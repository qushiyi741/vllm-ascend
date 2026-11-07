# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off

"""Standalone accepted-prefix checkpoint replay for Mega KDA."""

# Tensor operands are explicit to preserve the kernel ABI and DSL lexical order.
# Hardware stages and synchronization/movement order are fixed expansions;
# extracting helpers would change DSL lowering or measured kernel performance.

from __future__ import annotations

import functools
import sys
import threading
from types import ModuleType
import cannbotdsl
import torch
from typing import NamedTuple

from cannbotdsl import make_layout, MemLoc, Tensor, TensorSpec, dtypes
import torch._dynamo as torch_dynamo
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    UnpackMode,
    update_mask,
    vcast,
    vload,
    vload_broadcast,
    vload_unpack,
    vmadd,
    vmul,
    vstore,
)
from cannbotdsl.ops.sync import channel_rewind
from cannbotdsl.channel import Channel
from cannbotdsl.tensor import idx2crd, reinterpret, tile_slice

TILE = 128
VL = 64
KDA_STATE_UPDATE_UNROLL = 4
SUPPORTED_BLOCK_NUMS = (8, 24, 28, 32)
SMALL_BATCH_HEAD_LIMIT = 16
MEDIUM_BATCH_HEAD_LIMIT = 32
SMALL_BATCH_ROW_BLOCK = 32
_COMPILED_COMMIT_KERNEL = None


def _device_block_num(ref: torch.Tensor) -> int:
    """Return the current stream's effective AIC count, capped at 32."""
    if ref.device.type not in {"npu", "privateuseone"}:
        raise ValueError("commit_recurrent_kda requires NPU inputs")
    device_index = ref.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    stream = torch.npu.current_stream(device_index)
    properties = cannbotdsl.get_platform_info(stream=stream)
    try:
        cube_core_num = int(properties.cube_core_num)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"NPU {device_index} does not expose a valid AIC count"
        ) from error
    if cube_core_num <= 0:
        raise RuntimeError(
            f"NPU {device_index} reports invalid cube_core_num={cube_core_num}"
        )
    return min(cube_core_num, 32)


def _resolve_block_num(ref: torch.Tensor | None, block_num: int | None) -> int:
    """Resolve the public AIC-equivalent quota."""
    if block_num is None:
        if ref is None:
            raise ValueError("block_num=None requires a reference NPU tensor")
        available = _device_block_num(ref)
        choices = tuple(value for value in SUPPORTED_BLOCK_NUMS if value <= available)
        if not choices:
            raise ValueError(
                f"device exposes {available} AIC cores; at least 8 are required"
            )
        return choices[-1]
    if isinstance(block_num, bool) or not isinstance(block_num, int):
        raise ValueError(
            f"block_num must be one of {SUPPORTED_BLOCK_NUMS}; got {block_num!r}"
        )
    if block_num not in SUPPORTED_BLOCK_NUMS:
        raise ValueError(
            f"block_num must be one of {SUPPORTED_BLOCK_NUMS}; got {block_num}"
        )
    if ref is not None and ref.device.type in {"npu", "privateuseone"}:
        available = _device_block_num(ref)
        if block_num > available:
            raise ValueError(
                f"block_num={block_num} exceeds the device limit {available}"
            )
    return block_num


def _commit_launch_blocks(batch_size: int, num_heads: int, aic_blocks: int) -> int:
    """Use disjoint checkpoint row blocks to expose enough work to both AIV subblocks."""
    batch_heads = batch_size * num_heads
    if batch_heads <= SMALL_BATCH_HEAD_LIMIT:
        row_block = SMALL_BATCH_ROW_BLOCK
    elif batch_heads <= MEDIUM_BATCH_HEAD_LIMIT:
        row_block = VL
    else:
        row_block = TILE
    total_items = batch_heads * (TILE // row_block)
    return min(total_items, 2 * aic_blocks)


def _commit_row_schedule(batch_size: int, num_heads: int, aic_blocks: int):
    """Describe the kernel's disjoint state-row assignments for host tests."""
    batch_heads = batch_size * num_heads
    lane_count = _commit_launch_blocks(batch_size, num_heads, aic_blocks)
    schedule = [[] for _ in range(lane_count)]
    if batch_heads <= MEDIUM_BATCH_HEAD_LIMIT:
        row_block = (
            SMALL_BATCH_ROW_BLOCK if batch_heads <= SMALL_BATCH_HEAD_LIMIT else VL
        )
        row_blocks = TILE // row_block
        total_items = batch_heads * row_blocks
        for lane in range(lane_count):
            for work_item in range(lane, total_items, lane_count):
                head_item, row_index = divmod(work_item, row_blocks)
                schedule[lane].append((head_item, row_index * row_block, row_block))
        return tuple(tuple(lane) for lane in schedule)

    total_half_rows = 2 * batch_heads
    base_half_rows, extra_cores = divmod(total_half_rows, lane_count)
    base_full_heads = base_half_rows // 2
    extra_full_heads = (base_half_rows + 1) // 2
    extra_full_delta = extra_full_heads - base_full_heads
    full_heads = lane_count * base_full_heads + extra_cores * extra_full_delta
    for lane in range(lane_count):
        full_head_count = extra_full_heads if lane < extra_cores else base_full_heads
        prefix_extra = min(lane, extra_cores)
        full_head_start = lane * base_full_heads + prefix_extra * extra_full_delta
        for full_offset in range(full_head_count):
            schedule[lane].append((full_head_start + full_offset, 0, TILE))
        if base_half_rows % 2 == 0 and lane < extra_cores:
            schedule[lane].append((full_heads + lane // 2, (lane % 2) * VL, VL))
        if base_half_rows % 2 == 1 and lane >= extra_cores:
            half_rank = lane - extra_cores
            schedule[lane].append(
                (full_heads + half_rank // 2, (half_rank % 2) * VL, VL)
            )
    return tuple(tuple(lane) for lane in schedule)


class _CheckpointPool(NamedTuple):
    data: Tensor
    indices: Tensor
    stride0: int


@jit
def _checkpoint_head(pool, batch, head):
    slot = dtypes.int64(pool.indices[batch])
    begin = slot * pool.stride0 + head * TILE * TILE
    return pool.data[begin:begin + TILE * TILE,].view(TILE, TILE)


class CommitRecurrentKDAStage:
    """Replay one accepted token stream for each batch/head checkpoint."""

    def __init__(self):
        self.state_current = Buffer(MemLoc.UB, dtype=dtypes.float32, shape=(TILE, TILE))
        # The paired schedule keeps the next state tile in UB while the
        # current head drains its accepted-token stream.  The tile shape is
        # independent of B, N and S; only the scheduler is runtime-shaped.
        self.state_next = Buffer(MemLoc.UB, dtype=dtypes.float32, shape=(TILE, TILE))
        self.replay_u = Channel(MemLoc.UB, dtype=dtypes.float32, shape=(1, TILE), depth=2)
        self.replay_k = Channel(MemLoc.UB, dtype=dtypes.float16, shape=(1, TILE), depth=2)
        self.replay_decay = Channel(MemLoc.UB, dtype=dtypes.float32, shape=(1, TILE), depth=2)
        self.u_fp32 = Buffer(MemLoc.UB, dtype=dtypes.float32, shape=(1, TILE))

    @jit
    def prefetch_record(
        self,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        batch_index,
        token,
        value_head,
        row_index,
        active_rows: int,
    ):
        replay_u = self.replay_u.produce()
        if active_rows < TILE:
            replay_u = reinterpret(replay_u, offset=0, shape=(1, active_rows))
        mem_copy(
            replay_u,
            tile_slice(
                replay_u_gm[batch_index, value_head, None, None],
                (1, active_rows),
                (token, row_index),
            ),
        )
        mem_copy(self.replay_k.produce(), tile_slice(replay_k_gm[batch_index, value_head, None, None], (1, TILE), (token, 0)))
        mem_copy(self.replay_decay.produce(), tile_slice(replay_decay_gm[batch_index, value_head, None, None], (1, TILE), (token, 0)))

    @jit
    def apply_current_record(self, state, active_rows: int):
        replay_u = self.replay_u.consume()
        replay_k = self.replay_k.consume()
        replay_decay = self.replay_decay.consume()
        # Preserve the separate U staging phase and depth-two record pipeline.
        # U is already FP32 under the paged-state ABI, so this phase copies without quantization.
        with vf(mode="simd"):
            for segment in range((active_rows + VL - 1) // VL):
                offset = segment * VL
                segment_rows = active_rows - offset
                segment_mask, _ = update_mask(segment_rows, elem_bits=32)
                update = vload(replay_u, offset)
                vstore(self.u_fp32, offset, update, segment_mask)

        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            key_pre_packed = vload_unpack(
                replay_k, 0, unpack_mode=UnpackMode.B16_TO_B32
            )
            key_post_packed = vload_unpack(
                replay_k, VL, unpack_mode=UnpackMode.B16_TO_B32
            )
            key_pre = vcast(key_pre_packed, dtypes.float32, mask=full)
            key_post = vcast(key_post_packed, dtypes.float32, mask=full)
            decay_pre = vload(replay_decay, 0)
            decay_post = vload(replay_decay, VL)
            for row in dsl_range(0, active_rows, 1, unroll=KDA_STATE_UPDATE_UNROLL):
                offset = row * TILE
                u_row = vload_broadcast(self.u_fp32, row)
                state_pre = vload(state, offset)
                state_post = vload(state, offset + VL)
                state_pre = vmul(state_pre, decay_pre, mask=full)
                state_post = vmul(state_post, decay_post, mask=full)
                state_pre = vmadd(u_row, key_pre, state_pre, mask=full)
                state_post = vmadd(u_row, key_post, state_post, mask=full)
                vstore(state, offset, state_pre, full)
                vstore(state, offset + VL, state_post, full)

    @jit
    def execute_item(
        self,
        state_gm,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
        item,
        batch_size,
        num_heads,
        sequence_length,
        row_index,
        active_rows: int,
    ):
        batch_index, value_head = idx2crd(item, [batch_size, num_heads])
        accepted = dtypes.int64(accepted_gm[batch_index])
        if accepted > 0:
            state = self.state_current
            if active_rows < TILE:
                state = reinterpret(self.state_current, offset=0, shape=(active_rows, TILE))
            state_tile = tile_slice(
                _checkpoint_head(state_gm, batch_index, value_head),
                (active_rows, TILE),
                (row_index, 0),
            )
            mem_copy(state, state_tile)
            self.prefetch_record(
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                batch_index,
                0,
                value_head,
                row_index,
                active_rows,
            )
            for token in range(sequence_length):
                if token < accepted:
                    if token + 1 < accepted:
                        self.prefetch_record(
                            replay_u_gm,
                            replay_k_gm,
                            replay_decay_gm,
                            batch_index,
                            token + 1,
                            value_head,
                            row_index,
                            active_rows,
                        )
                    self.apply_current_record(state, active_rows)
            mem_copy(state_tile, state)

    @jit
    def execute_pair(
        self,
        state_gm,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
        current_item,
        next_item,
        batch_size,
        num_heads,
        sequence_length,
        next_row_index,
        next_active_rows: int,
    ):
        """Drain two disjoint state tiles through one UB double buffer."""
        current_batch, current_head = idx2crd(current_item, [batch_size, num_heads])
        next_batch, next_head = idx2crd(next_item, [batch_size, num_heads])
        current_accepted = dtypes.int64(accepted_gm[current_batch])
        next_accepted = dtypes.int64(accepted_gm[next_batch])
        current_state_tile = tile_slice(
            _checkpoint_head(state_gm, current_batch, current_head),
            (TILE, TILE),
            (0, 0),
        )
        next_state_tile = tile_slice(
            _checkpoint_head(state_gm, next_batch, next_head),
            (next_active_rows, TILE),
            (next_row_index, 0),
        )
        next_state = self.state_next
        if next_active_rows < TILE:
            next_state = reinterpret(self.state_next, offset=0, shape=(next_active_rows, TILE))

        if current_accepted > 0:
            mem_copy(self.state_current, current_state_tile)
            self.prefetch_record(
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                current_batch,
                0,
                current_head,
                0,
                TILE,
            )
            for token in range(sequence_length):
                if token < current_accepted:
                    if token + 1 < current_accepted:
                        self.prefetch_record(
                            replay_u_gm,
                            replay_k_gm,
                            replay_decay_gm,
                            current_batch,
                            token + 1,
                            current_head,
                            0,
                            TILE,
                        )
                    elif next_accepted > 0:
                        mem_copy(next_state, next_state_tile)
                        self.prefetch_record(
                            replay_u_gm,
                            replay_k_gm,
                            replay_decay_gm,
                            next_batch,
                            0,
                            next_head,
                            next_row_index,
                            next_active_rows,
                        )
                    self.apply_current_record(self.state_current, TILE)
            mem_copy(current_state_tile, self.state_current)

        if next_accepted > 0:
            if current_accepted == 0:
                mem_copy(next_state, next_state_tile)
                self.prefetch_record(
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    next_batch,
                    0,
                    next_head,
                    next_row_index,
                    next_active_rows,
                )
            for token in range(sequence_length):
                if token < next_accepted:
                    if token + 1 < next_accepted:
                        self.prefetch_record(
                            replay_u_gm,
                            replay_k_gm,
                            replay_decay_gm,
                            next_batch,
                            token + 1,
                            next_head,
                            next_row_index,
                            next_active_rows,
                        )
                    self.apply_current_record(next_state, next_active_rows)
            mem_copy(next_state_tile, next_state)

    @jit
    def execute_row_blocks(
        self,
        state_gm,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
        row_block: int,
    ):
        batch_size = accepted_gm.shape[0]
        num_heads = replay_u_gm.shape[1]
        sequence_length = replay_u_gm.shape[2]
        row_blocks = TILE // row_block
        total_items = batch_size * num_heads * row_blocks
        block_idx = get_block_idx()
        aiv_block_count = get_block_num()
        for work_item in range(block_idx, total_items, aiv_block_count):
            item = work_item // row_blocks
            row_index = work_item - item * row_blocks
            self.execute_item(
                state_gm,
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                accepted_gm,
                item,
                batch_size,
                num_heads,
                sequence_length,
                row_index,
                row_block,
            )

    @jit
    def execute_minimal_split(
        self,
        state_gm,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
    ):
        batch_size = accepted_gm.shape[0]
        num_heads = replay_u_gm.shape[1]
        sequence_length = replay_u_gm.shape[2]
        block_idx = get_block_idx()
        aiv_block_count = get_block_num()
        batch_heads = batch_size * num_heads
        total_half_rows = 2 * batch_heads
        base_half_rows = total_half_rows // aiv_block_count
        extra_cores = total_half_rows - base_half_rows * aiv_block_count
        base_full_heads = base_half_rows // 2
        extra_full_heads = (base_half_rows + 1) // 2
        extra_full_delta = extra_full_heads - base_full_heads
        full_head_count = base_full_heads
        if block_idx < extra_cores:
            full_head_count = extra_full_heads
        prefix_extra = block_idx if block_idx < extra_cores else extra_cores
        full_head_start = block_idx * base_full_heads + prefix_extra * extra_full_delta

        for full_offset in range(full_head_count):
            self.execute_item(
                state_gm,
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                accepted_gm,
                full_head_start + full_offset,
                batch_size,
                num_heads,
                sequence_length,
                0,
                TILE,
            )

        full_heads = aiv_block_count * base_full_heads + extra_cores * extra_full_delta
        if base_half_rows % 2 == 0 and block_idx < extra_cores:
            self.execute_item(
                state_gm,
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                accepted_gm,
                full_heads + block_idx // 2,
                batch_size,
                num_heads,
                sequence_length,
                block_idx % 2,
                VL,
            )
        if base_half_rows % 2 == 1 and block_idx >= extra_cores:
            half_rank = block_idx - extra_cores
            self.execute_item(
                state_gm,
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                accepted_gm,
                full_heads + half_rank // 2,
                batch_size,
                num_heads,
                sequence_length,
                half_rank % 2,
                VL,
            )

    @jit
    def execute_paired(
        self,
        state_gm,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
    ):
        """Use paired full/half-head work for large even head counts.

        The mapping is the generalized form of the former B16/N6 schedule:
        low AIV grids process several full-head pairs, while larger grids use
        a full head plus one half of a second head per lane.
        """
        batch_size = accepted_gm.shape[0]
        num_heads = replay_u_gm.shape[1]
        sequence_length = replay_u_gm.shape[2]
        batch_heads = batch_size * num_heads
        block_idx = get_block_idx()
        aiv_block_count = get_block_num()

        # If the launch grid has one lane per head, do not manufacture a
        # second item.  This is the tail case for small dynamic B*N values.
        if aiv_block_count >= batch_heads:
            if block_idx < batch_heads:
                self.execute_item(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                    block_idx,
                    batch_size,
                    num_heads,
                    sequence_length,
                    0,
                    TILE,
                )
        elif aiv_block_count <= batch_heads // 2:
            # Every lane drains one or more full-head pairs.
            for pair_index in range(block_idx, batch_heads // 2, aiv_block_count):
                self.execute_pair(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                    2 * pair_index,
                    2 * pair_index + 1,
                    batch_size,
                    num_heads,
                    sequence_length,
                    0,
                    TILE,
                )
        elif 3 * aiv_block_count <= 2 * batch_heads:
            # This is the Disjoint full/half-row split schedule.  `pair_lanes`
            # process two full heads; the remaining lanes process one full
            # head plus one half of the remaining head through state_next.
            pair_lanes = 2 * batch_heads - 3 * aiv_block_count
            if block_idx < pair_lanes:
                self.execute_pair(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                    2 * block_idx,
                    2 * block_idx + 1,
                    batch_size,
                    num_heads,
                    sequence_length,
                    0,
                    TILE,
                )
            else:
                split_rank = block_idx - pair_lanes
                remaining_lanes = aiv_block_count - pair_lanes
                current_start = 2 * pair_lanes
                self.execute_pair(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                    current_start + split_rank,
                    current_start + remaining_lanes + split_rank // 2,
                    batch_size,
                    num_heads,
                    sequence_length,
                    split_rank % 2,
                    VL,
                )
        else:
            # When only a small tail remains, pair full heads instead of
            # splitting a tile.  This keeps every item in bounds for shapes
            # such as B=7,N=6 while still using the UB double buffer.
            pair_lanes = batch_heads - aiv_block_count
            if block_idx < pair_lanes:
                self.execute_pair(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                    2 * block_idx,
                    2 * block_idx + 1,
                    batch_size,
                    num_heads,
                    sequence_length,
                    0,
                    TILE,
                )
            else:
                self.execute_item(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                    2 * pair_lanes + block_idx - pair_lanes,
                    batch_size,
                    num_heads,
                    sequence_length,
                    0,
                    TILE,
                )

    @jit
    def execute(
        self,
        state_gm,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
    ):
        batch_heads = accepted_gm.shape[0] * replay_u_gm.shape[1]
        channel_rewind(reset_sync_id=True)
        if batch_heads <= SMALL_BATCH_HEAD_LIMIT:
            self.execute_row_blocks(
                state_gm,
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                accepted_gm,
                SMALL_BATCH_ROW_BLOCK,
            )
        elif batch_heads <= MEDIUM_BATCH_HEAD_LIMIT:
            self.execute_row_blocks(
                state_gm, replay_u_gm, replay_k_gm, replay_decay_gm, accepted_gm, VL
            )
        else:
            if batch_heads % 2 == 0:
                self.execute_paired(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                )
            else:
                self.execute_minimal_split(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                )


class CommitRecurrentKDAKernel:
    """Original optimized AIV schedule with direct paged-state checkpoint addressing."""

    @kernel
    def kernel(self, checkpoint_state, replay_u, replay_k, replay_decay,
        num_accepted_tokens, state_indices, checkpoint_stride0):
        pool = _CheckpointPool(checkpoint_state, state_indices, checkpoint_stride0)
        CommitRecurrentKDAStage().execute(
            pool, replay_u, replay_k, replay_decay, num_accepted_tokens)

    @host
    def run(self, checkpoint_state, replay_u, replay_k, replay_decay,
        num_accepted_tokens, state_indices, checkpoint_stride0, launch_blocks):
        self.kernel[launch_blocks](checkpoint_state, replay_u, replay_k,
            replay_decay, num_accepted_tokens, state_indices, checkpoint_stride0)


def _commit_specification():
    span=cannbotdsl.Dim('physical_span',min=1);mb=cannbotdsl.Dim('max_batch',min=1)
    n=cannbotdsl.Dim('heads',min=1);s=cannbotdsl.Dim('sequence',min=1);b=cannbotdsl.Dim('batch',min=1)
    return (TensorSpec((span,),dtypes.float32),TensorSpec((mb,n,s,128),dtypes.float32),
        TensorSpec((mb,n,s,128),dtypes.float16),TensorSpec((mb,n,s,128),dtypes.float32),
        TensorSpec((b,),dtypes.int32),TensorSpec((b,),dtypes.int32),dtypes.int64,dtypes.int64)


def _compile_commit_kernel():
    return cannbotdsl.compile(CommitRecurrentKDAKernel().run, *_commit_specification())


def _get_compiled_commit_kernel():
    global _COMPILED_COMMIT_KERNEL
    if _COMPILED_COMMIT_KERNEL is None:
        _COMPILED_COMMIT_KERNEL=_compile_commit_kernel()
    return _COMPILED_COMMIT_KERNEL


def _shared_kda_call_state():
    """Reuse one lock/event state across the independently loaded KDA modules."""
    key = "_cannbotdsl_kda_call_state"
    state = sys.modules.get(key)
    if state is None:
        candidate = ModuleType(key)
        candidate.call_lock = threading.RLock()
        candidate.pending = {"event": None, "keep": None}
        state = sys.modules.setdefault(key, candidate)
    return state


_KDA_CALL_STATE = _shared_kda_call_state()
_CALL_LOCK = _KDA_CALL_STATE.call_lock
_PENDING = _KDA_CALL_STATE.pending


class InputValidation:
    """Input contracts and writable-buffer protection; no reference mathematics."""

    @staticmethod
    def require(condition, message):
        if isinstance(condition, bool):
            if not condition:
                raise ValueError(message)
        else:
            torch._check(condition, lambda: message)

    @staticmethod
    def tensor_metadata(name, tensor, shape, dtype, device):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'{name} must be a torch.Tensor')
        shape_label = '[max_batch,N,S,128]' if name in ('replay_u', 'replay_k', 'replay_decay') else str(shape)
        InputValidation.require(tensor.ndim == len(shape), f'{name} must have shape {shape_label}')
        for actual, expected in zip(tensor.shape, shape):
            if expected is not None:
                InputValidation.require(actual == expected, f'{name} must have shape {shape_label}')
        label = {torch.float16: 'FP16', torch.float32: 'FP32', torch.bfloat16: 'BF16', torch.int32: 'int32', torch.int64: 'int64'}[dtype]
        if name == 'hidden_states':
            InputValidation.require(tensor.dtype == torch.bfloat16 and tensor.is_contiguous(), 'hidden_states must be contiguous BF16')
        InputValidation.require(tensor.dtype == dtype, f'{name} must have dtype {label}')
        InputValidation.require(tensor.layout == torch.strided, f'{name} must have strided layout')
        InputValidation.require(tensor.is_contiguous(), f'{name} must be contiguous')
        InputValidation.require(tensor.device == device, f'{name} must share the input device')
        InputValidation.require(not tensor.is_neg() and (not tensor.is_conj()), f'{name} must resolve negative/conjugate view bits')

    @staticmethod
    def state_pool_metadata(name, tensor, shape, dtype, device):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'{name} must be a torch.Tensor')
        InputValidation.require(tensor.ndim == len(shape), f'{name} must have shape {shape}')
        for actual, expected in zip(tensor.shape, shape):
            if expected is not None:
                InputValidation.require(actual == expected, f'{name} must have shape {shape}')
        InputValidation.require(tensor.dtype == dtype, f'{name} must have dtype {dtype}')
        InputValidation.require(tensor.layout == torch.strided and tensor.device == device, f'{name} must share input device and strided layout')
        expected = 1
        for axis in range(tensor.ndim - 1, 0, -1):
            InputValidation.require(tensor.stride(axis) == expected, f'{name} only axis 0 may be non-contiguous')
            expected = expected * tensor.shape[axis]
        InputValidation.require(tensor.stride(0) >= expected, f'{name} axis-0 stride must not overlap paged-state blocks')
        InputValidation.require(not tensor.is_neg() and (not tensor.is_conj()), f'{name} must resolve view bits')

    @staticmethod
    def _records(a, batch, heads, device):
        mb = a.get('max_batch', 512)
        InputValidation.require(isinstance(mb, int) and (not isinstance(mb, bool)) and (mb > 0), 'max_batch must be a positive integer')
        InputValidation.require((batch > 0) & (batch <= mb), 'batch must be in [1,max_batch]')
        u = a['replay_u']
        if not isinstance(u, torch.Tensor):
            raise TypeError('replay_u must be a torch.Tensor')
        InputValidation.require(u.ndim == 4, 'replay_u must be [max_batch,N,S,128]')
        capacity = u.shape[2]
        InputValidation.require(capacity > 0, 'record S capacity must be positive')
        for name, dt in (('replay_u', torch.float32), ('replay_k', torch.float16), ('replay_decay', torch.float32)):
            InputValidation.tensor_metadata(name, a[name], (mb, heads, capacity, 128), dt, device)
        return capacity

    @staticmethod
    def commit_metadata(a):
        checkpoint = a['checkpoint_state']
        if not isinstance(checkpoint, torch.Tensor):
            raise TypeError('checkpoint_state must be a torch.Tensor')
        InputValidation.require(checkpoint.ndim == 4, 'checkpoint_state must be [block_num,N,128,128]')
        blocks, heads = checkpoint.shape[:2]
        device = checkpoint.device
        InputValidation.require((blocks > 0) & (heads > 0), 'paged-state block count and heads must be positive')
        InputValidation.state_pool_metadata('checkpoint_state', checkpoint, (blocks, heads, 128, 128), torch.float32, device)
        idx = a['state_indices']
        if not isinstance(idx, torch.Tensor):
            raise TypeError('state_indices must be a torch.Tensor')
        InputValidation.require(idx.ndim == 1, 'state_indices must be [B]')
        batch = idx.shape[0]
        InputValidation.tensor_metadata('state_indices', idx, (batch,), torch.int32, device)
        InputValidation.tensor_metadata('num_accepted_tokens', a['num_accepted_tokens'], (batch,), torch.int32, device)
        InputValidation._records(a, batch, heads, device)
        block = a['block_num']
        InputValidation.require(block is None or (isinstance(block, int) and (not isinstance(block, bool)) and (block in (8, 24, 28, 32))), 'block_num must be one of (8,24,28,32) or None')
        InputValidation.require(not checkpoint.requires_grad, 'checkpoint_state is an inference-only writable buffer')
        return {name: value for name, value in a.items() if isinstance(value, torch.Tensor)}

    @staticmethod
    def _bounded(values, low, high, name, message=None):
        if any((v < low or v > high for v in values)):
            raise ValueError(message or f'{name} values must be in [{low},{high}]')

    @staticmethod
    def validate_commit_values(a):
        indices = a['state_indices'].detach().cpu().tolist()
        InputValidation._bounded(indices, 0, a['checkpoint_state'].shape[0] - 1, 'state_indices')
        InputValidation.require(len(indices) == len(set(indices)), 'state_indices must be distinct across batch')
        InputValidation._bounded(a['num_accepted_tokens'].detach().cpu().tolist(), 0, a['replay_u'].shape[2], 'num_accepted_tokens')

    @staticmethod
    def validate_aliases(tensors, writable):
        spans = {}
        for name, tensor in tensors.items():
            InputValidation.require(tensor.device.type != 'meta', 'physical alias validation requires real tensors')
            start = tensor.data_ptr()
            size = tensor.numel() * tensor.element_size()
            if name in ('conv_state', 'checkpoint_state'):
                width = tensor[0].numel() * tensor.element_size()
                ranges = [(start + i * tensor.stride(0) * tensor.element_size(), start + i * tensor.stride(0) * tensor.element_size() + width) for i in range(tensor.shape[0])]
            else:
                if tensor.device.type in ('npu', 'privateuseone'):
                    import torch_npu
                    if torch_npu.get_npu_format(tensor) == 29:
                        size = int(torch_npu.get_storage_size(tensor)) * tensor.element_size()
                ranges = [(start, start + size)]
            spans[name] = ranges
        names = tuple(tensors)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                if left not in writable and right not in writable:
                    continue
                if tensors[left].device != tensors[right].device:
                    continue
                x = y = 0
                a = spans[left]
                b = spans[right]
                while x < len(a) and y < len(b):
                    if a[x][0] < b[y][1] and b[y][0] < a[x][1]:
                        raise ValueError(f'writable buffer overlap: {left} and {right}')
                    if a[x][1] <= b[y][0]:
                        x += 1
                    else:
                        y += 1

    @staticmethod
    def serialized(function):

        @functools.wraps(function)
        def checked(*args, **kwargs):
            ref = args[0] if args else kwargs.get('hidden_states', kwargs.get('checkpoint_state'))
            if not isinstance(ref, torch.Tensor) or ref.device.type not in ('npu', 'privateuseone'):
                return function(*args, **kwargs)
            device = ref.device
            with torch.npu.device(device):
                if torch.npu.is_current_stream_capturing():
                    raise RuntimeError('strict KDA validation does not support NPU graph capture; use uncaptured checked calls')
                with _CALL_LOCK:
                    if _PENDING['event'] is not None:
                        _PENDING['event'].synchronize()
                        _PENDING['event'] = None
                        _PENDING['keep'] = None
                    result = None
                    try:
                        result = function(*args, **kwargs)
                        return result
                    finally:
                        event = torch.npu.Event()
                        event.record(torch.npu.current_stream(device))
                        _PENDING['event'] = event
                        _PENDING['keep'] = (args, kwargs, result)
        return checked

    @staticmethod
    def real_boundary(tensors, writable):
        for name, tensor in tensors.items():
            InputValidation.require(tensor.device.type in ('npu', 'privateuseone'), f'{name} requires NPU inputs')
            InputValidation.require(type(tensor) is torch.Tensor, f'{name} must be an exact Tensor; detach Parameter inputs')
        InputValidation.validate_aliases(tensors, writable)
        import torch_npu
        nz_weights = ('qkv_projection_weight', 'decay_projection_a_weight', 'decay_projection_b_weight', 'beta_projection_weight', 'output_gate_projection_weight', 'output_projection_weight')
        for name, tensor in tensors.items():
            if name not in nz_weights:
                InputValidation.require(torch_npu.get_npu_format(tensor) in (0, 2), f'{name} requires ND storage; paged-state addressing does not support blocked physical formats')
        if 'hidden_states' in tensors:
            import cannbotdsl
            for name in ('qkv_projection_weight', 'decay_projection_a_weight', 'decay_projection_b_weight', 'beta_projection_weight', 'output_gate_projection_weight', 'output_projection_weight'):
                InputValidation.require(torch_npu.get_npu_format(tensors[name]) == 29, f'{name} requires BF16 FRACTAL_NZ storage')
            ref = tensors['hidden_states']
            hardware = torch.npu.get_device_properties(ref.device)
            info = cannbotdsl.get_platform_info(stream=torch.npu.current_stream(ref.device))
            available = min(int(hardware.cube_core_num), int(info.cube_core_num), int(hardware.vector_core_num) // 2, int(info.vector_core_num) // 2)
            InputValidation.require(available >= 32 or available in (8, 16, 24, 28), f'AIC quota {available} is not supported; use 8/16/24/28/32 or a full stream with at least 32 AIC and 64 AIV')


@InputValidation.serialized
def commit_recurrent_kda(checkpoint_state:torch.Tensor,replay_u:torch.Tensor,replay_k:torch.Tensor,
    replay_decay:torch.Tensor,num_accepted_tokens:torch.Tensor,state_indices:torch.Tensor,
    block_num:int|None=None,max_batch:int=512)->None:
    """Apply each logical batch's accepted prefix to its selected paged-state checkpoint.

    Args:
        checkpoint_state (Tensor): Writable FP32 [block_num,N,128,128] paged-state pool;
            only axis zero may be non-contiguous. Its actual stride is passed to the kernel.
        replay_u (Tensor): Read-only FP32 [max_batch,N,S,128].
        replay_k (Tensor): Read-only FP16 [max_batch,N,S,128].
        replay_decay (Tensor): Read-only FP32 [max_batch,N,S,128].
        num_accepted_tokens (Tensor): INT32 [B], every value in [0,S].
        state_indices (Tensor): INT32 [B], distinct paged-state slots; slot zero is valid.
        block_num (int|None): AIC quota 8/24/28/32; None selects current-stream maximum.
        max_batch (int): Record capacity, default 512.
    Returns:
        None; selected checkpoints are changed in place. Zero-accepted slots,
        unselected slots, paged-state gaps and all replay records remain unchanged.
    Notes:
        checkpoint = checkpoint * decay + outer(U,K), in original FP32 arithmetic.
        No fixed S=8 restriction and no CPU mathematical fallback.
        Original 32/64/128-row schedules, paired state prefetch and depth-two
        record queues are preserved; paged-state indices affect addressing only.
    """
    a=locals().copy();tensors=InputValidation.commit_metadata(a)
    InputValidation.real_boundary(tensors,('checkpoint_state',));InputValidation.validate_commit_values(a)
    batch=state_indices.shape[0];heads=checkpoint_state.shape[1]
    cores=_resolve_block_num(checkpoint_state,block_num)
    lanes=_commit_launch_blocks(batch,heads,cores)
    span=(checkpoint_state.shape[0]-1)*checkpoint_state.stride(0)+checkpoint_state[0].numel()
    _get_compiled_commit_kernel()(checkpoint_state.as_strided((span,),(1,)),replay_u,replay_k,replay_decay,
        num_accepted_tokens,state_indices,checkpoint_state.stride(0),lanes)


@export('commit_recurrent_kda')
def export_commit_recurrent_kda():
    handle=_compile_commit_kernel();handle.close()


_GRAPH_LIBRARY=torch.library.Library('cannbotdsl_commit_recurrent_kda','DEF')
_GRAPH_LIBRARY.define('commit_recurrent_kda(Tensor(a!) checkpoint_state, Tensor replay_u, Tensor replay_k, '
    'Tensor replay_decay, Tensor num_accepted_tokens, Tensor state_indices, int? block_num=None, int max_batch=512) -> ()')


@torch.library.impl(_GRAPH_LIBRARY,'commit_recurrent_kda','Meta')
def _commit_recurrent_kda_meta(checkpoint_state,replay_u,replay_k,replay_decay,num_accepted_tokens,
    state_indices,block_num=None,max_batch=512):
    InputValidation.commit_metadata(locals());return None


@torch.library.impl(_GRAPH_LIBRARY,'commit_recurrent_kda','PrivateUse1')
def _commit_recurrent_kda_privateuse1(*args,**kwargs):return commit_recurrent_kda(*args,**kwargs)


commit_recurrent_kda_op=torch.ops.cannbotdsl_commit_recurrent_kda.commit_recurrent_kda
torch_dynamo.substitute_in_graph(commit_recurrent_kda,can_constant_fold_through=False)(
    lambda *args,**kwargs:commit_recurrent_kda_op(*args,**kwargs))
