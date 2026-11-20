#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from collections.abc import Mapping, Sequence
from copy import copy
from typing import Any, cast

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_dcp_group
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    DSparkSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import get_eagle3_aux_layers_from_config

from vllm_ascend import envs
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState
from vllm_ascend.attention.mla_v1 import AscendMLABackend, all_mla_use_flashmla
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    DSparkContextChunk,
    initialize_draft_context_chunk,
)
from vllm_ascend.spec_decode.dspark_proposer import (
    _draft_config_value,
    _get_native_window_gqa_width,
    _mapping_value,
    _is_window_gqa_draft,
)
from vllm_ascend.utils import lmhead_tp_enable, lmhead_tp_max_num_logits
from vllm_ascend.worker.dcp_utils import DCPManager
from vllm_ascend.worker.v2.aclgraph_utils import _get_graph_update_backend
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_factory,
    build_attn_metadata_wrapper,
)
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    prepare_dflash_inputs,
    prepare_dflash_inputs_factory,
)
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin
from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
    disable_profiling_chunk_for_draft,
    prepare_replicated_pcp_config,
)


class AscendDSparkSpeculator(LmheadTPDraftSamplingMixin, DSparkSpeculator):
    _speculator_name = "DSpark"
    # DSpark samples via compute_draft_logits and never calls sample_draft, so
    # the mixin sample_draft alignment is not used; instead load_draft_model
    # wraps the draft model's compute_draft_logits to pad the LM-head input to
    # the group-agreed capacity and trim the logits back (no _sample_sequential
    # override, upstream sampling logic untouched).
    _lmhead_tp_sample_draft_supported = True

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        vllm_config, self.replicated_pcp = prepare_replicated_pcp_config(vllm_config)
        super().__init__(vllm_config, device)
        self._lmhead_tp_validate_draft_sampling()
        self.input_batch: InputBatch | None = None
        self.attn_architecture: str | None = None
        self._draft_mla_uses_device_lengths = False
        self._window_gqa_dcp_fallback = _is_window_gqa_draft(self.draft_model_config.hf_config)
        self._window_gqa_attn_vllm_config: VllmConfig | None = None
        self._verify_num_speculative_steps = self.num_speculative_steps
        self._native_num_speculative_steps = _get_native_window_gqa_width(
            self.draft_model_config.hf_config,
            self.num_speculative_steps,
        )
        if self._native_num_speculative_steps is not None:
            self._use_native_window_gqa_width()
        if self._window_gqa_dcp_fallback:
            # The generic DFlash initializer derives this from the model's
            # non-causal flags.  Window-GQA checkpoints may still expose the
            # MLA hook, but their FIA call uses sparse_mode=4 and must retain
            # the checkpoint's causal setting so a mask is generated.
            dflash_config = _draft_config_value(self.draft_model_config.hf_config, "dflash_config")
            self._group_causal = bool(_mapping_value(dflash_config, "causal", True))
        self._init_dcp()

    def _use_native_window_gqa_width(self) -> None:
        """Resize V2 DSpark-only buffers for RedHat's fixed eight-row graph."""
        native_steps = self._native_num_speculative_steps
        assert native_steps is not None
        if native_steps <= self._verify_num_speculative_steps:
            return
        self.num_speculative_steps = native_steps
        self.num_query_per_req = native_steps if self.sample_from_anchor else 1 + native_steps
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            native_steps,
            dtype=torch.int64,
            device=self.device,
        )
        self._step_cols = torch.arange(native_steps, dtype=torch.int32, device=self.device)
        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=self.device)
            * self.num_query_per_req
        )
        sample_capacity = self.max_num_reqs * native_steps
        self.sample_indices = torch.zeros(sample_capacity, dtype=torch.int64, device=self.device)
        self.sample_pos = torch.zeros(sample_capacity, dtype=torch.int64, device=self.device)
        # ``-1`` marks padded rows as inert.  A zero here would make a full
        # graph replay write padding logits into request-state row zero.
        self.sample_idx_mapping = torch.full(
            (sample_capacity,), -1, dtype=torch.int32, device=self.device
        )
        self.sample_col = torch.arange(native_steps, dtype=torch.int32, device=self.device).repeat(self.max_num_reqs)
        # Adaptive verification writes one confidence value per native step.
        # The upstream buffer is allocated from the public verification width;
        # widen it together with ``draft_tokens`` so the fixed-width forward
        # cannot fail when a confidence head is enabled.
        if hasattr(self, "draft_token_confidence_probs"):
            confidence = self.draft_token_confidence_probs
            self.draft_token_confidence_probs = confidence.new_empty(
                self.max_num_reqs,
                native_steps,
            )
        if self.draft_logits is not None:
            # Current vLLM exposes this as a tensor.  A few development
            # branches wrapped it in a ``DraftLogitsCache`` with a ``values``
            # member, so preserve either representation while widening only
            # the step dimension.
            cache = self.draft_logits
            cache_values = cache if isinstance(cache, torch.Tensor) else getattr(cache, "values", cache)
            widened = cache_values.new_zeros(cache_values.shape[0], native_steps, cache_values.shape[2])
            if cache_values is cache:
                self.draft_logits = widened
            else:
                cache.values = widened

    def _public_draft_tokens(self, draft_tokens: torch.Tensor) -> torch.Tensor:
        if self._native_num_speculative_steps is None:
            return draft_tokens
        return draft_tokens[:, : self._verify_num_speculative_steps].contiguous()

    @property
    def attn_vllm_config(self) -> VllmConfig:
        """Use a cached local DCP=1 config for a replicated Window-GQA draft."""
        if not getattr(self, "_window_gqa_dcp_fallback", False):
            return super().attn_vllm_config
        if getattr(self, "_window_gqa_attn_vllm_config", None) is None:
            config = copy(self.vllm_config)
            config.attention_config = copy(config.attention_config)
            config.attention_config.use_non_causal = self.requires_non_causal
            config.parallel_config = copy(config.parallel_config)
            config.parallel_config.decode_context_parallel_size = 1
            config.parallel_config.dcp_kv_cache_interleave_size = 1
            config.parallel_config.cp_kv_cache_interleave_size = 1
            self._window_gqa_attn_vllm_config = config
        return self._window_gqa_attn_vllm_config

    def _init_dcp(self) -> None:
        self.use_dcp = self.attn_vllm_config.parallel_config.decode_context_parallel_size > 1
        self.dcp_manager: DCPManager | None = None
        if not self.use_dcp:
            return
        self.dcp_manager = DCPManager(
            dcp_world_size=self.attn_vllm_config.parallel_config.decode_context_parallel_size,
            dcp_rank=get_dcp_group().rank_in_group,
            max_buffer_num_tokens=self.max_num_tokens,
            max_num_reqs=self.max_num_reqs,
            device=self.device,
            vllm_config=self.attn_vllm_config,
            use_async_scheduling=self.vllm_config.scheduler_config.async_scheduling,
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        draft_config = self.vllm_config
        if self._window_gqa_dcp_fallback:
            target_vllm_config = self.vllm_config
            draft_config = self.attn_vllm_config
            self.vllm_config = draft_config
            try:
                with disable_profiling_chunk_for_draft(self.vllm_config):
                    model = super().load_draft_model(target_model, target_attn_layer_names)
                if hasattr(model, "post_process"):
                    model.post_process(draft_config)
            finally:
                self.vllm_config = target_vllm_config
        else:
            with disable_profiling_chunk_for_draft(self.vllm_config):
                model = super().load_draft_model(target_model, target_attn_layer_names)
            if hasattr(model, "post_process"):
                model.post_process(draft_config)
        if hasattr(model, "configure_target_aux_hidden_capture"):
            model.configure_target_aux_hidden_capture(target_model)

        self._lmhead_tp_wrap_draft_logits(model)

        return model

    def _lmhead_tp_wrap_draft_logits(self, model: torch.nn.Module) -> None:
        """Pad/trim the DSpark draft LM head around its collectives.

        DSpark feeds its vocab-sharded draft LM head directly through
        ``compute_draft_logits`` (inside upstream ``_sample_sequential`` /
        ``_sample_sequential_topk``), which bypasses the mixin's
        ``sample_draft`` alignment. Wrap the method instead of overriding the
        sampling loop: every rank feeds the group-agreed capacity
        (``max_num_reqs * num_speculative_steps``) into the LM-head
        collectives, then the logits are trimmed back to the real rows.
        """
        if not lmhead_tp_enable():
            return

        original = model.compute_draft_logits
        capacity = lmhead_tp_max_num_logits(self.max_num_reqs, self.num_speculative_steps)

        def aligned(hidden_states: torch.Tensor) -> torch.Tensor:
            num_logits = hidden_states.shape[0]
            if num_logits > capacity:
                raise ValueError(
                    f"lmhead TP DSpark draft rows ({num_logits}) exceed the group-agreed "
                    f"capacity ({capacity} = max_num_reqs * num_speculative_steps)."
                )
            padded = hidden_states
            if num_logits < capacity:
                # Zero rows carry no draft token; they are trimmed back off.
                padded = torch.nn.functional.pad(hidden_states, (0, 0, 0, capacity - num_logits))
            return original(padded)[:num_logits]

        model.compute_draft_logits = aligned

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        if self._window_gqa_dcp_fallback:
            target_vllm_config = self.vllm_config
            self.vllm_config = self.attn_vllm_config
            try:
                super().init_cudagraph_manager(cudagraph_mode)
            finally:
                self.vllm_config = target_vllm_config
        else:
            super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph manager is patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # It needs this speculator to update full-graph params, so set it here.
        self.query_cudagraph_manager.speculator = self
        self.query_cudagraph_manager.update_stream = self.update_stream

    @torch.inference_mode()
    def get_draft_context_group_layout(self) -> tuple[tuple[int, ...], tuple[int, ...], dict[int, int]]:
        """Return the loaded draft's actual layer-to-cache-group mapping."""
        layer_names = self.model.get_draft_kv_cache_layer_names()
        group_ids = tuple(self.draft_kv_cache_group_ids)
        if self._layer_group_idx is None:
            if len(group_ids) != 1:
                raise ValueError("DSpark requires an explicit cache-group map for multiple draft KV groups")
            layer_group_ids = (group_ids[0],) * len(layer_names)
        else:
            if len(self._layer_group_idx) != len(layer_names):
                raise ValueError("DSpark cache-group map does not match the loaded draft layers")
            layer_group_ids = tuple(group_ids[index] for index in self._layer_group_idx)
        if not group_ids or not layer_group_ids:
            raise ValueError("DSpark context initialization requires loaded draft attention layers")
        block_sizes = {
            group_id: self.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.block_size for group_id in group_ids
        }
        return group_ids, layer_group_ids, block_sizes

    @torch.inference_mode()
    def initialize_local_context(
        self,
        chunk: DSparkContextChunk,
        aux_hidden_states: torch.Tensor,
        draft_block_ids_by_group: Mapping[int, Sequence[int]],
    ) -> None:
        """Write this P worker's draft KV for a chunk of prompt auxiliary states.

        The feature tensor never leaves P. Run between the target prefill and
        MemFabric read notification, then synchronize before the draft pages
        may be read or reused.
        """
        if self.attn_architecture != "MLA" or self.use_dcp:
            raise ValueError("P-side DSpark prompt initialization requires MLA without DCP")
        parallel = self.attn_vllm_config.parallel_config
        if parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
            raise ValueError("P-side DSpark prompt initialization does not support context parallelism")
        expected_layers = get_eagle3_aux_layers_from_config(self.speculative_config)
        descriptor = chunk.descriptor
        if (
            not expected_layers
            or descriptor.aux_layer_ids != tuple(expected_layers)
            or descriptor.hidden_size != self.vllm_config.model_config.get_hidden_size()
            or descriptor.prompt_tokens > self.max_model_len
        ):
            raise ValueError("P-side DSpark context does not match the loaded target/draft schema")
        if aux_hidden_states.dtype != torch.bfloat16 or tuple(aux_hidden_states.shape) != (
            chunk.num_tokens,
            descriptor.feature_width,
        ):
            raise ValueError("P-side DSpark context must contain ordered BF16 auxiliary features")
        group_ids, layer_group_ids, block_sizes_by_group = self.get_draft_context_group_layout()
        block_ids = {group_id: tuple(draft_block_ids_by_group[group_id]) for group_id in group_ids}
        with set_current_vllm_config(self.attn_vllm_config):
            initialize_draft_context_chunk(
                self.model,
                chunk,
                aux_hidden_states.to(device=self.device),
                draft_group_ids=group_ids,
                draft_block_ids_by_group=block_ids,
                block_sizes_by_group=block_sizes_by_group,
                layer_group_ids=layer_group_ids,
                device=self.device,
            )

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        if self._window_gqa_dcp_fallback:
            # Keep the shared per-group topology, but expose a scalar DCP=1
            # view to upstream draft input preparation.
            block_tables = copy(block_tables)
            block_tables.cp_size = 1
            block_tables.cp_rank = 0
            block_tables.cp_interleave = 1
        # Initialize the draft attention backend with its PCP=1 config.
        with set_current_vllm_config(self.attn_vllm_config):
            super().set_attn(
                model_state,
                kv_cache_config,
                block_tables,
                target_input_buffers,
                target_attn_groups,
            )
            self._context_slot_mappings = self._context_slot_mappings.to(torch.int32)  # type: ignore[has-type]
            # npu needs attn_backends to update full graph params in run_fullgraph.
            attn_backends: dict[str, type[AttentionBackend]] = {}
            draft_impl_c8_quant = False
            active_layer_names = self.draft_attn_layer_names
            for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
                layer_names = kv_cache_group_spec.layer_names
                if active_layer_names is not None:
                    # Preserve cache-group order so captured graph tasks and
                    # runtime metadata stay aligned.
                    layer_names = [name for name in layer_names if name in active_layer_names]

                layer_type = cast(type[Any], AttentionLayerBase)
                attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

                for layer_name in layer_names:
                    attn_backends[layer_name] = attn_layers[layer_name].get_attn_backend()
                    draft_impl_c8_quant |= getattr(attn_layers[layer_name].impl, "enable_c8_quant", False)

            self.attn_backends = attn_backends
            backend = _get_graph_update_backend(self.attn_groups)
            if issubclass(backend, AscendMLABackend):
                self.attn_architecture = "MLA"
            elif issubclass(backend, AscendAttentionBackend):
                self.attn_architecture = "GQA"
            else:
                self.attn_architecture = None
            # Read each draft MLA layer's final backend selection.
            self._draft_mla_uses_device_lengths = self.attn_architecture == "MLA" and all_mla_use_flashmla(self.model)
            # Window-GQA drafts own their exact KV lengths only on NPU
            # (rejection-corrected positions). FIA's ABI consumes host lists,
            # so stage those lengths with an early non-blocking D2H enqueued
            # right behind the input-preparation kernel: the copy completes
            # while precompute still runs, so the later host wait is free and
            # replaces the blocking tolist() of the NPU tensor inside the
            # metadata build. C8-quantized layers keep the plain factory;
            # their FIA ABI must not share the staged mirror.
            self._use_device_seq_lens_draft = (
                self._window_gqa_dcp_fallback
                and self.attn_architecture == "GQA"
                and not draft_impl_c8_quant
            )
            # A/B mode: derive the host KV lengths purely from the
            # upper-bound arithmetic (num_computed mirror + scheduled + draft
            # width) like the DCP MLA draft does, skipping the staged exact
            # values entirely. On steps with rejected tokens this overstates
            # the length by the rejection count, so the draft reads a few
            # stale slots past its query block; measure the acceptance rate
            # before keeping this mode.
            self._draft_seq_lens_host_bound = (
                self._use_device_seq_lens_draft and envs.VLLM_ASCEND_DRAFT_SEQ_LENS_HOST_BOUND
            )
            if self._use_device_seq_lens_draft and not self._draft_seq_lens_host_bound:
                self._draft_seq_lens_pinned = torch.empty_like(
                    self.input_buffers.seq_lens, device="cpu", pin_memory=True
                )
                self._draft_seq_lens_copy_event = torch.npu.Event()
                dflash_speculator.prepare_dflash_inputs = self._prepare_dflash_inputs_with_length_stage
            else:
                self._draft_seq_lens_pinned = None
                self._draft_seq_lens_copy_event = None
                dflash_speculator.prepare_dflash_inputs = prepare_dflash_inputs_factory(
                    self.vllm_config.cache_config.block_size
                )

    def _prepare_dflash_inputs_with_length_stage(self, *args: Any, **kwargs: Any) -> None:
        """Run the input-preparation kernel, then stage its lengths to host.

        The preparation kernel is the only producer of the draft's exact
        (rejection-corrected) KV lengths on NPU. Enqueue the D2H directly
        behind it so the pinned mirror is ready long before any host
        consumer runs: the copy sits at the front of the device queue, so
        waiting on its event later is effectively free.
        """
        prepare_dflash_inputs(
            *args, **kwargs, kv_cache_block_size=self.vllm_config.cache_config.block_size
        )
        self._draft_seq_lens_pinned.copy_(self.input_buffers.seq_lens, non_blocking=True)
        self._draft_seq_lens_copy_event.record()

    def _prepare_draft_dcp_metadata_inputs(
        self, num_reqs: int, num_reqs_padded: int, step: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        is_prefilling = torch.zeros(num_reqs_padded, dtype=torch.bool)
        if not self.use_dcp:
            if self.attn_architecture == "MLA" and not self._draft_mla_uses_device_lengths:
                # FIA consumes a host list of *valid* KV lengths. The target's
                # optimistic upper bound includes rejected/lookahead tokens;
                # adding the draft width again exposes unwritten cache slots
                # to the non-causal block. Read the lengths produced alongside
                # the actual draft positions instead. One batched blocking
                # transfer is required until FIA accepts device-side lengths;
                # this runs before forward/graph replay, not inside capture.
                seq_lens_cpu = torch.zeros(num_reqs_padded, dtype=torch.int32)
                seq_lens_cpu[:num_reqs].copy_(self.input_buffers.seq_lens[:num_reqs])
                return seq_lens_cpu, is_prefilling
            return None, is_prefilling
        assert self.dcp_manager is not None
        return self.dcp_manager.prepare_draft_dcp_metadata_inputs(
            target_seq_lens_cpu=self.target_input_buffers.seq_lens_cpu,
            is_prefilling=is_prefilling,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            step=step,
            max_model_len=self.max_model_len,
        )

    # Graph capture builds metadata from the padded request shape.
    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        assert self.input_batch is not None
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        seq_lens_cpu, is_prefilling = self._prepare_draft_dcp_metadata_inputs(
            self.input_batch.num_reqs, num_reqs_padded, self.num_query_per_req
        )
        with (
            build_attn_metadata_wrapper(),
            build_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens_padded,
                is_prefilling=is_prefilling,
                seq_lens_cpu=seq_lens_cpu,
                attn_state=AscendAttentionState.ChunkedPrefill,
                parallel_config=self.attn_vllm_config.parallel_config,
            ),
        ):
            # vLLM main (#56181) replaced _build_draft_attn_metadata with
            # _build_uniform_attn_metadata (BatchExecutionDescriptor).
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=num_tokens_padded,
                num_reqs=num_reqs_padded,
            )
            # This build feeds the replay-time task update, which runs only
            # after the graph is enqueued. Mark it so the metadata build
            # waits for the staged exact lengths instead of publishing the
            # FULL-path upper bound.
            self._await_staged_lengths = True
            try:
                attn_metadata = self._build_uniform_attn_metadata(
                    num_reqs=self.input_batch.num_reqs,
                    batch_desc=batch_desc,
                    num_query_per_req=self.num_query_per_req,
                    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                    step=self.num_query_per_req,
                    causal=self._group_causal,
                )
            finally:
                self._await_staged_lengths = False

        return [attn_metadata]

    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: BatchExecutionDescriptor,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        """Build Ascend draft metadata through the upstream attention hook."""
        if self.attn_architecture not in ("GQA", "MLA"):
            # Non-MLA/GQA (e.g. SFA) delegates straight to upstream without
            # Ascend CPU-side DCP preparation.
            return super()._build_attn_metadata(
                num_reqs=num_reqs,
                batch_desc=batch_desc,
                query_start_loc_np=query_start_loc_np,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                causal=causal,
                dcp_local_seq_lens=dcp_local_seq_lens,
            )

        num_reqs_padded = batch_desc.num_reqs or num_reqs
        # The draft forward may process padded inputs outside FULL graphs, but
        # attention only consumes the tokens owned by real requests there.
        num_tokens = batch_desc.num_tokens if batch_desc.cg_mode == CUDAGraphMode.FULL else int(query_start_loc_np[-1])
        seq_lens_cpu, is_prefilling = self._prepare_draft_dcp_metadata_inputs(num_reqs, num_reqs_padded, step)
        if (
            seq_lens_cpu is None
            and getattr(self, "_use_device_seq_lens_draft", False)
            and getattr(self, "_draft_seq_lens_pinned", None) is not None
        ):
            if (
                batch_desc.cg_mode == CUDAGraphMode.FULL
                and not getattr(self, "_await_staged_lengths", False)
                and seq_lens_cpu_upper_bound is not None
            ):
                # FULL replays discard this build's values: the graph's FIA
                # host lists come from the task-update build, which runs
                # after the replay is enqueued. Publish the upper bound and
                # stay synchronization-free -- waiting here would sleep the
                # host before the draft graph is dispatched and let the
                # device run dry mid-step.
                seq_lens_cpu = seq_lens_cpu_upper_bound
            else:
                # Eager drafts consume these values directly, and the
                # task-update build runs only after the graph replay is
                # enqueued. In both positions the event wait cannot starve
                # the device of already-dispatched work.
                self._draft_seq_lens_copy_event.synchronize()
                seq_lens_cpu = self._draft_seq_lens_pinned
        with (
            build_attn_metadata_wrapper(),
            build_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens,
                is_prefilling=is_prefilling,
                seq_lens_cpu=seq_lens_cpu,
                attn_state=AscendAttentionState.ChunkedPrefill,
                parallel_config=self.attn_vllm_config.parallel_config,
                use_device_seq_lens=getattr(self, "_use_device_seq_lens_draft", False),
            ),
        ):
            attn_metadata = super()._build_attn_metadata(
                num_reqs=num_reqs,
                batch_desc=batch_desc,
                query_start_loc_np=query_start_loc_np,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                causal=causal,
                dcp_local_seq_lens=dcp_local_seq_lens,
            )
        if batch_desc.cg_mode == CUDAGraphMode.FULL and attn_metadata is not None:
            # FULL replay uses the captured padded Q, so FIA needs cumulative
            # query lengths through every padded request slot.
            self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)
        return attn_metadata

    def _update_draft_attn_metadata(self, attn_metadata, num_reqs_padded):
        """Match FULL-graph query lengths to the padded request count."""
        query_lens_list = [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)]
        for metadata in attn_metadata.values():
            if getattr(metadata, "external_flashmla", None) is not None:
                continue
            decode_metadata = metadata.decode if self.attn_architecture == "MLA" else metadata
            decode_metadata.actual_seq_lengths_q = query_lens_list
        return attn_metadata

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        if self._window_gqa_dcp_fallback:
            draft_vllm_config = self.attn_vllm_config
            # The upstream implementation already installs the complete
            # forward context and invokes the model with the current runner
            # contract. Temporarily swap only the config so all of those
            # details are retained while attention sees DCP=1.
            target_vllm_config = self.vllm_config
            self.vllm_config = draft_vllm_config
            try:
                with set_current_vllm_config(draft_vllm_config):
                    hidden_states = super()._run_model(
                        num_tokens,
                        attn_metadata,
                        slot_mappings,
                        num_tokens_across_dp,
                        cudagraph_runtime_mode,
                    )
            finally:
                self.vllm_config = target_vllm_config
        else:
            hidden_states = super()._run_model(
                num_tokens, attn_metadata, slot_mappings, num_tokens_across_dp, cudagraph_runtime_mode
            )
        # PCP replicas must propose identical tokens for the next joint target
        # verification. Share the backbone output before sequential Markov sampling.
        hidden_states, _ = AscendPCPManager.broadcast_replicated_hidden_states(
            hidden_states, hidden_states, num_tokens, replicated_pcp=self.replicated_pcp
        )
        return hidden_states

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        self.input_batch = input_batch
        assert self.input_batch is not None
        # TODO: Remove the adaptive verification condition once upstream vLLM
        # synchronizes the draft DP shape independently of the target.
        # Replicated drafts consume global requests, unlike sharded targets.
        # Adaptive verification also negotiates the draft shape independently.
        # Every DP rank must synchronize the draft batch independently.
        sync_state = dp_sync
        if self.replicated_pcp or (self.enable_adaptive_verification and not is_profile):
            sync_state = None
        if dummy_run and skip_attn_for_dummy_run:
            # Profiling runs the draft with its own query token count, which
            # can differ from the target batch. Let forward_context coordinate
            # the actual draft counts instead of reusing the target DP state.
            # TODO: Remove this guard once main2main includes upstream vLLM
            # #54856 (facd9a74a1), which resets the profiling DP counts.
            sync_state = None
        seq_lens_cpu = None
        is_prefilling = torch.from_numpy(self.input_batch.is_prefilling_np)
        if self.use_dcp and self.attn_architecture in ("GQA", "MLA") and not (dummy_run and skip_attn_for_dummy_run):
            # DSpark drafts one block with a fixed step; zero unused request slots
            # before upstream selects the padded batch size and slices this view.
            seq_lens_cpu, is_prefilling = self._prepare_draft_dcp_metadata_inputs(
                input_batch.num_reqs, self.max_num_reqs, self.num_query_per_req
            )
        with (
            build_attn_metadata_wrapper(),
            build_attn_metadata_factory(
                self.input_buffers.positions,
                self.max_num_tokens,
                is_prefilling,
                seq_lens_cpu=seq_lens_cpu,
                parallel_config=self.attn_vllm_config.parallel_config,
            ),
        ):
            draft_tokens = super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                sync_state,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
            )
        return self._public_draft_tokens(draft_tokens)
