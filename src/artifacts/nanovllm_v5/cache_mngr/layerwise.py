from src.core.artifact_base import Artifact
from src.core.service_base import BaseService

from ..attention.flashinfer_attention import (
    Attention,
    store_kvcache,
    read_kvcache,
    read_q_cache,
)
from src.services.nanovllm_v5.engine.sequence import Sequence
import torch

import itertools

from src.services.nanovllm_v5.utils.logging import get_log, set_log
# all implemntation here


class CacheManager(BaseService):
    @property
    def name(self):
        return "CacheManagerLayerwise"

    """
    This version of implementation only 
    """

    def __init__(self, attention_backend: Artifact, config, compressor=None):
        super().__init__()

        self.cu_seqs: list[Sequence]
    
        attention_backend.register(self)

        self.num_layers = config.hf_config.num_hidden_layers

        self.seq_to_layer_block_table = {}
        
        self.cu_page_indices = self.cu_seq_lens = None

        self.compressor = compressor
        self.layer_budget = config.layer_budget
        self.strict_prefill_chunk_size = config.strict_prefill_chunk_size
        self.strict_budget = bool(getattr(compressor, "strict_budget", False))

    def prepare_indices_flashinfer(self, seqs):
        # move to model runner before capturing cuda graph
        self.cu_seqs = seqs
        occupied_pages = 0
        cu_page_indices = torch.tensor(
            list(itertools.chain(*[seq.block_table for seq in seqs]))
        ).to(torch.int32)
        occupied_pages = cu_page_indices.shape[0]
        seq_lens = torch.tensor(
            [len(seq.block_table) for seq in self.cu_seqs]
        ).to(torch.int32)
        self.page_indices = cu_page_indices
        self.seq_lens = seq_lens
        log = get_log()
        log.occupied_pages = occupied_pages
        set_log(log)
        
    def update_indices(self):
        self.prepare_metadata_for_attn(
            self.seq_lens,
            self.page_indices, 
        )

    def update_indices_capture(self, bs: int):
        self.init_forward_metadata_capture_cuda_graph(
            bs,
            self.seq_lens,
            self.page_indices,
        )

    def update_indices_replay(self, bs: int):
        self.init_forward_metadata_replay_cuda_graph(
            bs,
            self.seq_lens,
            self.page_indices, 
        )
        
    def read_and_store_cache(
        self,
        q_cache,
        k_cache,
        v_cache,
        slot_mappings=None,
        layer_id=None,
        question_q_cache=None,
    ):
        """
        option 1: per-sequence handling

        option 2: like flashinfer's layout, handling with packed indices,
        """
        assert len(self.cu_seqs) == 1, "Currently only support single request"
        if slot_mappings is None:
            slot_mappings = []
            for seq in self.cu_seqs:
                slot_mappings.extend(seq.block_table)

        seq = self.cu_seqs[0]
        protected_len = 0
        if getattr(seq, "protected_token_span", None) is not None:
            protected_start, protected_end = seq.protected_token_span
            if protected_start != 0:
                raise ValueError("Protected KV span must start at token 0.")
            protected_len = min(protected_end, len(slot_mappings))
        compress_slot_mappings = slot_mappings[protected_len:]
        if not compress_slot_mappings:
            return len(slot_mappings)
        token_metadata = self._build_table_metadata(
            seq,
            layer_id,
            protected_len,
            len(compress_slot_mappings),
        )

        slot_mappings_tensor = torch.tensor(compress_slot_mappings, device="cuda").to(
            torch.int32
        )

        if question_q_cache is not None and getattr(seq, "question_cache_len", 0) > 0:
            query = question_q_cache[
                seq.query_block_id : seq.query_block_id + 1,
                : seq.question_cache_len,
            ].contiguous()
        else:
            query_slot_mapping = [seq.query_block_id for seq in self.cu_seqs]

            query_slot_mapping_tensor = torch.tensor(query_slot_mapping, device="cuda").to(
                torch.int32
            )

            query = read_q_cache(
                q_cache=q_cache,
                query_slot_mapping=query_slot_mapping_tensor,
            )
            query_cache_len = getattr(seq, "query_cache_len", None)
            if query_cache_len is not None and 0 < query_cache_len < query.shape[1]:
                query = query[:, :query_cache_len]
            question_span = seq.question_token_span
            if question_span is not None:
                start, end = question_span
                q_window_start = max(0, seq.num_tokens - query.shape[1])
                span_start = max(start - q_window_start, 0)
                span_end = min(end - q_window_start, query.shape[1])
                if span_start < span_end:
                    query = query[:, span_start:span_end]

        key, value = read_kvcache(
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=slot_mappings_tensor,
        )

        key = key.unsqueeze(0)
        value = value.unsqueeze(0)

        compressor_kwargs = {}
        if getattr(self.compressor, "requires_table_metadata", False):
            compressor_kwargs["token_metadata"] = token_metadata

        updated_k, updated_v = self.compressor.update_kv(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            **compressor_kwargs,
        )
        if (
            getattr(self.compressor, "requires_table_metadata", False)
            and layer_id is not None
            and getattr(self.compressor, "last_kept_metadata", None) is not None
        ):
            seq.layer_tabula_metadata[layer_id] = self.compressor.last_kept_metadata

        key = updated_k.transpose(1, 2).squeeze(0).contiguous()
        value = updated_v.transpose(1, 2).squeeze(0).contiguous()

        # for single request only
        slot_mappings_tensor = slot_mappings_tensor[: key.shape[0]]

        store_kvcache(
            key=key,
            value=value,
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=slot_mappings_tensor,
        )
        return protected_len + key.shape[0]

    def _metadata_from_token_positions(self, seq, token_positions):
        source = getattr(seq, "tabula_token_metadata", None) or {}
        result = {}
        for name in ("header_columns", "header_cell_ids", "cell_columns"):
            values = source.get(name) or []
            result[name] = [
                values[position] if 0 <= position < len(values) else -1
                for position in token_positions
            ]
        return result

    def _build_table_metadata(self, seq, layer_id, protected_len, compress_len):
        if not getattr(self.compressor, "requires_table_metadata", False):
            return None

        if layer_id is not None and layer_id in seq.layer_tabula_metadata:
            previous_metadata = seq.layer_tabula_metadata[layer_id]
            previous_len = len(previous_metadata.get("header_columns", []))
        else:
            previous_metadata = None
            previous_len = 0

        if getattr(seq, "strict_kv_budget", False) and previous_metadata is not None:
            previous_len = min(previous_len, compress_len)
            result = {
                "header_columns": previous_metadata.get("header_columns", [])[:previous_len],
                "header_cell_ids": previous_metadata.get("header_cell_ids", [])[:previous_len],
                "cell_columns": previous_metadata.get("cell_columns", [])[:previous_len],
            }
            new_len = compress_len - previous_len
            if new_len > 0:
                if not getattr(seq, "prompt_prefill_done", False):
                    start = seq.prompt_prefill_cursor
                else:
                    start = seq.num_tokens
                new_positions = list(range(start, start + new_len))
                new_metadata = self._metadata_from_token_positions(seq, new_positions)
                for name in ("header_columns", "header_cell_ids", "cell_columns"):
                    result[name].extend(new_metadata[name])
            return result

        if getattr(seq, "strict_kv_budget", False):
            start = protected_len
            token_positions = list(range(start, start + compress_len))
        else:
            token_positions = list(range(protected_len, protected_len + compress_len))
        return self._metadata_from_token_positions(seq, token_positions)

    def read_and_store_cache_for_seq(
        self,
        seq,
        q_cache,
        k_cache,
        v_cache,
        slot_mappings,
        question_q_cache=None,
    ):
        previous_cu_seqs = getattr(self, "cu_seqs", None)
        self.cu_seqs = [seq]
        try:
            return self.read_and_store_cache(
                q_cache,
                k_cache,
                v_cache,
                slot_mappings,
                question_q_cache=question_q_cache,
            )
        finally:
            if previous_cu_seqs is not None:
                self.cu_seqs = previous_cu_seqs
