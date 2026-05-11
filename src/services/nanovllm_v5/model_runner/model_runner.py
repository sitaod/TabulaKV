import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from ..config import Config
from ..engine.sequence import Sequence
from .models.qwen3 import Qwen3ForCausalLM
from .layers.sampler import Sampler
from ..utils.context import set_context, get_context, reset_context
from ..utils.loader import load_model

from src.core.service_base import BaseService
import itertools
from ..engine.io_struct import SamplingInfo

from src.services.nanovllm_v5.utils.context import set_cuda_graph_flag

from src.artifacts.nanovllm_v5.cache_mngr.layerwise import CacheManager

from src.artifacts.nanovllm_v5.cache_mngr.snapKV import SnapKV
from src.artifacts.nanovllm_v5.cache_mngr.snapKV_new import SnapKVNew
from src.artifacts.nanovllm_v5.cache_mngr.RKV import RKV
from src.artifacts.nanovllm_v5.cache_mngr.RKV_new import RKVNew
from src.artifacts.nanovllm_v5.cache_mngr.sliding import SlidingWindowKV
from src.artifacts.nanovllm_v5.cache_mngr.sliding_new import SlidingWindowKVNew
from src.artifacts.nanovllm_v5.cache_mngr.query import QueryAwareKV
from src.artifacts.nanovllm_v5.cache_mngr.question_new import QuestionAwareKVNew
from src.artifacts.nanovllm_v5.cache_mngr.tabula import TabulaKV

from src.services.nanovllm_v5.model_runner.models.qwen3 import Qwen3AttentionArtifacts

from enum import Enum

class RunningStage(Enum):
    WARMUP = 1
    INFERENCE = 2

stage = RunningStage.WARMUP


def build_compressor(config: Config):
    compressor_name = config.cache_compressor.lower()
    if compressor_name in ("", "none", "full", "full_kv"):
        return None
    if compressor_name == "snapkv":
        return SnapKV(window_size=config.query_window_size, budget=config.layer_budget)
    if compressor_name == "snapkv_new":
        return SnapKVNew(window_size=config.query_window_size, budget=config.layer_budget)
    if compressor_name == "rkv":
        return RKV(window_size=config.query_window_size, budget=config.layer_budget)
    if compressor_name == "rkv_new":
        return RKVNew(window_size=config.query_window_size, budget=config.layer_budget)
    if compressor_name in ("sliding", "sliding_window"):
        return SlidingWindowKV(budget=config.layer_budget)
    if compressor_name in ("sliding_new", "sliding_window_new"):
        return SlidingWindowKVNew(budget=config.layer_budget)
    if compressor_name in ("query", "query_aware", "tablekv_query"):
        return QueryAwareKV(window_size=config.query_window_size, budget=config.layer_budget)
    if compressor_name in ("question_new", "question_aware_new"):
        return QuestionAwareKVNew(window_size=config.query_window_size, budget=config.layer_budget)
    if compressor_name in ("tabula", "tabula_new", "tabulakv"):
        return TabulaKV(
            window_size=config.query_window_size,
            budget=config.layer_budget,
            tabula_lambda=config.tabula_lambda,
        )
    raise ValueError(f"Unknown cache_compressor: {config.cache_compressor}")


class ModelRunner(BaseService):
    @property
    def name(self):
        return f"ModelRunner-Rank{self.rank}"

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        BaseService.__init__(self)
        self.config = config
        hf_config = config.hf_config
        self.query_window_size = config.query_window_size
        self.question_window_size = config.question_window_size
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        self.device = torch.device("cuda", rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        self.attention_backend = Qwen3AttentionArtifacts.init_new(self, hf_config)
        self.attention_backend.register(self)
        self.model = Qwen3ForCausalLM(self.attention_backend, hf_config)
        load_model(self.model, config.model)
        
        self.compressor = build_compressor(config)
        
        self.cache_mngr = CacheManager(self.attention_backend, config, self.compressor)
        self.cache_mngr._register_method("prepare_indices_flashinfer", self)
        self.cache_mngr._register_method("update_indices", self)
        self.cache_mngr._register_method("update_indices_capture", self)
        self.cache_mngr._register_method("update_indices_replay", self)

        self.sampler = Sampler()
        global stage
        stage = RunningStage.WARMUP
        # self.warmup_model()
        stage = RunningStage.INFERENCE
        
        self.allocate_kv_cache()
        if not self.enforce_eager:
            set_cuda_graph_flag()
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def compress(self, seqs=None):
        if self.cache_mngr.compressor is None:
            return
        if seqs is not None:
            self.cache_mngr.cu_seqs = seqs
        assert len(self.cache_mngr.cu_seqs) == 1, "Currently only support single request"
        seq = self.cache_mngr.cu_seqs[0]
        original_block_table = list(seq.block_table)
        compressed_len = None
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache") and hasattr(module, "q_cache"):
                layer_len = self.cache_mngr.read_and_store_cache(
                    module.q_cache,
                    module.k_cache,
                    module.v_cache,
                    original_block_table,
                    layer_id=layer_id,
                    question_q_cache=getattr(module, "question_q_cache", None)
                    if getattr(self.compressor, "requires_question_cache", False)
                    else None,
                )
                if compressed_len is None:
                    compressed_len = layer_len
                else:
                    assert compressed_len == layer_len
                layer_id += 1

        if compressed_len is None:
            return
        evicted_block_ids = original_block_table[compressed_len:]
        seq.block_table = original_block_table[:compressed_len]
        if self.rank == 0:
            for block_id in evicted_block_ids:
                self.cache_mngr._deallocate_block(block_id)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence.from_prompt([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        try:
            self.q_cache = torch.zeros(hf_config.num_hidden_layers, self.config.max_num_seqs, self.query_window_size, hf_config.num_attention_heads, hf_config.head_dim, dtype=hf_config.torch_dtype)
            if getattr(self.compressor, "requires_question_cache", False):
                self.question_q_cache = torch.zeros(
                    hf_config.num_hidden_layers,
                    self.config.max_num_seqs,
                    self.question_window_size,
                    hf_config.num_attention_heads,
                    hf_config.head_dim,
                    dtype=hf_config.torch_dtype,
                )
            else:
                self.question_q_cache = None
        except:
            raise ValueError(
                "Not enough memory for q_cache/question_q_cache, try to lower memory "
                "occupation of other process or lower \"config.max_num_seqs\""
            )
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        available_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        if getattr(self.compressor, "strict_budget", False):
            required_blocks = (
                config.protected_kv_cache_size
                + config.layer_budget
                + config.strict_prefill_chunk_size
            )
            if available_blocks < required_blocks:
                budget_gb = total * config.gpu_memory_utilization / (1024**3)
                used_gb = used / (1024**3)
                peak_gb = peak / (1024**3)
                current_gb = current / (1024**3)
                block_mb = block_bytes / (1024**2)
                raise ValueError(
                    "Not enough GPU memory left for strict-budget KV cache. "
                    f"required_blocks={required_blocks}, available_blocks={available_blocks}, "
                    f"gpu_memory_utilization budget={budget_gb:.2f} GiB, "
                    f"used={used_gb:.2f} GiB, peak={peak_gb:.2f} GiB, "
                    f"current={current_gb:.2f} GiB, block={block_mb:.2f} MiB."
                )
            config.num_kvcache_blocks = required_blocks
        else:
            config.num_kvcache_blocks = available_blocks
        if config.num_kvcache_blocks <= 0:
            budget_gb = total * config.gpu_memory_utilization / (1024**3)
            used_gb = used / (1024**3)
            peak_gb = peak / (1024**3)
            current_gb = current / (1024**3)
            block_mb = block_bytes / (1024**2)
            raise ValueError(
                "Not enough GPU memory left for KV cache. "
                f"gpu_memory_utilization budget={budget_gb:.2f} GiB, "
                f"used={used_gb:.2f} GiB, peak={peak_gb:.2f} GiB, "
                f"current={current_gb:.2f} GiB, block={block_mb:.2f} MiB. "
                "For 32B single-GPU evaluation, increase model.gpu_memory_utilization "
                "or reduce model.max_num_seqs/model.max_model_len in eval/config_eval.yaml."
            )
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache") and hasattr(module, "q_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                module.q_cache = self.q_cache[layer_id]
                module.question_q_cache = (
                    self.question_q_cache[layer_id]
                    if self.question_q_cache is not None
                    else None
                )
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_question_prefill(self, seqs: list[Sequence]):
        assert len(seqs) == 1, "question_new currently only supports single request"
        seq = seqs[0]
        if not seq.question_token_ids:
            raise ValueError("question_new requires question_token_ids in the prompt input.")

        question_token_ids = seq.question_token_ids
        question_start = 0
        if seq.question_token_span is not None:
            question_start = seq.question_token_span[0]
        if len(question_token_ids) > self.question_window_size:
            start = len(question_token_ids) - self.question_window_size
            question_token_ids = question_token_ids[start:]
            question_start += start

        seq.question_cache_len = len(question_token_ids)
        input_ids = torch.tensor(
            question_token_ids,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True)
        positions = torch.tensor(
            list(range(question_start, question_start + seq.question_cache_len)),
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True)
        cu_seqlens = torch.tensor(
            [0, seq.question_cache_len],
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        query_slot_mapping = torch.tensor(
            list(
                range(
                    seq.query_block_id * self.question_window_size,
                    seq.query_block_id * self.question_window_size + seq.question_cache_len,
                )
            ),
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        query_window_pos = torch.arange(
            seq.question_cache_len,
            dtype=torch.int32,
            device=input_ids.device,
        )

        set_context(
            True,
            cu_seqlens,
            cu_seqlens,
            seq.question_cache_len,
            seq.question_cache_len,
            torch.empty(0, dtype=torch.int32, device=input_ids.device),
            None,
            None,
            query_slot_mapping,
            query_window_pos,
            None,
            True,
            True,
        )
        return input_ids, positions

    def prepare_prefill(self, seqs: list[Sequence]):
        self.cache_mngr.cu_seqs = seqs
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        query_window_pos = []
        query_slot_mapping = []
        prefill_cache_slot_mapping = []
        block_tables = None

        for seq in seqs:
            if getattr(seq, "strict_kv_budget", False) and not seq.prompt_prefill_done:
                assert self.block_size == 1, "Strict-budget prefill currently requires kvcache_block_size=1"
                chunk_start = seq.prompt_prefill_cursor
                chunk_end = seq.num_tokens
                seqlen_q = chunk_end - chunk_start
                assert seqlen_q > 0
                cached_kv_len = getattr(
                    seq,
                    "cached_kv_len_before_chunk",
                    max(len(seq.block_table) - seqlen_q, 0),
                )

                input_ids.extend(seq.token_ids[chunk_start:chunk_end])
                positions.extend(list(range(chunk_start, chunk_end)))
                seqlen_k = cached_kv_len + seqlen_q
                cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
                cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
                max_seqlen_q = max(seqlen_q, max_seqlen_q)
                max_seqlen_k = max(seqlen_k, max_seqlen_k)

                query_cache_len = min(seq.query_window_size, seqlen_q)
                seq.query_cache_len = query_cache_len
                query_window_pos.extend(list(range(seqlen_q - query_cache_len, seqlen_q)))
                query_slot_mapping.extend(
                    list(
                        range(
                            seq.query_block_id * self.query_window_size,
                            seq.query_block_id * self.query_window_size + query_cache_len,
                        )
                    )
                )

                for block_id in seq.block_table[cached_kv_len:cached_kv_len + seqlen_q]:
                    slot_mapping.append(block_id * self.block_size)
                if cached_kv_len > 0:
                    for block_id in seq.block_table[:cached_kv_len + seqlen_q]:
                        prefill_cache_slot_mapping.append(block_id * self.block_size)
                continue

            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            # prepare query window metadata
            query_window_pos.extend(list(range(seq.num_tokens - seq.query_window_num_tokens, seq.num_tokens)))
            query_slot_mapping.extend(list(range(seq.query_block_id * self.query_window_size, seq.query_block_id * self.query_window_size + seq.query_window_num_tokens)))

            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1] and not prefill_cache_slot_mapping:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        query_slot_mapping = torch.tensor(query_slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        query_window_pos = torch.tensor(query_window_pos, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        if prefill_cache_slot_mapping:
            prefill_cache_slot_mapping = torch.tensor(
                prefill_cache_slot_mapping,
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True)
        else:
            prefill_cache_slot_mapping = None

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            query_slot_mapping,
            query_window_pos,
            prefill_cache_slot_mapping,
        )
        
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        query_slot_mapping = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
            
            # prepare query window metadata
            query_slot_mapping.append(seq.query_block_id * self.query_window_size + seq.last_query_window_index)
            if getattr(seq, "strict_kv_budget", False):
                seq.query_cache_len = min(
                    seq.query_window_size,
                    getattr(seq, "query_cache_len", seq.query_window_size) + 1,
                )
            
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        query_slot_mapping = torch.tensor(query_slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, query_slot_mapping=query_slot_mapping)
        
        self.prepare_indices_flashinfer(seqs)
        
        if not self.enforce_eager and stage != RunningStage.WARMUP:
            # cuda_graph enabled
            self.update_indices_replay(bs=len(seqs))
        else:
            self.update_indices()
        return input_ids, positions

    # def prepare_sample(self, seqs: list[Sequence]):
    #     temperatures = []
    #     for seq in seqs:
    #         temperatures.append(seq.temperature)
    #     temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
    #     return temperatures

    def prepare_sample(self, seqs: list[Sequence]):
        return SamplingInfo.from_sequence(seqs)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["query_slot_mapping"][:bs] = context.query_slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        sampling_infos = self.prepare_sample(seqs).to(input_ids.device) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, sampling_infos).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def strict_prefill_chunk(self, seqs: list[Sequence]) -> list[int] | None:
        assert len(seqs) == 1, "Strict-budget prefill currently only supports single request"
        input_ids, positions = self.prepare_prefill(seqs)
        sampling_infos = self.prepare_sample(seqs).to(input_ids.device) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, True)
        token_ids = self.sampler(logits[-len(seqs):], sampling_infos).tolist() if self.rank == 0 else None
        reset_context()
        self.compress(seqs)
        return token_ids

    @torch.inference_mode()
    def prepare_question_cache(self, seqs: list[Sequence]):
        assert getattr(self.compressor, "requires_question_cache", False)
        assert self.question_q_cache is not None
        input_ids, positions = self.prepare_question_prefill(seqs)
        try:
            self.model(input_ids, positions)
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        query_slot_mapping = torch.zeros(max_bs, dtype=torch.int32) 
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        
        seqs = [Sequence.for_capture([0]) for _ in range(max_bs)]

        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], query_slot_mapping=query_slot_mapping[:bs])
            # self.init_forward_metadata_capture_cuda_graph(bs, seq_lens[:bs], cu_page_indices)
            self.prepare_indices_flashinfer(seqs[:bs])
            self.update_indices_capture(bs)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            query_slot_mapping=query_slot_mapping, 
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
