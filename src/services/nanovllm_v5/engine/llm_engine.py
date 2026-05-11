import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from ..config import Config
from ..sampling_params import SamplingParams
from .sequence import Sequence
from .scheduler import Scheduler
from src.services.nanovllm_v5.model_runner import ModelRunner

from src.services.nanovllm_v5.utils.logging import get_log, LogCollector


def is_strict_budget_compressor(cache_compressor: str) -> bool:
    return cache_compressor.lower() in {
        "rkv_new",
        "snapkv_new",
        "question_new",
        "sliding_new",
        "sliding_window_new",
        "tabula",
        "tabula_new",
        "tabulakv",
    }


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        self.log_collector = LogCollector()
        
        self.model_runner = ModelRunner(config, 0, self.events)
        self.scheduler = Scheduler(config)
        
        self.scheduler.block_manager._register_method("_deallocate_block", self.model_runner.cache_mngr)
        self.scheduler.block_manager._register_method("append_token_blocks", self.model_runner.cache_mngr)
        self.scheduler.block_manager._register_obj("blocks", self.model_runner.cache_mngr)
        
        self.model_runner.cache_mngr._register_obj("seq_to_layer_block_table", self.scheduler.block_manager)
        self.model_runner.cache_mngr._register_obj("num_layers", self.scheduler.block_manager)
                
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        self.cur_step = 0
        
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int] | dict, sampling_params: SamplingParams):
        question_token_span = None
        question_token_ids = None
        protected_token_span = None
        tabula_token_metadata = None
        if isinstance(prompt, dict):
            question_token_span = prompt.get("question_token_span")
            question_token_ids = prompt.get("question_token_ids")
            protected_token_span = prompt.get("protected_token_span")
            tabula_token_metadata = prompt.get("tabula_token_metadata")
            prompt = prompt["token_ids"] if "token_ids" in prompt else prompt["prompt"]
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence.from_prompt(
            prompt,
            sampling_params,
            self.config.kvcache_block_size,
            self.config.query_window_size,
            question_token_span,
            protected_token_span,
        )
        seq.question_token_ids = question_token_ids
        seq.tabula_token_metadata = tabula_token_metadata
        seq.strict_kv_budget = is_strict_budget_compressor(self.config.cache_compressor)
        seq.strict_prefill_chunk_size = self.config.strict_prefill_chunk_size
        if seq.strict_kv_budget:
            if self.config.kvcache_block_size != 1:
                raise ValueError("Strict-budget *_new compressors require kvcache_block_size=1.")
            if self.config.strict_prefill_chunk_size <= 0:
                raise ValueError("strict_prefill_chunk_size must be positive.")
            if self.config.strict_prefill_chunk_size > self.config.layer_budget:
                raise ValueError(
                    "strict_prefill_chunk_size must be <= layer_budget for strict-budget *_new compressors."
                )
            if self.config.cache_compressor.lower() in {"question_new", "tabula", "tabula_new", "tabulakv"}:
                if not seq.question_token_ids:
                    raise ValueError(
                        f"{self.config.cache_compressor} requires question_token_ids in prompt input."
                    )
                if self.config.question_window_size <= 0:
                    raise ValueError(
                        f"question_window_size must be positive for {self.config.cache_compressor}."
                    )
            protected_end = protected_token_span[1] if protected_token_span else 0
            if protected_end > self.config.protected_kv_cache_size:
                raise ValueError(
                    "The protected few-shot KV prefix is larger than "
                    f"protected_kv_cache_size: protected_tokens={protected_end}, "
                    f"protected_kv_cache_size={self.config.protected_kv_cache_size}. "
                    "Increase model.protected_kv_cache_size in the eval config."
                )
            first_chunk_end = min(seq.num_prompt_tokens, self.config.strict_prefill_chunk_size)
            first_chunk_end = max(first_chunk_end, min(protected_end, seq.num_prompt_tokens))
            seq.num_tokens = first_chunk_end
            seq.last_token = seq.token_ids[first_chunk_end - 1]
            seq.cached_kv_len_before_chunk = 0
            seq.query_cache_len = min(seq.query_window_size, first_chunk_end)
        self.scheduler.add(seq)

    def _strict_budget_enabled(self) -> bool:
        return is_strict_budget_compressor(self.config.cache_compressor)

    def _advance_strict_prefill_chunk(self, seq: Sequence) -> None:
        chunk_start = seq.num_tokens
        chunk_end = min(
            seq.num_prompt_tokens,
            chunk_start + self.config.strict_prefill_chunk_size,
        )
        chunk_token_ids = seq.token_ids[chunk_start:chunk_end]
        if len(self.scheduler.block_manager.free_block_ids) < len(chunk_token_ids):
            raise RuntimeError(
                "Not enough KV cache blocks for strict-budget prefill chunk. "
                "Increase gpu_memory_utilization or lower strict_prefill_chunk_size."
            )
        seq.prompt_prefill_cursor = chunk_start
        seq.num_cached_tokens = chunk_start
        seq.num_tokens = chunk_end
        seq.last_token = seq.token_ids[chunk_end - 1]
        seq.cached_kv_len_before_chunk = len(seq.block_table)
        self.scheduler.block_manager.append_token_blocks(seq, chunk_token_ids)
        self.scheduler.query_block_manager.append_tokens(seq, chunk_token_ids)

    def _run_strict_prefill(self, seqs: list[Sequence]) -> list[int]:
        if len(seqs) != 1:
            raise ValueError("Strict-budget *_new compressors currently support batch_size=1.")
        seq = seqs[0]
        final_next_token = None
        if self.config.cache_compressor.lower() in {"question_new", "tabula", "tabula_new", "tabulakv"}:
            self.model_runner.call("prepare_question_cache", seqs)
        while True:
            token_ids = self.model_runner.call("strict_prefill_chunk", seqs)
            if token_ids:
                final_next_token = token_ids[-1]
            protected_end = seq.protected_token_span[1] if seq.protected_token_span else 0
            active_budget = self.config.layer_budget + min(
                protected_end,
                self.config.protected_kv_cache_size,
            )
            if len(seq.block_table) > active_budget:
                raise RuntimeError(
                    f"Strict KV budget violated: active_kv={len(seq.block_table)} "
                    f"> protected_kv + layer_budget={active_budget}"
                )
            if seq.num_tokens >= seq.num_prompt_tokens:
                seq.mark_prompt_prefilled()
                seq.last_token = seq.token_ids[seq.num_prompt_tokens - 1]
                seq.cached_kv_len_before_chunk = len(seq.block_table)
                break
            self._advance_strict_prefill_chunk(seq)
        if final_next_token is None:
            raise RuntimeError("Strict-budget prefill did not produce a next-token candidate.")
        return [final_next_token]

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if is_prefill and self._strict_budget_enabled():
            token_ids = self._run_strict_prefill(seqs)
        else:
            token_ids = self.model_runner.call("run", seqs, is_prefill)
        cache_compressor = self.config.cache_compressor.lower()
        if (
            cache_compressor not in ("", "none", "full", "full_kv")
            and not is_prefill
            and self.cur_step % self.config.steps_between_cache_compressions == 0
        ):
            self.model_runner.call("compress")
        self.cur_step += 1
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[dict],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            # self.log_collector.append(perf_counter(), get_log().occupied_pages)
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                    "Occupied Pages": get_log().occupied_pages,
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
            pbar.update(1)
        self.log_collector.save(self.config.log_path)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
