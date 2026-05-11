from copy import copy
from enum import Enum, auto
from itertools import count

from ..sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    query_window_size = 128
    block_size = 1
    counter = count()
    cuda_graph_counter = count()
    
    def __init__(self):
        self.block_table: list[int] = []
        self.query_block_id: int = -1
        self.last_query_window_index: int = -1
        self.num_tokens: int = 0
        self.num_prompt_tokens: int = 0
        self.num_cached_tokens: int = 0
        self.prompt_prefill_cursor: int = 0
        self.prompt_prefill_done: bool = False
        self.strict_kv_budget: bool = False
        self.strict_prefill_chunk_size: int = 0
        self.cached_kv_len_before_chunk: int = 0
        self.query_cache_len: int | None = None
        self.question_token_ids: list[int] | None = None
        self.question_cache_len: int = 0
        self.protected_token_span: tuple[int, int] | None = None
        self.tabula_token_metadata: dict[str, list[int]] | None = None
        self.layer_tabula_metadata: dict[int, dict[str, list[int]]] = {}
    
    @classmethod
    def for_capture(cls, block_table: list[int]):
        seq = cls()
        seq.seq_id = next(Sequence.cuda_graph_counter)
        seq.block_table = block_table
        seq.num_tokens = len(block_table) * cls.block_size
        return seq
    
    @classmethod
    def from_prompt(
        cls,
        token_ids: list[int],
        sampling_params=SamplingParams(),
        kvcache_block_size=1,
        query_window_size=128,
        question_token_span: tuple[int, int] | None = None,
        protected_token_span: tuple[int, int] | None = None,
    ):
        seq = cls()
        seq.block_size = kvcache_block_size
        seq.query_window_size = query_window_size
        seq.seq_id = next(Sequence.counter)
        seq.status = SequenceStatus.WAITING
        seq.token_ids = copy(token_ids)
        seq.last_token = token_ids[-1]
        seq.num_tokens = len(seq.token_ids)
        seq.num_prompt_tokens = len(token_ids)
        seq.num_cached_tokens = 0
        seq.prompt_prefill_cursor = 0
        seq.prompt_prefill_done = False
        seq.strict_kv_budget = False
        seq.strict_prefill_chunk_size = 0
        seq.cached_kv_len_before_chunk = 0
        seq.query_cache_len = None
        seq.question_token_ids = None
        seq.question_cache_len = 0
        seq.protected_token_span = protected_token_span
        seq.tabula_token_metadata = None
        seq.layer_tabula_metadata = {}
        
        seq.block_table = []
        seq.temperature = sampling_params.temperature
        seq.top_k = sampling_params.top_k
        seq.top_p = sampling_params.top_p
        seq.min_p = sampling_params.min_p
        seq.max_tokens = sampling_params.max_tokens
        seq.ignore_eos = sampling_params.ignore_eos
        seq.question_token_span = question_token_span

        return seq

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def query_window_num_tokens(self):
        return min(self.query_window_size, self.num_tokens)

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert -1 <= i < self.num_blocks
        if i == -1:
            return self.token_ids[-self.block_size:]
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def mark_prompt_prefilled(self):
        self.prompt_prefill_cursor = self.num_prompt_tokens
        self.num_cached_tokens = self.num_prompt_tokens
        self.prompt_prefill_done = True

    def __getstate__(self):
        return {
            "seq_id": self.seq_id,
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "prompt_prefill_cursor": self.prompt_prefill_cursor,
            "prompt_prefill_done": self.prompt_prefill_done,
            "strict_kv_budget": self.strict_kv_budget,
            "strict_prefill_chunk_size": self.strict_prefill_chunk_size,
            "cached_kv_len_before_chunk": self.cached_kv_len_before_chunk,
            "query_cache_len": self.query_cache_len,
            "question_token_ids": self.question_token_ids,
            "question_cache_len": self.question_cache_len,
            "protected_token_span": self.protected_token_span,
            "tabula_token_metadata": self.tabula_token_metadata,
            "layer_tabula_metadata": self.layer_tabula_metadata,
            "block_table": self.block_table,
            "block_size": self.block_size,
            "query_window_size": self.query_window_size,
            "query_block_id": self.query_block_id,
            "last_query_window_index": self.last_query_window_index,
            "question_token_span": self.question_token_span,
            "token_state": self.token_ids if self.num_tokens <= self.num_prompt_tokens else self.last_token,
        }

    def __setstate__(self, state):
        if not isinstance(state, dict):
            self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
            self.prompt_prefill_cursor = self.num_cached_tokens
            self.prompt_prefill_done = self.prompt_prefill_cursor >= self.num_prompt_tokens
            self.strict_kv_budget = False
            self.strict_prefill_chunk_size = 0
            self.cached_kv_len_before_chunk = 0
            self.query_cache_len = None
            self.question_token_ids = None
            self.question_cache_len = 0
            self.protected_token_span = None
            self.tabula_token_metadata = None
            self.layer_tabula_metadata = {}
            self.block_size = Sequence.block_size
            self.query_window_size = Sequence.query_window_size
            self.query_block_id = -1
            self.last_query_window_index = -1
            self.question_token_span = None
            token_state = state[-1]
        else:
            self.seq_id = state["seq_id"]
            self.num_tokens = state["num_tokens"]
            self.num_prompt_tokens = state["num_prompt_tokens"]
            self.num_cached_tokens = state["num_cached_tokens"]
            self.prompt_prefill_cursor = state.get("prompt_prefill_cursor", self.num_cached_tokens)
            self.prompt_prefill_done = state.get(
                "prompt_prefill_done",
                self.prompt_prefill_cursor >= self.num_prompt_tokens,
            )
            self.strict_kv_budget = state.get("strict_kv_budget", False)
            self.strict_prefill_chunk_size = state.get("strict_prefill_chunk_size", 0)
            self.cached_kv_len_before_chunk = state.get("cached_kv_len_before_chunk", 0)
            self.query_cache_len = state.get("query_cache_len")
            self.question_token_ids = state.get("question_token_ids")
            self.question_cache_len = state.get("question_cache_len", 0)
            self.protected_token_span = state.get("protected_token_span")
            self.tabula_token_metadata = state.get("tabula_token_metadata")
            self.layer_tabula_metadata = state.get("layer_tabula_metadata", {})
            self.block_table = state["block_table"]
            self.block_size = state["block_size"]
            self.query_window_size = state["query_window_size"]
            self.query_block_id = state["query_block_id"]
            self.last_query_window_index = state["last_query_window_index"]
            self.question_token_span = state.get("question_token_span")
            token_state = state["token_state"]
        if self.num_tokens <= self.num_prompt_tokens:
            self.token_ids = token_state
            self.last_token = self.token_ids[-1]
        else:
            self.last_token = token_state
