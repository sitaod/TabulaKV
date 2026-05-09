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
    
    @classmethod
    def for_capture(cls, block_table: list[int]):
        seq = cls()
        seq.seq_id = next(Sequence.cuda_graph_counter)
        seq.block_table = block_table
        seq.num_tokens = len(block_table) * cls.block_size
        return seq
    
    @classmethod
    def from_prompt(cls, token_ids: list[int], sampling_params = SamplingParams(), kvcache_block_size = 1, query_window_size = 128):
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
        
        seq.block_table = []
        seq.temperature = sampling_params.temperature
        seq.top_k = sampling_params.top_k
        seq.top_p = sampling_params.top_p
        seq.min_p = sampling_params.min_p
        seq.max_tokens = sampling_params.max_tokens
        seq.ignore_eos = sampling_params.ignore_eos
        
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

    def __getstate__(self):
        return {
            "seq_id": self.seq_id,
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "block_table": self.block_table,
            "block_size": self.block_size,
            "query_window_size": self.query_window_size,
            "query_block_id": self.query_block_id,
            "last_query_window_index": self.last_query_window_index,
            "token_state": self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        }

    def __setstate__(self, state):
        if not isinstance(state, dict):
            self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
            self.block_size = Sequence.block_size
            self.query_window_size = Sequence.query_window_size
            self.query_block_id = -1
            self.last_query_window_index = -1
            token_state = state[-1]
        else:
            self.seq_id = state["seq_id"]
            self.num_tokens = state["num_tokens"]
            self.num_prompt_tokens = state["num_prompt_tokens"]
            self.num_cached_tokens = state["num_cached_tokens"]
            self.block_table = state["block_table"]
            self.block_size = state["block_size"]
            self.query_window_size = state["query_window_size"]
            self.query_block_id = state["query_block_id"]
            self.last_query_window_index = state["last_query_window_index"]
            token_state = state["token_state"]
        if self.num_completion_tokens == 0:
            self.token_ids = token_state
            self.last_token = self.token_ids[-1]
        else:
            self.last_token = token_state
