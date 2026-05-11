import torch

from .utils import compute_attention_scores


class QuestionAwareKVNew:
    """Strict-budget question-aware iterative KV compression.

    The question states score the union of the previous compressed cache and
    the current prefill chunk. The most recent window is always retained, and
    the remaining budget is filled by top-scored history positions.
    """

    strict_budget = True
    requires_question_cache = True

    def __init__(
        self,
        budget=128,
        window_size=32,
        score_reduce="max",
        record_kept_token_indices=False,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        assert score_reduce in ("mean", "max"), "score_reduce must be 'mean' or 'max'"
        self.budget = budget
        self.window_size = window_size
        self.score_reduce = score_reduce
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []

    def update_kv(self, query_states, key_states, value_states):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len <= self.budget:
            return key_states, value_states

        attn_weights = compute_attention_scores(query_states, key_states)
        history_scores = torch.softmax(
            attn_weights[:, :, :, : -self.window_size],
            dim=-1,
            dtype=torch.float32,
        )
        if self.score_reduce == "mean":
            history_scores = history_scores.mean(dim=-2).to(query_states.dtype)
        else:
            history_scores = history_scores.max(dim=-2).values.to(query_states.dtype)

        history_budget = self.budget - self.window_size
        indices = history_scores.topk(history_budget, dim=-1).indices

        if self.record_kept_token_indices:
            indices_cpu = indices.clone().squeeze(0).to("cpu")
            recent_indices = torch.arange(
                kv_cache_len - self.window_size,
                kv_cache_len,
                device="cpu",
            ).expand(indices_cpu.shape[0], -1)
            self.kept_token_indices.append(torch.cat([indices_cpu, recent_indices], dim=-1))
            self.kept_attention_scores.append(
                torch.gather(history_scores.squeeze(0).to("cpu"), dim=1, index=indices_cpu)
            )
            self.evicted_token_num += kv_cache_len - self.budget

        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_history = key_states[:, :, : -self.window_size, :].gather(
            dim=2,
            index=gather_indices,
        )
        v_history = value_states[:, :, : -self.window_size, :].gather(
            dim=2,
            index=gather_indices,
        )
        k_recent = key_states[:, :, -self.window_size :, :]
        v_recent = value_states[:, :, -self.window_size :, :]
        return (
            torch.cat([k_history, k_recent], dim=2),
            torch.cat([v_history, v_recent], dim=2),
        )
