import torch

from .utils import compute_attention_scores


class TabulaKV:
    """Strict-budget table-aware question-aware KV compression."""

    strict_budget = True
    requires_question_cache = True
    requires_table_metadata = True

    def __init__(
        self,
        budget=128,
        window_size=32,
        tabula_lambda=0.5,
        score_reduce="max",
        **kwargs,
    ):
        assert budget > 0, "budget must be positive"
        assert 0.0 <= tabula_lambda <= 1.0, "tabula_lambda must be in [0, 1]"
        assert score_reduce in ("mean", "max"), "score_reduce must be 'mean' or 'max'"
        self.budget = budget
        self.window_size = window_size
        self.tabula_lambda = tabula_lambda
        self.score_reduce = score_reduce
        self.last_kept_metadata = None

    def _metadata_tensor(self, token_metadata, name, kv_cache_len, device):
        if token_metadata is None:
            return torch.full((kv_cache_len,), -1, dtype=torch.long, device=device)
        values = token_metadata.get(name) or []
        if len(values) < kv_cache_len:
            values = [*values, *([-1] * (kv_cache_len - len(values)))]
        return torch.tensor(values[:kv_cache_len], dtype=torch.long, device=device)

    def _apply_header_scores(self, token_scores, header_columns, cell_columns):
        adjusted_scores = token_scores.clone()
        header_mask = header_columns >= 0
        cell_mask = cell_columns >= 0
        if not header_mask.any() or not cell_mask.any():
            return adjusted_scores

        for column in torch.unique(cell_columns[cell_mask]):
            column_header_mask = header_columns == column
            if not column_header_mask.any():
                continue
            header_score = token_scores[column_header_mask].max()
            column_cell_mask = cell_columns == column
            adjusted_scores[column_cell_mask] = (
                self.tabula_lambda * token_scores[column_cell_mask]
                + (1.0 - self.tabula_lambda) * header_score
            )
        return adjusted_scores

    def _select_indices(self, adjusted_scores, header_columns, header_cell_ids):
        kv_cache_len = adjusted_scores.shape[0]
        if kv_cache_len <= self.budget:
            return torch.arange(kv_cache_len, device=adjusted_scores.device)

        required_header_indices = []
        for cell_id in torch.unique(header_cell_ids[header_cell_ids >= 0]):
            cell_indices = torch.nonzero(header_cell_ids == cell_id, as_tuple=False).flatten()
            if cell_indices.numel() == 0:
                continue
            best_offset = adjusted_scores[cell_indices].argmax()
            required_header_indices.append(cell_indices[best_offset])
        if required_header_indices:
            header_indices = torch.stack(required_header_indices).unique()
        else:
            header_indices = torch.empty(0, dtype=torch.long, device=adjusted_scores.device)

        if header_indices.numel() > self.budget:
            raise ValueError(
                "Table header cell count exceeds layer_budget; increase layer_budget "
                "or shorten the table."
            )

        recent_start = max(0, kv_cache_len - self.window_size)
        recent_indices = torch.arange(recent_start, kv_cache_len, device=adjusted_scores.device)

        forced_mask = torch.zeros(kv_cache_len, dtype=torch.bool, device=adjusted_scores.device)
        forced_mask[header_indices] = True
        if self.window_size > 0:
            forced_mask[recent_indices] = True

        forced_indices = torch.nonzero(forced_mask, as_tuple=False).flatten()
        if forced_indices.numel() > self.budget:
            header_mask = torch.zeros(kv_cache_len, dtype=torch.bool, device=adjusted_scores.device)
            header_mask[header_indices] = True
            keep_mask = header_mask.clone()
            remaining = self.budget - int(header_indices.numel())
            if remaining > 0:
                recent_non_header = recent_indices[~header_mask[recent_indices]]
                keep_mask[recent_non_header[-remaining:]] = True
            return torch.nonzero(keep_mask, as_tuple=False).flatten()

        remaining = self.budget - int(forced_indices.numel())
        if remaining <= 0:
            return forced_indices

        candidate_scores = adjusted_scores.masked_fill(forced_mask, -torch.inf)
        top_indices = candidate_scores.topk(remaining, dim=-1).indices
        keep_indices = torch.cat([forced_indices, top_indices], dim=0)
        return keep_indices.sort().values

    def _slice_metadata(self, token_metadata, kept_indices):
        kept_cpu = kept_indices.to("cpu").tolist()
        result = {}
        for name in ("header_columns", "header_cell_ids", "cell_columns"):
            values = (token_metadata or {}).get(name) or []
            result[name] = [
                values[index] if index < len(values) else -1
                for index in kept_cpu
            ]
        return result

    def update_kv(self, query_states, key_states, value_states, token_metadata=None):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len <= self.budget:
            kept_indices = torch.arange(kv_cache_len, device=key_states.device)
            self.last_kept_metadata = self._slice_metadata(token_metadata, kept_indices)
            return key_states, value_states

        attn_weights = compute_attention_scores(query_states, key_states)
        attn_scores = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
        if self.score_reduce == "mean":
            attn_scores = attn_scores.mean(dim=-2)
        else:
            attn_scores = attn_scores.max(dim=-2).values

        token_scores = attn_scores.mean(dim=(0, 1))
        header_columns = self._metadata_tensor(
            token_metadata,
            "header_columns",
            kv_cache_len,
            key_states.device,
        )
        header_cell_ids = self._metadata_tensor(
            token_metadata,
            "header_cell_ids",
            kv_cache_len,
            key_states.device,
        )
        cell_columns = self._metadata_tensor(
            token_metadata,
            "cell_columns",
            kv_cache_len,
            key_states.device,
        )
        adjusted_scores = self._apply_header_scores(
            token_scores,
            header_columns,
            cell_columns,
        )
        kept_indices = self._select_indices(
            adjusted_scores,
            header_columns,
            header_cell_ids,
        )
        self.last_kept_metadata = self._slice_metadata(token_metadata, kept_indices)

        gather_indices = kept_indices.view(1, 1, -1, 1).expand(
            key_states.shape[0],
            key_states.shape[1],
            -1,
            head_dim,
        )
        return (
            key_states.gather(dim=2, index=gather_indices),
            value_states.gather(dim=2, index=gather_indices),
        )
