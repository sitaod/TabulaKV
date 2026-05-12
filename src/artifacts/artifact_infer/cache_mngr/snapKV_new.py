from .snapKV import SnapKV


class SnapKVNew(SnapKV):
    """SnapKV compressor used by the strict-budget prefill path."""

    strict_budget = True

    def update_kv(self, query_states, key_states, value_states):
        if key_states.shape[-2] <= self.budget:
            return key_states, value_states
        return super().update_kv(query_states, key_states, value_states)
