class SlidingWindowKV:
    """Keep only the most recent KV tokens after compression."""

    def __init__(self, budget=128, **kwargs):
        assert budget > 0, "budget must be positive"
        self.budget = budget

    def update_kv(self, query_states, key_states, value_states):
        kv_cache_len = key_states.shape[-2]
        if kv_cache_len <= self.budget:
            return key_states, value_states
        return key_states[:, :, -self.budget :, :], value_states[:, :, -self.budget :, :]
