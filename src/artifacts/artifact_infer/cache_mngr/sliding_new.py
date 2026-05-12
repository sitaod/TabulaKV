from .sliding import SlidingWindowKV


class SlidingWindowKVNew(SlidingWindowKV):
    """Sliding-window compressor for the strict-budget prefill path."""

    strict_budget = True
