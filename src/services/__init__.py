def __getattr__(name):
    if name in {"LLM", "SamplingParams"}:
        from .nanovllm_v5 import LLM, SamplingParams

        return {"LLM": LLM, "SamplingParams": SamplingParams}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
