def __getattr__(name):
    if name in {"LLM", "SamplingParams"}:
        from .artifact_infer import LLM, SamplingParams

        return {"LLM": LLM, "SamplingParams": SamplingParams}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
