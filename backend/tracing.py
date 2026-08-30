"""One optional dependency, isolated.

LangChain traces itself, but `search()` is plain Python — without this the whole
retrieval pipeline is one opaque box in LangSmith. `traceable` is a no-op unless
LANGSMITH_TRACING=true, and a machine that has never heard of LangSmith still runs
the pipeline unchanged.

It lives alone because BOTH retrieval and graph decorate with it, and neither should
have to depend on the other to get it.
"""

try:
    from langsmith import traceable
except ImportError:                                     # degrade, do not fail
    def traceable(*args, **kwargs):
        def decorate(fn):
            return fn
        return decorate(args[0]) if args and callable(args[0]) else decorate
