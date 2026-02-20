import time
from functools import wraps
from typing import Any, Callable

from kc.config import settings


def profile(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure and print the execution time of a function if profiling is enabled."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if settings.profiling:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            print(
                f"[{func.__module__}.{func.__qualname__}] execution time: {end_time - start_time:.4f}s"
            )
            return result
        return func(*args, **kwargs)

    return wrapper
