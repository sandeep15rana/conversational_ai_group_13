from collections.abc import Callable
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def retry_external_call(
    attempts: int = 3,
    minimum_wait: int = 1,
    maximum_wait: int = 8,
) -> Callable[..., Any]:
    """Retry temporary failures from external services."""

    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(
            multiplier=1,
            min=minimum_wait,
            max=maximum_wait,
        ),
        retry=retry_if_exception_type(
            (TimeoutError, ConnectionError)
        ),
        reraise=True,
    )
