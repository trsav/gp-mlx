from typing import Any

import mlx.core as mx


def function(x: Any) -> Any:
    return mx.array(x**2).item()
