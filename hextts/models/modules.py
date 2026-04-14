"""Reusable model building blocks."""

from __future__ import annotations


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Return symmetric Conv1d padding preserving sequence length for odd kernels."""
    return (kernel_size * dilation - dilation) // 2