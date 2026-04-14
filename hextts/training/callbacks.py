"""Training callback interfaces for HexTTs."""

from __future__ import annotations

from typing import Protocol


class TrainingCallback(Protocol):
    """Minimal callback protocol for future trainer integrations."""

    def on_train_start(self) -> None:
        ...

    def on_train_end(self) -> None:
        ...


class NoOpCallback:
    """Default callback that intentionally does nothing."""

    def on_train_start(self) -> None:
        return None

    def on_train_end(self) -> None:
        return None