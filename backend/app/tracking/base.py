from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from numpy.typing import NDArray

from backend.app.schemas import Detection, TrackedObject


class MultiObjectTracker(Protocol):
    def update(self, detections: Sequence[Detection], frame: NDArray) -> list[TrackedObject]: ...

    def reset(self) -> None: ...
