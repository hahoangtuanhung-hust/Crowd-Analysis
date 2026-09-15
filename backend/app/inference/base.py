from __future__ import annotations

from typing import Protocol

from numpy.typing import NDArray

from backend.app.schemas import Detection


class PersonDetector(Protocol):
    def detect(self, frame: NDArray) -> list[Detection]: ...
