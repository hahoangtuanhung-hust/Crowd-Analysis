from __future__ import annotations

from queue import Empty, Full, Queue
from typing import Generic, TypeVar

T = TypeVar("T")


class BoundedFrameQueue(Generic[T]):
    def __init__(self, maxsize: int, drop_oldest: bool) -> None:
        self._queue: Queue[T] = Queue(maxsize=maxsize)
        self.drop_oldest = drop_oldest
        self.dropped = 0

    @property
    def size(self) -> int:
        return self._queue.qsize()

    def put(self, item: T, timeout: float = 0.1) -> bool:
        if not self.drop_oldest:
            self._queue.put(item, timeout=timeout)
            return True

        try:
            self._queue.put_nowait(item)
            return True
        except Full:
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except Empty:
                pass
            self._queue.put_nowait(item)
            return True

    def get(self, timeout: float = 0.1) -> T:
        return self._queue.get(timeout=timeout)
