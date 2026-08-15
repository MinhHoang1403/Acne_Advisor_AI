"""Truyền một deadline chung xuyên suốt các bước của một request.

Budget dùng monotonic clock để không bị ảnh hưởng bởi thay đổi system time.
``remaining = max(0, deadline_at - now)`` và timeout của mỗi stage là
``min(configured_timeout, remaining)``. Vì các stage dùng cùng object, retry hoặc
provider fallback không được cấp lại toàn bộ thời gian từ đầu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Callable


@dataclass(frozen=True)
class DeadlineBudget:
    """Mốc bắt đầu/deadline bất biến cùng clock có thể thay bằng test double."""
    started_at: float
    deadline_at: float
    clock: Callable[[], float] = field(default=monotonic, compare=False, repr=False)

    @classmethod
    def from_timeout(
        cls,
        timeout_seconds: float,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> "DeadlineBudget":
        """Tạo deadline tại ``clock() + timeout_seconds``."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        started_at = clock()
        return cls(started_at=started_at, deadline_at=started_at + timeout_seconds, clock=clock)

    def elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_at - self.clock())

    def cap_timeout(self, configured_timeout: float) -> float:
        """Giới hạn timeout của stage theo phần ngân sách còn lại."""
        if configured_timeout <= 0:
            raise ValueError("configured_timeout must be > 0")
        remaining = self.remaining_seconds()
        if remaining <= 0:
            return 0.0
        return min(configured_timeout, remaining)

    def expired(self) -> bool:
        return self.remaining_seconds() <= 0


__all__ = ["DeadlineBudget"]
