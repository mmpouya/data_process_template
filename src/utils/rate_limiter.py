import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """Async-safe sliding window rate limiter."""

    def __init__(self, rpm: int, name: str = "limiter"):
        self.rpm = max(1, rpm)
        self.name = name
        self._lock = asyncio.Lock()
        self._timestamps: list[float] = []
        self._paused_until: float = 0.0

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                if now < self._paused_until:
                    wait_pause = self._paused_until - now
                else:
                    cutoff = now - 60.0
                    self._timestamps = [t for t in self._timestamps if t > cutoff]
                    if len(self._timestamps) < self.rpm:
                        self._timestamps.append(now)
                        return
                    oldest = self._timestamps[0]
                    wait_pause = max(0.05, 60.0 - (now - oldest) + 0.05)

            if wait_pause > 0:
                await asyncio.sleep(wait_pause)

    def pause(self, seconds: float) -> None:
        now = time.monotonic()
        target = now + max(0.0, seconds)
        if target > self._paused_until:
            self._paused_until = target
            logger.info("Rate limiter %s paused for %.1f seconds", self.name, seconds)
