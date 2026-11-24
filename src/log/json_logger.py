import asyncio
import json
from pathlib import Path
from typing import Any

import aiofiles


class AsyncJsonlStorage:
    """
    Asynchronously writes JSON-serializable records to a JSONL file.

    Records are buffered in an internal asyncio queue and written by a
    background worker in batches to reduce disk I/O overhead. Use `append()`
    to enqueue records, `close()` to flush pending items and stop the worker,
    and `delete()` to remove the output file.

    Args:
        file_path: Output JSONL file path relative to the `assets/` directory.
        max_queue_size: Maximum number of queued records allowed before
            `append()` blocks.
        batch_size: Maximum number of queued records written per batch.

    Attributes:
        file_path: Full path to the JSONL output file.
        batch_size: Maximum number of records written in one batch.
        _queue: Internal asyncio queue holding pending records.
        _worker_task: Background task that writes queued records to disk.
    usage example:
        ```
        async with AsyncJsonlStorage() as storage:
            await storage.append({"x": 1})
        ```
    """

    def __init__(
        self,
        file_path: Path | str = Path("logs.jsonl"),
        max_queue_size: int = 10000,
        batch_size: int = 500,
    ):
        self.file_path = Path("assets") / file_path
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

        # Defer initialization to avoid "no running event loop" errors
        self._queue: asyncio.Queue | None = None
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> "AsyncJsonlStorage":
        """Initializes the asyncio components and starts the worker."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)
            self._worker_task = asyncio.create_task(self._writer_worker())
        return self

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def append(self, data: dict[str, Any]) -> None:
        """Async-safe append. Puts data in the asyncio queue."""
        if self._queue is None:
            raise RuntimeError(
                "Storage must be started before appending data. use `async with ...` block"
            )
        await self._queue.put(data)

    async def _writer_worker(self) -> None:
        # Ensure the parent directory exists before opening the file
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if self._queue is None:
            raise RuntimeError("Storage must be started before appending data.")

        async with aiofiles.open(str(self.file_path), mode="a", encoding="utf-8") as f:
            shutdown = False
            while not shutdown:
                try:
                    item = await self._queue.get()
                    if item is None:  # Sentinel value to trigger graceful shutdown
                        self._queue.task_done()
                        break

                    batch = [item]

                    # Drain available items up to batch_size
                    while len(batch) < self.batch_size:
                        try:
                            next_item = self._queue.get_nowait()
                            if next_item is None:
                                self._queue.task_done()
                                shutdown = True
                                break
                            batch.append(next_item)
                        except asyncio.QueueEmpty:
                            break

                    # Serialize and write
                    lines = [
                        json.dumps(record, ensure_ascii=False) + "\n"
                        for record in batch
                    ]
                    await f.writelines(lines)
                    await f.flush()

                    for _ in batch:
                        self._queue.task_done()

                except asyncio.CancelledError:
                    # Allow the task to exit cleanly if forcefully cancelled
                    break
                except Exception as e:
                    # Prevent silent background failures
                    print(f"AsyncJsonlStorage worker error: {e}")

    async def close(self) -> None:
        """Wait for the queue to drain and gracefully stop the worker."""
        if self._queue and self._worker_task and not self._worker_task.done():
            # Send sentinel value to ensure graceful shutdown without data loss
            await self._queue.put(None)
            await self._queue.join()

            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def delete(self) -> None:
        """Deletes the log file if it exists."""
        if self.file_path.exists():
            self.file_path.unlink()
