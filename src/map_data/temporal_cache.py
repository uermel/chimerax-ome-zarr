"""Bounded, priority-ordered decoded read-ahead for time-series grids."""

import heapq
import math
import time
import weakref
from dataclasses import dataclass
from threading import Condition, Event, RLock, Thread
from typing import Callable, Optional, Sequence, Tuple

MEBIBYTE = 1024**2
MAX_TEMPORAL_CACHE_BYTES = 512 * MEBIBYTE
MAX_TEMPORAL_TASKS = 256


def temporal_cache_size(matrix_cache_size: float) -> int:
    """Return the decoded read-ahead budget for a ChimeraX matrix cache."""

    return max(0, min(MAX_TEMPORAL_CACHE_BYTES, int(matrix_cache_size // 16)))


def _sample_count(size: int, step: int) -> int:
    return (size + step - 1) // step


def decoded_matrix_bytes(size: Tuple[int, int, int], step: Tuple[int, int, int], itemsize: int) -> int:
    """Return the byte size of a sampled three-dimensional matrix."""

    return math.prod(
        _sample_count(axis_size, axis_step) for axis_size, axis_step in zip(size, step, strict=True)
    ) * int(
        itemsize,
    )


@dataclass
class _SequenceState:
    current_index: Optional[int] = None
    direction: int = 1
    request: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]] = None
    decoded_bytes: int = 0
    pending: Optional[
        Tuple[
            int,
            Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]],
            int,
        ]
    ] = None


class _Task:
    def __init__(
        self,
        sequence,
        grid,
        grid_index: int,
        request,
        decoded_bytes: int,
        priority,
        grid_deleted,
    ) -> None:
        self.sequence_ref = weakref.ref(sequence)
        self.grid_ref = weakref.ref(grid, grid_deleted)
        self.grid_id = id(grid)
        self.grid_index = grid_index
        self.request = request
        self.decoded_bytes = decoded_bytes
        self.priority = priority
        self.queue_version = 0
        self.status = "queued"
        self.discard = False
        self.result = None
        self.done = Event()

    @property
    def key(self):
        return (self.grid_id, *self.request)


class TemporalSequence:
    """A time-ordered group of equivalent resolution/channel grids."""

    def __init__(self, manager, grids: Sequence[object], depth: Optional[int], order: int) -> None:
        self._manager_ref = weakref.ref(manager)
        self._grids = tuple(weakref.ref(grid) for grid in grids)
        self.depth = depth
        self.order = order

    def __len__(self) -> int:
        return len(self._grids)

    def grid(self, index: int):
        return self._grids[index]()

    def observe(self, index: int, origin, size, step, decoded_bytes: int) -> None:
        manager = self._manager_ref()
        if manager is not None:
            manager.observe(self, index, origin, size, step, decoded_bytes)

    def consume(self, index: int, origin, size, step):
        manager = self._manager_ref()
        grid = self.grid(index)
        if manager is None or grid is None:
            return None
        return manager.consume(grid, origin, size, step)


class TemporalReadAheadManager:
    """Coordinate background matrix reads without touching ChimeraX's cache."""

    def __init__(
        self,
        max_size: int | Callable[[], int],
        *,
        max_tasks: int = MAX_TEMPORAL_TASKS,
    ) -> None:
        if not callable(max_size) and max_size < 0:
            raise ValueError(f"Read-ahead cache size must be nonnegative, got {max_size}.")
        if max_tasks < 0:
            raise ValueError(f"Read-ahead task limit must be nonnegative, got {max_tasks}.")
        self._max_size = max_size
        self.max_tasks = max_tasks
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._states = weakref.WeakKeyDictionary()
        self._tasks = {}
        self._detached_running = set()
        self._queue = []
        self._sequence_order = 0
        self._queue_order = 0
        self._reserved_bytes = 0
        self._worker = None
        self._stopped = False

    @property
    def max_size(self) -> int:
        value = self._max_size() if callable(self._max_size) else self._max_size
        return max(0, int(value))

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return self._reserved_bytes

    @property
    def ready_count(self) -> int:
        with self._lock:
            return sum(task.status == "ready" for task in self._tasks.values())

    @property
    def queued_count(self) -> int:
        with self._lock:
            return sum(task.status == "queued" for task in self._tasks.values())

    @property
    def running_count(self) -> int:
        with self._lock:
            return sum(task.status == "running" for task in self._tasks.values()) + len(self._detached_running)

    def create_sequence(self, grids: Sequence[object], depth: Optional[int] = None) -> TemporalSequence:
        if depth is not None and depth < 0:
            raise ValueError(f"Read-ahead depth must be nonnegative, got {depth}.")
        if not grids:
            raise ValueError("A temporal sequence requires at least one grid.")
        with self._lock:
            sequence = TemporalSequence(self, grids, depth, self._sequence_order)
            self._sequence_order += 1
            self._states[sequence] = _SequenceState()
        return sequence

    def observe(self, sequence: TemporalSequence, index: int, origin, size, step, decoded_bytes: int) -> None:
        request = (tuple(origin), tuple(size), tuple(step))
        if decoded_bytes <= 0 or sequence.depth == 0:
            return
        with self._lock:
            if self._stopped or sequence not in self._states:
                return
            if index < 0 or index >= len(sequence):
                raise IndexError(f"Temporal grid index {index} is outside a sequence of length {len(sequence)}.")
            self._states[sequence].pending = (index, request, int(decoded_bytes))

    def flush(self) -> None:
        """Apply observations and refill the globally prioritized read-ahead queue."""

        with self._condition:
            if self._stopped:
                return
            observed = False
            for sequence, state in tuple(self._states.items()):
                pending = state.pending
                if pending is None:
                    continue
                observed = True
                state.pending = None
                index, request, decoded_bytes = pending
                state.direction = self._direction(state.current_index, index, len(sequence), state.direction)
                state.current_index = index
                state.request = request
                state.decoded_bytes = decoded_bytes
            if observed:
                self._rebuild_queue_locked()
            self._condition.notify_all()

    @staticmethod
    def _direction(previous: Optional[int], current: int, count: int, existing: int) -> int:
        if previous is None or previous == current:
            return existing
        if previous == count - 1 and current == 0:
            return 1
        if previous == 0 and current == count - 1:
            return -1
        return 1 if current > previous else -1

    def _candidates_locked(self):
        candidates = []
        for sequence, state in tuple(self._states.items()):
            if state.current_index is None or state.request is None or sequence.depth == 0:
                continue
            available = len(sequence) - state.current_index - 1 if state.direction > 0 else state.current_index
            distance_limit = available if sequence.depth is None else min(available, sequence.depth)
            for distance in range(1, distance_limit + 1):
                grid_index = state.current_index + state.direction * distance
                grid = sequence.grid(grid_index)
                if grid is None:
                    continue
                priority = (distance, sequence.order, grid_index)
                candidates.append(
                    (priority, sequence, grid, grid_index, state.request, state.decoded_bytes),
                )
        candidates.sort(key=lambda candidate: candidate[0])
        return candidates

    def _rebuild_queue_locked(self) -> None:
        candidates = self._candidates_locked()
        running = [task for task in self._tasks.values() if task.status == "running"]
        running.extend(self._detached_running)
        reserved = sum(task.decoded_bytes for task in running)
        task_count = len(running)
        selected = set()
        candidate_by_key = {}
        max_size = self.max_size

        for priority, sequence, grid, grid_index, request, decoded_bytes in candidates:
            key = (id(grid), *request)
            if key in candidate_by_key:
                continue
            candidate_by_key[key] = (priority, sequence, grid, grid_index, request, decoded_bytes)
            existing = self._tasks.get(key)
            if existing is not None and existing.status == "running":
                selected.add(key)
                continue
            if task_count >= self.max_tasks or reserved + decoded_bytes > max_size:
                continue
            selected.add(key)
            reserved += decoded_bytes
            task_count += 1

        for key, task in tuple(self._tasks.items()):
            if key not in selected:
                self._cancel_locked(task)

        for key in selected:
            priority, sequence, grid, grid_index, request, decoded_bytes = candidate_by_key[key]
            task = self._tasks.get(key)
            if task is None:
                task = _Task(sequence, grid, grid_index, request, decoded_bytes, priority, self._grid_deleted)
                self._tasks[key] = task
                self._reserved_bytes += decoded_bytes
                self._enqueue_locked(task)
            elif task.status == "queued" and task.priority != priority:
                task.priority = priority
                task.queue_version += 1
                self._enqueue_locked(task)

        if any(task.status == "queued" for task in self._tasks.values()):
            self._ensure_worker_locked()

    def _enqueue_locked(self, task: _Task) -> None:
        self._queue_order += 1
        heapq.heappush(
            self._queue,
            (*task.priority, self._queue_order, task.queue_version, task),
        )

    def _ensure_worker_locked(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = Thread(target=self._run, name="OME-Zarr temporal read-ahead", daemon=True)
            self._worker.start()

    def _next_task_locked(self):
        while self._queue:
            *_, queue_version, task = heapq.heappop(self._queue)
            if task.status == "queued" and task.queue_version == queue_version:
                task.status = "running"
                return task
        return None

    def _run(self) -> None:
        while True:
            with self._condition:
                task = self._next_task_locked()
                if task is None:
                    self._worker = None
                    self._condition.notify_all()
                    return
            grid = task.grid_ref()
            result = None
            if grid is not None:
                try:
                    origin, size, step = task.request
                    result = grid.read_matrix(origin, size, step, None)
                except Exception:
                    result = None
            with self._condition:
                if self._stopped or task.discard or grid is None or result is None:
                    self._finish_task_locked(task)
                else:
                    task.result = result
                    task.status = "ready"
                    task.done.set()
                self._condition.notify_all()

    def consume(self, grid, origin, size, step):
        """Return a matching decoded matrix, waiting only for the same running read."""

        request = (tuple(origin), tuple(size), tuple(step))
        key = (id(grid), *request)
        with self._condition:
            task = self._tasks.get(key)
            if task is None:
                return None
            if task.status == "ready":
                return self._consume_ready_locked(task)
            if task.status == "queued":
                self._cancel_locked(task)
                return None
            if task.status != "running":
                return None
            done = task.done

        done.wait()
        with self._condition:
            task = self._tasks.get(key)
            if task is not None and task.status == "ready":
                return self._consume_ready_locked(task)
        return None

    def _consume_ready_locked(self, task: _Task):
        result = task.result
        self._finish_task_locked(task)
        self._condition.notify_all()
        return result

    def _cancel_locked(self, task: _Task) -> None:
        if task.status == "running":
            task.discard = True
            if self._tasks.get(task.key) is task:
                self._tasks.pop(task.key)
            self._detached_running.add(task)
            return
        self._finish_task_locked(task)

    def _finish_task_locked(self, task: _Task) -> None:
        if task.status not in {"cancelled", "finished"}:
            self._reserved_bytes -= task.decoded_bytes
        self._detached_running.discard(task)
        if self._tasks.get(task.key) is task:
            self._tasks.pop(task.key)
        task.result = None
        task.status = "finished"
        task.done.set()

    def _grid_deleted(self, grid_ref) -> None:
        with self._condition:
            for task in tuple(self._tasks.values()):
                if task.grid_ref is grid_ref:
                    self._cancel_locked(task)
            self._condition.notify_all()

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        """Wait for queued/running work; intended for tests and orderly shutdown."""

        deadline = time.monotonic() + timeout
        with self._condition:
            while self._detached_running or any(task.status in {"queued", "running"} for task in self._tasks.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, *, wait: bool = False) -> None:
        with self._condition:
            if self._stopped:
                return
            self._stopped = True
            for task in tuple(self._tasks.values()):
                self._cancel_locked(task)
            worker = self._worker
            self._condition.notify_all()
        if wait and worker is not None:
            worker.join()


def session_temporal_read_ahead(session) -> TemporalReadAheadManager:
    """Return the session's decoded temporal read-ahead manager."""

    manager = getattr(session, "_ome_zarr_temporal_read_ahead", None)
    if manager is not None:
        return manager

    from chimerax.map.volume import data_cache

    manager = TemporalReadAheadManager(lambda: temporal_cache_size(data_cache(session).size))
    session._ome_zarr_temporal_read_ahead = manager

    def flush_after_frame(*_):
        manager.flush()

    def close_on_quit(*_):
        manager.close()

    manager._trigger_handlers = (  # Keep the handlers alive for the session lifetime.
        session.triggers.add_handler("frame drawn", flush_after_frame),
        session.triggers.add_handler("app quit", close_on_quit),
    )
    return manager
