"""Portable tests for decoded temporal read-ahead scheduling."""

from threading import Event
from time import sleep

import numpy as np

from src.map_data.temporal_cache import (
    MEBIBYTE,
    TemporalReadAheadManager,
    decoded_matrix_bytes,
    temporal_cache_size,
)

ORIGIN = (0, 0, 0)
SIZE = (4, 3, 2)
STEP = (1, 1, 1)
MATRIX_BYTES = 4 * 3 * 2 * np.dtype(np.uint16).itemsize


class FakeGrid:
    def __init__(self, name, reads, *, block=None, fail=False):
        self.name = name
        self.reads = reads
        self.block = block
        self.fail = fail

    def read_matrix(self, origin, size, step, progress):
        del progress
        self.reads.append((self.name, origin, size, step))
        if self.block is not None:
            self.block.wait()
        if self.fail:
            raise OSError("test read failed")
        shape = tuple((axis_size + axis_step - 1) // axis_step for axis_size, axis_step in zip(size, step, strict=True))
        return np.full(shape[::-1], int(self.name.rsplit("t", 1)[-1]), dtype=np.uint16)


def _sequence(manager, prefix, count, reads, **grid_kwargs):
    grids = [FakeGrid(f"{prefix}t{time}", reads, **grid_kwargs) for time in range(count)]
    return grids, manager.create_sequence(grids)


def test_temporal_cache_budget_and_matrix_size_are_adaptive():
    assert temporal_cache_size(8 * MEBIBYTE) == MEBIBYTE // 2
    assert temporal_cache_size(16 * 1024 * MEBIBYTE) == 512 * MEBIBYTE
    assert decoded_matrix_bytes(SIZE, (2, 2, 1), 2) == 2 * 2 * 2 * 2


def test_adaptive_read_ahead_prioritizes_near_times_across_channels():
    reads = []
    manager = TemporalReadAheadManager(20 * MATRIX_BYTES)
    channel0, sequence0 = _sequence(manager, "c0", 4, reads)
    channel1, sequence1 = _sequence(manager, "c1", 4, reads)

    sequence0.observe(0, ORIGIN, SIZE, STEP, MATRIX_BYTES)
    sequence1.observe(0, ORIGIN, SIZE, STEP, MATRIX_BYTES)
    manager.flush()

    assert manager.wait_for_idle()
    assert [name for name, *_ in reads] == ["c0t1", "c1t1", "c0t2", "c1t2", "c0t3", "c1t3"]
    assert manager.ready_count == 6
    np.testing.assert_array_equal(sequence0.consume(1, ORIGIN, SIZE, STEP), np.full((2, 3, 4), 1, np.uint16))
    assert manager.ready_count == 5
    assert manager.reserved_bytes == 5 * MATRIX_BYTES
    manager.close(wait=True)
    del channel0, channel1


def test_memory_and_depth_limits_bound_adaptive_filling():
    reads = []
    manager = TemporalReadAheadManager(2 * MATRIX_BYTES)
    grids, sequence = _sequence(manager, "", 6, reads)

    sequence.observe(0, ORIGIN, SIZE, STEP, MATRIX_BYTES)
    manager.flush()

    assert manager.wait_for_idle()
    assert [name for name, *_ in reads] == ["t1", "t2"]
    assert manager.ready_count == 2
    assert manager.reserved_bytes == 2 * MATRIX_BYTES
    manager.close(wait=True)

    reads = []
    manager = TemporalReadAheadManager(20 * MATRIX_BYTES)
    limited_grids = [FakeGrid(f"limited-t{time}", reads) for time in range(6)]
    limited = manager.create_sequence(limited_grids, depth=1)
    limited.observe(0, ORIGIN, SIZE, STEP, MATRIX_BYTES)
    manager.flush()
    assert manager.wait_for_idle()
    assert [name for name, *_ in reads] == ["limited-t1"]
    manager.close(wait=True)
    del grids, limited_grids


def test_direction_and_request_changes_discard_stale_results():
    reads = []
    manager = TemporalReadAheadManager(20 * MATRIX_BYTES)
    grids, sequence = _sequence(manager, "", 5, reads)

    sequence.observe(2, ORIGIN, SIZE, STEP, MATRIX_BYTES)
    manager.flush()
    assert manager.wait_for_idle()
    assert [name for name, *_ in reads] == ["t3", "t4"]

    sequence.observe(1, ORIGIN, SIZE, STEP, MATRIX_BYTES)
    manager.flush()
    assert manager.wait_for_idle()
    assert [name for name, *_ in reads][-1] == "t0"
    assert sequence.consume(3, ORIGIN, SIZE, STEP) is None
    assert sequence.consume(0, ORIGIN, SIZE, STEP) is not None

    changed_step = (2, 1, 1)
    changed_bytes = decoded_matrix_bytes(SIZE, changed_step, 2)
    sequence.observe(2, ORIGIN, SIZE, changed_step, changed_bytes)
    manager.flush()
    assert manager.wait_for_idle()
    assert sequence.consume(3, ORIGIN, SIZE, STEP) is None
    assert sequence.consume(3, ORIGIN, SIZE, changed_step) is not None
    manager.close(wait=True)
    del grids


def test_queued_foreground_request_bypasses_queue_and_failures_do_not_poison_cache():
    reads = []
    release = Event()
    manager = TemporalReadAheadManager(20 * MATRIX_BYTES)
    grids = [FakeGrid("t0", reads), FakeGrid("t1", reads, block=release), FakeGrid("t2", reads)]
    sequence = manager.create_sequence(grids)
    sequence.observe(0, ORIGIN, SIZE, STEP, MATRIX_BYTES)
    manager.flush()

    for _ in range(1000):
        if manager.running_count:
            break
        sleep(0.001)
    assert manager.running_count == 1
    assert sequence.consume(2, ORIGIN, SIZE, STEP) is None
    release.set()
    assert manager.wait_for_idle()
    manager.close(wait=True)

    reads = []
    manager = TemporalReadAheadManager(20 * MATRIX_BYTES)
    failing_grids = [FakeGrid("t0", reads), FakeGrid("t1", reads, fail=True)]
    failing = manager.create_sequence(failing_grids)
    failing.observe(0, ORIGIN, SIZE, STEP, MATRIX_BYTES)
    manager.flush()
    assert manager.wait_for_idle()
    assert failing.consume(1, ORIGIN, SIZE, STEP) is None
    assert manager.reserved_bytes == 0
    manager.close(wait=True)
    del grids, failing_grids
