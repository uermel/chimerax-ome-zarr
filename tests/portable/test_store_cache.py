"""Tests for the Zarr 3 encoded-store cache."""

import asyncio
from collections import Counter

import zarr
from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype

from src.map_data.store_cache import (
    GIBIBYTE,
    MEBIBYTE,
    CachedStore,
    EncodedStoreCache,
    store_cache_size,
)


class CountingStore(zarr.storage.WrapperStore):
    def __init__(self, store) -> None:
        super().__init__(store)
        self.gets = Counter()

    async def get(self, key, prototype, byte_range=None):
        self.gets[(key, repr(byte_range))] += 1
        return await self._store.get(key, prototype, byte_range)


def _memory_store_with_payload(key="payload", value=b"abcdef"):
    prototype = default_buffer_prototype()
    store = zarr.storage.MemoryStore()
    asyncio.run(store.set(key, prototype.buffer.from_bytes(value)))
    return store.with_read_only(True), prototype


def test_cached_store_reuses_full_range_and_missing_reads():
    memory_store, prototype = _memory_store_with_payload()
    counting_store = CountingStore(memory_store)
    cache = EncodedStoreCache(1024)
    store = CachedStore(counting_store, cache)

    async def read_values():
        full_first = await store.get("payload", prototype)
        full_second = await store.get("payload", prototype)
        byte_range = RangeByteRequest(1, 4)
        range_first = await store.get("payload", prototype, byte_range)
        range_second = await store.get("payload", prototype, byte_range)
        missing_first = await store.get("missing", prototype)
        missing_second = await store.get("missing", prototype)
        missing_exists = await store.exists("missing")
        partial_values = await store.get_partial_values(
            prototype,
            [("payload", byte_range), ("missing", None)],
        )
        many_values = [
            value
            async for _, value in store._get_many(
                [("payload", prototype, None), ("missing", prototype, None)],
            )
        ]
        return (
            full_first,
            full_second,
            range_first,
            range_second,
            missing_first,
            missing_second,
            missing_exists,
            partial_values,
            many_values,
        )

    (
        full_first,
        full_second,
        range_first,
        range_second,
        missing_first,
        missing_second,
        missing_exists,
        partial_values,
        many_values,
    ) = asyncio.run(read_values())

    assert full_first.to_bytes() == full_second.to_bytes() == b"abcdef"
    assert range_first.to_bytes() == range_second.to_bytes() == b"bcd"
    assert missing_first is missing_second is None
    assert missing_exists is False
    assert partial_values[0].to_bytes() == b"bcd"
    assert partial_values[1] is None
    assert many_values[0].to_bytes() == b"abcdef"
    assert many_values[1] is None
    assert sum(counting_store.gets.values()) == 3
    assert cache.entry_count == 3
    assert cache.used == 10


def test_encoded_store_cache_evicts_lru_and_skips_oversized_values():
    cache = EncodedStoreCache(5)
    namespace = object()
    first = (namespace, "first", None)
    second = (namespace, "second", None)
    third = (namespace, "third", None)

    cache.store(first, b"123")
    cache.store(second, b"45")
    assert cache.lookup(first) == (True, b"123")
    cache.store(third, b"abc")

    assert cache.lookup(second) == (False, None)
    assert cache.lookup(first) == (False, None)
    assert cache.lookup(third) == (True, b"abc")
    cache.store(first, b"123456")
    assert cache.lookup(first) == (False, None)
    assert cache.used == 3


def test_store_cache_size_is_adaptive_and_bounded():
    assert store_cache_size(128 * MEBIBYTE) == 256 * MEBIBYTE
    assert store_cache_size(16 * GIBIBYTE) == 2 * GIBIBYTE
    assert store_cache_size(64 * GIBIBYTE) == 4 * GIBIBYTE
