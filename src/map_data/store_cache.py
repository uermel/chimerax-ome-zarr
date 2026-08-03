"""Bounded, session-shared caching for Zarr 3 stores."""

import asyncio
from collections import OrderedDict
from collections.abc import AsyncGenerator, Iterable
from threading import RLock
from typing import Optional

import zarr
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.common import BytesLike

MEBIBYTE = 1024**2
GIBIBYTE = 1024**3
MIN_STORE_CACHE_BYTES = 256 * MEBIBYTE
MAX_STORE_CACHE_BYTES = 4 * GIBIBYTE

_MISSING = object()


def _byte_request_key(byte_range: Optional[ByteRequest]):
    if byte_range is None:
        return None
    return type(byte_range).__name__, tuple(sorted(vars(byte_range).items()))


class EncodedStoreCache:
    """A thread-safe byte-size-limited LRU shared by cached stores."""

    def __init__(self, max_size: int) -> None:
        if max_size < 0:
            raise ValueError(f"Cache size must be nonnegative, got {max_size}.")
        self.max_size = max_size
        self._entries = OrderedDict()
        self._used = 0
        self._lock = RLock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def lookup(self, cache_key):
        with self._lock:
            value = self._entries.pop(cache_key, _MISSING)
            if value is _MISSING:
                return False, None
            self._entries[cache_key] = value
            return True, value

    def store(self, cache_key, value: Optional[bytes]) -> None:
        size = self._entry_size(value)
        if size > self.max_size:
            return
        with self._lock:
            previous = self._entries.pop(cache_key, _MISSING)
            if previous is not _MISSING:
                self._used -= self._entry_size(previous)
            self._entries[cache_key] = value
            self._used += size
            while self._used > self.max_size:
                _, evicted = self._entries.popitem(last=False)
                self._used -= self._entry_size(evicted)

    def invalidate_key(self, namespace, key: str) -> None:
        with self._lock:
            cache_keys = [cache_key for cache_key in self._entries if cache_key[:2] == (namespace, key)]
            self._remove(cache_keys)

    def invalidate_prefix(self, namespace, prefix: str) -> None:
        with self._lock:
            cache_keys = [
                cache_key
                for cache_key in self._entries
                if cache_key[0] is namespace and cache_key[1].startswith(prefix)
            ]
            self._remove(cache_keys)

    def invalidate_namespace(self, namespace) -> None:
        with self._lock:
            cache_keys = [cache_key for cache_key in self._entries if cache_key[0] is namespace]
            self._remove(cache_keys)

    def _remove(self, cache_keys) -> None:
        for cache_key in cache_keys:
            value = self._entries.pop(cache_key)
            self._used -= self._entry_size(value)

    @staticmethod
    def _entry_size(value: Optional[bytes]) -> int:
        # Count a negative entry so missing-key probes cannot grow without bound.
        return 1 if value is None else len(value)


class CachedStore(zarr.storage.WrapperStore):
    """Cache full and byte-range reads from any Zarr 3 store."""

    def __init__(self, store, cache: EncodedStoreCache, namespace=None) -> None:
        super().__init__(store)
        self.cache = cache
        self.namespace = object() if namespace is None else namespace

    def with_read_only(self, read_only: bool = False):
        return type(self)(self._store.with_read_only(read_only), self.cache, self.namespace)

    def _cache_key(self, key: str, byte_range: Optional[ByteRequest]):
        return self.namespace, key, _byte_request_key(byte_range)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: Optional[ByteRequest] = None,
    ) -> Optional[Buffer]:
        cache_key = self._cache_key(key, byte_range)
        found, value = self.cache.lookup(cache_key)
        if found:
            return None if value is None else prototype.buffer.from_bytes(value)

        value = await self._store.get(key, prototype, byte_range)
        self.cache.store(cache_key, None if value is None else value.to_bytes())
        return value

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, Optional[ByteRequest]]],
    ) -> list[Optional[Buffer]]:
        return list(await asyncio.gather(*(self.get(key, prototype, byte_range) for key, byte_range in key_ranges)))

    async def _get_many(
        self,
        requests: Iterable[tuple[str, BufferPrototype, Optional[ByteRequest]]],
    ) -> AsyncGenerator[tuple[str, Optional[Buffer]], None]:
        request_list = list(requests)
        values = await asyncio.gather(
            *(self.get(key, prototype, byte_range) for key, prototype, byte_range in request_list),
        )
        for (key, _, _), value in zip(request_list, values, strict=True):
            yield key, value

    async def exists(self, key: str) -> bool:
        found, value = self.cache.lookup(self._cache_key(key, None))
        return value is not None if found else await self._store.exists(key)

    async def set(self, key: str, value: Buffer) -> None:
        await self._store.set(key, value)
        self.cache.invalidate_key(self.namespace, key)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        await self._store.set_if_not_exists(key, value)
        self.cache.invalidate_key(self.namespace, key)

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        value_list = list(values)
        await self._store._set_many(value_list)
        for key, _ in value_list:
            self.cache.invalidate_key(self.namespace, key)

    async def set_partial_values(self, key_start_values: Iterable[tuple[str, int, BytesLike]]) -> None:
        value_list = list(key_start_values)
        await self._store.set_partial_values(value_list)
        for key in {key for key, _, _ in value_list}:
            self.cache.invalidate_key(self.namespace, key)

    async def delete(self, key: str) -> None:
        await self._store.delete(key)
        self.cache.invalidate_key(self.namespace, key)

    async def delete_dir(self, prefix: str) -> None:
        await self._store.delete_dir(prefix)
        self.cache.invalidate_prefix(self.namespace, prefix)

    async def clear(self) -> None:
        await self._store.clear()
        self.cache.invalidate_namespace(self.namespace)


def store_cache_size(matrix_cache_size: float) -> int:
    """Return the adaptive encoded-store cache budget for a ChimeraX session."""

    return max(MIN_STORE_CACHE_BYTES, min(MAX_STORE_CACHE_BYTES, int(matrix_cache_size // 8)))


def session_store_cache(session) -> EncodedStoreCache:
    cache = getattr(session, "_ome_zarr_store_cache", None)
    if cache is None:
        from chimerax.map.volume import data_cache

        cache = EncodedStoreCache(store_cache_size(data_cache(session).size))
        session._ome_zarr_store_cache = cache
    return cache


def cached_store(session, store):
    if isinstance(store, CachedStore):
        return store
    return CachedStore(store, session_store_cache(session))


def cached_group(session, root):
    """Open a group through the session cache while preserving a nested group path."""

    if isinstance(root, zarr.Group):
        if isinstance(root.store, CachedStore):
            return root
        store = root.store
        path = root.path
    else:
        store = root
        path = ""
    return zarr.open_group(store=cached_store(session, store), path=path, mode="r")
