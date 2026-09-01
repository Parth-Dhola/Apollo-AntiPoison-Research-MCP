"""cache.py — Resilient disk and in-memory caching layer to prevent API rate limits."""

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional
from apollo.config import get_settings

logger = logging.getLogger("apollo.cache")


class SimpleCache:
    """Wrapper that tries DiskCache, with an in-memory TTL dictionary fallback."""

    def __init__(self, cache_dir: str = ".apollo_cache", default_ttl: int = 86400):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self._disk_cache = None
        self._memory_cache = {}

        try:
            import diskcache
            os.makedirs(cache_dir, exist_ok=True)
            self._disk_cache = diskcache.Cache(cache_dir)
            logger.debug(f"DiskCache initialized at {cache_dir}")
        except Exception as e:
            logger.warning(f"DiskCache not available ({e}), using in-memory cache.")

    def _make_key(self, namespace: str, key_data: Any) -> str:
        serialized = json.dumps(key_data, sort_keys=True) if not isinstance(key_data, str) else key_data
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, namespace: str, key_data: Any) -> Optional[Any]:
        key = self._make_key(namespace, key_data)
        if self._disk_cache is not None:
            try:
                return self._disk_cache.get(key)
            except Exception as e:
                logger.error(f"DiskCache get error: {e}")

        # In-memory fallback check
        item = self._memory_cache.get(key)
        if item:
            val, expire_at = item
            if expire_at is None or time.time() < expire_at:
                return val
            else:
                del self._memory_cache[key]
        return None

    def set(self, namespace: str, key_data: Any, value: Any, ttl: Optional[int] = None) -> None:
        key = self._make_key(namespace, key_data)
        expire_seconds = ttl if ttl is not None else self.default_ttl

        if self._disk_cache is not None:
            try:
                self._disk_cache.set(key, value, expire=expire_seconds)
                return
            except Exception as e:
                logger.error(f"DiskCache set error: {e}")

        # In-memory fallback
        expire_at = time.time() + expire_seconds if expire_seconds > 0 else None
        self._memory_cache[key] = (value, expire_at)

    def clear(self) -> None:
        if self._disk_cache is not None:
            self._disk_cache.clear()
        self._memory_cache.clear()


_cache_instance: Optional[SimpleCache] = None


def get_cache() -> SimpleCache:
    global _cache_instance
    if _cache_instance is None:
        settings = get_settings()
        _cache_instance = SimpleCache(
            cache_dir=settings.CACHE_DIR,
            default_ttl=settings.CACHE_TTL_SECONDS
        )
    return _cache_instance

