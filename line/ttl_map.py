"""A dictionary with per-entry TTL-based expiration."""

import time
from typing import Dict, Generic, Iterator, Optional, Tuple, TypeVar, Union, overload

KT = TypeVar("KT")
VT = TypeVar("VT")
_T = TypeVar("_T")

_MISSING = object()


class TTLMap(Generic[KT, VT]):
    """A dictionary where entries expire after a configurable TTL (in seconds).

    Expired entries are lazily evicted on access (get/pop/contains/iter/len).
    """

    # TTL in seconds for entries in this map. Can be overridden per-entry on set().
    def __init__(self, default_ttl: float):
        self._default_ttl = default_ttl
        self._data: Dict[KT, Tuple[VT, float]] = {}  # key -> (value, expiry_time)

    def set(self, key: KT, value: VT, ttl: Optional[float] = None) -> None:
        """Insert or update a key with an optional per-key TTL override."""
        self._evict()
        expiry = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        self._data[key] = (value, expiry)

    @overload
    def get(self, key: KT) -> Optional[VT]: ...
    @overload
    def get(self, key: KT, default: Union[VT, _T]) -> Union[VT, _T]: ...
    def get(self, key: KT, default: Union[VT, _T, object] = None) -> Union[VT, _T, None]:
        """Return the value for *key* if present and not expired, else *default*."""
        entry = self._data.get(key)
        if entry is None:
            return default  # type: ignore[return-value]
        value, expiry = entry
        if time.monotonic() > expiry:
            del self._data[key]
            return default  # type: ignore[return-value]
        return value

    @overload
    def pop(self, key: KT) -> VT: ...
    @overload
    def pop(self, key: KT, default: Union[VT, _T]) -> Union[VT, _T]: ...
    def pop(self, key: KT, default: object = _MISSING) -> Union[VT, object]:
        """Remove and return the value for *key* if present and not expired."""
        entry = self._data.pop(key, None)
        if entry is None:
            if default is _MISSING:
                raise KeyError(key)
            return default
        value, expiry = entry
        if time.monotonic() > expiry:
            if default is _MISSING:
                raise KeyError(key)
            return default
        return value

    def __contains__(self, key: object) -> bool:
        entry = self._data.get(key)  # type: ignore[arg-type]
        if entry is None:
            return False
        _, expiry = entry
        if time.monotonic() > expiry:
            del self._data[key]  # type: ignore[arg-type]
            return False
        return True

    def __len__(self) -> int:
        self._evict()
        return len(self._data)

    def __iter__(self) -> Iterator[KT]:
        self._evict()
        return iter(self._data)

    def __setitem__(self, key: KT, value: VT) -> None:
        self.set(key, value)

    def __getitem__(self, key: KT) -> VT:
        entry = self._data.get(key)
        if entry is None:
            raise KeyError(key)
        value, expiry = entry
        if time.monotonic() > expiry:
            del self._data[key]
            raise KeyError(key)
        return value

    def __delitem__(self, key: KT) -> None:
        del self._data[key]

    def _evict(self) -> None:
        """Remove all expired entries."""
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._data.items() if now > exp]
        for k in expired:
            del self._data[k]
