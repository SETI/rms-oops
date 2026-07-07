##########################################################################################
# oops/cache.py: Support for caching of OOPS objects
##########################################################################################

import numpy as np
from polymath import Qube


class Cache:
    """Class that can be indexed like a dictionary, where `maxsize` items are preserved.
    When the size of the cache exceeds `maxsize` by ~ 10%, the least-recently accessed
    items are deleted.

    Indexing a Cache using a key that is not present, or has been deleted, returns None.
    A KeyError is never raised.

    Dictionary keys can include mutable items, which are converted to immutable. The class
    method `clean_key` performs this conversion.
    """

    # These are filled in by oops/__init__.py to avoid circular imports
    FRAME_CLASS = None
    PATH_CLASS = None

    def __init__(self, maxsize=100):
        """Constructor for a Cache.

        Parameters:
            maxsize (int, optional): The rough limit on the number of items stored in the
                Cache. When this value is exceeded by ~ 10%, the number of elements is
                reduced back to `maxsize` by removing the items accessed least recently.
        """

        self._maxsize = maxsize
        self._extras = max(3, maxsize//10)
        self._limit = maxsize + self._extras
        self._dict = {}
        self._counter = 0

    def __len__(self):
        """The number of items currently in this Cache."""
        return len(self._dict)

    @staticmethod
    def clean_key(key):
        """Convert the given key to immutable so it can be used as a dictionary key."""

        def clean_item(item):
            match item:
                case Qube():
                    vals = tuple(item.vals.ravel()) if np.shape(item.vals) else item.vals
                    mask = tuple(item.mask.ravel()) if np.shape(item.mask) else item.mask
                    return (type(item).__name__, item.shape, vals, mask)
                case np.ndarray():
                    return (item.shape, tuple(item.ravel()))
                case Cache.PATH_CLASS():
                    return Cache.PATH_CLASS.as_waypoint(item)
                case Cache.FRAME_CLASS():
                    return Cache.FRAME_CLASS.as_wayframe(item)
                case x if hasattr(x, '__data__'):
                    return id(item)
                case list():
                    return tuple(item)
                case _:
                    return item

        if isinstance(key, (list, tuple)):
            return tuple(clean_item(item) for item in key)

        return clean_item(key)

    def __contains__(self, key):
        """True if the given key is currently in the Cache."""

        if self._maxsize:
            key = Cache.clean_key(key)
            if key in self._dict:
                self._counter += 1
                self._dict[key][0] = self._counter
                return True
        return False

    def __getitem__(self, key):
        """The value associated with the given key, or None if the key is missing.

        Supports index notation using square brackets "[]".
        """

        if self._maxsize:
            key = Cache.clean_key(key)
            if key in self._dict:
                self._counter += 1
                count_key_value = self._dict[key]
                count_key_value[0] = self._counter
                return count_key_value[2]

        return None

    def __setitem__(self, key, value):
        """Set the value associated with the given key.

        Supports index notation using square brackets "[]".
        """

        if self._maxsize:
            key = Cache.clean_key(key)
            self._counter += 1
            self._dict[key] = [self._counter, key, value]

            if len(self._dict) > self._limit:
                tuples = list(self._dict.values())
                tuples.sort()
                extras = tuples[:-self._maxsize]
                for (_, k, _) in extras:
                    del self._dict[k]

##########################################################################################
