##########################################################################################
# oops/_cache.py
##########################################################################################

import numpy as np

from polymath  import Qube
from oops.oops import Oops


class _Cache(Oops):
    """A dictionary-like cache preserving a fixed number of items.

    When the size of the cache exceeds `maxsize` by about 10%, the least-recently accessed
    items are deleted.

    Indexing a _Cache using a key that is not present, or has been deleted, returns None.
    A KeyError is never raised.

    Dictionary keys can include mutable items, which are converted to immutable. The class
    method :meth:`clean_key` performs this conversion.
    """

    # These are filled in by oops/__init__.py to avoid circular imports
    _Frame = None
    _Path = None

    def __init__(self, maxsize=100):
        """Constructor for a _Cache.

        Parameters:
            maxsize (int, optional): The rough limit on the number of items stored in the
                _Cache. When this value is exceeded by ~ 10%, the number of elements is
                reduced back to `maxsize` by removing the items accessed least recently.
        """

        self._maxsize = maxsize
        self._extras = max(3, maxsize//10)
        self._limit = maxsize + self._extras
        self._dict = {}
        self._counter = 0

    def __len__(self):
        """The number of items currently in this _Cache."""
        return len(self._dict)

    @staticmethod
    def clean_key(key):
        """Convert the given key to immutable so it can be used as a dictionary key.

        Lists and tuples are converted recursively, so that a key containing a nested
        collection of Qubes is still hashable.

        Parameters:
            key (object): The key to convert. Qubes, NumPy arrays, Paths, Frames, and
                lists and tuples of these are converted; anything else is returned
                unchanged.

        Returns:
            object: An immutable version of `key`.
        """

        def clean_item(item):
            """An immutable version of one item of a key.

            Parameters:
                item (Any): The item to convert.

            Returns:
                Any: The immutable version of `item`, or `item` itself if it needs no
                conversion.
            """
            match item:
                case Qube():
                    vals = tuple(item.vals.ravel()) if np.shape(item.vals) else item.vals
                    mask = tuple(item.mask.ravel()) if np.shape(item.mask) else item.mask
                    return (type(item).__name__, item.shape, vals, mask)
                case np.ndarray():
                    if item.dtype == object:
                        return (item.shape,
                                tuple(clean_item(x) for x in item.ravel()))
                    return (item.shape, tuple(item.ravel()))
                case _Cache._Path():
                    return _Cache._Path.as_primary_path(item)
                case _Cache._Frame():
                    return _Cache._Frame.as_primary_frame(item)
                case x if hasattr(x, '__data__'):
                    # Keyed by identity, because hashing the object's own data buffer
                    # would cost more than the lookup saves. An id() is unique only among
                    # live objects, so a collected object's address can be reused and a
                    # later object can match its entry; that entry is evicted in turn.
                    return id(item)
                case list() | tuple():
                    return tuple(clean_item(subitem) for subitem in item)
                case _:
                    return item

        return clean_item(key)

    def __contains__(self, key):
        """True if the given key is currently in the _Cache."""

        if self._maxsize:
            key = _Cache.clean_key(key)
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
            key = _Cache.clean_key(key)
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
            key = _Cache.clean_key(key)
            self._counter += 1
            self._dict[key] = [self._counter, key, value]

            if len(self._dict) > self._limit:
                tuples = list(self._dict.values())
                tuples.sort()
                extras = tuples[:-self._maxsize]
                for (_, k, _) in extras:
                    del self._dict[k]


def _cache_for(key, shared):
    """The dictionary that holds the entry for a key of the Frame or Path cache.

    The keys of `Frame._FRAME_CACHE` and `Path._PATH_CACHE` are a wayframe or waypoint, or
    a tuple of them. A wayframe or waypoint that was created without an ID carries a
    dictionary of its own in its `_local_cache` attribute until it is given an ID; see
    `_give_up_local_cache`. An entry whose key includes one or more of them is kept
    in the dictionary of the first of them in the key, its owner, so that the entry is
    freed with its owner; until then, the entry holds the rest of its key and its value.
    Every other entry is kept in the process-wide dictionary. A stream of unregistered
    Frames and Paths over registered ones, such as the camera frames of observations built
    one at a time, therefore leaves the process-wide dictionaries as they were.

    Parameters:
        key (Frame | Path | tuple[Frame | Path, ...]): The key of the entry.
        shared (dict): The process-wide dictionary, `Frame._FRAME_CACHE` or
            `Path._PATH_CACHE`.

    Returns:
        dict: The `_local_cache` of the first item of `key` that has one; otherwise,
        `shared`.
    """

    for item in (key if isinstance(key, tuple) else (key,)):
        if item._local_cache is not None:
            return item._local_cache

    return shared


def _give_up_local_cache(owner, shared):
    """Move the cache entries of a wayframe or waypoint that has been given an ID.

    A wayframe or waypoint created without an ID is given one when a registered Frame or
    Path shares it, and it then stays alive through the registry. It stops keeping cache
    entries itself, and each of its entries moves to the dictionary that `_cache_for`
    selects for the entry's key from then on: that of the next member of the key that has
    no ID, or else the process-wide dictionary. An entry whose key
    includes an unregistered Frame is therefore still freed with that Frame.

    Parameters:
        owner (Frame | Path): The wayframe or waypoint that has been given an ID.
        shared (dict): The process-wide dictionary, `Frame._FRAME_CACHE` or
            `Path._PATH_CACHE`.
    """

    entries = owner._local_cache
    if entries is None:
        return

    owner._local_cache = None
    for key, value in entries.items():
        _cache_for(key, shared).setdefault(key, value)

##########################################################################################
