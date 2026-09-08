##########################################################################################
# test/test_cache.py
##########################################################################################

import numpy as np
from oops._cache import _Cache
from oops.frame import Rotation
from oops.path  import LinearPath
from polymath   import Scalar, Vector


def test_clean_key():
    clean_key = _Cache.clean_key

    key = 1
    assert clean_key(key) == 1
    assert isinstance(clean_key(key), int)

    key = 2.
    assert clean_key(key) == 2.
    assert isinstance(clean_key(key), float)

    key = True
    assert clean_key(key) is True
    assert isinstance(clean_key(key), bool)

    key = False
    assert clean_key(key) is False
    assert isinstance(clean_key(key), bool)

    key = 'abc'
    assert clean_key(key) == 'abc'

    key = None
    assert clean_key(key) is None

    key = [1]
    assert clean_key(key) == (1,)

    key = [2, 3., 'four']
    assert clean_key(key) == (2, 3., 'four')

    key = np.array(4.)
    assert clean_key(key) == ((), (4.,))
    assert isinstance(clean_key(key)[1][0], np.float64)

    key = np.array([[1,2],[3,4]])
    assert clean_key(key) == ((2,2), (1,2,3,4))
    assert isinstance(clean_key(key)[-1][-1], np.int64)

    key = Scalar(3.14)
    assert clean_key(key) == ('Scalar', (), 3.14, False)

    key = Scalar((2.718, 3.14))
    assert clean_key(key) == ('Scalar', (2,), (2.718, 3.14), False)

    key = Scalar((2.718, 3.14), True)
    assert clean_key(key) == ('Scalar', (2,), (2.718, 3.14), True)

    key = Scalar((2.718, 3.14), (False,True))
    assert clean_key(key) == ('Scalar', (2,), (2.718, 3.14), (False,True))

    key = Vector([[1,2],[3,4]])
    assert clean_key(key) == ('Vector', (2,), (1,2,3,4), False)

    key = Vector([[1,2],[3,4]], (False,True))
    assert clean_key(key) == ('Vector', (2,), (1,2,3,4), (False,True))

    key = Vector([[1,2],[3,4]], drank=1)
    assert clean_key(key) == ('Vector', (), (1,2,3,4), False)

    path = LinearPath((0,0,0), 0., 'SSB')
    assert clean_key(path) == path.waypoint
    {path.waypoint}             # a set literal: TypeError if the key is unhashable

    frame = Rotation(1., 2, 'J2000')
    assert clean_key(frame) == frame.wayframe
    {frame.wayframe}            # a set literal: TypeError if the key is unhashable

    key = (1, Vector([[1,2],[3,4]]), path, frame)
    assert (clean_key(key)
            == (1, ('Vector', (2,), (1, 2, 3, 4), False), path.waypoint, frame.wayframe))
    {clean_key(key)}            # a set literal: TypeError if the key is unhashable

def test_clean_key_is_recursive():
    clean_key = _Cache.clean_key

    scalars = (Scalar(2.718), Scalar(3.14))
    cleaned = (('Scalar', (), 2.718, False), ('Scalar', (), 3.14, False))

    # A tuple nested inside the key is cleaned, not passed through
    key = ('coords', scalars)
    assert clean_key(key) == ('coords', cleaned)
    assert isinstance(hash(clean_key(key)), int)    # TypeError if unhashable

    # A nested list becomes a tuple, and its contents are cleaned too
    key = ('coords', list(scalars))
    assert clean_key(key) == ('coords', cleaned)
    assert isinstance(hash(clean_key(key)), int)    # TypeError if unhashable

    # Nesting can be arbitrarily deep
    key = (1, (2, (3, Scalar(3.14))))
    assert clean_key(key) == (1, (2, (3, ('Scalar', (), 3.14, False))))
    assert isinstance(hash(clean_key(key)), int)    # TypeError if unhashable

    # An object array is cleaned element by element; a numeric array is not
    key = np.empty((2,), dtype=object)
    key[0] = scalars[0]
    key[1] = scalars[1]
    assert clean_key(key) == ((2,), cleaned)
    assert isinstance(hash(clean_key(key)), int)    # TypeError if unhashable

    # Paths and Frames are converted at depth exactly as they are at the top level
    path = LinearPath((0,0,0), 0., 'SSB')
    frame = Rotation(1., 2, 'J2000')
    assert clean_key(((path,), (frame,))) == ((clean_key(path),), (clean_key(frame),))

def test_Cache():
    cache = _Cache()
    assert cache._maxsize == 100
    assert cache._extras == 10
    assert cache._limit == 110

    for key in range(110):
        cache[key] = str(key)

    assert len(cache) == 110
    assert 0 in cache
    assert 109 in cache
    assert cache[0] == '0'
    assert cache[109] == '109'
    assert cache[-1] is None

    cache[110] = '110'
    assert len(cache) == 100
    assert cache[0] == '0'
    assert cache[1] is None
    assert cache[11] is None
    assert cache[12] == '12'
    assert cache[110] == '110'
    assert 0 in cache
    assert 1 not in cache
    assert 11 not in cache
    assert 12 in cache

    # maxsize = 0
    cache = _Cache(maxsize=0)
    assert len(cache) == 0
    cache['pi'] = 3.14
    assert len(cache) == 0
    assert cache['pi'] is None

    # maxsize = 2
    cache = _Cache(maxsize=2)
    assert cache._maxsize == 2
    assert cache._extras == 3
    assert cache._limit == 5
    assert len(cache) == 0

    cache['pi'] = 3.14
    cache['e'] = 2.718
    cache['c'] = 3.e8
    cache['avogadro'] = 6.e23
    cache['h-bar'] = 1.054e-34
    assert len(cache) == 5

    cache['e']                  # touch 'e' so that it is not the next eviction
    cache['G'] = 6.67e-11
    assert len(cache) == 2
    assert 'e' in cache
    assert 'G' in cache
##########################################################################################
