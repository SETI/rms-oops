##########################################################################################
# test/test_cache.py
##########################################################################################

import unittest
import numpy as np
from oops.cache import Cache
from oops.frame import Rotation
from oops.path  import LinearPath
from polymath   import Scalar, Vector


class Test_Cache(unittest.TestCase):

    def test_clean_key(self):

        clean_key = Cache.clean_key

        key = 1
        self.assertEqual(clean_key(key), 1)
        self.assertIsInstance(clean_key(key), int)

        key = 2.
        self.assertEqual(clean_key(key), 2.)
        self.assertIsInstance(clean_key(key), float)

        key = True
        self.assertEqual(clean_key(key), True)
        self.assertIsInstance(clean_key(key), bool)

        key = False
        self.assertEqual(clean_key(key), False)
        self.assertIsInstance(clean_key(key), bool)

        key = 'abc'
        self.assertEqual(clean_key(key), 'abc')

        key = None
        self.assertIs(clean_key(key), None)

        key = [1]
        self.assertEqual(clean_key(key), (1,))

        key = [2, 3., 'four']
        self.assertEqual(clean_key(key), (2, 3., 'four'))

        key = np.array(4.)
        self.assertEqual(clean_key(key), ((), (4.,)))
        self.assertIsInstance(clean_key(key)[1][0], np.float64)

        key = np.array([[1,2],[3,4]])
        self.assertEqual(clean_key(key), ((2,2), (1,2,3,4)))
        self.assertIsInstance(clean_key(key)[-1][-1], np.int64)

        key = Scalar(3.14)
        self.assertEqual(clean_key(key), ('Scalar', (), 3.14, False))

        key = Scalar((2.718, 3.14))
        self.assertEqual(clean_key(key), ('Scalar', (2,), (2.718, 3.14), False))

        key = Scalar((2.718, 3.14), True)
        self.assertEqual(clean_key(key), ('Scalar', (2,), (2.718, 3.14), True))

        key = Scalar((2.718, 3.14), (False,True))
        self.assertEqual(clean_key(key), ('Scalar', (2,), (2.718, 3.14), (False,True)))

        key = Vector([[1,2],[3,4]])
        self.assertEqual(clean_key(key), ('Vector', (2,), (1,2,3,4), False))

        key = Vector([[1,2],[3,4]], (False,True))
        self.assertEqual(clean_key(key), ('Vector', (2,), (1,2,3,4), (False,True)))

        key = Vector([[1,2],[3,4]], drank=1)
        self.assertEqual(clean_key(key), ('Vector', (), (1,2,3,4), False))

        path = LinearPath((0,0,0), 0., 'SSB')
        self.assertEqual(clean_key(path), path.waypoint)
        test = {path.waypoint}      # TypeError if unhashable

        frame = Rotation(1., 2, 'J2000')
        self.assertEqual(clean_key(frame), frame.wayframe)
        test = {frame.wayframe}     # TypeError if unhashable

        key = (1, Vector([[1,2],[3,4]]), path, frame)
        self.assertEqual(clean_key(key), (1, ('Vector', (2,), (1, 2, 3, 4), False),
                                          path.waypoint, frame.wayframe))
        test = {key}                # TypeError if unhashable

    def test_Cache(self):

        cache = Cache()
        self.assertEqual(cache._maxsize, 100)
        self.assertEqual(cache._extras, 10)
        self.assertEqual(cache._limit, 110)

        for key in range(110):
            cache[key] = str(key)

        self.assertEqual(len(cache), 110)
        self.assertIn(0, cache)
        self.assertIn(109, cache)
        self.assertEqual(cache[0], '0')
        self.assertEqual(cache[109], '109')
        self.assertEqual(cache[-1], None)

        cache[110] = '110'
        self.assertEqual(len(cache), 100)
        self.assertEqual(cache[0], '0')
        self.assertEqual(cache[1], None)
        self.assertEqual(cache[11], None)
        self.assertEqual(cache[12], '12')
        self.assertEqual(cache[110], '110')
        self.assertIn(0, cache)
        self.assertNotIn(1, cache)
        self.assertNotIn(11, cache)
        self.assertIn(12, cache)

        # maxsize = 0
        cache = Cache(maxsize=0)
        self.assertEqual(len(cache), 0)
        cache['pi'] = 3.14
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache['pi'], None)

        # maxsize = 2
        cache = Cache(maxsize=2)
        self.assertEqual(cache._maxsize, 2)
        self.assertEqual(cache._extras, 3)
        self.assertEqual(cache._limit, 5)
        self.assertEqual(len(cache), 0)

        cache['pi'] = 3.14
        cache['e'] = 2.718
        cache['c'] = 3.e8
        cache['avogadro'] = 6.e23
        cache['h-bar'] = 1.054e-34
        self.assertEqual(len(cache), 5)

        ignore = cache['e']
        cache['G'] = 6.67e-11
        self.assertEqual(len(cache), 2)
        self.assertIn('e', cache)
        self.assertIn('G', cache)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
##########################################################################################
