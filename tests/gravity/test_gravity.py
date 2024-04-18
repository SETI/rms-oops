################################################################################
# tests/gravity/test_gravity.py
################################################################################

import numpy as np
import unittest

from oops.gravity.oblategravity import (JUPITER, SATURN, URANUS, NEPTUNE,
                                        PLUTO_CHARON)

ERROR_TOLERANCE = 1.e-15

class Test_Gravity(unittest.TestCase):

    def runTest(self):

        np.random.seed(6950)

        # Testing scalars in a loop...
        tests = 100
        planets = [JUPITER, SATURN, URANUS, NEPTUNE]
        factors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        for test in range(tests):
          for obj in planets:
            for e in (0., 0.1):
              for i in (0., 0.1):
                a = obj.rp * 10. ** (np.random.rand() * 2.)
                for f in factors:
                    b = obj.solve_a(obj.combo(a,f,e,i), f, e, i)
                    c = abs((b - a) / a)
                    self.assertTrue(c < ERROR_TOLERANCE)

        # PLUTO_CHARON with factors (1,0,0) and (0,0,1)
        for test in range(tests):
          for obj in [PLUTO_CHARON]:
            for e in (0., 0.1):
              for i in (0., 0.1):
                a = obj.rp * 10. ** (np.random.rand() * 2.)
                for f in [(1,0,0),(0,0,1)]:
                    b = obj.solve_a(obj.combo(a,f,e,i), f, e, i)
                    c = abs((b - a) / a)
                    self.assertTrue(c < ERROR_TOLERANCE)

        # PLUTO_CHARON with factors (0,1,0) can have duplicated values...
        for test in range(tests):
          for obj in [PLUTO_CHARON]:
            a = obj.rp * 10. ** (np.random.rand() * 2.)
            if obj.kappa2(a) < 0.:
                continue        # this would raise RuntimeError

            for f in [(0,1,0)]:
                combo1 = obj.combo(a,f)
                b = obj.solve_a(combo1, f)
                combo2 = obj.combo(b,f)
                c = abs((combo2 - combo1) / combo1)
                self.assertTrue(c < ERROR_TOLERANCE)

        # Testing a 100x100 array
        for obj in planets:
          a = obj.rp * 10. ** (np.random.rand(100,100) * 2.)
          for e in (0., 0.1):
            for i in (0., 0.1):
              for f in factors:
                b = obj.solve_a(obj.combo(a,f,e,i), f, e, i)
                c = abs((b - a) / a)
                self.assertTrue(np.all(c < ERROR_TOLERANCE))

        # Testing with first-order cancellation
        factors = [(1, -1, 0), (1, 0, -1), (0, 1, -1)]
        planets = [JUPITER, SATURN, URANUS, NEPTUNE]

        for obj in planets:
            a = obj.rp * 10. ** (np.random.rand(100,100) * 2.)
            for f in factors:
                b = obj.solve_a(obj.combo(a, f), f)
                c = abs((b - a) / a)
                self.assertTrue(np.all(c < ERROR_TOLERANCE))

        # Testing with second-order cancellation
        factors = [(2, -1, -1)]
        planets = [JUPITER, SATURN, URANUS, NEPTUNE]

        for obj in planets:
            a = obj.rp * 10. ** (np.random.rand(100,100) * 2.)
            for f in factors:
                b = obj.solve_a(obj.combo(a, f), f)
                c = abs((b - a) / a)
                self.assertTrue(np.all(c < ERROR_TOLERANCE))

########################################

if __name__ == '__main__':
    unittest.main()

################################################################################
