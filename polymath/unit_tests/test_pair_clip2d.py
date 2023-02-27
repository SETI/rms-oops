################################################################################
# Pair.swapxy() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Pair, Scalar, Units

class Test_Pair_clip2d(unittest.TestCase):

  # runTest
  def runTest(self):

    ############################################################################
    # clip2d()
    ############################################################################

    a = Pair([[1,2],[3,4],[5,6]])

    self.assertEqual(a.clip2d([2,3],[4,5],False), [[2,3],[3,4],[4,5]])
    self.assertTrue(np.all(a.clip2d([2,3],[4,5],True).mask == [True,False,True]))

    self.assertEqual(a.clip2d(None,[4,5],False), [[1,2],[3,4],[4,5]])
    self.assertTrue(np.all(a.clip2d(None,[4,5],True).mask == [False,False,True]))

    lower = Pair([2,3], True)
    self.assertEqual(a.clip2d(lower,[4,5],False), [[1,2],[3,4],[4,5]])
    self.assertTrue(np.all(a.clip2d(lower,[4,5],True).mask == [False,False,True]))

############################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
