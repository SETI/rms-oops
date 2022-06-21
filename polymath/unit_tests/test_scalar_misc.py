################################################################################
# Old Scalar tests, updated by MRS 2/18/14
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Vector, Boolean, Units

#*******************************************************************************
# Test_Scalar_misc
#*******************************************************************************
class Test_Scalar_misc(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    #----------------------------
    # Arithmetic operations	     
    #----------------------------
    ints = Scalar((1,2,3))
    test = Scalar(np.array([1,2,3]))
    self.assertEqual(ints, test)

    test = Scalar(test)
    self.assertEqual(ints, test)

    self.assertEqual(ints, (1,2,3))
    self.assertEqual(ints, [1,2,3])

    self.assertEqual(ints.shape, (3,))

    self.assertEqual(-ints, [-1,-2,-3])
    self.assertEqual(+ints, [1,2,3])

    self.assertEqual(ints, abs(ints))
    self.assertEqual(ints, abs(Scalar(( 1, 2, 3))))
    self.assertEqual(ints, abs(Scalar((-1,-2,-3))))

    self.assertEqual(ints * 2, [2,4,6])
    self.assertEqual(ints / 2., [0.5,1,1.5])
    #self.assertEqual(ints / 2, [0,1,1])             # now truediv
    self.assertEqual(ints / 2, [0.5,1,1.5])         # now truediv
    self.assertEqual(ints + 1, [2,3,4])
    self.assertEqual(ints - 0.5, (0.5,1.5,2.5))
    self.assertEqual(ints % 2, (1,0,1))

    self.assertEqual(ints + Scalar([1,2,3]), [2,4,6])
    self.assertEqual(ints - Scalar((1,2,3)), [0,0,0])
    self.assertEqual(ints * [1,2,3], [1,4,9])
    self.assertEqual(ints / [1,2,3], [1,1,1])
    self.assertEqual(ints % [1,3,3], [0,2,0])

    self.assertRaises(ValueError, ints.__add__, (4,5))
    self.assertRaises(ValueError, ints.__sub__, (4,5))
    self.assertRaises(ValueError, ints.__mul__, (4,5))
    self.assertRaises(ValueError, ints.__div__, (4,5))
    self.assertRaises(ValueError, ints.__mod__, (4,5))

    self.assertRaises(ValueError, ints.__add__, Scalar((4,5)))
    self.assertRaises(ValueError, ints.__sub__, Scalar((4,5)))
    self.assertRaises(ValueError, ints.__mul__, Scalar((4,5)))
    self.assertRaises(ValueError, ints.__div__, Scalar((4,5)))
    self.assertRaises(ValueError, ints.__mod__, Scalar((4,5)))

    #------------------
    # Integer ops          
    #------------------
    ints = Scalar((1,2,3))
    ints += 1
    self.assertEqual(ints, [2,3,4])

    ints -= 1
    self.assertEqual(ints, [1,2,3])

    ints *= 2
    self.assertEqual(ints, [2,4,6])

    ints //= 2
    self.assertEqual(ints, [1,2,3])

    ints *= (3,2,1)
    self.assertEqual(ints, [3,4,3])

    ints //= (1,2,3)
    self.assertEqual(ints, [3,2,1])

    ints += (1,2,3)
    self.assertEqual(ints, 4)
    self.assertEqual(ints, [4])
    self.assertEqual(ints, [4,4,4])
    self.assertEqual(ints, Scalar([4,4,4]))

    ints -= (3,2,1)
    self.assertEqual(ints, [1,2,3])

    test = Scalar((10,10,10))
    test %= 4
    self.assertEqual(test, 2)

    test = Scalar((10,10,10))
    test %= (4,3,2)
    self.assertEqual(test, [2,1,0])

    test = Scalar((10,10,10))
    test %= Scalar((5,4,3))
    self.assertEqual(test, [0,2,1])

    self.assertRaises(ValueError, ints.__iadd__, (4,5))
    self.assertRaises(ValueError, ints.__isub__, (4,5))
    self.assertRaises(ValueError, ints.__imul__, (4,5))
    self.assertRaises(ValueError, ints.__imod__, (4,5))
    self.assertRaises(ValueError, ints.__ifloordiv__, (4,5))

    self.assertRaises(ValueError, ints.__iadd__, Scalar((4,5)))
    self.assertRaises(ValueError, ints.__isub__, Scalar((4,5)))
    self.assertRaises(ValueError, ints.__imul__, Scalar((4,5)))
    self.assertRaises(ValueError, ints.__imod__, Scalar((4,5)))
    self.assertRaises(ValueError, ints.__ifloordiv__, Scalar((4,5)))

    self.assertRaises(TypeError, ints.__idiv__, (4,5))
    self.assertRaises(TypeError, ints.__idiv__, Scalar((4,5)))

    #---------------
    # Float ops     	
    #---------------
    floats = Scalar((1.,2.,3.))
    floats += 1
    self.assertEqual(floats, [2,3,4])

    floats -= 1
    self.assertEqual(floats, [1,2,3])

    floats *= 2
    self.assertEqual(floats, [2,4,6])

    floats /= 2
    self.assertEqual(floats, [1,2,3])

    floats *= (3,2,1)
    self.assertEqual(floats, [3,4,3])

    floats /= (1,2,3)
    self.assertEqual(floats, [3,2,1])

    floats += (1,2,3)
    self.assertEqual(floats, 4)
    self.assertEqual(floats, [4])
    self.assertEqual(floats, [4,4,4])
    self.assertEqual(floats, Scalar([4,4,4]))

    floats -= (3,2,1)
    self.assertEqual(floats, [1,2,3])

    test = Scalar((10,10,10))
    test %= 4
    self.assertEqual(test, 2)

    test = Scalar((10,10,10))
    test %= (4,3,2)
    self.assertEqual(test, [2,1,0])

    test = Scalar((10,10,10))
    test %= Scalar((5,4,3))
    self.assertEqual(test, [0,2,1])

    self.assertRaises(ValueError, floats.__iadd__, (4,5))
    self.assertRaises(ValueError, floats.__isub__, (4,5))
    self.assertRaises(ValueError, floats.__imul__, (4,5))
    self.assertRaises(ValueError, floats.__idiv__, (4,5))
    self.assertRaises(ValueError, floats.__imod__, (4,5))
    self.assertRaises(ValueError, floats.__ifloordiv__, (4,5))

    self.assertRaises(ValueError, floats.__iadd__, Scalar((4,5)))
    self.assertRaises(ValueError, floats.__isub__, Scalar((4,5)))
    self.assertRaises(ValueError, floats.__imul__, Scalar((4,5)))
    self.assertRaises(ValueError, floats.__idiv__, Scalar((4,5)))
    self.assertRaises(ValueError, floats.__imod__, Scalar((4,5)))
    self.assertRaises(ValueError, floats.__ifloordiv__, Scalar((4,5)))

    #------------------------
    # Generic operations     	 
    #------------------------
    self.assertEqual(ints[0], 1)

    floats = ints.as_float()
    self.assertEqual(floats[0], 1.)

    six = Scalar([1,2,3,4,5,6])
    self.assertEqual(six.shape, (6,))

    test = six.copy().reshape((3,1,2))
    self.assertEqual(test.shape, (3,1,2))
    self.assertEqual(test, [[[1,2]],[[3,4]],[[5,6]]])
    self.assertEqual(test.swap_axes(0,1).shape, (1,3,2))
    self.assertEqual(test.swap_axes(0,2).shape, (2,1,3))
    self.assertEqual(test.flatten().shape, (6,))

    four = Scalar([1,2,3,4]).reshape((2,2))
    self.assertEqual(four, [[1,2],[3,4]])

    self.assertEqual(Qube.broadcasted_shape(four,test), (3,2,2))
    self.assertEqual(four.broadcast_into_shape((3,2,2)),
                           [[[1,2],[3,4]],
                            [[1,2],[3,4]],
                            [[1,2],[3,4]]])
    self.assertEqual(test.broadcast_into_shape((3,2,2)),
                           [[[1,2],[1,2]],
                            [[3,4],[3,4]],
                            [[5,6],[5,6]]])
    self.assertEqual([[[1,2],[3,4]],
                      [[1,2],[3,4]],
                      [[1,2],[3,4]]], four.broadcast_into_shape((3,2,2)))
    self.assertEqual([[[1,2],[1,2]],
                      [[3,4],[3,4]],
                      [[5,6],[5,6]]], test.broadcast_into_shape((3,2,2)))

    ten = four + test
    self.assertEqual(ten.shape, (3,2,2))
    self.assertEqual(ten, [[[2, 4], [4, 6]],
                           [[4, 6], [6, 8]],
                           [[6, 8], [8,10]]])

    x24 = four * test
    self.assertEqual(x24.shape, (3,2,2))
    self.assertEqual(x24, [[[1, 4], [ 3, 8]],
                           [[3, 8], [ 9,16]],
                           [[5,12], [15,24]]])

    #-----------------
    # Mask tests          
    #-----------------
    test = Scalar(list(range(6)))
    self.assertEqual(str(test), "Scalar(0 1 2 3 4 5)")

    test = Scalar(test, mask=(3*[True] + 3*[False]))

    self.assertEqual(str(test),   "Scalar(-- -- -- 3 4 5; mask)")
    self.assertEqual(str(test+1), "Scalar(-- -- -- 4 5 6; mask)")
    self.assertEqual(str(test-2), "Scalar(-- -- -- 1 2 3; mask)")
    self.assertEqual(str(test*2), "Scalar(-- -- -- 6 8 10; mask)")
    self.assertEqual(str(test/2), "Scalar(-- -- -- 1.5 2.0 2.5; mask)")
    self.assertEqual(str(test%2), "Scalar(-- -- -- 1 0 1; mask)")

    self.assertEqual(str(test-2.), "Scalar(-- -- -- 1.0 2.0 3.0; mask)")
    self.assertEqual(str(test+2.), "Scalar(-- -- -- 5.0 6.0 7.0; mask)")
    self.assertEqual(str(test*2.), "Scalar(-- -- -- 6.0 8.0 10.0; mask)")
    self.assertEqual(str(test/2.), "Scalar(-- -- -- 1.5 2.0 2.5; mask)")

    self.assertEqual(str(test + [1, 2, 3, 4, 5, 6]),
                     "Scalar(-- -- -- 7 9 11; mask)")
    self.assertEqual(str(test - [1, 2, 3, 4, 5, 6]),
                     "Scalar(-- -- -- -1 -1 -1; mask)")
    self.assertEqual(str(test * [1, 2, 3, 4, 5, 6]),
                     "Scalar(-- -- -- 12 20 30; mask)")
    self.assertEqual(str(test / [1, 7, 5, 1, 2, 1]),
                     "Scalar(-- -- -- 3.0 2.0 5.0; mask)")
    self.assertEqual(str(test / [0, 7, 5, 1, 2, 0]),
                     "Scalar(-- -- -- 3.0 2.0 --; mask)")
    self.assertEqual(str(test % [0, 7, 5, 1, 2, 0]),
                     "Scalar(-- -- -- 0 0 --; mask)")

    temp = Scalar(6*[1], 5*[False] + [True])
    self.assertEqual(str(temp), "Scalar(1 1 1 1 1 --; mask)")


    self.assertEqual(str(test + temp), "Scalar(-- -- -- 4 5 --; mask)")

    foo = test + temp
    self.assertTrue(foo.vals[0] == test.vals[0] + temp.vals[0])

    foo.vals[0] = 99
    self.assertFalse(foo.vals[0] == test.vals[0] + temp.vals[0])

    self.assertEqual(foo, test + temp)

    self.assertEqual(test[5], 5)
    self.assertEqual(test[-1], 5)
    self.assertEqual(test[3:], [3,4,5])
    self.assertEqual(test[3:5], [3,4])
    self.assertEqual(test[3:-1], [3,4])

    self.assertEqual(test[0], Scalar(0, True))

    self.assertEqual(str(test[0]), "Scalar(--; mask)")
    self.assertEqual(str(test[0:4]), "Scalar(-- -- -- 3; mask)")
    self.assertEqual(str(test[0:1]), "Scalar(--; mask)")
    self.assertEqual(str(test[5]), "Scalar(5)")
    self.assertEqual(str(test[4:]), "Scalar(4 5)")
    self.assertEqual(str(test[5:]), "Scalar(5)")
    self.assertEqual(str(test[0:6:2]), "Scalar(-- -- 4; mask)")

    mvals = test.mvals
    self.assertEqual(type(mvals), np.ma.MaskedArray)
    self.assertEqual(str(mvals), "[-- -- -- 3 4 5]")

    temp = Scalar(list(range(6)))
    mvals = temp.mvals
    self.assertEqual(type(mvals), np.ma.MaskedArray)
    self.assertEqual(str(mvals), "[0 1 2 3 4 5]")
    self.assertEqual(mvals.mask, np.ma.nomask)

    temp = Scalar(temp, mask=True)
    self.assertEqual(str(temp), "Scalar(-- -- -- -- -- --; mask)")

    mvals = temp.mvals
    self.assertEqual(type(mvals), np.ma.MaskedArray)
    self.assertEqual(str(mvals), "[-- -- -- -- -- --]")

    #--------------------
    # Test of units	     
    #--------------------
    test = Scalar(list(range(6)))
    self.assertEqual(test, np.arange(6))
    eps = 1.e-7

    km = Scalar(list(range(6)), units=Units.KM)
    cm = Scalar(np.arange(6), units=Units.CM)
    self.assertTrue(np.all(km.values == cm.values))

    cm = cm.into_units()
    EPS = 1.e-15
    self.assertTrue(np.all(np.abs(km.values - cm.values/1.e5) < 1.e5*EPS))

    self.assertRaises(ValueError, cm.set_units, Units.SECONDS)
  #=============================================================================



#*******************************************************************************



################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
