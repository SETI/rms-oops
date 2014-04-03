################################################################################
# General index tests - masked and non-masked
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Scalar, Pair, Vector, Matrix

class Test_Indices(unittest.TestCase):

  def runTest(self):

        def make_masked(orig, mask_list):
            ret = orig.copy()
            ret[mask_list] = np.ma.masked
            return ret

        def extract(a, indices):
            ret = []
            for index in indices:
                ret.append(a[index])
            return np.ma.array(ret)

        #
        # An unmasked Scalar with traditional indexing
        #

        b = np.ma.arange(10)
        a = Scalar(b)

        self.assertEqual(a, b)
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[1:5], b[1:5])
        self.assertEqual(a[:], b[:])
        self.assertEqual(a[...,:], b[...,:])

        #
        # An unmasked Scalar indexed by a Scalar
        #

        # Single element
        self.assertEqual(a[Scalar(1)], b[1])
        self.assertEqual(a[Scalar(1,True)], make_masked(b, [1])[1])

        # Two elements
        self.assertEqual(a[Scalar((1,2))], b[1:3])
        self.assertEqual(a[Scalar((1,2),(True,False))], make_masked(b, [1])[1:3])
        self.assertEqual(a[Scalar((1,2),True)], make_masked(b, [1,2])[1:3])

        #
        # A fully masked Scalar with traditional indexing
        #

        b = np.ma.arange(10)
        b[:] = np.ma.masked
        a = Scalar(b)

        self.assertEqual(a, b)
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[1:5], b[1:5])
        self.assertEqual(a[:], b[:])
        self.assertEqual(a[...,:], b[...,:])

        #
        # A fully masked Scalar indexed by a Scalar
        #

        b = np.ma.arange(10)
        b[:] = np.ma.masked
        a = Scalar(b)

        # Single element
        self.assertEqual(a[Scalar(1)], b[1])
        self.assertEqual(a[Scalar(1,True)], make_masked(b, [1])[1])

        # Two elements
        self.assertEqual(a[Scalar((1,2))], b[1:3])
        self.assertEqual(a[Scalar((1,2),(True,False))], make_masked(b, [1])[1:3])
        self.assertEqual(a[Scalar((1,2),True)], make_masked(b, [1,2])[1:3])

        #
        # A partially masked Scalar with traditional indexing
        #

        b = np.ma.arange(10)
        b[3] = np.ma.masked
        a = Scalar(b)

        self.assertEqual(a, b)
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[1:5], b[1:5])
        self.assertEqual(a[:], b[:])
        self.assertEqual(a[...,:], b[...,:])

        #
        # A partially masked Scalar indexed by a Scalar
        #

        b = np.ma.arange(10)
        b[3] = np.ma.masked
        a = Scalar(b)

        # Single element
        self.assertEqual(a[Scalar(1)], b[1])
        self.assertEqual(a[Scalar(1,True)], make_masked(b, [1])[1])

        # Two elements
        self.assertEqual(a[Scalar((1,2))], b[1:3])
        self.assertEqual(a[Scalar((1,2),(True,False))], make_masked(b, [1])[1:3])
        self.assertEqual(a[Scalar((1,2),True)], make_masked(b, [1,2])[1:3])

        #
        # An unmasked 2-D Scalar with traditional indexing
        #

        b = np.ma.arange(25).reshape(5,5)
        a = Scalar(b)

        # Traditional indexing
        self.assertEqual(a, b)
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[1:5], b[1:5])
        self.assertEqual(a[:], b[:])
        self.assertEqual(a[...,:], b[...,:])

        #
        # An unmasked 2-D Scalar indexed by a Scalar
        #

        b = np.ma.arange(25).reshape(5,5)
        a = Scalar(b)

        # Single element
        self.assertEqual(a[Scalar(1)], b[1])
        self.assertEqual(a[Scalar(1,True)], make_masked(b, [1])[1])

        # Two elements
        self.assertEqual(a[Scalar((1,2))], b[1:3])
        self.assertEqual(a[Scalar((1,2),(True,False))], make_masked(b, [1])[1:3])
        self.assertEqual(a[Scalar((1,2),True)], make_masked(b, [1,2])[1:3])

        #
        # An unmasked 2-D Scalar indexed by a Pair
        #

        b = np.ma.arange(25).reshape(5,5)
        a = Scalar(b)

        self.assertEqual(a[Pair((1,1))], b[1,1])
        self.assertEqual(a[Pair((1,1),True)], make_masked(b, [[1,1]])[1,1])
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)))],
                         extract(b, ((1,1),(2,2),(3,3))))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),False)],
                         extract(b, ((1,1),(2,2),(3,3))))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),True)],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [0,1,2]))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),(True,False,False))],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [0]))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),(False,True,False))],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [1]))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),(False,False,True))],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [2]))

        #
        # A partially masked 2-D Scalar indexed by a Pair
        #

        b = np.ma.arange(25).reshape(5,5)
        b[1,1] = np.ma.masked
        a = Scalar(b)

        self.assertEqual(a[Pair((1,1))], b[1,1])
        self.assertEqual(a[Pair((1,1),True)], make_masked(b, [[1,1]])[1,1])
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)))],
                         extract(b, ((1,1),(2,2),(3,3))))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),False)],
                         extract(b, ((1,1),(2,2),(3,3))))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),True)],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [0,1,2]))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),(True,False,False))],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [0]))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),(False,True,False))],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [1]))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),(False,False,True))],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [2]))

        #
        # An unmasked 3-D Scalar with traditional indexing
        #

        b = np.ma.arange(125).reshape(5,5,5)
        a = Scalar(b)

        self.assertEqual(a, b)
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[1:5], b[1:5])
        self.assertEqual(a[:], b[:])
        self.assertEqual(a[...,:], b[...,:])
        self.assertEqual(a[0,0,0], b[0,0,0])
        self.assertEqual(a[:,0,0], b[:,0,0])
        self.assertEqual(a[0,:,0], b[0,:,0])
        self.assertEqual(a[0,0,:], b[0,0,:])
        self.assertEqual(a[:,:,0], b[:,:,0])
        self.assertEqual(a[0,:,:], b[0,:,:])
        self.assertEqual(a[...,0], b[...,0])
        self.assertEqual(a[0,...], b[0,...])

        #
        # An unmasked 3-D Scalar indexed by a Scalar
        #

        b = np.ma.arange(125).reshape(5,5,5)
        a = Scalar(b)

        # Single element
        self.assertEqual(a[Scalar(1)], b[1])
        self.assertEqual(a[Scalar(1,True)], make_masked(b, [1])[1])

        # Two elements
        self.assertEqual(a[Scalar((1,2))], b[1:3])
        self.assertEqual(a[Scalar((1,2),(True,False))], make_masked(b, [1])[1:3])
        self.assertEqual(a[Scalar((1,2),True)], make_masked(b, [1,2])[1:3])

        #
        # An unmasked 3-D Scalar indexed by a Pair
        #

        b = np.ma.arange(125).reshape(5,5,5)
        a = Scalar(b)

        self.assertEqual(a[Pair((1,1))], b[1,1])
        self.assertEqual(a[Pair((1,1),True)], make_masked(b, [[1,1]])[1,1])
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)))],
                         extract(b, ((1,1),(2,2),(3,3))))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),False)],
                         extract(b, ((1,1),(2,2),(3,3))))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),True)],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [0,1,2]))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),(True,False,False))],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [0]))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),(False,True,False))],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [1]))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),(False,False,True))],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [2]))

        #
        # A partially masked 3-D Scalar indexed by a Pair
        #

        b = np.ma.arange(125).reshape(5,5,5)
        b[1,1,1] = np.ma.masked
        a = Scalar(b)

        self.assertEqual(a[Pair((1,1))], b[1,1])
        self.assertEqual(a[Pair((1,1),True)], make_masked(b, [[1,1]])[1,1])
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)))],
                         extract(b, ((1,1),(2,2),(3,3))))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),False)],
                         extract(b, ((1,1),(2,2),(3,3))))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),True)],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [0,1,2]))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),(True,False,False))],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [0]))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),(False,True,False))],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [1]))
        self.assertEqual(a[Pair(((1,1),(2,2),(3,3)),(False,False,True))],
                         make_masked(extract(b, ((1,1),(2,2),(3,3))), [2]))

        #
        # An unmasked 3-D Scalar indexed by a Vector
        #

        b = np.ma.arange(125).reshape(5,5,5)
        a = Scalar(b)

        self.assertEqual(a[Vector((1,1,1))], b[1,1,1])
        self.assertEqual(a[Vector((1,1,1),True)],
                         make_masked(b, [[1,1,1]])[1,1,1])
        self.assertEqual(a[Vector(((1,1,1),(2,2,2),(3,3,3)))],
                         extract(b, ((1,1,1),(2,2,2),(3,3,3))))
        self.assertEqual(a[Vector(((1,1,1),(2,2,2),(3,3,3)),False)],
                         extract(b, ((1,1,1),(2,2,2),(3,3,3))))
        self.assertEqual(a[Vector(((1,1,1),(2,2,2),(3,3,3)),True)],
                         make_masked(extract(b, ((1,1,1),(2,2,2),(3,3,3))),
                                     [0,1,2]))
        self.assertEqual(a[Vector(((1,1,1),(2,2,2),(3,3,3)),(True,False,False))],
                         make_masked(extract(b, ((1,1,1),(2,2,2),(3,3,3))),
                                     [0]))
        self.assertEqual(a[Vector(((1,1,1),(2,2,2),(3,3,3)),(False,True,False))],
                         make_masked(extract(b, ((1,1,1),(2,2,2),(3,3,3))),
                                     [1]))
        self.assertEqual(a[Vector(((1,1,1),(2,2,2),(3,3,3)),(False,False,True))],
                         make_masked(extract(b, ((1,1,1),(2,2,2),(3,3,3))),
                                     [2]))

        #
        # An unmasked 3-D Scalar indexed by mixed types
        #

        b = np.ma.arange(125).reshape(5,5,5)
        a = Scalar(b)

        self.assertEqual(a[0,Scalar(0),0], 0.)
        self.assertRaises(ValueError, a.__getitem__, [0,Scalar(0,True),0])
        self.assertRaises(ValueError, a.__getitem__, [slice(Ellipsis), Scalar(0,True)])
        self.assertRaises(ValueError, a.__getitem__, [Scalar(0,True), slice(Ellipsis)])

        #
        # An unmasked 2-D Matrix indexed by a Pair
        #

        b = np.ma.arange(16).reshape(2,2,2,2)
        a = Matrix(b)

        self.assertEqual(a[Pair((1,1))], b[1,1])
        self.assertEqual(a[Pair((1,1),True)], make_masked(b, [[1,1]])[1,1])
        self.assertEqual(a[...,1], b[...,1,:,:])
        self.assertEqual(a[...,Scalar(1)], b[...,1,:,:])
        self.assertRaises(ValueError, a.__getitem__, [slice(Ellipsis),Scalar(1,True)])

        #
        # A partially masked 2-D Matrix indexed by a Pair
        #

        b = np.ma.arange(16.).reshape(2,2,2,2)
        a = Matrix(b.copy(), ((False,True),(False,False)))
        b[0,1,:,:] = np.ma.masked

        self.assertEqual(a[Pair((1,1))], b[1,1])
        self.assertEqual(a[Pair((1,1),True)], make_masked(b, [[1,1]])[1,1])
#        self.assertEqual(a[...,1], b[...,1,:,:])
#        self.assertEqual(a[...,Scalar(1)], b[...,1,:,:])
        self.assertRaises(ValueError, a.__getitem__, [slice(Ellipsis),Scalar(1,True)])

        #
        # Assignment to a 1-D Scalar
        #

        b = np.zeros(10)
        a = Scalar(b)

        a[2] = 1
        self.assertEqual(a, Scalar((0,0,1,0,0,0,0,0,0,0)))
        a[Scalar(3)] = 1
        self.assertEqual(a, Scalar((0,0,1,1,0,0,0,0,0,0)))
        a[Scalar(4,True)] = 1
        self.assertEqual(a, Scalar((0,0,1,1,0,0,0,0,0,0)))
        a[Scalar((5,6,7),(True,False,True))] = 1
        self.assertEqual(a, Scalar((0,0,1,1,0,0,1,0,0,0)))
        a[Scalar(1)] = Scalar(2,True)
        self.assertEqual(a, Scalar((0,0,1,1,0,0,1,0,0,0),
                                   (False,True,False,False,False,False,
                                    False,False,False,False)))
        a[Scalar(0,True)] = Scalar(3,True)
        self.assertEqual(a, Scalar((0,0,1,1,0,0,1,0,0,0),
                                   (False,True,False,False,False,False,
                                    False,False,False,False)))
        a[Scalar((0,2,4))] = Scalar(4,True)
        self.assertEqual(a, Scalar((0,0,1,1,0,0,1,0,0,0),
                                   (True,True,True,False,True,False,
                                    False,False,False,False)))
        a[Scalar((0,2,4))] = Scalar((5,6,7))
        self.assertEqual(a, Scalar((5,0,6,1,7,0,1,0,0,0),
                                   (False,True,False,False,False,False,
                                    False,False,False,False)))
        a[:] = 9
        self.assertEqual(a, Scalar([9]*10))

        #
        # Assignment to a 2-D Scalar
        #

        a = Scalar(((0,0,0),(0,0,0)))
        a[Pair((1,2))] = 1
        self.assertEqual(a, Scalar(((0,0,0),(0,0,1))))
        a[Pair((1,2),True)] = 2
        self.assertEqual(a, Scalar(((0,0,0),(0,0,1))))
        a[Pair((1,2))] = Scalar(0,True)
        self.assertEqual(a, Scalar(((0,0,0),(0,0,1)),
                                   ((False,False,False),
                                    (False,False,True))))
        a[Scalar(1,True)] = Scalar(1,True)
        self.assertEqual(a, Scalar(((0,0,0),(0,0,1)),
                                   ((False,False,False),
                                    (False,False,True))))
        a[Scalar(1)] = Scalar(1,True)
        self.assertEqual(a, Scalar(((0,0,0),(0,0,1)),
                                   ((False,False,False),
                                    (True,True,True))))
        a[Pair(((0,0),(0,1),(0,2)),(False,True,False))] = 2
        self.assertEqual(a, Scalar(((2,0,2),(0,0,1)),
                                   ((False,False,False),
                                    (True,True,True))))


        #
        # Assignment to a 2-D Matrix indexed 2-D
        #

        a = Matrix(np.zeros(16).reshape(2,2,2,2))

        a[Pair((1,1))] = Matrix(((1,2),(3,4)))
        self.assertEqual(a, Matrix(((((0,0),(0,0)), ((0,0),(0,0))),
                                    (((0,0),(0,0)), ((1,2),(3,4))))))
        a[Pair((1,1),True)] = Matrix(((5,5),(5,5)))
        self.assertEqual(a, Matrix(((((0,0),(0,0)), ((0,0),(0,0))),
                                    (((0,0),(0,0)), ((1,2),(3,4))))))
        a[...,1] = Matrix(((5,6),(7,8)))
        self.assertEqual(a, Matrix(((((0,0),(0,0)), ((5,6),(7,8))),
                                    (((0,0),(0,0)), ((5,6),(7,8))))))
        a[...,Scalar(1)] = Matrix(((1,2),(3,4)))
        self.assertEqual(a, Matrix(((((0,0),(0,0)), ((1,2),(3,4))),
                                    (((0,0),(0,0)), ((1,2),(3,4))))))
        a[...,Scalar(1,True)] = Matrix(((8,8),(8,8)))
        self.assertEqual(a, Matrix(((((0,0),(0,0)), ((1,2),(3,4))),
                                    (((0,0),(0,0)), ((1,2),(3,4))))))
        a[Pair((0,1))] = Matrix(((9,9),(9,9)),True)
        self.assertEqual(a, Matrix(((((0,0),(0,0)), ((1,2),(3,4))),
                                    (((0,0),(0,0)), ((1,2),(3,4)))),
                                   ((((False,True), (False,False))))))
        a[Pair((0,0),True)] = Matrix(((10,10),(10,10)),True)
        self.assertEqual(a, Matrix(((((0,0),(0,0)), ((1,2),(3,4))),
                                    (((0,0),(0,0)), ((1,2),(3,4)))),
                                   ((((False,True), (False,False))))))

        self.assertRaises(ValueError, a.__getitem__, [slice(Ellipsis),Scalar(1,True)])

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
