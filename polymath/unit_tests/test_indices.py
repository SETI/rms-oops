################################################################################
# General index tests - masked and non-masked
################################################################################

from __future__ import division
import warnings
import numpy as np
import unittest

from polymath import Scalar, Pair, Vector, Matrix

class Test_Indices(unittest.TestCase):

  def runTest(self):

        def make_masked(orig, mask_list):
            ret = orig.copy()
            ret[np.array(mask_list)] = np.ma.masked
            return ret

        def extract(a, indices):
            ret = []
            for index in indices:
                ret.append(a[index])

            # NOTE: can raise UserWarning:
            #    Warning: converting a masked element to nan.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = np.ma.array(ret)

            return result

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

        self.assertEqual(a[(0,Scalar(3),0)], 15.)
        self.assertEqual(a[(0,Scalar(3))], [15,16,17,18,19])
        self.assertEqual(a[(0,Scalar(3,True),0)], Scalar.MASKED)
        self.assertEqual(a[(0,Scalar(3,True))].shape, (5,))
        self.assertTrue(np.all(a[(0,Scalar(3,True))].mask == True))

        self.assertEqual(a[(Ellipsis, Scalar([0,1],False))], b[...,(0,1)])
        self.assertTrue(np.all(a[(Ellipsis, Scalar([0,1],True))].mask == True))

        indx = (Scalar([1,2]), Ellipsis, Scalar([0,1]))
        self.assertEqual(a[indx], b[(1,2),...,(0,1)])

        indx = (Scalar([1,2],True), Ellipsis, Scalar([0,1]))
        self.assertTrue(np.all(a[indx].mask == True))

        indx = (Scalar([1,2],True), Ellipsis, Scalar([0,1],True))
        self.assertTrue(np.all(a[indx].mask == True))

        #
        # An unmasked 2-D Matrix indexed by a Pair
        #

        b = np.ma.arange(16).reshape(2,2,2,2)
        a = Matrix(b)

        self.assertEqual(a[Pair((1,1))], b[1,1])
        self.assertEqual(a[Pair((1,1),True)], make_masked(b, [[1,1]])[1,1])
        self.assertEqual(a[...,1], b[...,1,:,:])
        self.assertEqual(a[...,Scalar(1)], b[...,1,:,:])
        self.assertTrue(np.all(a[Scalar(1,True),Scalar(1,True)].mask == True))

        #
        # A partially masked 2-D Matrix indexed by a Pair
        #

        b = np.ma.arange(16.).reshape(2,2,2,2)
        a = Matrix(b.copy(), ((False,True),(False,False)))
        b[0,1,:,:] = np.ma.masked

        self.assertEqual(a[Pair((1,1))], b[1,1])
        self.assertEqual(a[Pair((1,1),True)], make_masked(b, [[1,1]])[1,1])

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
        self.assertTrue(np.all(a.values == (0,0,1,1,0,0,0,0,0,0)))
        self.assertTrue(not np.any(a.mask))

        a[Scalar((5,6,7),(True,False,True))] = 2
        self.assertTrue(np.all(a.values == (0,0,1,1,0,0,2,0,0,0)))
        self.assertTrue(not np.any(a.mask))

        a[Scalar(1)] = Scalar(3,True)
        self.assertTrue(np.all(a.values == (0,3,1,1,0,0,2,0,0,0)))
        self.assertTrue(np.all(a.mask   == (0,1,0,0,0,0,0,0,0,0)))

        a[Scalar(0,True)] = a[2] + 3
        self.assertTrue(np.all(a.values == (0,3,1,1,0,0,2,0,0,0)))
        self.assertTrue(np.all(a.mask   == (0,1,0,0,0,0,0,0,0,0)))

        a[Scalar(0,False)] = a[2] + 3
        self.assertTrue(np.all(a.values == (4,3,1,1,0,0,2,0,0,0)))
        self.assertTrue(np.all(a.mask   == (0,1,0,0,0,0,0,0,0,0)))

        a[Scalar((0,2,4))] = Scalar(4,True)
        self.assertTrue(np.all(a.values == (4,3,4,1,4,0,2,0,0,0)))
        self.assertTrue(np.all(a.mask   == (1,1,1,0,1,0,0,0,0,0)))

        a[Scalar((0,2,4))] = Scalar((5,6,7))
        self.assertTrue(np.all(a.values == (5,3,6,1,7,0,2,0,0,0)))
        self.assertTrue(np.all(a.mask   == (0,1,0,0,0,0,0,0,0,0)))

        a[Scalar((-1,-2,-3))] = a[Scalar((0,1,2))]
        self.assertTrue(np.all(a.values == (5,3,6,1,7,0,2,6,3,5)))
        self.assertTrue(np.all(a.mask   == (0,1,0,0,0,0,0,0,1,0)))

        a[Scalar((5,6,5),(True,False,False))] = Scalar((5,6,7))
        self.assertTrue(np.all(a.values == (5,3,6,1,7,7,6,6,3,5)))
        self.assertTrue(np.all(a.mask   == (0,1,0,0,0,0,0,0,1,0)))

        a[Scalar((5,6,5),(False,False,True))] = Scalar((5,6,7))
        self.assertTrue(np.all(a.values == (5,3,6,1,7,5,6,6,3,5)))
        self.assertTrue(np.all(a.mask   == (0,1,0,0,0,0,0,0,1,0)))

        a[:] = 9
        self.assertEqual(a, Scalar([9]*10))

        #
        # Assignment to a 2-D Scalar
        #

        a = Scalar(((0,0,0),(0,0,0)))
        a[Pair((1,2))] = 1
        self.assertEqual(a, Scalar([[0,0,0],[0,0,1]]))

        a[Pair((1,2),True)] = 2
        self.assertTrue(np.all(a.values == [[0,0,0],[0,0,1]]))
        self.assertTrue(not np.any(a.mask))

        a[Pair((1,2),False)] = 2
        self.assertTrue(np.all(a.values == [[0,0,0],[0,0,2]]))
        self.assertTrue(not np.any(a.mask))

        a[Pair((1,2))] = Scalar(0,True)
        self.assertTrue(np.all(a.values == [[0,0,0],[0,0,0]]))
        self.assertTrue(np.all(a.mask   == [[0,0,0],[0,0,1]]))

        a[Scalar(1,True)] = Scalar(1,True)
        self.assertTrue(np.all(a.values == [[0,0,0],[0,0,0]]))
        self.assertTrue(np.all(a.mask   == [[0,0,0],[0,0,1]]))

        a[Scalar(1,False)] = Scalar(1,False)
        self.assertTrue(np.all(a.values == [[0,0,0],[1,1,1]]))
        self.assertTrue(not np.any(a.mask))

        a[Scalar(1)] = Scalar(1,True)
        self.assertTrue(np.all(a.values == [[0,0,0],[1,1,1]]))
        self.assertTrue(np.all(a.mask   == [[0,0,0],[1,1,1]]))

        a[Scalar(1)] = Scalar(2)
        self.assertTrue(np.all(a.values == [[0,0,0],[2,2,2]]))
        self.assertTrue(not np.any(a.mask))

        a[Pair(((0,0),(0,1),(0,2)),True)] = 'abc'   # would raise an error if
                                                    # not for the mask
        self.assertTrue(np.all(a.values == [[0,0,0],[2,2,2]]))
        self.assertTrue(not np.any(a.mask))

        a[Pair(((0,0),(1,1)))] = 7
        self.assertTrue(np.all(a.values == [[7,0,0],[2,7,2]]))
        self.assertTrue(not np.any(a.mask))

        a[Pair(((0,0),(-1,-1)))] = 8
        self.assertTrue(np.all(a.values == [[8,0,0],[2,7,8]]))
        self.assertTrue(not np.any(a.mask))

        #
        # Assignment to a 2-D Matrix indexed 2-D
        #

        a = Matrix(np.zeros(16).reshape(2,2,2,2))

        a[Pair((1,1))] = Matrix([[1,2],[3,4]])
        self.assertEqual(a, Matrix([[[[0,0],[0,0]], [[0,0],[0,0]]],
                                    [[[0,0],[0,0]], [[1,2],[3,4]]]]))
        self.assertTrue(np.all(a.values == [[[[0,0],[0,0]], [[0,0],[0,0]]],
                                            [[[0,0],[0,0]], [[1,2],[3,4]]]]))
        self.assertTrue(np.all(a.mask   == False))

        a[Pair((1,1))] = Matrix([[4,5],[6,7]],True)
        self.assertTrue(np.all(a.values == [[[[0,0],[0,0]], [[0,0],[0,0]]],
                                            [[[0,0],[0,0]], [[4,5],[6,7]]]]))
        self.assertTrue(np.all(a.mask   == [[0,0],[0,1]]))

        a[Pair((1,1),True)] = Matrix([[5,5],[5,5]])
        self.assertTrue(np.all(a.values == [[[[0,0],[0,0]], [[0,0],[0,0]]],
                                            [[[0,0],[0,0]], [[4,5],[6,7]]]]))
        self.assertTrue(np.all(a.mask   == [[0,0],[0,1]]))

        a[Pair((1,1),False)] = Matrix([[5,5],[5,5]])
        self.assertTrue(np.all(a.values == [[[[0,0],[0,0]], [[0,0],[0,0]]],
                                            [[[0,0],[0,0]], [[5,5],[5,5]]]]))
        self.assertTrue(not np.any(a.mask))

        a[...,1] = Matrix([[5,6],[7,8]])
        self.assertEqual(a, Matrix([[[[0,0],[0,0]], [[5,6],[7,8]]],
                                    [[[0,0],[0,0]], [[5,6],[7,8]]]]))
        self.assertTrue(not np.any(a.mask))

        a[...,Scalar(1)] = Matrix([[1,2],[3,4]])
        self.assertEqual(a, Matrix([[[[0,0],[0,0]], [[1,2],[3,4]]],
                                    [[[0,0],[0,0]], [[1,2],[3,4]]]]))
        self.assertTrue(not np.any(a.mask))

        a[...,Scalar(1,True)] = Matrix([[8,8],[8,8]])
        self.assertEqual(a, Matrix([[[[0,0],[0,0]], [[1,2],[3,4]]],
                                    [[[0,0],[0,0]], [[1,2],[3,4]]]]))
        self.assertTrue(not np.any(a.mask))

        a[...,1] = Matrix([[8,8],[8,8]])
        self.assertTrue(np.all(a.values == [[[[0,0],[0,0]], [[8,8],[8,8]]],
                                            [[[0,0],[0,0]], [[8,8],[8,8]]]]))
        self.assertTrue(not np.any(a.mask))

        a[...,0] = Matrix([[9,9],[9,9]],True)
        self.assertTrue(np.all(a.values[:,1] == [[[8,8],[8,8]],
                                                  [[8,8],[8,8]]]))
        self.assertTrue(not np.any(a.mask[:,1]))
        self.assertTrue(np.all(a.mask[:,0]))

        a[Pair((0,0))] = Matrix([[5,5],[5,5]],False)
        a[Pair((-1,0))] = Matrix([[6,6],[6,6]],False)
        self.assertTrue(np.all(a.values == [[[[5,5],[5,5]], [[8,8],[8,8]]],
                                            [[[6,6],[6,6]], [[8,8],[8,8]]]]))
        self.assertTrue(not np.any(a.mask))

        a[Pair((1,0))] = Matrix([[7,7],[7,7]],True)
        self.assertTrue(np.all(a.values == [[[[5,5],[5,5]], [[8,8],[8,8]]],
                                            [[[7,7],[7,7]], [[8,8],[8,8]]]]))
        self.assertTrue(np.all(a.mask   == [[0,0],[1,0]]))

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
