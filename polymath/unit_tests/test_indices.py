################################################################################
# General index tests - masked and non-masked
################################################################################

from __future__ import division
import warnings
import numpy as np
import unittest

from polymath import Scalar, Pair, Vector, Matrix, Boolean, Qube

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
                warnings.simplefilter('ignore')
                result = np.ma.array(ret)

            return result

        def compare_a_b_1d(a, b, class_):
            """Input a is a Qube subclass made from MaskedArray b, at least 1-D."""

            # Traditional indexing
            self.assertEqual(a, b, class_)
            self.assertEqual(a[1], b[1])
            self.assertEqual(a[-1], b[-1])
            self.assertEqual(a[1:5], b[1:5])
            self.assertEqual(a[1:5:2], b[1:5:2])
            self.assertEqual(a[-5:], b[-5:])
            self.assertEqual(a[:], b[:])
            self.assertEqual(a[...], b[...])
            self.assertEqual(a[...,:], b[...,:])
            self.assertEqual(a[::-1], b[::-1])

            # Single Scalar
            self.assertEqual(a[Scalar(1)], b[1])
            self.assertEqual(a[Scalar(1,True)], make_masked(b, [1])[1])

            # Two elements
            self.assertEqual(a[Scalar((1,3))], b[1:4:2])
            self.assertEqual(a[Scalar((1,3),(True,False))],
                             make_masked(b, [1])[1:4:2])
            self.assertEqual(a[Scalar((1,3),(True,False))],
                             Qube.stack(a[3].as_all_masked(), a[3]))
            self.assertEqual(a[Scalar((1,3),True)],
                             make_masked(b, [1,3])[1:4:2])
            self.assertEqual(a[Scalar((1,3),True)],
                             class_.zeros((), denom=a.denom, numer=a.numer, mask=True))
            self.assertEqual(a[Scalar((1,3),True)].shape, (2,) + a.shape[1:])

            # Boolean
            self.assertEqual(a[True], b)
            self.assertEqual(a[True].shape, a.shape)
            self.assertEqual(a[False].shape, (0,) + a.shape[1:])
            self.assertEqual(a[Boolean(True)], b)
            self.assertEqual(a[Boolean(True)].shape, a.shape)
            self.assertEqual(a[Boolean(False)].shape, (0,) + a.shape[1:])
            self.assertEqual(a[Boolean.MASKED].shape, (1,) + a.shape[1:])
            self.assertEqual(a[Boolean.MASKED].mask, True)

        def compare_a_b_2d(a, b, class_):
            """Input a is a Qube subclass made from MaskedArray b, at least 2-D."""

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

        def compare_a_b_3d(a, b, class_):
            """Input a is a Qube subclass made from MaskedArray b, at least 3-D.
            """

            # Indexed by 3-D Vector
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
            self.assertEqual(a[Vector(((1,1,1),(2,2,2),(3,3,3)),(True,0,0))],
                             make_masked(extract(b, ((1,1,1),(2,2,2),(3,3,3))),
                                         [0]))
            self.assertEqual(a[Vector(((1,1,1),(2,2,2),(3,3,3)),(0,True,0))],
                             make_masked(extract(b, ((1,1,1),(2,2,2),(3,3,3))),
                                         [1]))
            self.assertEqual(a[Vector(((1,1,1),(2,2,2),(3,3,3)),(0,0,True))],
                             make_masked(extract(b, ((1,1,1),(2,2,2),(3,3,3))),
                                         [2]))

            # Indexed by mixed types
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

        def check_derivs_1d(c):
            """Alternative ways of indexing a 1-D derivative."""

            self.assertEqual(c[1].d_dt, c.d_dt[1])
            self.assertEqual(c[-1].d_dt, c.d_dt.vals[-1])
            self.assertEqual(c[1:5].d_dt, c.d_dt[1:5])
            self.assertEqual(c[1:5:2].d_dt, c.d_dt[1:5:2])
            self.assertEqual(c[-5:].d_dt, c.d_dt[-5:])
            self.assertEqual(c[:].d_dt, c.d_dt)
            self.assertEqual(c[...].d_dt, c.d_dt)
            self.assertEqual(c[::-1].d_dt, c.d_dt[::-1])

            self.assertEqual(c[1].d_dxy, c.d_dxy[1])
            self.assertEqual(c[-1].d_dxy, c.d_dxy[-1])
            self.assertEqual(c[1:3].d_dxy, c.d_dxy[1:3])
            self.assertEqual(c[1:4:2].d_dxy, c.d_dxy[1:4:2])
            self.assertEqual(c[-3:].d_dxy, c.d_dxy[-3:])
            self.assertEqual(c.d_dxy[:], c.d_dxy)
            self.assertEqual(c[:].d_dxy, c.d_dxy)
            self.assertEqual(c[...].d_dxy, c.d_dxy)
            self.assertEqual(c[::-1].d_dxy, c.d_dxy[::-1])

        def check_derivs_2d(c, ellipses=True):
            """Alternative ways of indexing a 2-D derivative."""

            self.assertEqual(c[1,0].d_dt, c.d_dt[1,0])
            self.assertEqual(c[-1,0].d_dt, c.d_dt.vals[-1,0])
            self.assertEqual(c[1:5,3].d_dt, c.d_dt[1:5,3])
            self.assertEqual(c[:-1,1:5:2].d_dt, c.d_dt[:-1,1:5:2])
            self.assertEqual(c[-1,-5:].d_dt, c.d_dt[-1,-5:])
            self.assertEqual(c[:,0].d_dt, c[:,0].d_dt)
            self.assertEqual(c[:,0:].d_dt, c[:,0:].d_dt)
            self.assertEqual(c[:,-1].d_dt, c[:,-1].d_dt)
            self.assertEqual(c[:,-1:].d_dt, c[:,-1:].d_dt)
            self.assertEqual(c[::-1,:2].d_dt, c.d_dt[::-1,:2])
            if ellipses:
                self.assertEqual(c[...,2].d_dt, c[...,2].d_dt)
                self.assertEqual(c[-2,...].d_dt, c[-2,...].d_dt)
                self.assertEqual(c[:-2,...,1].d_dt, c[:-2,...,1].d_dt)

            self.assertEqual(c[Scalar(1),0].d_dt, c.d_dt[1,0])
            self.assertEqual(c[Scalar(-1),0].d_dt, c.d_dt.vals[-1,0])
            self.assertEqual(c[1:5,Scalar((3,4))].d_dt, c.d_dt[1:5,3:5])
            self.assertEqual(c[-1,-5:].d_dt, c.d_dt[Scalar(-1),-5:])
            if ellipses:
                self.assertEqual(c[...,Scalar(2)].d_dt, c[...,2].d_dt)
                self.assertEqual(c[Scalar(-2),...].d_dt, c[-2,...].d_dt)
                self.assertEqual(c[:-2,...,Scalar(1)].d_dt, c[:-2,...,1].d_dt)
                self.assertEqual(c[Scalar(0),...,Scalar(-1)].d_dt, c.d_dt[0,-1])
                self.assertEqual(c[Scalar((1,0)),...,Scalar(-1)].d_dt, c.d_dt[Pair(((1,-1),(0,-1)))])

            self.assertEqual(c[Pair((1,0))].d_dt, c.d_dt[1,0])
            self.assertEqual(c[Pair((-1,0))].d_dt, c.d_dt.vals[-1,0])
            self.assertEqual(c[Pair([(1,3),(2,3),(3,3),(4,3)])].d_dt, c.d_dt[1:5,3])

        # An unmasked Scalar
        b = np.ma.arange(10)
        a = Scalar(b.data, False)
        c = a.copy()
        c.insert_deriv('t', Scalar([5,4,3,2,1,0,9,8,7,6]))
        c.insert_deriv('xy', Scalar(-2*np.arange(20.).reshape(10,2), drank=1))
        compare_a_b_1d(a, b, Scalar)
        check_derivs_1d(c)

        # A fully masked Scalar
        b = np.ma.arange(10)
        b[:] = np.ma.masked
        a = Scalar(b, True)
        c = a.copy()
        c.insert_deriv('t', Scalar([5,4,3,2,1,0,9,8,7,6]))
        c.insert_deriv('xy', Scalar(-2*np.arange(20.).reshape(10,2), drank=1))
        compare_a_b_1d(a, b, Scalar)
        check_derivs_1d(c)

        # A partially masked Scalar
        b = np.ma.arange(10)
        b[3] = np.ma.masked
        a = Scalar(b)
        c = a.copy()
        c.insert_deriv('t', Scalar([5,4,3,2,1,0,9,8,7,6],
                                   mask=[0,0,0,1,0,0,0,0,0,0]))
        c.insert_deriv('xy', Scalar(-2*np.arange(20.).reshape(10,2), drank=1,
                                    mask=[0,0,0,1,0,0,0,0,0,0]))
        compare_a_b_1d(a, b, Scalar)
        check_derivs_1d(c)

        # An unmasked 2-D Scalar
        b = np.ma.arange(25).reshape(5,5)
        a = Scalar(b, False)
        c = a.copy()
        c.insert_deriv('t', Scalar(np.random.randn(5,5)))
        c.insert_deriv('xy', Scalar(np.random.randn(5,5,2), drank=1))
        compare_a_b_1d(a, b, Scalar)
        check_derivs_1d(c)

        # An unmasked 2-D Scalar indexed by a Pair
        b = np.ma.arange(25).reshape(5,5)
        a = Scalar(b)
        c = a.copy()
        c.insert_deriv('t', Scalar(np.random.randn(5,5)))
        c.insert_deriv('xy', Scalar(np.random.randn(5,5,2), drank=1))
        compare_a_b_1d(a, b, Scalar)
        compare_a_b_2d(a, b, Scalar)
        check_derivs_1d(c)
        check_derivs_2d(c)

        # A partially masked 2-D Scalar indexed by a Pair
        b = np.ma.arange(25).reshape(5,5)
        b[1,1] = np.ma.masked
        a = Scalar(b)
        c = a.copy()
        c.insert_deriv('t', Scalar(np.random.randn(5,5)))
        c.insert_deriv('xy', Scalar(np.random.randn(5,5,2), drank=1))
        compare_a_b_1d(a, b, Scalar)
        compare_a_b_2d(a, b, Scalar)
        compare_a_b_1d(a, b, Scalar)
        compare_a_b_2d(a, b, Scalar)
        check_derivs_1d(c)
        check_derivs_2d(c)

        # An unmasked 3-D Scalar
        b = np.ma.arange(125).reshape(5,5,5)
        a = Scalar(b)
        c = a.copy()
        c.insert_deriv('t', Scalar(np.random.randn(5,5,5)))
        c.insert_deriv('xy', Scalar(np.random.randn(5,5,5,2), drank=1))
        compare_a_b_1d(a, b, Scalar)
        compare_a_b_2d(a, b, Scalar)
        check_derivs_1d(c)
        check_derivs_2d(c, ellipses=False)

        # An unmasked 3-D Scalar
        b = np.ma.arange(72).reshape(6,6,2)
        a = Scalar(b)
        c = a.copy()
        c.insert_deriv('t', Scalar(np.random.randn(6,6,2)))
        c.insert_deriv('xy', Scalar(np.random.randn(6,6,2,2), drank=1))
        compare_a_b_1d(a, b, Scalar)
        compare_a_b_2d(a, b, Scalar)
        check_derivs_1d(c)
        check_derivs_2d(c, ellipses=False)

        # A partially masked 3-D Scalar
        b = np.ma.arange(75).reshape(5,5,3)
        b[1,1,1] = np.ma.masked
        a = Scalar(b)
        c = a.copy()
        c.insert_deriv('t', Scalar(np.random.randn(5,5,3)))
        c.insert_deriv('xy', Scalar(np.random.randn(5,5,3,2), drank=1))
        compare_a_b_1d(a, b, Scalar)
        compare_a_b_2d(a, b, Scalar)
        check_derivs_1d(c)
        check_derivs_2d(c, ellipses=False)

        # An unmasked 3-D Scalar
        b = np.ma.arange(125).reshape(5,5,5)
        a = Scalar(b)
        c = a.copy()
        c.insert_deriv('t', Scalar(np.random.randn(5,5,5)))
        c.insert_deriv('xy', Scalar(np.random.randn(5,5,5,2), drank=1))
        compare_a_b_1d(a, b, Scalar)
        compare_a_b_2d(a, b, Scalar)
        compare_a_b_3d(a, b, Scalar)
        check_derivs_1d(c)
        check_derivs_2d(c, ellipses=False)

        # An unmasked 1-D Matrix
        b = np.ma.arange(20).reshape(5,2,2)
        a = Matrix(b)
        c = a.copy()
        c.insert_deriv('t', Matrix(np.random.randn(5,2,2)))
        c.insert_deriv('xy', Matrix(np.random.randn(5,2,2,2), drank=1))
        compare_a_b_1d(a, b, Matrix)
        check_derivs_1d(c)

        # An unmasked 2-D Matrix
        b = np.ma.arange(100).reshape(5,5,2,2)
        a = Matrix(b)
        c = a.copy()
        c.insert_deriv('t', Matrix(np.random.randn(5,5,2,2)))
        c.insert_deriv('xy', Matrix(np.random.randn(5,5,2,2,2), drank=1))
        compare_a_b_1d(a, b, Matrix)
        compare_a_b_2d(a, b, Matrix)
        check_derivs_1d(c)
        check_derivs_2d(c)

        # Boolean mask
        a = Pair(np.arange(6).reshape((3,2)), mask=[False, False, True])
        self.assertEqual(a[2], Pair.MASKED)
        self.assertEqual(a[np.array([True,False,True])], [Pair((0,1)),Pair.MASKED])
        self.assertEqual(a[Boolean([True,False,True])] , [Pair((0,1)),Pair.MASKED])

        a = a.insert_deriv('t', Pair(-np.arange(6).reshape((3,2)), mask=a.mask))
        self.assertEqual(a[2].d_dt, Pair.MASKED)
        self.assertEqual(a[np.array([True,False,True])].d_dt, [Pair((0,-1)),Pair.MASKED])
        self.assertEqual(a[Boolean([True,False,True])].d_dt , [Pair((0,-1)),Pair.MASKED])

        # Indexing a shapeless object
        a = Scalar(0.)
        self.assertEqual(a[True], a)
        self.assertEqual(a[..., True], a)
        self.assertEqual(a[..., True].shape, ())
        self.assertEqual(a[..., True, None, None], a)
        self.assertEqual(a[..., True, None, None].shape, (1,1))
        self.assertEqual(a[None, ..., True, None], a)
        self.assertEqual(a[None, ..., None, True].shape, (1,1))
        self.assertEqual(a[None, ..., None], a)
        self.assertEqual(a[None, ..., None].shape, (1,1))

        self.assertEqual(a[False].shape, (0,))
        self.assertEqual(a[..., False].shape, (0,))
        self.assertEqual(a[..., False, None, None].shape, (0,1,1))
        self.assertEqual(a[None, ..., False, None].shape, (1,0,1))

        BM = Boolean.MASKED
        self.assertEqual(a[BM], Scalar.MASKED)
        self.assertEqual(a[BM].shape, ())
        self.assertEqual(a[..., BM], Scalar.MASKED)
        self.assertEqual(a[..., BM].shape, ())
        self.assertEqual(a[..., BM, None, None], Scalar.MASKED)
        self.assertEqual(a[..., BM, None, None].shape, (1,1))
        self.assertEqual(a[None, ..., BM, None], Scalar.MASKED)
        self.assertEqual(a[None, ..., BM, None].shape, (1,1))

        a.insert_deriv('xy', Scalar((1.,2.), drank=1))
        self.assertEqual(a[True].d_dxy, a.d_dxy)
        self.assertEqual(a[True].d_dxy.shape, ())
        self.assertEqual(a[..., True].d_dxy, a.d_dxy)
        self.assertEqual(a[..., True].d_dxy.shape, ())
        self.assertEqual(a[..., True, None, None].d_dxy, a.d_dxy)
        self.assertEqual(a[..., True, None, None].d_dxy.shape, (1,1))
        self.assertEqual(a[None, ..., True, None].d_dxy, a.d_dxy)
        self.assertEqual(a[None, ..., None, True].d_dxy.shape, (1,1))
        self.assertEqual(a[None, ..., None].d_dxy, a.d_dxy)
        self.assertEqual(a[None, ..., None].d_dxy.shape, (1,1))

        self.assertEqual(a[False].d_dxy.shape, (0,))
        self.assertEqual(a[..., False].d_dxy.shape, (0,))
        self.assertEqual(a[..., False, None, None].d_dxy.shape, (0,1,1))
        self.assertEqual(a[None, ..., False, None].d_dxy.shape, (1,0,1))

        dxy_masked = Scalar((0.,0.), drank=1, mask=True)
        self.assertEqual(a[BM].d_dxy, dxy_masked)
        self.assertEqual(a[BM].d_dxy.shape, ())
        self.assertEqual(a[..., BM].d_dxy, dxy_masked)
        self.assertEqual(a[..., BM].d_dxy.shape, ())
        self.assertEqual(a[..., BM, None, None].d_dxy, dxy_masked)
        self.assertEqual(a[..., BM, None, None].d_dxy.shape, (1,1))
        self.assertEqual(a[None, ..., BM, None].d_dxy, dxy_masked)
        self.assertEqual(a[None, ..., BM, None].d_dxy.shape, (1,1))

        self.assertRaises(IndexError, a.__getitem__, (Ellipsis, None, Ellipsis))
        self.assertRaises(IndexError, a.__getitem__, (True, False))
        self.assertRaises(IndexError, a.__getitem__, (True, True))

        ######## __setitem__

        # Assignment to a 0-D Scalar with boolean indexing
        a = Scalar(1.)
        self.assertEqual(a, 1)

        a[True] = 7
        self.assertEqual(a, 7)

        a[False] = -7
        self.assertEqual(a, 7)

        a[Boolean(True)] = 4
        self.assertEqual(a, 4)

        a[Boolean(False)] = -7
        self.assertEqual(a, 4)

        a[Boolean.MASKED] = -7
        self.assertEqual(a, 4)

        # Assignment to a 1-D Scalar with boolean indexing
        a = Scalar(np.arange(3))
        a[True] = np.arange(4,7)
        self.assertEqual(a, np.arange(4,7))
        a[..., True] = np.arange(3)
        self.assertEqual(a, np.arange(3))
        a[None, None, ..., True] = np.arange(4,7)
        self.assertEqual(a, np.arange(4,7))
        a[None, ..., True, None] = np.arange(3).reshape(3,1)
        self.assertEqual(a, np.arange(3))

        a = Scalar(np.arange(4,7))
        a[False] = np.arange(3)
        self.assertEqual(a, np.arange(4,7))
        a[..., False] = np.arange(3)
        self.assertEqual(a, np.arange(4,7))
        a[None, ..., False, None] = np.arange(3).reshape(3,1)
        self.assertEqual(a, np.arange(4,7))
        a[Boolean(True)] = np.arange(8,11)
        self.assertEqual(a, np.arange(8,11))
        a[Boolean(False)] = np.arange(3)
        self.assertEqual(a, np.arange(8,11))
        a[Boolean.MASKED] = np.arange(3)
        self.assertEqual(a, np.arange(8,11))

        a[np.array([True, True, False])] = 7
        self.assertEqual(a, [7,7,10])
        a[Boolean([False, True, True])] = -7
        self.assertEqual(a, [7,-7,-7])
        a[Boolean([False, True, True], mask=(0,0,1))] = 3
        self.assertEqual(a, [7,3,-7])

        self.assertEqual(a.derivs, {})
        five = Scalar(5, derivs={'t': Scalar(-5)})
        a[Boolean([False, False, True], mask=(0,0,1))] = five
        self.assertEqual(a.derivs, {})
        a[Boolean([False, True, True], mask=(0,0,1))] = five
        self.assertEqual(a.derivs, {'t': Scalar([0,-5,0])})

        # Assignment to a 1-D Scalar
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

        # Assignment to a 2-D Scalar
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

        # Assignment to a 2-D Matrix indexed 2-D
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

        # Assignment to a shapeless object
        a = Scalar(0.)
        a[False] = 7
        self.assertEqual(a, 0.)
        self.assertTrue(a.is_float())

        a[True] = 7
        self.assertEqual(a, 7.)
        self.assertTrue(a.is_float())

        a[..., np.newaxis, False] = 3
        self.assertEqual(a, 7.)
        self.assertTrue(a.is_float())

        a[..., np.newaxis, True] = 3
        self.assertEqual(a, 3.)
        self.assertTrue(a.is_float())

        a = Scalar(0.)
        a.insert_deriv('xy', Scalar((2,3), drank=1))

        a[False] = 7
        self.assertEqual(a.d_dxy, Scalar((2,3), drank=1))

        a[..., True] = 7
        self.assertEqual(a.d_dxy, Scalar((0,0), drank=1))

        a = Scalar(0.)
        a.insert_deriv('xy', Scalar((2,3), drank=1))

        b = Scalar(7.)
        b.insert_deriv('ab', Scalar((4,3), drank=1))

        a[None, ..., False] = b
        self.assertEqual(a.d_dxy, Scalar((2,3), drank=1))
        self.assertFalse('ab' in a.derivs)

        a[None, ..., True] = b
        self.assertEqual(a.d_dxy, Scalar((0,0), drank=1))
        self.assertEqual(a.d_dab, Scalar((4,3), drank=1))

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
