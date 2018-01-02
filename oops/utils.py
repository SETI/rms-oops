################################################################################
# oops/array_/utils.py
#
# Low-level operations on numpy arrays, mimicking SPICE routines but fully
# supporting broadcasted shapes.
#
# Mark Showalter, PDS Rings Node, SETI Institute, October 2011
################################################################################

import numpy as np

def dot(a,b):
    """dot(a,b) = dot product of 2/3-vectors a and b."""
    return np.sum(np.asarray(a)*np.asarray(b), axis=-1)

def norm(a):
    """norm(a) = length of a 2/3-vector."""
    return np.sqrt(np.sum(np.asarray(a)**2, axis=-1))

def unit(a):
    """unit(a) = a 2/3-vector re-scaled to unit length."""
    return a / norm(a)[..., np.newaxis]

def cross2d(a,b):
    """cross2d(a,b) = magnitude of the cross product of 2-vectors a and b."""
    a = np.asarray(a)
    b = np.asarray(b)
    return a[...,0]*b[...,1] - a[...,1]*b[...,0]

def cross3d(a,b):
    """cross3d(a,b) = cross product of 3-vectors a and b."""

    # It appears that np.cross only works properly if the arguments have the
    # same number of dimensions, ending in 3.

    a = np.asarray(a)
    b = np.asarray(b)

    while len(a.shape) < len(b.shape): a = a[np.newaxis]
    while len(b.shape) < len(a.shape): b = b[np.newaxis]

    return np.cross(a,b)

def ucross3d(a,b):
    """ucross3d(a,b) = cross product of 3-vectors a and b, scaled to unit
    length."""
    return unit(cross3d(a,b))

def proj(a,b):
    """proj(a,b) = 3-vector a projected onto 3-vector b."""
    b = np.asfarray(b)
    return b * dot(a,b)[..., np.newaxis] / dot(b,b)[..., np.newaxis]

def perp(a,b):
    """perp(a,b) = component of 3-vector a perpendicular to 3-vector b."""
    a = np.asfarray(a)
    return a - proj(a,b)

def sep(a,b):
    """sep(a,b) = angular separation between 2/3-vectors a and b."""

    # Algorithm translated directly from the SPICE routine
    signs = np.sign(dot(a,b))
    return (((1-signs)/2) * np.pi +
             2.*signs*np.arcsin(0.5 * norm(unit(a) -
                signs[..., np.newaxis] * unit(b))))

def xpose(m):
    """xpose(m) = transpose of matrix m."""
    return np.asfarray(m).swapaxes(-2,-1)

def mxv(m,v):
    """mxv(m,v)= matrix m times 3-vector v"""
    return np.sum(np.asfarray(m) * np.asfarray(v)[..., np.newaxis, :], axis=-1)

def mtxv(m,v):
    """mtxv(m,v) = transpose (inverse) of matrix m times 3-vector v."""
    return np.sum(np.asfarray(m) * np.asfarray(v)[..., np.newaxis], axis=-2)

# The standard numpy method for multiplying matrices uses the dot() function,
# but does not generalize to the case of multiplying arrays of 3x3 matrices,
# incorporating the rules of broadcasting.

# The matrix multiply functions below are written in a way that uses tricky
# indexing to avoid loops, which would slow down the operation substantially.
# The first case below is written out in explicit detail. The others are just
# variations that involve alternative index orders to transpose either or both
# of the matrices prior to the multiplication.

# Element ordering needed below for matrix multiplies
_ORDER1 = [0]*9 + [1]*9 + [2]*9
_ORDER2 = [0,1,2]*9
_ORDER3 = [0,0,0,1,1,1,2,2,2]*3

# _ORDER1 = [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2]
# _ORDER2 = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2]
# _ORDER3 = [0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2]

def mxm(m1,m2):
    """mxm(m1,m2) = matrix m1 times matrix m2."""

    # Duplicate the final 3x3 elements, reorder and pair-wise multiply

    prods = (np.asfarray(m1)[..., _ORDER1, _ORDER2] *
             np.asfarray(m2)[..., _ORDER2, _ORDER3])

    # Note that m1 and m2 need not have the same shape, as long as they
    # broadcast to the same shape. For purposes of this illustation, we neglect
    # any leading axes, so m1 and m2 are just 3x3 matrices.
    #
    # m1[_ORDER1, _ORDER2] = is a 1-D array with 27 elements:
    #   = array(m1[0,0], m1[0,1], m1[0,2],
    #           m1[0,0], m1[0,1], m1[0,2],
    #           m1[0,0], m1[0,1], m1[0,2],
    #           m1[1,0], m1[1,1], m1[1,2],
    #           m1[1,0], m1[1,1], m1[1,2],
    #           m1[1,0], m1[1,1], m1[1,2],
    #           m1[2,0], m1[2,1], m1[2,2],
    #           m1[2,0], m1[2,1], m1[2,2],
    #           m1[2,0], m1[2,1], m1[2,2])
    #
    # m2[_ORDER2, _ORDER3] = is a 1-D array with 27 elements:
    #   = array(m2[0,0], m2[1,0], m2[2,0],
    #           m2[0,1], m2[1,1], m2[2,1],
    #           m2[0,2], m2[1,2], m2[2,2],
    #           m2[0,0], m2[1,0], m2[2,0],
    #           m2[0,1], m2[1,1], m2[2,1],
    #           m2[0,2], m2[1,2], m2[2,2],
    #           m2[0,0], m2[1,0], m2[2,0],
    #           m2[0,1], m2[1,1], m2[2,1],
    #           m2[0,2], m2[1,2], m2[2,2])
    #
    # prods = the pairwise product of all 27 elements:
    #   = array(m1[0,0]*m2[0,0], m1[0,1]*m2[1,0], m1[0,2]*m2[2,0],
    #           m1[0,0]*m2[0,1], m1[0,1]*m2[1,1], m1[0,2]*m2[2,1],
    #           m1[0,0]*m2[0,2], m1[0,1]*m2[1,2], m1[0,2]*m2[2,2],
    #           m1[1,0]*m2[0,0], m1[1,1]*m2[1,0], m1[1,2]*m2[2,0],
    #           m1[1,0]*m2[0,1], m1[1,1]*m2[1,1], m1[1,2]*m2[2,1],
    #           m1[1,0]*m2[0,2], m1[1,1]*m2[1,2], m1[1,2]*m2[2,2],
    #           m1[2,0]*m2[0,0], m1[2,1]*m2[1,0], m1[2,2]*m2[2,0],
    #           m1[2,0]*m2[0,1], m1[2,1]*m2[1,1], m1[2,2]*m2[2,1],
    #           m1[2,0]*m2[0,2], m1[2,1]*m2[1,2], m1[2,2]*m2[2,2])

    # Reshape to (...,3,3,3)
    prods = prods.reshape(list(prods.shape)[0:-1]+[3,3,3])

    # prods is replaced by a 3x3x3 array
    # The old shape is (...,27)
    # The new shape is (...,3,3,3)

    # prods = the pairwise product of all 27 elements as a 3x3x3 array:
    #   = array((((m1[0,0]*m2[0,0], m1[0,1]*m2[1,0], m1[0,2]*m2[2,0]),
    #             (m1[0,0]*m2[0,1], m1[0,1]*m2[1,1], m1[0,2]*m2[2,1]),
    #             (m1[0,0]*m2[0,2], m1[0,1]*m2[1,2], m1[0,2]*m2[2,2])),
    #            ((m1[1,0]*m2[0,0], m1[1,1]*m2[1,0], m1[1,2]*m2[2,0]),
    #             (m1[1,0]*m2[0,1], m1[1,1]*m2[1,1], m1[1,2]*m2[2,1]),
    #             (m1[1,0]*m2[0,2], m1[1,1]*m2[1,2], m1[1,2]*m2[2,2])),
    #            ((m1[2,0]*m2[0,0], m1[2,1]*m2[1,0], m1[2,2]*m2[2,0]),
    #             (m1[2,0]*m2[0,1], m1[2,1]*m2[1,1], m1[2,2]*m2[2,1]),
    #             (m1[2,0]*m2[0,2], m1[2,1]*m2[1,2], m1[2,2]*m2[2,2]))))

    # Sum over the final axis and return

    #   = array(((m1[0,0]*m2[0,0] + m1[0,1]*m2[1,0] + m1[0,2]*m2[2,0]),
    #             m1[0,0]*m2[0,1] + m1[0,1]*m2[1,1] + m1[0,2]*m2[2,1]),
    #             m1[0,0]*m2[0,2] + m1[0,1]*m2[1,2] + m1[0,2]*m2[2,2])),
    #            (m1[1,0]*m2[0,0] + m1[1,1]*m2[1,0] + m1[1,2]*m2[2,0]),
    #             m1[1,0]*m2[0,1] + m1[1,1]*m2[1,1] + m1[1,2]*m2[2,1]),
    #             m1[1,0]*m2[0,2] + m1[1,1]*m2[1,2] + m1[1,2]*m2[2,2])),
    #            (m1[2,0]*m2[0,0] + m1[2,1]*m2[1,0] + m1[2,2]*m2[2,0]),
    #             m1[2,0]*m2[0,1] + m1[2,1]*m2[1,1] + m1[2,2]*m2[2,1]),
    #             m1[2,0]*m2[0,2] + m1[2,1]*m2[1,2] + m1[2,2]*m2[2,2]))))
    #
    # The above result is the matrix product sought.

    return np.sum(prods, axis=-1)

def mtxm(m1,m2):
    """mtxm(m1,m2) = transpose of matrix m1 times matrix m2."""

    # Duplicate the final 3x3 elements, reorder and pair-wise multiply
    prods = (np.asfarray(m1)[..., _ORDER2, _ORDER1] *
             np.asfarray(m2)[..., _ORDER2, _ORDER3])

    # Reshape and sum over the final axis
    prods = prods.reshape(list(prods.shape)[0:-1]+[3,3,3])
    return np.sum(prods, axis=-1)

def mxmt(m1,m2):
    """mxmt(m1,m2) = matrix m1 times transpose of matrix m2."""

    # Duplicate the final 3x3 elements, reorder and pair-wise multiply
    prods = (np.asfarray(m1)[..., _ORDER1, _ORDER2] *
             np.asfarray(m2)[..., _ORDER3, _ORDER2])

    # Reshape and sum over the final axis
    prods = prods.reshape(list(prods.shape)[0:-1]+[3,3,3])
    return np.sum(prods, axis=-1)

def mtxmt(m1,m2):
    """mtxmt(m1,m2) = transpose of matrix m1 times transpose of matrix m2."""

    # Duplicate the final 3x3 elements, reorder and pair-wise multiply
    prods = (np.asfarray(m1)[..., _ORDER2, _ORDER1] *
             np.asfarray(m2)[..., _ORDER3, _ORDER2])

    # Reshape and sum over the final axis
    prods = prods.reshape(list(prods.shape)[0:-1] + [3,3,3])
    return np.sum(prods, axis=-1)

def twovec(a,i,b,j):
    """twovec(a,i,b,j) = the transformation to the right-handed frame having
    a given vector as a specified axis and having a second given vector
    lying in a specified coordinate plane. Axes are indexed 0 to 2."""

    # Force arrays to appear to have the same shape; convert a to unit length
    (aa,bb) = np.broadcast_arrays(unit(a),b)

    # Create the buffer and fill in the first axis
    result = np.empty(list(aa.shape) + [3])
    result[...,i,:] = aa

    k = 3 - i - j
    if (3+j-i)%3 == 1:
        result[...,k,:] = ucross3d(aa,bb)
        result[...,j,:] = ucross3d(result[...,k,:],aa)
    else:
        result[...,k,:] = ucross3d(bb,aa)
        result[...,j,:] = ucross3d(aa,result[...,k,:])

    return result

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_utils(unittest.TestCase):

    def runTest(self):

        # dot
        self.assertEqual(dot((1,2),(3,4)), 11)
        self.assertEqual(dot((1,2,3),(3,4,5)), 26)
        self.assertEqual(dot((1.,2.),(3.,4.)), 11.)
        self.assertEqual(dot((1.,2.,3.),(3.,4.,5.)), 26.)
        self.assertTrue(np.all(dot([(1.,2.),(-1.,-2.)],(3.,4.)) == [11.,-11]))
        self.assertTrue(np.all(dot([(1.,2.,3.),(3.,2.,1.)],(3.,4.,5.))
                                   == (26.,22.)))

        # norm
        self.assertEqual(norm((3,4)), 5.)
        self.assertEqual(norm((3,4,12)), 13.)
        self.assertTrue(np.all(norm([(3,4),(5,12)]) == [5.,13.]))
        self.assertTrue(np.all(norm([(3,4,12),(5,12,84)]) == [13.,85.]))

        # unit, sep
        eps = 3.e-16
        lo = 1. - eps
        hi = 1. + eps
        self.assertTrue(norm(unit((3,4))) > lo)
        self.assertTrue(norm(unit((3,4))) < hi)

        test2 = [[(1,2),(3,4)],[(5,6),(7,8)]]
        self.assertTrue(np.all(norm(unit(test2)) > lo))
        self.assertTrue(np.all(norm(unit(test2)) < hi))
        self.assertTrue(np.all(norm(unit(test2)) > [lo,lo]))
        self.assertTrue(np.all(norm(unit(test2)) < [hi,hi]))
        self.assertTrue(np.all(norm(unit(test2)) > [[lo,lo],[lo,lo]]))
        self.assertTrue(np.all(norm(unit(test2)) < [[hi,hi],[hi,hi]]))

        self.assertTrue(np.all(sep(test2,unit(test2)) <  eps))
        self.assertTrue(np.all(sep(test2,unit(test2)) > -eps))
        self.assertTrue(np.all(sep(test2,unit(test2)) < [ eps, eps]))
        self.assertTrue(np.all(sep(test2,unit(test2)) > [-eps,-eps]))

        self.assertTrue(norm(unit((3,4,5))) > lo)
        self.assertTrue(norm(unit((3,4,5))) < hi)

        test3 = [[(1,2,-3),(3,4,-5)],[(5,6,-7),(7,8,-9)]]
        self.assertTrue(np.all(norm(unit(test3)) > lo))
        self.assertTrue(np.all(norm(unit(test3)) < hi))
        self.assertTrue(np.all(norm(unit(test3)) > [lo,lo]))
        self.assertTrue(np.all(norm(unit(test3)) < [hi,hi]))
        self.assertTrue(np.all(norm(unit(test3)) > [[lo,lo],[lo,lo]]))
        self.assertTrue(np.all(norm(unit(test3)) < [[hi,hi],[hi,hi]]))

        self.assertTrue(np.all(sep(test3,unit(test3)) <  eps))
        self.assertTrue(np.all(sep(test3,unit(test3)) > -eps))
        self.assertTrue(np.all(sep(test3,unit(test3)) < [ eps, eps]))
        self.assertTrue(np.all(sep(test3,unit(test3)) > [-eps,-eps]))

        # cross2d, sep
        self.assertEqual(cross2d((1,0),(0,1)), 1.)
        self.assertEqual(cross2d((1,0),(1,1)), 1.)
        self.assertEqual(cross2d((1,0),(111,1)), 1.)
        self.assertEqual(cross2d((0,1),(111,1)), -111.)

        dirs = np.asfarray([[[( 5, 0),( 4, 3),( 3, 4)],
                             [( 0, 5),(-3, 4),(-4, 3)]],
                            [[(-5, 0),(-4,-3),(-3,-4)],
                             [( 0,-5),( 3,-4),( 4,-3)]]])
        self.assertTrue(np.all(cross2d(dirs,(1,0)) == -dirs[...,1]))
        self.assertTrue(np.all(cross2d(dirs,(0,1)) ==  dirs[...,0]))

        # cross3d
        self.assertTrue(np.all(cross3d((1,0,0),(0,1,0)) == (0, 0,1)))
        self.assertTrue(np.all(cross3d((1,0,0),(0,0,1)) == (0,-1,0)))

        self.assertTrue(np.all(cross3d([(1 ,0,0),(0,2,0)],(0,0,1)) ==
                                       [(0,-1,0),(2,0,0)]))

        # ucross3d, sep, norm
        eps = 1.e-15
        vec1 = [(7,-1,1),(1,2,-3),(-1,3,3)]
        vec2 = (3,1,-3)

        test = ucross3d(vec1, vec2)
        self.assertTrue(np.all(norm(test) > 1. - eps))
        self.assertTrue(np.all(norm(test) < 1. + eps))
        self.assertTrue(np.all(sep(test,vec2) > np.pi/2. - eps))
        self.assertTrue(np.all(sep(test,vec2) < np.pi/2. + eps))

        vec2 = [(3,2,1),(-4,-1,0),(7,6,5)]

        test = ucross3d(vec1, vec2)
        self.assertTrue(np.all(norm(test) > 1. - eps))
        self.assertTrue(np.all(norm(test) < 1. + eps))
        self.assertTrue(np.all(sep(test,vec2) > np.pi/2. - eps))
        self.assertTrue(np.all(sep(test,vec2) < np.pi/2. + eps))

        # proj, perp, sep, norm
        eps = 3.e-15
        perps = perp(vec1, vec2)
        projs = proj(vec1, vec2)

        self.assertTrue(np.all(dot(perps,vec2) > -eps))
        self.assertTrue(np.all(dot(perps,vec2) <  eps))

        self.assertTrue(np.all(sep(perps,vec2) > np.pi/2. - eps))
        self.assertTrue(np.all(sep(perps,vec2) < np.pi/2. + eps))

        self.assertTrue(np.all(sep(projs,vec2) % np.pi > -eps))
        self.assertTrue(np.all(sep(projs,vec2) % np.pi <  eps))

        test = vec1 - (projs + perps)
        self.assertTrue(np.all(test > -eps))
        self.assertTrue(np.all(test <  eps))

        vec2 = [(3,2,1),(-4,-1,0),(7,6,5)]

        self.assertTrue(np.all(dot(perps,vec2) > -eps))
        self.assertTrue(np.all(dot(perps,vec2) <  eps))

        self.assertTrue(np.all(sep(perps,vec2) > np.pi/2. - eps))
        self.assertTrue(np.all(sep(perps,vec2) < np.pi/2. + eps))

        self.assertTrue(np.all(sep(projs,vec2) % np.pi > -eps))
        self.assertTrue(np.all(sep(projs,vec2) % np.pi <  eps))

        test = vec1 - (projs + perps)
        self.assertTrue(np.all(test > -eps))
        self.assertTrue(np.all(test <  eps))

        # xpose
        mat = [[[1,2,3],[4,5,6],[7,8,9]]] * 7
        self.assertEqual(np.shape(mat),(7,3,3))
        self.assertEqual(np.shape(xpose(mat)),(7,3,3))
        self.assertTrue(np.all(np.array(mat)[...,0,1] == xpose(mat)[...,1,0]))

        # twovec, mxv, mtxv, twovec
        eps = 1.e-14

        mat1 = twovec((1,0,0),0,(0,1,0),1)
        mat2 = twovec((1,0,0),0,(0,0,4),2)
        self.assertTrue(np.all(mat1 == mat2))
        self.assertTrue(np.all(mat1 == [[1,0,0,],[0,1,0],[0,0,1]]))

        self.assertTrue(np.all(mxv( mat1,vec1) == vec1))
        self.assertTrue(np.all(mtxv(mat1,vec1) == vec1))
        self.assertTrue(np.all(mxv( mat1,vec1[0]) == vec1[0]))
        self.assertTrue(np.all(mtxv(mat1,vec1[0]) == vec1[0]))

        # Rotate vectors along the axes into the frame
        mat = twovec((1,1,1),2,[(1,0,-1),(-1,0,1)],0)
        vec = (3,3,3)

        self.assertTrue(np.all(mxv(mat,vec)[...,0:2] > -eps))
        self.assertTrue(np.all(mxv(mat,vec)[...,0:2] <  eps))
        self.assertTrue(np.all(mxv(mat,vec)[...,2] > np.sqrt(27) - eps))
        self.assertTrue(np.all(mxv(mat,vec)[...,2] < np.sqrt(27) + eps))

        vec = [(2,0,-2),[-2,0,2]]
        result = self.assertTrue(np.all(mxv(mat,vec)[:,1:3] > -eps))
        result = self.assertTrue(np.all(mxv(mat,vec)[:,1:3] <  eps))
        result = self.assertTrue(np.all(mxv(mat,vec)[:,0] > np.sqrt(8) - eps))
        result = self.assertTrue(np.all(mxv(mat,vec)[:,0] < np.sqrt(8) + eps))

        # Rotate axis vectors out of the frame
        vec = [[(1,0,0),[0,1,0]],[(2,3,4),[0,0,2]]]
        result = mtxv(mat,vec)

        self.assertEqual(result[1,1,0], result[1,1,1])
        self.assertEqual(result[1,1,0], result[1,1,2])
        self.assertEqual(result[0,0,0],-result[0,0,2])
        self.assertEqual(result[0,0,1], 0.)

        result = mxv(xpose(mat),vec)
        self.assertEqual(result[1,1,0], result[1,1,1])
        self.assertEqual(result[1,1,0], result[1,1,2])
        self.assertEqual(result[0,0,0],-result[0,0,2])
        self.assertEqual(result[0,0,1], 0.)

        mat = [[1,2,3],[4,5,6],[7,8,9]]
        vec = [1,0,0]
        self.assertTrue(np.all(mxv(mat,vec)  - [1,4,7]) == 0.)
        self.assertTrue(np.all(mtxv(mat,vec) - [1,2,3]) == 0.)
        vec = [0,1,0]
        self.assertTrue(np.all(mxv(mat,vec)  - [2,5,8]) == 0.)
        self.assertTrue(np.all(mtxv(mat,vec) - [4,5,6]) == 0.)

        # mxv, mtxv, mxm, mtxm, mxmt, mtxmt, with shape broadcasting
        a = np.random.rand(2,1,4,3,3)
        b = np.random.rand(  3,4,3,3)
        v = np.random.rand(1,3,1,3,1)

        axb   = mxm(a,b)
        atxb  = mtxm(a,b)
        axbt  = mxmt(a,b)
        atxbt = mtxmt(a,b)

        axv  = mxv(a,v[...,0])
        atxv = mtxv(a,v[...,0])

        self.assertEqual(axb.shape, (2,3,4,3,3))
        self.assertEqual(axv.shape, (2,3,4,3))

        eps = 1.e-15

        for i in range(2):
          for j in range(3):
            for k in range(4):
                am = np.matrix(a[i,0,k])
                bm = np.matrix(b[  j,k])
                amt = np.matrix(a[i,0,k].T)
                bmt = np.matrix(b[  j,k].T)
                vm  = np.matrix(v[0,j,0])

                test = am * bm
                self.assertTrue(np.all(test == np.matrix(axb[i,j,k])))

                test = amt * bm
                self.assertTrue(np.all(test == np.matrix(atxb[i,j,k])))

                test = am * bmt
                self.assertTrue(np.all(test == np.matrix(axbt[i,j,k])))

                test = amt * bmt
                self.assertTrue(np.all(test == np.matrix(atxbt[i,j,k])))

                test = am * vm
                self.assertTrue(np.all(test > np.matrix(axv[i,j,k,:,
                                                        np.newaxis]) - eps))
                self.assertTrue(np.all(test < np.matrix(axv[i,j,k,:,
                                                        np.newaxis]) + eps))

                test = amt * vm
                self.assertTrue(np.all(test > np.matrix(atxv[i,j,k,:,
                                                        np.newaxis]) - eps))
                self.assertTrue(np.all(test < np.matrix(atxv[i,j,k,:,
                                                        np.newaxis]) + eps))

########################################
if __name__ == "__main__":
    unittest.main(verbosity=2)
################################################################################
