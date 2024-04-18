################################################################################
# oops/utils.py
#
# Low-level operations on numpy arrays, mimicking SPICE routines but fully
# supporting broadcasted shapes.
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

    while len(a.shape) < len(b.shape):
        a = a[np.newaxis]
    while len(b.shape) < len(a.shape):
        b = b[np.newaxis]

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
