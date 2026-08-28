################################################################################
# tests/test_utils.py
################################################################################

import numpy as np

from oops.utils import *


def test_utils():
    np.random.seed(6167)

    # dot
    assert dot((1,2),(3,4)) == 11
    assert dot((1,2,3),(3,4,5)) == 26
    assert dot((1.,2.),(3.,4.)) == 11.
    assert dot((1.,2.,3.),(3.,4.,5.)) == 26.
    assert np.all(dot([(1.,2.),(-1.,-2.)],(3.,4.)) == [11.,-11])
    assert np.all(dot([(1.,2.,3.),(3.,2.,1.)],(3.,4.,5.)) == (26.,22.))

    # norm
    assert norm((3,4)) == 5.
    assert norm((3,4,12)) == 13.
    assert np.all(norm([(3,4),(5,12)]) == [5.,13.])
    assert np.all(norm([(3,4,12),(5,12,84)]) == [13.,85.])

    # unit, sep
    eps = 3.e-16
    lo = 1. - eps
    hi = 1. + eps
    assert norm(unit((3,4))) > lo
    assert norm(unit((3,4))) < hi

    test2 = [[(1,2),(3,4)],[(5,6),(7,8)]]
    assert np.all(norm(unit(test2)) > lo)
    assert np.all(norm(unit(test2)) < hi)
    assert np.all(norm(unit(test2)) > [lo,lo])
    assert np.all(norm(unit(test2)) < [hi,hi])
    assert np.all(norm(unit(test2)) > [[lo,lo],[lo,lo]])
    assert np.all(norm(unit(test2)) < [[hi,hi],[hi,hi]])

    assert np.all(sep(test2,unit(test2)) <  eps)
    assert np.all(sep(test2,unit(test2)) > -eps)
    assert np.all(sep(test2,unit(test2)) < [ eps, eps])
    assert np.all(sep(test2,unit(test2)) > [-eps,-eps])

    assert norm(unit((3,4,5))) > lo
    assert norm(unit((3,4,5))) < hi

    test3 = [[(1,2,-3),(3,4,-5)],[(5,6,-7),(7,8,-9)]]
    assert np.all(norm(unit(test3)) > lo)
    assert np.all(norm(unit(test3)) < hi)
    assert np.all(norm(unit(test3)) > [lo,lo])
    assert np.all(norm(unit(test3)) < [hi,hi])
    assert np.all(norm(unit(test3)) > [[lo,lo],[lo,lo]])
    assert np.all(norm(unit(test3)) < [[hi,hi],[hi,hi]])

    assert np.all(sep(test3,unit(test3)) <  eps)
    assert np.all(sep(test3,unit(test3)) > -eps)
    assert np.all(sep(test3,unit(test3)) < [ eps, eps])
    assert np.all(sep(test3,unit(test3)) > [-eps,-eps])

    # cross2d, sep
    assert cross2d((1,0),(0,1)) == 1.
    assert cross2d((1,0),(1,1)) == 1.
    assert cross2d((1,0),(111,1)) == 1.
    assert cross2d((0,1),(111,1)) == -111.

    dirs = np.asarray([[[( 5, 0),( 4, 3),( 3, 4)],
                        [( 0, 5),(-3, 4),(-4, 3)]],
                       [[(-5, 0),(-4,-3),(-3,-4)],
                        [( 0,-5),( 3,-4),( 4,-3)]]], dtype=np.float64)
    assert np.all(cross2d(dirs,(1,0)) == -dirs[...,1])
    assert np.all(cross2d(dirs,(0,1)) ==  dirs[...,0])

    # cross3d
    assert np.all(cross3d((1,0,0),(0,1,0)) == (0, 0,1))
    assert np.all(cross3d((1,0,0),(0,0,1)) == (0,-1,0))

    assert np.all(cross3d([(1 ,0,0),(0,2,0)],(0,0,1)) == [(0,-1,0),(2,0,0)])

    # ucross3d, sep, norm
    eps = 1.e-15
    vec1 = [(7,-1,1),(1,2,-3),(-1,3,3)]
    vec2 = (3,1,-3)

    test = ucross3d(vec1, vec2)
    assert np.all(norm(test) > 1. - eps)
    assert np.all(norm(test) < 1. + eps)
    assert np.all(sep(test,vec2) > np.pi/2. - eps)
    assert np.all(sep(test,vec2) < np.pi/2. + eps)

    vec2 = [(3,2,1),(-4,-1,0),(7,6,5)]

    test = ucross3d(vec1, vec2)
    assert np.all(norm(test) > 1. - eps)
    assert np.all(norm(test) < 1. + eps)
    assert np.all(sep(test,vec2) > np.pi/2. - eps)
    assert np.all(sep(test,vec2) < np.pi/2. + eps)

    # proj, perp, sep, norm
    eps = 3.e-15
    perps = perp(vec1, vec2)
    projs = proj(vec1, vec2)

    assert np.all(dot(perps,vec2) > -eps)
    assert np.all(dot(perps,vec2) <  eps)

    assert np.all(sep(perps,vec2) > np.pi/2. - eps)
    assert np.all(sep(perps,vec2) < np.pi/2. + eps)

    assert np.all(sep(projs,vec2) % np.pi > -eps)
    assert np.all(sep(projs,vec2) % np.pi <  eps)

    test = vec1 - (projs + perps)
    assert np.all(test > -eps)
    assert np.all(test <  eps)

    vec2 = [(3,2,1),(-4,-1,0),(7,6,5)]

    assert np.all(dot(perps,vec2) > -eps)
    assert np.all(dot(perps,vec2) <  eps)

    assert np.all(sep(perps,vec2) > np.pi/2. - eps)
    assert np.all(sep(perps,vec2) < np.pi/2. + eps)

    assert np.all(sep(projs,vec2) % np.pi > -eps)
    assert np.all(sep(projs,vec2) % np.pi <  eps)

    test = vec1 - (projs + perps)
    assert np.all(test > -eps)
    assert np.all(test <  eps)

    # xpose
    mat = [[[1,2,3],[4,5,6],[7,8,9]]] * 7
    assert np.shape(mat) == (7,3,3)
    assert np.shape(xpose(mat)) == (7,3,3)
    assert np.all(np.array(mat)[...,0,1] == xpose(mat)[...,1,0])

    # twovec, mxv, mtxv, twovec
    eps = 1.e-14

    mat1 = twovec((1,0,0),0,(0,1,0),1)
    mat2 = twovec((1,0,0),0,(0,0,4),2)
    assert np.all(mat1 == mat2)
    assert np.all(mat1 == [[1,0,0,],[0,1,0],[0,0,1]])

    assert np.all(mxv( mat1,vec1) == vec1)
    assert np.all(mtxv(mat1,vec1) == vec1)
    assert np.all(mxv( mat1,vec1[0]) == vec1[0])
    assert np.all(mtxv(mat1,vec1[0]) == vec1[0])

    # Rotate vectors along the axes into the frame
    mat = twovec((1,1,1),2,[(1,0,-1),(-1,0,1)],0)
    vec = (3,3,3)

    assert np.all(mxv(mat,vec)[...,0:2] > -eps)
    assert np.all(mxv(mat,vec)[...,0:2] <  eps)
    assert np.all(mxv(mat,vec)[...,2] > np.sqrt(27) - eps)
    assert np.all(mxv(mat,vec)[...,2] < np.sqrt(27) + eps)

    vec = [(2,0,-2),[-2,0,2]]
    assert np.all(mxv(mat,vec)[:,1:3] > -eps)
    assert np.all(mxv(mat,vec)[:,1:3] <  eps)
    assert np.all(mxv(mat,vec)[:,0] > np.sqrt(8) - eps)
    assert np.all(mxv(mat,vec)[:,0] < np.sqrt(8) + eps)

    # Rotate axis vectors out of the frame
    vec = [[(1,0,0),[0,1,0]],[(2,3,4),[0,0,2]]]
    result = mtxv(mat,vec)

    assert result[1,1,0] == result[1,1,1]
    assert result[1,1,0] == result[1,1,2]
    assert result[0,0,0] == -result[0,0,2]
    assert result[0,0,1] == 0.

    result = mxv(xpose(mat),vec)
    assert result[1,1,0] == result[1,1,1]
    assert result[1,1,0] == result[1,1,2]
    assert result[0,0,0] == -result[0,0,2]
    assert result[0,0,1] == 0.

    mat = [[1,2,3],[4,5,6],[7,8,9]]
    vec = [1,0,0]
    assert np.all(mxv(mat,vec)  - [1,4,7]) == 0.
    assert np.all(mtxv(mat,vec) - [1,2,3]) == 0.
    vec = [0,1,0]
    assert np.all(mxv(mat,vec)  - [2,5,8]) == 0.
    assert np.all(mtxv(mat,vec) - [4,5,6]) == 0.

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

    assert axb.shape == (2,3,4,3,3)
    assert axv.shape == (2,3,4,3)

    eps = 1.e-15

    for i in range(2):
      for j in range(3):
        for k in range(4):
            am = np.array(a[i,0,k])
            bm = np.array(b[  j,k])
            amt = np.array(a[i,0,k].T)
            bmt = np.array(b[  j,k].T)
            vm  = np.array(v[0,j,0])

            test = am @ bm
            assert np.abs(test - axb[i,j,k]).max() < eps

            test = amt @ bm
            assert np.abs(test - atxb[i,j,k]).max() < eps

            test = am @ bmt
            assert np.abs(test - axbt[i,j,k]).max() < eps

            test = amt @ bmt
            assert np.abs(test - atxbt[i,j,k]).max() < eps

            test = am @ vm
            assert np.abs(test - axv[i,j,k,:,np.newaxis]).max() < eps

            test = amt @ vm
            assert np.abs(test - atxv[i,j,k,:,np.newaxis]).max() < eps
################################################################################
