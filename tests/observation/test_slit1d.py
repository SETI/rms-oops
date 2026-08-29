##########################################################################################
# tests/observation/test_slit1d.py
##########################################################################################

import numpy as np

from polymath         import Pair, Vector
from oops.observation import Slit1D
from oops.fov         import FlatFOV


def test_slit1d():
    fov = FlatFOV((0.001,0.001), (20,1))
    obs = Slit1D(('u'), tstart=0., texp=10., fov=fov, path='SSB', frame='J2000')

    indices = Vector([(0,0),(1,0),(20,0),(21,0)])
    indices_ = indices.copy()       # clipped at 20
    indices_.vals[:,0][indices_.vals[:,0] == 20] -= 1

    # uvt() with remask == False
    (uv,time) = obs.uvt(indices)
    assert not np.any(uv.mask)
    assert not np.any(time.mask)
    assert time == 5.
    assert uv.to_scalar(0) == indices.to_scalar(0)
    assert uv.to_scalar(1) == 0.5

    # uvt() with remask == True
    (uv,time) = obs.uvt(indices, remask=True)

    assert np.all(uv.mask == np.array(3*[False] + [True]))
    assert np.all(time.mask == uv.mask)
    assert time[:3] == (5,5,5)
    assert uv[:3].to_scalar(0) == indices[:3].to_scalar(0)
    assert uv[:3].to_scalar(1) == 0.5

    # uvt_range() with remask == False
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert not np.any(uv_min.mask)
    assert not np.any(uv_max.mask)
    assert not np.any(time_min.mask)
    assert not np.any(time_max.mask)

    assert uv_min.to_scalar(0) == indices_.to_scalar(0)
    assert uv_min.to_scalar(1) == 0
    assert uv_max.to_scalar(0) == indices_.to_scalar(0) + 1
    assert uv_max.to_scalar(1) == 1
    assert time_min == 0.
    assert time_max == 10.

    # uvt_range() with remask == True
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices,
                                                         remask=True)
    assert np.all(uv_min.mask == np.array(3*[False] + [True]))
    assert np.all(uv_max.mask == uv_min.mask)
    assert np.all(time_min.mask == uv_min.mask)
    assert np.all(time_max.mask == uv_min.mask)

    assert uv_min.to_scalar(0)[:2] == indices.to_scalar(0)[:2]
    assert uv_min.to_scalar(1)[:2] == 0
    assert uv_max.to_scalar(0)[:2] == indices.to_scalar(0)[:2] + 1
    assert uv_max.to_scalar(1)[:2] == 1
    assert time_min[:2] == 0.
    assert time_max[:2] == 10.

    # time_range_at_uv() with remask == False
    uv = Pair([(0,0),(0,0.5),(0,1),(0,2),
               (20,0),(20,0.5),(20,1),(20,2),
               (21,0)])

    (time0, time1) = obs.time_range_at_uv(uv)

    assert time0 == 0.
    assert time1 == 10.

    # time_range_at_uv() with remask == True
    (time0, time1) = obs.time_range_at_uv(uv, remask=True)

    assert np.all(time0.mask == 3*[False] + [True] + 3*[False] + 2*[True])
    assert np.all(time1.mask == time0.mask)
    assert time0[:3] == 0.
    assert time1[:3] == 10.

    ######################################################################################

    # Alternative axis order ('a','u','b')

    fov = FlatFOV((0.001,0.001), (20,1))
    obs = Slit1D(('a','u', 'b'), tstart=0., texp=10.,
                 fov=fov, path='SSB', frame='J2000')

    indices = Vector([(0,0,0),(0,1,99),(0,19,99),(10,20,99),(10,21,99)])
    indices_ = indices.copy()       # clipped at 20
    indices_.vals[:,1][indices_.vals[:,1] == 20] -= 1

    (uv,time) = obs.uvt(indices)

    assert uv.to_scalar(0) == indices.to_scalar(1)
    assert uv.to_scalar(1) == 0.5
    assert time == 5.

    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert uv_min.to_scalar(0) == indices_.to_scalar(1)
    assert uv_max.to_scalar(0) == indices_.to_scalar(1)+1
    assert uv_min.to_scalar(1) == 0.
    assert uv_max.to_scalar(1) == 1
    assert time_min == 0.
    assert time_max == 10.
##########################################################################################
