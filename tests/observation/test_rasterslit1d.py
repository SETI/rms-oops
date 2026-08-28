################################################################################
# tests/observation/test_rasterslit1d.py
################################################################################

import numpy as np

from polymath         import Scalar, Pair
from oops.cadence     import Metronome
from oops.fov         import FlatFOV
from oops.observation import RasterSlit1D


def test_rasterslit1d():
    ############################################
    # Continuous 2-D observation
    # First axis = U and T with length 10
    # Second axis ignored
    ############################################

    fov = FlatFOV((0.001,0.001), (10,1))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
    obs = RasterSlit1D(axes=('ut','a'), cadence=cadence,
                       fov=fov, path='SSB', frame='J2000')

    indices = Pair([(0,0),(10,0),(11,0)])
    indices_ = indices.copy()   # clipped at top
    indices_.vals[:,0][indices_.vals[:,0] == 10] -= 1

    # uvt() with remask == False
    (uv, time) = obs.uvt(indices)

    assert not np.any(uv.mask)
    assert not np.any(time.mask)
    assert time == cadence.time_at_tstep(indices.to_scalar(0))
    assert uv.to_scalar(0) == indices.to_scalar(0)
    assert uv.to_scalar(1) == 0.5

    # uvt() with remask == True
    (uv, time) = obs.uvt(indices, remask=True)

    assert np.all(uv.mask == np.array(2*[False] + [True]))
    assert np.all(time.mask == uv.mask)
    assert time[:2] == cadence.tstride * indices.to_scalar(0)[:2]
    assert uv[:2].to_scalar(0) == indices[:2].to_scalar(0)
    assert uv[:2].to_scalar(1) == 0.5

    # uvt() with remask == True, new indices
    non_ints = indices + (0.2,0.9)
    (uv, time) = obs.uvt(non_ints, remask=True)

    assert np.all(uv.mask == np.array([False] + 2*[True]))
    assert np.all(time.mask == uv.mask)

    # uvt_range() with remask == False
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints)

    assert not np.any(uv_min.mask)
    assert not np.any(uv_max.mask)
    assert not np.any(time_min.mask)
    assert not np.any(time_max.mask)

    assert uv_min.to_scalar(0) == indices.to_scalar(0)
    assert uv_min.to_scalar(1) == 0
    assert uv_max.to_scalar(0) == indices.to_scalar(0) + 1
    assert uv_max.to_scalar(1) == 1
    assert time_min == cadence.time_range_at_tstep(indices.to_scalar(0))[0]
    assert time_max == time_min + 10.

    # uvt_range() with remask == True
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices,
                                                         remask=True)

    assert np.all(uv_min.mask == np.array([False, False, True]))
    assert np.all(uv_max.mask == uv_min.mask)
    assert np.all(time_min.mask == uv_min.mask)
    assert np.all(time_max.mask == uv_min.mask)

    assert uv_min.to_scalar(0)[:2] == indices_.to_scalar(0)[:2]
    assert uv_min.to_scalar(1)[:2] == 0
    assert uv_max.to_scalar(0)[:2] == indices_.to_scalar(0)[:2] + 1
    assert uv_max.to_scalar(1)[:2] == 1
    assert time_min[:2] == cadence.tstride*indices_.to_scalar(0)[:2]
    assert time_max[:2] == time_min[:2] + cadence.texp

    assert uv_min[2] == Pair.MASKED
    assert time_min[2] == Scalar.MASKED
    assert time_min[2] == Scalar.MASKED

    # time_range_at_uv() with remask == False
    uv = Pair([(0,0),(0,20),(10,0),(10,20),(11,21)])
    uv_ = uv.copy()
    uv_.vals[:,0][uv_.vals[:,0] == 10] -= 1

    (time0, time1) = obs.time_range_at_uv(uv)

    assert time0 == cadence.time_range_at_tstep(uv_.to_scalar(0))[0]
    assert time1 == time0 + cadence.texp

    # time_range_at_uv() with remask == True
    (time0, time1) = obs.time_range_at_uv(uv, remask=True)

    assert np.all(time0.mask == 4*[False] + [True])
    assert np.all(time1.mask == 4*[False] + [True])
    assert time0[:4] == cadence.tstride * uv_.to_scalar(0)[:4]
    assert time1[:4] == time0[:4] + cadence.texp

    ############################################################
    # Alternative axis order ('a', 'vt')
    # Second axis = V and T with length 10
    # Discontinuous time sampling [0-8], [10-18], ..., [90-98]
    # First axis ignored
    ############################################################

    fov = FlatFOV((0.001,0.001), (1,10))
    cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
    obs = RasterSlit1D(axes=('a','vt'), cadence=cadence,
                       fov=fov, path='SSB', frame='J2000')

    indices = Pair([(0,0),(0,9),(0,10),(0,11)])
    indices_ = indices.copy()   # clipped at top
    indices_.vals[:,1][indices_.vals[:,1] == 10] -= 1

    (uv,time) = obs.uvt(indices)

    assert uv.to_scalar(0) == 0.5
    assert uv.to_scalar(1) == indices.to_scalar(1)
    assert time == [0,90,98,98]

    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert uv_min.to_scalar(0) == 0
    assert uv_min.to_scalar(1) == indices_.to_scalar(1)
    assert uv_max.to_scalar(0) == 1
    assert uv_max.to_scalar(1) == indices_.to_scalar(1) + 1
    assert time_min == cadence.time_range_at_tstep(indices_.to_scalar(1))[0]
    assert time_max == time_min + cadence.texp

    uv = Pair([(11,0),(11,9),(11,10),(11,11)])
    uv_ = uv.copy()
    uv_.vals[:,1][uv_.vals[:,1] == 10] -= 1

    (time0, time1) = obs.time_range_at_uv(uv)

    assert time0 == cadence.time_range_at_tstep(uv_.to_scalar(1))[0]
    assert time1 == time0 + cadence.texp

    ############################################################
    # Similar to above but 1-D observation
    # First axis = V and T with length 10
    # Discontinuous time sampling [0-8], [10-18], ..., [90-98]
    ############################################################

    fov = FlatFOV((0.001,0.001), (1,10))
    cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
    obs = RasterSlit1D(axes=('vt',), cadence=cadence,
                       fov=fov, path='SSB', frame='J2000')

    indices = Scalar([0,9,10,11])
    indices_ = indices.copy()   # clipped at top
    indices_.vals[indices_.vals == 10] -= 1

    (uv,time) = obs.uvt(indices)

    assert uv.to_scalar(0) == 0.5
    assert uv.to_scalar(1) == indices
    assert time == [0,90,98,98]

    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert uv_min.to_scalar(0) == 0
    assert uv_min.to_scalar(1) == indices_
    assert uv_max.to_scalar(0) == 1
    assert uv_max.to_scalar(1) == indices_ + 1
    assert time_min == cadence.time_range_at_tstep(indices_)[0]
    assert time_max == time_min + cadence.texp

    uv = Pair([(11,0),(11,9),(11,10),(11,11)])
    uv_ = uv.copy()
    uv_.vals[:,1][uv_.vals[:,1] == 10] -= 1

    (time0, time1) = obs.time_range_at_uv(uv)

    assert time0 == cadence.time_range_at_tstep(uv_.to_scalar(1))[0]
    assert time1 == time0 + cadence.texp

    ############################################################
    # Alternative axis order ('ut',), 1-D
    # First axis = U and T with length 10
    # Discontinuous time sampling [0-8], [10-18], ..., [90-98]
    ############################################################

    fov = FlatFOV((0.001,0.001), (10,1))
    cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
    obs = RasterSlit1D(axes=('ut',), cadence=cadence,
                       fov=fov, path='SSB', frame='J2000')

    assert obs.time[1] == 98.

    assert obs.uvt(0,True)[1] == 0.
    assert obs.uvt(5,True)[1] == 50.
    assert obs.uvt(5.5,True)[1] == 54.
    assert obs.uvt(9.5,True)[1] == 94.
    assert obs.uvt(10.,True)[1] == 98.
    assert obs.uvt(10.001,True)[1].mask

    eps = 1.e-14
    delta = 1.e-13
    assert abs(obs.uvt((6.     ),True)[0] - (6.0,0.5)) < delta
    assert abs(obs.uvt((6.2    ),True)[0] - (6.2,0.5)) < delta
    assert abs(obs.uvt((6.4    ),True)[0] - (6.4,0.5)) < delta
    assert abs(obs.uvt((6.6    ),True)[0] - (6.6,0.5)) < delta
    assert abs(obs.uvt((6.8    ),True)[0] - (6.8,0.5)) < delta
    assert abs(obs.uvt((7.     ),True)[0] - (7.0,0.5)) < delta
    assert abs(obs.uvt((10     ),True)[0] - (10.,0.5)) < delta
    assert obs.uvt(10.+eps,True)[0].mask

    indices = Scalar([10-eps, 10, 10+eps])

    (uv,t) = obs.uvt(indices, remask=True)
    assert np.all(t.mask == np.array(2*[False] + [True]))
################################################################################
