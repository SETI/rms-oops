##########################################################################################
# tests/observation/test_timedimage.py
##########################################################################################

import numpy as np

from polymath         import Pair, Vector, Boolean, Scalar
from oops.cadence     import DualCadence, Metronome, TDICadence
from oops.fov         import FlatFOV
from oops.observation import TimedImage



def test_timedimage():
    ######################################################################################
    # Old RasterScan unit tests
    ######################################################################################

    RasterScan = TimedImage

    ######################################################################################
    # Continuous observation, shape (10,20)
    # Axes are (fast,slow)
    ######################################################################################

    fov = FlatFOV((0.001,0.001), (10,20))
    slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
    fast_cadence = Metronome(tstart=0., tstride=1., texp=1., steps=10)
    cadence = DualCadence(slow_cadence, fast_cadence)
    obs = RasterScan(axes=('ufast','vslow'), cadence=cadence, fov=fov,
                     path='SSB', frame='J2000')

    indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(0,21),(10,21)])
    indices_ = indices.copy()
    indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
    indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

    # uvt() with remask == False
    (uv, time) = obs.uvt(indices)

    assert not np.any(uv.mask)
    assert not np.any(time.mask)
    assert time == [0, 100, 190, 10, 110, 200, 190, 200]
    assert uv == Pair.as_pair(indices)

    # uvt() with remask == True
    (uv, time) = obs.uvt(indices, remask=True)

    assert np.all(uv.mask == np.array(6*[False] + 2*[True]))
    assert np.all(time.mask == uv.mask)
    assert time[:6] == [0, 100, 190, 10, 110, 200]
    assert uv[:6] == Pair.as_pair(indices)[:6]

    # uvt_range() with remask == False
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert not np.any(uv_min.mask)
    assert not np.any(uv_max.mask)
    assert not np.any(time_min.mask)
    assert not np.any(time_max.mask)

    assert uv_min == Pair.as_pair(indices_)
    assert uv_max == Pair.as_pair(indices_) + (1,1)
    assert time_min == [0, 100, 190,  9, 109, 199, 190, 199]
    assert time_max == [1, 101, 191, 10, 110, 200, 191, 200]

    # uvt_range() with remask == True
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices, remask=True)

    assert Boolean(uv_min.mask) == 6*[False] + 2*[True]
    assert Boolean(uv_max.mask) == 6*[False] + 2*[True]
    assert Boolean(time_min.mask) == 6*[False] + 2*[True]
    assert Boolean(time_max.mask) == 6*[False] + 2*[True]

    assert uv_min[:6] == Pair.as_pair(indices_)[:6]
    assert uv_max[:6] == Pair.as_pair(indices_)[:6] + (1,1)
    assert time_min[:6] == [0, 100, 190,  9, 109, 199]
    assert time_max[:6] == time_min[:6] + fast_cadence._texp

    # uvt() with remask == False, non-integer indices
    non_ints = indices + (0.2, 0.9)
    (uv, time) = obs.uvt(non_ints)

    assert not np.any(uv.mask)
    assert not np.any(time.mask)
    assert time == cadence.time_at_tstep(uv.swapxy())
    assert uv == Pair.as_pair(non_ints)

    # uvt() with remask == True, non-integer indices
    non_ints = indices + (0.2, 0.9)
    (uv, time) = obs.uvt(non_ints, remask=True)

    assert Boolean(uv.mask) == 2*[False] + 6*[True]
    assert Boolean(time.mask) == 2*[False] + 6*[True]
    assert (time[:2]
            == (slow_cadence._tstride * non_ints.to_scalar(1).int()
                + fast_cadence._tstride * non_ints.to_scalar(0))[:2])
    assert uv[:2] == Pair.as_pair(non_ints)[:2]

    # uvt_range() with remask == False, non-integer indices
    non_ints = indices + (0.2, 0.9)
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints)

    assert not np.any(uv_min.mask)
    assert not np.any(uv_max.mask)
    assert not np.any(time_min.mask)
    assert not np.any(time_max.mask)

    assert uv_min == Pair.as_pair(indices)
    assert uv_max == Pair.as_pair(indices) + (1,1)
    assert time_min == cadence.time_range_at_tstep(Pair.as_pair(non_ints).swapxy())[0]
    assert time_max == time_min + fast_cadence._texp

    # uvt_range() with remask == True, non-integer indices
    non_ints = indices + (0.2, 0.9)
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints, remask=True)

    assert Boolean(uv_min.mask) == 2*[False] + 6*[True]
    assert Boolean(uv_max.mask) == 2*[False] + 6*[True]
    assert Boolean(time_min.mask) == 2*[False] + 6*[True]
    assert Boolean(time_max.mask) == 2*[False] + 6*[True]

    assert uv_min[:2] == Pair.as_pair(indices)[:2]
    assert uv_max[:2] == Pair.as_pair(indices)[:2] + (1,1)
    assert (time_min[:2]
            == (slow_cadence._tstride * non_ints.to_scalar(1).int()
                + fast_cadence._tstride * non_ints.to_scalar(0).int())[:2])
    assert time_max[:2] == time_min[:2] + fast_cadence._texp

    # time_range_at_uv() with remask == False
    uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])

    (time0, time1) = obs.time_range_at_uv(uv)

    assert time0 == [0, 190, 9, 199, 199]
    assert time1 == time0 + fast_cadence._texp

    # time_range_at_uv() with remask == True
    (time0, time1) = obs.time_range_at_uv(uv, remask=True)

    assert np.all(time0.mask == 4*[False] + [True])
    assert np.all(time1.mask == 4*[False] + [True])
    assert time0[:4] == [0, 190, 9, 199]
    assert time1[:4] == time0[:4] + fast_cadence._texp

    ######################################################################################
    # Fast cadence is discontinuous
    # Axes are (slow,fast)
    # Shape (10,20)
    # [[0-1, 10-11, 20-21, ..., 190-191],
    #  [1000-1001, 1010-1011, ..., 1190-1191],
    #  ...
    #  [9000-9001, 9010-9011, ..., 9190, 9191]]
    ######################################################################################

    fov = FlatFOV((0.001,0.001), (10,20))
    slow_cadence = Metronome(tstart=0., tstride=1000., texp=1., steps=10)
    fast_cadence = Metronome(tstart=0., tstride=10., texp=1., steps=20)
    cadence = DualCadence(slow_cadence, fast_cadence)
    obs = RasterScan(axes=('uslow','vfast'), cadence=cadence, fov=fov,
                                             path='SSB', frame='J2000')

    indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
    indices_ = indices.copy()
    indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
    indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

    (uv, time) = obs.uvt(indices)

    assert not np.any(uv.mask)
    assert not np.any(time.mask)
    assert time == [0, 100, 191, 9000, 9100, 9191, 9191]
    assert uv == Pair.as_pair(indices)

    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert uv_min == Pair.as_pair(indices_)
    assert uv_max == Pair.as_pair(indices_) + (1,1)
    assert time_min == cadence.time_range_at_tstep(indices_)[0]
    assert time_max == time_min + fast_cadence._texp

    (time0,time1) = obs.time_range_at_uv(indices)

    assert time0 == cadence.time_range_at_tstep(indices_)[0]
    assert time1 == time0 + fast_cadence._texp

    ######################################################################################
    # Fast cadence is discontinuous
    # Axes are (fast,slow)
    # Shape (10,20)
    ######################################################################################

    fov = FlatFOV((0.001,0.001), (10,20))
    slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
    fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
    cadence = DualCadence(slow_cadence, fast_cadence)
    obs = RasterScan(axes=('ufast','vslow'),
                     cadence=cadence, fov=fov, path='SSB', frame='J2000')

    assert obs.time[1] == 199.8

    assert obs.uvt((0,0))[1] == 0.
    assert obs.uvt((5,0))[1] == 5.
    assert obs.uvt((5,5))[1] == 55.
    assert obs.uvt((5.0, 5.5))[1] == 55.
    assert obs.uvt((5.5, 5.0))[1] == 55.4

    eps = 1.e-15
    delta = 1.e-13
    assert abs(obs.uvt((6.     ,0))[1] - 6. ) < delta
    assert abs(obs.uvt((6.25   ,0))[1] - 6.2) < delta
    assert abs(obs.uvt((6.5    ,0))[1] - 6.4) < delta
    assert abs(obs.uvt((6.75   ,0))[1] - 6.6) < delta
    assert abs(obs.uvt((7 - eps,0))[1] - 6.8) < delta
    assert abs(obs.uvt((7.     ,0))[1] - 7.0) < delta

    assert obs.uvt((0,0))[0] == (0.,0.)
    assert obs.uvt((5,0))[0] == (5.,0.)
    assert obs.uvt((5,5))[0] == (5.,5.)

    assert abs(obs.uvt((6.     ,0))[0] - (6.0,0.)) < delta
    assert abs(obs.uvt((6.2    ,1))[0] - (6.2,1.)) < delta
    assert abs(obs.uvt((6.4    ,2))[0] - (6.4,2.)) < delta
    assert abs(obs.uvt((6.6    ,3))[0] - (6.6,3.)) < delta
    assert abs(obs.uvt((6.8    ,4))[0] - (6.8,4.)) < delta
    assert abs(obs.uvt((7.     ,6))[0] - (7.0,6.)) < delta

    assert abs(obs.uvt((1, 0      ))[0] - (1.,0.0)) < delta
    assert abs(obs.uvt((2, 1.2    ))[0] - (2.,1.2)) < delta
    assert abs(obs.uvt((3, 2.5    ))[0] - (3.,2.5)) < delta
    assert abs(obs.uvt((4, 3.8    ))[0] - (4.,3.8)) < delta
    assert abs(obs.uvt((5, 5.     ))[0] - (5.,5.0)) < delta

    ######################################################################################
    # Alternative tstride for even more discontinuous indices
    ######################################################################################

    fov = FlatFOV((0.001,0.001), (10,20))
    slow_cadence = Metronome(tstart=0., tstride=11., texp=10., steps=20)
    fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
    cadence = DualCadence(slow_cadence, fast_cadence)
    obs = RasterScan(axes=('ufast','vslow'),
                     cadence=cadence, fov=fov, path='SSB', frame='J2000')

    assert obs.time[1] == 218.8

    assert obs.uvt((0,0))[1] == 0.
    assert obs.uvt((5,0))[1] == 5.
    assert obs.uvt((5,5))[1] == 60.
    assert obs.uvt((5.0, 5.5))[1] == 60.
    assert obs.uvt((5.5, 5.0))[1] == 60.4
    assert obs.uvt((5.5, 5.5))[1] == 60.4

    eps = 1.e-14
    delta = 1.e-13
    assert (obs.uvt((6.     ,0.))[1] - 6.0).abs() < delta
    assert (obs.uvt((6.25   ,0.))[1] - 6.2).abs() < delta
    assert (obs.uvt((6.5    ,0.))[1] - 6.4).abs() < delta
    assert (obs.uvt((6.75   ,0.))[1] - 6.6).abs() < delta
    assert (obs.uvt((7. -eps,0.))[1] - 6.8).abs() < delta
    assert (obs.uvt((7.     ,0.))[1] - 7.0).abs() < delta

    assert (obs.uvt((9.      ,0.))[1] -  9.0).abs() < delta
    assert (obs.uvt((9.25    ,0.))[1] -  9.2).abs() < delta
    assert (obs.uvt((9.5     ,0.))[1] -  9.4).abs() < delta
    assert (obs.uvt((9.75    ,0.))[1] -  9.6).abs() < delta
    assert (obs.uvt((10 - eps,0.))[1] -  9.8).abs() < delta
    assert (obs.uvt((0.      ,1.))[1] - 11.0).abs() < delta

    assert (obs.uvt((6.00, 0.    ))[1] -  6.0).abs() < delta
    assert (obs.uvt((6.25, 0.    ))[1] -  6.2).abs() < delta
    assert (obs.uvt((6.25, 1.    ))[1] - 17.2).abs() < delta
    assert (obs.uvt((6.25, 2.-eps))[1] - 17.2).abs() < delta
    assert (obs.uvt((6.25, 2.    ))[1] - 28.2).abs() < delta

    # Test the upper edge
    pair = (10-eps, 20-eps)
    assert obs.uvt(pair)[0] == pair
    assert (obs.uvt(pair)[1] - 218.8).abs() < delta

    pair = (10, 20-eps)
    assert obs.uvt(pair)[0] == pair
    assert (obs.uvt(pair)[1] - 218.8).abs() < delta

    pair = (10-eps, 20)
    assert obs.uvt(pair)[0] == pair
    assert (obs.uvt(pair)[1] - 218.8).abs() < delta

    pair = (10, 20)
    assert obs.uvt(pair)[0] == pair
    assert (obs.uvt(pair)[1] - 218.8).abs() < delta

    assert obs.uvt((10+eps, 20), remask=True)[0].mask
    assert obs.uvt((10, 20+eps), remask=True)[0].mask

    # Try all at once
    indices = Pair([(10-eps,20-eps), (10,20-eps), (10-eps,20), (10,20),
                    (10+eps,20), (10,20+eps)])
    (test_uv, time) = obs.uvt(indices, remask=True)

    assert Boolean(test_uv.mask) == 4*[False] + 2*[True]
    assert test_uv[:4] == indices[:4]
    assert ((time[:4] - 218.8).abs() < delta).all()
    assert Boolean(time.mask) == test_uv.mask

    ######################################################################################
    # Alternative texp and axes
    ######################################################################################

    fov = FlatFOV((0.001,0.001), (10,20))
    slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
    fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
    cadence = DualCadence(slow_cadence, fast_cadence)
    obs = RasterScan(axes=('a','vslow','b','ufast','c'),
                     cadence=cadence, fov=fov, path='SSB', frame='J2000')

    assert obs.time[1] == 199.8

    assert obs.uvt((1,0,3,0,4))[1] == 0.
    assert obs.uvt((1,0,3,5,4))[1] == 5.
    assert obs.uvt((1,0,3,5.5,4))[1] == 5.4

    eps = 1.e-15
    delta = 1.e-13
    assert abs(obs.uvt((1,0,0,6      ,0))[1] - 6. ) < delta
    assert abs(obs.uvt((1,0,0,6.25   ,0))[1] - 6.2) < delta
    assert abs(obs.uvt((1,0,0,6.5    ,0))[1] - 6.4) < delta
    assert abs(obs.uvt((1,0,0,6.75   ,0))[1] - 6.6) < delta
    assert abs(obs.uvt((1,0,0,7 - eps,0))[1] - 6.8) < delta
    assert abs(obs.uvt((1,0,0,7.     ,0))[1] - 7.0) < delta

    assert obs.uvt((0,0,0,0,0))[0] == (0.,0.)
    assert obs.uvt((0,0,0,5,0))[0] == (5.,0.)
    assert obs.uvt((0,5,0,5,0))[0] == (5.,5.)

    assert abs(obs.uvt((1,0,4,6   ,7))[0] - (6.0,0.)) < delta
    assert abs(obs.uvt((1,1,4,6.2 ,7))[0] - (6.2,1.)) < delta
    assert abs(obs.uvt((1,2,4,6.4 ,7))[0] - (6.4,2.)) < delta
    assert abs(obs.uvt((1,3,4,6.6 ,7))[0] - (6.6,3.)) < delta
    assert abs(obs.uvt((1,4,4,6.8 ,7))[0] - (6.8,4.)) < delta
    assert abs(obs.uvt((1,6,4,7.  ,7))[0] - (7.0,6.)) < delta

    assert abs(obs.uvt((1, 0      ,4,1,7))[0] - (1.,0.0)) < delta
    assert abs(obs.uvt((1, 1.2    ,4,2,7))[0] - (2.,1.2)) < delta
    assert abs(obs.uvt((1, 2.5    ,4,3,7))[0] - (3.,2.5)) < delta
    assert abs(obs.uvt((1, 3.7    ,4,4,7))[0] - (4.,3.7)) < delta
    assert abs(obs.uvt((1, 5.     ,4,5,7))[0] - (5.,5.0)) < delta

    ######################################################################################
    # Old Pushbroom unit tests
    ######################################################################################

    Pushbroom = TimedImage

    ######################################################################################
    # Overall shape (10,20)
    # Time is second axis; time = v * 10.
    ######################################################################################

    flatfov = FlatFOV((0.001,0.001), (10,20))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
    obs = Pushbroom(axes=('u','vt'), cadence=cadence, fov=flatfov,
                                     path='SSB', frame='J2000')

    indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
    tstep = indices.to_scalar(1)

    indices_ = indices.copy()   # clipped at top
    indices_.vals[:,0][indices_.vals[:,0] == 10] -= 1
    indices_.vals[:,1][indices_.vals[:,1] == 20] -= 1

    # uvt() with remask == False
    (uv, time) = obs.uvt(indices)

    assert not np.any(uv.mask)
    assert not np.any(time.mask)
    assert time == cadence.time_at_tstep(tstep)
    assert uv == Pair.as_pair(indices)

    # uvt() with remask == True
    (uv, time) = obs.uvt(indices, remask=True)

    assert np.all(uv.mask == np.array(6*[False] + [True]))
    assert np.all(time.mask == uv.mask)
    assert time[:6] == cadence._tstride * indices.to_scalar(1)[:6]
    assert uv[:6] == Pair.as_pair(indices)[:6]

    # uvt_range() with remask == False
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert not np.any(uv_min.mask)
    assert not np.any(uv_max.mask)
    assert not np.any(time_min.mask)
    assert not np.any(time_max.mask)

    assert uv_min == Pair.as_pair(indices_)
    assert uv_max == Pair.as_pair(indices_) + (1,1)
    assert time_min == cadence.time_range_at_tstep(tstep)[0]
    assert time_max == time_min + cadence._texp

    # uvt_range() with remask == False, new indices
    non_ints = indices + (0.2,0.9)
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints)

    assert not np.any(uv_min.mask)
    assert not np.any(uv_max.mask)
    assert not np.any(time_min.mask)
    assert not np.any(time_max.mask)

    assert uv_min == Pair.as_pair(indices)
    assert uv_max == Pair.as_pair(indices) + (1,1)
    assert time_min == cadence.time_range_at_tstep(tstep)[0]
    assert time_max == time_min + cadence._texp

    # uvt_range() with remask == True, new indices
    non_ints = indices + (0.2,0.9)
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints,
                                                         remask=True)

    assert np.all(uv_min.mask == np.array(2*[False] + 5*[True]))
    assert np.all(uv_max.mask == uv_min.mask)
    assert np.all(time_min.mask == uv_min.mask)
    assert np.all(time_max.mask == uv_min.mask)

    assert uv_min[:2] == Pair.as_pair(indices)[:2]
    assert uv_max[:2] == Pair.as_pair(indices)[:2] + (1,1)
    assert time_min[:2] == cadence.time_range_at_tstep(tstep)[0][:2]
    assert time_max[:2] == time_min[:2] + cadence._texp

    # time_range_at_uv() with remask == False
    uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])
    tstep = uv.to_scalar(1)

    uv_ = uv.copy()
    uv_.vals[:,0][uv_.vals[:,0] == 10] -= 1
    uv_.vals[:,1][uv_.vals[:,1] == 20] -= 1

    (time0, time1) = obs.time_range_at_uv(uv)
    assert time0 == cadence.time_range_at_tstep(tstep)[0]
    assert time1 == time0 + cadence._texp

    # time_range_at_uv() with remask == True
    (time0, time1) = obs.time_range_at_uv(uv, remask=True)
    assert np.all(time0.mask == 4*[False] + [True])
    assert np.all(time1.mask == 4*[False] + [True])
    assert time0[:4] == cadence._tstride * uv_.to_scalar(1)[:4]
    assert time1[:4] == time0[:4] + cadence._texp

    ######################################################################################
    # Alternative axis order ('ut','v')
    # Overall shape (10,20)
    # Time is first axis; time = v * 10.
    ######################################################################################

    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
    obs = Pushbroom(axes=('ut','v'), cadence=cadence, fov=flatfov,
                                     path='SSB', frame='J2000')

    indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
    indices_ = indices.copy()
    indices_.vals[:,0][indices_.vals[:,0] == 10] -= 1
    indices_.vals[:,1][indices_.vals[:,1] == 20] -= 1

    (uv, time) = obs.uvt(indices)

    uv_ = uv.copy()
    uv_.vals[:,0][uv_.vals[:,0] == 10] -= 1
    uv_.vals[:,1][uv_.vals[:,1] == 20] -= 1

    assert uv == Pair.as_pair(indices)
    assert time == cadence._tstride * indices.to_scalar(0)

    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert uv_min == Pair.as_pair(indices_)
    assert uv_max == Pair.as_pair(indices_) + (1,1)
    assert time_min == cadence._tstride * indices_.to_scalar(0)
    assert time_max == time_min + cadence._texp

    (time0, time1) = obs.time_range_at_uv(indices)

    assert time0 == cadence._tstride * uv_.to_scalar(0)
    assert time1 == time0 + cadence._texp

    ######################################################################################
    # Alternative texp for discontinuous time index
    # Overall shape (10,20)
    # Time is first axis; time = [0-8, 10-18, ..., 90-98]
    ######################################################################################

    cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
    obs = Pushbroom(axes=('ut','v'), cadence=cadence, fov=flatfov,
                                     path='SSB', frame='J2000')

    assert obs.time[1] == 98.

    assert obs.uvt((0,0))[1] == 0.
    assert obs.uvt((5,0))[1] == 50.
    assert obs.uvt((5,5))[1] == 50.

    eps = 1.e-14
    delta = 1.e-13
    assert abs(obs.uvt((6      ,0))[1] - 60.) < delta
    assert abs(obs.uvt((6.25   ,0))[1] - 62.) < delta
    assert abs(obs.uvt((6.5    ,0))[1] - 64.) < delta
    assert abs(obs.uvt((6.75   ,0))[1] - 66.) < delta
    assert abs(obs.uvt((7 - eps,0))[1] - 68.) < delta
    assert abs(obs.uvt((7.     ,0))[1] - 70.) < delta

    assert obs.uvt((0,0))[0] == (0.,0.)
    assert obs.uvt((5,0))[0] == (5.,0.)
    assert obs.uvt((5,5))[0] == (5.,5.)

    assert abs(obs.uvt((6      ,0))[0] - (6.0,0.)) < delta
    assert abs(obs.uvt((6.2    ,1))[0] - (6.2,1.)) < delta
    assert abs(obs.uvt((6.4    ,2))[0] - (6.4,2.)) < delta
    assert abs(obs.uvt((6.6    ,3))[0] - (6.6,3.)) < delta
    assert abs(obs.uvt((6.8    ,4))[0] - (6.8,4.)) < delta
    assert abs(obs.uvt((7 - eps,5))[0] - (7.0,5.)) < delta
    assert abs(obs.uvt((7.     ,6))[0] - (7.0,6.)) < delta

    # Test the upper edge
    uv_list = []
    uvt_list = []
    for i,u in enumerate([10.-eps, 10., 10.+eps]):
      for j,v in enumerate([20.-eps, 20., 20.+eps]):
        uv_list.append((u,v))

        uvt = obs.uvt((u,v), remask=True)
        uvt_list.append(uvt)
        if (i < 2) and (j < 2):
            assert uvt[0] == (u,v)
        else:
            assert uvt[0] == Pair.MASKED

        if (i < 2) and (j < 2):
            assert (uvt[1] - (10. * u - 2.)).abs() < delta
        else:
            assert uvt[1] == Scalar.MASKED

    # Try all at once
    uvt = obs.uvt(uv_list, remask=True)
    assert uvt[0] == [a[0] for a in uvt_list]
    assert uvt[1] == [a[1] for a in uvt_list]

    ######################################################################################
    # Old Slit unit tests
    ######################################################################################

    Slit = TimedImage

    fov = FlatFOV((0.001,0.001), (10,1))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
    obs = Slit(axes=('u','vt'),
               cadence=cadence, fov=fov, path='SSB', frame='J2000')

    indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
    tstep = indices.to_scalar(1)
    indices_ = indices.copy()
    indices_.vals[:,0][indices_.vals[:,0] == 10] -= 1
    indices_.vals[:,1][indices_.vals[:,1] == 20] -= 1

    # uvt() with remask == False
    (uv, time) = obs.uvt(indices)

    assert not np.any(uv.mask)
    assert not np.any(time.mask)
    assert time == cadence.time_at_tstep(tstep)
    assert uv.to_scalar(0) == indices.to_scalar(0)
    assert uv.to_scalar(1) == 0.5

    # uvt() with remask == True
    (uv, time) = obs.uvt(indices, remask=True)

    assert np.all(uv.mask == np.array(6*[False] + [True]))
    assert np.all(time.mask == uv.mask)
    assert time[:6] == cadence._tstride * indices.to_scalar(1)[:6]
    assert uv[:6].to_scalar(0) == indices[:6].to_scalar(0)
    assert uv[:6].to_scalar(1) == 0.5

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
    assert time_min == cadence.time_range_at_tstep(tstep)[0]
    assert time_max == time_min + cadence._texp

    # uvt_range() with remask == True
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices,
                                                         remask=True)

    assert np.all(uv_min.mask == np.array(6*[False] + [True]))
    assert np.all(uv_max.mask == uv_min.mask)
    assert np.all(time_min.mask == uv_min.mask)
    assert np.all(time_max.mask == uv_min.mask)

    assert uv_min.to_scalar(0)[:6] == indices_.to_scalar(0)[:6]
    assert uv_min.to_scalar(1)[:6] == 0
    assert uv_max.to_scalar(0)[:6] == indices_.to_scalar(0)[:6] + 1
    assert uv_max.to_scalar(1)[:6] == 1
    assert time_min[:6] == cadence._tstride * indices_.to_scalar(1)[:6]
    assert time_max[:6] == time_min[:6] + cadence._texp

    # time_range_at_uv() with remask == False
    uv = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
    tstep = indices.to_scalar(1)

    (time0, time1) = obs.time_range_at_uv(uv)
    uv_ = uv.copy()
    uv_.vals[:,0][uv_.vals[:,0] == 10] -= 1
    uv_.vals[:,1][uv_.vals[:,1] == 20] -= 1

    assert time0 == cadence.time_range_at_tstep(tstep)[0]
    assert time1 == time0 + cadence._texp

    # time_range_at_uv() with remask == True
    (time0, time1) = obs.time_range_at_uv(uv, remask=True)

    assert np.all(time0.mask == 6*[False] + [True])
    assert np.all(time1.mask == time0.mask)
    assert time0[:6] == cadence.time_range_at_tstep(tstep)[0][:6]
    assert time1[:6] == time0[:6] + cadence._texp

    ######################################################################################

    # Alternative axis order ('ut','v')

    fov = FlatFOV((0.001,0.001), (1,20))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
    obs = Slit(axes=('ut','v'),
               cadence=cadence, fov=fov, path='SSB', frame='J2000')

    indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
    indices_ = indices.copy()
    indices_.vals[:,0][indices_.vals[:,0] == 10] -= 1
    indices_.vals[:,1][indices_.vals[:,1] == 20] -= 1

    (uv,time) = obs.uvt(indices)

    assert uv.to_scalar(0) == 0.5
    assert uv.to_scalar(1) == indices.to_scalar(1)
    assert time == cadence._tstride * indices.to_scalar(0)

    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert uv_min.to_scalar(0) == 0
    assert uv_min.to_scalar(1) == indices_.to_scalar(1)
    assert uv_max.to_scalar(0) == 1
    assert uv_max.to_scalar(1) == indices_.to_scalar(1) + 1
    assert time_min == cadence._tstride * indices_.to_scalar(0)
    assert time_max == time_min + cadence._texp

    ######################################################################################

    # Alternative texp for discontinuous indices

    fov = FlatFOV((0.001,0.001), (1,20))
    cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
    obs = Slit(axes=('ut','v'),
               cadence=cadence, fov=fov, path='SSB', frame='J2000')

    assert obs.time[1] == 98.

    assert obs.uvt((0,0))[1] == 0.
    assert obs.uvt((5,0))[1] == 50.
    assert obs.uvt((5,5))[1] == 50.

    eps = 1.e-14
    delta = 1.e-13
    assert abs(obs.uvt((6      ,0))[1] - 60.) < delta
    assert abs(obs.uvt((6.25   ,0))[1] - 62.) < delta
    assert abs(obs.uvt((6.5    ,0))[1] - 64.) < delta
    assert abs(obs.uvt((6.75   ,0))[1] - 66.) < delta
    assert abs(obs.uvt((7 - eps,0))[1] - 68.) < delta
    assert abs(obs.uvt((7.     ,0))[1] - 70.) < delta

    assert obs.uvt((0,0))[0] == (0.5,0.)
    assert obs.uvt((5,0))[0] == (0.5,0.)
    assert obs.uvt((5,5))[0] == (0.5,5.)

    assert abs(obs.uvt((6      ,0))[0] - (0.5,0.)) < delta
    assert abs(obs.uvt((6.2    ,1))[0] - (0.5,1.)) < delta
    assert abs(obs.uvt((6.4    ,2))[0] - (0.5,2.)) < delta
    assert abs(obs.uvt((6.6    ,3))[0] - (0.5,3.)) < delta
    assert abs(obs.uvt((6.8    ,4))[0] - (0.5,4.)) < delta
    assert abs(obs.uvt((7 - eps,5))[0] - (0.5,5.)) < delta
    assert abs(obs.uvt((7.     ,6))[0] - (0.5,6.)) < delta

    # Test using scalar indices
    below = obs.uvt((0,20 - eps), remask=True)[0].to_scalar(1)
    exact = obs.uvt((0,20      ), remask=True)[0].to_scalar(1)
    above = obs.uvt((0,20 + eps), remask=True)[0].to_scalar(1)

    assert below < 20.
    assert 20. - below < delta
    assert exact == 20.
    assert above == Scalar.MASKED
    assert above.mask

    # Test using a Vector index
    indices = Vector([(0,20 - eps), (0,20), (0,20 + eps)])

    u = obs.uvt(indices, remask=True)[0].to_scalar(1)
    assert u == (below, exact, above)

    # Alternative texp and axes
    cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
    obs = Slit(axes=('a','v','b','ut','c'),
               cadence=cadence, fov=fov, path='SSB', frame='J2000')

    assert obs.time[1] == 98.

    assert obs.uvt((1,0,3,0,4))[1] == 0.
    assert obs.uvt((1,0,3,5,4))[1] == 50.
    assert obs.uvt((1,0,3,5,4))[1] == 50.

    eps = 1.e-15
    delta = 1.e-13
    assert abs(obs.uvt((1,0,0,6      ,0))[1] - 60.) < delta
    assert abs(obs.uvt((1,0,0,6.25   ,0))[1] - 62.) < delta
    assert abs(obs.uvt((1,0,0,6.5    ,0))[1] - 64.) < delta
    assert abs(obs.uvt((1,0,0,6.75   ,0))[1] - 66.) < delta
    assert abs(obs.uvt((1,0,0,7 - eps,0))[1] - 68.) < delta
    assert abs(obs.uvt((1,0,0,7.     ,0))[1] - 70.) < delta

    assert obs.uvt((0,0,0,0,0))[0] == (0.5,0.)
    assert obs.uvt((0,0,0,5,0))[0] == (0.5,0.)
    assert obs.uvt((0,5,0,5,0))[0] == (0.5,5.)

    assert abs(obs.uvt((1,0,4,6      ,7))[0] - (0.5,0.)) < delta
    assert abs(obs.uvt((1,1,4,6.2    ,7))[0] - (0.5,1.)) < delta
    assert abs(obs.uvt((1,2,4,6.4    ,7))[0] - (0.5,2.)) < delta
    assert abs(obs.uvt((1,3,4,6.6    ,7))[0] - (0.5,3.)) < delta
    assert abs(obs.uvt((1,4,4,6.8    ,7))[0] - (0.5,4.)) < delta
    assert abs(obs.uvt((1,5,4,7 - eps,7))[0] - (0.5,5.)) < delta
    assert abs(obs.uvt((1,6,4,7.     ,7))[0] - (0.5,6.)) < delta

    ######################################################################################
    # Old RasterSlit unit tests
    ######################################################################################

    RasterSlit = TimedImage

    fov = FlatFOV((0.001,0.001), (10,1))
    slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
    fast_cadence = Metronome(tstart=0., tstride=1., texp=1., steps=10)
    cadence = DualCadence(slow_cadence, fast_cadence)
    obs = RasterSlit(axes=('ufast','vslow'), cadence=cadence, fov=fov,
                     path='SSB', frame='J2000')

    indices = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
    indices_ = indices.copy()
    indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
    indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

    # uvt() with remask == False
    (uv, time) = obs.uvt(indices)

    assert not np.any(uv.mask)
    assert not np.any(time.mask)
    assert time == [0, 100, 190, 10, 110, 200, 200]
    assert uv.to_scalar(0) == indices.to_scalar(0)
    assert uv.to_scalar(1) == 0.5

    # uvt() with remask == True
    (uv, time) = obs.uvt(indices, remask=True)

    assert np.all(uv.mask == np.array(6*[False] + [True]))
    assert np.all(time.mask == uv.mask)
    assert time[:6] == [0, 100, 190, 10, 110, 200]
    assert uv[:6].to_scalar(0) == indices[:6].to_scalar(0)
    assert uv[:6].to_scalar(1) == 0.5

    # uvt() with remask == True, new indices
    non_ints = indices + (0.2, 0.9)
    (uv, time) = obs.uvt(non_ints, remask=True)

    assert np.all(uv.mask == np.array(2*[False] + 5*[True]))
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
    assert time_min == [0, 100, 190,  9, 109, 199, 199]
    assert time_max == time_min + fast_cadence._texp

    # uvt_range() with remask == True
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints,
                                                         remask=True)

    assert Boolean(uv_min.mask) == 2*[False] + 5*[True]
    assert Boolean(uv_max.mask) == uv_min.mask
    assert Boolean(time_min.mask) == uv_min.mask
    assert Boolean(time_max.mask) == uv_min.mask

    assert uv_min.to_scalar(0)[:2] == indices.to_scalar(0)[:2]
    assert uv_min.to_scalar(1)[:2] == 0
    assert uv_max.to_scalar(0)[:2] == indices.to_scalar(0)[:2] + 1
    assert uv_max.to_scalar(1)[:2] == 1
    assert (time_min[:2]
            == (slow_cadence._tstride * indices.to_scalar(1)
                + fast_cadence._tstride * indices.to_scalar(0))[:2])
    assert time_max[:2] == time_min[:2] + fast_cadence._texp

    # time_range_at_uv() with remask == False
    uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])

    (time0, time1) = obs.time_range_at_uv(uv)

    assert time0 == [0, 190, 9, 199, 199]
    assert time1 == time0 + fast_cadence._texp

    # time_range_at_uv() with remask == True
    (time0, time1) = obs.time_range_at_uv(uv, remask=True)

    assert np.all(time0.mask == 4*[False] + [True])
    assert np.all(time1.mask == 4*[False] + [True])
    assert time0[:4] == [0, 190, 9, 199]
    assert time1[:4] == time0[:4] + fast_cadence._texp

    ######################################################################################
    # Alternative axis order ('uslow','vfast')
    ######################################################################################

    fov = FlatFOV((0.001,0.001), (1,20))
    slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
    fast_cadence = Metronome(tstart=0., tstride=0.5, texp=0.5, steps=20)
    cadence = DualCadence(slow_cadence, fast_cadence)
    obs = RasterSlit(axes=('uslow','vfast'),
                     cadence=cadence, fov=fov, path='SSB', frame='J2000')

    indices = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
    indices_ = indices.copy()
    indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
    indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

    (uv, time) = obs.uvt(indices)

    assert uv.to_scalar(0) == 0.5
    assert uv.to_scalar(1) == indices.to_scalar(1)
    assert time == [0, 5, 10, 90, 95, 100, 100]

    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert uv_min.to_scalar(0) == 0
    assert uv_min.to_scalar(1) == indices_.to_scalar(1)
    assert uv_max.to_scalar(0) == 1
    assert uv_max.to_scalar(1) == indices_.to_scalar(1) + 1
    assert time_min == [0, 5, 9.5, 90, 95, 99.5, 99.5]
    assert time_max == time_min + fast_cadence._texp

    (time0, time1) = obs.time_range_at_uv(indices)

    assert time0 == time_min
    assert time1 == time0 + fast_cadence._texp

    ######################################################################################
    # Alternative texp for discontinuous indices
    ######################################################################################

    fov = FlatFOV((0.001,0.001), (10,1))
    slow_cadence = Metronome(tstart=0., tstride=10., texp=8., steps=20)
    fast_cadence = Metronome(tstart=0., tstride=1., texp=0.5, steps=10)
    cadence = DualCadence(slow_cadence, fast_cadence)
    obs = RasterSlit(axes=('ufast','vslow'), cadence=cadence, fov=fov,
                                             path='SSB', frame='J2000')

    assert obs.time[1] == 199.5

    assert obs.uvt((0,0))[1] == 0.
    assert obs.uvt((5,0))[1] == 5.
    assert obs.uvt((5,5))[1] == 55.
    assert obs.uvt((5.0, 5.5))[1] == 55.
    assert obs.uvt((5.5, 5.0))[1] == 55.25

    eps = 1.e-15
    delta = 1.e-13
    assert abs(obs.uvt((6.     ,0))[1] - 6.000) < delta
    assert abs(obs.uvt((6.25   ,0))[1] - 6.125) < delta
    assert abs(obs.uvt((6.5    ,0))[1] - 6.250) < delta
    assert abs(obs.uvt((6.75   ,0))[1] - 6.375) < delta
    assert abs(obs.uvt((7 - eps,0))[1] - 6.500) < delta
    assert abs(obs.uvt((7.     ,0))[1] - 7.000) < delta

    assert obs.uvt((0,0))[0] == (0.,0.5)
    assert obs.uvt((5,0))[0] == (5.,0.5)
    assert obs.uvt((5,5))[0] == (5.,0.5)

    assert abs(obs.uvt((6.     ,0))[0] - (6.0,0.5)) < delta
    assert abs(obs.uvt((6.2    ,1))[0] - (6.2,0.5)) < delta
    assert abs(obs.uvt((6.4    ,2))[0] - (6.4,0.5)) < delta
    assert abs(obs.uvt((6.6    ,3))[0] - (6.6,0.5)) < delta
    assert abs(obs.uvt((6.8    ,4))[0] - (6.8,0.5)) < delta
    assert abs(obs.uvt((7.     ,6))[0] - (7.0,0.5)) < delta

    assert abs(obs.uvt((1, 0      ))[0] - (1.,0.5)) < delta
    assert abs(obs.uvt((2, 1.25   ))[0] - (2.,0.5)) < delta
    assert abs(obs.uvt((3, 2.5    ))[0] - (3.,0.5)) < delta
    assert abs(obs.uvt((4, 3.75   ))[0] - (4.,0.5)) < delta
    assert abs(obs.uvt((5, 5 - eps))[0] - (5.,0.5)) < delta
    assert abs(obs.uvt((5, 5.     ))[0] - (5.,0.5)) < delta

    ######################################################################################
    # Alternative tstride for even more discontinuous indices
    ######################################################################################

    fov = FlatFOV((0.001,0.001), (10,1))
    slow_cadence = Metronome(tstart=0., tstride=11., texp=10., steps=20)
    fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
    cadence = DualCadence(slow_cadence, fast_cadence)
    obs = RasterSlit(axes=('ufast','vslow'), cadence=cadence, fov=fov,
                                             path='SSB', frame='J2000')

    assert obs.time[1] == 218.8

    assert obs.uvt((0,0))[1] == 0.
    assert obs.uvt((5,0))[1] == 5.
    assert obs.uvt((5,5))[1] == 60.
    assert obs.uvt((5.0, 5.5))[1] == 60.
    assert obs.uvt((5.5, 5.0))[1] == 60.4
    assert obs.uvt((5.5, 5.5))[1] == 60.4

    eps = 1.e-14
    delta = 1.e-13
    assert abs(obs.uvt((6      ,0))[1] - 6. ) < delta
    assert abs(obs.uvt((6.25   ,0))[1] - 6.2) < delta
    assert abs(obs.uvt((6.5    ,0))[1] - 6.4) < delta
    assert abs(obs.uvt((6.75   ,0))[1] - 6.6) < delta
    assert abs(obs.uvt((7 - eps,0))[1] - 6.8) < delta
    assert abs(obs.uvt((7.     ,0))[1] - 7.0) < delta

    assert abs(obs.uvt((9       ,0))[1] -  9. ) < delta
    assert abs(obs.uvt((9.25    ,0))[1] -  9.2) < delta
    assert abs(obs.uvt((9.5     ,0))[1] -  9.4) < delta
    assert abs(obs.uvt((9.75    ,0))[1] -  9.6) < delta
    assert abs(obs.uvt((10 - eps,0))[1] -  9.8) < delta
    assert abs(obs.uvt((0.      ,1))[1] - 11. ) < delta

    assert abs(obs.uvt((6.00, 0.   ))[1] -  6. ) < delta
    assert abs(obs.uvt((6.25, 0.   ))[1] -  6.2) < delta
    assert abs(obs.uvt((6.25, 1.   ))[1] - 17.2) < delta
    assert abs(obs.uvt((6.25, 2-eps))[1] - 17.2) < delta
    assert abs(obs.uvt((6.25, 2    ))[1] - 28.2) < delta

    # Test the upper edge
    pair = (10-eps, 0)
    assert (obs.uvt(pair, remask=True)[0] - (10, 0.5)).rms() < delta
    assert (obs.uvt(pair, remask=True)[1] - 9.8).abs() < delta
    assert not obs.uvt(pair, remask=True)[0].mask

    pair = (10, 0)
    assert (obs.uvt(pair, remask=True)[0] - (10, 0.5)).rms() < delta
    assert (obs.uvt(pair, remask=True)[1] - 9.8).abs() < delta
    assert not obs.uvt(pair, remask=True)[0].mask

    pair = (10+eps, 0)
    assert obs.uvt(pair, remask=True)[0].mask

    pair = (10-eps, 1-eps)
    assert (obs.uvt(pair, remask=True)[0] - (10, 0.5)).rms() < delta
    assert (obs.uvt(pair, remask=True)[1] - 9.8).abs() < delta
    assert not obs.uvt(pair, remask=True)[0].mask

    pair = (10, 1)
    assert (obs.uvt(pair, remask=True)[0] - (10, 0.5)).rms() < delta
    assert (obs.uvt(pair, remask=True)[1] - 20.8).abs() < delta
    assert not obs.uvt(pair, remask=True)[0].mask

    pair = (10, 20)
    assert (obs.uvt(pair, remask=True)[0] - (10, 0.5)).rms() < delta
    assert (obs.uvt(pair, remask=True)[1] - 218.8).abs() < delta
    assert not obs.uvt(pair, remask=True)[0].mask

    pair = (10, 20+eps)
    assert obs.uvt(pair, remask=True)[0].mask

    ######################################################################################
    # Alternative, discontinuous and weird axes
    ######################################################################################

    fov = FlatFOV((0.001,0.001), (10,1))
    slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
    fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
    cadence = DualCadence(slow_cadence, fast_cadence)
    obs = RasterSlit(axes=('a','vslow','b','ufast','c'),
                     cadence=cadence, fov=fov, path='SSB', frame='J2000')

    assert obs.time[1] == 199.8

    assert obs.uvt((1,0,3,0,4))[1] == 0.
    assert obs.uvt((1,0,3,5,4))[1] == 5.
    assert obs.uvt((1,0,3,5.5,4))[1] == 5.4

    eps = 1.e-15
    delta = 1.e-13
    assert abs(obs.uvt((1,0,0,6      ,0))[1] - 6. ) < delta
    assert abs(obs.uvt((1,0,0,6.25   ,0))[1] - 6.2) < delta
    assert abs(obs.uvt((1,0,0,6.5    ,0))[1] - 6.4) < delta
    assert abs(obs.uvt((1,0,0,6.75   ,0))[1] - 6.6) < delta
    assert abs(obs.uvt((1,0,0,7 - eps,0))[1] - 6.8) < delta
    assert abs(obs.uvt((1,0,0,7.     ,0))[1] - 7.0) < delta

    assert obs.uvt((0,0,0,0,0))[0] == (0.,0.5)
    assert obs.uvt((0,0,0,5,0))[0] == (5.,0.5)
    assert obs.uvt((0,5,0,5,0))[0] == (5.,0.5)

    assert abs(obs.uvt((1,0,4,6      ,7))[0] - (6.0,0.5)) < delta
    assert abs(obs.uvt((1,1,4,6.2    ,7))[0] - (6.2,0.5)) < delta
    assert abs(obs.uvt((1,2,4,6.4    ,7))[0] - (6.4,0.5)) < delta
    assert abs(obs.uvt((1,3,4,6.6    ,7))[0] - (6.6,0.5)) < delta
    assert abs(obs.uvt((1,4,4,6.8    ,7))[0] - (6.8,0.5)) < delta
    assert abs(obs.uvt((1,6,4,7.     ,7))[0] - (7.0,0.5)) < delta

    assert abs(obs.uvt((1, 0      ,4,1,7))[0] - (1.,0.5)) < delta
    assert abs(obs.uvt((1, 1.25   ,4,2,7))[0] - (2.,0.5)) < delta
    assert abs(obs.uvt((1, 2.5    ,4,3,7))[0] - (3.,0.5)) < delta
    assert abs(obs.uvt((1, 3.75   ,4,4,7))[0] - (4.,0.5)) < delta
    assert abs(obs.uvt((1, 5 - eps,4,5,7))[0] - (5.,0.5)) < delta
    assert abs(obs.uvt((1, 5.     ,4,5,7))[0] - (5.,0.5)) < delta

    ######################################################################################
    # Old Pushframe unit tests
    ######################################################################################

    Pushframe = TimedImage

    flatfov = FlatFOV((0.001,0.001), (10,20))
    cadence = TDICadence(lines=20, tstart=100., tdi_texp=10., tdi_stages=2,
                         tdi_sign=-1)
    obs = Pushframe(axes=('u','vt'),
                    cadence=cadence, fov=flatfov, path='SSB', frame='J2000')

    indices = Vector([( 0,0),( 0,1),( 0,10),( 0,18),( 0,19),( 0,20),( 0,21),
                      (10,0),(10,1),(10,10),(10,18),(10,19),(10,20),(10,21)])
    tstep = indices.to_scalar(1)

    # uvt() with remask == False
    (uv,time) = obs.uvt(indices)

    assert not np.any(uv.mask)
    assert not np.any(time.mask)
    assert uv == Pair.as_pair(indices)
    assert time == 2*[100,100,100,100,110,120,120]

    # uvt() with remask == True
    (uv,time) = obs.uvt(indices, remask=True)

    assert np.all(uv.mask == np.array(2*(6*[False]+[True])))
    assert np.all(time.mask == uv.mask)
    assert time == cadence.time_at_tstep(tstep, remask=True)
    assert uv[:6] == Pair.as_pair(indices)[:6]

    # uvt_range() with remask == False
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert not np.any(uv_min.mask)
    assert not np.any(uv_max.mask)
    assert not np.any(time_min.mask)
    assert not np.any(time_max.mask)

    assert (uv_min
            == [(0,0),(0,1),(0,10),(0,18),(0,19),(0,19),(0,21),
                (9,0),(9,1),(9,10),(9,18),(9,19),(9,19),(9,21)])
    assert uv_max == uv_min + (1,1)
    assert time_min == cadence.time_range_at_tstep(tstep)[0]
    assert time_max == cadence.time_range_at_tstep(tstep)[1]

    # uvt_range() with remask == False, new indices
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9))

    assert not np.any(uv_min.mask)
    assert not np.any(uv_max.mask)
    assert not np.any(time_min.mask)
    assert not np.any(time_max.mask)

    assert uv_min == Pair.as_pair(indices)
    assert uv_max == Pair.as_pair(indices) + (1,1)
    assert time_min == cadence.time_range_at_tstep(tstep)[0]
    assert time_max == cadence.time_range_at_tstep(tstep)[1]

    # uvt_range() with remask == True, new indices
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9),
                                                         remask=True)

    assert np.all(uv_min.mask == np.array(5*[False] + 9*[True]))
    assert np.all(uv_max.mask == uv_min.mask)
    assert np.all(time_min.mask == uv_min.mask)
    assert np.all(time_max.mask == uv_min.mask)

    assert uv_min[:2] == Pair.as_pair(indices)[:2]
    assert uv_max[:2] == Pair.as_pair(indices)[:2] + (1,1)
    assert time_min[:2] == cadence.time_range_at_tstep(tstep)[0][:2]
    assert time_max[:2] == cadence.time_range_at_tstep(tstep)[1][:2]

    # time_range_at_uv() with remask == False
    uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])
    tstep = uv.to_scalar(1)

    (time0, time1) = obs.time_range_at_uv(uv)
    assert time0 == cadence.time_range_at_tstep(tstep)[0]
    assert time1 == cadence.time_range_at_tstep(tstep)[1]

    # time_range_at_uv() with remask == True
    (time0, time1) = obs.time_range_at_uv(uv, remask=True)

    assert np.all(time0.mask == 4*[False] + [True])
    assert np.all(time1.mask == 4*[False] + [True])
    assert time0[:4] == cadence.time_range_at_tstep(tstep)[0][:4]
    assert time1[:4] == cadence.time_range_at_tstep(tstep)[1][:4]

    # Alternative axis order ('ut','v')
    cadence = TDICadence(lines=10, tstart=100., tdi_texp=10., tdi_stages=10,
                         tdi_sign=-1)
    obs = Pushframe(axes=('ut','v'),
                    cadence=cadence, fov=flatfov, path='SSB', frame='J2000')

    indices = Vector([(-1,0),(0,-1),(0,0),(0,20),(9,0),(10,0),(11,0),(11,20)])
    tstep = indices.to_scalar(0)

    (uv,time) = obs.uvt(indices)

    assert uv == Pair.as_pair(indices)
    assert time == cadence.time_at_tstep(tstep)

    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert uv_min == [(-1,0),(0,-1),(0,0),(0,19),(9,0),(9,0),(11,0),(11,19)]
    assert uv_max == uv_min + (1,1)
    assert time_min == cadence.time_range_at_tstep(tstep)[0]
    assert time_max == cadence.time_range_at_tstep(tstep)[1]

    (time0,time1) = obs.time_range_at_uv(indices)

    assert time0 == cadence.time_range_at_tstep(tstep)[0]
    assert time1 == cadence.time_range_at_tstep(tstep)[1]

    # Alternative texp for discontinuous indices
    cadence = TDICadence(lines=10, tstart=100., tdi_texp=10., tdi_stages=10,
                                   tdi_sign=1)
    obs = Pushframe(axes=('ut','v'),
                    cadence=cadence, fov=flatfov, path='SSB', frame='J2000')

    assert obs.time[0] == 100.

    assert obs.uvt((-1,0))[0] == (-1,0)
    assert obs.uvt(( 0,0))[0] == ( 0,0)
    assert obs.uvt(( 5,0))[0] == ( 5,0)
    assert obs.uvt(( 5,5))[0] == ( 5,5)
    assert obs.uvt(( 9,5))[0] == ( 9,5)
    assert obs.uvt((9.5,5))[0] == (9.5,5)
    assert obs.uvt((10,5))[0] == (10,5)

    assert obs.uvt((-1,0))[1] == 190.
    assert obs.uvt(( 0,0))[1] == 190.
    assert obs.uvt(( 5,0))[1] == 140.
    assert obs.uvt(( 5,5))[1] == 140.
    assert obs.uvt(( 9,5))[1] == 100.
    assert obs.uvt((9.5,5))[1] == 150.
    assert obs.uvt((10,5))[1] == 200.
##########################################################################################
