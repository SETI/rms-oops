##########################################################################################
# tests/observation/test_rasterslit1d.py
##########################################################################################

import numpy as np
import pytest

from polymath         import Scalar, Pair
from oops.cadence     import Metronome
from oops.fov         import FlatFOV
from oops.observation import RasterSlit1D


def test_rasterslit1d():
    ######################################################################################
    # Continuous 2-D observation
    # First axis = U and T with length 10
    # Second axis ignored
    ######################################################################################

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
    assert time[:2] == cadence._tstride * indices.to_scalar(0)[:2]
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
    assert time_min[:2] == cadence._tstride*indices_.to_scalar(0)[:2]
    assert time_max[:2] == time_min[:2] + cadence._texp

    assert uv_min[2] == Pair.MASKED
    assert time_min[2] == Scalar.MASKED
    assert time_min[2] == Scalar.MASKED

    # time_range_at_uv() with remask == False
    uv = Pair([(0,0),(0,20),(10,0),(10,20),(11,21)])
    uv_ = uv.copy()
    uv_.vals[:,0][uv_.vals[:,0] == 10] -= 1

    (time0, time1) = obs.time_range_at_uv(uv)

    assert time0 == cadence.time_range_at_tstep(uv_.to_scalar(0))[0]
    assert time1 == time0 + cadence._texp

    # time_range_at_uv() with remask == True
    (time0, time1) = obs.time_range_at_uv(uv, remask=True)

    assert np.all(time0.mask == 4*[False] + [True])
    assert np.all(time1.mask == 4*[False] + [True])
    assert time0[:4] == cadence._tstride * uv_.to_scalar(0)[:4]
    assert time1[:4] == time0[:4] + cadence._texp

    ######################################################################################
    # Alternative axis order ('a', 'vt')
    # Second axis = V and T with length 10
    # Discontinuous time sampling [0-8], [10-18], ..., [90-98]
    # First axis ignored
    ######################################################################################

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
    assert time_max == time_min + cadence._texp

    uv = Pair([(11,0),(11,9),(11,10),(11,11)])
    uv_ = uv.copy()
    uv_.vals[:,1][uv_.vals[:,1] == 10] -= 1

    (time0, time1) = obs.time_range_at_uv(uv)

    assert time0 == cadence.time_range_at_tstep(uv_.to_scalar(1))[0]
    assert time1 == time0 + cadence._texp

    ######################################################################################
    # Similar to above but 1-D observation
    # First axis = V and T with length 10
    # Discontinuous time sampling [0-8], [10-18], ..., [90-98]
    ######################################################################################

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
    assert time_max == time_min + cadence._texp

    uv = Pair([(11,0),(11,9),(11,10),(11,11)])
    uv_ = uv.copy()
    uv_.vals[:,1][uv_.vals[:,1] == 10] -= 1

    (time0, time1) = obs.time_range_at_uv(uv)

    assert time0 == cadence.time_range_at_tstep(uv_.to_scalar(1))[0]
    assert time1 == time0 + cadence._texp

    ######################################################################################
    # Alternative axis order ('ut',), 1-D
    # First axis = U and T with length 10
    # Discontinuous time sampling [0-8], [10-18], ..., [90-98]
    ######################################################################################

    fov = FlatFOV((0.001,0.001), (10,1))
    cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
    obs = RasterSlit1D(axes=('ut',), cadence=cadence,
                       fov=fov, path='SSB', frame='J2000')

    assert obs.time[1] == 98.

    assert obs.uvt(0, remask=True)[1] == 0.
    assert obs.uvt(5, remask=True)[1] == 50.
    assert obs.uvt(5.5, remask=True)[1] == 54.
    assert obs.uvt(9.5, remask=True)[1] == 94.
    assert obs.uvt(10., remask=True)[1] == 98.
    assert obs.uvt(10.001, remask=True)[1].mask

    eps = 1.e-14
    delta = 1.e-13
    assert abs(obs.uvt((6.     ), remask=True)[0] - (6.0,0.5)) < delta
    assert abs(obs.uvt((6.2    ), remask=True)[0] - (6.2,0.5)) < delta
    assert abs(obs.uvt((6.4    ), remask=True)[0] - (6.4,0.5)) < delta
    assert abs(obs.uvt((6.6    ), remask=True)[0] - (6.6,0.5)) < delta
    assert abs(obs.uvt((6.8    ), remask=True)[0] - (6.8,0.5)) < delta
    assert abs(obs.uvt((7.     ), remask=True)[0] - (7.0,0.5)) < delta
    assert abs(obs.uvt((10     ), remask=True)[0] - (10.,0.5)) < delta
    assert obs.uvt(10.+eps, remask=True)[0].mask

    indices = Scalar([10-eps, 10, 10+eps])

    (uv,t) = obs.uvt(indices, remask=True)
    assert np.all(t.mask == np.array(2*[False] + [True]))

##########################################################################################
# Constructor validation, alternative cadence forms, subfields, and time_shift
##########################################################################################

SLIT_FOV = FlatFOV((0.001, 0.001), (10, 1))
CADENCE = Metronome(tstart=0., tstride=10., texp=10., steps=10)


def _rasterslit(**kwargs) -> RasterSlit1D:
    """A RasterSlit1D sweeping the u-axis of a ten-pixel slit.

    Parameters:
        kwargs: Overrides of the default constructor arguments.

    Returns:
        RasterSlit1D: The observation.
    """

    args = {'axes': ('ut', 'a'), 'cadence': CADENCE, 'fov': SLIT_FOV,
            'path': 'SSB', 'frame': 'J2000'}
    args.update(kwargs)

    return RasterSlit1D(**args)


@pytest.mark.parametrize('axes', [('u', 'a'), ('ut', 'vt'), ('ut', 't')],
                         ids=['neither', 'both', 'with-t'])
def test_the_axes_must_name_exactly_one_swept_axis(axes: tuple[str, ...]) -> None:
    """Exactly one axis is swept in time, and no axis is time alone."""

    with pytest.raises(ValueError, match='invalid axes for RasterSlit1D'):
        _rasterslit(axes=axes)


def test_the_cross_slit_axis_of_the_fov_must_be_one_pixel_wide() -> None:
    """A slit is one pixel across; a wider FOV is not a slit."""

    with pytest.raises(ValueError, match='cross-slit axis must have length 1'):
        _rasterslit(fov=FlatFOV((0.001, 0.001), (10, 2)))


def test_a_cadence_given_as_a_tuple_is_built_for_the_slit() -> None:
    """A tuple supplies the Metronome arguments other than the number of steps."""

    obs = _rasterslit(cadence=(0., 10.))

    assert obs.cadence.shape == (10,)
    assert obs.time == (0., 100.)


def test_a_cadence_given_as_a_dictionary_is_built_for_the_slit() -> None:
    """A dictionary supplies the same arguments by keyword."""

    obs = _rasterslit(cadence={'tstart': 0., 'texp': 10.})

    assert obs.cadence.shape == (10,)
    assert obs.time == (0., 100.)


def test_a_cadence_of_the_wrong_length_is_rejected() -> None:
    """The cadence has one step per pixel along the slit."""

    wrong = Metronome(tstart=0., tstride=10., texp=10., steps=9)

    with pytest.raises(ValueError, match='Cadence and FOV shapes'):
        _rasterslit(cadence=wrong)


def test_a_cadence_of_an_unusable_type_is_rejected() -> None:
    """Anything that is neither a Cadence nor its arguments is refused."""

    with pytest.raises(TypeError, match='Invalid cadence class: float'):
        _rasterslit(cadence=10.)


def test_a_subfield_becomes_an_attribute() -> None:
    """Optional keywords are inserted as subfields, and so as attributes."""

    obs = _rasterslit(data=Scalar([1., 2., 3.]))

    assert obs.data == Scalar([1., 2., 3.])
    assert obs.subfields['data'] == Scalar([1., 2., 3.])


def test_time_shift_moves_the_cadence_and_keeps_the_subfields() -> None:
    """A shifted observation is the same observation at a later time."""

    obs = _rasterslit(data=Scalar([1., 2., 3.]))

    shifted = obs.time_shift(100.)

    assert shifted.time == (100., 200.)
    assert shifted.shape == obs.shape
    assert shifted.data == obs.data


def test_a_slit_along_the_v_axis_is_swept_the_same_way() -> None:
    """"vt" sweeps the v-axis, so the FOV is one pixel wide in u instead."""

    obs = _rasterslit(axes=('a', 'vt'), fov=FlatFOV((0.001, 0.001), (1, 10)))

    assert obs.u_axis == -1
    assert obs.v_axis == 1
    assert obs.t_axis == 1
    assert obs.uv_shape == (1, 10)

##########################################################################################
