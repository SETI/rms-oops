##########################################################################################
# tests/observation/test_slit1d.py
##########################################################################################

import numpy as np
import pytest

from polymath         import Pair, Scalar, Vector
from oops.cadence     import Metronome
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
# Constructor validation, a cadence in place of a start time, subfields, and time_shift
##########################################################################################

SLIT_FOV = FlatFOV((0.001, 0.001), (20, 1))


def _slit(**kwargs) -> Slit1D:
    """A Slit1D along the u-axis of a twenty-pixel slit.

    Parameters:
        kwargs: Overrides of the default constructor arguments.

    Returns:
        Slit1D: The observation.
    """

    args = {'axes': ('u',), 'tstart': 0., 'texp': 10., 'fov': SLIT_FOV,
            'path': 'SSB', 'frame': 'J2000'}
    args.update(kwargs)

    return Slit1D(**args)


@pytest.mark.parametrize('axes', [('a',), ('u', 'v')], ids=['neither', 'both'])
def test_the_axes_must_name_exactly_one_spatial_axis(axes: tuple[str, ...]) -> None:
    """A slit runs along one axis, so exactly one of "u" and "v" appears."""

    with pytest.raises(ValueError, match='axes are incompatible with Slit1D'):
        _slit(axes=axes)


def test_the_cross_slit_axis_of_the_fov_must_be_one_pixel_wide() -> None:
    """A slit is one pixel across; a wider FOV is not a slit."""

    with pytest.raises(ValueError, match='cross-slit FOV axis must have length 1'):
        _slit(fov=FlatFOV((0.001, 0.001), (20, 2)))


def test_a_slit_along_the_v_axis_runs_the_other_way() -> None:
    """"v" puts the slit on the v-axis, so the FOV is one pixel wide in u instead."""

    obs = _slit(axes=('v',), fov=FlatFOV((0.001, 0.001), (1, 20)))

    assert obs.u_axis == -1
    assert obs.v_axis == 0
    assert tuple(obs.shape) == (20,)


def test_a_cadence_can_stand_in_for_the_start_time() -> None:
    """A one-step Cadence defines both the start time and the exposure."""

    obs = _slit(tstart=Metronome(tstart=5., tstride=20., texp=20., steps=1), texp=None)

    assert obs.time == (5., 25.)
    assert obs._texp == 20.


def test_a_cadence_of_more_than_one_step_is_rejected() -> None:
    """Every pixel of a Slit1D is exposed at once, so the cadence has one step."""

    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=2)

    with pytest.raises(ValueError, match="cadence must be \\(1,\\)"):
        _slit(tstart=cadence)


def test_a_subfield_becomes_an_attribute() -> None:
    """Optional keywords are inserted as subfields, and so as attributes."""

    obs = _slit(data=Scalar([1., 2., 3.]))

    assert obs.data == Scalar([1., 2., 3.])
    assert obs.subfields['data'] == Scalar([1., 2., 3.])


def test_time_shift_moves_the_exposure_and_keeps_the_subfields() -> None:
    """A shifted observation is the same observation at a later time."""

    obs = _slit(data=Scalar([1., 2., 3.]))

    shifted = obs.time_shift(100.)

    assert shifted.time == (100., 110.)
    assert shifted.shape == obs.shape
    assert shifted.data == obs.data

##########################################################################################
