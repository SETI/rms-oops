##########################################################################################
# tests/observation/test_pixel.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Pair

from oops.cadence     import DualCadence, Metronome
from oops.fov         import FlatFOV
from oops.observation import Pixel


def test_pixel():
    fov = FlatFOV((0.001,0.001), (1,1))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
    obs = Pixel(axes=('t'),
                cadence=cadence, fov=fov, path='SSB', frame='J2000')

    indices = Scalar([(0,),(1,),(20,),(21,)])
    indices_ = indices.copy()
    indices_.vals[indices_.vals == 20] -= 1         # clip the top

    # uvt() with remask == False
    (uv, time) = obs.uvt(indices)

    assert not np.any(uv.mask)
    assert not np.any(time.mask)
    assert time == cadence.time_at_tstep(indices)
    assert uv == (0.5,0.5)

    # uvt() with remask == True
    (uv, time) = obs.uvt(indices, remask=True)

    assert np.all(uv.mask == np.array([3*[[False]] + [[True]]]))
    assert np.all(time.mask == uv.mask)
    assert time[:3] == cadence._tstride * indices[:3]
    assert uv[:3] == (0.5,0.5)

    # uvt_range() with remask == False
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert not np.any(uv_min.mask)
    assert not np.any(uv_max.mask)
    assert not np.any(time_min.mask)
    assert not np.any(time_max.mask)

    assert uv_min == (0,0)
    assert uv_max == (1,1)

    assert time_min == cadence.time_range_at_tstep(indices_)[0]
    assert time_max == time_min + cadence._texp

    # uvt_range() with remask == False, new indices
    non_ints = indices + 0.2
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints)

    assert not np.any(uv_min.mask)
    assert not np.any(uv_max.mask)
    assert not np.any(time_min.mask)
    assert not np.any(time_max.mask)

    assert uv_min == (0,0)
    assert uv_max == (1,1)

    assert time_min == cadence.time_range_at_tstep(non_ints)[0]
    assert time_max == time_min + cadence._texp

    # uvt_range() with remask == True, new indices
    non_ints = indices + 0.2
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints, remask=True)

    assert np.all(uv_min.mask == np.array(2*[[False]] + 2*[[True]]))
    assert np.all(uv_max.mask == uv_min.mask)
    assert np.all(time_min.mask == uv_min.mask)
    assert np.all(time_max.mask == uv_min.mask)

    assert uv_min[:2] == (0,0)
    assert uv_max[:2] == (1,1)
    assert time_min[:2] == indices[:2] * cadence._tstride
    assert time_max[:2] == time_min[:2] + cadence._texp

    # time_range_at_uv() with remask == False
    uv = Pair([(0,0),(0,1),(1,0),(1,1),(1,2)])

    (time0, time1) = obs.time_range_at_uv(uv)

    assert time0 == obs.time[0]
    assert time1 == obs.time[1]

    # time_range_at_uv() with remask == True
    (time0, time1) = obs.time_range_at_uv(uv, remask=True)

    assert np.all(time0.mask == 4*[False] + [True])
    assert np.all(time1.mask == time0.mask)
    assert time0[:4] == obs.time[0]
    assert time1[:4] == obs.time[1]

    ######################################################################################

    # Alternative axis order ('a','t')

    fov = FlatFOV((0.001,0.001), (1,1))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
    obs = Pixel(axes=('a','t'),
                cadence=cadence, fov=fov, path='SSB', frame='J2000')

    indices = Pair([(0,0),(1,1),(0,20,),(1,21)])
    indices_ = indices.copy()
    indices_.vals[indices_.vals == 20] -= 1         # clip the top

    # uvt() with remask == False
    (uv,time) = obs.uvt(indices)

    assert not uv.mask
    assert not np.any(time.mask)
    assert time.without_mask() == cadence.time_at_tstep(indices.to_scalar(1))
    assert uv == (0.5,0.5)

    # uvt() with remask == True
    (uv,time) = obs.uvt(indices, remask=True)

    assert np.all(uv.mask == np.array(3*[False] + [True]))
    assert np.all(time.mask == uv.mask)
    assert time[:3] == cadence._tstride * indices[:3].to_scalar(1)
    assert uv[:3] == (0.5,0.5)

    # uvt_range() with remask == False
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert not uv_min.mask
    assert not uv_max.mask
    assert not time_min.mask
    assert not time_max.mask

    assert uv_min == (0,0)
    assert uv_max == (1,1)

    assert time_min == cadence.time_range_at_tstep(indices.to_scalar(1))[0]
    assert time_max == time_min + cadence._texp

    # uvt_range() with remask == False, new indices
    non_ints = indices + (0.2,0.9)
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints)

    assert not uv_min.mask
    assert not uv_max.mask
    assert not time_min.mask
    assert not time_max.mask

    assert uv_min == (0,0)
    assert uv_max == (1,1)

    assert time_min == cadence.time_range_at_tstep(indices.to_scalar(1))[0]
    assert time_max == time_min + cadence._texp

    # uvt_range() with remask == True, new indices
    non_ints = indices + (0.2,0.2)
    (uv_min, uv_max, time_min,
                     time_max) = obs.uvt_range(non_ints, remask=True)

    assert np.all(uv_min.mask == np.array(2*[False] + 2*[True]))
    assert np.all(uv_max.mask == uv_min.mask)
    assert np.all(time_min.mask == uv_min.mask)
    assert np.all(time_max.mask == uv_min.mask)

    assert uv_min[:2] == (0,0)
    assert uv_max[:2] == (1,1)
    assert uv_min[2:] == Pair.MASKED
    assert uv_max[2:] == Pair.MASKED

    assert time_min[:2] == indices.to_scalar(1)[:2] * cadence._tstride
    assert time_max[:2] == time_min[:2] + cadence._texp

    # time_range_at_uv() with remask == False
    uv = Pair([(0,0),(0,1),(1,0),(1,1),(1,2)])

    (time0, time1) = obs.time_range_at_uv(uv)
    assert time0 == obs.time[0]
    assert time1 == obs.time[1]

    # time_range_at_uv() with remask == True
    (time0, time1) = obs.time_range_at_uv(uv, remask=True)

    assert np.all(time0.mask == 4*[False] + [True])
    assert np.all(time1.mask == time0.mask)
    assert time0[:4] == obs.time[0]
    assert time1[:4] == obs.time[1]


def test_pixel_event_at_grid():
    fov = FlatFOV((0.001,0.001), (1,1))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=4)
    obs = Pixel(axes=('t'),
                cadence=cadence, fov=fov, path='SSB', frame='J2000')
    meshgrid = obs.meshgrid()

    # One time per sample of the cadence, on a leading axis
    event = obs.event_at_grid(meshgrid)

    assert event.shape == (4, 1)
    assert event.time.flatten() == Scalar([5., 15., 25., 35.])
    assert event.neg_arr_ap.shape == (1,)

    # tfrac selects the point within each sample's exposure
    event = obs.event_at_grid(meshgrid, tfrac=0.)
    assert event.time.flatten() == Scalar([0., 10., 20., 30.])

    event = obs.event_at_grid(meshgrid, tfrac=1.)
    assert event.time.flatten() == Scalar([10., 20., 30., 40.])

    # An explicit time overrides the cadence and tfrac
    event = obs.event_at_grid(meshgrid, time=Scalar(7.))
    assert event.time == Scalar(7.)


def test_pixel_gridless_event():
    fov = FlatFOV((0.001,0.001), (1,1))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=4)
    obs = Pixel(axes=('t'),
                cadence=cadence, fov=fov, path='SSB', frame='J2000')
    meshgrid = obs.meshgrid()

    event = obs.gridless_event(meshgrid)

    assert event.shape == (4, 1)
    assert event.time.flatten() == Scalar([5., 15., 25., 35.])

    # shapeless collapses to the mean of the times
    event = obs.gridless_event(meshgrid, shapeless=True)

    assert event.shape == ()
    assert event.time == Scalar(20.)

    # An explicit time overrides the cadence and tfrac
    event = obs.gridless_event(meshgrid, time=Scalar(7.))
    assert event.time == Scalar(7.)

    # Passing tfrac=None alongside a time is also accepted
    event = obs.gridless_event(meshgrid, tfrac=None, time=Scalar(7.))
    assert event.time == Scalar(7.)


def test_pixel_gridless_event_without_a_meshgrid():
    fov = FlatFOV((0.001,0.001), (1,1))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=4)
    obs = Pixel(axes=('t'),
                cadence=cadence, fov=fov, path='SSB', frame='J2000')

    # The meshgrid only shapes the event, so omitting it leaves the cadence times alone
    event = obs.gridless_event()

    assert event.shape == (4,)
    assert event.time == Scalar([5., 15., 25., 35.])


def test_pixel_gridless_event_without_a_meshgrid_is_shapeless():
    fov = FlatFOV((0.001,0.001), (1,1))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=4)
    obs = Pixel(axes=('t'),
                cadence=cadence, fov=fov, path='SSB', frame='J2000')

    event = obs.gridless_event(shapeless=True)

    assert event.time == Scalar(20.)


def test_pixel_uvt_accepts_a_number_without_a_time_axis():
    """`_scalar_from_indices` accepts a number, so `uvt` must too."""

    fov = FlatFOV((0.001,0.001), (1,1))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=4)
    obs = Pixel(axes=('u'),
                cadence=cadence, fov=fov, path='SSB', frame='J2000')

    (uv, time) = obs.uvt(0)

    assert uv == Pair((0.5,0.5))


def test_pixel_uvt_returns_the_midtime_without_a_time_axis():
    fov = FlatFOV((0.001,0.001), (1,1))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=4)
    obs = Pixel(axes=('u'),
                cadence=cadence, fov=fov, path='SSB', frame='J2000')

    (uv, time) = obs.uvt(0)

    assert time == Scalar(cadence.midtime)


def test_pixel_uvt_accepts_a_list_of_index_components():
    """A flat list is one index vector, so the (u,v) result is shapeless."""

    fov = FlatFOV((0.001,0.001), (1,1))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=4)
    obs = Pixel(axes=('u','v'),
                cadence=cadence, fov=fov, path='SSB', frame='J2000')

    (uv, time) = obs.uvt([0, 0])

    assert uv.shape == ()


def test_pixel_uvt_range_accepts_a_number():
    """`uvt_range` reads the same raw `indices` argument that `uvt` does."""

    fov = FlatFOV((0.001,0.001), (1,1))
    cadence = Metronome(tstart=0., tstride=10., texp=10., steps=4)
    obs = Pixel(axes=('t'),
                cadence=cadence, fov=fov, path='SSB', frame='J2000')

    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(0)

    assert uv_min == Pair((0,0))

##########################################################################################
# Constructor validation, an untimed pixel, subfields, and time_shift
##########################################################################################

PIXEL_FOV = FlatFOV((0.001, 0.001), (1, 1))
CADENCE = Metronome(tstart=0., tstride=10., texp=10., steps=5)


def _pixel(**kwargs) -> Pixel:
    """A Pixel observed at five consecutive time steps.

    Parameters:
        kwargs: Overrides of the default constructor arguments.

    Returns:
        Pixel: The observation.
    """

    args = {'axes': ('t',), 'cadence': CADENCE, 'fov': PIXEL_FOV,
            'path': 'SSB', 'frame': 'J2000'}
    args.update(kwargs)

    return Pixel(**args)


def test_the_fov_must_be_a_single_pixel() -> None:
    """A Pixel observation sees exactly one pixel."""

    with pytest.raises(ValueError, match=r'FOV must have shape \(1,1\)'):
        _pixel(fov=FlatFOV((0.001, 0.001), (2, 1)))


def test_the_cadence_must_be_one_dimensional() -> None:
    """The one axis of a Pixel is time, so its cadence has one dimension."""

    two_d = DualCadence(Metronome(tstart=0., tstride=50., texp=50., steps=2), CADENCE)

    with pytest.raises(ValueError, match='requires a 1-D cadence'):
        _pixel(cadence=two_d)


def test_a_pixel_without_a_time_axis_is_shapeless() -> None:
    """Axes that do not include "t" leave the observation with no time index."""

    obs = _pixel(axes=('a',))

    assert obs.t_axis == -1
    assert obs.shape == (0,)


def test_an_untimed_pixel_spans_the_whole_cadence() -> None:
    """With no time index, every index covers the pixel and the full time range."""

    obs = _pixel(axes=('a',))

    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(Scalar(0.))

    assert uv_min == Pair((0, 0))
    assert uv_max == Pair((1, 1))
    assert time_min == Scalar(0.)
    assert time_max == Scalar(50.)


def test_a_subfield_becomes_an_attribute() -> None:
    """Optional keywords are inserted as subfields, and so as attributes."""

    obs = _pixel(data=Scalar([1., 2., 3.]))

    assert obs.data == Scalar([1., 2., 3.])
    assert obs.subfields['data'] == Scalar([1., 2., 3.])


def test_time_shift_moves_the_cadence_and_keeps_the_subfields() -> None:
    """A shifted observation is the same observation at a later time."""

    obs = _pixel(data=Scalar([1., 2., 3.]))

    shifted = obs.time_shift(100.)

    assert shifted.time == (100., 150.)
    assert shifted.shape == obs.shape
    assert shifted.data == obs.data

##########################################################################################
