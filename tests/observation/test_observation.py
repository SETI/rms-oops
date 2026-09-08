##########################################################################################
# tests/observation/test_observation.py
##########################################################################################

import pickle

import numpy as np
import pytest

from oops             import mutable
from oops.cadence     import DualCadence, Metronome
from oops.fov         import FlatFOV
from oops.frame       import Frame, Navigation
from oops.observation import (InSitu, Observation, Pixel, RasterSlit1D, Slit1D,
                              Snapshot, TimedImage)
from oops.path        import Path
from polymath         import Matrix3, Pair, Scalar


def _observations() -> dict[str, Observation]:
    """One instance of every concrete Observation subclass, keyed by name."""

    image_fov = FlatFOV((0.001,0.001), (10,20))
    pixel_fov = FlatFOV((0.001,0.001), (1,1))
    slit_fov  = FlatFOV((0.001,0.001), (10,1))
    sweep_fov = FlatFOV((0.001,0.001), (5,1))

    steps = Metronome(tstart=0., tstride=10., texp=10., steps=5)
    rows  = Metronome(tstart=0., tstride=1., texp=1., steps=20)
    slow  = Metronome(tstart=0., tstride=20., texp=20., steps=10)

    return {
        'Snapshot':     Snapshot(('u','v'), 0., 10., image_fov, 'SSB', 'J2000'),
        'Pixel':        Pixel(('t',), steps, pixel_fov, 'SSB', 'J2000'),
        'Slit1D':       Slit1D(('u',), 0., 10., slit_fov, 'SSB', 'J2000'),
        'RasterSlit1D': RasterSlit1D(('ut',), steps, sweep_fov, 'SSB', 'J2000'),
        'TimedImage':   TimedImage(('u','vt'), rows, image_fov, 'SSB', 'J2000'),
        'TimedImage2D': TimedImage(('uslow','vfast'), DualCadence(slow, rows),
                                   image_fov, 'SSB', 'J2000'),
        'InSitu':       InSitu(steps, 'SSB'),
    }


@pytest.mark.parametrize('name', sorted(_observations()))
def test_observation_survives_a_pickle_round_trip(name: str) -> None:
    obs = _observations()[name]

    restored = pickle.loads(pickle.dumps(obs))

    assert type(restored) is type(obs)
    assert restored.shape == obs.shape
    assert restored.time == obs.time
    assert restored.uv_shape == obs.uv_shape
    assert restored.t_axis == obs.t_axis

    # The SSB path and J2000 frame are singletons, so the round trip must return the
    # very same objects; the registries and the `is` tests in oops depend on it
    assert restored.path is Path.SSB
    assert restored.frame is Frame.J2000


def _parallel_pair() -> tuple[Snapshot, Snapshot]:
    """Two observations sharing an origin and a time, for the parallel_* methods.

    Returns:
        tuple[Snapshot, Snapshot]: This observation and a parallel one. They share a
        frame and FOV, so a pointing offset maps to the identical offset.
    """

    fov = FlatFOV((1.e-4, 1.e-4), (64, 64))
    return (Snapshot(('u','v'), 0., 10., fov, 'SSB', 'J2000'),
            Snapshot(('u','v'), 0., 10., fov, 'SSB', 'J2000'))


def test_parallel_offset_duv_maps_an_offset_through_an_identical_fov() -> None:
    """A parallel observation with the same frame and FOV sees the same pixel offset."""

    (obs, parallel) = _parallel_pair()

    assert obs.parallel_offset_duv(parallel, Pair((1., 2.))) == Pair((1., 2.))


def test_parallel_offset_duv_assumes_the_midtime() -> None:
    """Omitting the time gives the same answer as passing the midtime, as documented."""

    (obs, parallel) = _parallel_pair()
    duv = Pair((1., 2.))

    assert obs.parallel_offset_duv(parallel, duv) \
           == obs.parallel_offset_duv(parallel, duv, time=obs.midtime)


def test_parallel_offset_duv_measures_from_the_given_origin() -> None:
    """The origin selects the reference point the offset is measured from."""

    (obs, parallel) = _parallel_pair()
    duv = Pair((1., 2.))

    from_center = obs.parallel_offset_duv(parallel, duv)
    from_corner = obs.parallel_offset_duv(parallel, duv, origin=Pair((10., 20.)))

    assert from_center == Pair((1., 2.))
    assert from_corner != from_center           # a different reference, a different map


@pytest.mark.parametrize('time', [None, 5.])
def test_parallel_offset_duv_inverts_the_fov_offset_angles(time: float | None) -> None:
    """The method is the composition its docstring describes, at any time."""

    (obs, parallel) = _parallel_pair()
    duv = Pair((1., 2.))
    at = obs.midtime if time is None else time

    angles = obs.fov.offset_angles_from_duv(duv, time=at)
    angles = obs.parallel_offset_angles(parallel, angles, time=at)
    expected = parallel.fov.offset_duv_from_angles(angles, time=at)

    assert obs.parallel_offset_duv(parallel, duv, time=time) == expected



@pytest.mark.parametrize('tfrac', [0.5, 0.25, Scalar(0.5), Scalar([0.5, 0.25])],
                         ids=['float-midtime', 'float-other', 'shapeless', 'array'])
def test_uv_from_ra_and_dec_accepts_any_shape_of_tfrac(tfrac) -> None:
    """A shapeless tfrac must work as well as an array one.

    Comparing a shapeless Scalar returns a Python bool rather than a Boolean, so the
    iteration count test has to convert the result before calling all() on it.
    """

    fov = FlatFOV((1.e-4, 1.e-4), (64, 64))
    obs = Snapshot(('u','v'), 0., 10., fov, 'SSB', 'J2000')

    # The FlatFOV axis is +Z, which is a declination of 90 degrees: the FOV center.
    uv = obs.uv_from_ra_and_dec(0., np.pi/2, tfrac=tfrac)

    assert float((uv - Pair((32., 32.))).norm().vals) == pytest.approx(0., abs=1.e-9)



def test_inventory_flag_marks_the_subclasses_that_implement_it() -> None:
    """The flag records which subclasses can answer inventory(); the base class cannot."""

    assert Observation._INVENTORY_IMPLEMENTED is False
    assert Snapshot._INVENTORY_IMPLEMENTED is True
    assert TimedImage._INVENTORY_IMPLEMENTED is True
    assert Pixel._INVENTORY_IMPLEMENTED is False


def test_a_timed_image_with_an_extended_fov_disowns_inventory() -> None:
    """A cadence longer than the FOV extends it, leaving inventory unable to answer.

    The instance value has to override the class default for this to work.
    """

    fov = FlatFOV((1.e-3, 1.e-3), (10, 20))
    plain = TimedImage(('u','vt'), cadence=Metronome(tstart=0., tstride=1., texp=1.,
                                                     steps=20),
                       fov=fov, path='SSB', frame='J2000')
    extended = TimedImage(('u','vt'), cadence=Metronome(tstart=0., tstride=1., texp=1.,
                                                        steps=25),
                          fov=fov, path='SSB', frame='J2000')

    # Truthiness, not identity: the comparison that sets _extended_fov can yield a
    # numpy bool, which is not the True or False singleton.
    assert not plain._extended_fov
    assert plain._INVENTORY_IMPLEMENTED

    assert extended._extended_fov
    assert not extended._INVENTORY_IMPLEMENTED



# The (u,v) range each subclass reports for its first two time steps, and the number of
# steps in its cadence.
UV_RANGE_AT_TSTEP = {
    'Snapshot':     ((( 0, 0), (10, 20)), (( 0, 0), (10, 20)),  1),
    'Pixel':        ((( 0, 0), ( 1,  1)), (( 0, 0), ( 1,  1)),  5),
    'Slit1D':       ((( 0, 0), (10,  1)), (( 0, 0), (10,  1)),  1),
    'RasterSlit1D': ((( 0, 0), ( 1,  1)), (( 1, 0), ( 2,  1)),  5),
    'TimedImage':   ((( 0, 0), (10,  1)), (( 0, 1), (10,  2)), 20),
}


@pytest.mark.parametrize('name', sorted(UV_RANGE_AT_TSTEP))
def test_uv_range_at_tstep_covers_the_pixels_of_each_step(name: str) -> None:
    """Each subclass reports the pixels its cadence exposes at a given time step."""

    (first, second, _) = UV_RANGE_AT_TSTEP[name]
    obs = _observations()[name]

    assert obs.uv_range_at_tstep(0) == (Pair(first[0]), Pair(first[1]))
    assert obs.uv_range_at_tstep(1) == (Pair(second[0]), Pair(second[1]))


def test_uv_range_at_tstep_indexes_a_two_dimensional_cadence_by_a_pair() -> None:
    """A TimedImage with a 2-D cadence exposes one pixel per (slow, fast) time step."""

    obs = _observations()['TimedImage2D']

    assert obs.uv_range_at_tstep(Pair((0, 0))) == (Pair((0, 0)), Pair((1, 1)))
    assert obs.uv_range_at_tstep(Pair((3, 7))) == (Pair((3, 7)), Pair((4, 8)))
    assert obs.uv_range_at_tstep(Pair((9, 19))) == (Pair((9, 19)), Pair((10, 20)))


@pytest.mark.parametrize('name', sorted(UV_RANGE_AT_TSTEP))
def test_uv_range_at_tstep_agrees_with_uv_range_at_time(name: str) -> None:
    """The pixels of a time step are those active at a time within that step.

    uv_range_at_time is implemented independently, so this ties the two together.
    """

    (_, _, steps) = UV_RANGE_AT_TSTEP[name]
    obs = _observations()[name]

    for tstep in range(steps):
        midtime = obs.cadence.time_at_tstep(tstep + 0.5)

        assert obs.uv_range_at_tstep(tstep) == obs.uv_range_at_time(midtime)


def test_uv_range_at_tstep_agrees_with_uv_range_at_time_in_two_dimensions() -> None:
    """The same agreement holds over the whole grid of a 2-D cadence."""

    obs = _observations()['TimedImage2D']

    for i in range(10):
        for j in range(20):
            midtime = obs.cadence.time_at_tstep(Pair((i + 0.5, j + 0.5)))

            assert obs.uv_range_at_tstep(Pair((i, j))) == obs.uv_range_at_time(midtime)


def test_uv_range_at_tstep_masks_steps_outside_the_cadence() -> None:
    """With remask, a time step outside the cadence has no pixels to report."""

    obs = _observations()['RasterSlit1D']       # five time steps
    (uv_min, uv_max) = obs.uv_range_at_tstep(Scalar([-1, 4, 5, 6]), remask=True)

    # Index 5 is the inclusive end of the last step, so it belongs to the cadence.
    assert list(uv_min.mask) == [True, False, False, True]
    assert list(uv_max.mask) == [True, False, False, True]


def test_uv_range_at_tstep_clips_outside_steps_without_remask() -> None:
    """Without remask, a time step beyond either end reports the nearest one."""

    obs = _observations()['RasterSlit1D']       # five time steps

    assert obs.uv_range_at_tstep(-1) == obs.uv_range_at_tstep(0)
    assert obs.uv_range_at_tstep(99) == obs.uv_range_at_tstep(4)


def test_uv_range_at_tstep_propagates_the_mask_of_its_time_step() -> None:
    """A masked time step yields a masked range even when remask is False."""

    obs = _observations()['RasterSlit1D']
    (uv_min, uv_max) = obs.uv_range_at_tstep(Scalar([0, 2], [True, False]),
                                             remask=False)

    assert list(uv_min.mask) == [True, False]
    assert uv_min[1] == Pair((2, 0))


def test_uv_range_at_tstep_is_not_implemented_for_an_insitu_observation() -> None:
    """InSitu has no field of view, so it declines this as it declines its sibling."""

    obs = _observations()['InSitu']

    with pytest.raises(NotImplementedError, match='uv_range_at_tstep'):
        obs.uv_range_at_tstep(0)


##########################################################################################
# The (u,v) and time range helpers of the Observation base class
##########################################################################################

IMAGE_FOV = FlatFOV((0.001, 0.001), (10, 20))


def _snapshot() -> Snapshot:
    """A Snapshot: two spatial axes and no time-dependence."""

    return Snapshot(('u', 'v'), 0., 10., IMAGE_FOV, 'SSB', 'J2000')


def _timed_image() -> TimedImage:
    """A TimedImage whose v-axis is swept in time by a 1-D cadence."""

    rows = Metronome(tstart=0., tstride=1., texp=1., steps=20)

    return TimedImage(('u', 'vt'), rows, IMAGE_FOV, 'SSB', 'J2000')


def _timed_image_2d() -> TimedImage:
    """A TimedImage whose two spatial axes are swept by a 2-D cadence."""

    slow = Metronome(tstart=0., tstride=20., texp=20., steps=10)
    fast = Metronome(tstart=0., tstride=1., texp=1., steps=20)

    return TimedImage(('uslow', 'vfast'), DualCadence(slow, fast), IMAGE_FOV,
                      'SSB', 'J2000')


def test_time_and_midtime_come_from_the_cadence() -> None:
    """The overall time limits and mid-time are inherited from the cadence."""

    obs = _snapshot()

    assert obs.time == (0., 10.)
    assert obs.midtime == 5.


def test_midtime_at_uv_defaults_to_the_middle_of_the_exposure() -> None:
    """tfrac=0.5 is the mid-time of the pixel's integration."""

    assert _snapshot().midtime_at_uv(Pair((5., 10.))) == Scalar(5.)


@pytest.mark.parametrize('tfrac, expected', [(0., 0.), (0.25, 2.5), (1., 10.)])
def test_midtime_at_uv_honors_tfrac(tfrac: float, expected: float) -> None:
    """tfrac=0 is the beginning of the exposure and 1 is the end."""

    assert _snapshot().midtime_at_uv(Pair((5., 10.)), tfrac=tfrac) == Scalar(expected)


def test_uv_is_outside_marks_coordinates_beyond_the_fov() -> None:
    """The test is the FOV's, applied to the observation's coordinates."""

    outside = _snapshot().uv_is_outside(Pair([(5., 10.), (50., 10.), (5., 90.)]))

    assert list(outside.vals) == [False, True, True]


def test_uv_is_outside_can_exclude_the_upper_corner() -> None:
    """inclusive=False treats the upper end of each range as outside."""

    obs = _snapshot()

    assert not obs.uv_is_outside(Pair((10., 20.)))
    assert obs.uv_is_outside(Pair((10., 20.)), inclusive=False)


def test_time_range_at_uv_0d_covers_the_whole_cadence() -> None:
    """With time decoupled from the spatial axes, every pixel spans the exposure."""

    (tmin, tmax) = _snapshot()._time_range_at_uv_0d(Pair((5., 10.)))

    assert tmin == Scalar(0.)
    assert tmax == Scalar(10.)


def test_time_range_at_uv_1d_follows_the_spatial_axis() -> None:
    """With a 1-D cadence, the pixel's time range is that of its row."""

    (tmin, tmax) = _timed_image()._time_range_at_uv_1d(Pair((5., 10.)), axis=1)

    assert tmin == Scalar(10.)
    assert tmax == Scalar(11.)


def test_time_range_at_uv_2d_follows_both_axes() -> None:
    """With a 2-D cadence, both spatial axes select the time step."""

    (tmin, tmax) = _timed_image_2d()._time_range_at_uv_2d(Pair((5., 10.)), fast=1)

    assert tmin == Scalar(110.)
    assert tmax == Scalar(111.)


def test_uv_range_at_time_0d_covers_the_whole_fov() -> None:
    """With time decoupled from the spatial axes, every pixel is observed at once."""

    (uv_min, uv_max) = _snapshot()._uv_range_at_time_0d(Scalar(5.), (10, 20))

    assert uv_min == Pair((0, 0))
    assert uv_max == Pair((10, 20))


def test_uv_range_at_time_1d_selects_one_row() -> None:
    """With a 1-D cadence, only the row being swept is active."""

    (uv_min, uv_max) = _timed_image()._uv_range_at_time_1d(Scalar(5.), (10, 20), axis=1)

    assert uv_min == Pair((0, 5))
    assert uv_max == Pair((10, 6))


def test_uv_range_at_time_2d_selects_one_pixel() -> None:
    """With a 2-D cadence, a single pixel is active at a time."""

    (uv_min, uv_max) = _timed_image_2d()._uv_range_at_time_2d(Scalar(25.), (10, 20),
                                                             slow=0, fast=1)

    assert uv_min == Pair((1, 5))
    assert uv_max == Pair((2, 6))


def test_uv_range_at_tstep_0d_covers_the_whole_fov() -> None:
    """Every pixel is active at every time step when time is decoupled."""

    (uv_min, uv_max) = _snapshot()._uv_range_at_tstep_0d(Scalar(0.), (10, 20))

    assert uv_min == Pair((0, 0))
    assert uv_max == Pair((10, 20))


def test_uv_range_at_tstep_1d_selects_one_row() -> None:
    """One pixel is active along the cadence axis, the whole FOV along the other."""

    (uv_min, uv_max) = _timed_image()._uv_range_at_tstep_1d(Scalar(5.), (10, 20), axis=1)

    assert uv_min == Pair((0, 5))
    assert uv_max == Pair((10, 6))


def test_uv_range_at_tstep_2d_selects_one_pixel() -> None:
    """Both indices of a 2-D cadence select a single active pixel."""

    (uv_min, uv_max) = _timed_image_2d()._uv_range_at_tstep_2d(Pair((1., 5.)), (10, 20),
                                                              slow=0, fast=1)

    assert uv_min == Pair((1, 5))
    assert uv_max == Pair((2, 6))


def test_uv_range_at_time_agrees_with_the_0d_helper() -> None:
    """The general entry point dispatches to the helper for this observation."""

    obs = _snapshot()

    assert obs.uv_range_at_time(Scalar(5.)) \
           == obs._uv_range_at_time_0d(Scalar(5.), obs.uv_shape)


##########################################################################################
# Copies, navigation, subfields, and the SPICE C matrix
##########################################################################################

def test_copy_is_a_new_object() -> None:
    """A copy is independent of the original."""

    obs = _snapshot()

    assert obs.copy() is not obs


def test_copy_shares_the_canonical_sub_objects() -> None:
    """The frame, path, FOV and cadence are shared rather than duplicated."""

    obs = _snapshot()
    copied = obs.copy()

    assert copied.fov is obs.fov
    assert copied.cadence is obs.cadence


def test_copy_gets_its_own_subfield_dictionary() -> None:
    """A subfield inserted into the copy does not reach the original."""

    obs = _snapshot()
    copied = obs.copy()
    copied.insert_subfield('sample', 1)

    assert 'sample' not in obs.subfields


def test_navigate_gives_the_copy_a_navigation_frame() -> None:
    """The re-pointed copy carries a Navigation frame of its own."""

    obs = _snapshot()
    navigated = obs.navigate((0.001, 0.002))

    assert isinstance(navigated.frame, Navigation)


def test_navigate_leaves_the_original_alone() -> None:
    """Re-pointing the copy does not disturb this observation."""

    obs = _snapshot()
    frame_before = obs.frame
    obs.navigate((0.001, 0.002))

    assert obs.frame is frame_before


def test_navigate_accepts_three_angles() -> None:
    """A third angle rotates about the z-axis."""

    navigated = _snapshot().navigate((0.001, 0.002, 0.003))

    assert isinstance(navigated.frame, Navigation)


def test_set_frame_replaces_the_frame_in_place() -> None:
    """set_frame modifies the observation rather than returning a copy."""

    obs = _snapshot()
    obs.set_frame(Frame.J2000)

    assert obs.frame == Frame.J2000.wayframe


def test_insert_and_delete_a_subfield() -> None:
    """A subfield is readable as an attribute until it is deleted."""

    obs = _snapshot()
    obs.insert_subfield('sample', 42)
    assert obs.sample == 42

    obs.delete_subfield('sample')
    assert 'sample' not in obs.subfields


def test_delete_a_missing_subfield_is_harmless() -> None:
    """Deleting a subfield that is not present does nothing."""

    obs = _snapshot()
    obs.delete_subfield('never_inserted')

    assert 'never_inserted' not in obs.subfields


def test_delete_subfields_removes_them_all() -> None:
    """delete_subfields empties the dictionary."""

    obs = _snapshot()
    obs.insert_subfield('a', 1)
    obs.insert_subfield('b', 2)
    obs.delete_subfields()

    assert obs.subfields == {}


def test_get_spice_cmatrix_requires_spice_to_frame() -> None:
    """The observation must carry the rotation inserted by its host module."""

    with pytest.raises(AttributeError, match='spice_to_frame'):
        _snapshot().get_spice_cmatrix()


def test_get_spice_cmatrix_defaults_to_the_midtime() -> None:
    """With neither tstep nor time given, the mid-time of the observation is used."""

    obs = _snapshot()
    obs.insert_subfield('spice_to_frame', Matrix3.IDENTITY)

    assert obs.get_spice_cmatrix() == obs.get_spice_cmatrix(time=obs.midtime)


def test_get_spice_cmatrix_rejects_both_tstep_and_time() -> None:
    """At most one of tstep and time can be specified."""

    obs = _snapshot()
    obs.insert_subfield('spice_to_frame', Matrix3.IDENTITY)

    with pytest.raises(ValueError, match='cannot both be specified'):
        obs.get_spice_cmatrix(tstep=0., time=5.)


def test_set_spice_cmatrix_fixes_the_pointing() -> None:
    """Setting the C matrix leaves the observation with a fixed pointing."""

    obs = _snapshot()
    obs.insert_subfield('spice_to_frame', Matrix3.IDENTITY)
    obs.set_spice_cmatrix(Matrix3.IDENTITY)

    assert obs.get_spice_cmatrix() == Matrix3.IDENTITY


##########################################################################################
# meshgrid and timegrid
##########################################################################################

def test_meshgrid_takes_the_shape_of_the_observation() -> None:
    """The (u,v) axes are placed where the observation's axis ordering puts them."""

    obs = _snapshot()

    assert obs.meshgrid().shape == tuple(obs.uv_shape)


def test_meshgrid_undersamples() -> None:
    """An undersample of 2 samples every other pixel along each axis."""

    assert _snapshot().meshgrid(undersample=2).shape == (5, 10)


def test_meshgrid_oversamples() -> None:
    """An oversample of 2 creates a 2x2 array of samples inside each pixel."""

    assert _snapshot().meshgrid(oversample=2).shape == (20, 40)


def test_meshgrid_honors_the_limit() -> None:
    """The limit replaces the shape of the FOV as the upper bound."""

    assert _snapshot().meshgrid(limit=(4, 8)).shape == (4, 8)


def test_timegrid_of_an_untimed_observation_is_the_midtime() -> None:
    """With no time-dependence, the default grid is a single mid-exposure time."""

    obs = _snapshot()

    assert obs.timegrid(obs.meshgrid()) == Scalar(obs.midtime)


@pytest.mark.parametrize('tfrac_limits, expected', [(0., 0.), (0.25, 2.5), (1., 10.)])
def test_timegrid_honors_a_single_tfrac_limit(tfrac_limits: float,
                                              expected: float) -> None:
    """A single number is interpreted as a pair of identical limits."""

    obs = _snapshot()

    assert obs.timegrid(obs.meshgrid(), tfrac_limits=tfrac_limits) == Scalar(expected)


def test_timegrid_oversampling_gives_more_times() -> None:
    """An oversample above one samples the exposure more finely in time."""

    obs = _snapshot()

    assert obs.timegrid(obs.meshgrid(), oversample=2).shape[0] == 2


def test_timegrid_broadcasts_with_a_timed_meshgrid() -> None:
    """With time coupled to a spatial axis, the grid follows the meshgrid's shape."""

    obs = _timed_image()
    meshgrid = obs.meshgrid()

    assert obs.timegrid(meshgrid).shape == meshgrid.shape

def test_uv_range_at_time_0d_masks_a_time_outside_the_cadence() -> None:
    """With remask, a time outside the exposure marks the whole range as masked."""

    obs = _snapshot()

    (uv_min, uv_max) = obs._uv_range_at_time_0d(Scalar([5., 50.]), (10, 20), remask=True)

    assert list(uv_min.mask) == [False, True]
    assert list(uv_max.mask) == [False, True]
    assert uv_min[0] == Pair((0, 0))
    assert uv_max[0] == Pair((10, 20))


def test_uv_range_at_time_0d_stays_shapeless_when_nothing_is_masked() -> None:
    """With every time inside the cadence, remask leaves the shapeless answer alone."""

    obs = _snapshot()

    (uv_min, uv_max) = obs._uv_range_at_time_0d(Scalar([2., 8.]), (10, 20), remask=True)

    assert uv_min == Pair((0, 0))
    assert uv_max == Pair((10, 20))
    assert uv_min.shape == ()


def test_uv_range_at_time_1d_falls_back_to_the_0d_helper() -> None:
    """axis=-1 means the time axis is not a spatial axis, so the whole FOV is active."""

    obs = _timed_image()

    assert obs._uv_range_at_time_1d(Scalar(5.), (10, 20), axis=-1) \
           == obs._uv_range_at_time_0d(Scalar(5.), (10, 20))


def test_uv_range_at_time_2d_swaps_the_axes_when_the_slow_index_is_v() -> None:
    """A cadence whose slow index sweeps v reports its steps in (u,v) order."""

    obs = _timed_image_2d()

    (uv_min, uv_max) = obs._uv_range_at_time_2d(Scalar(25.), (20, 10), slow=1, fast=0)

    assert uv_min == Pair((5, 1))
    assert uv_max == Pair((6, 2))


def test_uv_range_at_time_2d_spans_an_axis_no_cadence_index_sweeps() -> None:
    """An index marked -1 sweeps no spatial axis, so that axis stays fully active."""

    obs = _timed_image_2d()

    (uv_min, uv_max) = obs._uv_range_at_time_2d(Scalar(25.), (10, 20), slow=-1, fast=1)

    assert uv_min == Pair((0, 5))
    assert uv_max == Pair((10, 6))


def test_uv_range_at_time_2d_spans_the_fov_when_neither_index_sweeps_it() -> None:
    """With the fast index detached as well, every pixel is active."""

    obs = _timed_image_2d()

    (uv_min, uv_max) = obs._uv_range_at_time_2d(Scalar(25.), (10, 20), slow=-1, fast=-1)

    assert uv_min == Pair((0, 0))
    assert uv_max == Pair((10, 20))


def test_uv_range_at_tstep_0d_masks_a_step_outside_the_cadence() -> None:
    """With remask, a step beyond the cadence marks the whole range as masked."""

    obs = _snapshot()

    (uv_min, uv_max) = obs._uv_range_at_tstep_0d(Scalar([0., -1.]), (10, 20), remask=True)

    assert list(uv_min.mask) == [False, True]
    assert list(uv_max.mask) == [False, True]
    assert uv_min[0] == Pair((0, 0))
    assert uv_max[0] == Pair((10, 20))


def test_uv_range_at_tstep_0d_expands_a_masked_step_without_remask() -> None:
    """A step that is already masked forces the array form even without remask."""

    obs = _snapshot()

    (uv_min, uv_max) = obs._uv_range_at_tstep_0d(Scalar([0., 1.], [False, True]),
                                                (10, 20))

    assert list(uv_min.mask) == [False, True]
    assert uv_max[0] == Pair((10, 20))


def test_uv_range_at_tstep_1d_falls_back_to_the_0d_helper() -> None:
    """axis=-1 means the time axis is not a spatial axis, so the whole FOV is active."""

    obs = _timed_image()

    assert obs._uv_range_at_tstep_1d(Scalar(5.), (10, 20), axis=-1) \
           == obs._uv_range_at_tstep_0d(Scalar(5.), (10, 20))


def test_uv_range_at_tstep_2d_swaps_the_axes_when_the_slow_index_is_v() -> None:
    """A cadence whose slow index sweeps v reports its step in (u,v) order."""

    obs = _timed_image_2d()

    (uv_min, uv_max) = obs._uv_range_at_tstep_2d(Pair((1., 5.)), (20, 10),
                                                slow=1, fast=0)

    assert uv_min == Pair((5, 1))
    assert uv_max == Pair((6, 2))


def test_uv_range_at_tstep_2d_spans_an_axis_no_cadence_index_sweeps() -> None:
    """An index marked -1 sweeps no spatial axis, so that axis stays fully active."""

    obs = _timed_image_2d()

    (uv_min, uv_max) = obs._uv_range_at_tstep_2d(Pair((1., 5.)), (10, 20),
                                                slow=-1, fast=1)

    assert uv_min == Pair((0, 5))
    assert uv_max == Pair((10, 6))


def test_uv_range_at_tstep_2d_spans_the_fov_when_neither_index_sweeps_it() -> None:
    """With both indices detached, every pixel is active at every time step."""

    obs = _timed_image_2d()

    (uv_min, uv_max) = obs._uv_range_at_tstep_2d(Pair((1., 5.)), (10, 20),
                                                slow=-1, fast=-1)

    assert uv_min == Pair((0, 0))
    assert uv_max == Pair((10, 20))


def test_copy_duplicates_a_fittable_frame() -> None:
    """A fittable sub-object is duplicated, so fitting the copy leaves the original."""

    navigated = _snapshot().navigate((1.e-6, 2.e-6))

    duplicate = navigated.copy()

    assert duplicate.frame is not navigated.frame
    assert duplicate.frame.transform_at_time(Scalar(5.)).matrix \
           == navigated.frame.transform_at_time(Scalar(5.)).matrix


def test_copy_duplicates_a_fittable_subfield() -> None:
    """A fittable that is also a subfield is replaced in both records at once."""

    obs = _snapshot().navigate((1.e-6, 2.e-6))
    obs.insert_subfield('frame', obs.frame)

    duplicate = obs.copy()

    assert duplicate.subfields['frame'] is duplicate.frame
    assert duplicate.subfields['frame'] is not obs.frame


def test_navigating_twice_replaces_the_navigation() -> None:
    """A second navigation supersedes the first rather than stacking on top of it."""

    once = _snapshot().navigate((1.e-6, 2.e-6))

    twice = once.navigate((3.e-6, 4.e-6))

    assert isinstance(twice.frame, Navigation)
    assert twice.frame.reference is once.frame.reference


def test_set_frame_is_refused_on_a_frozen_observation() -> None:
    """Freezing an observation fixes its pointing, so the frame cannot be replaced."""

    obs = _snapshot().navigate((1.e-6, 2.e-6))
    mutable.freeze(obs)

    with pytest.raises(ValueError, match='Snapshot object is frozen'):
        obs.set_frame(Frame.J2000)


def test_get_spice_cmatrix_accepts_a_time_step() -> None:
    """A time step is converted to a time through the cadence."""

    obs = _timed_image()
    obs.insert_subfield('spice_to_frame', Matrix3.IDENTITY)

    assert obs.get_spice_cmatrix(tstep=5) \
           == obs.get_spice_cmatrix(time=obs.cadence.time_at_tstep(5))


def test_timegrid_oversamples_a_time_coupled_spatial_axis() -> None:
    """With time on a spatial axis, oversampling adds a leading axis of times."""

    obs = _timed_image()
    meshgrid = obs.meshgrid()

    times = obs.timegrid(meshgrid, oversample=3)

    assert times.shape == (3,) + meshgrid.shape
    assert times[0] != times[-1]


def test_timegrid_samples_a_time_axis_of_its_own() -> None:
    """When no spatial axis carries the time, the grid runs along the cadence."""

    pixel_fov = FlatFOV((0.001, 0.001), (1, 1))
    steps = Metronome(tstart=0., tstride=10., texp=10., steps=5)
    obs = Pixel(('t',), steps, pixel_fov, 'SSB', 'J2000')

    times = obs.timegrid(obs.meshgrid())

    assert times == Scalar([0., 10., 20., 30., 40., 50.])


def test_timegrid_of_a_two_dimensional_cadence_is_the_pixel_midtime() -> None:
    """With both axes swept, one sample per pixel is that pixel's mid-exposure time."""

    obs = _timed_image_2d()
    meshgrid = obs.meshgrid()

    times = obs.timegrid(meshgrid)

    assert times.shape == meshgrid.shape
    assert times == obs.midtime_at_uv(meshgrid.uv)


def test_timegrid_of_a_two_dimensional_cadence_oversamples() -> None:
    """Oversampling a 2-D cadence adds a leading axis of times."""

    obs = _timed_image_2d()
    meshgrid = obs.meshgrid()

    times = obs.timegrid(meshgrid, oversample=3)

    assert times.shape == (3,) + meshgrid.shape
    assert times[0] != times[-1]


def test_gridless_event_without_a_meshgrid_spans_the_exposure() -> None:
    """With no meshgrid, the time comes from the observation's own time limits."""

    obs = _snapshot()

    event = obs.gridless_event(tfrac=0.25)

    assert event.time == Scalar(2.5)
    assert event.arr is None


def test_scalar_from_indices_selects_an_axis_of_an_array() -> None:
    """A bare NumPy array is indexed along its last axis."""

    indices = np.array([[1., 2.], [3., 4.]])

    assert Observation._scalar_from_indices(indices, 1) == Scalar([2., 4.])


def test_scalar_from_indices_passes_a_narrow_array_through() -> None:
    """An array with no room for the requested axis is returned as it stands."""

    indices = np.array([[1.], [2.]])

    assert Observation._scalar_from_indices(indices, 1) == Scalar([[1.], [2.]])


def test_parallel_los_is_unchanged_between_identical_frames() -> None:
    """Two observations sharing a frame see a line of sight identically."""

    (obs, parallel) = _parallel_pair()
    los = obs.fov.los_from_uvt(Pair((10., 20.)), time=obs.midtime)

    assert obs.parallel_los(parallel, los) == los


def test_parallel_los_rotates_into_the_parallel_frame() -> None:
    """A parallel observation on a rotated frame sees the line of sight rotated."""

    (obs, _) = _parallel_pair()
    rotated = obs.navigate((0.1, 0.))
    los = obs.fov.los_from_uvt(Pair((10., 20.)), time=obs.midtime)

    turned = obs.parallel_los(rotated, los)

    assert turned != los
    assert float(turned.norm().vals) == pytest.approx(float(los.norm().vals), abs=1.e-12)


def test_parallel_uv_maps_a_pixel_through_an_identical_fov() -> None:
    """With the same frame and FOV, a pixel maps to itself."""

    (obs, parallel) = _parallel_pair()

    assert obs.parallel_uv(parallel, Pair((10., 20.))) == Pair((10., 20.))


def test_parallel_uv_assumes_the_midtime() -> None:
    """Omitting the time gives the same answer as passing the midtime, as documented."""

    (obs, parallel) = _parallel_pair()

    assert obs.parallel_uv(parallel, Pair((10., 20.))) \
           == obs.parallel_uv(parallel, Pair((10., 20.)), time=obs.midtime)


def test_parallel_offset_angles_accepts_a_frame_and_fov_in_place_of_an_observation():
    """A (frame, FOV) pair stands in for the parallel observation itself."""

    (obs, parallel) = _parallel_pair()
    angles = (1.e-5, 2.e-5)

    assert obs.parallel_offset_angles((parallel.frame, parallel.fov), angles) \
           == obs.parallel_offset_angles(parallel, angles)

def test_time_range_at_uv_0d_masks_a_pixel_outside_the_fov() -> None:
    """With remask, a pixel beyond the field of view has no time range."""

    obs = _snapshot()

    (tmin, tmax) = obs._time_range_at_uv_0d(Pair([(5., 10.), (50., 10.)]), remask=True)

    assert list(tmin.mask) == [False, True]
    assert list(tmax.mask) == [False, True]
    assert tmin[0] == Scalar(0.)


def test_uv_range_at_time_2d_spans_only_the_axis_the_fast_index_misses() -> None:
    """With the fast index detached, the slow index still selects its own axis."""

    obs = _timed_image_2d()

    (uv_min, uv_max) = obs._uv_range_at_time_2d(Scalar(25.), (10, 20), slow=0, fast=-1)

    assert uv_min == Pair((1, 0))
    assert uv_max == Pair((2, 20))


def test_uv_range_at_tstep_2d_spans_only_the_axis_the_fast_index_misses() -> None:
    """With the fast index detached, the slow index still selects its own axis."""

    obs = _timed_image_2d()

    (uv_min, uv_max) = obs._uv_range_at_tstep_2d(Pair((1., 5.)), (10, 20),
                                                slow=0, fast=-1)

    assert uv_min == Pair((1, 0))
    assert uv_max == Pair((2, 20))


def test_set_spice_cmatrix_requires_spice_to_frame() -> None:
    """The observation must carry the rotation inserted by its host module."""

    with pytest.raises(AttributeError, match='spice_to_frame'):
        _snapshot().set_spice_cmatrix(Matrix3.IDENTITY)


def test_uv_from_ra_and_dec_accepts_an_absolute_time() -> None:
    """An absolute time replaces the guess, so a single iteration suffices."""

    obs = _timed_image()

    # The FlatFOV axis is +Z, which is a declination of 90 degrees: the FOV center.
    uv = obs.uv_from_ra_and_dec(0., np.pi/2, time=Scalar(5.))

    assert uv.vals == pytest.approx((5., 10.), abs=1.e-9)


def test_uv_from_ra_and_dec_stops_once_the_pixel_stops_moving() -> None:
    """Iteration ends early when a pass reproduces the pixel the previous one found."""

    obs = _timed_image()

    uv = obs.uv_from_ra_and_dec(0., np.pi/2, iters=4)

    assert uv == obs.uv_from_ra_and_dec(0., np.pi/2, iters=2)

##########################################################################################
