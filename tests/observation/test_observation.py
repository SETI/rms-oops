##########################################################################################
# tests/observation/test_observation.py
##########################################################################################

import pickle

import numpy as np
import pytest

from oops.cadence     import DualCadence, Metronome
from oops.fov         import FlatFOV
from oops.frame       import Frame
from oops.observation import (InSitu, Observation, Pixel, RasterSlit1D, Slit1D,
                              Snapshot, TimedImage)
from oops.path        import Path
from polymath         import Pair, Scalar


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
