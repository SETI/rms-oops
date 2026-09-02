##########################################################################################
# tests/observation/test_observation.py
##########################################################################################

import pickle

import pytest

from oops.cadence     import DualCadence, Metronome
from oops.fov         import FlatFOV
from oops.frame       import Frame
from oops.observation import (InSitu, Observation, Pixel, RasterSlit1D, Slit1D,
                              Snapshot, TimedImage)
from oops.path        import Path
from polymath         import Pair


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


##########################################################################################
