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

##########################################################################################
