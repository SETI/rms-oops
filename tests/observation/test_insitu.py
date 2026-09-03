##########################################################################################
# tests/observation/test_insitu.py
##########################################################################################

import pickle

import numpy as np
import pytest

from oops.cadence     import Instant, Metronome
from oops.fov         import NullFOV
from oops.observation import InSitu, Observation
from oops.path        import Path
from polymath         import Pair, Scalar

TSTART = 0.
TSTRIDE = 10.
TEXP = 10.
STEPS = 5


def steps_cadence() -> Metronome:
    """A five-step cadence covering 0 to 50 seconds."""

    return Metronome(tstart=TSTART, tstride=TSTRIDE, texp=TEXP, steps=STEPS)


def test_insitu_takes_its_shape_from_the_cadence() -> None:
    """The shape of the cadence defines the shape of the observation."""

    obs = InSitu(steps_cadence(), 'SSB')

    assert isinstance(obs, Observation)
    assert obs.shape == (STEPS,)
    assert obs.cadence.shape == (STEPS,)
    assert obs.time == (TSTART, TSTART + STEPS * TSTRIDE)
    assert obs.midtime == (TSTART + STEPS * TSTRIDE) / 2.


def test_insitu_has_no_pointing() -> None:
    """An InSitu observation has timing and path information, but no field of view."""

    obs = InSitu(steps_cadence(), 'SSB')

    assert isinstance(obs.fov, NullFOV)
    assert obs.u_axis == -1
    assert obs.v_axis == -1
    assert obs.path == Path.as_path('SSB').waypoint


def test_insitu_time_axis_follows_the_cadence() -> None:
    """The single axis of the observation is the time axis."""

    obs = InSitu(steps_cadence(), 'SSB')

    assert obs.t_axis == [0]


def test_insitu_accepts_a_scalar_as_an_instant() -> None:
    """A Scalar is converted to a Cadence of subclass Instant."""

    obs = InSitu(Scalar(1234.), 'SSB')

    assert isinstance(obs.cadence, Instant)
    assert obs.midtime == 1234.
    assert obs.time == (1234., 1234.)


@pytest.mark.parametrize('cadence', ['not a cadence', 1234., None],
                         ids=['str', 'float', 'None'])
def test_insitu_rejects_anything_but_a_cadence_or_a_scalar(cadence: object) -> None:
    """A cadence that is neither a Cadence nor a Scalar raises TypeError."""

    with pytest.raises(TypeError, match='Invalid cadence class'):
        InSitu(cadence, 'SSB')


def test_insitu_time_shift() -> None:
    """A time shift moves the whole observation later and leaves its shape alone."""

    obs = InSitu(steps_cadence(), 'SSB')
    shifted = obs.time_shift(100.)

    assert isinstance(shifted, InSitu)
    assert shifted.time == (obs.time[0] + 100., obs.time[1] + 100.)
    assert shifted.midtime == obs.midtime + 100.
    assert shifted.shape == obs.shape
    assert shifted.path == obs.path

    # The original is unchanged
    assert obs.time == (TSTART, TSTART + STEPS * TSTRIDE)


def test_insitu_time_shift_by_zero_preserves_the_times() -> None:
    """A zero shift leaves the cadence where it was."""

    obs = InSitu(steps_cadence(), 'SSB')

    assert obs.time_shift(0.).time == obs.time


def test_insitu_subfields_are_attributes() -> None:
    """Optional attributes given to the constructor become subfields."""

    data = np.arange(STEPS)
    obs = InSitu(steps_cadence(), 'SSB', data=data)

    assert obs.subfields['data'] is data
    assert obs.data is data


@pytest.mark.parametrize('method, arg', [('uvt', Scalar(0.5)),
                                         ('uvt_range', Scalar(0.)),
                                         ('time_range_at_uv', Pair((0., 0.))),
                                         ('uv_range_at_time', Scalar(25.)),
                                         ('uv_range_at_tstep', Scalar(0.))])
def test_insitu_has_no_directional_methods(method: str, arg: object) -> None:
    """An InSitu observation carries no directional information, so the methods that
    would need it are not implemented."""

    obs = InSitu(steps_cadence(), 'SSB')

    with pytest.raises(NotImplementedError, match=f'InSitu.{method} is not implemented'):
        getattr(obs, method)(arg)


def test_insitu_pickle() -> None:
    """Pickling restores the cadence and the path."""

    obs = InSitu(steps_cadence(), 'SSB')
    restored = pickle.loads(pickle.dumps(obs))

    assert isinstance(restored, InSitu)
    assert restored.shape == obs.shape
    assert restored.time == obs.time
    assert restored.midtime == obs.midtime
    assert restored.path == obs.path


def test_insitu_getstate_roundtrip() -> None:
    """The state captured by __getstate__ fully restores the object."""

    obs = InSitu(steps_cadence(), 'SSB')
    state = obs.__getstate__()

    copied = Observation.__new__(InSitu)
    copied.__setstate__(state)
    assert copied.time == obs.time
    assert copied.shape == obs.shape

##########################################################################################
