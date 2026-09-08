##########################################################################################
# oops/cadence/timeshift.py: TimeShift subclass of class Cadence
##########################################################################################

import pickle

import numpy as np

from polymath import Scalar
from oops.cadence import Metronome, TimeShift


def test_timeshift():
    DT = 10.

    # 100-110, 110-120, 120-130, 130-140
    cadence = Metronome(100., 10., 10., 4)
    shifted = TimeShift(DT, cadence)
    assert shifted.dt == DT
    assert shifted.link is None

    # The shift is held privately and published as a property, as in PathShift and
    # FrameShift. The wrapped cadence is an implementation detail, like the internals of
    # the other Cadence subclasses; only the attributes the Cadence contract documents
    # stay public.
    assert isinstance(type(shifted).dt, property)
    assert isinstance(type(shifted).link, property)
    assert sorted(a for a in vars(shifted)
                  if a.startswith('_')) == ['_cadence', '_dt', '_link']
    for name in ('time', 'midtime', 'lasttime', 'shape', 'is_continuous', 'is_unique',
                 'min_tstride', 'max_tstride'):
        assert name in vars(shifted)

    # Every time is offset by dt, and the shape is unchanged
    assert shifted.shape == cadence.shape
    assert shifted.time == (cadence.time[0] + DT, cadence.time[1] + DT)
    assert shifted.midtime == cadence.midtime + DT

    tstep = Scalar(np.arange(0., 4., 0.25))
    assert shifted.time_at_tstep(tstep) == cadence.time_at_tstep(tstep) + DT
    assert shifted.tstep_at_time(cadence.time_at_tstep(tstep) + DT) == tstep

    # A linked TimeShift tracks the offset of the object it is linked to
    linked = TimeShift(shifted, cadence)
    assert linked.link is shifted
    assert linked.dt == DT

    shifted.set_params(np.array([2. * DT]))
    linked._refresh()
    assert linked.dt == 2. * DT
    assert linked.time_at_tstep(tstep) == cadence.time_at_tstep(tstep) + 2. * DT

##########################################################################################
# The rest of the Cadence API, which TimeShift forwards with its offset applied
##########################################################################################

DT = 10.
TSTART = 100.
TSTRIDE = 10.
TEXP = 8.                       # shorter than the stride, so the cadence has gaps
STEPS = 4


def _pair() -> tuple[Metronome, TimeShift]:
    """A Metronome with gaps between its steps, and the same cadence shifted later.

    Returns:
        tuple[Metronome, TimeShift]: The original cadence and the shifted one.
    """

    cadence = Metronome(TSTART, TSTRIDE, TEXP, STEPS)

    return (cadence, TimeShift(DT, cadence))


def test_the_time_range_of_a_step_is_shifted() -> None:
    """Both ends of a time step move later by the offset."""

    (cadence, shifted) = _pair()
    tstep = Scalar([0., 1., 2., 3.])

    (tmin, tmax) = shifted.time_range_at_tstep(tstep)
    (expected_min, expected_max) = cadence.time_range_at_tstep(tstep)

    assert tmin == expected_min + DT
    assert tmax == expected_max + DT


def test_the_step_range_at_a_time_is_taken_from_the_shifted_time() -> None:
    """A time is converted back to the unshifted cadence before it is looked up."""

    (cadence, shifted) = _pair()
    time = Scalar([105., 115., 125.])

    assert shifted.tstep_range_at_time(time + DT) \
           == cadence.tstep_range_at_time(time)


def test_a_time_outside_the_shifted_cadence_is_reported() -> None:
    """The gaps and the endpoints move later along with everything else."""

    (_, shifted) = _pair()

    # The shifted cadence samples 110-118, 120-128, 130-138 and 140-148
    outside = shifted.time_is_outside(Scalar([109., 115., 119., 151.]))

    assert list(outside.vals) == [True, False, True, True]


def test_the_end_of_the_cadence_can_be_excluded() -> None:
    """inclusive=False treats the final instant as outside, as it does unshifted."""

    (_, shifted) = _pair()
    end = Scalar(shifted.time[1])

    assert not shifted.time_is_outside(end)
    assert shifted.time_is_outside(end, inclusive=False)


def test_shifting_a_shifted_cadence_shifts_the_cadence_it_wraps() -> None:
    """A further shift moves the wrapped cadence, leaving the fitted offset alone."""

    (_, shifted) = _pair()

    twice = shifted.time_shift(100.)

    assert isinstance(twice, TimeShift)
    assert twice.dt == DT
    assert twice.time == (shifted.time[0] + 100., shifted.time[1] + 100.)


def test_a_shifted_cadence_can_be_made_continuous() -> None:
    """Continuity is a property of the wrapped cadence, and the offset survives it."""

    (_, shifted) = _pair()

    continuous = shifted.as_continuous()

    assert isinstance(continuous, TimeShift)
    assert continuous.is_continuous
    assert not shifted.is_continuous
    assert continuous.dt == DT


def test_a_linked_shift_keeps_the_link_through_a_further_shift() -> None:
    """A linked TimeShift stays linked when it is shifted or made continuous."""

    (cadence, shifted) = _pair()
    linked = TimeShift(shifted, cadence)

    assert linked.time_shift(100.).link is shifted
    assert linked.as_continuous().link is shifted


def test_setting_the_parameters_of_a_linked_shift_moves_the_original() -> None:
    """Fitting a linked shift redefines the offset of the object it follows."""

    (cadence, shifted) = _pair()
    linked = TimeShift(shifted, cadence)

    linked.set_params(np.array([3. * DT]))

    assert shifted.dt == 3. * DT
    assert linked.dt == 3. * DT


def test_a_shifted_cadence_survives_a_round_trip_through_pickle() -> None:
    """Unpickling rebuilds the shift from its offset and the cadence it wraps."""

    (_, shifted) = _pair()

    revived = pickle.loads(pickle.dumps(shifted))

    assert revived.dt == DT
    assert revived.time == shifted.time
    assert revived.time_at_tstep(Scalar(1.5)) == shifted.time_at_tstep(Scalar(1.5))

##########################################################################################
