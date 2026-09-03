##########################################################################################
# tests/cadence/test_snapcadence.py
##########################################################################################

import pickle

import numpy as np

from polymath import Boolean, Scalar
from oops.cadence import Metronome, SnapCadence

TSTART = 100.
TEXP = 10.


def test_snapcadence_is_a_single_step_metronome() -> None:
    """A SnapCadence has one time step spanning the whole exposure."""

    cadence = SnapCadence(TSTART, TEXP)

    assert isinstance(cadence, Metronome)
    assert cadence.shape == (1,)
    assert cadence.time == (TSTART, TSTART + TEXP)
    assert cadence.midtime == TSTART + TEXP / 2.
    assert cadence.lasttime == TSTART
    assert cadence.is_continuous
    assert cadence.is_unique


def test_snapcadence_time_at_tstep() -> None:
    """The single step interpolates linearly from tstart to tstart + texp."""

    cadence = SnapCadence(TSTART, TEXP)

    assert cadence.time_at_tstep(Scalar(0.)) == Scalar(TSTART)
    assert cadence.time_at_tstep(Scalar(0.5)) == Scalar(TSTART + TEXP / 2.)
    assert cadence.time_at_tstep(Scalar(1.)) == Scalar(TSTART + TEXP)


def test_snapcadence_time_range_at_tstep() -> None:
    """The one time step covers the full exposure."""

    cadence = SnapCadence(TSTART, TEXP)
    (tmin, tmax) = cadence.time_range_at_tstep(Scalar(0.))

    assert tmin == Scalar(TSTART)
    assert tmax == Scalar(TSTART + TEXP)


def test_snapcadence_tstep_at_time() -> None:
    """A time within the exposure maps to a fractional index within the single step."""

    cadence = SnapCadence(TSTART, TEXP)

    assert cadence.tstep_at_time(Scalar(TSTART)) == Scalar(0.)
    assert cadence.tstep_at_time(Scalar(TSTART + TEXP / 2.)) == Scalar(0.5)
    assert cadence.tstep_at_time(Scalar(TSTART + TEXP)) == Scalar(1.)


def test_snapcadence_tstep_range_at_time() -> None:
    """The only active step is step zero, wherever the time falls inside."""

    cadence = SnapCadence(TSTART, TEXP)
    (first, last) = cadence.tstep_range_at_time(Scalar(TSTART + 1.))

    assert first == Scalar(0)
    assert last == Scalar(1)

    # Outside the cadence, the range is empty: the two limits are equal
    (first, last) = cadence.tstep_range_at_time(Scalar(TSTART - 50.))
    assert first == last


def test_snapcadence_time_is_inside_and_outside() -> None:
    """The exposure interval is inside; its end is inside only when inclusive."""

    cadence = SnapCadence(TSTART, TEXP)
    times = Scalar([TSTART - 1., TSTART, TSTART + 5., TSTART + TEXP, TSTART + TEXP + 1.])

    assert cadence.time_is_inside(times) \
           == Boolean([False, True, True, True, False])
    assert cadence.time_is_inside(times, inclusive=False) \
           == Boolean([False, True, True, False, False])
    assert cadence.time_is_outside(times) \
           == Boolean([True, False, False, False, True])


def test_snapcadence_clipping() -> None:
    """With clip=True, index values outside the single step are pulled back inside."""

    clipped = SnapCadence(TSTART, TEXP, clip=True)

    assert clipped.time_at_tstep(Scalar(-5.)) == Scalar(TSTART)
    assert clipped.time_at_tstep(Scalar(5.)) == Scalar(TSTART + TEXP)

    # Without clipping, the same indices extrapolate along the step
    unclipped = SnapCadence(TSTART, TEXP, clip=False)
    assert unclipped.time_at_tstep(Scalar(2.)) == Scalar(TSTART + 2. * TEXP)


def test_snapcadence_remask_masks_indices_outside() -> None:
    """remask=True masks index values that fall outside the cadence."""

    cadence = SnapCadence(TSTART, TEXP)
    tstep = Scalar([0., 0.5, 1., 3.])

    assert not np.any(cadence.time_at_tstep(tstep, remask=False).mask)
    assert list(cadence.time_at_tstep(tstep, remask=True).mask) \
           == [False, False, False, True]


def test_snapcadence_time_shift() -> None:
    """A time shift moves both ends of the exposure and preserves its duration."""

    cadence = SnapCadence(TSTART, TEXP)
    shifted = cadence.time_shift(25.)

    assert isinstance(shifted, Metronome)
    assert shifted.time == (TSTART + 25., TSTART + TEXP + 25.)
    assert shifted.midtime == cadence.midtime + 25.
    assert shifted.shape == cadence.shape


def test_snapcadence_as_continuous() -> None:
    """A SnapCadence has no gaps, so forcing continuity changes nothing."""

    cadence = SnapCadence(TSTART, TEXP)
    continuous = cadence.as_continuous()

    assert continuous.is_continuous
    assert continuous.time == cadence.time
    assert continuous.shape == cadence.shape


def test_snapcadence_pickle() -> None:
    """Pickling restores the start time and the exposure."""

    cadence = SnapCadence(TSTART, TEXP)
    restored = pickle.loads(pickle.dumps(cadence))

    assert isinstance(restored, SnapCadence)
    assert restored.time == cadence.time
    assert restored.shape == cadence.shape
    assert restored.midtime == cadence.midtime
    assert restored.time_at_tstep(Scalar(0.5)) == cadence.time_at_tstep(Scalar(0.5))

##########################################################################################
