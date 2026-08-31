##########################################################################################
# tests/cadence/test_instant.py
##########################################################################################

import pickle

import pytest

from polymath import Scalar
import oops


def test_instant_defines_every_cadence_attribute() -> None:
    """An Instant fills in all eight attributes the Cadence contract requires."""

    cad = oops.cadence.Instant([100., 110., 130.])

    assert cad.shape == (3,)
    assert cad.time == (100., 130.)
    assert cad.midtime == 115.
    assert cad.lasttime == 130.
    assert cad.is_continuous is False
    assert cad.is_unique is True
    assert cad.min_tstride == 10.
    assert cad.max_tstride == 20.


def test_instant_is_not_unique_when_a_time_repeats() -> None:
    """Two time steps sharing one time make the cadence non-unique."""

    assert oops.cadence.Instant([100., 110., 100.]).is_unique is False


def test_instant_of_a_single_time_has_zero_strides() -> None:
    """A lone time step has no neighbor, so both strides are zero."""

    cad = oops.cadence.Instant(100.)

    assert cad.shape == ()
    assert cad.time == (100., 100.)
    assert cad.min_tstride == 0.
    assert cad.max_tstride == 0.
    assert cad.is_unique is True


def test_instant_ignores_masked_times() -> None:
    """Masked times take no part in the time limits or in the strides."""

    cad = oops.cadence.Instant(Scalar([100., 500., 110.], [False, True, False]))

    assert cad.time == (100., 110.)
    assert cad.max_tstride == 10.


def test_instant_requires_one_unmasked_time() -> None:
    """A fully masked input leaves no times to define the cadence."""

    with pytest.raises(ValueError, match='at least one unmasked time'):
        oops.cadence.Instant(Scalar([100., 110.], [True, True]))


def test_instant_survives_a_pickle_round_trip() -> None:
    """__getstate__ returns a tuple, so the cadence can be pickled."""

    cad = pickle.loads(pickle.dumps(oops.cadence.Instant([100., 110., 130.])))

    assert cad.time == (100., 130.)
    assert cad.max_tstride == 20.

##########################################################################################
