##########################################################################################
# tests/cadence/test_reshapedcadence.py
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath import Scalar, Pair, Vector
import oops
from oops.cadence import Metronome, ReshapedCadence

from tests.cadence.test_dualcadence import case_dual_metronome


def case_reshape_roundtrip(oldshape, newshape, arg):
    """A complete there-and-back test of ReshapedCadence._reshape_tstep()."""

    oldstride = np.cumprod((oldshape + (1,))[::-1])[-2::-1]
    newstride = np.cumprod((newshape + (1,))[::-1])[-2::-1]
    oldrank = len(oldshape)
    newrank = len(newshape)

    arg1 = oops.cadence.ReshapedCadence._reshape_tstep(arg,
                                          oldshape, oldstride, oldrank,
                                          newshape, newstride, newrank,
                                          np.prod(oldshape))
    arg2 = oops.cadence.ReshapedCadence._reshape_tstep(arg1,
                                          newshape, newstride, newrank,
                                          oldshape, oldstride, oldrank,
                                          np.prod(oldshape))

    assert arg == arg2

    assert type(arg) is type(arg2)

    if arg.is_int():
        assert arg2.is_int()
    else:
        assert arg2.is_float()

def test_reshapedcadence():
    from oops.cadence.metronome import Metronome

    case_reshape_roundtrip((10,), (10,), Scalar(1))
    case_reshape_roundtrip((10,), (2,5), Scalar(1))
    case_reshape_roundtrip((10,), (2,5), Scalar(1.5))
    case_reshape_roundtrip((10,), (2,5), Scalar(np.arange(10)))
    case_reshape_roundtrip((10,), (2,5), Scalar(np.arange(20)/2.))
    case_reshape_roundtrip((10,), (2,5), Scalar(np.arange(10).reshape(5,2)))
    case_reshape_roundtrip((10,), (2,5), Scalar((np.arange(20)/2.).reshape(2,5,2)))

    case_reshape_roundtrip((2,3,4), (24,), Vector((1,2,3)))
    case_reshape_roundtrip((2,3,4), (24,), Vector((1,2,3.5)))
    case_reshape_roundtrip((2,3,4), (24,), Vector([(1,2,3),(1,2,3.5),(0,0,0.25)]))

    case_reshape_roundtrip((2,3,4), (4,6), Vector((1,2,3)))
    case_reshape_roundtrip((2,3,4), (4,6), Vector((1,2,3.5)))
    case_reshape_roundtrip((2,3,4), (4,6), Vector([(1,2,3),(1,2,3.5),(0,0,0.25)]))

    ######################################################################################
    # Compare a Metronome reshaped to 2-D to an equivalent DualCadence
    # cad1d: 100-101, 102-103, 104-105, ... 198-199.

    cad1d = Metronome(100., 2., 1., 50)

#         long = oops.cadence.Metronome(100., 10., 1., 10)
#         short = oops.cadence.Metronome(0, 2., 1., 5)
#         cad2d = oops.cadence.DualCadence(long, short)
    cad2d = oops.cadence.ReshapedCadence(cad1d, (10,5))

    case_dual_metronome(cad1d, cad2d)

    ######################################################################################
    # Weirdly reshaped case, 100 -> (25,4)
    # 100-110, 110-120, 120-130, ...
    ######################################################################################

    cadence = oops.cadence.Metronome(100., 10., 10., 100)
    reshaped = oops.cadence.ReshapedCadence(cadence, (25,4))

    assert reshaped.is_continuous
    assert reshaped.is_unique

    assert reshaped.time_at_tstep((0,0)) == 100.
    assert reshaped.time_at_tstep((0,1)) == 110.
    assert reshaped.time_at_tstep((1,0)) == 140.
    assert reshaped.time_at_tstep((1,1)) == 150.
    assert reshaped.time_at_tstep((1,1.5)) == 155.

    tstep = Pair([[(0,0),(0,1)],[(1,0),(1,1)]])
    assert reshaped.time_at_tstep(tstep) == [[100,110],[140,150]]

    tstep = Pair([[(0,0),(0,1)],[(1,0),(1,1)]], [[1,0],[0,0]])
    time = reshaped.time_at_tstep(tstep)
    assert np.all(tstep.mask == time.mask)
    assert time[0,0] == Scalar.MASKED
    assert reshaped.time_at_tstep(tstep) == [[Scalar.MASKED,110],[140,150]]

    assert reshaped.tstep_at_time(100.) == (0,0)
    assert reshaped.tstep_at_time(110.) == (0,1)
    assert reshaped.tstep_at_time(140.) == (1,0)
    assert reshaped.tstep_at_time(150.) == (1,1)
    assert reshaped.tstep_at_time(155.) == (1,1.5)

    for i in np.arange(-2., 28., 0.5):
      for j in np.arange(-1., 5., 0.25):
        k = 4*np.floor(i) + j
        (time1a, time1b) = cadence.time_range_at_tstep(k)
        (time2a, time2b) = reshaped.time_range_at_tstep((i,j), remask=True,
                                                        inclusive=False)
        if not time2a.mask:
            assert time1a == time2a
            assert time1b == time2b

            time = reshaped.time_at_tstep((i,j), remask=True)
            tstep = reshaped.tstep_at_time(time)
            assert tstep == (np.floor(i),j)

    ######################################################################################
    # Weirdly reshaped case, 100 -> (25,4), discontinuous
    # [100-108, 116-124, 132-140, 148-156], [164-172, ...], [..., 1684-1692]
    ######################################################################################

    cadence = oops.cadence.Metronome(100., 16., 8., 100)
    reshaped = oops.cadence.ReshapedCadence(cadence, (25,4))
    assert not reshaped.is_continuous
    assert reshaped.is_unique
    assert reshaped.time_at_tstep((0,0)) == 100.
    assert reshaped.time_at_tstep((0,1)) == 116.
    assert reshaped.time_at_tstep((1,0)) == 164.
    assert reshaped.time_at_tstep((1,1)) == 180.
    assert reshaped.time_at_tstep((1,1.5)) == 184.
    assert reshaped.time_at_tstep((1.5,1.5)) == 184.

    new_cadence = reshaped.as_continuous()
    assert new_cadence.is_continuous
    assert new_cadence.time_at_tstep((0,0)) == 100.
    assert new_cadence.time_at_tstep((1,0)) == 164.
    assert new_cadence.time_at_tstep((1,1)) == 180.
    assert new_cadence.time_at_tstep((1,1.5)) == 188
    assert new_cadence.time_at_tstep((1.5,1.5)) == 188

    assert reshaped.tstep_at_time( 99.) == (0,0)
    assert reshaped.tstep_at_time( 99., remask=True) == Pair.MASKED
    assert reshaped.tstep_at_time(100.) == (0,0)
    assert reshaped.tstep_at_time(106.) == (0,0.75)
    assert reshaped.tstep_at_time(116.) == (0,1)
    assert reshaped.tstep_at_time(108.) == (0,1)
    assert reshaped.tstep_at_time(108., remask=True) == Pair.MASKED
    assert reshaped.tstep_at_time(132.) == (0,2)
    assert reshaped.tstep_at_time(148.) == (0,3)
    assert reshaped.tstep_at_time(140.) == (0,3)
    assert reshaped.tstep_at_time(140., remask=True) == Pair.MASKED
    assert reshaped.tstep_at_time(155.) == (0,3.875)
    assert reshaped.tstep_at_time(156.) == (0,4)
    assert reshaped.tstep_at_time(156., remask=True) == Pair.MASKED
    assert reshaped.tstep_at_time(163.99999) == (0,4)
    assert reshaped.tstep_at_time(163.99999, remask=True) == Pair.MASKED
    assert reshaped.tstep_at_time(164.) == (1,0)
    assert reshaped.tstep_at_time(1684) == (24,3)
    assert reshaped.tstep_at_time(1692) == (24,4)
    assert reshaped.tstep_at_time(1692, inclusive=False) == (25,4)
    assert reshaped.tstep_at_time(1692, inclusive=False, remask=True) == Pair.MASKED

    assert reshaped.tstep_range_at_time( 99.) == ((0,0), (1,0))
    assert reshaped.tstep_range_at_time( 99., remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(100.) == ((0,0), (1,1))
    assert reshaped.tstep_range_at_time(106.) == ((0,0), (1,1))
    assert reshaped.tstep_range_at_time(108.) == ((0,0), (1,0))
    assert reshaped.tstep_range_at_time(108., remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(115.999) == ((0,0), (1,0))
    assert reshaped.tstep_range_at_time(115.999, remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(116.) == ((0,1), (1,2))
    assert reshaped.tstep_range_at_time(148.) == ((0,3), (1,4))
    assert reshaped.tstep_range_at_time(140.) == ((0,2), (1,2))
    assert reshaped.tstep_range_at_time(140., remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(156.) == ((0,3), (1,3))
    assert reshaped.tstep_range_at_time(156., remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(163.999) == ((0,3), (1,3))
    assert reshaped.tstep_range_at_time(163.999, remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(164.) == ((1,0), (2,1))
    assert reshaped.tstep_range_at_time(1684) == ((24,3), (25,4))
    assert reshaped.tstep_range_at_time(1692) == ((24,3), (25,4))
    assert reshaped.tstep_range_at_time(1692, inclusive=False) == ((24,3), (25,3))
    assert (reshaped.tstep_range_at_time(1692, inclusive=False, remask=True)[0]
            == Pair.MASKED)

    for i in np.arange(-2., 28., 0.5):
      for j in np.arange(-1., 5., 0.25):
        k = 4*np.floor(i) + j
        (time1a, time1b) = cadence.time_range_at_tstep(k)
        (time2a, time2b) = reshaped.time_range_at_tstep((i,j), remask=True,
                                                        inclusive=False)
        if not time2a.mask:
            assert time1a == time2a
            assert time1b == time2b

            time = reshaped.time_at_tstep((i,j), remask=True)
            tstep = reshaped.tstep_at_time(time)
            assert tstep == (np.floor(i),j)

    ######################################################################################
    # Weirdly reshaped case, 100 -> (25,4), overlapping
    # [100-116, 110-126, 120-136, 130-146], [140-156, ...], [..., 1090-1106]
    ######################################################################################

    cadence = oops.cadence.Metronome(100., 10., 16., 100)
    reshaped = oops.cadence.ReshapedCadence(cadence, (25,4))
    assert reshaped.is_continuous
    assert not reshaped.is_unique
    assert reshaped.time_at_tstep((0,0)) == 100.
    assert reshaped.time_at_tstep((0,1)) == 110.
    assert reshaped.time_at_tstep((1,0)) == 140.
    assert reshaped.time_at_tstep((1,1)) == 150.
    assert reshaped.time_at_tstep((1,1.5)) == 158.
    assert reshaped.time_at_tstep((1.5,1.5)) == 158.

    tstep = Pair([(0,0),(0,1),(1,0),(1,1),(1,1.5),(1.5,1.5)],
                 [True] + 5*[False])
    assert reshaped.time_at_tstep(tstep) == (Scalar.MASKED, 110, 140, 150, 158, 158)
    tstep.insert_deriv('t' , Pair(np.arange(12).reshape(6,2)))
    tstep.insert_deriv('xy', Pair(np.ones((6,2,2)), drank=1))
    assert reshaped.time_at_tstep(tstep) == (Scalar.MASKED, 110, 140, 150, 158, 158)
    assert (reshaped.time_at_tstep(tstep, derivs=True).d_dt
            == (Scalar.MASKED, 48, 80, 112, 144, 176))
    assert (reshaped.time_at_tstep(tstep, derivs=True).d_dxy
            == Scalar(16 * np.ones((6,2)), [True] + 5*[False], drank=1))

    assert reshaped.time_range_at_tstep((0,0)) == (100, 116)
    assert reshaped.time_range_at_tstep((0,1)) == (110, 126)
    assert reshaped.time_range_at_tstep((1,0)) == (140, 156)
    assert reshaped.time_range_at_tstep((1,1)) == (150, 166)
    assert reshaped.time_range_at_tstep((1,1.5)) == (150, 166)
    assert reshaped.time_range_at_tstep((1.5,1.5)) == (150, 166)

    assert (reshaped.time_range_at_tstep(tstep)
            == ((Scalar.MASKED, 110, 140, 150, 150, 150),
                (Scalar.MASKED, 126, 156, 166, 166, 166)))

    assert reshaped.tstep_at_time( 99.) == (0,0)
    assert reshaped.tstep_at_time( 99., remask=True) == Pair.MASKED
    assert reshaped.tstep_at_time(100.) == (0,0)
    assert reshaped.tstep_at_time(108.) == (0,0.5)
    assert reshaped.tstep_at_time(110.) == (0,1)
    assert reshaped.tstep_at_time(140.) == (1,0)
    assert reshaped.tstep_at_time(1090) == (24,3)
    assert reshaped.tstep_at_time(1106) == (24,4)
    assert reshaped.tstep_at_time(1106, inclusive=False) == (25,4)
    assert reshaped.tstep_at_time(1106, inclusive=False, remask=True) == Pair.MASKED

    assert reshaped.tstep_range_at_time( 99.) == ((0,0), (1,0))
    assert reshaped.tstep_range_at_time( 99., remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(100.) == ((0,0), (1,1))
    assert reshaped.tstep_range_at_time(106.) == ((0,0), (1,1))
    assert reshaped.tstep_range_at_time(110.) == ((0,0), (1,2))
    assert reshaped.tstep_range_at_time(139.999) == ((0,3), (1,4))
    with pytest.raises(ValueError):
        reshaped.tstep_range_at_time(140)

    assert reshaped.tstep_range_at_time(1090) == ((24,2), (25,4))
    assert reshaped.tstep_range_at_time(1095.999) == ((24,2), (25,4))
    assert reshaped.tstep_range_at_time(1096) == ((24,3), (25,4))
    assert reshaped.tstep_range_at_time(1106) == ((24,3), (25,4))
    assert reshaped.tstep_range_at_time(1106, inclusive=False) == ((24,3), (25,3))
    assert (reshaped.tstep_range_at_time(1106, inclusive=False, remask=True)[0]
            == Pair.MASKED)


##########################################################################################
# Reshaping a 1-D Metronome into a 2-D cadence
##########################################################################################

TSTART = 0.
TSTRIDE = 10.
TEXP = 10.
STEPS = 12


def _metronome() -> Metronome:
    """A continuous 12-step cadence covering 0 to 120 seconds."""

    return Metronome(tstart=TSTART, tstride=TSTRIDE, texp=TEXP, steps=STEPS)


def _reshaped() -> ReshapedCadence:
    """The same cadence, reshaped to 3 rows of 4 steps."""

    return ReshapedCadence(_metronome(), (3, 4))


def test_reshaping_keeps_the_overall_timing() -> None:
    """The time steps are unchanged; only their indexing differs."""

    reshaped = _reshaped()
    original = _metronome()

    assert reshaped.shape == (3, 4)
    assert reshaped.time == original.time
    assert reshaped.midtime == original.midtime
    assert reshaped.lasttime == original.lasttime


def test_reshaping_keeps_the_continuity_and_uniqueness() -> None:
    """A reshaped cadence samples the same times, so these properties carry over."""

    reshaped = _reshaped()
    original = _metronome()

    assert reshaped.is_continuous == original.is_continuous
    assert reshaped.is_unique == original.is_unique


def test_reshaping_keeps_the_time_strides() -> None:
    """The interval between successive steps is unchanged."""

    reshaped = _reshaped()
    original = _metronome()

    assert reshaped.min_tstride == original.min_tstride
    assert reshaped.max_tstride == original.max_tstride


@pytest.mark.parametrize('row, col, step', [(0, 0, 0), (0, 3, 3), (1, 2, 6),
                                            (2, 3, 11)])
def test_a_two_dimensional_index_selects_the_same_step(row: int, col: int,
                                                       step: int) -> None:
    """The new index (row, col) names the old step row * 4 + col."""

    reshaped = _reshaped()
    original = _metronome()

    assert reshaped.time_at_tstep(Pair((row, col))) \
           == original.time_at_tstep(Scalar(step))


def test_time_range_at_tstep_matches_the_original() -> None:
    """The exposure of a reshaped step is that of the step it stands for."""

    assert _reshaped().time_range_at_tstep(Pair((1, 2))) \
           == _metronome().time_range_at_tstep(Scalar(6))


def test_tstep_at_time_returns_a_pair() -> None:
    """A 2-D cadence is indexed by a Pair rather than a Scalar."""

    tstep = _reshaped().tstep_at_time(Scalar(65.))

    assert isinstance(tstep, Pair)
    assert tstep == Pair((1., 2.5))


def test_tstep_at_time_inverts_time_at_tstep() -> None:
    """Converting a time to an index and back returns the original time."""

    reshaped = _reshaped()
    time = Scalar(65.)

    assert reshaped.time_at_tstep(reshaped.tstep_at_time(time)) == time


def test_tstep_range_at_time_selects_one_step() -> None:
    """Exactly one step of a continuous, unique cadence is active at a time."""

    (first, last) = _reshaped().tstep_range_at_time(Scalar(65.))

    assert first == Pair((1, 2))
    assert last == Pair((2, 3))


def test_time_is_inside_matches_the_original() -> None:
    """Reshaping does not change which times the cadence samples."""

    times = Scalar([-1., 0., 65., 120., 121.])

    assert _reshaped().time_is_inside(times) == _metronome().time_is_inside(times)


def test_time_shift_moves_the_whole_cadence() -> None:
    """A time shift moves both ends and keeps the new shape."""

    shifted = _reshaped().time_shift(100.)

    assert shifted.time == (TSTART + 100., TSTART + STEPS * TSTRIDE + 100.)
    assert shifted.shape == (3, 4)


def test_as_continuous_of_an_already_continuous_cadence() -> None:
    """A continuous cadence stays continuous."""

    assert _reshaped().as_continuous().is_continuous


def test_a_gapped_cadence_can_be_forced_continuous() -> None:
    """as_continuous closes the gaps between the time steps."""

    gapped = ReshapedCadence(Metronome(tstart=0., tstride=10., texp=5., steps=12),
                             (3, 4))

    assert not gapped.is_continuous
    assert gapped.as_continuous().is_continuous


def test_reshaping_to_one_dimension_is_allowed() -> None:
    """A 1-D shape is a valid target."""

    assert ReshapedCadence(_metronome(), (STEPS,)).shape == (STEPS,)


def test_reshaping_rejects_an_incompatible_size() -> None:
    """The new shape must hold exactly as many steps as the old one."""

    with pytest.raises(ValueError, match='size and shape are incompatible'):
        ReshapedCadence(_metronome(), (5, 5))


def test_reshaping_rejects_three_dimensions() -> None:
    """A reshaped cadence may have at most two dimensions."""

    with pytest.raises(ValueError, match='3-D cadences are not supported'):
        ReshapedCadence(_metronome(), (2, 2, 3))


def test_reshaped_cadence_survives_a_pickle_round_trip() -> None:
    """Pickling restores the underlying cadence and the new shape."""

    reshaped = _reshaped()
    restored = pickle.loads(pickle.dumps(reshaped))

    assert isinstance(restored, ReshapedCadence)
    assert restored.shape == reshaped.shape
    assert restored.time == reshaped.time
    assert restored.time_at_tstep(Pair((1, 2))) == reshaped.time_at_tstep(Pair((1, 2)))

##########################################################################################
