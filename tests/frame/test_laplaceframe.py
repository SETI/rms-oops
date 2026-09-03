##########################################################################################
# tests/frame/test_laplaceframe.py
##########################################################################################

import pickle
from collections.abc import Iterator
from typing import cast

import pytest

from polymath     import Scalar, Vector3
from oops.body    import Body
from oops.frame   import Frame, LaplaceFrame
from oops.gravity import Gravity
from oops.path    import KeplerPath, Path

_A = 140000.                    # semimajor axis of the test orbit, km
TILT = 0.05                     # radians
EPOCH = 0.


@pytest.fixture(scope='module', autouse=True)
def _solar_system() -> Iterator[None]:
    """The bodies of the solar system, which a KeplerPath needs for its gravity."""

    Body.reset_registry()
    Body._undefine_solar_system()
    Body.define_solar_system('2000-01-01', '2010-01-01')

    yield

    Frame._reset_caches()
    Path._reset_caches()
    Body.reset_registry()


def _orbit(**kwargs) -> KeplerPath:
    """A KeplerPath about Saturn, inclined to Saturn's ring plane."""

    saturn = Gravity.lookup('SATURN')
    elements = (_A, 1., saturn.n(_A),
                0.2, 3., saturn.dperi_dt(_A),
                0.1, 5., saturn.dnode_dt(_A))

    return KeplerPath(Body.lookup('SATURN'), EPOCH, elements, **kwargs)


def _pole(tilt: float, time: float = 0.) -> Vector3:
    """The Z-axis of a Laplace Plane, expressed in the reference frame."""

    frame = LaplaceFrame(_orbit(), tilt=tilt)
    matrix = frame.transform_at_time(Scalar(time)).matrix

    return cast(Vector3, matrix.unrotate(Vector3.ZAXIS))


def _separation(one: Vector3, other: Vector3) -> float:
    """The angle in radians between two poles."""

    return float(cast(Scalar, one.sep(other)).vals)


def test_laplaceframe_is_defined_relative_to_j2000() -> None:
    """The orbit's own reference frame places the Laplace Plane in J2000."""

    assert LaplaceFrame(_orbit(), tilt=TILT).reference == Frame.J2000.wayframe


def test_a_zero_tilt_leaves_the_pole_where_it_is() -> None:
    """With no tilt, the Z-axis is the planet's own pole."""

    assert _separation(_pole(0.), _pole(0.)) == pytest.approx(0., abs=1.e-12)


@pytest.mark.parametrize('tilt', [0.02, 0.05, 0.1])
def test_the_pole_is_rotated_by_the_tilt_angle(tilt: float) -> None:
    """The Z-axis is the planet's pole rotated by exactly `tilt`."""

    assert _separation(_pole(tilt), _pole(0.)) == pytest.approx(tilt, abs=1.e-9)


def test_a_negative_tilt_rotates_the_other_way() -> None:
    """The tilt angles should be negative for a retrograde orbit like Triton's."""

    assert _separation(_pole(-TILT), _pole(0.)) == pytest.approx(TILT, abs=1.e-9)
    assert _separation(_pole(TILT), _pole(-TILT)) \
           == pytest.approx(2. * TILT, abs=1.e-9)


def test_the_pole_precesses_with_the_orbit() -> None:
    """The rotation follows the orbit's regressing node, so the pole moves in time."""

    early = _pole(TILT, time=0.)
    late = _pole(TILT, time=1.e7)

    assert _separation(early, late) > 1.e-6


def test_the_tilt_is_fixed_as_the_pole_precesses() -> None:
    """The angle from the planet's pole is a constant of the Laplace Plane."""

    assert _separation(_pole(TILT, time=1.e7), _pole(0., time=1.e7)) \
           == pytest.approx(TILT, abs=1.e-9)


def test_laplaceframe_transform_is_cached() -> None:
    """A transform is cached, so asking twice at one time gives the same matrix."""

    frame = LaplaceFrame(_orbit(), tilt=TILT)

    assert frame.transform_at_time(Scalar(1000.)).matrix \
           == frame.transform_at_time(Scalar(1000.)).matrix


def test_laplaceframe_cache_size_may_be_set() -> None:
    """A small cache still returns the same transforms, just holding fewer of them."""

    small = LaplaceFrame(_orbit(), tilt=TILT, cache_size=2)
    large = LaplaceFrame(_orbit(), tilt=TILT, cache_size=100)

    for time in (0., 1000., 2000., 3000., 0.):
        assert small.transform_at_time(Scalar(time)).matrix \
               == large.transform_at_time(Scalar(time)).matrix


def test_laplaceframe_accepts_an_array_of_times() -> None:
    """A shaped time gives a shaped Transform."""

    frame = LaplaceFrame(_orbit(), tilt=TILT)

    assert frame.transform_at_time(Scalar([0., 1000., 2000.])).shape == (3,)


def test_laplaceframe_accepts_an_orbit_id() -> None:
    """The orbit may be named by the ID of a registered KeplerPath."""

    _orbit(path_id='TEST_LAPLACE_ORBIT')
    frame = LaplaceFrame('TEST_LAPLACE_ORBIT', tilt=TILT)

    assert frame.transform_at_time(Scalar(0.)).matrix \
           == LaplaceFrame(_orbit(), tilt=TILT).transform_at_time(Scalar(0.)).matrix


def test_laplaceframe_rejects_an_unregistered_orbit_id() -> None:
    """An orbit ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        LaplaceFrame('NOT_A_REGISTERED_PATH', tilt=TILT)


def test_laplaceframe_auto_generated_frame_id() -> None:
    """A frame_id of "+" appends "_LAPLACE" to the orbit's Path ID."""

    _orbit(path_id='TEST_LAPLACE_PLUS')
    frame = LaplaceFrame('TEST_LAPLACE_PLUS', tilt=TILT, frame_id='+')

    assert frame.frame_id == 'TEST_LAPLACE_PLUS_LAPLACE'


def test_laplaceframe_registration() -> None:
    """A frame_id registers the Frame under that name."""

    frame = LaplaceFrame(_orbit(), tilt=TILT, frame_id='TEST_LAPLACE_FRAME')

    assert frame.frame_id == 'TEST_LAPLACE_FRAME'
    assert Frame.as_frame('TEST_LAPLACE_FRAME').frame_id == 'TEST_LAPLACE_FRAME'


def test_laplaceframe_pickle() -> None:
    """Pickling restores the orbit and the tilt."""

    frame = LaplaceFrame(_orbit(), tilt=TILT)
    restored = pickle.loads(pickle.dumps(frame))
    time = Scalar(5000.)

    assert isinstance(restored, LaplaceFrame)
    assert restored.transform_at_time(time).matrix == frame.transform_at_time(time).matrix


def test_laplaceframe_getstate_roundtrip() -> None:
    """The state captured by __getstate__ fully restores the object."""

    frame = LaplaceFrame(_orbit(), tilt=TILT)
    state = frame.__getstate__()

    copied = Frame.__new__(LaplaceFrame)
    copied.__setstate__(state)
    time = Scalar(5000.)

    assert copied.transform_at_time(time).matrix == frame.transform_at_time(time).matrix

##########################################################################################
