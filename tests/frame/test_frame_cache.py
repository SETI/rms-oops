##########################################################################################
# tests/frame/test_frame_cache.py: the process-wide Frame caches and unregistered Frames
##########################################################################################

import gc

from collections.abc import Callable

from polymath   import Matrix3, Vector3
from oops.frame import Cmatrix, Frame, Navigation, SpiceErrorFrame

# Enough Frames that a cache keeping each one would be seen to grow
COUNT = 40


def _matrix(k: int) -> Matrix3:
    """A rotation matrix that differs for each value of `k`."""

    return Matrix3.twovec(Vector3((1., 1e-3 * (k + 1), 2e-3 * k)), 2,
                          Vector3((0., 0., 1.)), 0)


def _cache_growth(build: Callable[[int], object]) -> int:
    """The growth of Frame._FRAME_CACHE after `build` is called COUNT times.

    The objects built are dropped and the garbage collector is run before the cache is
    measured. One object is built beforehand, so that whatever the first one adds to the
    cache, such as a registered Frame, is there before the cache is first measured.
    """

    build(-1)
    gc.collect()
    before = len(Frame._FRAME_CACHE)
    for k in range(COUNT):
        build(k)
    gc.collect()
    return len(Frame._FRAME_CACHE) - before


def _spice_error_frame(k: int) -> SpiceErrorFrame:
    try:
        raise RuntimeError(f'SPICE(CKINSUFFDATA) -- no pointing {k}')
    except RuntimeError as err:
        return SpiceErrorFrame(err, description=f'no pointing {k}')


def test_unregistered_frames_leave_the_frame_cache_unchanged() -> None:
    assert _cache_growth(lambda k: Cmatrix(_matrix(k))) == 0


def test_spice_error_frames_leave_the_frame_cache_unchanged() -> None:
    assert _cache_growth(_spice_error_frame) == 0


def test_a_frame_relative_to_an_unregistered_frame_leaves_the_cache_unchanged() -> None:
    assert _cache_growth(lambda k: Navigation((1e-4, 0., 0.),
                                              Cmatrix(_matrix(k))).wrt(Frame.J2000)) == 0


def test_j2000_relative_to_an_unregistered_frame_leaves_the_cache_unchanged() -> None:
    assert _cache_growth(lambda k: Frame.J2000.wrt(Cmatrix(_matrix(k)))) == 0


def test_a_registered_frame_relative_to_an_unregistered_one_leaves_it_unchanged() -> None:
    registered = Cmatrix(_matrix(-2), frame_id='TEST_FRAME_CACHE_REGISTERED')
    assert _cache_growth(lambda k: registered.wrt(Cmatrix(_matrix(k)))) == 0


def test_unregistered_frames_leave_the_pool_of_wayframes_unchanged() -> None:
    gc.collect()
    before = len(Cmatrix._WAYFRAMES)
    for k in range(COUNT):
        Cmatrix(_matrix(1000 + k))
    gc.collect()
    assert len(Cmatrix._WAYFRAMES) == before


def test_identical_unregistered_frames_share_a_wayframe() -> None:
    first = Cmatrix(_matrix(2000))
    assert Cmatrix(_matrix(2000)).wayframe is first.wayframe


def test_a_frame_identical_to_a_registered_one_shares_its_wayframe() -> None:
    Cmatrix(_matrix(3000), frame_id='TEST_FRAME_CACHE_ID')
    gc.collect()
    assert Cmatrix(_matrix(3000)).wayframe.frame_id == 'TEST_FRAME_CACHE_ID'


def test_an_unregistered_frame_gives_the_same_frame_wrt_j2000_twice() -> None:
    frame = Navigation((1e-4, 0., 0.), Cmatrix(_matrix(4000)))
    assert frame.wrt(Frame.J2000) is frame.wrt(Frame.J2000)

##########################################################################################
