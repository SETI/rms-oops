##########################################################################################
# tests/path/test_path_cache.py: the process-wide Path caches and unregistered Paths
##########################################################################################

import gc

from collections.abc import Callable

from polymath   import Matrix3, Vector3
from oops.frame import Cmatrix, Frame
from oops.path  import FixedPath, Path

# Enough Paths that a cache keeping each one would be seen to grow
COUNT = 40


def _position(k: int) -> Vector3:
    """A position that differs for each value of `k`."""

    return Vector3((1.e5, 2.e5 + k, -3.e5))


def _matrix(k: int) -> Matrix3:
    """A rotation matrix that differs for each value of `k`."""

    return Matrix3.twovec(Vector3((1., 1e-3 * (k + 1), 2e-3 * k)), 2,
                          Vector3((0., 0., 1.)), 0)


def _cache_growth(build: Callable[[int], object]) -> int:
    """The growth of Path._PATH_CACHE after `build` is called COUNT times.

    The objects built are dropped and the garbage collector is run before the cache is
    measured. One object is built beforehand, so that whatever the first one adds to the
    cache, such as a registered Path, is there before the cache is first measured.
    """

    build(-1)
    gc.collect()
    before = len(Path._PATH_CACHE)
    for k in range(COUNT):
        build(k)
    gc.collect()
    return len(Path._PATH_CACHE) - before


def test_unregistered_paths_leave_the_path_cache_unchanged() -> None:
    assert _cache_growth(lambda k: FixedPath(_position(k), Path.SSB, Frame.J2000)) == 0


def test_a_path_relative_to_an_unregistered_path_leaves_the_cache_unchanged() -> None:
    def build(k: int) -> Path:
        origin = FixedPath(_position(k), Path.SSB, Frame.J2000)
        return FixedPath(_position(0), origin, Frame.J2000).wrt(Path.SSB)

    assert _cache_growth(build) == 0


def test_a_path_in_an_unregistered_frame_leaves_the_cache_unchanged() -> None:
    assert _cache_growth(lambda k: Path.SSB.wrt(Path.SSB, Cmatrix(_matrix(k)))) == 0


def test_unregistered_paths_leave_the_pool_of_waypoints_unchanged() -> None:
    gc.collect()
    before = len(FixedPath._WAYPOINTS)
    for k in range(COUNT):
        FixedPath(_position(1000 + k), Path.SSB, Frame.J2000)
    gc.collect()
    assert len(FixedPath._WAYPOINTS) == before


def test_identical_unregistered_paths_share_a_waypoint() -> None:
    first = FixedPath(_position(2000), Path.SSB, Frame.J2000)
    assert FixedPath(_position(2000), Path.SSB, Frame.J2000).waypoint is first.waypoint


def test_a_path_identical_to_a_registered_one_shares_its_waypoint() -> None:
    FixedPath(_position(3000), Path.SSB, Frame.J2000, path_id='TEST_PATH_CACHE_ID')
    gc.collect()
    path = FixedPath(_position(3000), Path.SSB, Frame.J2000)
    assert path.waypoint.path_id == 'TEST_PATH_CACHE_ID'


def test_an_unregistered_path_gives_the_same_path_wrt_the_ssb_twice() -> None:
    origin = FixedPath(_position(4000), Path.SSB, Frame.J2000)
    path = FixedPath(_position(0), origin, Frame.J2000)
    assert path.wrt(Path.SSB) is path.wrt(Path.SSB)

##########################################################################################
