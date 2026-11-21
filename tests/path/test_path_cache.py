##########################################################################################
# tests/path/test_path_cache.py: the process-wide Path caches and unregistered Paths
##########################################################################################

import gc
import weakref

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from polymath   import Matrix3, Vector3
from oops.frame import Cmatrix, Frame
from oops.path  import FixedPath, LinearPath, NullPath, Path, PathShift

# Enough Paths that a cache keeping each one would be seen to grow
COUNT = 40


def _subclasses(cls: type) -> Iterator[type]:
    """Every subclass of `cls`, at any depth."""

    subclasses: list[type] = cls.__subclasses__()
    for subclass in subclasses:
        yield subclass
        yield from _subclasses(subclass)


# The Path classes of oops that keep a pool of waypoints of their own
POOLED_PATH_CLASSES = sorted({cls for cls in _subclasses(Path)
                              if cls.__module__.startswith('oops.')
                              and '_WAYPOINTS' in vars(cls)},
                             key=lambda cls: cls.__name__)


@pytest.fixture(autouse=True)
def _registry_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test its own copy of the Path and Frame registries and caches, so that
    the IDs it registers are gone when it ends."""

    monkeypatch.setattr(Path, '_PATH_REGISTRY', dict(Path._PATH_REGISTRY))
    monkeypatch.setattr(Path, '_PATH_CACHE', dict(Path._PATH_CACHE))
    monkeypatch.setattr(Frame, '_FRAME_REGISTRY', dict(Frame._FRAME_REGISTRY))
    monkeypatch.setattr(Frame, '_FRAME_CACHE', dict(Frame._FRAME_CACHE))


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


def test_a_null_path_in_a_frame_sharing_a_wayframe_leaves_the_cache_unchanged() -> None:
    # Each Frame built in the loop repeats the definition of this one and shares its
    # wayframe, which stays in use until the test ends
    wayframe = Cmatrix(_matrix(6000))
    assert Cmatrix(_matrix(6000)).wayframe is wayframe
    assert _cache_growth(lambda k: NullPath(Path.SSB, Cmatrix(_matrix(6000)))) == 0


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


def test_a_waypoint_given_an_id_later_frees_the_frames_used_with_it() -> None:
    # The second Path shares the first one's waypoint and gives it an ID, so the waypoint
    # stays alive through the registry from then on
    first = FixedPath(_position(7000), Path.SSB, Frame.J2000)
    FixedPath(_position(7000), Path.SSB, Frame.J2000, path_id='TEST_PATH_CACHE_GIVEN_AN_ID')
    refs = []
    for k in range(COUNT):
        frame = Cmatrix(_matrix(k))
        first.wrt(Path.SSB, frame)
        refs.append(weakref.ref(frame))
    del frame
    gc.collect()
    assert sum(ref() is not None for ref in refs) == 0


def test_a_wayframe_given_an_id_later_keeps_the_paths_cached_in_it() -> None:
    # The Path relative to the SSB in the unregistered Frame is cached by that Frame
    # until the second Frame shares its wayframe and gives it an ID; the Path is then
    # looked up again in the process-wide Path cache
    path = FixedPath(_position(8000), Path.SSB, Frame.J2000, path_id='TEST_PATH_CACHE_PATH')
    frame = Cmatrix(_matrix(8000))
    linked = path.wrt(Path.SSB, frame)
    Cmatrix(_matrix(8000), frame_id='TEST_PATH_CACHE_FRAME')
    assert path.wrt(Path.SSB, frame) is linked


def test_the_ssb_relative_to_a_registered_path_frees_the_frame_it_is_given() -> None:
    # The SSB relative to the registered Path is that Path reversed and then rotated into
    # the unregistered Frame
    path = FixedPath(_position(9000), Path.SSB, Frame.J2000, path_id='TEST_PATH_CACHE_FREE')
    refs = []
    for k in range(COUNT):
        frame = Cmatrix(_matrix(k))
        Path.SSB.wrt(path, frame)
        refs.append(weakref.ref(frame))
    del frame
    gc.collect()
    assert sum(ref() is not None for ref in refs) == 0


def test_a_path_rotated_into_one_frame_is_not_reused_for_another() -> None:
    # The SSB relative to the registered Path in the first Frame is a rotated Path. The
    # reverse request in the second Frame must not reverse that rotated Path as though
    # its state were in the frame of the registered Path.
    path = FixedPath(_position(9100), Path.SSB, Frame.J2000, path_id='TEST_PATH_CACHE_ROT')
    Path.SSB.wrt(path, Cmatrix(_matrix(9101)))
    frame = Cmatrix(_matrix(9102))
    expected = frame.transform_at_time(0.).rotate(_position(9100))
    result = path.wrt(Path.SSB, frame).event_at_time(0.).pos
    assert (result - expected).norm().vals < 1e-6


def test_an_unregistered_path_gives_the_same_path_wrt_the_ssb_twice() -> None:
    origin = FixedPath(_position(4000), Path.SSB, Frame.J2000)
    path = FixedPath(_position(0), origin, Frame.J2000)
    assert path.wrt(Path.SSB) is path.wrt(Path.SSB)


def test_every_path_class_with_a_pool_is_found() -> None:
    assert {FixedPath, LinearPath, PathShift} <= set(POOLED_PATH_CLASSES)


@pytest.mark.parametrize('cls', POOLED_PATH_CLASSES, ids=lambda cls: cls.__name__)
def test_every_pool_of_waypoints_holds_its_paths_weakly(cls: type) -> None:
    assert isinstance(vars(cls)['_WAYPOINTS'], weakref.WeakValueDictionary)


class _VanishingPool(weakref.WeakValueDictionary[Any, Any]):
    """A pool whose entry the garbage collector frees between the test for the key and
    the read that follows it."""

    def __contains__(self, key: object) -> bool:
        return True

    def __getitem__(self, key: Any) -> Any:
        raise KeyError(key)


def test_a_path_is_reregistered_when_a_pool_entry_is_freed_during_the_lookup(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FixedPath, '_WAYPOINTS', _VanishingPool())
    path = FixedPath(_position(5000), Path.SSB, Frame.J2000)
    path._reregister()
    assert path.waypoint is path

##########################################################################################
