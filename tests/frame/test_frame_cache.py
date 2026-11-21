##########################################################################################
# tests/frame/test_frame_cache.py: the process-wide Frame caches and unregistered Frames
##########################################################################################

import gc
import weakref

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from polymath   import Matrix3, Vector3
from oops.frame import Cmatrix, Frame, Navigation

# Enough Frames that a cache keeping each one would be seen to grow
COUNT = 40


def _subclasses(cls: type) -> Iterator[type]:
    """Every subclass of `cls`, at any depth."""

    subclasses: list[type] = cls.__subclasses__()
    for subclass in subclasses:
        yield subclass
        yield from _subclasses(subclass)


# The Frame classes of oops that keep a pool of wayframes of their own
POOLED_FRAME_CLASSES = sorted({cls for cls in _subclasses(Frame)
                               if cls.__module__.startswith('oops.')
                               and '_WAYFRAMES' in vars(cls)},
                              key=lambda cls: cls.__name__)


@pytest.fixture(autouse=True)
def _registry_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test its own copy of the Frame registry and cache, so that the IDs it
    registers are gone when it ends."""

    monkeypatch.setattr(Frame, '_FRAME_REGISTRY', dict(Frame._FRAME_REGISTRY))
    monkeypatch.setattr(Frame, '_FRAME_CACHE', dict(Frame._FRAME_CACHE))


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


def test_unregistered_frames_leave_the_frame_cache_unchanged() -> None:
    assert _cache_growth(lambda k: Cmatrix(_matrix(k))) == 0


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


def test_a_wayframe_given_an_id_later_frees_the_frames_related_to_it() -> None:
    # The second Frame shares the first one's wayframe and gives it an ID, so the
    # wayframe stays alive through the registry from then on
    first = Cmatrix(_matrix(7000))
    Cmatrix(_matrix(7000), frame_id='TEST_FRAME_CACHE_GIVEN_AN_ID')
    refs = []
    for k in range(COUNT):
        frame = Cmatrix(_matrix(k))
        first.wrt(frame)
        refs.append(weakref.ref(frame))
    del frame
    gc.collect()
    assert sum(ref() is not None for ref in refs) == 0


def test_an_unregistered_frame_gives_the_same_frame_wrt_j2000_twice() -> None:
    frame = Navigation((1e-4, 0., 0.), Cmatrix(_matrix(4000)))
    assert frame.wrt(Frame.J2000) is frame.wrt(Frame.J2000)


def test_every_frame_class_with_a_pool_is_found() -> None:
    assert {Cmatrix, Navigation} <= set(POOLED_FRAME_CLASSES)


@pytest.mark.parametrize('cls', POOLED_FRAME_CLASSES, ids=lambda cls: cls.__name__)
def test_every_pool_of_wayframes_holds_its_frames_weakly(cls: type) -> None:
    assert isinstance(vars(cls)['_WAYFRAMES'], weakref.WeakValueDictionary)


class _VanishingPool(weakref.WeakValueDictionary[Any, Any]):
    """A pool whose entry the garbage collector frees between the test for the key and
    the read that follows it."""

    def __contains__(self, key: object) -> bool:
        return True

    def __getitem__(self, key: Any) -> Any:
        raise KeyError(key)


def test_a_frame_is_reregistered_when_a_pool_entry_is_freed_during_the_lookup(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Cmatrix, '_WAYFRAMES', _VanishingPool())
    frame = Cmatrix(_matrix(5000))
    frame._reregister()
    assert frame.wayframe is frame

##########################################################################################
