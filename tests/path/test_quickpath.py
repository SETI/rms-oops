##########################################################################################
# oops/path/quickpath.py: Subclass QuickPath of class Path
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath     import Scalar, Vector3
from oops.body    import Body
from oops.config  import LOGGING, QUICK
from oops.frame   import Frame
from oops.gravity import Gravity
from oops.path    import FixedPath, KeplerPath, Path, QuickPath, SpicePath


@pytest.fixture(autouse=True)
def _reset_body_registry(core_kernels):
    yield

    Body.reset_registry()

def test_quickpath():
    np.random.seed(9033)

    mars = SpicePath('MARS', 'SSB')
    epoch = 1.e8
    time = Scalar(epoch + np.arange(0., 100., 0.01))

    ######################################################################################
    # Tabulating a Path does not spawn a second, nested QuickPath
    ######################################################################################

    # SpicePath quickens itself when handed an array of times, so the tabulation
    # inside QuickPath must not re-enter that machinery. The span is short enough
    # that a QuickPath of the tabulation times would otherwise be judged worthwhile.
    short_time = Scalar(epoch + np.arange(0., 0.5, 0.001))
    assert isinstance(mars.quick_path(short_time, quick={}), QuickPath)
    assert len(mars._quickpaths) == 1

    ######################################################################################
    # A Path whose state is fixed in time is never tabulated
    ######################################################################################

    # FixedPath returns one position and velocity regardless of the times requested,
    # so a QuickPath could not interpolate it and would gain nothing if it could
    fixed = FixedPath(Vector3((1.e3, 0., 0.)), mars, path_id='fixed_path')
    assert not fixed._USE_QUICKPATHS
    assert fixed.quick_path(time, quick={}) is fixed

    with pytest.raises(ValueError):
        QuickPath(fixed, epoch, epoch + 100., QUICK.dictionary)

    ######################################################################################
    # A composite Path inherits the opt-in of the paths it combines
    ######################################################################################

    # The fixed offset above contributes nothing, but the SpicePath it is measured
    # from does, so the composite is worth tabulating
    linked = fixed.wrt(Path.SSB, Frame.J2000)
    assert linked._USE_QUICKPATHS
    quick = linked.quick_path(time, quick={})
    assert isinstance(quick, QuickPath)

    exact = linked.event_at_time(time, quick=False)
    interpolated = quick.event_at_time(time, quick=False)
    assert np.max(np.abs((interpolated.pos - exact.pos).norm().vals)) < 1.e-6

    ######################################################################################
    # A tabulation of a fittable Path is redone after that Path is re-fit
    ######################################################################################

    Body._undefine_solar_system()
    Body.define_solar_system('2000-01-01', '2010-01-01')

    a = 140000.
    saturn = Gravity.lookup('SATURN')
    elements = [a, 1., saturn.n(a), 0.2, 3., saturn.dperi_dt(a),
                0.1, 5., saturn.dnode_dt(a)]
    kepler = KeplerPath(Body.lookup('SATURN'), 0., elements, path_id='fitted_kepler')
    assert kepler._USE_QUICKPATHS

    quick = kepler.quick_path(time, quick={})
    assert isinstance(quick, QuickPath)

    # Enlarge the orbit by 10,000 km, which moves the body much farther than any
    # interpolation error
    elements[0] = a + 10000.
    kepler.set_params(np.array(elements))
    exact = kepler.event_at_time(time, quick=False)

    # The same QuickPath is handed back, but tabulated afresh
    reused = kepler.quick_path(time, quick={})
    assert reused is quick
    error = (reused.event_at_time(time, quick=False).pos - exact.pos).norm()
    assert np.max(error.vals) < 1.e-3


##########################################################################################
# QuickPath.for_path: creation, re-use, and extension
##########################################################################################

_EPOCH = 1.e8


def _dense_times(start: float, stop: float) -> Scalar:
    """Enough closely-spaced times that a QuickPath is worth building."""

    return Scalar(_EPOCH + np.arange(start, stop, 0.01))


def test_for_path_builds_a_quickpath_when_it_is_worthwhile(core_kernels) -> None:
    """A dense set of times justifies the overhead of tabulating the path."""

    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})

    assert isinstance(quick, QuickPath)


def test_for_path_returns_the_path_when_quick_is_false(core_kernels) -> None:
    """quick=False creates no QuickPath and returns the path itself."""

    mars = SpicePath('MARS', 'SSB')

    assert QuickPath.for_path(mars, _dense_times(0., 100.), quick=False) is mars


def test_for_path_accepts_a_tuple_of_time_limits(core_kernels) -> None:
    """The times may be given simply as (tmin, tmax)."""

    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, (_EPOCH, _EPOCH + 100.), quick={})

    assert isinstance(quick, (QuickPath, SpicePath))


def test_for_path_saves_the_quickpath_on_the_path(core_kernels) -> None:
    """A QuickPath is saved in the list inside path._quickpaths."""

    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})

    assert quick in mars._quickpaths


def test_for_path_reuses_a_covering_quickpath(core_kernels) -> None:
    """A pre-existing QuickPath that covers the range is returned as it is."""

    mars = SpicePath('MARS', 'SSB')
    first = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})
    second = QuickPath.for_path(mars, _dense_times(20., 80.), quick={})

    assert second is first
    assert len(mars._quickpaths) == 1


def test_for_path_extends_a_partially_covering_quickpath(core_kernels) -> None:
    """A QuickPath covering part of the range is extended rather than duplicated."""

    mars = SpicePath('MARS', 'SSB')
    first = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})
    original_end = first._times[-1]

    second = QuickPath.for_path(mars, _dense_times(50., 200.), quick={})

    assert second is first
    assert len(mars._quickpaths) == 1
    assert second._times[-1] > original_end


def test_for_path_extends_backward(core_kernels) -> None:
    """Extension works in the earlier direction too."""

    mars = SpicePath('MARS', 'SSB')
    first = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})
    original_start = first._times[0]

    first.extend(_EPOCH - 100., _EPOCH + 100.)

    assert first._times[0] < original_start


def test_for_path_builds_a_second_quickpath_for_a_distant_time(core_kernels) -> None:
    """A range nowhere near the first tabulation gets a QuickPath of its own."""

    mars = SpicePath('MARS', 'SSB')
    first = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})
    second = QuickPath.for_path(mars, Scalar(5.e8 + np.arange(0., 100., 0.01)), quick={})

    assert second is not first
    assert len(mars._quickpaths) == 2


def test_quickpath_matches_the_path_it_emulates(core_kernels) -> None:
    """Interpolation reproduces the underlying path to well within a kilometer."""

    np.random.seed(3391)
    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})

    times = Scalar(_EPOCH + np.random.rand(50) * 100.)
    error = abs(quick.event_at_time(times).pos - mars.event_at_time(times).pos).max()

    assert error < 1.e-3


def test_quickpath_velocity_matches_the_path(core_kernels) -> None:
    """The interpolated velocity matches the underlying path as well."""

    np.random.seed(7712)
    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})

    times = Scalar(_EPOCH + np.random.rand(50) * 100.)
    error = abs(quick.event_at_time(times).vel - mars.event_at_time(times).vel).max()

    assert error < 1.e-6


def test_extend_widens_the_tabulated_interval(core_kernels) -> None:
    """extend() re-tabulates the path over the wider interval."""

    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})

    quick.extend(_EPOCH - 200., _EPOCH + 300.)

    assert quick._times[0] <= _EPOCH - 200.
    assert quick._times[-1] >= _EPOCH + 300.


def test_extend_to_a_narrower_interval_changes_nothing(core_kernels) -> None:
    """An interval already covered leaves the tabulation alone."""

    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})
    before = (quick._times[0], quick._times[-1])

    quick.extend(_EPOCH + 20., _EPOCH + 80.)

    assert (quick._times[0], quick._times[-1]) == before


def test_extended_quickpath_is_still_accurate(core_kernels) -> None:
    """The path is still reproduced accurately over the extended interval."""

    np.random.seed(5150)
    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})
    quick.extend(_EPOCH - 200., _EPOCH + 300.)

    times = Scalar(_EPOCH + np.random.rand(50) * 500. - 200.)
    error = abs(quick.event_at_time(times).pos - mars.event_at_time(times).pos).max()

    assert error < 1.e-3


def test_quickpath_rejects_a_shaped_path(core_kernels) -> None:
    """A QuickPath can only emulate a path of shape ()."""

    shaped = FixedPath(Vector3([(1.e5, 0., 0.), (0., 2.e5, 0.)]), Path.SSB)

    with pytest.raises(ValueError):
        QuickPath(shaped, _EPOCH, _EPOCH + 100., QUICK.dictionary)


def test_quickpath_rejects_a_quickpath(core_kernels) -> None:
    """A QuickPath cannot be built on top of another QuickPath."""

    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})

    with pytest.raises(ValueError):
        QuickPath(quick, _EPOCH, _EPOCH + 100., QUICK.dictionary)

##########################################################################################
# Serialization, empty inputs, validation, and the creation diagnostics
##########################################################################################

def test_a_quickpath_survives_a_round_trip_through_pickle(core_kernels) -> None:
    """By default only the path and its limits are pickled, and the table is rebuilt."""

    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})
    time = Scalar(_EPOCH + 50.)
    expected = quick.event_at_time(time).pos

    restored = pickle.loads(pickle.dumps(quick))

    assert isinstance(restored, QuickPath)
    assert restored.event_at_time(time).pos == expected


def test_a_quickpath_can_pickle_its_tabulated_details(core_kernels) -> None:
    """With the flag set, the interpolation table is pickled along with the path."""

    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})
    time = Scalar(_EPOCH + 50.)
    expected = quick.event_at_time(time).pos

    quick.pickle_quickpath_details = True
    try:
        restored = pickle.loads(pickle.dumps(quick))
    finally:
        quick.pickle_quickpath_details = False

    assert restored.event_at_time(time).pos == expected


def test_an_empty_array_of_times_gives_an_empty_state(core_kernels) -> None:
    """With no times to evaluate, the position and velocity are masked."""

    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})

    (pos, vel) = quick._interpolate_pos_vel(Scalar(np.zeros((0,))))

    assert pos.shape == (0,)
    assert vel.shape == (0,)


def test_extending_a_quickpath_over_a_covered_interval_does_nothing(
        core_kernels) -> None:
    """An interval already inside the table leaves the table unchanged."""

    mars = SpicePath('MARS', 'SSB')
    quick = QuickPath.for_path(mars, _dense_times(0., 100.), quick={})
    before = quick._times.size

    quick.extend(quick._tmin, quick._tmax)

    assert quick._times.size == before


def test_for_path_rejects_an_unusable_quick_argument(core_kernels) -> None:
    """The `quick` argument is a dictionary of overrides, None, or False."""

    mars = SpicePath('MARS', 'SSB')

    with pytest.raises(ValueError, match='invalid `quick` input'):
        QuickPath.for_path(mars, _dense_times(0., 100.), quick=17)


def test_for_path_returns_a_shaped_path_unchanged(core_kernels) -> None:
    """A QuickPath tabulates one path, so a shaped path cannot be quickened."""

    shaped = FixedPath(Vector3([(1.e5, 0., 0.), (0., 1.e5, 0.)]), 'SSB', 'J2000',
                       path_id='TEST_QUICK_SHAPED')

    assert QuickPath.for_path(shaped, _dense_times(0., 100.), quick={}) is shaped


def test_for_path_returns_the_path_when_quickpaths_are_disabled(core_kernels) -> None:
    """The configured switch turns the optimization off entirely."""

    mars = SpicePath('MARS', 'SSB')

    assert QuickPath.for_path(mars, _dense_times(0., 100.),
                              quick={'use_quickpaths': False}) is mars


def test_for_path_returns_the_path_when_every_time_is_masked(core_kernels) -> None:
    """With no unmasked time there is no interval to tabulate."""

    mars = SpicePath('MARS', 'SSB')
    masked = Scalar(_EPOCH + np.arange(0., 100., 0.01), True)

    assert QuickPath.for_path(mars, masked, quick={}) is mars


def test_building_a_quickpath_is_reported_as_a_diagnostic(
        core_kernels, capsys: pytest.CaptureFixture[str]) -> None:
    """A new QuickPath and an extension of one are both logged."""

    mars = SpicePath('MARS', 'SSB')

    LOGGING.on()
    LOGGING.quickpath_creation = True
    try:
        QuickPath.for_path(mars, _dense_times(0., 100.), quick={})
        QuickPath.for_path(mars, _dense_times(100., 300.), quick={})
    finally:
        LOGGING.quickpath_creation = False
        LOGGING.off()

    printed = capsys.readouterr().out
    assert 'New QuickPath for' in printed
    assert 'Extending QuickPath for' in printed


def test_the_quickpath_cache_holds_only_as_many_as_it_is_allowed(core_kernels) -> None:
    """A new QuickPath displaces the oldest one once the cache is full."""

    mars = SpicePath('MARS', 'SSB')
    quick = {'quickpath_cache_size': 2}

    for offset in (0., 1.e6, 2.e6, 3.e6):
        QuickPath.for_path(mars, _dense_times(offset, offset + 100.), quick=quick)

    assert len(mars._quickpaths) == 2

##########################################################################################
