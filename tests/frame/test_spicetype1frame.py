##########################################################################################
# tests/frame/test_spicetype1frame.py
##########################################################################################

import pickle

import cspyce
import pytest

from spicedb                       import get_spice_filecache_prefix
from polymath                      import Scalar
from oops.frame                    import (Frame, LinkedFrame, Rotation, SpiceFrame,
                                           SpinFrame)
from oops.frame.spicetype1frame    import SpiceType1Frame

# Galileo's scan platform is defined by a Type 1 C kernel. Its frame definition and its
# spacecraft clock are all these tests need; none of them evaluates a transform, so no C
# kernel is required.
GALILEO_KERNELS = ['General/LSK/naif0012.tls',
                   'Galileo/SCLK/mk00062a.tsc',
                   'Galileo/FK/gll_v0.tf']

FRAME = 'GLL_SCAN_PLATFORM'
TICKS = 40                      # the tolerance the Galileo host uses
GALILEO_SCLK = -77              # the spacecraft ID, which is also its clock ID


@pytest.fixture
def galileo_kernels():
    """Furnish the kernels defining Galileo's scan platform, with the registries cleared.

    Yields:
        None
    """

    for path in get_spice_filecache_prefix().retrieve(GALILEO_KERNELS):
        cspyce.furnsh(path)
    Frame._reset_caches()

    yield

    Frame._reset_caches()


def test_get_reuses_a_cached_frame(galileo_kernels) -> None:
    """`get` returns the frame it built before, rather than building another."""

    frame = SpiceType1Frame.get(FRAME, TICKS)

    assert SpiceType1Frame.get(FRAME, TICKS) is frame

    # An unconstrained cache size matches the frame built with the default size
    assert SpiceType1Frame.get(FRAME, TICKS, cache_size=None) is frame


def test_get_matches_a_tolerance_given_as_a_string(galileo_kernels) -> None:
    """A tolerance written in clock ticks matches the same tolerance given as a number.

    The constructor converts a string to ticks, so a key built from the unconverted string
    could never match what it stored.
    """

    frame = SpiceType1Frame.get(FRAME, '0:40')

    assert SpiceType1Frame.get(FRAME, '0:40') is frame
    assert SpiceType1Frame.get(FRAME, frame._tick_tolerance) is frame


def test_the_lookup_is_separate_from_the_one_in_spiceframe(galileo_kernels) -> None:
    """Type 1 frames are cached apart from ordinary SpiceFrames.

    The two classes key their caches differently, so sharing one dictionary between them
    would rely on the two kinds of key never colliding.
    """

    frame = SpiceType1Frame.get(FRAME, TICKS)

    assert frame in SpiceType1Frame._FRAME_LOOKUP.values()
    assert frame not in SpiceFrame._FRAME_LOOKUP.values()


def test_resetting_the_caches_empties_the_frame_lookup(galileo_kernels) -> None:
    """The lookup holds frames the registry no longer knows, so a reset clears it."""

    SpiceType1Frame.get(FRAME, TICKS)
    assert SpiceType1Frame._FRAME_LOOKUP

    Frame._reset_caches()

    assert not SpiceType1Frame._FRAME_LOOKUP


def test_a_frame_on_another_reference_leaves_the_registered_frame_alone(galileo_kernels):
    """Building the frame relative to another frame does not displace the registered one.

    The registered ID must keep referring to the frame defined relative to J2000; the
    version on another reference is cached without claiming that ID.
    """

    primary = SpiceType1Frame(FRAME, TICKS)
    assert Frame._FRAME_REGISTRY[FRAME] is primary
    assert primary._reference == Frame.J2000

    secondary = SpiceType1Frame(FRAME, TICKS, SpiceFrame.get('B1950'))

    assert Frame._FRAME_REGISTRY[FRAME] is primary
    assert secondary._reference == SpiceFrame.get('B1950').wayframe
    assert secondary._wayframe is primary._wayframe


def test_get_rejects_a_reference_that_is_not_a_spice_frame(galileo_kernels) -> None:
    """A reference the constructor could not use is refused by `get` as well.

    The error names the problem rather than surfacing as a missing attribute.
    """

    spinning = SpinFrame(0., 1.e-6, 0., 2, Frame.J2000, frame_id='TEST_SPIN')

    with pytest.raises(ValueError, match='must be a SpiceFrame or J2000'):
        SpiceType1Frame.get(FRAME, TICKS, spinning)


# The C kernel covering Galileo's scan platform in November 1996. Evaluating a transform
# needs it; the tests above do not.
CK_KERNEL = 'Galileo/CK/ckc03b_plt.bc'


@pytest.fixture
def galileo_pointing(galileo_kernels):
    """Furnish a C kernel and return times at which the scan platform is pointed.

    Yields:
        list[float]: Four times in seconds TDB, each at an actual pointing instance, so
        that a query with the frame's own tick tolerance succeeds.
    """

    cspyce.furnsh(get_spice_filecache_prefix().retrieve(CK_KERNEL))

    spacecraft = -77
    instrument = -77001
    times = set()
    for offset in (0., 500., 1000., 2000.):
        ticks = cspyce.sce2c(spacecraft, -99777477.0 + offset)
        (_, true_tick) = cspyce.ckgp(instrument, ticks, 1.e9, 'J2000')
        times.add(cspyce.sct2e(spacecraft, true_tick))

    yield sorted(times)


# A time far outside the coverage of any Galileo C kernel
UNCOVERED = 0.


def test_transforms_are_returned_for_every_covered_time(galileo_pointing) -> None:
    """An array of covered times yields a transform of the same shape."""

    frame = SpiceType1Frame(FRAME, TICKS)

    (valid, xform) = frame.transform_at_time_if_possible(Scalar(galileo_pointing))

    assert valid.shape == (len(galileo_pointing),)
    assert xform.matrix.shape == (len(galileo_pointing),)


def test_uncovered_times_are_omitted_rather_than_raising(galileo_pointing) -> None:
    """A time outside the C kernel coverage drops out of the result.

    The transform is defined at the times that remain, and those are what is returned.
    """

    frame = SpiceType1Frame(FRAME, TICKS)
    times = [galileo_pointing[0], UNCOVERED, galileo_pointing[1]]

    (valid, xform) = frame.transform_at_time_if_possible(Scalar(times))

    assert valid.shape == (2,)
    assert xform.matrix.shape == (2,)
    assert valid == Scalar([galileo_pointing[0], galileo_pointing[1]])


def test_a_partial_result_is_not_reused_for_a_later_call(galileo_pointing) -> None:
    """The transform for a subset of times is not cached against the full input shape."""

    frame = SpiceType1Frame(FRAME, TICKS)
    times = Scalar([galileo_pointing[0], UNCOVERED, galileo_pointing[1]])

    (first, _) = frame.transform_at_time_if_possible(times)
    (second, xform) = frame.transform_at_time_if_possible(times)

    assert second == first
    assert xform.matrix.shape == (2,)


def test_every_time_uncovered_raises(galileo_pointing) -> None:
    """With nothing to return, the error from the Toolkit stands."""

    frame = SpiceType1Frame(FRAME, TICKS)

    with pytest.raises(OSError, match='CKINSUFFDATA'):
        frame.transform_at_time_if_possible(Scalar([UNCOVERED, UNCOVERED + 1000.]))

def test_a_tolerance_string_is_converted_by_the_constructor(galileo_kernels) -> None:
    """The constructor accepts ticks written as a clock string and converts them."""

    frame = SpiceType1Frame(FRAME, '0:40')

    assert frame._tick_tolerance == cspyce.sctiks(GALILEO_SCLK, '0:40')


def test_get_returns_a_matching_frame_it_was_handed(galileo_kernels) -> None:
    """`get` given a SpiceType1Frame returns it unchanged when the request matches."""

    frame = SpiceType1Frame(FRAME, TICKS)

    assert SpiceType1Frame.get(frame, TICKS) is frame
    assert SpiceType1Frame.get(frame, TICKS, cache_size=None) is frame


def test_get_rebuilds_from_a_frame_whose_tolerance_differs(galileo_kernels) -> None:
    """A request that differs from the frame handed in is satisfied by its name."""

    frame = SpiceType1Frame(FRAME, TICKS)

    other = SpiceType1Frame.get(frame, 2 * TICKS)

    assert other is not frame
    assert other._tick_tolerance == 2 * TICKS
    assert other._spice_frame_name == frame._spice_frame_name


def test_refreshing_a_frame_empties_its_quickframe_cache(galileo_kernels) -> None:
    """`_refresh` drops the QuickFrames built against the transforms it is discarding."""

    frame = SpiceType1Frame(FRAME, TICKS)
    frame._quickframes = {'key': 'quickframe'}

    frame._refresh()

    assert frame._quickframes == {}


def test_a_shortcut_connects_through_a_non_spice_reference(galileo_kernels) -> None:
    """A reference that is not a SpiceFrame is reached through its nearest SPICE ancestor.
    """

    frame = SpiceType1Frame(FRAME, TICKS)
    tilted = Rotation(0.1, 2, Frame.J2000, frame_id='TEST_ROTATION')

    shortcut = frame._get_shortcut(tilted.wayframe)

    assert isinstance(shortcut, LinkedFrame)
    assert shortcut.reference == tilted.wayframe


def test_a_shortcut_to_a_spice_reference_needs_no_link(galileo_kernels) -> None:
    """When the reference is already a SpiceFrame, no linking frame is needed."""

    frame = SpiceType1Frame(FRAME, TICKS)
    b1950 = SpiceFrame.get('B1950')

    shortcut = frame._get_shortcut(b1950.wayframe)

    assert isinstance(shortcut, SpiceType1Frame)
    assert shortcut.reference == b1950.wayframe


def test_a_scalar_transform_is_cached(galileo_pointing) -> None:
    """A shapeless time is evaluated once and returned from the cache thereafter."""

    frame = SpiceType1Frame(FRAME, TICKS)
    time = Scalar(galileo_pointing[0])

    xform = frame.transform_at_time(time)

    assert xform.matrix.shape == ()
    assert frame.transform_at_time(time) is xform


def test_an_array_of_spread_times_is_evaluated_elementwise(galileo_pointing) -> None:
    """Times too far apart to share one transform each get their own rotation matrix."""

    frame = SpiceType1Frame(FRAME, TICKS)

    xform = frame.transform_at_time(Scalar(galileo_pointing))

    assert xform.matrix.shape == (len(galileo_pointing),)
    assert xform.matrix[0] != xform.matrix[-1]


def test_nearly_equal_times_share_one_transform(galileo_pointing) -> None:
    """When every time falls inside the tolerance, one SPICE query serves them all."""

    frame = SpiceType1Frame(FRAME, TICKS)
    time = Scalar(galileo_pointing[0]) + Scalar([0., 1.e-6, 2.e-6])

    xform = frame.transform_at_time(time)

    assert xform.matrix.shape == ()
    assert frame._cached_shape == (3,)


def test_a_repeated_shaped_query_reuses_the_cached_transform(galileo_pointing) -> None:
    """A second query at the same times and shape returns the transform already built."""

    frame = SpiceType1Frame(FRAME, TICKS)
    time = Scalar(galileo_pointing[0]) + Scalar([0., 1.e-6, 2.e-6])

    xform = frame.transform_at_time(time)

    assert frame.transform_at_time(time) is xform


def test_a_scalar_transform_is_cached_by_the_tolerant_variant(galileo_pointing) -> None:
    """`transform_at_time_if_possible` caches a shapeless time the same way."""

    frame = SpiceType1Frame(FRAME, TICKS)
    time = Scalar(galileo_pointing[0])

    (valid, xform) = frame.transform_at_time_if_possible(time)

    assert valid is time
    assert frame.transform_at_time_if_possible(time)[1] is xform


def test_the_tolerant_variant_reuses_a_shaped_transform(galileo_pointing) -> None:
    """Nearly equal times yield one transform, which a repeated query returns again."""

    frame = SpiceType1Frame(FRAME, TICKS)
    time = Scalar(galileo_pointing[0]) + Scalar([0., 1.e-6, 2.e-6])

    (_, xform) = frame.transform_at_time_if_possible(time)

    assert xform.matrix.shape == ()
    assert frame.transform_at_time_if_possible(time)[1] is xform


def test_a_masked_time_is_skipped_by_the_tolerant_variant(galileo_pointing) -> None:
    """A masked time is passed over rather than reported as an uncovered time."""

    frame = SpiceType1Frame(FRAME, TICKS)
    time = Scalar([galileo_pointing[0], UNCOVERED, galileo_pointing[1]],
                  [False, True, False])

    (valid, xform) = frame.transform_at_time_if_possible(time)

    assert valid.shape == (2,)
    assert xform.matrix.shape == (2,)


def test_a_two_dimensional_uncovered_time_raises(galileo_pointing) -> None:
    """The tolerant variant drops times only from a 1-D array; otherwise the error stands.
    """

    frame = SpiceType1Frame(FRAME, TICKS)
    times = Scalar([[galileo_pointing[0], UNCOVERED],
                    [galileo_pointing[1], galileo_pointing[2]]])

    with pytest.raises(OSError, match='CKINSUFFDATA'):
        frame.transform_at_time_if_possible(times)


def test_a_frame_survives_a_round_trip_through_pickle(galileo_pointing) -> None:
    """Unpickling rebuilds the frame from its SPICE name, tolerance and cache size."""

    frame = SpiceType1Frame(FRAME, TICKS, cache_size=17)
    expected = frame.transform_at_time(Scalar(galileo_pointing[0])).matrix

    revived = pickle.loads(pickle.dumps(frame))

    assert revived._spice_frame_name == frame._spice_frame_name
    assert revived._tick_tolerance == frame._tick_tolerance
    assert revived._cache_size == 17
    assert revived.transform_at_time(Scalar(galileo_pointing[0])).matrix == expected

def test_a_shortcut_through_a_rotating_reference_needs_a_common_origin(galileo_kernels):
    """Two rotating frames centered on different bodies cannot be linked.

    The scan platform turns about the spacecraft, so it cannot be linked through a frame
    that turns about the solar system barycenter.
    """

    frame = SpiceType1Frame(FRAME, TICKS)
    spinning = SpinFrame(0., 1.e-6, 0., 2, Frame.J2000, frame_id='TEST_SPIN')

    with pytest.raises(ValueError, match='LinkedFrame origin mismatch'):
        frame._get_shortcut(spinning.wayframe)

##########################################################################################
