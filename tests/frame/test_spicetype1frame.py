def test_get_rejects_a_reference_that_is_not_a_spice_frame(galileo_kernels) -> None:
    """A reference the constructor could not use is refused by `get` as well.

    The error names the problem rather than surfacing as a missing attribute.
    """

    with pytest.raises(ValueError, match='must be a SpiceFrame or J2000'):
        SpiceType1Frame.get(FRAME, TICKS, Frame.as_wayframe('B1950_ROTATED'))

##########################################################################################
# tests/frame/test_spicetype1frame.py
def test_get_rejects_a_reference_that_is_not_a_spice_frame(galileo_kernels) -> None:
    """A reference the constructor could not use is refused by `get` as well.

    The error names the problem rather than surfacing as a missing attribute.
    """

    with pytest.raises(ValueError, match='must be a SpiceFrame or J2000'):
        SpiceType1Frame.get(FRAME, TICKS, Frame.as_wayframe('B1950_ROTATED'))

##########################################################################################

import cspyce
import pytest

from spicedb                       import get_spice_filecache_prefix
from oops.frame                    import Frame, SpiceFrame, SpinFrame
from oops.frame.spicetype1frame    import SpiceType1Frame

# Galileo's scan platform is defined by a Type 1 C kernel. Its frame definition and its
# spacecraft clock are all these tests need; none of them evaluates a transform, so no C
# kernel is required.
GALILEO_KERNELS = ['General/LSK/naif0012.tls',
                   'Galileo/SCLK/mk00062a.tsc',
                   'Galileo/FK/gll_v0.tf']

FRAME = 'GLL_SCAN_PLATFORM'
TICKS = 40                      # the tolerance the Galileo host uses


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

##########################################################################################
