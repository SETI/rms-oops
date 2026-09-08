##########################################################################################
# tests/frame/test_frame_show.py: Frame.show() across the Frame subclasses
##########################################################################################

import pytest

from polymath      import Matrix3, Scalar, Vector3
from oops.frame    import (Cmatrix, Frame, FrameShift, InclinedFrame, Navigation,
                           PoleFrame, PosTargFrame, RingFrame, Rotation, SpiceFrame,
                           SpinFrame, TrackerFrame, TwoVectorFrame)
from oops.frame.quickframe import QuickFrame
from oops.path.spicepath   import SpicePath

# `show(2)` is the shallowest level that expands a Frame's own definition; levels 0 and 1
# report the ID and the one-line summary instead, which every Frame shares. A Frame named
# one level down is summarized rather than expanded, so it appears without quotes.
EXPANDED = 2


@pytest.fixture(autouse=True)
def _kernels(core_kernels) -> None:
    """Furnish the core kernels for every test in this module."""


def test_j2000_describes_itself_by_name() -> None:
    """The root frame has no definition to expand, so it names itself at every level."""

    assert Frame.J2000.show(EXPANDED) == '"J2000"'


def test_a_spiceframe_on_j2000_is_named_alone() -> None:
    """A SPICE frame defined relative to J2000 needs only its SPICE name."""

    frame = SpiceFrame('IAU_SATURN')

    assert frame.show(EXPANDED) == 'SpiceFrame("IAU_SATURN")'


def test_a_spiceframe_on_another_reference_names_both() -> None:
    """A SPICE frame on some other reference reports that reference too."""

    frame = SpiceFrame('IAU_SATURN', SpiceFrame('B1950'))

    description = frame.show(EXPANDED)

    assert description.startswith('SpiceFrame("IAU_SATURN",')
    assert 'B1950' in description


def test_a_cmatrix_reports_its_rotation_matrix() -> None:
    """A Cmatrix is defined by its matrix and its reference."""

    frame = Cmatrix(Matrix3.IDENTITY, frame_id='TEST_SHOW_CMATRIX')

    description = frame.show(EXPANDED)

    assert description.startswith('Cmatrix([[1. 0. 0.]')
    assert description.endswith('"J2000")')


def test_a_cmatrix_on_another_reference_expands_it() -> None:
    """A reference other than J2000 is summarized in place of the literal "J2000"."""

    reference = SpiceFrame('IAU_SATURN')
    frame = Cmatrix(Matrix3.IDENTITY, reference, frame_id='TEST_SHOW_CMATRIX_2')

    assert frame.show(EXPANDED).endswith('SpiceFrame(IAU_SATURN))')


def test_a_twovectorframe_reports_both_vectors_and_axes() -> None:
    """Two vectors and the axes they define are what a TwoVectorFrame is built from."""

    frame = TwoVectorFrame(Frame.J2000, Vector3.XAXIS, 0, Vector3.YAXIS, 1,
                           frame_id='TEST_SHOW_TWOVECTOR')

    description = frame.show(EXPANDED)

    assert description.startswith('TwoVectorFrame(J2000,')
    assert '[1. 0. 0.], 0,' in description
    assert description.endswith('[0. 1. 0.], 1)')


def test_a_spinframe_reports_its_rate_and_epoch() -> None:
    """A SpinFrame is defined by an offset, a rate, an epoch, an axis and a reference."""

    frame = SpinFrame(0.25, 1.e-4, 1000., 2, Frame.J2000, frame_id='TEST_SHOW_SPIN')

    description = frame.show(EXPANDED)

    assert description.startswith('SpinFrame(offset = 0.25,')
    assert 'rate = 0.0001,' in description
    assert 'epoch = 1000.0,' in description
    assert 'axis = 2,' in description
    assert 'reference = J2000)' in description


def test_an_inclinedframe_reports_its_elements() -> None:
    """An InclinedFrame is defined by an inclination, a node, a rate and an epoch."""

    frame = InclinedFrame(0.1, 0.2, 1.e-5, 0., frame_id='TEST_SHOW_INCLINED')

    description = frame.show(EXPANDED)

    assert description.startswith('InclinedFrame(inc = 0.1,')
    assert 'node = 0.2,' in description
    assert 'rate = 1e-05,' in description
    assert 'reference = "J2000",' in description
    assert 'despin = True)' in description


def test_an_inclinedframe_expands_a_reference_that_is_not_j2000() -> None:
    """A reference other than J2000 is summarized in place of the literal "J2000"."""

    frame = InclinedFrame(0.1, 0.2, 1.e-5, 0., reference=SpiceFrame('IAU_SATURN'),
                          frame_id='TEST_SHOW_INCLINED_2')

    assert 'reference = SpiceFrame(IAU_SATURN),' in frame.show(EXPANDED)


def test_a_postargframe_reports_its_two_offsets() -> None:
    """A PosTargFrame is defined by two angular offsets and a reference."""

    frame = PosTargFrame(1.e-5, 2.e-5, Frame.J2000, frame_id='TEST_SHOW_POSTARG')

    assert frame.show(EXPANDED) == 'PosTargFrame(1e-05, 2e-05,\n             J2000)'


def test_a_rotation_reports_its_angle_and_axis() -> None:
    """A Rotation is defined by an angle about an axis of its reference."""

    frame = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_SHOW_ROTATION')

    assert frame.show(EXPANDED) == 'Rotation(0.5, 2\n         J2000)'


def test_a_linked_rotation_expands_the_frame_it_follows() -> None:
    """A Rotation that tracks another Rotation reports that Rotation, not an angle."""

    tracked = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_SHOW_TRACKED')
    frame = Rotation(tracked, 2, Frame.J2000, frame_id='TEST_SHOW_LINKED')

    assert frame.show(EXPANDED) == ('Rotation(Rotation(TEST_SHOW_TRACKED)\n'
                                    '         2\n'
                                    '         J2000)')


def test_a_navigation_reports_its_angles() -> None:
    """A Navigation frame is defined by its pointing offsets and its reference."""

    frame = Navigation((1.e-6, 2.e-6, 3.e-6), Frame.J2000, frame_id='TEST_SHOW_NAV')

    assert frame.show(EXPANDED) == ('Navigation((1e-06, 2e-06, 3e-06),\n'
                                    '           J2000)')


def test_a_linked_navigation_expands_the_frame_it_follows() -> None:
    """A Navigation that tracks another reports that frame in place of its angles."""

    tracked = Navigation((1.e-6, 2.e-6, 3.e-6), Frame.J2000, frame_id='TEST_SHOW_NAV_2')
    frame = Navigation(tracked, Frame.J2000, frame_id='TEST_SHOW_NAV_3')

    assert frame.show(EXPANDED).startswith('Navigation(Navigation(TEST_SHOW_NAV_2),')


def test_a_ringframe_reports_the_frame_it_is_derived_from() -> None:
    """A RingFrame names the planet frame whose equator it adopts."""

    frame = RingFrame(SpiceFrame('IAU_SATURN'), frame_id='TEST_SHOW_RING')

    assert frame.show(EXPANDED) == 'RingFrame(frame = SpiceFrame(IAU_SATURN))'


def test_a_ringframe_reports_its_optional_settings() -> None:
    """An epoch, a retrograde sense and an Aries longitude origin each appear."""

    frame = RingFrame(SpiceFrame('IAU_SATURN'), epoch=0., retrograde=True, aries=True,
                      frame_id='TEST_SHOW_RING_2')

    description = frame.show(EXPANDED)

    assert 'epoch = Scalar(0.0),' in description
    assert 'retrograde = True,' in description
    assert 'aries = True)' in description


def test_a_poleframe_reports_its_invariable_pole() -> None:
    """A PoleFrame names the planet frame and the pole it precesses about."""

    frame = PoleFrame(SpiceFrame('IAU_NEPTUNE'), Vector3.ZAXIS,
                      frame_id='TEST_SHOW_POLE')

    description = frame.show(EXPANDED)

    assert description.startswith('PoleFrame(frame = SpiceFrame(IAU_NEPTUNE),')
    assert description.endswith('pole = [0. 0. 1.])')


def test_a_poleframe_reports_its_optional_settings() -> None:
    """A retrograde sense and an Aries longitude origin each appear when they are set."""

    frame = PoleFrame(SpiceFrame('IAU_NEPTUNE'), Vector3.ZAXIS, retrograde=True,
                      aries=True, frame_id='TEST_SHOW_POLE_2')

    description = frame.show(EXPANDED)

    assert 'retrograde = True,' in description
    assert 'aries = True)' in description


def test_a_trackerframe_reports_the_target_and_the_observer() -> None:
    """A TrackerFrame names the frame it freezes, the target, the observer and the epoch.
    """

    frame = TrackerFrame(SpiceFrame('IAU_MARS'), SpicePath('MARS'), SpicePath('EARTH'),
                         0., frame_id='TEST_SHOW_TRACKER')

    description = frame.show(EXPANDED)

    assert description.startswith('TrackerFrame(frame = SpiceFrame(IAU_MARS),')
    assert 'target = SpicePath(MARS),' in description
    assert 'observer = SpicePath(EARTH),' in description
    assert description.endswith('epoch = Scalar(0.0))')


def test_a_frameshift_by_a_time_offset_reports_the_offset() -> None:
    """A FrameShift built on a fixed offset reports that offset and the frame."""

    frame = FrameShift(60., SpiceFrame('IAU_MARS'), frame_id='TEST_SHOW_SHIFT')

    assert frame.show(EXPANDED).startswith('FrameShift(60.0, SpiceFrame(IAU_MARS)')


def test_a_frameshift_reports_a_frame_it_takes_its_offset_from() -> None:
    """A FrameShift linked to another FrameShift expands that FrameShift."""

    linked = FrameShift(60., SpiceFrame('IAU_MARS'), frame_id='TEST_SHOW_SHIFT_2')
    frame = FrameShift(linked, SpiceFrame('IAU_SATURN'), frame_id='TEST_SHOW_SHIFT_3')

    assert frame.show(EXPANDED) == ('FrameShift(FrameShift(TEST_SHOW_SHIFT_2),\n'
                                    '           SpiceFrame(IAU_SATURN))')


def test_a_frameshift_reports_an_offset_that_cannot_be_expanded() -> None:
    """An offset that is a Scalar rather than a Frame is written out directly."""

    frame = FrameShift(Scalar(60.), SpiceFrame('IAU_MARS'), frame_id='TEST_SHOW_SHIFT_4')

    assert frame.show(EXPANDED).startswith('FrameShift(Scalar(60.0), '
                                           'SpiceFrame(IAU_MARS)')


def test_a_quickframe_reports_the_frame_and_the_time_span_it_samples() -> None:
    """A QuickFrame is an interpolation of one frame over one span of time.

    The span it reports is the padded one it actually samples, not the one requested.
    """

    frame = QuickFrame(SpiceFrame('IAU_MARS'), 0., 100., quick={'frame_time_step': 1.})

    description = frame.show(EXPANDED)

    assert description.startswith('QuickFrame(SpiceFrame(IAU_MARS),')
    assert description.endswith(f'{frame._tmin}, {frame._tmax})')


def test_the_expansion_of_a_deeper_frame_indents_each_level() -> None:
    """Every line after the first is indented past the name of the frame that owns it."""

    inner = Rotation(0.5, 2, SpiceFrame('IAU_MARS'), frame_id='TEST_SHOW_DEPTH_INNER')
    frame = Rotation(0.25, 1, inner, frame_id='TEST_SHOW_DEPTH')

    assert frame.show(3) == ('Rotation(0.25, 1\n'
                             '         Rotation(0.5, 2\n'
                             '                  SpiceFrame(IAU_MARS)))')

##########################################################################################
