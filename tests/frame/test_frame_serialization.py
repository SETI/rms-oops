##########################################################################################
# tests/frame/test_frame_serialization.py: pickling and generated IDs across the Frames
##########################################################################################

import pickle

import pytest

from polymath   import Matrix3, Scalar, Vector3
from oops       import mutable
from oops.frame import (Cmatrix, Frame, FrameShift, InclinedFrame, Navigation, PoleFrame,
                        PosTargFrame, RingFrame, Rotation, SpiceFrame, SpinFrame,
                        TrackerFrame, TwoVectorFrame)
from oops.frame.frame_   import Frame as FrameClass
from oops.path.spicepath import SpicePath

# A time at which every one of these frames is defined
TIME = Scalar([0., 1000.])


@pytest.fixture(autouse=True)
def _kernels(core_kernels) -> None:
    """Furnish the core kernels for every test in this module."""


# The names of the frames _frames() builds. The dictionary itself cannot be built at
# collection time, because the SPICE kernels are not furnished until a test runs.
FRAME_NAMES = ['Cmatrix', 'FrameShift', 'InclinedFrame', 'Navigation', 'PoleFrame',
               'PosTargFrame', 'RingFrame', 'Rotation', 'SpiceFrame', 'SpinFrame',
               'TrackerFrame', 'TwoVectorFrame']


def _frames() -> dict[str, FrameClass]:
    """One instance of every pickleable Frame subclass, keyed by name.

    Returns:
        dict[str, Frame]: The frames to test.
    """

    mars = SpiceFrame('IAU_MARS')

    return {
        'Cmatrix':        Cmatrix(Matrix3.IDENTITY, frame_id='TEST_PICKLE_CMATRIX'),
        'FrameShift':     FrameShift(60., mars, frame_id='TEST_PICKLE_SHIFT'),
        'InclinedFrame':  InclinedFrame(0.1, 0.2, 1.e-5, 0.,
                                        frame_id='TEST_PICKLE_INCLINED'),
        'Navigation':     Navigation((1.e-6, 2.e-6, 3.e-6), Frame.J2000,
                                     frame_id='TEST_PICKLE_NAV'),
        'PoleFrame':      PoleFrame(SpiceFrame('IAU_NEPTUNE'), Vector3.ZAXIS,
                                    frame_id='TEST_PICKLE_POLE'),
        'PosTargFrame':   PosTargFrame(1.e-5, 2.e-5, Frame.J2000,
                                       frame_id='TEST_PICKLE_POSTARG'),
        'RingFrame':      RingFrame(SpiceFrame('IAU_SATURN'),
                                    frame_id='TEST_PICKLE_RING'),
        'Rotation':       Rotation(0.5, 2, Frame.J2000, frame_id='TEST_PICKLE_ROTATION'),
        'SpiceFrame':     mars,
        'SpinFrame':      SpinFrame(0.25, 1.e-4, 1000., 2, Frame.J2000,
                                    frame_id='TEST_PICKLE_SPIN'),
        'TrackerFrame':   TrackerFrame(mars, SpicePath('MARS'), SpicePath('EARTH'), 0.,
                                       frame_id='TEST_PICKLE_TRACKER'),
        'TwoVectorFrame': TwoVectorFrame(Frame.J2000, Vector3.XAXIS, 0, Vector3.YAXIS, 1,
                                         frame_id='TEST_PICKLE_TWOVECTOR'),
    }


@pytest.mark.parametrize('name', FRAME_NAMES)
def test_a_frame_survives_a_round_trip_through_pickle(name: str) -> None:
    """Unpickling rebuilds the frame and reproduces the transform it defines."""

    frame = _frames()[name]
    expected = frame.transform_at_time(TIME).matrix

    restored = pickle.loads(pickle.dumps(frame))

    assert type(restored) is type(frame)
    assert restored.transform_at_time(TIME).matrix == expected


@pytest.mark.parametrize('name', FRAME_NAMES)
def test_an_unpickled_frame_is_frozen(name: str) -> None:
    """Unpickling restores the values as they stood, so the result is not refittable."""

    frame = _frames()[name]

    assert mutable.is_frozen(pickle.loads(pickle.dumps(frame)))


##########################################################################################
# Generated frame IDs
##########################################################################################

@pytest.mark.parametrize('class_, args, suffix',
                         [(PosTargFrame, (1.e-5, 2.e-5), '_POSTARG'),
                          (SpinFrame, (0.25, 1.e-4, 1000., 2), '_SPIN')])
def test_a_generated_frame_id_extends_the_id_of_the_reference(class_, args,
                                                              suffix: str) -> None:
    """A frame_id of "+" appends a suffix to the ID of the reference frame."""

    reference = SpiceFrame('IAU_MARS')

    frame = class_(*args, reference, frame_id='+')

    assert frame.frame_id == 'IAU_MARS' + suffix


def test_a_tracker_frame_id_extends_the_id_of_the_frame_it_freezes() -> None:
    """A TrackerFrame takes its generated ID from the frame it holds fixed."""

    frame = TrackerFrame(SpiceFrame('IAU_MARS'), SpicePath('MARS'), SpicePath('EARTH'),
                         0., frame_id='+')

    assert frame.frame_id == 'IAU_MARS_TRACKER'


##########################################################################################
# Frames linked to other frames, and the frozen forms they yield
##########################################################################################

def test_a_frame_shift_linked_to_a_frozen_shift_is_frozen() -> None:
    """Linking to a frozen offset gives a frozen frame with that offset."""

    linked = FrameShift(60., SpiceFrame('IAU_MARS'), frame_id='TEST_FROZEN_SOURCE')
    linked.freeze()

    frame = FrameShift(linked, SpiceFrame('IAU_SATURN'), frame_id='TEST_FROZEN_SHIFT')

    assert frame.dt == 60.
    assert frame.link is None
    assert mutable.is_frozen(frame)


def test_a_frame_shift_can_be_linked_by_the_id_of_another() -> None:
    """A string is looked up in the frame registry."""

    FrameShift(60., SpiceFrame('IAU_MARS'), frame_id='TEST_LINK_BY_ID')

    frame = FrameShift('TEST_LINK_BY_ID', SpiceFrame('IAU_SATURN'),
                       frame_id='TEST_LINKED_BY_ID')

    assert frame.dt == 60.


def test_a_linked_frame_shift_names_its_source() -> None:
    """The source of a chain of shifts is the one that holds the offset."""

    original = FrameShift(60., SpiceFrame('IAU_MARS'), frame_id='TEST_SHIFT_SOURCE')
    linked = FrameShift(original, SpiceFrame('IAU_SATURN'), frame_id='TEST_SHIFT_LINKED')

    assert linked._source() is original
    assert original._source() is original


def test_setting_the_parameters_of_a_linked_frame_shift_moves_the_original() -> None:
    """Fitting a linked shift redefines the offset of the frame it follows."""

    original = FrameShift(60., SpiceFrame('IAU_MARS'), frame_id='TEST_SHIFT_FIT_SOURCE')
    linked = FrameShift(original, SpiceFrame('IAU_SATURN'),
                        frame_id='TEST_SHIFT_FIT_LINKED')

    linked.set_params((120.,))

    assert original.dt == 120.
    assert linked.dt == 120.


def test_a_rotation_linked_to_a_frozen_rotation_is_frozen() -> None:
    """Linking to a frozen angle gives a frozen frame with that angle."""

    tracked = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_FROZEN_ROTATION_SOURCE')
    tracked.freeze()

    frame = Rotation(tracked, 2, Frame.J2000, frame_id='TEST_FROZEN_ROTATION')

    assert frame.params == (0.5,)
    assert mutable.is_frozen(frame)


def test_a_rotation_can_be_linked_by_the_id_of_another() -> None:
    """A string is looked up in the frame registry."""

    Rotation(0.5, 2, Frame.J2000, frame_id='TEST_ROTATION_BY_ID')

    frame = Rotation('TEST_ROTATION_BY_ID', 2, Frame.J2000,
                     frame_id='TEST_ROTATION_LINKED_BY_ID')

    assert frame.params == (0.5,)


def test_a_linked_rotation_names_its_source() -> None:
    """The source of a chain of rotations is the one that holds the angle."""

    original = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_ROTATION_SOURCE')
    linked = Rotation(original, 2, Frame.J2000, frame_id='TEST_ROTATION_LINKED')

    assert linked._source() is original
    assert original._source() is original


def test_setting_the_parameters_of_a_linked_rotation_moves_the_original() -> None:
    """Fitting a linked rotation redefines the angle of the frame it follows."""

    original = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_ROTATION_FIT_SOURCE')
    linked = Rotation(original, 2, Frame.J2000, frame_id='TEST_ROTATION_FIT_LINKED')

    linked.set_params((1.25,))

    assert original.params == (1.25,)
    assert linked.params == (1.25,)


def test_freezing_a_linked_rotation_adopts_the_angle_and_drops_the_link() -> None:
    """A frozen rotation keeps the angle it had and stops following its source."""

    original = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_ROTATION_FREEZE_SOURCE')
    linked = Rotation(original, 2, Frame.J2000, frame_id='TEST_ROTATION_FREEZE_LINKED')

    linked.freeze()

    assert linked._link is None
    assert linked.params == (0.5,)


def test_a_shaped_rotation_is_fitted_element_by_element() -> None:
    """An array of angles is reported as a flat tuple and restored to its own shape."""

    frame = Rotation(Scalar([0.5]), 2, Frame.J2000, frame_id='TEST_ROTATION_SHAPED')

    assert frame.params == (0.5,)

    frame.set_params((0.25,))

    assert frame.params == (0.25,)
    assert frame.transform_at_time(Scalar(0.)).matrix.shape == (1,)


##########################################################################################
# The node of a PoleFrame, which follows the planet's pole
##########################################################################################

def test_the_node_of_a_pole_frame_follows_the_planet() -> None:
    """The ascending node is 90 degrees ahead of the planet's pole."""

    frame = PoleFrame(SpiceFrame('IAU_NEPTUNE'), Vector3.ZAXIS,
                      frame_id='TEST_POLE_NODE')

    node = frame.node_at_time(TIME)

    assert node.shape == (2,)
    assert node.vals[0] != node.vals[1]     # Neptune's pole precesses


def test_the_node_of_a_retrograde_pole_frame_is_half_a_turn_away() -> None:
    """Reversing the pole moves the ascending node to the opposite side."""

    prograde = PoleFrame(SpiceFrame('IAU_NEPTUNE'), Vector3.ZAXIS,
                         frame_id='TEST_POLE_NODE_PROGRADE')
    retrograde = PoleFrame(SpiceFrame('IAU_NEPTUNE'), Vector3.ZAXIS, retrograde=True,
                           frame_id='TEST_POLE_NODE_RETROGRADE')

    difference = (prograde.node_at_time(TIME) - retrograde.node_at_time(TIME)) % 6.28318

    assert difference.vals[0] == pytest.approx(3.14159, abs=1.e-4)


##########################################################################################
# The transform cache of a TrackerFrame
##########################################################################################

def test_a_tracker_frame_caches_a_shapeless_transform() -> None:
    """A single time is evaluated once and returned from the cache thereafter."""

    frame = TrackerFrame(SpiceFrame('IAU_MARS'), SpicePath('MARS'), SpicePath('EARTH'),
                         0., frame_id='TEST_TRACKER_CACHE')

    xform = frame.transform_at_time(Scalar(0.))

    assert frame.transform_at_time(Scalar(0.)) is xform

def test_a_navigation_linked_to_a_frozen_navigation_is_frozen() -> None:
    """Linking to frozen angles gives a frozen frame with those angles."""

    tracked = Navigation((1.e-6, 2.e-6, 3.e-6), Frame.J2000,
                         frame_id='TEST_FROZEN_NAV_SOURCE')
    tracked.freeze()

    frame = Navigation(tracked, Frame.J2000, frame_id='TEST_FROZEN_NAV')

    assert frame.params == tracked.params
    assert mutable.is_frozen(frame)


def test_a_navigation_can_be_linked_by_the_id_of_another() -> None:
    """A string is looked up in the frame registry."""

    original = Navigation((1.e-6, 2.e-6, 3.e-6), Frame.J2000,
                          frame_id='TEST_NAV_BY_ID')

    frame = Navigation('TEST_NAV_BY_ID', Frame.J2000, frame_id='TEST_NAV_LINKED_BY_ID')

    assert frame.params == original.params


def test_a_linked_navigation_names_its_source() -> None:
    """The source of a chain of navigations is the one that holds the angles."""

    original = Navigation((1.e-6, 2.e-6, 3.e-6), Frame.J2000,
                          frame_id='TEST_NAV_SOURCE')
    linked = Navigation(original, Frame.J2000, frame_id='TEST_NAV_LINKED')

    assert linked._source() is original
    assert original._source() is original

##########################################################################################
