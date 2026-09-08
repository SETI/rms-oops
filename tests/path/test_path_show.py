##########################################################################################
# tests/path/test_path_show.py: Path.show() across the Path subclasses
##########################################################################################

import pytest

from polymath                    import Scalar, Vector3
from oops.frame                  import SpiceFrame
from oops.path                   import Path
from oops.path.circlepath        import CirclePath
from oops.path.coordpath         import CoordPath
from oops.path.linearcoordpath   import LinearCoordPath
from oops.path.linearpath        import LinearPath
from oops.path.multipath         import MultiPath
from oops.path.pathshift         import PathShift
from oops.path.quickpath         import QuickPath
from oops.path.spicepath         import SpicePath
from oops.surface.ringplane      import RingPlane

# `show(2)` is the shallowest level that expands a Path's own definition; levels 0 and 1
# report the ID and the one-line summary instead, which every Path shares. A Path named
# one level down is summarized rather than expanded, so it appears without quotes.
EXPANDED = 2

# A point on a ring plane, and the rate at which the point moves along it
COORDS = (Scalar(1.e5), Scalar(0.), Scalar(0.))
COORDS_DOT = (Scalar(1.), Scalar(0.), Scalar(0.))

# Enough of a QuickPath configuration to interpolate one path over one span of time
QUICK = {'path_time_step': 1.,
         'path_time_extension': 10.,
         'path_extra_steps': 4,
         'path_interpolation_order': 3,
         'path_self_check': None,
         'path_extend_ends': False}


@pytest.fixture(autouse=True)
def _kernels(core_kernels) -> None:
    """Furnish the core kernels for every test in this module."""


@pytest.fixture
def ring_plane() -> RingPlane:
    """A ring plane on which a CoordPath can name a point.

    Returns:
        RingPlane: The equatorial plane of the J2000 frame, centered on the SSB.
    """

    return RingPlane('SSB', 'J2000')


def test_the_ssb_describes_itself_by_name() -> None:
    """The root path has no definition to expand, so it names itself at every level."""

    assert Path.SSB.show(EXPANDED) == '"SSB"'


def test_a_spicepath_on_the_ssb_is_named_alone() -> None:
    """A SPICE path on the SSB in J2000 needs only its SPICE name."""

    assert SpicePath('MARS').show(EXPANDED) == 'SpicePath("MARS")'


def test_a_spicepath_on_another_origin_names_that_origin() -> None:
    """An origin other than the SSB is reported below the name."""

    path = SpicePath('MARS', SpicePath('EARTH'))

    assert path.show(EXPANDED) == 'SpicePath("MARS",\n          SpicePath(EARTH))'


def test_a_spicepath_in_another_frame_names_that_frame_too() -> None:
    """A frame other than J2000 is reported below the origin."""

    path = SpicePath('MARS', SpicePath('EARTH'), SpiceFrame('IAU_MARS'))

    assert path.show(EXPANDED) == ('SpicePath("MARS",\n'
                                   '          SpicePath(EARTH),\n'
                                   '          SpiceFrame(IAU_MARS))')


def test_a_circlepath_reports_its_orbital_elements() -> None:
    """A CirclePath is defined by a radius, a longitude, a rate and an epoch."""

    path = CirclePath(1.e5, 0., 1.e-4, 0., 'SSB', path_id='TEST_SHOW_CIRCLE')

    description = path.show(EXPANDED)

    assert description.startswith('CirclePath(radius = 100000.0,')
    assert 'lon = 0.0,' in description
    assert 'rate = 0.0001,' in description
    assert 'epoch = 0.0,' in description
    assert description.endswith('origin = SSB)')


def test_a_circlepath_reports_a_frame_that_is_not_the_origin_frame() -> None:
    """A frame the path does not inherit from its origin is named as well."""

    path = CirclePath(1.e5, 0., 1.e-4, 0., 'SSB', frame=SpiceFrame('IAU_MARS'),
                      path_id='TEST_SHOW_CIRCLE_2')

    assert path.show(EXPANDED).endswith('frame = SpiceFrame(IAU_MARS))')


def test_a_linearpath_reports_its_position_and_velocity() -> None:
    """A LinearPath is defined by a position, a velocity, an epoch and an origin."""

    path = LinearPath((Vector3([1., 2., 3.]), Vector3([4., 5., 6.])), 0., 'SSB',
                      path_id='TEST_SHOW_LINEAR')

    description = path.show(EXPANDED)

    assert description.startswith('LinearPath(pos = ([1. 2. 3.],')
    assert '[4. 5. 6.]),' in description
    assert 'epoch = Scalar(0.0),' in description
    assert description.endswith('origin = SSB)')


def test_a_linearpath_reports_a_frame_that_is_not_the_origin_frame() -> None:
    """A frame the path does not inherit from its origin is named as well."""

    path = LinearPath((Vector3([1., 2., 3.]), Vector3([4., 5., 6.])), 0., 'SSB',
                      frame=SpiceFrame('IAU_MARS'), path_id='TEST_SHOW_LINEAR_2')

    assert path.show(EXPANDED).endswith('frame = SpiceFrame(IAU_MARS))')


def test_a_multipath_reports_every_path_it_gathers() -> None:
    """A MultiPath lists each of its paths, first, middle and last alike."""

    path = MultiPath([SpicePath('MARS'), SpicePath('VENUS'), SpicePath('MERCURY')],
                     path_id='TEST_SHOW_MULTI')

    description = path.show(EXPANDED)

    assert description.startswith('MultiPath(paths = (SpicePath(MARS),')
    assert 'SpicePath(VENUS),' in description
    assert 'SpicePath(MERCURY)),' in description
    assert description.endswith('origin = SSB)')


def test_a_multipath_reports_a_frame_that_is_not_the_origin_frame() -> None:
    """A frame the path does not inherit from its origin is named as well."""

    path = MultiPath([SpicePath('MARS'), SpicePath('VENUS')],
                     frame=SpiceFrame('IAU_MARS'), path_id='TEST_SHOW_MULTI_2')

    assert path.show(EXPANDED).endswith('frame = SpiceFrame(IAU_MARS))')


def test_a_pathshift_by_a_time_offset_reports_the_offset() -> None:
    """A PathShift built on a fixed offset reports that offset and the path."""

    path = PathShift(60., SpicePath('MARS'), path_id='TEST_SHOW_SHIFT')

    assert path.show(EXPANDED).startswith('PathShift(60.0, SpicePath(MARS)')


def test_a_pathshift_reports_a_path_it_takes_its_offset_from() -> None:
    """A PathShift linked to another PathShift expands that PathShift."""

    linked = PathShift(60., SpicePath('MARS'), path_id='TEST_SHOW_SHIFT_2')
    path = PathShift(linked, SpicePath('VENUS'), path_id='TEST_SHOW_SHIFT_3')

    assert path.show(EXPANDED) == ('PathShift(PathShift(TEST_SHOW_SHIFT_2),\n'
                                   '          SpicePath(VENUS))')


def test_a_pathshift_reports_an_offset_that_cannot_be_expanded() -> None:
    """An offset that is a Scalar rather than a Path is written out directly."""

    path = PathShift(Scalar(60.), SpicePath('MARS'), path_id='TEST_SHOW_SHIFT_4')

    assert path.show(EXPANDED).startswith('PathShift(Scalar(60.0), SpicePath(MARS)')


def test_a_coordpath_reports_the_coordinates_of_its_point(ring_plane) -> None:
    """A CoordPath is a point on a surface, named by its surface coordinates."""

    path = CoordPath(ring_plane, COORDS, path_id='TEST_SHOW_COORD')

    description = path.show(EXPANDED)

    assert description.startswith('CoordPath(surface = ')
    assert 'coords = (100000.0,' in description
    assert description.endswith('0.0))')


def test_a_coordpath_reports_an_observer_when_it_has_one(ring_plane) -> None:
    """An observer, which makes the coordinates apparent rather than actual, is named."""

    path = CoordPath(ring_plane, COORDS, obs=SpicePath('EARTH'),
                     path_id='TEST_SHOW_COORD_2')

    assert path.show(EXPANDED).endswith('obs = SpicePath(EARTH))')


def test_a_linearcoordpath_reports_its_coordinates_and_their_rates(ring_plane) -> None:
    """A LinearCoordPath adds a rate for each coordinate, and an epoch."""

    path = LinearCoordPath(ring_plane, COORDS, COORDS_DOT, 0.,
                           path_id='TEST_SHOW_LINEAR_COORD')

    description = path.show(EXPANDED)

    assert description.startswith('LinearCoordPath(surface = ')
    assert 'coords = (100000.0,' in description
    assert 'coords_dot = (1.0,' in description
    assert description.endswith('epoch = Scalar(0.0))')


def test_a_linearcoordpath_reports_an_observer_when_it_has_one(ring_plane) -> None:
    """An observer, which makes the coordinates apparent rather than actual, is named."""

    path = LinearCoordPath(ring_plane, COORDS, COORDS_DOT, 0., obs=SpicePath('EARTH'),
                           path_id='TEST_SHOW_LINEAR_COORD_2')

    assert path.show(EXPANDED).endswith('obs = SpicePath(EARTH))')


def test_a_quickpath_reports_the_path_and_the_time_span_it_samples() -> None:
    """A QuickPath is an interpolation of one path over one span of time.

    The span it reports is the padded one it actually samples, not the one requested.
    """

    path = QuickPath(SpicePath('MARS'), 0., 100., QUICK)

    description = path.show(EXPANDED)

    assert description.startswith('QuickPath(SpicePath(MARS),')
    assert description.endswith(f'{path._tmin}, {path._tmax})')


def test_a_deeper_expansion_reaches_the_path_inside(ring_plane) -> None:
    """Each extra level expands one more layer of the definition.

    The inner path is summarized at level 2 and expanded at level 3, and every line
    after the first is indented past the name of the path that owns it.
    """

    inner = CirclePath(1.e5, 0., 1.e-4, 0., 'SSB', path_id='TEST_SHOW_DEPTH_INNER')
    path = MultiPath([inner, SpicePath('VENUS')], path_id='TEST_SHOW_DEPTH')

    assert 'CirclePath(TEST_SHOW_DEPTH_INNER)' in path.show(EXPANDED)

    lines = path.show(3).split('\n')

    assert lines[0] == 'MultiPath(paths = (CirclePath(radius = 100000.0,'
    assert all(line.startswith(' ' * len('MultiPath(')) for line in lines[1:])

##########################################################################################
