##########################################################################################
# tests/surface/test_surface.py: Abstract class Surface
##########################################################################################

import numpy as np
import pytest

from polymath               import Boolean, Scalar, Vector3
from oops.event             import Event
from oops.surface           import Surface
from oops.surface.ringplane import RingPlane
from oops.surface.spheroid  import Spheroid


def test_surface():

    np.random.seed(6631)

    # Most methods are heavily tested elsewhere

    # Surface.resolution...

    # Make sure the rotated resolution vectors are perpendicular
    dpos_duv = Vector3(np.random.randn(10000, 3, 2), drank=1)
    (new_du, new_dv) = Surface._resolution(dpos_duv, _unittest=True)
    assert new_du.dot(new_dv).max() < 1.e-12

    # We also expect area to be conserved
    dpos_du = Vector3(dpos_duv.values[...,0])
    dpos_dv = Vector3(dpos_duv.values[...,1])
    diffs = dpos_du.cross(dpos_dv) - new_du.cross(new_dv)
    assert diffs.norm().max() < 1.e-14

##########################################################################################
# The base-class methods the subclasses inherit
##########################################################################################

def _ringplane() -> RingPlane:
    """A ring plane, which is a surface with no interior.

    Returns:
        RingPlane: The equatorial plane of J2000, centered on the SSB.
    """

    return RingPlane('SSB', 'J2000')


def _spheroid() -> Spheroid:
    """An oblate spheroid, which is a surface with an interior.

    Returns:
        Spheroid: A body 60268 km across the equator and 50000 km pole to pole.
    """

    return Spheroid('SSB', 'J2000', (60268., 50000.))


def test_coords_from_vector3_needs_two_or_three_axes() -> None:
    """A surface has two or three coordinates, and nothing else."""

    with pytest.raises(ValueError, match='axes must be 2 or 3 in RingPlane'):
        _ringplane().coords_from_vector3(Vector3((1.e5, 0., 0.)), axes=4)


def test_vector3_from_coords_needs_two_or_three_coordinates() -> None:
    """The same holds in the other direction."""

    with pytest.raises(ValueError, match='2 or 3 coords required in RingPlane'):
        _ringplane().vector3_from_coords((Scalar(1.e5),))


def test_a_surface_with_no_interior_holds_nothing_inside_it() -> None:
    """A ring plane has no interior, so no position is inside it."""

    assert _ringplane().position_is_inside(Vector3((1.e5, 0., 0.))) is Boolean.FALSE


def test_a_surface_that_has_an_interior_must_implement_the_test() -> None:
    """The base-class method refuses to answer for a surface with a real interior."""

    surface = _ringplane()
    surface.HAS_INTERIOR = True     # a RingPlane does not override the base method

    with pytest.raises(NotImplementedError, match='RingPlane.position_is_inside'):
        surface.position_is_inside(Vector3((1.e5, 0., 0.)))


def test_a_surface_is_its_own_reference_by_default() -> None:
    """Only a surface derived from another overrides this."""

    surface = _ringplane()

    assert surface.reference() is surface


def test_apply_coords_to_event_adds_the_coordinate_subfields() -> None:
    """The coordinates of the event's position are attached to a copy of it."""

    surface = _ringplane()
    pos = Vector3([(1.e5, 0., 0.), (0., 2.e5, 0.)])
    event = Event(Scalar(0.), pos, surface.origin, surface.frame)

    result = surface.apply_coords_to_event(event, axes=3)

    assert result is not event
    assert result.coord1 == Scalar([1.e5, 2.e5])
    assert result.coord3 == Scalar([0., 0.])


def test_apply_coords_to_event_can_add_only_two_coordinates() -> None:
    """With axes=2 the third coordinate is left off."""

    surface = _ringplane()
    event = Event(Scalar(0.), Vector3((1.e5, 0., 0.)), surface.origin, surface.frame)

    result = surface.apply_coords_to_event(event, axes=2)

    assert 'coord2' in result.subfields
    assert 'coord3' not in result.subfields


def test_apply_coords_to_event_masks_the_event_where_the_coordinates_are_masked() -> None:
    """A position outside the surface's limits masks the event that names it."""

    surface = RingPlane('SSB', 'J2000', radii=(74000., 140000.))
    pos = Vector3([(1.e5, 0., 0.), (1.e6, 0., 0.)])
    event = Event(Scalar(0.), pos, surface.origin, surface.frame)

    result = surface.apply_coords_to_event(event)

    assert list(result.mask) == [False, True]


def test_event_at_coords_places_the_event_on_the_surface() -> None:
    """Two coordinates put the event in the plane; the third defaults to zero."""

    surface = _ringplane()

    event = surface.event_at_coords(Scalar(0.), (Scalar(1.e5), Scalar(0.)))

    assert isinstance(event, Event)
    assert event.pos == Vector3((1.e5, 0., 0.))
    assert event.origin is surface.origin


def test_event_at_coords_accepts_a_third_coordinate() -> None:
    """A third coordinate lifts the event off the plane."""

    surface = _ringplane()

    event = surface.event_at_coords(Scalar(0.),
                                    (Scalar(1.e5), Scalar(0.), Scalar(500.)))

    assert event.pos == Vector3((1.e5, 0., 500.))


def test_event_at_coords_strips_the_derivatives_of_its_arguments() -> None:
    """Without derivs, only the time derivatives survive into the event."""

    surface = _ringplane()
    radius = Scalar(1.e5)
    radius.insert_deriv('rad', Scalar(1.))
    obs = Vector3((0., 0., 1.e6))
    obs.insert_deriv('obs', Vector3.IDENTITY)

    event = surface.event_at_coords(Scalar(0.), (radius, Scalar(0.)), obs=obs)

    assert 'rad' not in event.pos.derivs
    assert 'obs' not in event.pos.derivs


def test_the_public_resolution_reports_the_extremes() -> None:
    """The finest and coarsest resolutions are the lengths of the orthogonalized axes."""

    dpos_duv = Vector3(np.array([[3., 0.], [0., 4.], [0., 0.]]), drank=1)

    (minres, maxres) = Surface.resolution(dpos_duv)

    assert minres.vals == pytest.approx(3., abs=1.e-12)
    assert maxres.vals == pytest.approx(4., abs=1.e-12)

##########################################################################################
