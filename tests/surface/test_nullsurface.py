##########################################################################################
# tests/surface/test_nullsurface.py
##########################################################################################

import pickle

from polymath                  import Scalar, Vector3
from oops.frame                import Frame
from oops.path                 import Path
from oops.surface.nullsurface  import NullSurface


def _surface():
    """A NullSurface centered on the solar system barycenter.

    Returns:
        NullSurface: The surface, which intercepts nothing.
    """

    return NullSurface('SSB', 'J2000')


def test_intercept_returns_masked_values() -> None:
    """Nothing intercepts this surface, so the results are masked throughout."""

    (pos, t) = _surface().intercept(Vector3((1., 2., 3.)), Vector3((0., 0., -1.)))

    assert pos.mask
    assert t.mask


def test_intercept_matches_the_shape_of_its_inputs() -> None:
    """The masked results take the shape they would have had."""

    obs = Vector3([(1., 2., 3.), (4., 5., 6.)])

    (pos, t) = _surface().intercept(obs, Vector3((0., 0., -1.)))

    assert pos.shape == (2,)
    assert t.shape == (2,)


def test_intercept_keeps_the_derivatives_of_its_inputs() -> None:
    """The masked results carry the derivatives the caller asked to propagate."""

    obs = Vector3((1., 2., 3.))
    obs.insert_deriv('los', Vector3.IDENTITY)

    (pos, t) = _surface().intercept(obs, Vector3((0., 0., -1.)), derivs=True)

    assert 'los' in pos.derivs
    assert 'los' in t.derivs


def test_intercept_returns_hints_when_given() -> None:
    """A hints value passes through to the returned tuple."""

    result = _surface().intercept(Vector3((1., 2., 3.)), Vector3((0., 0., -1.)),
                                  hints=True)

    assert len(result) == 3
    assert result[2] is True


def test_coords_from_vector3_returns_the_rectangular_coordinates() -> None:
    """The coordinates are the (x,y) of the position in the surface's frame."""

    (x, y) = _surface().coords_from_vector3(Vector3((1., 2., 3.)))

    assert x == Scalar(1.)
    assert y == Scalar(2.)


def test_coords_from_vector3_can_return_three_coordinates() -> None:
    """axes=3 adds the z-coordinate."""

    (x, y, z) = _surface().coords_from_vector3(Vector3((1., 2., 3.)), axes=3)

    assert x == Scalar(1.)
    assert y == Scalar(2.)
    assert z == Scalar(3.)


def test_coords_from_vector3_returns_hints_when_given() -> None:
    """A hints value passes through to the returned tuple."""

    result = _surface().coords_from_vector3(Vector3((1., 2., 3.)), hints=True)

    assert len(result) == 3
    assert result[2] is True


def test_coords_from_vector3_keeps_the_derivatives() -> None:
    """derivs=True propagates the derivatives inside pos into the coordinates."""

    pos = Vector3((1., 2., 3.))
    pos.insert_deriv('t', Vector3((1., 1., 1.)))

    (x, y) = _surface().coords_from_vector3(pos, derivs=True)

    assert 't' in x.derivs
    assert 't' in y.derivs


def test_coords_from_vector3_drops_the_derivatives_by_default() -> None:
    """derivs=False leaves the derivatives out of the returned coordinates."""

    pos = Vector3((1., 2., 3.))
    pos.insert_deriv('t', Vector3((1., 1., 1.)))

    (x, _) = _surface().coords_from_vector3(pos, derivs=False)

    assert 't' not in x.derivs


def test_vector3_from_coords_inverts_coords_from_vector3() -> None:
    """The two conversions are exact inverses of one another."""

    pos = Vector3((1., 2., 3.))
    coords = _surface().coords_from_vector3(pos, axes=3)

    assert _surface().vector3_from_coords(coords) == pos


def test_vector3_from_coords_accepts_two_coordinates() -> None:
    """With only (x,y) given, the z-coordinate is zero."""

    pos = _surface().vector3_from_coords((Scalar(1.), Scalar(2.)))

    assert pos == Vector3((1., 2., 0.))


def test_vector3_from_coords_returns_hints_when_given() -> None:
    """A hints value passes through to the returned tuple."""

    result = _surface().vector3_from_coords((Scalar(1.), Scalar(2.)), hints=True)

    assert len(result) == 2
    assert result[1] is True


def test_normal_is_the_z_axis() -> None:
    """The surface's normal is defined by the z-axis of its frame."""

    normal = _surface().normal(Vector3((1., 2., 3.)))

    assert normal.unit() == Vector3((0., 0., 1.))


def test_normal_returns_hints_when_given() -> None:
    """A hints value passes through to the returned tuple."""

    result = _surface().normal(Vector3((1., 2., 3.)), hints=True)

    assert len(result) == 2
    assert result[1] is True


def test_velocity_is_zero() -> None:
    """A NullSurface has no local motion."""

    assert _surface().velocity(Vector3((1., 2., 3.))) == Vector3((0., 0., 0.))


def test_velocity_is_zero_for_every_position() -> None:
    """A shaped position is still motionless everywhere."""

    velocity = _surface().velocity(Vector3([(1., 2., 3.), (4., 5., 6.)]))

    assert velocity == Vector3((0., 0., 0.))


def test_surface_reports_its_origin_and_frame() -> None:
    """The surface is centered on the given path and uses the given frame."""

    surface = _surface()

    assert surface.origin == Path.as_path('SSB').waypoint
    assert surface.frame == Frame.as_frame('J2000').wayframe


def test_pickle_restores_the_surface() -> None:
    """Pickling restores the origin and the frame."""

    surface = _surface()
    restored = pickle.loads(pickle.dumps(surface))

    assert isinstance(restored, NullSurface)
    assert restored.origin == surface.origin
    assert restored.frame == surface.frame

##########################################################################################
