##########################################################################################
# tests/surface/test_nullsurface.py
##########################################################################################

from polymath                  import Vector3
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

##########################################################################################
