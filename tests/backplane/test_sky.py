##########################################################################################
# tests/backplane/test_sky.py
##########################################################################################

import numpy as np
import pytest

from oops.backplane import Backplane
from oops.constants import TWOPI

HALFPI = np.pi / 2.


def test_right_ascension_spans_the_full_circle(bp: Backplane) -> None:
    """Right ascension is an angle, reported between 0 and 2*pi."""

    ra = bp.right_ascension()

    assert np.all(ra.vals >= 0.)
    assert np.all(ra.vals < TWOPI)


def test_declination_stays_within_the_poles(bp: Backplane) -> None:
    """Declination runs from -pi/2 at the south pole to +pi/2 at the north."""

    dec = bp.declination()

    assert np.all(dec.vals >= -HALFPI)
    assert np.all(dec.vals <= HALFPI)


def test_right_ascension_has_the_shape_of_the_meshgrid(bp: Backplane) -> None:
    """A sky backplane is evaluated at every sample of the meshgrid."""

    assert bp.right_ascension().shape == bp._shape


def test_right_ascension_varies_across_the_field(bp: Backplane) -> None:
    """Neighboring pixels look in different directions."""

    ra = bp.right_ascension()

    assert ra.vals[0, 0] != ra.vals[-1, -1]


def test_apparent_and_geometric_directions_differ(bp: Backplane) -> None:
    """Stellar aberration shifts the apparent direction away from the geometric one."""

    apparent = bp.right_ascension(apparent=True)
    geometric = bp.right_ascension(apparent=False)

    assert np.any(apparent.vals != geometric.vals)


def test_apparent_aberration_is_small(bp: Backplane) -> None:
    """Aberration from the Earth's motion is well under a milliradian."""

    apparent = bp.right_ascension(apparent=True)
    geometric = bp.right_ascension(apparent=False)

    assert np.max(np.abs(apparent.vals - geometric.vals)) < 1.e-3


def test_ra_and_dec_are_cached(bp: Backplane) -> None:
    """A backplane already computed is returned rather than recomputed."""

    assert bp.right_ascension() is bp.right_ascension()
    assert bp.declination() is bp.declination()


def test_ra_dec_rejects_an_unknown_direction(bp: Backplane) -> None:
    """The photon direction must be 'arr' or 'dep'."""

    with pytest.raises(ValueError):
        bp.right_ascension(direction='sideways')


def test_celestial_north_angle_is_an_angle(bp: Backplane) -> None:
    """The angle is measured from the U-axis toward the V-axis."""

    angle = bp.celestial_north_angle()

    assert angle.shape == bp._shape
    assert np.all(np.abs(angle.vals) <= TWOPI)


def test_celestial_east_is_a_quarter_turn_from_north(bp: Backplane) -> None:
    """Celestial east lies 90 degrees from celestial north."""

    north = bp.celestial_north_angle()
    east = bp.celestial_east_angle()
    separation = np.abs((east.vals - north.vals) % TWOPI)

    assert np.allclose(np.minimum(separation, TWOPI - separation), HALFPI, atol=1.e-6)


def test_center_right_ascension_is_gridless(bp: Backplane) -> None:
    """A center backplane refers to the body's path, so it has no spatial extent."""

    assert bp.center_right_ascension('SATURN').shape == ()


def test_center_declination_is_gridless(bp: Backplane) -> None:
    """A center backplane refers to the body's path, so it has no spatial extent."""

    assert bp.center_declination('SATURN').shape == ()


def test_center_direction_falls_inside_the_disk(bp: Backplane) -> None:
    """The camera points at Saturn, so the planet's center is near the field center."""

    ra = bp.right_ascension()
    center_ra = bp.center_right_ascension('SATURN')

    assert np.min(ra.vals) <= center_ra.vals <= np.max(ra.vals)


def test_center_ra_dec_rejects_an_unknown_direction(bp: Backplane) -> None:
    """The photon direction must be 'arr' or 'dep'."""

    with pytest.raises(ValueError):
        bp.center_right_ascension('SATURN', direction='sideways')

##########################################################################################
