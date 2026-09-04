##########################################################################################
# tests/backplane/test_where.py
##########################################################################################

import numpy as np
import pytest

from oops.backplane import Backplane

PLANET = 'SATURN'
RING = 'SATURN:RING'


def test_where_intercepted_marks_the_disk(bp: Backplane) -> None:
    """The planet covers part, but not all, of the field of view."""

    intercepted = bp.where_intercepted(PLANET)

    assert intercepted.shape == bp._shape
    assert 0 < np.sum(intercepted.vals) < intercepted.size


def test_where_intercepted_agrees_with_the_antimask_of_a_backplane(bp: Backplane) -> None:
    """A surface backplane is unmasked exactly where the surface was intercepted."""

    intercepted = bp.where_intercepted(PLANET)

    assert np.all(intercepted.vals == bp.incidence_angle(PLANET).antimask)


def test_where_intercepted_of_a_ring_plane_covers_everything(bp: Backplane) -> None:
    """A ring plane is unbounded, so every line of sight crosses it somewhere."""

    assert np.all(bp.where_intercepted(RING).vals)


def test_sunward_and_antisunward_partition_the_surface(bp: Backplane) -> None:
    """Every intercepted pixel faces either toward or away from the Sun."""

    sunward = bp.where_sunward(PLANET)
    antisunward = bp.where_antisunward(PLANET)
    intercepted = bp.where_intercepted(PLANET)

    assert np.all((sunward.vals | antisunward.vals) == intercepted.vals)
    assert not np.any(sunward.vals & antisunward.vals)


def test_where_sunward_agrees_with_the_incidence_angle(bp: Backplane) -> None:
    """The sunlit side is where the incidence angle is less than 90 degrees."""

    sunward = bp.where_sunward(PLANET)
    lit = bp.incidence_angle(PLANET).vals < np.pi / 2.

    assert np.all(sunward.vals == (lit & bp.where_intercepted(PLANET).vals))


def test_three_valued_logic_masks_the_pixels_off_the_surface(bp: Backplane) -> None:
    """With tvl=True, locations outside the surface are masked rather than False."""

    plain = bp.where_sunward(PLANET)
    tvl = bp.where_sunward(PLANET, tvl=True)

    assert not np.any(plain.mask)
    assert np.any(tvl.mask)


def test_three_valued_logic_agrees_where_it_is_not_masked(bp: Backplane) -> None:
    """Where both are defined, the two forms give the same answer."""

    plain = bp.where_sunward(PLANET)
    tvl = bp.where_sunward(PLANET, tvl=True)

    assert np.all(plain.vals[tvl.antimask] == tvl.vals[tvl.antimask])


def test_in_front_and_in_back_partition_the_ring(bp: Backplane) -> None:
    """Each intercepted ring pixel is either in front of the planet or behind it."""

    in_front = bp.where_in_front(RING, PLANET)
    in_back = bp.where_in_back(RING, PLANET)

    assert np.all((in_front.vals | in_back.vals) == bp.where_intercepted(RING).vals)
    assert not np.any(in_front.vals & in_back.vals)


def test_the_ring_passes_in_front_of_the_planet_somewhere(bp: Backplane) -> None:
    """Part of the ring plane lies between the observer and Saturn."""

    assert np.any(bp.where_in_front(RING, PLANET).vals)


def test_the_ring_passes_behind_the_planet_somewhere(bp: Backplane) -> None:
    """Part of the ring plane is hidden by the globe of Saturn."""

    assert np.any(bp.where_in_back(RING, PLANET).vals)


def test_inside_and_outside_shadow_cover_the_ring(bp: Backplane) -> None:
    """Every intercepted ring pixel is shadowed by the planet or outside its shadow."""

    inside = bp.where_inside_shadow(RING, PLANET)
    outside = bp.where_outside_shadow(RING, PLANET)

    assert np.all((inside.vals | outside.vals) == bp.where_intercepted(RING).vals)


def test_inside_and_outside_shadow_are_exclusive_where_the_ring_is_visible(
        bp: Backplane) -> None:
    """On the part of the ring plane in front of the planet, the two are complements."""

    inside = bp.where_inside_shadow(RING, PLANET)
    outside = bp.where_outside_shadow(RING, PLANET)
    visible = bp.where_in_front(RING, PLANET).vals

    assert not np.any(inside.vals[visible] & outside.vals[visible])


def test_the_planet_casts_a_shadow_on_the_ring(bp: Backplane) -> None:
    """Saturn's shadow falls across part of the ring plane."""

    assert np.any(bp.where_inside_shadow(RING, PLANET).vals)


def test_shadow_requires_an_event_key_of_two_items(bp: Backplane) -> None:
    """The standardized event key must name the light source and one body."""

    with pytest.raises(ValueError):
        bp.where_inside_shadow(('SUN<', 'MIMAS', RING), PLANET)


def test_inside_and_outside_partition_the_surface(bp: Backplane) -> None:
    """Each intercepted pixel is either inside the second surface or outside it."""

    inside = bp.where_inside(RING, PLANET)
    outside = bp.where_outside(RING, PLANET)

    assert np.all((inside.vals | outside.vals) == bp.where_intercepted(RING).vals)
    assert not np.any(inside.vals & outside.vals)


def test_where_below_selects_the_smaller_values(bp: Backplane) -> None:
    """A pixel is below the limit when its backplane value is at or under it."""

    limit = 1.
    below = bp.where_below(('incidence_angle', PLANET), limit)
    values = bp.incidence_angle(PLANET)

    assert np.all(below.vals == ((values.vals <= limit) & values.antimask))


def test_where_above_selects_the_larger_values(bp: Backplane) -> None:
    """A pixel is above the limit when its backplane value is at or over it."""

    limit = 1.
    above = bp.where_above(('incidence_angle', PLANET), limit)
    values = bp.incidence_angle(PLANET)

    assert np.all(above.vals == ((values.vals >= limit) & values.antimask))


def test_where_between_is_the_intersection_of_above_and_below(bp: Backplane) -> None:
    """A pixel between two limits is both above the lower and below the upper."""

    key = ('incidence_angle', PLANET)
    between = bp.where_between(key, 0.5, 1.5)
    above = bp.where_above(key, 0.5)
    below = bp.where_below(key, 1.5)

    assert np.all(between.vals == (above.vals & below.vals))


def test_where_not_inverts_a_mask(bp: Backplane) -> None:
    """where_not reverses the sense of a Boolean backplane."""

    sunward = bp.where_sunward(PLANET)
    inverted = bp.where_not(('where_sunward', PLANET))

    assert np.all(inverted.vals == ~sunward.vals)


def test_where_any_is_the_union(bp: Backplane) -> None:
    """where_any is True where any of the given backplanes is True."""

    sunward = bp.where_sunward(PLANET)
    in_front = bp.where_in_front(RING, PLANET)
    combined = bp.where_any(('where_sunward', PLANET),
                            ('where_in_front', RING, PLANET))

    assert np.all(combined.vals == (sunward.vals | in_front.vals))


def test_where_all_is_the_intersection(bp: Backplane) -> None:
    """where_all is True where every one of the given backplanes is True."""

    sunward = bp.where_sunward(PLANET)
    in_front = bp.where_in_front(RING, PLANET)
    combined = bp.where_all(('where_sunward', PLANET),
                            ('where_in_front', RING, PLANET))

    assert np.all(combined.vals == (sunward.vals & in_front.vals))


def test_masks_are_cached(bp: Backplane) -> None:
    """A mask already computed is returned rather than recomputed."""

    assert bp.where_intercepted(PLANET) is bp.where_intercepted(PLANET)
    assert bp.where_sunward(PLANET) is bp.where_sunward(PLANET)

##########################################################################################
