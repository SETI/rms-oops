##########################################################################################
# tests/backplane/test_backplane.py: the Backplane class itself, and the border masks
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath         import Scalar
from oops.backplane   import Backplane
from oops.observation import Snapshot

PLANET = 'SATURN'
RING = 'SATURN:RING'


##########################################################################################
# Event keys
##########################################################################################

def test_a_bare_surface_key_means_dispersed_sunlight() -> None:
    """A surface key alone is understood as dispersed illumination from the Sun."""

    assert Backplane.standardize_event_key(PLANET) == ('SUN<', 'SATURN')


def test_event_keys_are_uppercased() -> None:
    """Source and surface keys are not case-sensitive."""

    assert Backplane.standardize_event_key('saturn:ring') == ('SUN<', 'SATURN:RING')


def test_an_empty_event_key_stays_empty() -> None:
    """An empty event key refers to the observation itself."""

    assert Backplane.standardize_event_key(()) == ()


def test_an_explicit_source_is_kept() -> None:
    """A source given in the key replaces the default Sun."""

    assert Backplane.standardize_event_key(('SUN>', RING)) == ('SUN>', 'SATURN:RING')


@pytest.mark.parametrize('default, expected',
                         [('RING', ('SUN<', 'SATURN:RING')),
                          ('ANSA', ('SUN<', 'SATURN:ANSA')),
                          ('LIMB', ('SUN<', 'SATURN:LIMB'))])
def test_a_default_suffix_is_appended(default: str, expected: tuple,
                                      solar_system: None) -> None:
    """A default suffix names the associated surface of a bare body name."""

    assert Backplane.standardize_event_key(PLANET, default=default) == expected


def test_a_default_suffix_does_not_override_an_explicit_one(solar_system: None) -> None:
    """A key that already names a surface keeps it."""

    assert Backplane.standardize_event_key(RING, default='ANSA') \
           == ('SUN<', 'SATURN:RING')


def test_a_shadowing_key_carries_three_items() -> None:
    """An extra surface after the source describes one surface shadowing another."""

    key = Backplane.standardize_event_key(('SUN<', 'MIMAS', RING))

    assert len(key) == 3
    assert Backplane._is_shadowing(key)


def test_dispersed_illumination_rejects_a_key_of_four_items() -> None:
    """Dispersed illumination takes two or three items, never more."""

    with pytest.raises(ValueError):
        Backplane.standardize_event_key(('SUN<', 'MIMAS', RING, PLANET))


def test_occultation_illumination_takes_exactly_two_items() -> None:
    """Occultation illumination has no room for a shadowing surface."""

    with pytest.raises(ValueError):
        Backplane.standardize_event_key(('SUN>', 'MIMAS', RING))


def test_path_based_illumination_takes_exactly_two_items() -> None:
    """Path-based illumination has no room for a shadowing surface either."""

    with pytest.raises(ValueError):
        Backplane.standardize_event_key(('SUN-', 'MIMAS', RING))


@pytest.mark.parametrize('source, dispersed, occultation, gridless',
                         [('SUN<', True, False, False),
                          ('SUN>', False, True, False),
                          ('SUN-', False, False, True)])
def test_the_source_suffix_selects_the_kind_of_illumination(
        source: str, dispersed: bool, occultation: bool, gridless: bool) -> None:
    """'<' is dispersed, '>' is occultation and '-' is path-based."""

    key = Backplane.standardize_event_key((source, PLANET))

    assert Backplane._is_dispersed(key) is dispersed
    assert Backplane._is_occultation(key) is occultation
    assert Backplane._is_gridless(key) is gridless


def test_an_empty_key_counts_as_dispersed() -> None:
    """An empty key describes dispersed illumination, and nothing else."""

    assert Backplane._is_dispersed(())
    assert not Backplane._is_occultation(())
    assert not Backplane._is_gridless(())


def test_gridless_event_key_marks_the_source_as_path_based() -> None:
    """The gridless form of a key uses path-based illumination."""

    key = Backplane.gridless_event_key(PLANET)

    assert Backplane._is_gridless(key)
    assert key == ('SUN-', 'SATURN')


def test_gridless_event_key_leaves_an_empty_key_alone() -> None:
    """An empty key is returned unchanged."""

    assert Backplane.gridless_event_key(()) == ()


##########################################################################################
# Backplane keys
##########################################################################################

def test_a_backplane_key_string_becomes_an_uppercase_tuple(bp: Backplane) -> None:
    """A string key is turned into a one-item tuple and converted to upper case."""

    assert bp.standardize_backplane_key('right_ascension') == ('RIGHT_ASCENSION',)


def test_a_backplane_key_tuple_is_returned_as_it_is(bp: Backplane) -> None:
    """A key already given as a tuple needs no repair."""

    key = ('ring_radius', 'SATURN:RING')

    assert bp.standardize_backplane_key(key) == key


def test_a_registered_backplane_yields_its_own_key(bp: Backplane) -> None:
    """Handed a backplane array, the key is extracted from it."""

    array = bp.ring_radius(RING)

    assert bp.standardize_backplane_key(array) == array.key


def test_an_unregistered_array_is_not_a_backplane_key(bp: Backplane) -> None:
    """An array that was never registered has no key to extract."""

    with pytest.raises(ValueError):
        bp.standardize_backplane_key(Scalar(np.zeros(bp.shape)))


def test_a_backplane_key_must_be_a_string_or_a_tuple(bp: Backplane) -> None:
    """Anything else cannot index the dictionary of backplanes."""

    with pytest.raises(ValueError):
        bp.standardize_backplane_key(42)


def test_evaluate_calls_the_named_backplane(bp: Backplane) -> None:
    """evaluate() dispatches a key to the method that generates it."""

    assert bp.evaluate(('ring_radius', RING)) == bp.ring_radius(RING)


def test_evaluate_passes_the_extra_arguments(bp: Backplane) -> None:
    """The remaining items of the key become the method's arguments."""

    assert bp.evaluate(('ring_longitude', RING, 'obs')) \
           == bp.ring_longitude(RING, reference='obs')


##########################################################################################
# Construction and caching
##########################################################################################

def test_backplane_samples_every_pixel_by_default(saturn_obs: Snapshot) -> None:
    """The default meshgrid samples the center of every pixel."""

    assert Backplane(saturn_obs).shape == tuple(saturn_obs.uv_shape)


def test_backplane_accepts_a_meshgrid(saturn_obs: Snapshot) -> None:
    """An explicit meshgrid defines the sampling of the field of view."""

    meshgrid = saturn_obs.meshgrid(undersample=4)
    backplane = Backplane(saturn_obs, meshgrid=meshgrid)

    assert backplane.shape == meshgrid.shape


def test_backplane_accepts_a_time(saturn_obs: Snapshot) -> None:
    """An explicit time replaces the mid-time of every pixel."""

    backplane = Backplane(saturn_obs, time=Scalar(saturn_obs.midtime))

    assert backplane.right_ascension().shape == backplane.shape


def test_an_inventory_gives_the_same_geometry(saturn_obs: Snapshot) -> None:
    """Keeping an inventory of bodies speeds up the work without changing it."""

    plain = Backplane(saturn_obs)
    with_inventory = Backplane(saturn_obs, inventory=True, inventory_border=2)

    assert np.all(with_inventory.where_intercepted(PLANET).vals
                  == plain.where_intercepted(PLANET).vals)


def test_backplane_survives_a_pickle_round_trip(saturn_obs: Snapshot) -> None:
    """The events are preserved so backplanes generate quickly after unpickling."""

    backplane = Backplane(saturn_obs)
    expected = backplane.ring_radius(RING)

    restored = pickle.loads(pickle.dumps(backplane))

    assert restored.shape == backplane.shape
    assert restored.ring_radius(RING) == expected


##########################################################################################
# Borders
##########################################################################################

def test_border_above_is_inside_the_region(bp: Backplane) -> None:
    """The border pixels are the inner edge of the region, not a ring around it."""

    value = 100000.
    border = bp.border_above(('ring_radius', RING), value)
    above = bp.where_above(('ring_radius', RING), value)

    assert np.any(border.vals)
    assert np.all(above.vals[border.vals])


def test_border_below_is_inside_the_region(bp: Backplane) -> None:
    """The border pixels are the inner edge of the region below the value."""

    value = 100000.
    border = bp.border_below(('ring_radius', RING), value)
    below = bp.where_below(('ring_radius', RING), value)

    assert np.any(border.vals)
    assert np.all(below.vals[border.vals])


def test_the_two_borders_are_adjacent_but_disjoint(bp: Backplane) -> None:
    """One lies just inside the boundary and the other just outside it."""

    value = 100000.
    above = bp.border_above(('ring_radius', RING), value)
    below = bp.border_below(('ring_radius', RING), value)

    assert not np.any(above.vals & below.vals)


def test_border_atop_straddles_the_transition(bp: Backplane) -> None:
    """The pixels closest to the transition come from both sides of it."""

    value = 100000.
    atop = bp.border_atop(('ring_radius', RING), value)

    assert np.any(atop.vals)
    assert np.sum(atop.vals) < atop.size


def test_border_inside_is_a_subset_of_the_region(bp: Backplane) -> None:
    """The inner border is made of True pixels adjacent to False ones."""

    mask = bp.where_intercepted(PLANET)
    border = bp.border_inside(('where_intercepted', PLANET))

    assert np.any(border.vals)
    assert np.all(mask.vals[border.vals])


def test_border_outside_lies_beyond_the_region(bp: Backplane) -> None:
    """The outer border is made of False pixels adjacent to True ones."""

    mask = bp.where_intercepted(PLANET)
    border = bp.border_outside(('where_intercepted', PLANET))

    assert np.any(border.vals)
    assert not np.any(mask.vals[border.vals])


def test_the_inner_and_outer_borders_are_disjoint(bp: Backplane) -> None:
    """No pixel is both inside and outside the region."""

    inside = bp.border_inside(('where_intercepted', PLANET))
    outside = bp.border_outside(('where_intercepted', PLANET))

    assert not np.any(inside.vals & outside.vals)


def test_border_requires_a_boolean_backplane(bp: Backplane) -> None:
    """border_inside and border_outside act on masks, not on measurements."""

    with pytest.raises(ValueError):
        bp.border_inside(('ring_radius', RING))


def test_border_backplanes_are_cached(bp: Backplane) -> None:
    """A backplane already computed is returned rather than recomputed."""

    key = ('where_intercepted', PLANET)

    assert bp.border_inside(key) is bp.border_inside(key)

##########################################################################################
