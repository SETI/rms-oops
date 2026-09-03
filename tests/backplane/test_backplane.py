##########################################################################################
# tests/backplane/test_backplane.py: the Backplane class itself, and the border masks
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath         import Boolean, Scalar
from oops.backplane   import Backplane
from oops.config      import LOGGING
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
# The remaining event-key and surface-key conversions
##########################################################################################

# A ring plane defined as a body in its own right, rather than as the ":RING" surface of
# a planet. Its own surface is a ring, so it names an intercept the planet's ring shares.
RING_BODY = 'SATURN_RING_PLANE'

# An eccentric, inclined ring, whose intercept geometry is its own
ECCENTRIC_RING = 'ALPHA_RING'


def test_a_key_naming_two_surfaces_gains_the_sun(solar_system: None) -> None:
    """A tuple that names no source is understood as dispersed sunlight."""

    assert Backplane.standardize_event_key((PLANET, RING)) \
           == ('SUN<', 'SATURN', 'SATURN:RING')


def test_a_repeated_source_is_dropped(solar_system: None) -> None:
    """A key that names the Sun twice, as older keys did, names it once."""

    assert Backplane.standardize_event_key(('SUN<', 'SUN', PLANET)) \
           == ('SUN<', 'SATURN')


def test_an_ansa_suffix_is_added_to_a_ring_body(solar_system: None) -> None:
    """A body whose own surface is a ring accepts the "ANSA" default suffix."""

    assert Backplane.standardize_event_key(RING_BODY, default='ANSA') \
           == ('SUN<', 'SATURN_RING_PLANE:ANSA')


def test_an_empty_key_describes_no_shadowing() -> None:
    """A key with nothing in it names no surface, so it shadows nothing."""

    assert not Backplane._is_shadowing(())


def test_the_ring_of_a_ring_body_is_its_own_unmasked_surface(solar_system: None) -> None:
    """An uninclined ring on its planet's ring frame shares the planet's ring intercept.
    """

    assert Backplane.unmasked_surface_key(RING_BODY) == 'SATURN_RING_PLANE:RING'


def test_an_eccentric_ring_is_its_own_unmasked_surface(solar_system: None) -> None:
    """A ring with a non-zero inclination has intercept geometry of its own."""

    assert Backplane.unmasked_surface_key(ECCENTRIC_RING) == ECCENTRIC_RING


def test_the_ansa_of_a_ring_body_belongs_to_its_planet(solar_system: None) -> None:
    """An ansa surface hangs off the planet, so a ring body defers to its parent."""

    assert Backplane.unmasked_surface_key(RING_BODY + ':ANSA') == 'SATURN:ANSA'


def test_an_unrecognized_surface_modifier_is_refused(solar_system: None) -> None:
    """The modifier after the colon must name a surface the Backplane knows."""

    with pytest.raises(KeyError):
        Backplane.get_surface('SATURN:HALO')


##########################################################################################
# The cached derivative properties
##########################################################################################

@pytest.mark.parametrize('name', ['dlos_duv', 'dlos_duv1', 'duv_dlos',
                                  'center_dlos_duv', 'center_duv_dlos'])
def test_a_derivative_property_is_evaluated_once(name: str, bp: Backplane) -> None:
    """Each derivative of the pixel geometry is cached on first use."""

    assert getattr(bp, name) is getattr(bp, name)


def test_the_orthogonalized_pixel_axes_keep_the_pixel_area(bp: Backplane) -> None:
    """dlos_duv1 shifts the longer pixel edge to be orthogonal, conserving the area."""

    (dlos_du, dlos_dv) = bp.dlos_duv.extract_denoms()
    (dlos_du1, dlos_dv1) = bp.dlos_duv1.extract_denoms()

    original = dlos_du.cross(dlos_dv).norm()
    orthogonal = dlos_du1.cross(dlos_dv1).norm()

    assert np.allclose(original.vals, orthogonal.vals, rtol=1.e-9)
    assert np.allclose(dlos_du1.dot(dlos_dv1).vals, 0., atol=1.e-20)


##########################################################################################
# register_backplane, get_backplane and evaluate
##########################################################################################

def test_a_plain_boolean_is_registered_as_a_backplane(bp: Backplane) -> None:
    """A Python or NumPy bool becomes a shapeless Boolean."""

    registered = bp.register_backplane(('_test_flag',), True)

    assert registered.shape == ()
    assert bool(registered)


def test_a_numpy_array_is_registered_as_a_scalar(bp: Backplane) -> None:
    """A bare NumPy array becomes a Scalar of the same shape."""

    registered = bp.register_backplane(('_test_array',), np.zeros(bp.shape))

    assert isinstance(registered, Scalar)
    assert registered.shape == bp.shape


def test_a_shapeless_boolean_can_be_expanded_to_the_grid(bp: Backplane) -> None:
    """With expand, a constant Boolean is broadcast to the shape of the backplane."""

    registered = bp.register_backplane(('_test_expanded_flag',), Boolean(True),
                                       expand=True)

    assert registered.shape == bp.shape
    assert np.all(registered.vals)


def test_a_shapeless_scalar_can_be_expanded_to_the_grid(bp: Backplane) -> None:
    """With expand, a constant Scalar is broadcast to the shape of the backplane."""

    registered = bp.register_backplane(('_test_expanded_value',), Scalar(2.5),
                                       expand=True)

    assert registered.shape == bp.shape
    assert np.all(registered.vals == 2.5)


def test_a_backplane_with_derivatives_is_kept_separately(saturn_obs: Snapshot) -> None:
    """Asking for derivatives returns the array that still carries them.

    The key has to be the registered one, which names the standardized event key and
    every argument of the backplane function.
    """

    backplane = Backplane(saturn_obs)
    key = ('ring_radius', Backplane.standardize_event_key(RING), None, None)

    with_derivs = backplane.evaluate(key, derivs=True)

    assert with_derivs.derivs
    assert not backplane.evaluate(key).derivs
    assert np.all(with_derivs.vals == backplane.evaluate(key).vals)


def test_get_backplane_returns_the_version_with_derivatives(
        saturn_obs: Snapshot) -> None:
    """The same distinction applies to a direct lookup by key."""

    backplane = Backplane(saturn_obs)
    key = ('ring_radius', Backplane.standardize_event_key(RING), None, None)
    backplane.evaluate(key, derivs=True)

    assert backplane.get_backplane(key, derivs=True).derivs
    assert not backplane.get_backplane(key).derivs


def test_evaluate_accepts_a_bare_backplane_name(bp: Backplane) -> None:
    """A string key names a backplane function that takes no further arguments."""

    assert bp.evaluate('right_ascension') is bp.right_ascension()


def test_evaluate_rejects_an_unrecognized_name(bp: Backplane) -> None:
    """The first item of a backplane key must name a defined backplane function."""

    with pytest.raises(ValueError, match='unrecognized backplane function: not_a_name'):
        bp.evaluate(('not_a_name', RING))


##########################################################################################
# Surface, gridless and occultation events
##########################################################################################

def test_an_empty_event_key_gives_the_observer_event(bp: Backplane) -> None:
    """The empty key, used by the sky backplanes, resolves to the observation itself."""

    event = bp.get_surface_event(())

    assert event.shape == bp.shape
    assert event.origin is bp.obs.path.waypoint


def test_an_occultation_event_is_solved_from_the_source(saturn_obs: Snapshot) -> None:
    """A ">" source key describes the body occulting that source, seen from the observer.
    """

    backplane = Backplane(saturn_obs)

    event = backplane.get_surface_event(('SUN>', PLANET))

    assert event.shape == ()
    assert backplane.get_gridless_event(('SUN>', PLANET)) is not None


def test_a_gridless_arrival_is_solved_once_per_source(saturn_obs: Snapshot) -> None:
    """Two gridless keys naming the same body share the arrival event already solved."""

    backplane = Backplane(saturn_obs)

    first = backplane.get_gridless_event(Backplane.gridless_event_key(PLANET))
    second = backplane.get_gridless_event(Backplane.gridless_event_key(RING))

    assert first.shape == ()
    assert second.shape == ()


##########################################################################################
# The inventory antimasks
##########################################################################################

def test_a_surface_with_a_modifier_has_no_antimask(saturn_obs: Snapshot) -> None:
    """A name with a colon names a surface the inventory does not describe."""

    backplane = Backplane(saturn_obs, inventory={})

    assert backplane.get_antimask(RING) is True


def test_a_body_inside_the_field_is_masked_to_its_own_footprint(
        saturn_obs: Snapshot) -> None:
    """The antimask of a body in the inventory covers the pixels it occupies."""

    backplane = Backplane(saturn_obs, inventory={})

    antimask = backplane.get_antimask(PLANET)

    assert antimask.shape == backplane.shape
    assert np.any(antimask)
    assert not np.all(antimask)


def test_a_body_outside_the_field_has_an_empty_antimask(saturn_obs: Snapshot) -> None:
    """A body the inventory places outside the field of view is masked entirely."""

    backplane = Backplane(saturn_obs, inventory={})

    assert backplane.get_antimask('JUPITER') is False


##########################################################################################
# Diagnostic and performance logging
##########################################################################################

def test_reusing_an_intercept_is_reported_as_a_diagnostic(
        saturn_obs: Snapshot, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Two surfaces that share one intercept solve it once, and the reuse is logged."""

    monkeypatch.setattr(Backplane, 'DIAGNOSTICS', True)
    backplane = Backplane(saturn_obs)
    backplane.ring_radius(RING)
    LOGGING.on()
    try:
        backplane.ring_radius(RING_BODY)
    finally:
        LOGGING.off()

    assert 'INTERCEPT REUSED' in capsys.readouterr().out


def test_solving_an_intercept_is_reported_as_a_performance_measurement(
        saturn_obs: Snapshot, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The time taken to solve an intercept is logged when performance logging is on."""

    monkeypatch.setattr(Backplane, 'PERFORMANCE', True)
    backplane = Backplane(saturn_obs)
    LOGGING.on()
    try:
        backplane.ring_radius(RING)
    finally:
        LOGGING.off()

    assert 'INTERCEPT' in capsys.readouterr().out

##########################################################################################
