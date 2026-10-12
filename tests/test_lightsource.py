##########################################################################################
# tests/test_lightsource.py
##########################################################################################

import pickle
from collections.abc import Iterator

import pytest

import numpy as np

from polymath import Pair, Scalar, Vector3
from oops.body        import Body
from oops.constants   import AU, RPD, RPS
from oops.event       import Event
from oops.lightsource import DiskSource, LightSource, StarSource, star_lookup
from oops.path        import Path


@pytest.fixture(autouse=True)
def _restore_body_registry() -> Iterator[None]:
    """Undo the Body registry entries that constructing a LightSource creates."""

    saved = set(Body.BODY_REGISTRY)
    yield
    for name in set(Body.BODY_REGISTRY) - saved:
        del Body.BODY_REGISTRY[name]


def _observer_event() -> Event:
    """An observation event one million km from the solar system barycenter."""

    return Event(0., Vector3((1.e6, 0., 0.)), Path.SSB, 'J2000')


def test_lightsource_from_a_direction_has_no_path() -> None:
    """A LightSource given a (ra,dec) pair is a fixed direction, not a path."""

    source = LightSource('TEST_FIXED', (30., 45.))

    assert source.as_path() is None
    assert source.source_is_moving is False

    (departure, arrival) = source.photon_to_event(_observer_event())

    assert departure is None
    assert arrival.neg_arr_j2000.shape == ()


def test_lightsource_from_a_path_solves_for_the_photon() -> None:
    """A path-based LightSource delegates to Path.photon_to_event."""

    source = LightSource('TEST_PATH', Path.SSB)

    assert source.as_path() is Path.SSB
    assert source.source_is_moving is True

    result = source.photon_to_event(_observer_event())

    assert len(result) == 2


def test_disksource_from_a_path_returns_a_grid_of_lines_of_sight() -> None:
    """A moving DiskSource spreads the arrival direction over its disk."""

    disk = DiskSource('TEST_DISK_PATH', LightSource('TEST_DISK_PATH_CENTER', Path.SSB),
                      695990., size=5)

    assert disk.shape == (5, 5)

    (departure, arrival) = disk.photon_to_event(_observer_event())

    assert departure is None
    assert isinstance(arrival, Event)
    assert arrival.neg_arr_ap_j2000.shape == (5, 5)

    # The observer is 1e6 km along +X, so the center of the disk lies along -X
    center = arrival.neg_arr_ap_j2000[2, 2]
    assert center.sep(-Vector3.XAXIS).vals == pytest.approx(0., abs=1.e-12)
    # On a 5x5 grid the outermost pixel centers lie at 0.8 of the radius
    edge = arrival.neg_arr_ap_j2000[2, 4]
    assert edge.sep(center).vals == pytest.approx(np.arctan(0.8 * 695990. / 1.e6),
                                                  rel=1.e-6)


def test_disksource_from_a_direction_spreads_over_the_disk() -> None:
    """A fixed-direction DiskSource spreads the arrival direction over its disk."""

    disk = DiskSource('TEST_DISK_FIXED', LightSource('TEST_DISK_FIXED_CENTER', (30., 45.)),
                      2.0, size=5)

    assert disk.shape == (5, 5)

    (departure, arrival) = disk.photon_to_event(_observer_event())

    assert departure is None
    assert isinstance(arrival, Event)
    assert arrival.neg_arr_j2000.shape == (5, 5)


def test_disksource_compress_keeps_the_pixels_inside_the_circle() -> None:
    """Compressing discards the corners outside the circle, not the disk itself."""

    # A 5x5 grid has four corners outside the unit circle, so 21 pixels remain.
    disk = DiskSource('TEST_DISK_SMALL', LightSource('TEST_DISK_SMALL_CENTER', Path.SSB),
                      695990., size=5, compress=True)

    assert disk.shape == (21,)


def test_disksource_weight_matches_the_lightsource_contract() -> None:
    """DiskSource publishes `weight` as a normalized Scalar, as LightSource does."""

    disk = DiskSource('TEST_DISK_WEIGHT', LightSource('TEST_DISK_WEIGHT_CENTER', Path.SSB),
                      695990., size=5)

    assert isinstance(disk.weight, Scalar)
    assert disk.weight.shape == (5, 5)
    assert float(disk.weight.sum()) == pytest.approx(1.)
    assert np.count_nonzero(disk.weight.vals) == 21


def test_lightsource_rejects_a_source_of_no_recognized_form() -> None:
    """A source that is neither a path, an (RA, dec) pair, nor a line of sight is refused.

    Each form is recognized explicitly, so an input resembling none of them raises rather
    than being forced into whichever interpretation happens to accept it first.
    """

    for source in [5, (1., 2., 3., 4.), None]:
        with pytest.raises(ValueError, match='must be a Path, a path ID'):
            LightSource('TEST_BAD_SOURCE', source)


def test_lightsource_accepts_every_documented_source_form() -> None:
    """A Path, a path ID, an (RA, dec) pair, and a line of sight are all accepted."""

    assert LightSource('TEST_FORM_PATH', Path.SSB).source_is_moving is True
    assert LightSource('TEST_FORM_ID', 'SSB').source_is_moving is True
    assert LightSource('TEST_FORM_RADEC', (30., 45.)).source_is_moving is False
    assert LightSource('TEST_FORM_LOS', Vector3((1., 0., 0.))).source_is_moving is False

def test_a_lightsource_name_must_be_a_string() -> None:
    """The name is what registers the source, so it has to be a name."""

    with pytest.raises(TypeError, match='name must be a string'):
        LightSource(42, Path.SSB)


def test_a_lightsource_cannot_take_the_name_of_a_body() -> None:
    """A registered Body already owns its name in the registry."""

    Body.reset_registry()
    Body.define_solar_system('2000-01-01', '2010-01-01')
    try:
        with pytest.raises(ValueError, match='also a Body name: SATURN'):
            LightSource('SATURN', Path.SSB)
    finally:
        Body.reset_registry()


def test_a_source_given_as_a_pair_is_a_right_ascension_and_declination() -> None:
    """A Pair is read as (RA, dec) in degrees, as a bare pair of values is."""

    from_pair = LightSource('TEST_PAIR_CLASS', Pair((30., 45.)))
    from_tuple = LightSource('TEST_PAIR_TUPLE', (30., 45.))

    assert from_pair.source == from_tuple.source


def test_a_source_given_as_a_vector3_is_a_line_of_sight() -> None:
    """A Vector3 is read as a direction, as a bare triple of values is."""

    from_vector = LightSource('TEST_VECTOR_CLASS', Vector3((1., 1., 1.)))
    from_tuple = LightSource('TEST_VECTOR_TUPLE', (1., 1., 1.))

    assert from_vector.source == from_tuple.source


def test_a_source_of_an_unusable_type_is_refused() -> None:
    """Something that is not even array-like is none of the documented forms."""

    with pytest.raises(ValueError, match='LightSource source must be'):
        LightSource('TEST_UNUSABLE', {'ra': 30., 'dec': 45.})


def test_weights_are_normalized_to_sum_to_one() -> None:
    """Explicit weights are scaled so the source integrates to unity."""

    source = LightSource('TEST_WEIGHTED', Vector3([(1., 0., 0.), (0., 1., 0.)]),
                         weight=Scalar([1., 3.]))

    assert float(source.weight.sum().vals) == pytest.approx(1., abs=1.e-12)
    assert source.weight.vals[1] == pytest.approx(0.75, abs=1.e-12)


def test_a_lightsource_is_registered_under_its_key() -> None:
    """The name keeps its capitalization; the registry holds the source by key."""

    source = LightSource('Test Lamp', (30., 45.))

    assert source.name == 'Test Lamp'
    assert source.key == 'TEST_LAMP'
    assert Body.lookup('test lamp') is source


def test_a_lightsource_survives_a_round_trip_through_pickle() -> None:
    """Unpickling rebuilds the source from its name, its location and its weights."""

    source = LightSource('TEST_PICKLED', (30., 45.))

    revived = pickle.loads(pickle.dumps(source))

    assert revived.name == source.name
    assert revived.source == source.source
    assert revived.weight == source.weight


def test_a_fixed_lightsource_keeps_the_derivatives_of_the_event() -> None:
    """With derivs, the derivatives of the observation event survive into the result."""

    source = LightSource('TEST_FIXED_DERIVS', (30., 45.))
    event = _observer_event()
    event.state.insert_deriv('pos', Vector3.IDENTITY)

    (_, arrival) = source.photon_to_event(event, derivs=True)

    assert 'pos' in arrival.state.derivs
    assert 'pos' not in source.photon_to_event(event)[1].state.derivs


def test_a_disk_source_must_name_a_single_direction() -> None:
    """A disk has one center, so a shaped source is refused."""

    shaped = LightSource('TEST_SHAPED_CENTER', Vector3([(1., 0., 0.), (0., 1., 0.)]))

    with pytest.raises(ValueError, match=r'source must have shape \(\)'):
        DiskSource('TEST_SHAPED_DISK', shaped, 0.01)

    assert 'TEST_SHAPED_DISK' not in Body.BODY_REGISTRY


@pytest.mark.parametrize('source', [Path.SSB, (30., 45.)], ids=['path', 'pair'])
def test_a_disk_source_must_wrap_a_lightsource(source: Path | tuple[float, float]
                                               ) -> None:
    """The center of the disk is another LightSource, not a raw path or direction."""

    with pytest.raises(TypeError, match='DiskSource source must be a LightSource'):
        DiskSource('TEST_UNWRAPPED_DISK', source, 0.01)

    assert 'TEST_UNWRAPPED_DISK' not in Body.BODY_REGISTRY


def test_disk_source_size_and_compress_are_keyword_only() -> None:
    """The grid size and compress flag must be passed by keyword."""

    center = LightSource('TEST_POSITIONAL_DISK_CENTER', Path.SSB)
    with pytest.raises(TypeError, match='positional argument'):
        DiskSource('TEST_POSITIONAL_DISK', center, 0.01, 5)

    assert 'TEST_POSITIONAL_DISK' not in Body.BODY_REGISTRY


def test_a_disk_source_keeps_the_lightsource_it_wraps() -> None:
    """The wrapped source stays registered under its own key and drives the disk."""

    center = LightSource('TEST_WRAPPED_CENTER', Path.SSB)

    disk = DiskSource('TEST_WRAPPING_DISK', center, 695990., size=5)

    assert disk.lightsource is center
    assert disk.source is center.source
    assert disk.source_is_moving is True
    assert Body.lookup('TEST_WRAPPED_CENTER') is center
    assert Body.lookup('TEST_WRAPPING_DISK') is disk


def test_a_disk_source_is_centered_on_a_moving_star() -> None:
    """The disk follows the wrapped star's proper motion and parallax."""

    star = StarSource('TEST_DISK_STAR', 90., 0., parallax=1.e-5,
                      proper_motion=(0., 1.e-14))
    disk = DiskSource('TEST_STAR_DISK', star, 2.0, size=5)
    event = _ssb_event(1.e9, (AU, 0., 0.))

    star_los = star.photon_to_event(event)[1].neg_arr_j2000
    disk_los = disk.photon_to_event(event)[1].neg_arr_j2000

    assert disk_los[2, 2].sep(star_los).vals == pytest.approx(0., abs=1.e-15)
    assert disk_los[2, 4].sep(star_los).vals == pytest.approx(np.arctan(0.8 * 2.0 * RPS),
                                                              rel=1.e-6)


@pytest.mark.parametrize('source', [(30., 45.), Path.SSB], ids=['fixed', 'moving'])
@pytest.mark.parametrize('compress', [False, True], ids=['full', 'compressed'])
def test_a_disk_source_survives_a_round_trip_through_pickle(source: tuple[float, float]
                                                            | Path,
                                                            compress: bool) -> None:
    """Unpickling rebuilds the disk with its radius, grid size and compression."""

    center = LightSource('TEST_DISK_PICKLED_CENTER', source)
    disk = DiskSource('TEST_DISK_PICKLED', center, 0.01, size=5, compress=compress)

    revived = pickle.loads(pickle.dumps(disk))

    assert isinstance(revived, DiskSource)
    assert revived.name == disk.name
    assert revived.source_is_moving == disk.source_is_moving
    assert revived.radius == disk.radius
    assert revived.shape == disk.shape
    assert revived.xy_grid == disk.xy_grid
    assert revived.weight == disk.weight
    assert revived.lightsource.name == center.name


def test_a_fixed_disk_source_keeps_the_derivatives_of_the_event() -> None:
    """With derivs, the derivatives of the observation event survive into the result."""

    source = DiskSource('TEST_FIXED_DISK', LightSource('TEST_FIXED_CENTER', (30., 45.)),
                        0.01)
    event = _observer_event()
    event.state.insert_deriv('pos', Vector3.IDENTITY)

    (_, arrival) = source.photon_to_event(event, derivs=True)

    assert 'pos' in arrival.state.derivs
    assert 'pos' not in source.photon_to_event(event)[1].state.derivs


def test_a_star_source_is_a_fixed_lightsource_with_star_attributes() -> None:
    """A StarSource points along its RA and dec and carries a magnitude and a type."""

    star = StarSource('Test Star', 30., 45., magnitude=2.5, stellar_type='G2V')
    fixed = LightSource('TEST_STAR_REFERENCE', (30., 45.))

    assert isinstance(star, LightSource)
    assert star.source_is_moving is False
    assert star.source == fixed.source
    assert star.as_path() is None
    assert star.magnitude == 2.5
    assert star.stellar_type == 'G2V'
    assert Body.lookup('TEST_STAR') is star


def test_star_source_inputs_after_dec_are_keyword_only() -> None:
    """Only the name, RA and dec may be given by position."""

    with pytest.raises(TypeError, match='positional argument'):
        StarSource('TEST_STAR_POSITIONAL', 30., 45., 2.5)

    assert 'TEST_STAR_POSITIONAL' not in Body.BODY_REGISTRY


def test_a_star_source_magnitude_and_type_default_to_none() -> None:
    """A star of unknown brightness and type is still a usable source."""

    star = StarSource('TEST_STAR_BARE', 0., 90.)

    assert star.magnitude is None
    assert star.stellar_type is None
    assert star.parallax == 0.
    assert star.proper_motion == Pair.ZEROS
    assert star.epoch == 0.


def test_a_star_source_solves_for_the_photon_like_a_fixed_lightsource() -> None:
    """The arrival direction is the star's line of sight and there is no departure."""

    star = StarSource('TEST_STAR_PHOTON', 30., 45.)

    (departure, arrival) = star.photon_to_event(_observer_event())

    assert departure is None
    assert arrival.neg_arr_j2000 == star.source


@pytest.mark.parametrize('ra,dec', [([30., 40.], 45.), (30., [45., 50.])],
                         ids=['ra', 'dec'])
def test_a_star_source_must_name_a_single_direction(ra: float | list[float],
                                                    dec: float | list[float]) -> None:
    """A star has one position, so an array of coordinates is refused."""

    with pytest.raises(ValueError, match='StarSource ra and dec must be single values'):
        StarSource('TEST_STAR_SHAPED', ra, dec)

    assert 'TEST_STAR_SHAPED' not in Body.BODY_REGISTRY

    assert 'TEST_STAR_SHAPED' not in Body.BODY_REGISTRY


def test_a_star_source_survives_a_round_trip_through_pickle() -> None:
    """Unpickling rebuilds the star with its direction, magnitude and type."""

    star = StarSource('TEST_STAR_PICKLED', 30., 45., magnitude=-1.4,
                      stellar_type='A1Vm', parallax=1.e-6, proper_motion=(1.e-15, 2.e-15),
                      epoch=100.)

    revived = pickle.loads(pickle.dumps(star))

    assert isinstance(revived, StarSource)
    assert revived.name == star.name
    assert revived.source == star.source
    assert revived.magnitude == -1.4
    assert revived.stellar_type == 'A1Vm'
    assert revived.parallax == 1.e-6
    assert revived.proper_motion == Pair((1.e-15, 2.e-15))
    assert revived.epoch == 100.


def test_star_lookup_returns_a_star_source_from_the_catalog() -> None:
    """A catalog star comes back with its catalog magnitude and type."""

    star = star_lookup('Sirius')

    assert isinstance(star, StarSource)
    assert star.key == 'SIRIUS'
    assert star.magnitude == -1.44
    assert star.stellar_type == 'A1Vm'
    assert star_lookup('sirius') is star


def _ra_dec_of(los: Vector3) -> tuple[float, float]:
    """The right ascension and declination in radians of a single line of sight."""

    (ra, dec, _) = los.to_ra_dec_length(recursive=False)
    return (float(ra.vals), float(dec.vals))


def _ssb_event(time: float | list[float], pos: tuple[float, float, float]
               = (0., 0., 0.)) -> Event:
    """An observation event at the given time and SSB-relative J2000 position."""

    return Event(Scalar(time), Vector3(pos), Path.SSB, 'J2000')


def test_a_star_source_moves_by_its_proper_motion() -> None:
    """The (RA, dec) advances linearly from the epoch at the proper-motion rate."""

    (dra, ddec) = (3.e-15, -2.e-15)
    star = StarSource('TEST_STAR_PM', 30., 45., proper_motion=(dra, ddec), epoch=1.e7)

    (_, arrival) = star.photon_to_event(_ssb_event(1.e9))

    (ra, dec) = _ra_dec_of(arrival.neg_arr_j2000)
    assert ra == pytest.approx(30. * RPD + dra * (1.e9 - 1.e7), abs=1.e-14)
    assert dec == pytest.approx(45. * RPD + ddec * (1.e9 - 1.e7), abs=1.e-14)


def test_a_star_source_applies_proper_motion_at_the_mean_event_time() -> None:
    """An event with many times gets the one direction at the mean of those times."""

    star = StarSource('TEST_STAR_MEAN', 30., 45., proper_motion=(1.e-14, 0.))
    single = star.photon_to_event(_ssb_event(1.e9))[1].neg_arr_j2000

    (_, arrival) = star.photon_to_event(_ssb_event([0., 1.e9, 3.e9]))

    assert arrival.neg_arr_j2000.shape == ()
    assert arrival.neg_arr_j2000.sep(single).vals > 1.e-6

    (_, arrival) = star.photon_to_event(_ssb_event([0., 1.e9, 2.e9]))

    assert arrival.neg_arr_j2000.sep(single).vals == pytest.approx(0., abs=1.e-15)


def test_a_star_source_is_displaced_by_parallax_from_the_observer() -> None:
    """An observer 1 AU off the line of sight sees the star shifted by the parallax."""

    parallax = 1.e-5
    star = StarSource('TEST_STAR_PARALLAX', 90., 0., parallax=parallax)

    (_, arrival) = star.photon_to_event(_ssb_event(0., (AU, 0., 0.)))

    assert arrival.neg_arr_j2000.sep(star.source).vals == pytest.approx(parallax,
                                                                         rel=1.e-9)
    assert arrival.neg_arr_j2000.vals[0] < 0.


def test_a_star_source_has_no_parallax_shift_at_the_barycenter() -> None:
    """The catalog direction is barycentric, so an observer at the SSB sees it as is."""

    star = StarSource('TEST_STAR_AT_SSB', 90., 0., parallax=1.e-5)

    (_, arrival) = star.photon_to_event(_ssb_event(0.))

    assert arrival.neg_arr_j2000.sep(star.source).vals == pytest.approx(0., abs=1.e-15)


def test_a_star_source_parallax_follows_each_observer_position() -> None:
    """Observers at different positions each see their own parallax shift."""

    star = StarSource('TEST_STAR_POSITIONS', 90., 0., parallax=1.e-5)
    event = Event(0., Vector3([(AU, 0., 0.), (0., 0., 0.)]), Path.SSB, 'J2000')

    (_, arrival) = star.photon_to_event(event)

    seps = arrival.neg_arr_j2000.sep(star.source).vals
    assert seps[0] == pytest.approx(1.e-5, rel=1.e-9)
    assert seps[1] == pytest.approx(0., abs=1.e-15)

##########################################################################################
