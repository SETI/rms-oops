##########################################################################################
# tests/test_lightsource.py
##########################################################################################

import pickle
from collections.abc import Iterator

import pytest

import numpy as np

from polymath import Pair, Scalar, Vector3
from oops.body        import Body
from oops.event       import Event
from oops.lightsource import DiskSource, LightSource
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

    disk = DiskSource('TEST_DISK_PATH', Path.SSB, 695990., size=5)

    assert disk.shape == (5, 5)

    (departure, arrival) = disk.photon_to_event(_observer_event())

    assert departure is None
    assert isinstance(arrival, Event)
    assert arrival.neg_arr_ap_j2000.shape == (5, 5)


def test_disksource_from_a_direction_spreads_over_the_disk() -> None:
    """A fixed-direction DiskSource spreads the arrival direction over its disk."""

    disk = DiskSource('TEST_DISK_FIXED', (30., 45.), 2.0, size=5)

    assert disk.shape == (5, 5)

    (departure, arrival) = disk.photon_to_event(_observer_event())

    assert departure is None
    assert isinstance(arrival, Event)
    assert arrival.neg_arr_j2000.shape == (5, 5)


def test_disksource_compress_keeps_the_pixels_inside_the_circle() -> None:
    """Compressing discards the corners outside the circle, not the disk itself."""

    # A 5x5 grid has four corners outside the unit circle, so 21 pixels remain.
    disk = DiskSource('TEST_DISK_SMALL', Path.SSB, 695990., size=5, compress=True)

    assert disk.shape == (21,)


def test_disksource_weight_matches_the_lightsource_contract() -> None:
    """DiskSource publishes `weight` as a normalized Scalar, as LightSource does."""

    disk = DiskSource('TEST_DISK_WEIGHT', Path.SSB, 695990., size=5)

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

    with pytest.raises(ValueError, match=r'source must have shape \(\)'):
        DiskSource('TEST_SHAPED_DISK', Vector3([(1., 0., 0.), (0., 1., 0.)]), 0.01)

    assert 'TEST_SHAPED_DISK' not in Body.BODY_REGISTRY


def test_a_fixed_disk_source_keeps_the_derivatives_of_the_event() -> None:
    """With derivs, the derivatives of the observation event survive into the result."""

    source = DiskSource('TEST_FIXED_DISK', (30., 45.), 0.01)
    event = _observer_event()
    event.state.insert_deriv('pos', Vector3.IDENTITY)

    (_, arrival) = source.photon_to_event(event, derivs=True)

    assert 'pos' in arrival.state.derivs
    assert 'pos' not in source.photon_to_event(event)[1].state.derivs

##########################################################################################
