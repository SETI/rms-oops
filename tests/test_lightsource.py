##########################################################################################
# tests/test_lightsource.py
##########################################################################################

import pytest

import numpy as np

from polymath import Scalar, Vector3
import oops
from oops.body        import Body
from oops.event       import Event
from oops.lightsource import DiskSource, LightSource
from oops.path        import Path


@pytest.fixture(autouse=True)
def _restore_body_registry() -> None:
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

##########################################################################################
