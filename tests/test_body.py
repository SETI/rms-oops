##########################################################################################
# test/test_body.py
##########################################################################################

import pickle

import pytest

from oops.body  import Body
from oops.frame import Frame
from oops.path  import Path


@pytest.fixture(autouse=True)
def _solar_system():
    Body._undefine_solar_system()
    Body.define_solar_system('2000-01-01', '2020-01-01')

def test_body():
    assert Body.lookup('DAPHNIS').barycenter.key == 'SATURN'
    assert Body.lookup('PHOEBE').barycenter.key == 'SATURN_BARYCENTER'

    mars = Body.lookup('MARS')
    moons = mars.select_children(include_all=['SATELLITE'])
    assert len(moons) == 2     # Phobos, Deimos

    saturn = Body.lookup('SATURN')
    moons = saturn.select_children(include_all=['CLASSICAL', 'IRREGULAR'])
    assert len(moons) == 1     # Phoebe

    moons = saturn.select_children(exclude=['IRREGULAR', 'RING'], radius=160)
    assert len(moons) == 8     # Mimas-Iapetus

    rings = saturn.select_children(include_any=('RING'))
    assert len(rings) == 8     # A, B, C, AB, Main, all, plane, system

    moons = saturn.select_children(include_all='SATELLITE',
                                   exclude=('IRREGULAR'), radius=1000)
    assert len(moons) == 1     # Titan only

    sun = Body.lookup('SUN')
    planets = sun.select_children(include_any=['PLANET'])
    assert len(planets) == 9

    sun = Body.lookup('SUN')
    planets = sun.select_children(include_any=['PLANET', 'EARTH'])
    assert len(planets) == 9

    sun = Body.lookup('SUN')
    planets = sun.select_children(include_any=['PLANET', 'EARTH'],
                                  recursive=True)
    assert len(planets) == 10  # 9 planets plus Earth's moon

    sun = Body.lookup('SUN')
    planets = sun.select_children(include_any=['PLANET', 'JUPITER'],
                                  exclude=['IRREGULAR', 'BARYCENTER', 'IO', 'RING'],
                                  recursive=True)
    assert len(planets) == 16  # 9 planets + 7 Jovian moons


def test_has_gravity_filters_on_the_gravity_field() -> None:
    """Bodies with a gravity field pass has_gravity; the others pass has_no_gravity."""

    mimas = Body.lookup('MIMAS')
    janus = Body.lookup('JANUS')
    assert mimas.gravity is not None
    assert janus.gravity is None

    assert Body.has_gravity([mimas, janus, mimas]) == [mimas]
    assert Body.has_no_gravity([mimas, janus, janus]) == [janus]


def test_a_body_name_is_kept_and_its_key_is_normalized() -> None:
    """The name keeps its capitalization; the key is upper case with underscores."""

    body = Body('Test Rock', Path.SSB, Frame.J2000, parent='SUN')

    assert body.name == 'Test Rock'
    assert body.key == 'TEST_ROCK'
    assert body.keywords == ['TEST_ROCK', 'Test Rock']
    assert Body.BODY_REGISTRY['TEST_ROCK'] is body
    assert 'TEST_ROCK' in Body.lookup('SUN').child_keys


@pytest.mark.parametrize('lookup_name', ['Test Rock', 'test_rock', 'TEST ROCK'])
def test_lookup_accepts_a_name_or_key_in_any_case(lookup_name: str) -> None:
    """Lookup and exists normalize case and spaces before consulting the registry."""

    body = Body('Test Rock', Path.SSB, Frame.J2000)

    assert Body.lookup(lookup_name) is body
    assert Body.exists(lookup_name)


def test_as_body_key_normalizes_a_body_or_a_string() -> None:
    """A body yields its key; a string is converted to key form."""

    assert Body.as_body_key(Body.lookup('SATURN')) == 'SATURN'
    assert Body.as_body_key('Saturn Main Rings') == 'SATURN_MAIN_RINGS'


def test_name_in_matches_names_or_keys() -> None:
    """name_in and name_not_in compare keys, so case and spaces do not matter."""

    mimas = Body.lookup('MIMAS')
    janus = Body.lookup('JANUS')

    assert Body.name_in([mimas, janus], 'mimas') == [mimas]
    assert Body.name_not_in([mimas, janus], ['Mimas']) == [janus]


def test_a_ring_carries_its_planet_key_as_a_keyword() -> None:
    """Rings, like satellites, are tagged with the key of their planet."""

    ring = Body.lookup('SATURN_MAIN_RINGS')

    assert 'SATURN' in ring.keywords
    assert all(isinstance(keyword, str) for keyword in ring.keywords)


def test_define_small_body_keeps_the_given_name() -> None:
    """A small body is registered under the key of the name it is given."""

    Body.define_small_body(3, name='Test EMB')
    body = Body.lookup('TEST_EMB')

    assert body.name == 'Test EMB'
    assert body.parent is Body.lookup('SUN')


def test_define_small_body_without_a_name_uses_the_path_id() -> None:
    """With no name, the SPICE path ID names the body; an existing one is kept."""

    barycenter = Body.lookup('MARS_BARYCENTER')

    Body.define_small_body(4)

    assert Body.lookup('MARS_BARYCENTER') is barycenter


def test_a_body_survives_a_round_trip_through_pickle() -> None:
    """Unpickling re-registers the body under its key and relinks its parent."""

    body = Body('Test Rock', Path.SSB, Frame.J2000, parent='SATURN')
    state = pickle.dumps(body)
    del Body.BODY_REGISTRY['TEST_ROCK']
    Body.lookup('SATURN').child_keys.remove('TEST_ROCK')

    revived = pickle.loads(state)

    assert Body.lookup('Test Rock') is revived
    assert revived.parent is Body.lookup('SATURN')
    assert 'TEST_ROCK' in Body.lookup('SATURN').child_keys

##########################################################################################
