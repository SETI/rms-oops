##########################################################################################
# test/test_body.py
##########################################################################################

import pytest

from oops.body  import Body


@pytest.fixture(autouse=True)
def _solar_system():
    Body._undefine_solar_system()
    Body.define_solar_system('2000-01-01', '2020-01-01')

def test_body():
    assert Body.lookup('DAPHNIS').barycenter.name == 'SATURN'
    assert Body.lookup('PHOEBE').barycenter.name == 'SATURN_BARYCENTER'

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
                                  exclude=['IRREGULAR', 'BARYCENTER', 'IO'],
                                  recursive=True)
    assert len(planets) == 16  # 9 planets + 7 Jovian moons
##########################################################################################
