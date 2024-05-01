################################################################################
# test/test_body.py
################################################################################

import unittest

from oops.body  import Body
from oops.frame import Frame
from oops.path  import Path


class Test_Body(unittest.TestCase):

    def runTest(self):

        # Imports are here to avoid conflicts
        Path.reset_registry()
        Frame.reset_registry()
        Body.reset_registry()

        Body.define_solar_system('2000-01-01', '2020-01-01')

        self.assertEqual(Body.lookup('DAPHNIS').barycenter.name,
                         'SATURN')
        self.assertEqual(Body.lookup('PHOEBE').barycenter.name,
                         'SATURN BARYCENTER')

        mars = Body.lookup('MARS')
        moons = mars.select_children(include_all=['SATELLITE'])
        self.assertEqual(len(moons), 2)     # Phobos, Deimos

        saturn = Body.lookup('SATURN')
        moons = saturn.select_children(include_all=['CLASSICAL', 'IRREGULAR'])
        self.assertEqual(len(moons), 1)     # Phoebe

        moons = saturn.select_children(exclude=['IRREGULAR','RING'], radius=160)
        self.assertEqual(len(moons), 8)     # Mimas-Iapetus

        rings = saturn.select_children(include_any=('RING'))
        self.assertEqual(len(rings), 8)     # A, B, C, AB, Main, all, plane,
                                            # system

        moons = saturn.select_children(include_all='SATELLITE',
                                       exclude=('IRREGULAR'), radius=1000)
        self.assertEqual(len(moons), 1)     # Titan only

        sun = Body.lookup('SUN')
        planets = sun.select_children(include_any=['PLANET'])
        self.assertEqual(len(planets), 9)

        sun = Body.lookup('SUN')
        planets = sun.select_children(include_any=['PLANET', 'EARTH'])
        self.assertEqual(len(planets), 9)

        sun = Body.lookup('SUN')
        planets = sun.select_children(include_any=['PLANET', 'EARTH'],
                                      recursive=True)
        self.assertEqual(len(planets), 10)  # 9 planets plus Earth's moon

        sun = Body.lookup('SUN')
        planets = sun.select_children(include_any=['PLANET', 'JUPITER'],
                                      exclude=['IRREGULAR', 'BARYCENTER', 'IO'],
                                      recursive=True)
        self.assertEqual(len(planets), 16)  # 9 planets + 7 Jovian moons

        Path.reset_registry()
        Frame.reset_registry()
        Body.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
