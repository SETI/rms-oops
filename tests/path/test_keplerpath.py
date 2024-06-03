################################################################################
# tests/path/test_keplerpath.py
################################################################################

import numpy as np
import unittest

from oops.body    import Body
from oops.frame   import Frame
from oops.gravity import Gravity
from oops.path    import Path, KeplerPath

def _xyz_planet_derivative_test(kep, t, delta=1.e-7):
    """Error in position change based on numerical vs. analytic derivatives.
    """

    # Save the position and its derivatives
    (xyz, _d_xyz_dt) = kep.xyz_planet(t, partials=True)
    d_xyz_d_elem = xyz.d_delements.vals
    pos_norm = xyz.norm().vals

    # Create new Kepler objects for tweaking the parameters
    khi = kep.copy()
    klo = kep.copy()

    params = kep.get_params()

    # Loop through parameters...
    errors = np.zeros(np.shape(t) + (3,kep.nparams))
    for e in range(kep.nparams):

        # Tweak one parameter
        hi = params.copy()
        lo = params.copy()

        if params[e] == 0.:
            hi[e] += delta
            lo[e] -= delta
        else:
            hi[e] *= 1. + delta
            lo[e] *= 1. - delta

        denom = hi[e] - lo[e]

        khi.set_params(hi)
        klo.set_params(lo)

        # Compare the change with that derived from the partial derivative
        xyz_hi = khi.xyz_planet(t, partials=False)[0].vals
        xyz_lo = klo.xyz_planet(t, partials=False)[0].vals
        hi_lo_diff = xyz_hi - xyz_lo

        errors[...,:,e] = ((d_xyz_d_elem[...,:,e] * denom - hi_lo_diff) /
                           pos_norm[...,np.newaxis])

    return errors

#===============================================================================
def _pos_derivative_test(kep, t, delta=1.e-5):
    """Calculates numerical derivatives of (x,y,z) in the observer/J2000 frame
    relative to the orbital elements, at time(s) t. It returns a tuple of
    (numerical derivatives, analytic derivatives, relative errors). Used for
    debugging.
    """

    # Save the position and its derivatives
    event = kep.event_at_time(t, partials=True)
    d_xyz_d_elem = event.pos.d_delements.vals
    pos_norm = event.pos.norm().vals

    # Create new Kepler objects for tweaking the parameters
    khi = kep.copy()
    klo = kep.copy()

    params = kep.get_params()

    # Loop through parameters...
    errors = np.zeros(np.shape(t) + (3,kep.nparams))
    for e in range(kep.nparams):

        # Tweak one parameter
        hi = params.copy()
        lo = params.copy()

        if params[e] == 0.:
            hi[e] += delta
            lo[e] -= delta
        else:
            hi[e] *= 1. + delta
            lo[e] *= 1. - delta

        denom = hi[e] - lo[e]

        khi.set_params(hi)
        klo.set_params(lo)

        # Compare the change with that derived from the partial derivative
        xyz_hi = khi.event_at_time(t, partials=False).pos.vals
        xyz_lo = klo.event_at_time(t, partials=False).pos.vals
        hi_lo_diff = xyz_hi - xyz_lo

        errors[...,:,e] = ((d_xyz_d_elem[...,:,e] * denom - hi_lo_diff) /
                           pos_norm[...,np.newaxis])

    return errors

#===============================================================================
class Test_KeplerPath(unittest.TestCase):

    def setUp(self):
        Body.reset_registry()
        Body._undefine_solar_system()
        Body.define_solar_system("2000-01-01", "2010-01-01")

    def tearDown(self):
        pass

    def runTest(self):
        from oops.body import Body

        # SEMIM = 0    elements[SEMIM] = semimajor axis (km)
        # MEAN0 = 1    elements[MEAN0] = mean longitude at epoch (radians)
        # DMEAN = 2    elements[DMEAN] = mean motion (radians/s)
        # ECCEN = 3    elements[ECCEN] = eccentricity
        # PERI0 = 4    elements[PERI0] = pericenter at epoch (radians)
        # DPERI = 5    elements[DPERI] = pericenter precession rate (radians/s)
        # INCLI = 6    elements[INCLI] = inclination (radians)
        # NODE0 = 7    elements[NODE0] = longitude of ascending node at epoch
        # DNODE = 8    elements[DNODE] = nodal regression rate (radians/s)

        a = 140000.

        saturn = Gravity.lookup('SATURN')
        dmean_dt = saturn.n(a)
        dperi_dt = saturn.dperi_dt(a)
        dnode_dt = saturn.dnode_dt(a)

        TIMESTEPS = 100
        time = 3600. * np.arange(TIMESTEPS)

        kep = KeplerPath(Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt, 0.2, 3., dperi_dt, 0.1, 5., dnode_dt),
                       Path.as_path("EARTH"))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-8)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-8)

        ####################

        kep = KeplerPath(Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        dmean_dt * 0.10, 2., dmean_dt / 100.,
                        dperi_dt * 0.08, 4., dmean_dt / 50.,
                        dnode_dt * 0.12, 6., dmean_dt / 200.),
                       Path.as_path("EARTH"), wobbles=('mean', 'peri', 'node'))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-8)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-8)

        ####################

        kep = KeplerPath(Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        a * 0.10, 2., dmean_dt / 100.),
                       Path.as_path("EARTH"), wobbles=('a',))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-7)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-5)

        ####################

        kep = KeplerPath(Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        0.1, 4., dmean_dt / 50.),
                       Path.as_path("EARTH"), wobbles=('e',))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 3.e-7)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 3.e-5)

        ####################

        kep = KeplerPath(Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        0.15, 2., dmean_dt / 150.),
                       Path.as_path("EARTH"), wobbles=('i',))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-7)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-5)

        ####################

        kep = KeplerPath(Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        1.e-4, 3., dperi_dt/100.),
                       Path.as_path("EARTH"), wobbles=('e2d',))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-5)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-4)

        ####################

        kep = KeplerPath(Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        1.e-4, 2., dnode_dt/150.),
                       Path.as_path("EARTH"), wobbles=('i2d',))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-6)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-5)

        ####################

        kep = KeplerPath(Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        1.e-4, 2., dperi_dt/150.,
                        2.e-4, 3., dnode_dt/200.,
                        a * 1.e-3, 4., dmean_dt/150.),
                       Path.as_path("EARTH"), wobbles=('i2d','e2d','a'))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-5)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-4)

        Frame.reset_registry()
        Path.reset_registry()
        Body.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
