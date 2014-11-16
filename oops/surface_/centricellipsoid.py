################################################################################
# oops/surface_/centricellipsoid.py: CentricEllipsoid subclass of class Surface
################################################################################

import numpy as np
from polymath import *

from oops.surface_.surface   import Surface
from oops.surface_.ellipsoid import Ellipsoid
from oops.constants import *

class CentricEllipsoid(Surface):
    """CentricEllipsoid is a variant of Ellipsoid in which latitudes are
    planetocentric.
    """

    COORDINATE_TYPE = "spherical"
    IS_VIRTUAL = False

    def __init__(self, origin, frame, radii, exclusion=0.95):
        """Constructor for a CentricEllipsoid object.

        Input:
            origin      the Path object or ID defining the center of the
                        ellipsoid.
            frame       the Frame object or ID defining the coordinate frame in
                        which the ellipsoid is fixed, with the shortest radius
                        of the ellipsoid along the Z-axis and the longest radius
                        along the X-axis.
            radii       a tuple (a,b,c) containing the radii from longest to
                        shortest, in km.
            exclusion   the fraction of the polar radius within which
                        calculations of intercept_normal_to() are suppressed.
                        Values of less than 0.9 are not recommended because
                        the problem becomes numerically unstable.
        """

        self.ellipsoid = Ellipsoid(origin, frame, radii, exclusion)
        self.origin = self.ellipsoid.origin
        self.frame  = self.ellipsoid.frame

        self.squash_z_sq = self.ellipsoid.squash_z**2
        self.unsquash_z_sq = self.ellipsoid.unsquash_z**2

        self.squash_y_sq = self.ellipsoid.squash_y**2
        self.unsquash_y_sq = self.ellipsoid.unsquash_y**2

        self.radii = self.ellipsoid.radii

    def coords_from_vector3(self, pos, obs=None, axes=2, derivs=False):
        """Convert positions in the internal frame to surface coordinates.

        Input:
            pos         a Vector3 of positions at or near the surface.
            obs         a Vector3 of observer positions. Ignored for solid
                        surfaces but needed for virtual surfaces.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.

        Return:         coordinate values packaged as a tuple containing two or
                        three Scalars, one for each coordinate.
        """

        coords = self.ellipsoid.coords_from_vector3(pos, obs=obs, axes=axes,
                                                    derivs=derivs)
        centric_lon = self.ellipsoid.lon_to_centric(coords[0], derivs=derivs)
        centric_lat = self.ellipsoid.lat_to_centric(coords[1], coords[0],
                                                    derivs=derivs)
        return (centric_lon, centric_lat,) + coords[2:]

    def vector3_from_coords(self, coords, obs=None, derivs=False):
        """Convert surface coordinates to positions in the internal frame.

        Input:
            coords      a tuple of two or three Scalars defining the
                        coordinates.
            obs         position of the observer in the surface frame. Ignored
                        for solid surfaces but needed for virtual surfaces.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a Vector3 of intercept points defined by the
                        coordinates.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.
        """

        squashed_lon = self.ellipsoid.lon_from_centric(coords[0], derivs=derivs)
        squashed_lat = self.ellipsoid.lat_from_centric(coords[1], squashed_lon,
                                                       derivs=derivs)
        new_coords = (squashed_lon, squashed_lat,) + coords[2:]

        return self.ellipsoid.vector3_from_coords(new_coords, obs=obs,
                                                  derivs=derivs)

    def intercept(self, obs, los, derivs=False, guess=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       optional initial guess at the coefficient t such that:
                            intercept = obs + t * los

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
        """

        return self.ellipsoid.intercept(obs, los, derivs=derivs, guess=guess)

    def normal(self, pos, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.
        """

        return self.ellipsoid.normal(pos, derivs=derivs)

    def intercept_with_normal(self, normal, derivs=False, guess=None):
        """Intercept point where the normal vector parallels the given vector.

        Input:
            normal      a Vector3 of normal vectors.
            derivs      True to propagate derivatives in the normal vector into
                        the returned intercepts.
            guess       optional initial guess a coefficient array p such that:
                            pos = intercept + p * normal(intercept);
                        use guess=False for the converged value of p to be
                        returned even if an initial guess was not provided.

        Return:         a Vector3 of surface intercept points, in km. Where no
                        solution exists, the returned Vector3 will be masked.

                        If guess is not None, then it instead returns a tuple
                        (intercepts, p), where p is the converged solution such
                        that 
                            pos = intercept + p * normal(intercept).
        """

        return self.ellipsoid.intercept_with_normal(normal, derivs=derivs,
                                                    guess=guess)

    def intercept_normal_to(self, pos, derivs=False, guess=None):
        """Intercept point whose normal vector passes through a given position.

        Input:
            pos         a Vector3 of positions near the surface.
            derivs      True to propagate derivatives in pos into the returned
                        intercepts.
            guess       optional initial guess a coefficient array p such that:
                            intercept = pos + p * normal(intercept);
                        use guess=False for the converged value of p to be
                        returned even if an initial guess was not provided.

        Return:         a vector3 of surface intercept points, in km. Where no
                        solution exists, the returned vector will be masked.

                        If guess is not None, then it instead returns a tuple
                        (intercepts, p), where p is the converged solution such
                        that 
                            intercept = pos + p * normal(intercept).
        """

        return self.ellipsoid.intercept_normal_to(pos, derivs=derivs,
                                                  guess=guess)

    ############################################################################
    # Longitude and latitude conversions
    ############################################################################

    def lon_to_centric(self, lon, derivs=False):
        """Convert longitude in planetocentric ellipsoid coordinates to
        planetocentric.
        """

        return Scalar.as_scalar(lon, derivs)

    def lon_from_centric(self, lon, derivs=False):
        """Convert planetocentric longitude to planetocentric ellipsoid
        longitude.
        """

        return Scalar.as_scalar(lon, derivs)

    def lon_to_graphic(self, lon, derivs=False):
        """Convert longitude in planetocentric ellipsoid coordinates to
        planetographic.
        """

        lon = Scalar.as_scalar(lon, derivs)
        return (lon.sin() * self.unsquash_y_sq).arctan2(lon.cos())

    def lon_from_graphic(self, lon, derivs=False):
        """Convert planetographic longitude to planetocentric ellipsoid
        longitude.
        """

        lon = Scalar.as_scalar(lon, derivs)
        return (lon.sin() * self.squash_y_sq).arctan2(lon.cos())

    def lat_to_centric(self, lat, lon, derivs=False):
        """Convert latitude in planetocentric coordinates to planetocentric.
        """

        return Scalar.as_scalar(lat, derivs)

    def lat_from_centric(self, lat, lon, derivs=False):
        """Convert planetocentric latitude to planetocentric latitude.
        """

        return Scalar.as_scalar(lat, derivs)

    def lat_to_graphic(self, lat, lon, derivs=False):
        """Convert latitude in planetocentric coordinates to planetographic.
        """

        # This could be done more efficiently I'm sure
        squashed_lon = self.ellipsoid.lon_from_centric(lon, derivs=derivs)
        squashed_lat = self.ellipsoid.lat_from_centric(lat, squashed_lon,
                                                       derivs=derivs)

        return self.ellipsoid.lat_to_graphic(squashed_lat, squashed_lon,
                                             derivs=derivs)

    def lat_from_graphic(self, lat, lon, derivs=False):
        """Convert a planetographic latitude to planetocentric latitude.
        """

        squashed_lon = self.ellipsoid.lon_from_centric(lon, derivs=derivs)
        squashed_lat = self.ellipsoid.lat_from_graphic(lat, squashed_lon,
                                                       derivs=derivs)

        return self.ellipsoid.lat_to_centric(squashed_lat, squashed_lon,
                                             derivs=derivs)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_CentricEllipsoid(unittest.TestCase):

    def runTest(self):

        from oops.frame_.frame import Frame
        from oops.path_.path import Path

        REQ  = 60268.
        RMID = 54364.
        RPOL = 50000.
        planet = CentricEllipsoid("SSB", "J2000", (REQ, RMID, RPOL))

        # Coordinate/vector conversions
        NPTS = 10000
        pos = (2 * np.random.rand(NPTS,3) - 1.) * REQ   # range is -REQ to REQ

        (lon,lat,elev) = planet.coords_from_vector3(pos, axes=3)
        test = planet.vector3_from_coords((lon,lat,elev))
        self.assertTrue(abs(test - pos).max() < 1.e-8)

        # Make sure longitudes convert to planetocentric and back
        test_lon = np.arctan2(pos[...,1], pos[...,0])
        centric_lon = planet.lon_to_centric(lon)
        diffs = (centric_lon - test_lon + HALFPI) % PI - HALFPI
        self.assertTrue(abs(diffs).max() < 1.e-8)

        test_lon2 = planet.lon_from_centric(centric_lon)
        diffs = (test_lon2 - lon + HALFPI) % PI - HALFPI
        self.assertTrue(abs(diffs).max() < 1.e-8)

        # Make sure latitudes convert to planetocentric and back
        test_lat = np.arcsin(pos[...,2] / np.sqrt(np.sum(pos**2, axis=-1)))
        centric_lat = planet.lat_to_centric(lat,lon)
        self.assertTrue(abs(centric_lat - test_lat).max() < 1.e-8)

        test_lat2 = planet.lat_from_centric(centric_lat, lon)
        self.assertTrue(abs(test_lat2 - lat).max() < 1.e-8)

        # Make sure longitudes convert to planetographic and back
        normals = planet.normal(pos)
        test_lon = np.arctan2(normals.vals[...,1], normals.vals[...,0])
        graphic_lon = planet.lon_to_graphic(lon)
        diffs = (graphic_lon - test_lon + HALFPI) % PI - HALFPI
        self.assertTrue(abs(diffs).max() < 1.e-8)

        test_lon2 = planet.lon_from_centric(centric_lon)
        diffs = (test_lon2 - lon + HALFPI) % PI - HALFPI
        self.assertTrue(abs(diffs).max() < 1.e-8)

        # Make sure latitudes convert to planetographic and back
        test_lat = np.arcsin(normals.vals[...,2] / normals.norm().vals)
        graphic_lat = planet.lat_to_graphic(lat,lon)
        self.assertTrue(abs(graphic_lat - test_lat).max() < 1.e-8)

        test_lat2 = planet.lat_from_graphic(graphic_lat, lon)
        self.assertTrue(abs(test_lat2 - lat).max() < 1.e-8)

        # Ellipsoid intercepts & normals
        obs = REQ * (np.random.rand(NPTS,3) + 1.)       # range is REQ to 2*REQ
        los = -np.random.rand(NPTS,3)                   # range is -1 to 0

        (pts, t) = planet.intercept(obs, los)
        test = t * Vector3(los) + Vector3(obs)
        self.assertTrue(abs(test - pts).max() < 1.e-9)

        self.assertTrue(np.all(t.mask == pts.mask))
        self.assertTrue(np.all(pts.mask[t.vals < 0.]))

        normals = planet.normal(pts)

        pts.vals[...,1] *= REQ/RMID
        pts.vals[...,2] *= REQ/RPOL
        self.assertTrue(abs(pts.norm() - REQ).max() < 1.e-8)

        normals.vals[...,1] *= RMID/REQ
        normals.vals[...,2] *= RPOL/REQ
        self.assertTrue(abs(normals.unit() - pts.unit()).max() < 1.e-14)

        # Intercept derivatives

        # Lines of sight with grazing incidence can have large numerical errors,
        # but this is not to be considered an error in the analytic calculation.
        # As a unit test, we ignore the largest 3% of the errors, but require
        # that the rest of the errors be very small.

        obs = REQ * (np.random.rand(NPTS,3) + 1.)       # range is REQ to 2*REQ
        los = -np.random.rand(NPTS,3)                   # range is -1 to 0

        obs = Vector3(obs)
        los = Vector3(los).unit()
        obs.insert_deriv('obs', Vector3.IDENTITY)
        los.insert_deriv('los', Vector3.IDENTITY)

        eps = 1.
        frac = 0.97     # Ignore errors above this cutoff
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (cept,t) = planet.intercept(obs, los, derivs=True)
            (cept1,t1) = planet.intercept(obs + dobs[i], los, derivs=False)
            (cept2,t2) = planet.intercept(obs - dobs[i], los, derivs=False)

            dcept_dobs = (cept1 - cept2) / (2*eps)
            ref = Vector3(cept.d_dobs.vals[...,i], cept.d_dobs.mask)

            errors = abs(dcept_dobs - ref) / abs(ref)
            sorted = np.sort(errors.vals[np.logical_not(errors.mask)])
                        # mask=True where the line of sight missed the surface
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

            dt_dobs = (t1 - t2) / (2*eps)
            ref = t.d_dobs.vals[...,i]

            errors = abs(dt_dobs/ref - 1)
            sorted = np.sort(errors.vals[np.logical_not(errors.mask)])
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

        eps = 1.e-6
        frac = 0.97
        dlos = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (cept,t) = planet.intercept(obs, los, derivs=True)
            (cept1,t1) = planet.intercept(obs, los + dlos[i], derivs=False)
            (cept2,t2) = planet.intercept(obs, los - dlos[i], derivs=False)

            dcept_dlos = (cept1 - cept2) / (2*eps)
            ref = Vector3(cept.d_dlos.vals[...,i], cept.d_dlos.mask)

            errors = abs(dcept_dlos - ref) / abs(ref)
            sorted = np.sort(errors.vals[np.logical_not(errors.mask)])
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

            dt_dlos = (t1 - t2) / (2*eps)
            ref = t.d_dlos.vals[...,i]

            errors = abs(dt_dlos/ref - 1)
            sorted = np.sort(errors.vals[np.logical_not(errors.mask)])
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

        # Test normal()
        cept = Vector3(np.random.random((100,3))).unit().element_mul(planet.radii)
        perp = planet.normal(cept)
        test1 = cept.element_mul(planet.ellipsoid.unsquash).unit()
        test2 = perp.element_mul(planet.ellipsoid.squash).unit()

        self.assertTrue(abs(test1 - test2).max() < 1.e-12)

        eps = 1.e-7
        (lon,lat) = planet.coords_from_vector3(cept, axes=2)
        cept1 = planet.vector3_from_coords((lon+eps,lat,0.))
        cept2 = planet.vector3_from_coords((lon-eps,lat,0.))

        self.assertTrue(abs((cept2 - cept1).sep(perp) - HALFPI).max() < 1.e-8)

        (lon,lat) = planet.coords_from_vector3(cept, axes=2)
        cept1 = planet.vector3_from_coords((lon,lat+eps,0.))
        cept2 = planet.vector3_from_coords((lon,lat-eps,0.))

        self.assertTrue(abs((cept2 - cept1).sep(perp) - HALFPI).max() < 1.e-8)

        # Test intercept_with_normal()
        vector = Vector3(np.random.random((100,3)))
        cept = planet.intercept_with_normal(vector)
        sep = vector.sep(planet.normal(cept))
        self.assertTrue(sep.max() < 1.e-14)

        # Test intercept_normal_to()
        pos = Vector3(np.random.random((100,3)) * 4.*REQ + REQ)
        cept = planet.intercept_normal_to(pos)
        sep = (pos - cept).sep(planet.normal(cept))
        self.assertTrue(sep.max() < 3.e-12)
        self.assertTrue(abs(cept.element_mul(planet.ellipsoid.unsquash).norm() -
                        planet.ellipsoid.req).max() < 1.e-6)

        # Test normal() derivative
        cept = Vector3(np.random.random((100,3))).unit().element_mul(planet.radii)
        cept.insert_deriv('pos', Vector3.IDENTITY, override=True)
        perp = planet.normal(cept, derivs=True)
        eps = 1.e-5
        dpos = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            perp1 = planet.normal(cept + dpos[i])
            dperp_dpos = (perp1 - perp) / eps

            ref = Vector3(perp.d_dpos.vals[...,i,:], perp.d_dpos.mask)
            self.assertTrue(abs(dperp_dpos - ref).max() < 1.e-4)

        # Test intercept_normal_to() derivative
        pos = Vector3(np.random.random((3,3)) * 4.*REQ + REQ)
        pos.insert_deriv('pos', Vector3.IDENTITY, override=True)
        (cept,t) = planet.intercept_normal_to(pos, derivs=True, guess=False)
        self.assertTrue(abs(cept.element_mul(planet.ellipsoid.unsquash).norm() -
                        planet.ellipsoid.req).max() < 1.e-6)

        eps = 1.
        dpos = ((eps,0,0), (0,eps,0), (0,0,eps))
        perp = planet.normal(cept)
        for i in range(3):
            (cept1,t1) = planet.intercept_normal_to(pos + dpos[i], derivs=False,
                                                    guess=t)
            (cept2,t2) = planet.intercept_normal_to(pos - dpos[i], derivs=False,
                                                    guess=t)
            dcept_dpos = (cept1 - cept2) / (2*eps)
            self.assertTrue(abs(dcept_dpos.sep(perp) - HALFPI).max() < 1.e-5)

            ref = Vector3(cept.d_dpos.vals[...,i], cept.d_dpos.mask)
            self.assertTrue(abs(dcept_dpos - ref).max() < 1.e-5)

            dt_dpos = (t1 - t2) / (2*eps)
            ref = t.d_dpos.vals[...,i]
            self.assertTrue(abs(dt_dpos/ref - 1).max() < 1.e-5)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
