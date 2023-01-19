################################################################################
# oops/surface/centricspheroid.py: CentricSpheroid subclass of class Surface.
################################################################################

import numpy as np
from polymath              import Scalar, Vector3
from oops.frame            import Frame
from oops.path             import Path
from oops.surface          import Surface
from oops.surface.spheroid import Spheroid

class CentricSpheroid(Surface):
    """A variant of Spheroid in which latitudes are planetocentric."""

    COORDINATE_TYPE = 'spherical'
    IS_VIRTUAL = False

    #===========================================================================
    def __init__(self, origin, frame, radii, exclusion=0.95):
        """Constructor for a CentricSpheroid surface.

        Input:
            origin      the Path object or ID defining the center of the
                        spheroid.
            frame       the Frame object or ID defining the coordinate frame in
                        which the spheroid is fixed, with the short radius
                        along the Z-axis.
            radii       a tuple (a,c) or (a,a,c), defining the long and short
                        radii of the spheroid.
            exclusion   the fraction of the polar radius within which
                        calculations of intercept_normal_to() are suppressed.
                        Values of less than 0.9 are not recommended because
                        the problem becomes numerically unstable.
        """

        self.spheroid = Spheroid(origin, frame, radii, exclusion)
        self.origin = self.spheroid.origin
        self.frame  = self.spheroid.frame
        self.req_sq = self.spheroid.req_sq
        self.unsquash = self.spheroid.unsquash

        self.squash_sq = self.spheroid.squash_z**2
        self.unsquash_sq = self.spheroid.unsquash_z**2
        self.radii = self.spheroid.radii

        self.exclusion = float(exclusion)

        self.unmasked = self

        # Unique key for intercept calculations
        self.intercept_key = self.spheroid.intercept_key

    def __getstate__(self):
        return (Path.as_primary_path(self.origin),
                Frame.as_primary_frame(self.frame),
                tuple(self.radii), self.exclusion)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                  derivs=False):
        """Convert positions in the internal frame to surface coordinates.

        Input:
            pos         a Vector3 of positions at or near the surface.
            obs         a Vector3 of observer positions. Ignored for solid
                        surfaces but needed for virtual surfaces.
            time        a Scalar time at which to evaluate the surface; ignored.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.

        Return:         coordinate values packaged as a tuple containing two or
                        three Scalars, one for each coordinate.
        """

        coords = self.spheroid.coords_from_vector3(pos, obs=obs, axes=axes,
                                                   derivs=derivs)

        new_lat = self.spheroid.lat_to_centric(coords[1], derivs=derivs)
        return coords[:1] + (new_lat,) + coords[2:]

    #===========================================================================
    def vector3_from_coords(self, coords, obs=None, time=None, derivs=False):
        """Convert surface coordinates to positions in the internal frame.

        Input:
            coords      a tuple of two or three Scalars defining the
                        coordinates.
            obs         position of the observer in the surface frame. Ignored
                        for solid surfaces but needed for virtual surfaces.
            time        a Scalar time at which to evaluate the surface; ignored.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a Vector3 of intercept points defined by the
                        coordinates.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.
        """

        new_lat = self.spheroid.lat_from_centric(coords[1], derivs=derivs)
        new_coords = coords[:1] + (new_lat,) + coords[2:]

        return self.spheroid.vector3_from_coords(new_coords, obs=obs,
                                                 derivs=derivs)

    #===========================================================================
    def intercept(self, obs, los, time=None, derivs=False, guess=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            time        a Scalar time at which to evaluate the surface; ignored.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       optional initial guess at the coefficient t such that:
                            intercept = obs + t * los

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
        """

        return self.spheroid.intercept(obs, los, derivs=derivs, guess=guess)

    #===========================================================================
    def normal(self, pos, time=None, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            time        a Scalar time at which to evaluate the surface; ignored.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.
        """

        return self.spheroid.normal(pos, derivs=derivs)

    #===========================================================================
    def intercept_with_normal(self, normal, time=None, derivs=False,
                                    guess=None):
        """Intercept point where the normal vector parallels the given vector.

        Input:
            normal      a Vector3 of normal vectors.
            time        a Scalar time at which to evaluate the surface; ignored.
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

        return self.spheroid.intercept_with_normal(normal, derivs=derivs,
                                                   guess=guess)

    #===========================================================================
    def intercept_normal_to(self, pos, time=None, derivs=False, guess=None):
        """Intercept point whose normal vector passes through a given position.

        Input:
            pos         a Vector3 of positions near the surface.
            time        a Scalar time at which to evaluate the surface; ignored.
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

        return self.spheroid.intercept_normal_to(pos, derivs=derivs,
                                                 guess=guess)

    ############################################################################
    # Longitude conversions
    ############################################################################

    def lon_to_centric(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetocentric.

        This is a null operation for spheroids. The method is provided for
        compatibility with Ellipsoids.

        Input:
            lon         planetocentric longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric longitude.
        """

        return Scalar.as_scalar(lon, derivs)

    #===========================================================================
    def lon_from_centric(self, lon, derivs=False):
        """Convert planetocentric longitude to internal coordinates.

        This is a null operation for spheroids. The method is provided for
        compatibility with Ellipsoids.

        Input:
            lon         planetocentric longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric longitude.
        """

        return Scalar.as_scalar(lon, derivs)

    #===========================================================================
    def lon_to_graphic(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetographic.

        This is a null operation for spheroids. The method is provided for
        compatibility with Ellipsoids.

        Input:
            lon         planetographic longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetographic longitude.
        """

        return Scalar.as_scalar(lon, derivs)

    #===========================================================================
    def lon_from_graphic(self, lon, derivs=False):
        """Convert planetographic longitude to internal coordinates.

        This is a null operation for spheroids. The method is provided for
        compatibility with Ellipsoids.

        Input:
            lon         planetographic longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric longitude.
        """

        return Scalar.as_scalar(lon, derivs)

    ############################################################################
    # Latitude conversions
    ############################################################################

    def lat_to_centric(self, lat, lon=None, derivs=False):
        """Convert latitude in internal coordinates to planetocentric.

        Input:
            lat         planetocentric latitide, radians.
            lon         ignored, included for compatibility with Ellipsoids.
            derivs      True to include derivatives in returned result.

        Return          planetocentric latitude.
        """

        return Scalar.as_scalar(lat, derivs)

    #===========================================================================
    def lat_from_centric(self, lat, lon=None, derivs=False):
        """Convert planetocentric latitude to internal coordinates.

        Input:
            lat         planetocentric latitide, radians.
            lon         ignored, included for compatibility with Ellipsoids.
            derivs      True to include derivatives in returned result.

        Return          planetocentric latitude.
        """

        return Scalar.as_scalar(lat, derivs)

    #===========================================================================
    def lat_to_graphic(self, lat, lon=None, derivs=False):
        """Convert latitude in internal coordinates to planetographic.

        Input:
            lat         planetocentric latitide, radians.
            lon         ignored, included for compatibility with Ellipsoids.
            derivs      True to include derivatives in returned result.

        Return          planetographic latitude.
        """

        lat = Scalar.as_scalar(lat, derivs)
        return (lat.tan() * self.unsquash_sq).arctan()

    #===========================================================================
    def lat_from_graphic(self, lat, derivs=False):
        """Convert a planetographic latitude to internal coordinates.

        Input:
            lat         planetographic latitide, radians.
            lon         ignored, included for compatibility with Ellipsoids.
            derivs      True to include derivatives in returned result.

        Return          planetocentric latitude.
        """

        lat = Scalar.as_scalar(lat, derivs)
        return (lat.tan() * self.squash_sq).arctan()

################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops.constants import HALFPI

class Test_CentricSpheroid(unittest.TestCase):

    def runTest(self):

        np.random.seed(6738)

        REQ  = 60268.
        #RPOL = 54364.
        RPOL = 50000.
        planet = CentricSpheroid("SSB", "J2000", (REQ, RPOL))

        # Coordinate/vector conversions
        NPTS = 10000
        pos = (2 * np.random.rand(NPTS,3) - 1.) * REQ   # range is -REQ to REQ

        (lon,lat,elev) = planet.coords_from_vector3(pos, axes=3)
        test = planet.vector3_from_coords((lon,lat,elev))
        self.assertTrue(abs(test - pos).max() < 3.e-8)

        # Spheroid intercepts & normals
        obs = REQ * (np.random.rand(NPTS,3) + 1.)       # range is REQ to 2*REQ
        los = -np.random.rand(NPTS,3)                   # range is -1 to 0

        (pts, t) = planet.intercept(obs, los)
        test = t * Vector3(los) + Vector3(obs)
        self.assertTrue(abs(test - pts).max() < 1.e-9)

        self.assertTrue(np.all(t.mask == pts.mask))
        self.assertTrue(np.all(pts.mask[t.vals < 0.]))

        normals = planet.normal(pts)

        pts.vals[...,2] *= REQ/RPOL
        self.assertTrue(abs(pts.norm() - REQ).max() < 1.e-8)

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
            sorted = np.sort(errors.vals[errors.antimask])
                        # mask=True where the line of sight missed the surface
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

            dt_dobs = (t1 - t2) / (2*eps)
            ref = t.d_dobs.vals[...,i]

            errors = abs(dt_dobs/ref - 1)
            sorted = np.sort(errors.vals[errors.antimask])
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
            sorted = np.sort(errors.vals[errors.antimask])
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

            dt_dlos = (t1 - t2) / (2*eps)
            ref = t.d_dlos.vals[...,i]

            errors = abs(dt_dlos/ref - 1)
            sorted = np.sort(errors.vals[errors.antimask])
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

        # Test normal()
        cept = Vector3(np.random.random((100,3))).unit().element_mul(planet.radii)
        perp = planet.normal(cept)
        test1 = cept.element_mul(planet.spheroid.unsquash).unit()
        test2 = perp.element_mul(planet.spheroid.squash).unit()

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
        self.assertTrue(abs(cept.element_mul(planet.spheroid.unsquash).norm() -
                            planet.spheroid.req).max() < 1.e-6)

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
        self.assertTrue(abs(cept.element_mul(planet.spheroid.unsquash).norm() -
                        planet.spheroid.req).max() < 1.e-6)

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

        # Confirm that latitudes are planetocentric
        NPTS = 10000
        pos = (2 * np.random.rand(NPTS,3) - 1.) * REQ   # range is -REQ to REQ

        (lon,lat,elev) = planet.coords_from_vector3(pos, axes=3)
        sep = Vector3.ZAXIS.sep(pos)
        test_lat = HALFPI - sep
        self.assertTrue(abs(lat - test_lat).max() < 1.e-8)

        # Confirm that latitudes convert to planetographic
        new_lat = planet.lat_to_graphic(lat)

        sep = Vector3.ZAXIS.sep(planet.normal(pos))
        test_lat = HALFPI - sep
        self.assertTrue(abs(new_lat - test_lat).max() < 1.e-8)

        # Confirm that planetographic latitudes convert back to planetocentric
        newer_lat = planet.lat_from_graphic(new_lat)
        self.assertTrue(abs(newer_lat - lat).max() < 1.e-8)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
