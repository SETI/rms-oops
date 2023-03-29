################################################################################
# oops/surface/spheroid.py: Spheroid subclass of class Surface
################################################################################

import numpy as np
from polymath     import Scalar, Vector3
from oops.config  import SURFACE_PHOTONS, LOGGING
from oops.frame   import Frame
from oops.path    import Path
from oops.surface.ellipsoid import Ellipsoid

class Spheroid(Ellipsoid):
    """A spheroidal surface centered on the given path and fixed with respect to
    the given frame. The short radius of the spheroid is oriented along the
    Z-axis of the frame.

    The coordinates defining the surface grid are (longitude, latitude), based
    on the assumption that a spherical body has been "squashed" along the
    Z-axis. The latitude defined in this manner is neither planetocentric nor
    planetographic; functions are provided to perform the conversion to either
    choice. Longitudes are measured in a right-handed manner, increasing toward
    the east; values range from 0 to 2*pi.

    Elevations are defined by "unsquashing" the radial vectors and then
    subtracting off the equatorial radius of the body. Thus, the surface is
    defined as the locus of points where elevation equals zero. However, note
    that with this definition, the gradient of the elevation value is not
    exactly normal to the surface.
    """

    #===========================================================================
    def __init__(self, origin, frame, radii, exclusion=0.95):
        """Constructor for a Spheroid surface.

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

        # Allow either two or three radius values
        if len(radii) == 2:
            radii = (radii[0], radii[0], radii[1])

        Ellipsoid.__init__(self, origin, frame, radii=radii,
                                                exclusion=exclusion)

    #===========================================================================
    def intercept_normal_to(self, pos, time=None, direction='dep', derivs=False,
                                       guess=None):
        """Intercept point whose normal vector passes through a given position.

        This is a bit faster and more reliable than the default method for
        Ellipsoids, because the polynomial is only fourth-order instead of
        sixth-order.

        Input:
            pos         a Vector3 of positions at or near the surface relative
                        to this surface's origin and frame.
            time        a Scalar time at the surface; ignored here.
            direction   'arr' for a photon arriving at the surface; 'dep' for a
                        photon departing from the surface; ignored here.
            derivs      True to propagate derivatives in pos into the returned
                        intercepts.
            guess       optional initial guess a coefficient array p such that:
                            intercept + p * normal(intercept) = pos
                        Use guess=True for the converged value of p to be
                        returned even if an initial guess was not provided.

        Return:         intercept or (intercept, p).
            intercept   a vector3 of surface intercept points relative to this
                        surface's origin and frame, in km. Where no intercept
                        exists, the returned vector will be masked.
            p           the converged solution such that
                            intercept = pos + p * normal(intercept);
                        included if guess is not None.
        """

        pos = Vector3.as_vector3(pos, recursive=derivs)
        pos = self._apply_exclusion(pos)

        # If we work in the plane defined by the position and the Z-axis, this
        # becomes a 2-D problem, with pos = (pos_x, pos_z).

        (pos_x3d, pos_y3d, pos_z) = pos.to_scalars()
        pos_x = (pos_x3d**2 + pos_y3d**2).sqrt()

        # We can always recover the 3D (x,y) coordinates via
        #   pos_x3d = pos_x * cos_lon
        #   pos_y3d = pos_y * sin_lon
        # where:

        cos_lon = pos_x3d / pos_x
        sin_lon = pos_y3d / pos_x

        # We need to solve for p such that:
        #   cept + p * normal(cept) = pos
        # where
        #   normal(cept) = cept.element_mul(self.unsquash_sq)
        #
        # This is subject to the constraint that cept is the intercept point on
        # the surface, where
        #   cept_unsquashed = cept.element_mul(self.unsquash)
        # and
        #   cept_unsquashed.dot(cept_unsquashed) = self.req_sq
        #
        # Let:

        C = self.unsquash_z_sq
        R = self.req_sq

        # Three equations with three unknowns:
        # cept_x + p * cept_x = pos_x
        # cept_z + p * cept_z * C = pos_z
        # cept_x**2 + cept_z**2 * C = R
        #
        # Let:

        X = pos_x**2
        Z = pos_z**2 * C

        # Plug the first two into the third and rearrange:
        #
        # f(p) = (  X * ((1 + C*p))**2
        #         + Z * ((1 + p))**2
        #         - R * ((1 + p) * (1 + C*p))**2)
        #
        # This is a fourth-order polynomial, which we need to solve for f(p) = 0.
        #
        # Using SymPy, this expands to:
        #
        # f(p) = (-C**2*R) * p**4
        #        + (-2*C**2*R - 2*C*R) * p**3
        #        + (-C**2*R + C**2*X - 4*C*R - R + Z) * p**2
        #        + (-2*C*R + 2*C*X - 2*R + 2*Z) * p
        #        + (- R + X + Z)
        #
        # Let f(p) = (((f4*p + f3)*p + f2)*p + f1)*p + f0

        C2 = C**2

        # For efficiency, we segregate all the array ops (involving X and Z)
        f4 = -C2 * R
        f3 = -2 * R * (C2 + C)
        f2 = C2 * X + Z - R * (C2 + 4*C + 1)
        f1 = (2*C) * X + 2 * Z - 2 * R * (C + 1)
        f0 = X + Z - R

        g3 = 4 * f4
        g2 = 3 * f3
        g1 = 2 * f2
        g0 = f1

        # Make an initial guess at p if necessary
        if guess in (None, True):

            # Unsquash into coordinates where the surface is a sphere
            pos_unsq = pos.wod.element_mul(self.unsquash)   # without derivs!

            # Estimate the intercept point as on a straight line to the origin
            # (Note that this estimate is exact for points at the surface.)
            cept_guess_unsq = pos_unsq.with_norm(self.req)

            # Make a guess at the normal vector in unsquashed coordinates
            normal_guess_unsq = cept_guess_unsq.element_mul(self.unsquash_sq)

            # Estimate p
            p = ((pos_unsq.norm() - cept_guess_unsq.norm())
                 / normal_guess_unsq.norm())

        else:
            p = guess.wod.copy()

        # The precision of p should match the default geometric accuracy defined
        # by SURFACE_PHOTONS.km_precision. Set our precision goal on p
        # accordingly.
        km_scale = self.req
        precision = SURFACE_PHOTONS.km_precision / km_scale

        # Iterate until convergence stops
        max_dp = 1.e99
        converged = False

        # We typically need a few extra iterations to reach desired precision
        for count in range(SURFACE_PHOTONS.max_iterations + 5):

            # Calculate f and df/dp
            f = (((f4*p + f3)*p + f2)*p + f1)*p + f0
            df_dp = ((g3*p + g2)*p + g1)*p + g0

            # One step of Newton's method
            dp = f / df_dp
            p -= dp

            prev_max_dp = max_dp
            max_dp = dp.abs().max(builtins=True, masked=-1.)

            if LOGGING.surface_iterations or Ellipsoid.DEBUG:
                LOGGING.convergence(
                            '%s.intercept_normal_to(): iter=%d; change[km]=%.6g'
                            % (type(self).__name__, count+1, max_dp * km_scale))


            if max_dp <= precision:
                converged = True
                break

            if max_dp >= prev_max_dp:
                break

        if not converged:
            LOGGING.warn('%s.intercept_normal_to() did not converge: '
                         'iter=%d; change[km]=%.6g'
                         % (type(self).__name__, count+1, max_dp))

        cept_x = pos_x / (1. + p)
        cept_z = pos_z / (1. + C * p)
        cept = Vector3.from_scalars(cos_lon * cept_x, sin_lon * cept_x, cept_z)

        if guess is None:
            return cept
        else:
            return (cept, p)

    ############################################################################
    # Longitude conversions
    ############################################################################

    def lon_to_centric(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetocentric.

        This is a null operation for spheroids. The method is provided for
        compatibility with Ellipsoids.

        Input:
            lon         squashed longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    #===========================================================================
    def lon_from_centric(self, lon, derivs=False):
        """Convert planetocentric longitude to internal coordinates.

        This is a null operation for spheroids. The method is provided for
        compatibility with Ellipsoids.

        Input:
            lon         planetocentric longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          squashed longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    #===========================================================================
    def lon_to_graphic(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetographic.

        This is a null operation for spheroids. The method is provided for
        compatibility with Ellipsoids.

        Input:
            lon         squashed longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetographic longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    #===========================================================================
    def lon_from_graphic(self, lon, derivs=False):
        """Convert planetographic longitude to internal coordinates.

        This is a null operation for spheroids. The method is provided for
        compatibility with Ellipsoids.

        Input:
            lon         planetographic longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          squashed longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    ############################################################################
    # Latitude conversions
    ############################################################################

    def lat_to_centric(self, lat, lon=None, derivs=False):
        """Convert latitude in internal coordinates to planetocentric.

        Input:
            lat         squashed latitide, radians.
            lon         ignored, included for compatibility with Ellipsoids.
            derivs      True to include derivatives in returned result.

        Return          planetocentric latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan(derivs) * self.squash_z).arctan()

    #===========================================================================
    def lat_from_centric(self, lat, lon=None, derivs=False):
        """Convert planetocentric latitude to internal spheroid coordinates.

        Input:
            lat         planetocentric latitide, radians.
            lon         ignored, included for compatibility with Ellipsoids.
            derivs      True to include derivatives in returned result.

        Return          squashed latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self.unsquash_z).arctan()

    #===========================================================================
    def lat_to_graphic(self, lat, lon=None, derivs=False):
        """Convert latitude in internal coordinates to planetographic.

        Input:
            lat         squashed latitide, radians.
            lon         ignored, included for compatibility with Ellipsoids.
            derivs      True to include derivatives in returned result.

        Return          planetographic latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self.unsquash_z).arctan()

    #===========================================================================
    def lat_from_graphic(self, lat, lon=None, derivs=False):
        """Convert a planetographic latitude to internal spheroid latitude.

        Input:
            lat         planetographic latitide, radians.
            lon         ignored, included for compatibility with Ellipsoids.
            derivs      True to include derivatives in returned result.

        Return          squashed latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self.squash_z).arctan()

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Spheroid(unittest.TestCase):

    def runTest(self):

        np.random.seed(2344)

        REQ  = 60268.
        #RPOL = 54364.
        RPOL = 50000.
        planet = Spheroid('SSB', 'J2000', (REQ, RPOL))

        # Coordinate/vector conversions
        NPTS = 10000
        pos = (2 * np.random.rand(NPTS,3) - 1.) * REQ   # range is -REQ to REQ

        lon = Scalar(np.random.rand(NPTS) * Scalar.TWOPI)
        lat = Scalar(np.random.rand(NPTS) * Scalar.PI - Scalar.HALFPI)
        test = planet.vector3_from_coords((lon,lat))
        (lon2, lat2) = planet.coords_from_vector3(test, axes=2)
        self.assertTrue((lon - lon2).abs().max() < 1.e-15)
        self.assertTrue((lat - lat2).abs().max() < 1.e-11)

        test2 = planet.vector3_from_coords((lon2,lat2))
        self.assertTrue((test2 - test).norm() < 1.e-6)

        lon = Scalar(np.random.rand(NPTS) * Scalar.TWOPI)
        lat = Scalar(np.random.rand(NPTS) * Scalar.PI - Scalar.HALFPI)
        z = Scalar(np.random.rand(NPTS) * 1000.)
        test2 = planet.vector3_from_coords((lon,lat,z))
        test3 = planet.vector3_from_coords((lon,lat))
        diff = test2 - test3
        self.assertTrue(diff.norm() - z < 3.e-11)
        self.assertTrue(diff.sep(planet.normal(test3)).max() < 1.e-10)

        (lon2, lat2, z2) = planet.coords_from_vector3(test2, axes=3)
        (lon3, lat3, z3) = planet.coords_from_vector3(test3, axes=3)
        self.assertTrue((lon - lon2).abs().max() < 1.e-15)
        self.assertTrue((lat - lat2).abs().max() < 3.e-12)
        self.assertTrue((lon3 - lon2).abs().max() < 1.e-15)
        self.assertTrue((lat3 - lat2).abs().max() < 3.e-12)
        self.assertTrue(z3.abs().max() < 1.e-10)
        self.assertTrue((z2 - z).abs().max() < 1.e-10)

        (lon,lat,elev) = planet.coords_from_vector3(pos, axes=3)
        test = planet.vector3_from_coords((lon,lat,elev))
        self.assertTrue(abs(test - pos).max() < 1.e-8)

        (_, track1) = planet.vector3_from_coords((lon,lat,z), groundtrack=True)
        (_, _, track2) = planet.coords_from_vector3(test, axes=2, groundtrack=True)
        self.assertTrue((track1 - track2).norm().max() < 1.e-10)

        pos = (2 * np.random.rand(NPTS,3) - 1.) * REQ   # range is -REQ to REQ

        (lon,lat,elev,track) = planet.coords_from_vector3(pos, axes=3, groundtrack=True)
        test = planet.vector3_from_coords((lon,lat,elev))
        self.assertTrue((test - pos).norm().max() < 1.e-8)

        # Make sure longitudes convert to planetocentric and back
        test_lon = np.arctan2(track.vals[...,1], track.vals[...,0]) % Scalar.TWOPI
        centric_lon = planet.lon_to_centric(lon)
        diffs = (centric_lon - test_lon + Scalar.HALFPI) % Scalar.PI - Scalar.HALFPI
        self.assertTrue((diffs).abs().max() < 1.e-8)

        test_lon2 = planet.lon_from_centric(centric_lon)
        diffs = (test_lon2 - lon + Scalar.HALFPI) % Scalar.PI - Scalar.HALFPI
        self.assertTrue((diffs).abs().max() < 1.e-8)

        # Make sure latitudes convert to planetocentric and back
        test_lat = np.arcsin(track.vals[...,2] / np.sqrt(np.sum(track.vals**2, axis=-1)))
        centric_lat = planet.lat_to_centric(lat,lon)
        self.assertTrue(abs(centric_lat - test_lat).max() < 1.e-8)

        test_lat2 = planet.lat_from_centric(centric_lat, lon)
        self.assertTrue(abs(test_lat2 - lat).max() < 1.e-8)

        # Make sure longitudes convert to planetographic and back
        normals = planet.normal(track)
        test_lon = np.arctan2(normals.vals[...,1], normals.vals[...,0])
        graphic_lon = planet.lon_to_graphic(lon)
        diffs = (graphic_lon - test_lon + Scalar.HALFPI) % Scalar.PI - Scalar.HALFPI
        self.assertTrue(abs(diffs).max() < 1.e-8)

        test_lon2 = planet.lon_from_centric(centric_lon)
        diffs = (test_lon2 - lon + Scalar.HALFPI) % Scalar.PI - Scalar.HALFPI
        self.assertTrue(abs(diffs).max() < 1.e-8)

        # Make sure latitudes convert to planetographic and back
        test_lat = np.arcsin(normals.vals[...,2] / normals.norm().vals)
        graphic_lat = planet.lat_to_graphic(lat,lon)
        self.assertTrue(abs(graphic_lat - test_lat).max() < 1.e-8)

        test_lat2 = planet.lat_from_graphic(graphic_lat, lon)
        self.assertTrue(abs(test_lat2 - lat).max() < 1.e-8)

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
        test1 = cept.element_mul(planet.unsquash).unit()
        test2 = perp.element_mul(planet.squash).unit()

        self.assertTrue(abs(test1 - test2).max() < 1.e-12)

        eps = 1.e-7
        (lon,lat) = planet.coords_from_vector3(cept, axes=2)
        cept1 = planet.vector3_from_coords((lon+eps,lat,0.))
        cept2 = planet.vector3_from_coords((lon-eps,lat,0.))

        self.assertTrue(abs((cept2 - cept1).sep(perp) - Scalar.HALFPI).max() < 1.e-8)

        (lon,lat) = planet.coords_from_vector3(cept, axes=2)
        cept1 = planet.vector3_from_coords((lon,lat+eps,0.))
        cept2 = planet.vector3_from_coords((lon,lat-eps,0.))

        self.assertTrue(abs((cept2 - cept1).sep(perp) - Scalar.HALFPI).max() < 1.e-8)

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
        self.assertTrue(abs(cept.element_mul(planet.unsquash).norm() -
                            planet.req).max() < 1.e-6)

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
        N = 1000
        pos = Vector3(np.random.random((N,3)) * 4.*REQ + REQ)
        pos.insert_deriv('pos', Vector3.IDENTITY, override=True)
        (cept,t) = planet.intercept_normal_to(pos, derivs=True, guess=True)
        self.assertTrue(abs(cept.element_mul(planet.unsquash).norm() -
                            planet.req).max() < 1.e-6)

        eps = 1.
        dpos = ((eps,0,0), (0,eps,0), (0,0,eps))
        perp = planet.normal(cept)
        for i in range(3):
            (cept1,t1) = planet.intercept_normal_to(pos + dpos[i], derivs=False,
                                                    guess=t)
            (cept2,t2) = planet.intercept_normal_to(pos - dpos[i], derivs=False,
                                                    guess=t)
            dcept_dpos = (cept1 - cept2) / (2*eps)
            self.assertTrue(abs(dcept_dpos.sep(perp) - Scalar.HALFPI).max() < 1.e-5)

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
