################################################################################
# oops/surface/centricellipsoid.py: CentricEllipsoid subclass of class Surface
################################################################################

import numpy as np
from polymath               import Scalar, Vector3
from oops.frame             import Frame
from oops.path              import Path
from oops.surface.ellipsoid import Ellipsoid

class CentricEllipsoid(Ellipsoid):
    """A variant of Ellipsoid in which latitudes and longitudes are
    planetocentric.
    """

    #===========================================================================
    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                  derivs=False, hints=None, groundtrack=False):
        """Surface coordinates associated with a position vector.

        Input:
            pos         a Vector3 of positions at or near the surface, relative
                        to this surface's origin and frame.
            obs         a Vector3 of observer position relative to this
                        surface's origin and frame; ignored for this Surface
                        subclass.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            axes        2 or 3, indicating whether to return the first two
                        coordinates (lon, lat) or all three (lon, lat, z) as
                        Scalars.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.
            hints       optionally, the value of the coefficient p such that
                            ground + p * normal(ground) = pos;
                        ignored if the value is None (the default) or True.
            groundtrack True to return the intercept on the surface along with
                        the coordinates.

        Return:         a tuple of two to four items:
            lon         longitude at the surface in radians.
            lat         latitude at the surface in radians.
            z           vertical altitude in km normal to the surface; included
                        if axes == 3.
            track       intercept point on the surface (where z == 0); included
                        if input groundtrack is True.
        """

        # Validate inputs
        self._coords_from_vector3_check(axes)

        pos = Vector3.as_vector3(pos, recursive=derivs)

        # Use the quick solution for the body points if hints are provided
        if isinstance(hints, (type(None), bool, np.bool_)):
            (track, p) = self.intercept_normal_to(pos, guess=True)
        else:
            p = Scalar.as_scalar(hints, recursive=derivs)
            denom = Vector3.ONES + p * self.unsquash_sq
            track = pos.element_div(denom)

        # Derive the coordinates
        (x,y,z) = track.to_scalars()
        lat = (z/track.norm()).arcsin()
        lon = y.arctan2(x) % Scalar.TWOPI

        results = (lon, lat)

        if axes == 3:
            r = (pos - track).norm() * p.sign()
            results += (r,)

        if groundtrack:
            results += (track,)

        return results

    #===========================================================================
    def vector3_from_coords(self, coords, obs=None, time=None, derivs=False,
                                          groundtrack=False):
        """The position where a point with the given coordinates falls relative
        to this surface's origin and frame.

        Input:
            coords      a tuple of two or three Scalars defining coordinates at
                        or near this surface. These can have different shapes,
                        but must be broadcastable to a common shape.
                lon     longitude at the surface in radians.
                lat     latitude at the surface in radians.
                z       vertical altitude in km normal to the body surface.
            obs         a Vector3 of observer position relative to this
                        surface's origin and frame; ignored for this Surface
                        subclass.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.
            groundtrack True to include the associated groundtrack points on the
                        body surface in the returned result.

        Return:         pos or (pos, track), where
            pos         a Vector3 of points defined by the coordinates, relative
                        to this surface's origin and frame.
            track       intercept point on the surface (where z == 0); included
                        if input groundtrack is True.
        """

        # Validate inputs
        self._vector3_from_coords_check(coords)

        (lon, lat) = coords[:2]
        squashed_lon = Ellipsoid.lon_from_centric(self, lon, derivs=derivs)
        squashed_lat = Ellipsoid.lat_from_centric(self, lat, squashed_lon,
                                                        derivs=derivs)
        new_coords = (squashed_lon, squashed_lat,) + coords[2:]

        return Ellipsoid.vector3_from_coords(self, new_coords, derivs=derivs,
                                                   groundtrack=groundtrack)

    ############################################################################
    # Longitude conversions
    ############################################################################

    def lon_to_centric(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetocentric.

        Input:
            lon         planetocentric longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    #===========================================================================
    def lon_from_centric(self, lon, derivs=False):
        """Convert planetocentric longitude to internal coordinates.

        Input:
            lon         planetocentric longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    #===========================================================================
    def lon_to_graphic(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetographic.

        Input:
            lon         planetocentric longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetographic longitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self.unsquash_y_sq).arctan2(lon.cos())

    #===========================================================================
    def lon_from_graphic(self, lon, derivs=False):
        """Convert planetographic longitude to internal coordinates.

        Input:
            lon         planetographic longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric longitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self.squash_y_sq).arctan2(lon.cos())

    ############################################################################
    # Latitude conversions
    ############################################################################

    def lat_to_centric(self, lat, lon, derivs=False):
        """Convert latitude in internal coordinates to planetocentric.

        Input:
            lat         squashed latitide, radians.
            lon         planetocentric longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric latitude.
        """

        return Scalar.as_scalar(lat, recursive=derivs)

    #===========================================================================
    def lat_from_centric(self, lat, lon, derivs=False):
        """Convert planetocentric latitude to internal coordinates.

        Input:
            lat         planetocentric latitide, radians.
            lon         planetocentric longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric latitude.
        """

        return Scalar.as_scalar(lat, recursive=derivs)

    #===========================================================================
    def lat_to_graphic(self, lat, lon, derivs=False):
        """Convert latitude in internal coordinates to planetographic.

        Input:
            lat         squashed latitide, radians.
            lon         planetocentric longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          planetographic latitude.
        """

        # This could be done more efficiently I'm sure
        squashed_lon = Ellipsoid.lon_from_centric(self, lon, derivs=derivs)
        squashed_lat = Ellipsoid.lat_from_centric(self, lat, squashed_lon,
                                                        derivs=derivs)

        return Ellipsoid.lat_to_graphic(self, squashed_lat, squashed_lon,
                                              derivs=derivs)

    #===========================================================================
    def lat_from_graphic(self, lat, lon, derivs=False):
        """Convert a planetographic latitude to internal coordinates.

        Input:
            lat         planetographic latitide, radians.
            lon         planetocentric longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric latitude.
        """

        squashed_lon = Ellipsoid.lon_from_centric(self, lon, derivs=derivs)
        squashed_lat = Ellipsoid.lat_from_graphic(self, lat, squashed_lon,
                                                        derivs=derivs)

        return Ellipsoid.lat_to_centric(self, squashed_lat, squashed_lon,
                                              derivs=derivs)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_CentricEllipsoid(unittest.TestCase):

    def runTest(self):

        np.random.seed(9123)

        REQ  = 60268.
        RMID = 54364.
        RPOL = 50000.
        planet = CentricEllipsoid('SSB', 'J2000', (REQ, RMID, RPOL))
        ellipsoid = Ellipsoid('SSB', 'J2000', (REQ, RMID, RPOL))

        # Coordinate/vector conversions
        NPTS = 10000
        pos = (2 * np.random.rand(NPTS,3) - 1.) * REQ   # range is -REQ to REQ

        lon = Scalar(np.random.rand(NPTS) * Scalar.TWOPI)
        lat = Scalar(np.random.rand(NPTS) * Scalar.PI - Scalar.HALFPI)
        track = planet.vector3_from_coords((lon,lat))
        (lon2, lat2) = planet.coords_from_vector3(track, axes=2)
        self.assertTrue((lon - lon2).abs().max() < 1.e-15)
        self.assertTrue((lat - lat2).abs().max() < 1.e-11)

        track2 = planet.vector3_from_coords((lon2,lat2))
        self.assertTrue((track2 - track).norm() < 1.e-6)

        lon = Scalar(np.random.rand(NPTS) * Scalar.TWOPI)
        lat = Scalar(np.random.rand(NPTS) * Scalar.PI - Scalar.HALFPI)
        z = Scalar(np.random.rand(NPTS) * 1000.)
        test = planet.vector3_from_coords((lon,lat,z))
        track = planet.vector3_from_coords((lon,lat))
        diff = test - track
        self.assertTrue((diff.norm()).abs() - z < 3.e-11)
        self.assertTrue(diff.sep(planet.normal(track)).max() < 3.e-10)

        (lon2, lat2, z2) = planet.coords_from_vector3(test, axes=3)
        (lon3, lat3, z3) = planet.coords_from_vector3(track, axes=3)
        self.assertTrue((lon - lon2).abs().max() < 1.e-15)
        self.assertTrue((lat - lat2).abs().max() < 3.e-12)
        self.assertTrue((lon3 - lon2).abs().max() < 1.e-15)
        self.assertTrue((lat3 - lat2).abs().max() < 1.e-11)
        self.assertTrue(z3.abs().max() < 1.e-10)
        self.assertTrue((z2 - z).abs().max() < 1.e-10)

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
        self.assertTrue(abs(centric_lon - test_lon).max() < 1.e-8)

        test_lon2 = planet.lon_from_centric(centric_lon)
        self.assertTrue(abs(test_lon2 - lon).max() < 1.e-8)

        # Make sure latitudes convert to planetocentric and back
        test_lat = np.arcsin(track.vals[...,2] / np.sqrt(np.sum(track.vals**2, axis=-1)))
        centric_lat = planet.lat_to_centric(lat,lon)
        self.assertTrue(abs(centric_lat - test_lat).max() < 1.e-8)

        test_lat2 = planet.lat_from_centric(centric_lat, lon)
        self.assertTrue(abs(test_lat2 - lat).max() < 1.e-8)

        # Make sure longitudes convert to planetographic and back
        normals = planet.normal(track)
        test_lon = np.arctan2(normals.vals[...,1], normals.vals[...,0]) % Scalar.TWOPI
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

        # Ellipsoid intercepts & normals
        obs = REQ * (np.random.rand(NPTS,3) + 1.)       # range is REQ to 2*REQ
        los = -np.random.rand(NPTS,3)                   # range is -1 to 0

        (pts, t) = planet.intercept(obs, los)
        test = t * Vector3(los) + Vector3(obs)
        self.assertTrue(abs(test - pts).max() < 4.e-9)

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
            sorted = np.sort(errors.vals[errors.antimask])
                        # mask=True where the line of sight missed the surface
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 2.e-5)

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
            self.assertTrue(selected_error < 2.e-5)

            dt_dlos = (t1 - t2) / (2*eps)
            ref = t.d_dlos.vals[...,i]

            errors = abs(dt_dlos/ref - 1)
            sorted = np.sort(errors.vals[errors.antimask])
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 2.e-5)

        # Test normal()
        cept = Vector3(np.random.random((100,3))).unit().element_mul(planet.radii)
        perp = planet.normal(cept)
        test1 = cept.element_mul(ellipsoid.unsquash).unit()
        test2 = perp.element_mul(ellipsoid.squash).unit()

        self.assertTrue(abs(test1 - test2).max() < 1.e-12)

        eps = 1.e-7
        (lon,lat) = planet.coords_from_vector3(cept, axes=2)
        cept1 = planet.vector3_from_coords((lon+eps,lat,0.))
        cept2 = planet.vector3_from_coords((lon-eps,lat,0.))

        self.assertTrue(abs((cept2 - cept1).sep(perp) - Scalar.HALFPI).max() < 3.e-8)

        (lon,lat) = planet.coords_from_vector3(cept, axes=2)
        cept1 = planet.vector3_from_coords((lon,lat+eps,0.))
        cept2 = planet.vector3_from_coords((lon,lat-eps,0.))

        self.assertTrue(abs((cept2 - cept1).sep(perp) - Scalar.HALFPI).max() < 3.e-8)

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
        self.assertTrue(abs(cept.element_mul(ellipsoid.unsquash).norm() -
                        ellipsoid.req).max() < 1.e-6)

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
        (cept,t) = planet.intercept_normal_to(pos, derivs=True, guess=True)
        self.assertTrue(abs(cept.element_mul(ellipsoid.unsquash).norm() -
                        ellipsoid.req).max() < 1.e-6)

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
