################################################################################
# oops/surface_/limb.py: Limb subclass of class Surface
################################################################################

import numpy as np
from polymath import *

from oops.surface_.surface   import Surface
from oops.surface_.spheroid  import Spheroid
from oops.surface_.ellipsoid import Ellipsoid
from oops.path_.path         import Path
from oops.frame_.frame       import Frame

from oops.config             import SURFACE_PHOTONS, LOGGING
from oops.constants          import *

class Limb(Surface):
    """The Limb surface is defined as the locus of points where a surface normal
    from a spheroid or ellipsoid is perpendicular to the line of sight. This 
    provides a convenient coordinate system for describing cloud features on the
    limb of a body.

    The coordinates are (lon, lat, z), much the same as for the surface of the
    associated spheroid or ellipsoid. The key difference is in how the
    intercept point is derived.
        lon     longitude at the ground point beneath the limb point, using the
                same definition as that of the associated spheroid or ellipsoid.
        lat     latitude at the ground point beneath the limb point, using the
                same definition as that of the associated spheroid or ellipsoid.
        z       the elevation above the surface, as an actual distance measured
                normal to the ring plane. Note that this definition differs from
                that used by the spheroid and ellipsoid surface.
    """

    COORDINATE_TYPE = "limb"
    IS_VIRTUAL = True
    DEBUG = False   # True for convergence testing in intercept()
    SPEEDUP = True  # True to retain history of the most recent guess. This will
                    # fail if oops ever becomes multithreaded

    # Class constants to override where derivs are undefined
    coords_from_vector3_DERIVS_ARE_IMPLEMENTED = False
    vector3_from_coords_DERIVS_ARE_IMPLEMENTED = False
    intercept_DERIVS_ARE_IMPLEMENTED = False
    normal_DERIVS_ARE_IMPLEMENTED = False

    def __init__(self, ground, limits=None):
        """Constructor for a Limb surface.

        Input:
            ground      the Surface object relative to which limb points are to
                        be defined. It should be a Spheroid or Ellipsoid.
            limits      an optional single value or tuple defining the absolute
                        numerical limit(s) placed on the limb; values outside
                        this range are masked.
        """

        assert ground.COORDINATE_TYPE == "spherical"
        self.ground = ground
        self.origin = ground.origin
        self.frame  = ground.frame

        # Used if SPEEDUP = True to speed up repeated calls to
        #   self.ground.intercept_normal_to_iterated()
        self.ground_guess = None
        self.ground_shape = None

        if limits is None:
            self.limits = None
        else:
            self.limits = (limits[0], limits[1])

    def coords_from_vector3(self, pos, obs=None, axes=2, derivs=False):
        """Converts position vectors in the internal frame into the surface
        coordinate system.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            obs         a Vector3 of observer observer positions. Ignored for
                        solid surfaces but needed for virtual surfaces.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects
            derivs      a boolean or tuple of booleans. If True, then the
                        partial derivatives of each coordinate with respect to
                        surface position and observer position are returned as
                        well. Using a tuple, you can indicate whether to return
                        partial derivatives on a coordinate-by-coordinate
                        basis.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate. The
                        coordinates are (longitude, latitude, elevation). Units
                        are radians and km; longitude ranges from 0 to 2*pi.

                        If derivs is True, then the coordinate has extra
                        attributes "d_dpos" and "d_dobs", which contain the
                        partial derivatives with respect to the surface position
                        and the observer position, represented as a MatrixN
                        objects with item shape [1,3].
        """

        # Re-use the most recent guess if the shape is unchanged
        if pos.shape == self.ground_shape:
            guess = self.ground_guess
        else:
            guess = None

        (groundtrack, guess) = self.ground.intercept_normal_to_iterated(pos,
                                                        derivs, t_guess=guess)
        if Limb.SPEEDUP:
            self.ground_shape = guess.shape
            self.ground_guess = guess

        (lon, lat) = self.ground.coords_from_vector3(groundtrack, derivs)
        z = (pos - groundtrack).norm()
        z *= (pos.norm() - groundtrack.norm()).sign()

        # Mask based on elevation limits if necessary
        if self.limits is not None:
            mask = (lon.mask | (z.vals < self.limits[0]) |
                               (z.vals > self.limits[1]))
            lon.mask = mask
            lat.mask = mask
            z.mask = mask

        if axes == 2:
            return (lon, lat)
        else:
            return (lon, lat, z)

    def vector3_from_coords(self, coords, obs=None, derivs=False):
        """Returns the position where a point with the given surface coordinates
        would fall in the surface frame, given the location of the observer.

        Input:
            coords      a tuple of two or three Scalars defining the coordinates
                lon     longitude in radians.
                lat     latitude in radians
                z       the perpendicular distance from the surface, in km;
            obs         position of the observer in the surface frame.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to observer and to the coordinates.

        Return:         a unitless Vector3 of intercept points defined by the
                        coordinates.

                        If derivs is True, then pos is returned with subfields
                        "d_dobs" and "d_dcoords", where the former contains the
                        MatrixN of partial derivatives with respect to obs and
                        the latter is the MatrixN of partial derivatives with
                        respect to the coordinates. The MatrixN item shapes are
                        [3,3].
        """

        lon = coords[0]
        lat = coords[1]
        z = coords[2]

        groundtrack = self.ground.vector3_from_coords((lon,lat), derivs=derivs)
        normal = self.ground.normal(groundtrack, derivs=derivs).unit()
        return groundtrack + z * normal

    def intercept(self, obs, los, derivs=False, t_guess=None,
                  groundtrack=False):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.
            t_guess     initial guess at the t array, optional.
            groundtrack True to include the surface intercept points of the body
                        associated with each limb intercept. This array can
                        speed up any subsequent calculations such as calls to
                        normal(), and can be used to determine locations in
                        body coordinates.

        Return:         a tuple (pos, t) where:
            pos         a unitless Vector3 of intercept points on the surface,
                        in km.
            t           a unitless Scalar such that:
                            position = obs + t * los

                        If derivs is True, then pos and t are returned with
                        subfields "d_dobs" and "d_dlos", where the former
                        contains the MatrixN of partial derivatives with respect
                        to obs and the latter is the MatrixN of partial
                        derivatives with respect to los. The MatrixN item shapes
                        are [3,3] for the derivatives of pos, and [1,3] for the
                        derivatives of t. For purposes of differentiation, los
                        is assumed to have unit length.

                        If groundtrack is True, pos contains a subfield
                        "groundtrack" containing the associated positions on the
                        surface of the body.
        """

        if derivs:
            raise NotImplementedError("Limb.intercept() " +
                                      " does not implement derivatives")

        # Convert to standard units
        obs = Vector3.as_vector3(obs)
        los = Vector3.as_vector3(los)

        # Re-use the most recent guess if the shape is unchanged
        if obs.shape == self.ground_shape and los.shape == self.ground_shape:
            guess = self.ground_guess
        else:
            guess = None

        # Solve for the intercept distance where the line of sight is normal to
        # the surface.
        #
        # pos = obs + t * los
        # ground.normal(ground.intercept_normal_to(pos(t))) dot los = 0
        #
        # Solve for t.
        #
        # f(t) = perp(pos(t)) dot los
        #
        # df/dt = dperp/dpos(pos(t)) dpos/dt dot los
        #       = [dperp/dpos(pos(t)) los] dot los
        #
        # Initial guess is where los and pos are perpendicular:
        # (obs + t * los) dot los = 0
        #
        # t = -(obs dot los) / (los dot los)

        if t_guess is None:
            t = -obs.dot(los) / los.dot(los)
        else:
            t = t_guess.copy()

        max_dt = 1.e99
        for iter in range(SURFACE_PHOTONS.max_iterations):
            pos = obs + t * los

            (track, guess) = self.ground.intercept_normal_to_iterated(pos,
                                                    derivs=True, t_guess=guess)
            perp = self.ground.normal(track, derivs=True)

            df_dt = (perp.d_dpos * track.d_dpos * los).dot(los)
            dt = perp.plain().dot(los) / df_dt

            t = t - dt

            prev_max_dt = max_dt
            max_dt = abs(dt).max()

            if LOGGING.surface_iterations or Limb.DEBUG:
                print LOGGING.prefix, "Surface.limb.intercept", iter, max_dt

            if (max_dt <= SURFACE_PHOTONS.dlt_precision or
                max_dt >= prev_max_dt * 0.5): break

        pos = obs + t * los

        if derivs:
            raise NotImplementedError("Limb.intercept() " +
                                      " does not implement derivatives")

        if groundtrack:
            (track, guess) = self.ground.intercept_normal_to_iterated(pos,
                                                    derivs=True, t_guess=guess)
            pos.insert_subfield("groundtrack", track)

        if Limb.SPEEDUP:
            self.ground_shape = guess.shape
            self.ground_guess = guess

        return (pos, t)

    def normal(self, pos, derivs=False):
        """Returns the normal vector at a position at or near a surface. This
        is the normal to the planetary surface, not the normal to the limb
        surface.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            derivs      True to include a matrix of partial derivatives.

        Return:         a unitless Vector3 containing directions normal to the
                        surface that pass through the position. Lengths are
                        arbitrary.

                        If derivs is True, then the normal vectors returned have
                        a subfield "d_dpos", which contains the partial
                        derivatives with respect to components of the given
                        position vector, as a MatrixN object with item shape
                        [3,3].
        """

        pos = Vector3.as_vector3(pos)

        if pos.subfields.has_key("groundtrack"):
            groundtrack = pos.groundtrack
        else:

            # Re-use the most recent guess if the shape is unchanged
            if pos.shape == self.ground_shape:
                guess = self.ground_guess
            else:
                guess = None

            (groundtrack, guess) = self.ground.intercept_normal_to_iterated(pos,
                                                derivs=derivs, t_guess=guess)
            if Limb.SPEEDUP:
                self.ground_shape = guess.shape
                self.ground_guess = guess

        return self.ground.normal(groundtrack, derivs=derivs)

    ############################################################################
    # Latitude conversions
    ############################################################################

    def lat_to_centric(self, lat):
        """Converts a latitude value given in internal spheroid coordinates to
        its planetocentric equivalent.
        """

        return self.ground.lat_to_centric(lat)

    def lat_to_graphic(self, lat):
        """Converts a latitude value given in internal spheroid coordinates to
        its planetographic equivalent.
        """

        return self.ground.lat_to_graphic(lat)

    def lat_from_centric(self, lat):
        """Converts a latitude value given in planetocentric coordinates to its
        equivalent value in internal spheroid coordinates.
        """

        return self.ground.lat_from_centric(lat)

    def lat_from_graphic(self, lat):
        """Converts a latitude value given in planetographic coordinates to its
        equivalent value in internal spheroid coordinates.
        """

        return self.ground.lat_from_graphic(lat)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Limb(unittest.TestCase):

    def runTest(self):

        save_dlt_precision = SURFACE_PHOTONS.dlt_precision
        SURFACE_PHOTONS.dlt_precision = 0

        from oops.frame_.frame import Frame
        from oops.path_.path import Path

        REQ  = 60268.
        RPOL = 50000.
        ground = Spheroid("SSB", "J2000", (REQ, RPOL))
        limb = Limb(ground)

        obs = Vector3(np.random.random((1000,3)) * 4.*REQ + REQ)
        los = Vector3(np.random.random((1000,3)))

        (cept,t) = limb.intercept(obs, los, groundtrack=True)

        self.assertTrue(abs(limb.normal(cept).sep(los) - HALFPI) < 1.e-5)

        lon = np.random.random(10) * TWOPI
        lat = np.arcsin(np.random.random(10) * 2. - 1.)
        z = np.random.random(10) * 10000.

        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        self.assertTrue(abs(coords[0] - lon) < 1.e-6)
        self.assertTrue(abs(coords[1] - lat) < 1.e-6)
        self.assertTrue(abs(coords[2] - z) < 1.e-2)

        SURFACE_PHOTONS.dlt_precision = save_dlt_precision

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
