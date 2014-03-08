################################################################################
# oops/surface_/spheroid.py: Spheroid subclass of class Surface
################################################################################

import numpy as np
from polymath import *

from oops.surface_.surface import Surface
from oops.config           import SURFACE_PHOTONS, LOGGING

import oops.registry as registry

class Spheroid(Surface):
    """Spheroid defines a spheroidal surface centered on the given path and
    fixed with respect to the given frame. The short radius of the spheroid is
    oriented along the Z-axis of the frame.

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

    COORDINATE_TYPE = "spherical"
    IS_VIRTUAL = False

    DEBUG = False       # True for convergence testing in intercept_normal_to()

    # Class constants to override where derivs are undefined
    coords_from_vector3_DERIVS_ARE_IMPLEMENTED = False
    vector3_from_coords_DERIVS_ARE_IMPLEMENTED = False
    intercept_with_normal_DERIVS_ARE_IMPLEMENTED = False

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

        self.origin_id = registry.as_path_id(origin)
        self.frame_id  = registry.as_frame_id(frame)

        # Allow either two or three radius values
        if len(radii) == 2:
            self.radii = np.array((radii[0], radii[0], radii[1]))
        else:
            self.radii = np.array((radii[0], radii[1], radii[2]))
            assert radii[0] == radii[1]

        self.radii_sq = self.radii**2
        self.req    = self.radii[0]
        self.req_sq = self.req**2
        self.rpol   = self.radii[2]

        self.squash_z   = self.radii[2] / self.radii[0]
        self.unsquash_z = self.radii[0] / self.radii[2]

        self.squash    = Vector3((1., 1.,   self.squash_z))
        self.squash_sq = self.squash.element_mul(self.squash)
        self.unsquash  = Vector3((1,1,1)).element_div(self.squash)
        self.unsquash_sq = self.unsquash.element_mul(self.unsquash)

        self.unsquash_sq_2d = Matrix(([1,0,0],
                                      [0,1,0],
                                      [0,0,self.unsquash_z**2]))

        # This is the exclusion zone radius, within which calculations of
        # intercept_normal_to() are automatically masked due to the ill-defined
        # geometry.

        self.exclusion = exclusion * self.rpol

    def coords_from_vector3(self, pos, obs=None, axes=2, derivs=False):
        """Converts from position vectors in the internal frame into the surface
        coordinate system.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            obs         ignored.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      a boolean or tuple of booleans. If True, then the
                        partial derivatives of each coordinate with respect to
                        surface position and observer position are returned as
                        well. Using a tuple, you can indicate whether to return
                        partial derivatives on an coordinate-by-coordinate
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

        unsquashed = Vector3.as_vector3(pos).element_mul(self.unsquash)

        r = unsquashed.norm()
        (x,y,z) = unsquashed.to_scalars()
        lat = (z/r).arcsin()
        lon = y.arctan2(x) % (2.*np.pi)


        if np.any(derivs):
            raise NotImplementedError("Spheroid.coords_from_vector3() " +
                                      " does not implement derivatives")

        if axes == 2:
            return (lon, lat)
        else:
            return (lon, lat, r - self.req)

    def vector3_from_coords(self, coords, obs=None, derivs=False):
        """Returns the position where a point with the given surface coordinates
        would fall in the surface frame, given the location of the observer.

        Input:
            coords      a tuple of two or three Scalars defining the coordinates
                lon     longitude in radians.
                lat     latitude in radians
                elev    a rough measure of distance from the surface, in km;
            obs         position of the observer in the surface frame; ignored.
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

        # Convert to Scalars in standard units
        lon = coords[0]
        lat = coords[1]

        if len(coords) == 2:
            r = Scalar(self.req)
        else:
            r = coords[2] + self.req

        r_coslat = r * lat.cos()
        x = r_coslat * lon.cos()
        y = r_coslat * lon.sin()
        z = r * lat.sin() * self.squash_z

        pos = Vector3.from_scalars(x,y,z)

        if derivs:
            raise NotImplementedError("Spheroid.vector3_from_coords() " +
                                      " does not implement derivatives")

        return pos

    def intercept(self, obs, los, derivs=False, t_guess=None):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.
            t_guess     initial guess at the t array, optional.

        Return:         a tuple (pos, t) where
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
        """

        # Convert to standard units and un-squash
        obs = Vector3.as_vector3(obs)
        los = Vector3.as_vector3(los)

        obs_unsquashed = obs.element_mul(self.unsquash)
        los_unsquashed = los.element_mul(self.unsquash)

        # Solve for the intercept distance, masking lines of sight that miss

        # Use the quadratic formula...
        # The use of b.sign() below always selects the closer intercept point

        # a = los_unsquashed.dot(los_unsquashed)
        # b = los_unsquashed.dot(obs_unsquashed) * 2.
        # c = obs_unsquashed.dot(obs_unsquashed) - self.req_sq
        # d = b**2 - 4. * a * c
        #
        # t = (-b + b.sign() * d.sqrt()) / (2*a)
        # pos = obs + t*los

        # This is the same algorithm as is commented out above, but avoids a
        # few unnecessary math operations

        a      = los_unsquashed.dot(los_unsquashed)
        b_div2 = los_unsquashed.dot(obs_unsquashed)
        c      = obs_unsquashed.dot(obs_unsquashed) - self.req_sq
        d_div4 = b_div2**2 - a * c

        bsign_sqrtd_div2 = b_div2.sign() * d_div4.sqrt()
        t = (bsign_sqrtd_div2 - b_div2) / a
        pos = obs + t*los
        pos = self._apply_exclusion(pos)

        if derivs:
            # Using step-by-step differentiation of the equations above

            # da_dlos = 2 * los * self.unsquash_sq
            # db_dlos = 2 * obs * self.unsquash_sq
            # db_dobs = 2 * los * self.unsquash_sq
            # dc_dobs = 2 * obs * self.unsquash_sq

            da_dlos_div2 = los.element_mul(self.unsquash_sq)
            db_dlos_div2 = obs.element_mul(self.unsquash_sq)
            db_dobs_div2 = los.element_mul(self.unsquash_sq)
            dc_dobs_div2 = obs.element_mul(self.unsquash_sq)

            # dd_dlos = 2 * b * db_dlos - 4 * c * da_dlos
            # dd_dobs = 2 * b * db_dobs - 4 * a * dc_dobs

            dd_dlos_div8 = b_div2 * db_dlos_div2 - c * da_dlos_div2
            dd_dobs_div8 = b_div2 * db_dobs_div2 - a * dc_dobs_div2

            # dsqrt = d.sqrt()
            # d_dsqrt_dd = 0.5 / dsqrt
            # d_dsqrt_dlos = d_dsqrt_dd * dd_dlos
            # d_dsqrt_dobs = d_dsqrt_dd * dd_dobs

            # d[bsign_sqrtd]/d[x] = 1/2 / bsign_sqrtd * d[d]/d[x]
            #                     = 1/4 / bsign_sqrtd_div2 * d[d]/d[x]

            d_bsign_sqrtd_dlos_div2 = dd_dlos_div8 / bsign_sqrtd_div2
            d_bsign_sqrtd_dobs_div2 = dd_dobs_div8 / bsign_sqrtd_div2

            # inv2a = 0.5/a
            # d_inv2a_da = -2 * inv2a**2
            # 
            # dt_dlos = (inv2a * (b.sign()*d_dsqrt_dlos - db_dlos) +
            #           (b.sign()*dsqrt - b)*d_inv2a_da * da_dlos).as_vectorn()
            # dt_dobs = (inv2a * (b.sign()*d_dsqrt_dobs - db_dobs)).as_vectorn()
            # 
            # dpos_dobs = los.as_column() * dt_dobs.as_row() + MatrixN.UNIT33
            # dpos_dlos = los.as_column() * dt_dlos.as_row() + MatrixN.UNIT33*t

            dt_dlos = ((d_bsign_sqrtd_dlos_div2
                        - db_dlos_div2 - 2 * t * da_dlos_div2) / a)
            dt_dobs = ((d_bsign_sqrtd_dobs_div2
                        - db_dobs_div2) / a)

            dpos_dobs = los.as_column() * dt_dobs.as_row() + Matrix.UNIT33
            dpos_dlos = los.as_column() * dt_dlos.as_row() + Matrix.UNIT33 * t

            los_norm = los.norm()
            pos.insert_subfield("d_dobs", dpos_dobs)
            pos.insert_subfield("d_dlos", dpos_dlos * los_norm)
            t.insert_subfield("d_dobs", dt_dobs.as_row())
            t.insert_subfield("d_dlos", dt_dlos.as_row() * los_norm)

        return (pos, t)

    def normal(self, pos, derivs=False):
        """Returns the normal vector at a position at or near a surface.

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

        perp = Vector3.as_vector3(pos).element_mul(self.unsquash_sq)

        if derivs:
            perp.insert_subfield("d_dpos", self.unsquash_sq_2d)

        return perp

    def intercept_with_normal(self, normal, derivs=False):
        """Constructs the intercept point on the surface where the normal vector
        is parallel to the given vector.

        Input:
            normal      a Vector3 of normal vectors, with optional units.
            derivs      true to return a matrix of partial derivatives.

        Return:         a unitless Vector3 of surface intercept points, in km.
                        Where no solution exists, the components of the returned
                        vector should be masked.

                        If derivs is True, then the returned intercept points
                        have a subfield "d_dperp", which contains the partial
                        derivatives with respect to components of the normal
                        vector, as a MatrixN object with item shape [3,3].
        """

        cept = (Vector3.as_standard(normal) * self.squash).unit() * self.radii

        if derivs:
            raise NotImplementedError("Spheroid.intercept_with_normal() " +
                                      "does not implement derivatives")

        return cept

    def intercept_normal_to(self, pos, derivs=False):
        """Constructs the intercept point on the surface where a normal vector
        passes through a given position.

        Input:
            pos         a Vector3 of positions near the surface, with optional
                        units.
            derivs      true to return a matrix of partial derivatives
                        d(intercept)/d(pos).

        Return:         a unitless vector3 of surface intercept points. Where no
                        solution exists, the returned vector should be masked.

                        If derivs is True, then the returned intercept points
                        have a subfield "d_dpos", which contains the partial
                        derivatives with respect to components of the given
                        position vector, as a MatrixN object with item shape
                        [3,3].
        """

        return self.intercept_normal_to_iterated(pos, derivs)[0]

    def intercept_normal_to_iterated(self, pos, derivs=False, t_guess=None):
        """This is the same as above but allows an initial guess at the t array
        to be passed in, and returns a tuple containing both the intercept and
        the values of t. Also used by the Limb class.

        Input:
            pos         a Vector3 of positions near the surface, with optional
                        units.
            derivs      true to return a matrix of partial derivatives
                        d(intercept)/d(pos).
            t_guess     initial guess at the t array. This is the scalar such
                        that
                            intercept + t * perp(intercept) = pos

        Return:         a tuple (pos,t):
            intercept   a unitless vector3 of surface intercept points. Where no
                        solution exists, the returned vector should be masked.
            t           the scale factor such that:
                            cept + t * normal(cept) = pos
        """

        pos = Vector3.as_standard(pos)
        pos = self._apply_exclusion(pos)

        # The intercept point satisfies:
        #   cept + t * perp(cept) = pos
        # where
        #   perp(cept) = cept * unsquash_sq
        #
        # Let C1 == unsquash_z
        # Let C2 == unsquash_z**2
        #
        # Expanding,
        #   pos_x = (1 + t) cept_x
        #   pos_y = (1 + t) cept_y
        #   pos_z = (1 + C2 t) cept_z
        #
        # The intercept point must also satisfy
        #   |cept * unsquash| = req
        # or
        #   cept_x**2 + cept_y**2 + C2 cept_z**2 = req_sq
        #
        # Solve:
        #   cept_x**2 + cept_y**2 + C2 cept_z**2 - req_sq = 0
        #
        #   pos_x**2 / (1 + t)**2 +
        #   pos_y**2 / (1 + t)**2 +
        #   pos_z**2 / (1 + C2 t)**2 * C2 - req_sq = 0
        #
        # f(t) = the above expression
        #
        # df/dt = -2 pos_x**2 / (1 + t)**3 +
        #       = -2 pos_y**2 / (1 + t)**3 +
        #       = -2 pos_z**2 / (1 + C2 t)**3 C2**2
        #
        # Let denom = [1 + t, 1 + t, 1 + C2 t]
        # Let unsquash = [1, 1, C1]
        # Let unsquash_sq = [1, 1, C2]
        #
        # f(t) = (pos * scale) dot (pos * scale) - req_sq
        #
        # df/dt = -2 (pos * scale) dot (pos * scale**2)

        # Make an initial guess at t, if necessary
        if t_guess is None:
            cept = (pos * self.unsquash).unit() * self.radii
            t = (pos - cept).norm() / self.normal(cept).norm()
        else:
            t = t_guess.copy(False)

        # Terminate when accuracy stops improving by at least a factor of 2
        max_dt = 1.e99
        for iter in range(SURFACE_PHOTONS.max_iterations):
            denom = Vector3.ONES + t * self.unsquash_sq
            pos_scale = pos * self.unsquash / denom
            f = pos_scale.dot(pos_scale) - self.req_sq
            df_dt_div_neg2 = pos_scale.dot(pos_scale * self.unsquash_sq/denom)

            dt = -0.5 * f/df_dt_div_neg2
            t -= dt

            prev_max_dt = max_dt
            max_dt = abs(dt).max()

            if LOGGING.surface_iterations or Spheroid.DEBUG:
                print LOGGING.prefix, "Surface.spheroid.intercept_normal_to",
                print iter, max_dt

            if (np.all(Scalar.as_scalar(max_dt).mask) or
                max_dt <= SURFACE_PHOTONS.dlt_precision or
                max_dt >= prev_max_dt * 0.5): break

        denom = Vector3.ONES + t * self.unsquash_sq
        cept = pos / denom

        if derivs:
            # First, we need dt/dpos
            #
            # pos_x**2 / (1 + t)**2 +
            # pos_y**2 / (1 + t)**2 +
            # pos_z**2 / (1 + C2 t)**2 * C2 - req_sq = 0
            #
            # 2 pos_x / (1+t)**2
            #   + pos_x**2 * (-2)/(1+t)**3 dt/dpos_x
            #   + pos_y**2 * (-2)/(1+t)**3 dt/dpos_x
            #   + pos_z**2 * (-2)/(1+C2 t)**3 C2**2 dt/dpos_x = 0
            #
            # dt/dpos_x [(pos_x**2 + pos_y**2)/(1+t)**3 +
            #             pos_z**2 * C2**2/(1 + C2 t)**3] = pos_x / (1+t)**2
            #
            # Similar for dt/dpos_y
            #
            # (pos_x**2 + pos_y**2) * (-2)/(1+t)**3 dt/dpos_z
            #   + pos_z**2 * (-2)/(1 + C2 t)**3 C2**2 dt/dpos_z
            #   + 2 pos_z/(1 + C2 t)**2 C2 = 0
            #
            # dt/dpos_z [(pos_x**2 + pos_y**2) / (1+t)**3 +
            #             pos_z**2*C2**2/(1 + C2 t)**3] = pos_z C2/(1 + C2 t)**2
            #
            # Let denom = [1 + t, 1 + t, 1 + C2 t]
            # Let unsquash_sq = [1, 1, C2]
            #
            # Let denom1 = [(pos_x**2 + pos_y**2)/(1+t)**3 +
            #                pos_z**2 * C2**2 / (1 + C2 t)**3]
            # in the expressions for dt/dpos. Note that this is identical to
            # df_dt_div_neg2 in the expressions above.
            #
            # dt/dpos_x * denom1 = pos_x  / (1+t)**2
            # dt/dpos_y * denom1 = pos_y  / (1+t)**2
            # dt/dpos_z * denom1 = pos_z  * C2 / (1 + C2 t)**2

            dt_dpos = pos * self.unsquash_sq / (denom**2 * df_dt_div_neg2)
            dt_dpos = dt_dpos.as_row()

            # Now we can proceed with dcept/dpos
            #
            # cept + perp(cept) * t = pos
            #
            # dcept/dpos + perp(cept) dt/dpos + t dperp/dcept dcept/dpos = I
            #
            # (I + t dperp/dcept) dcept/dpos = I - perp(cept) dt/dpos
            #
            # dcept/dpos = (I + t dperp/dcept)**(-1) * (I - perp dt/dpos)

            perp = self.normal(cept, derivs=True)
            dperp_dcept = perp.d_dpos
            perp = perp.plain()

            # Note that (I + t dperp/dcept) is diagonal!
            scale = MatrixN.UNIT33 + t * dperp_dcept
            scale.vals[...,0,0] = 1 / scale.vals[...,0,0]
            scale.vals[...,1,1] = 1 / scale.vals[...,1,1]
            scale.vals[...,2,2] = 1 / scale.vals[...,2,2]

            dcept_dpos = scale * (MatrixN.UNIT33 - perp.as_column() * dt_dpos)

            t.insert_subfield("d_dpos", dt_dpos)
            cept.insert_subfield("d_dpos", dcept_dpos)

        return (cept, t)

    def _apply_exclusion(self, pos):
        """This internal method is used by intercept_normal_to() to exclude any
        positions that fall too close to the center of the surface. The math
        is poorly-behaved in this region.

        (1) It sets the mask on any of these points to True.
        (2) It sets the magnitude of any of these points to the edge of the
            exclusion zone, in order to avoid runtime errors in the math
            libraries.
        """

        # Define the exclusion zone
        pos.vals[np.isnan(pos.vals)] = 0.

        pos_unsquashed = pos.element_mul(self.unsquash)
        pos_sq_vals = pos_unsquashed.dot(pos_unsquashed).vals
        mask = (pos_sq_vals <= self.exclusion**2)

        if not np.any(mask): return pos

        # Suppress any exact zeros
        mask2 = (pos_sq_vals == 0)
        if np.any(mask2):

            # For the scalar case, replace with a nonzero vector and repeat
            if mask2 is True:
                return self._mask_pos(Vector3((0,0,self.exclusion/2.)))

            # Otherwise, just replace the zero values
            pos_sq_vals[mask][2] = self.exclusion**2/4.

        # Scale all masked vectors out the exclusion radius
        factor = self.exclusion / np.sqrt(pos_sq_vals)
        pos.vals[mask] *= factor[mask][...,np.newaxis]
        pos.mask = pos.mask | mask

        return pos

    ############################################################################
    # Latitude conversions
    ############################################################################

    def lat_to_centric(self, lat):
        """Converts a latitude value given in internal spheroid coordinates to
        its planetocentric equivalent.
        """

        return (lat.tan() * self.squash_z).arctan()

    def lat_to_graphic(self, lat):
        """Converts a latitude value given in internal spheroid coordinates to
        its planetographic equivalent.
        """

        return (lat.tan() * self.unsquash_z).arctan()

    def lat_from_centric(self, lat):
        """Converts a latitude value given in planetocentric coordinates to its
        equivalent value in internal spheroid coordinates.
        """

        return (lat.tan() * self.unsquash_z).arctan()

    def lat_from_graphic(self, lat):
        """Converts a latitude value given in planetographic coordinates to its
        equivalent value in internal spheroid coordinates.
        """

        return (lat.tan() * self.squash_z).arctan()

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Spheroid(unittest.TestCase):

    def runTest(self):

        from oops.frame_.frame import Frame
        from oops.path_.path import Path

        REQ  = 60268.
        #RPOL = 54364.
        RPOL = 50000.
        planet = Spheroid("SSB", "J2000", (REQ, RPOL))

        # Coordinate/vector conversions
        NPTS = 10000
        pos = (2 * np.random.rand(NPTS,3) - 1.) * REQ   # range is -REQ to REQ

        (lon,lat,elev) = planet.coords_from_vector3(pos, axes=3)
        test = planet.vector3_from_coords((lon,lat,elev))
        self.assertTrue(((test - pos).rms() < 1.e-8).all())

        # Spheroid intercepts & normals
        obs = REQ * (np.random.rand(NPTS,3) + 1.)       # range is REQ to 2*REQ
        los = -np.random.rand(NPTS,3)                   # range is -1 to 0

        (pts, t) = planet.intercept(obs, los)
        test = t * Vector3(los) + Vector3(obs)
        self.assertTrue(((test - pts).rms().mvals < 1.e-9).all())

        self.assertTrue(np.all(t.mask == pts.mask))
        self.assertTrue(np.all(pts.mask[t.vals < 0.]))

        normals = planet.normal(pts)

        pts.vals[...,2] *= REQ/RPOL
        self.assertTrue(((pts.norm()[np.logical_not(pts.mask)] - REQ).rms() < 1.e-8).all())

        normals.vals[...,2] *= RPOL/REQ
        self.assertTrue(((normals.unit() - pts.unit()).rms().mvals < 1.e-14).all())

        # Intercept derivatives

        # Lines of sight with grazing incidence can have large numerical errors,
        # but this is not to be considered an error in the analytic calculation.
        # As a unit test, we ignore the largest 3% of the errors, but require
        # that the rest of the errors be very small.
        eps = 1.
        frac = 0.97     # Ignore errors above this cutoff
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (cept,t) = planet.intercept(obs, los, derivs=True)
            (cept1,t1) = planet.intercept(obs + dobs[i], los, derivs=False)
            (cept2,t2) = planet.intercept(obs - dobs[i], los, derivs=False)

            dcept_dobs = (cept1 - cept2) / (2*eps)
            ref = cept.d_dobs.as_column(i).as_vector3()

            errors = abs(dcept_dobs - ref) / abs(ref)
            sorted = np.sort(errors.vals[np.logical_not(errors.mask)])
                        # mask=True where the line of sight missed the surface
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

            dt_dobs = (t1 - t2) / (2*eps)
            ref = t.d_dobs.vals[...,0,i]

            errors = abs(dt_dobs/ref - 1)
            sorted = np.sort(errors.vals[np.logical_not(errors.mask)])
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

        eps = 1.e-6
        frac = 0.97
        dlos = ((eps,0,0), (0,eps,0), (0,0,eps))
        norms = np.sqrt(los[...,0]**2 + los[...,1]**2 + los[...,2]**2)
        los /= norms[..., np.newaxis]
        for i in range(3):
            (cept,t) = planet.intercept(obs, los, derivs=True)
            (cept1,t1) = planet.intercept(obs, los + dlos[i], derivs=False)
            (cept2,t2) = planet.intercept(obs, los - dlos[i], derivs=False)

            dcept_dlos = (cept1 - cept2) / (2*eps)
            ref = cept.d_dlos.as_column(i).as_vector3()

            errors = abs(dcept_dlos - ref) / abs(ref)
            sorted = np.sort(errors.vals[np.logical_not(errors.mask)])
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

            dt_dlos = (t1 - t2) / (2*eps)
            ref = t.d_dlos.vals[...,0,i]

            errors = abs(dt_dlos/ref - 1)
            sorted = np.sort(errors.vals[np.logical_not(errors.mask)])
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

        # Test normal()
        cept = Vector3(np.random.random((100,3))).unit() * planet.radii
        perp = planet.normal(cept)
        test1 = (cept * planet.unsquash).unit()
        test2 = (perp * planet.squash).unit()

        self.assertTrue(abs(test1 - test2) < 1.e-12)

        eps = 1.e-7
        (lon,lat) = planet.coords_from_vector3(cept, axes=2)
        cept1 = planet.vector3_from_coords((lon+eps,lat,0.))
        cept2 = planet.vector3_from_coords((lon-eps,lat,0.))

        self.assertTrue(abs((cept2 - cept1).sep(perp) - np.pi/2) < 1.e-8)

        (lon,lat) = planet.coords_from_vector3(cept, axes=2)
        cept1 = planet.vector3_from_coords((lon,lat+eps,0.))
        cept2 = planet.vector3_from_coords((lon,lat-eps,0.))

        self.assertTrue(abs((cept2 - cept1).sep(perp) - np.pi/2) < 1.e-8)

        # Test intercept_with_normal()
        vector = Vector3(np.random.random((100,3)))
        cept = planet.intercept_with_normal(vector)
        sep = vector.sep(planet.normal(cept))
        self.assertTrue(sep < 1.e-14)

        # Test intercept_normal_to()
        pos = Vector3(np.random.random((100,3)) * 4.*REQ + REQ)
        cept = planet.intercept_normal_to(pos)
        sep = (pos - cept).sep(planet.normal(cept))
        self.assertTrue(sep < 3.e-12)
        self.assertTrue(abs((cept*planet.unsquash).norm() - planet.req) < 1.e-6)

        # Test normal() derivative
        cept = Vector3(np.random.random((100,3))).unit() * planet.radii
        perp = planet.normal(cept, derivs=True)
        eps = 1.e-5
        dpos = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            perp1 = planet.normal(cept + dpos[i])
            dperp_dpos = (perp1 - perp) / eps

            self.assertTrue(abs(dperp_dpos - perp.d_dpos.as_row(i)) < 1.e-4)

        # Test intercept_normal_to() derivative
        pos = Vector3(np.random.random((3,3)) * 4.*REQ + REQ)
        (cept,t) = planet.intercept_normal_to_iterated(pos, derivs=True)
        self.assertTrue(abs((cept*planet.unsquash).norm() - planet.req) < 1.e-6)

        eps = 1.
        dpos = ((eps,0,0), (0,eps,0), (0,0,eps))
        perp = planet.normal(cept)
        for i in range(3):
            (cept1,t1) = planet.intercept_normal_to_iterated(pos + dpos[i],
                                                             derivs=False,
                                                             t_guess=t.plain())
            (cept2,t2) = planet.intercept_normal_to_iterated(pos - dpos[i],
                                                             derivs=False,
                                                             t_guess=t.plain())
            dcept_dpos = (cept1 - cept2) / (2*eps)
            self.assertTrue(abs(dcept_dpos.sep(perp) - np.pi/2) < 1.e-5)

            ref = cept.d_dpos.as_column(i).as_vector3()
            self.assertTrue(abs(dcept_dpos - ref) < 1.e-5)

            dt_dpos = (t1 - t2) / (2*eps)
            ref = t.d_dpos.vals[...,0,i]
            self.assertTrue(abs(dt_dpos/ref - 1) < 1.e-5)

        registry.initialize()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
