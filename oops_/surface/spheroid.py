################################################################################
# oops_/surface/spheroid.py: Spheroid subclass of class Surface
#
# 2/15/12 Checked in (BSW)
# 2/17/12 Modified (MRS) - Inserted coordinate definitions; added use of trig
#   functions and sqrt() defined in Scalar class to enable cleaner algorithms.
#   Unit tests added.
# 3/4/12 MRS: cleaned up comments, added NotImplementedErrors for features still
#   TBD.
# 3/29/12 MRS: added derivatives to intercept_normal_to() and support for new
#   Limb surface. Lots of new unit tests.
################################################################################

import numpy as np

from oops_.surface.surface_ import Surface
from oops_.array.all import *
import oops_.registry as registry

class Spheroid(Surface):
    """Spheroid defines a spheroidal surface centered on the given path and
    fixed with respect to the given frame. The short radius of the spheroid is
    oriented along the Z-axis of the frame.

    The coordinates defining the surface grid are (longitude, latitude), based
    on the assumption that a spherical body has been "squashed" along the
    Z-axis. The latitude defined in this manner is neither planetocentric nor
    planetographic; functions are provided to perform the conversion to either
    choice. Longitudes are measured in a right-handed manner, increasing toward
    the east. Values range from 0 to 2*pi.

    Elevations are defined by "unsquashing" the radial vectors and then
    subtracting off the equatorial radius of the body. Thus, the surface is
    defined as the locus of points where elevation equals zero. However, the
    direction of increasing elevation is not exactly normal to the surface.
    """

    UNIT_MATRIX = MatrixN([(1,0,0),(0,1,0),(0,0,1)])
    ONES_VECTOR = Vector3([1,1,1])

    COORDINATE_TYPE = "spherical"

    DEBUG = False       # True for convergence testing in intercept_normal_to()

    def __init__(self, origin, frame, radii, exclusion=0.95):
        """Constructor for a Spheroid surface.

        Input:
            origin      the Path object or ID defining the center of the
                        spheroid.
            frame       the Frame object or ID defining the coordinate frame in
                        which the spheroid is fixed, with the short axis along
                        the Z-coordinate.
            radii       a tuple (a,c), defining the long and short radii of the
                        spheroid.
            exclusion   the fraction of the polar radius within which
                        calculations of intercept_normal_to() are suppressed.
                        Values of less than 0.9 are not recommended because
                        the problem becomes numerically unstable.
        """

        self.origin_id = registry.as_path_id(origin)
        self.frame_id  = registry.as_frame_id(frame)

        self.radii  = np.array((radii[0], radii[0], radii[1]))
        self.radii_sq = self.radii**2
        self.req    = radii[0]
        self.req_sq = self.req**2
        self.rpol   = self.radii[2]

        self.squash_z   = radii[1] / radii[0]
        self.unsquash_z = radii[0] / radii[1]

        self.squash    = Vector3((1., 1., self.squash_z))
        self.squash_sq = Vector3((1., 1., self.squash_z**2))

        self.unsquash     = Vector3((1., 1., self.unsquash_z))
        self.unsquash_sq  = Vector3((1., 1., self.unsquash_z**2))

        self.unsquash_sq_2d = MatrixN(([1,0,0],
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

        unsquashed = Vector3.as_standard(pos) * self.unsquash

        r = unsquashed.norm()
        (x,y,z) = unsquashed.as_scalars()
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
        lon = Scalar.as_standard(coords[0])
        lat = Scalar.as_standard(coords[1])

        if len(coords) == 2:
            r = Scalar(0.)
        else:
            r = Scalar.as_standard(coords[2]) + self.req

        r_coslat = r * lat.cos()
        x = r_coslat * lon.cos()
        y = r_coslat * lon.sin()
        z = r * lat.sin() * self.squash_z

        pos = Vector3.from_scalars(x,y,z)

        if derivs:
            raise NotImplementedError("Spheroid.vector3_from_coords() " +
                                      " does not implement derivatives")

        return pos

    def intercept(self, obs, los, derivs=False):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.

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
        obs = Vector3.as_standard(obs)
        los = Vector3.as_standard(los)

        obs_unsquashed = Vector3.as_standard(obs) * self.unsquash
        los_unsquashed = Vector3.as_standard(los) * self.unsquash

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

            da_dlos_div2 = los * self.unsquash_sq
            db_dlos_div2 = obs * self.unsquash_sq
            db_dobs_div2 = los * self.unsquash_sq
            dc_dobs_div2 = obs * self.unsquash_sq

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
            # dpos_dobs = (los.as_column() * dt_dobs.as_row() +
            #              Spheroid.UNIT_MATRIX)
            # dpos_dlos = (los.as_column() * dt_dlos.as_row() +
            #              Spheroid.UNIT_MATRIX * t)

            dt_dlos = ((d_bsign_sqrtd_dlos_div2
                        - db_dlos_div2 - 2 * t * da_dlos_div2) / a).as_vectorn()
            dt_dobs = ((d_bsign_sqrtd_dobs_div2
                        - db_dobs_div2) / a).as_vectorn()

            dpos_dobs = (los.as_column() * dt_dobs.as_row() +
                         Spheroid.UNIT_MATRIX)
            dpos_dlos = (los.as_column() * dt_dlos.as_row() +
                         Spheroid.UNIT_MATRIX * t)

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

        perp = Vector3.as_standard(pos) * self.unsquash_sq

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

    def intercept_normal_to_iterated(self, pos, derivs=False, guess=None):
        """This is the same as above but allows an initial guess at the t array
        to be passed in, and returns a tuple containing both the intercept and
        the values of t.

        Input:
            pos         a Vector3 of positions near the surface, with optional
                        units.
            derivs      true to return a matrix of partial derivatives
                        d(intercept)/d(pos).
            guess       initial guess at the t array.

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
        # Let A1 == unsquash_z
        # Let A2 == unsquash_z**2
        #
        # Expanding,
        #   pos_x = (1 + t) cept_x
        #   pos_y = (1 + t) cept_y
        #   pos_z = (1 + A2 t) cept_z
        #
        # The intercept point must also satisfy
        #   |cept * unsquash| = req
        # or
        #   cept_x**2 + cept_y**2 + A2 cept_z**2 = req_sq
        #
        # Solve:
        #   cept_x**2 + cept_y**2 + A2 cept_z**2 - req_sq = 0
        #
        #   pos_x**2 / (1 + t)**2 +
        #   pos_y**2 / (1 + t)**2 +
        #   pos_z**2 / (1 + A2 t)**2 * A2 - req_sq = 0
        #
        # f(t) = the above expression
        #
        # df/dt = -2 pos_x**2 / (1 + t)**3 +
        #       = -2 pos_y**2 / (1 + t)**3 +
        #       = -2 pos_z**2 / (1 + A2 t)**3 A2**2
        #
        # Let denom = [1 + t, 1 + t, 1 + A2 t]
        # Let unsquash = [1, 1, A1]
        # Let unsquash_sq = [1, 1, A2]
        #
        # f(t) = (pos * scale) dot (pos * scale) - req_sq
        #
        # df/dt = -2 (pos * scale) dot (pos * scale**2)

        # Make an initial guess at t, if necessary
        if guess is None:
            cept = (pos * self.unsquash).unit() * self.radii
            t = (pos - cept).norm() / self.normal(cept).norm()
        else:
            t = guess.copy(False)

        # Terminate when accuracy stops improving by at least a factor of 2
        prev_max_dt = 3.e99
        max_dt = 1.e99

        if Spheroid.DEBUG: print "SPHEROID START"

        while (max_dt < prev_max_dt * 0.5 or max_dt > 1.e-3):
            denom = Spheroid.ONES_VECTOR + t * self.unsquash_sq
            pos_scale = pos * self.unsquash / denom
            f = pos_scale.dot(pos_scale) - self.req_sq
            df_dt_div_neg2 = pos_scale.dot(pos_scale * self.unsquash_sq/denom)

            dt = -0.5 * f/df_dt_div_neg2
            t -= dt

            prev_max_dt = max_dt
            max_dt = abs(dt).max()
            if Spheroid.DEBUG: print "SPHEROID", max_dt, np.sum(t.mask)

        denom = Spheroid.ONES_VECTOR + t * self.unsquash_sq
        cept = pos / denom

        if derivs:
            # First, we need dt/dpos
            #
            # pos_x**2 / (1 + t)**2 +
            # pos_y**2 / (1 + t)**2 +
            # pos_z**2 / (1 + A2 t)**2 * A2 - req_sq = 0
            #
            # 2 pos_x / (1+t)**2
            #   + pos_x**2 * (-2)/(1+t)**3 dt/dpos_x
            #   + pos_y**2 * (-2)/(1+t)**3 dt/dpos_x
            #   + pos_z**2 * (-2)/(1+A2 t)**3 A2**2 dt/dpos_x = 0
            #
            # dt/dpos_x [(pos_x**2 + pos_y**2)/(1+t)**3 +
            #             pos_z**2 * A2**2/(1 + A2 t)**3] = pos_x / (1+t)**2
            #
            # Similar for dt/dpos_y
            #
            # (pos_x**2 + pos_y**2) * (-2)/(1+t)**3 dt/dpos_z
            #   + pos_z**2 * (-2)/(1 + A2 t)**3 A2**2 dt/dpos_z
            #   + 2 pos_z/(1 + A2 t)**2 A2 = 0
            #
            # dt/dpos_z [(pos_x**2 + pos_y**2) / (1+t)**3 +
            #             pos_z**2*A2**2/(1 + A2 t)**3] = pos_z A2/(1 + A2 t)**2
            #
            # Let denom = [1 + t, 1 + t, 1 + A2 t]
            # Let unsquash_sq = [1, 1, A2]
            #
            # Let denom1 = [(pos_x**2 + pos_y**2)/(1+t)**3 +
            #                pos_z**2 * A2**2 / (1 + A2 t)**3]
            # in the expressions for dt/dpos. Note that this is identical to
            # df_dt_div_neg2 in the expressions above.
            #
            # dt/dpos_x * denom1 = pos_x  / (1+t)**2
            # dt/dpos_y * denom1 = pos_y  / (1+t)**2
            # dt/dpos_z * denom1 = pos_z  * A2 / (1 + A2 t)**2

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
            scale = Spheroid.UNIT_MATRIX + t * dperp_dcept
            scale.vals[...,0,0] = 1 / scale.vals[...,0,0]
            scale.vals[...,1,1] = 1 / scale.vals[...,1,1]
            scale.vals[...,2,2] = 1 / scale.vals[...,2,2]

            dcept_dpos = scale * (Spheroid.UNIT_MATRIX
                                                - perp.as_column() * dt_dpos)

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

        pos_unsquashed = pos * self.unsquash
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

        from oops_.frame.frame_ import Frame
        from oops_.path.path_ import Path

        REQ  = 60268.
        RPOL = 54364.
        RPOL = 50000.
        planet = Spheroid("SSB", "J2000", (REQ, RPOL))

        # Coordinate/vector conversions
        NPTS = 10000
        obs = (2 * np.random.rand(NPTS,3) - 1.) * REQ

        (lon,lat,elev) = planet.coords_from_vector3(obs,axes=3)
        test = planet.vector3_from_coords((lon,lat,elev))
        self.assertTrue(abs(test - obs) < 1.e-8)

        # Spheroid intercepts & normals
        obs[...,0] = np.abs(obs[...,0])
        obs[...,0] += REQ

        los = (2 * np.random.rand(NPTS,3) - 1.)
        los[...,0] = -np.abs(los[...,0])

        (pts, t) = planet.intercept(obs, los)
        test = t * Vector3(los) + Vector3(obs)
        self.assertTrue(abs(test - pts) < 1.e-9)

        self.assertTrue(np.all(t.mask == pts.mask))
        self.assertTrue(np.all(pts.mask[t.vals < 0.]))

        normals = planet.normal(pts)

        pts.vals[...,2] *= REQ/RPOL
        self.assertTrue(abs(pts.norm()[~pts.mask] - REQ) < 1.e-8)

        normals.vals[...,2] *= RPOL/REQ
        self.assertTrue(abs(normals.unit() - pts.unit()) < 1.e-14)

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
        pos = Vector3(np.random.random((100,3)) * 4.*REQ + REQ)
        (cept,t) = planet.intercept_normal_to_iterated(pos, derivs=True)
        self.assertTrue(abs((cept*planet.unsquash).norm() - planet.req) < 1.e-6)

        eps = 1.
        dpos = ((eps,0,0), (0,eps,0), (0,0,eps))
        perp = planet.normal(cept)
        for i in range(3):
            (cept1,t1) = planet.intercept_normal_to_iterated(pos + dpos[i],
                                                             False, t.plain())
            (cept2,t2) = planet.intercept_normal_to_iterated(pos - dpos[i],
                                                             False, t.plain())
            dcept_dpos = (cept1 - cept2) / (2*eps)
            ref = cept.d_dpos.as_column(i).as_vector3()

            self.assertTrue(abs(dcept_dpos.sep(perp) - np.pi/2) < 1.e-5)
            self.assertTrue(abs(dcept_dpos - ref) < 1.e-5)

            dt_dpos = (t1 - t2) / (2*eps)
            ref = t.d_dpos.vals[...,0,i]
            self.assertTrue(abs(dt_dpos/ref - 1) < 1.e-5)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
