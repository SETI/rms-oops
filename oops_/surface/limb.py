################################################################################
# oops_/surface/limb.py: Limb subclass of class Surface
#
# 3/29/12 MRS: New, with initial unit tests of intercept().
################################################################################

import numpy as np

from oops_.surface.surface_ import Surface
from oops_.surface.spheroid import Spheroid
from oops_.surface.ellipsoid import Ellipsoid
from oops_.array.all import *
import oops_.registry as registry

class Limb(Surface):
    """The Limb surface is defined as the locus of points where a surface normal
    from a spheroid or ellipsoid is perpendicular to the line of sight. This 
    provides a convenient coordinate system for describing cloud features on the
    limb of a body. The coordinates are (z,clock) where
        z       vertical distance above the surface at the visible limb.
        clock   a clock angle measured from the projected north pole of the body
                in the clockwise direction.
        dist    a distance offset along the line of sight, where 0 corresponds
                to the limb point and positive values are more distant.
    """

    UNIT_MATRIX = MatrixN([(1,0,0),(0,1,0),(0,0,1)])

    COORDINATE_TYPE = "limb"

    DEBUG = False   # Set to True for convergence testing in intercept()

    def __init__(self, origin, frame, radii, exclusion=0.95):
        """Constructor for a Limb surface.

        Input:
            origin      the Path object or ID defining the center of the
                        spheroid.
            frame       the Frame object or ID defining the coordinate frame in
                        which the spheroid is fixed, with the short axis along
                        the Z-coordinate.
            radii       a tuple (a,b,c), defining the long and short radii of
                        the spheroid or ellipsoid.
            exclusion   the fraction of the polar radius within which
                        calculations of the surface are suppressed. Values of
                        less than 0.9 are not recommended because the problem
                        becomes numerically unstable.
        """

        self.origin_id = registry.as_path_id(origin)
        self.frame_id  = registry.as_frame_id(frame)

        self.radii = np.array(radii)
        self.req   = radii[0]
        self.rpol  = radii[-1]

        if radii[0] == radii[1]:
            self.ground = Spheroid(origin, frame, self.radii[1:],
                                                  exclusion=exclusion)
        else:
            self.ground = Ellipsoid(origin, frame, self.radii,
                                                   exclusion=exclusion)

        # This defines the boundary of the inner exclusion zone, where the limb
        # geometry is poorly defined. We define this region as follows:
        #
        # (1) "Unsquash" the geometry so that the body is spherical
        # (2) Exlude the central sphere of radius equal to the equatorial radius
        #     minus the polar radius.
        # (3) Re-apply the squash to restore the geometry. The excluded zone
        #     becomes similarly squashed.

        self.unsquash = self.ground.unsquash
        self.squash = self.ground.squash

    def coords_from_vector3(self, pos, obs=None, axes=2, derivs=False):
        """Converts position vectors in the internal frame into the surface
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

        pass

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

        pass

    def intercept(self, obs, los, derivs=False, groundtrack=False):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.
            groundtrack True to include the surface intercept points of the body
                        associated with each limb intercept. This array can
                        speed up any subsequent calculations such as calls to
                        normal(), and can be used to determine locations in
                        body coordinates.

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

                        If groundtrack is True, then pos contains a subfield
                        "groundtrack" that contains the associated points on
                        the body surface as a Vector3.
        """

        # Convert to standard units
        obs = Vector3.as_standard(obs)
        los = Vector3.as_standard(los)

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

        t = -obs.dot(los) / los.dot(los)

        prev_max_dt = 3.e99
        max_dt = 1.e98
        helper = None       # To speed up calls to intercept_normal_to()

        if Limb.DEBUG: print "LIMB START"
        while (max_dt < prev_max_dt * 0.5 or max_dt > 1.e-3):
            pos = obs + t * los

            (cept, helper) = self.ground.intercept_normal_to_iterated(pos,
                                                     derivs=True, guess=helper)
            perp = self.ground.normal(cept, derivs=True)

            df_dt = (perp.d_dpos * cept.d_dpos * los).dot(los)
            dt = perp.plain().dot(los) / df_dt

            t = t - dt

            prev_max_dt = max_dt
            max_dt = abs(dt).max()
            if Limb.DEBUG: print "LIMB:", max_dt, np.sum(t.mask)

        pos = obs + t * los

        if groundtrack:
            cept = self.ground.intercept_normal_to_iterated(pos,guess=helper)[0]
            pos.insert_subfield("groundtrack", cept)

        if derivs:
            raise NotImplementedError("Limb.intercept() " +
                                      " does not implement derivatives")

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

        if "groundtrack" in pos.subfields.keys():
            groundtrack = pos.groundtrack
        else:
            groundtrack = self.ground.intercept_normal_to(pos)

        perp = self.ground.normal(groundtrack)

        if derivs:
            raise NotImplementedError("Limb.normal() " +
                                      "does not implement derivatives")

        return perp

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Limb(unittest.TestCase):

    def runTest(self):

        from oops_.frame.frame_ import Frame
        from oops_.path.path_ import Path

        REQ  = 60268.
        RPOL = 50000.
        limb = Limb("SSB", "J2000", (REQ, REQ, RPOL), exclusion=1)

        obs = Vector3(np.random.random((1000,3)) * 4.*REQ + REQ)
        los = Vector3(np.random.random((1000,3)))

        (cept,t) = limb.intercept(obs, los, groundtrack=True)

        self.assertTrue(abs(limb.normal(cept).sep(los) - np.pi/2) < 1.e-8)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
