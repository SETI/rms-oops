################################################################################
# oops_/surface/surface.py: Abstract class Surface
#
# 2/5/12 Modified (MRS) - Minor updates for style
# 3/1/12 MRS: Modified convergence criteria in _solve_photons(), added quick
#   dictionary and config file.
################################################################################

import numpy as np

import oops_.registry as registry
import oops_.constants as constants
from oops_.array.all import *
from oops_.config import QUICK, SURFACE_PHOTONS, LOGGING
from oops_.event import Event

class Surface(object):
    """Surface is an abstract class describing a 2-D object that moves and
    rotates in space. A surface employs an internal coordinate system, not
    necessarily rectangular, in which two primary coordinates define locations
    on the surface, and an optional third coordinate can define points above or
    below that surface. The shape is always fixed."""

########################################
# Each subclass must override...
########################################

    def __init__(self):
        """Constructor for a Surface object.

        Every surface must have the following attributes:
            path_id     the ID of a Path object, which defines the motion of a
                        surface's origin point.
            frame_id    the ID of a Frame object, which defines the orientation
                        of the surface. The surface and its coordinates are
                        fixed within this frame.
        """

        pass

    def as_coords(self, position, axes=2, **keywords):
        """Converts from position vectors in the internal frame into the surface
        coordinate system.

        Input:
            position    a Vector3 of positions at or near the surface, with
                        optional units.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            **keywords  any additional keyword=value pairs needed in order to
                        perform the conversion from position to coordinates.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate. Values
                        are always in standard units.
        """

        pass

    def as_vector3(self, coord1, coord2, coord3=0., **keywords):
        """Converts coordinates in the surface's internal coordinate system into
        position vectors at or near the surface.

        Input:
            coord1      a Scalar of values for the first coordinate, with
                        optional units.
            coord2      a Scalar of values for the second coordinate, with
                        optional units.
            coord3      a Scalar of values for the third coordinate, with
                        optional units; default is Scalar(0.).
            **keywords  any additional keyword=value pairs needed in order to
                        perform the conversion from coordinates to position.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.

        Return:         the corresponding unitless Vector3 object of positions,
                        in km.
        """

        pass

    def intercept(self, obs, los, derivs=False):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.

        Return:         a tuple (position, factor) if derivs is False; a tuple
                        (position, factor, dpos_dobs, dpos_dlos) if derivs is
                        True.
            position    a unitless Vector3 of intercept points on the surface,
                        in km.
            factor      a unitless Scalar of factors such that:
                            position = obs + factor * los
            dpos_dobs   the partial derivatives of the position vector with
                        respect to the observer position, as a Matrix3.
            dpos_dlos   the partial derivatives of the position vector with
                        respect to the line of sight, as a Matrix3.
        """

        pass

    def normal(self, position):
        """Returns the normal vector at a position at or near a surface.

        Input:
            position    a Vector3 of positions at or near the surface, with
                        optional units.

        Return:         a unitless Vector3 containing directions normal to the
                        surface that pass through the position. Lengths are
                        arbitrary.
        """

        pass

    def gradient(self, position, axis=0, **keywords):
        """Returns the gradient vector at a specified position at or near the
        surface. The gradient of surface coordinate c is defined as a vector
            (dc/dx,dc/dy,dc/dz)
        It has the property that it points in the direction of the most rapid
        change in value of the coordinate, and its magnitude is the rate of
        change in that direction.

        Input:
            position    a Vector3 of positions at or near the surface, with
                        optional units.
            axis        0, 1 or 2, identifying the coordinate axis for which the
                        gradient is sought.
            **keywords  any other keyword=value pairs needed to define the
                        mapping between positions and coordinates.

        Return:         a unitless Vector3 of the gradients sought. Values are
                        always in standard units.
        """

        pass

    def velocity(self, position):
        """Returns the local velocity vector at a point within the surface.
        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            position    a Vector3 of positions at or near the surface, with
                        optional units.

        Return:         a unitless Vector3 of velocities, in units of km/s.
        """

        return Vector3((0,0,0))

    def intercept_with_normal(self, normal):
        """Constructs the intercept point on the surface where the normal vector
        is parallel to the given vector.

        Input:
            normal      a Vector3 of normal vectors, with optional units.

        Return:         a unitless Vector3 of surface intercept points, in km.
                        Where no solution exists, the components of the returned
                        vector should be masked.
        """

        pass

    def intercept_normal_to(self, position):
        """Constructs the intercept point on the surface where a normal vector
        passes through a given position.

        Input:
            position    a Vector3 of positions near the surface, with optional
                        units.

        Return:         a unitless vector3 of surface intercept points. Where no
                        solution exists, the returned vector should be masked.
        """

        pass

################################################################################
# Photon Solver
################################################################################

    def photon_from_event(self, event, quick=QUICK, derivs=False,
                                iters     = SURFACE_PHOTONS.max_iterations,
                                precision = SURFACE_PHOTONS.dlt_precision,
                                limit     = SURFACE_PHOTONS.dlt_limit):
        """Returns the photon arrival event at the body's surface, for photons
        departing earlier from the specified event.
        """

        return self._solve_photon(event, +1, quick, derivs,
                                         iters, precision, limit)

    def photon_to_event(self, event, quick=QUICK, derivs=False,
                              iters     = SURFACE_PHOTONS.max_iterations,
                              precision = SURFACE_PHOTONS.dlt_precision,
                              limit     = SURFACE_PHOTONS.dlt_limit):
        """Returns the photon departure event at the body's surface, for photons
        arriving later at the specified event. See _solve_photon() for details.
        """

        return self._solve_photon(event, -1, quick, derivs,
                                         iters, precision, limit)

    def _solve_photon(self, event, sign, quick=QUICK, derivs=False,
                            iters     = SURFACE_PHOTONS.max_iterations,
                            precision = SURFACE_PHOTONS.dlt_precision,
                            limit     = SURFACE_PHOTONS.dlt_limit):
        """Solve for the Event object located on the body's surface that falls
        at the other end of the photon's path to or from another Event.

        Input:
            event       the reference Event of a photon's arrival or departure.

            sign        -1 to return earlier Events, corresponding to photons
                           departing from the surface and arriving later at the
                           reference event.
                        +1 to return later Events, corresponding to photons
                           departing from the reference event and arriving later
                           at the surface.

            quick       True to use QuickPaths and QuickFrames, where warranted,
                        to speed up the calculation.

            derivs      if True, the following subfields are added to the
                        event:
                           dtime_dpos   partial derivative of event time with
                                        respect to changes in the observer
                                        position, a MatrixN with item [1,3].
                           dpos_dpos    partial derivative of the event position
                                        with respect to changes in the observer
                                        position, a MatrixN with item [3,3].
                           dtime_dlos   partial derivative of event time with
                                        respect to changes in the line of sight
                                        vector (arriving or departing) as a
                                        MatrixN with item shape [1,3].
                           dpos_dlos    partial derivative of the event position
                                        with respect to changes in the line of
                                        sight vector (arriving or departing), as
                                        a MatrixN with item shape [3,3].
                        All quantities are given in the frame of the surface.

            The following input parameters have default defined in file
            oops_.config.SURFACE_PHOTONS:

            iters       the maximum number of iterations of Newton's method to
                        perform. It should almost never need to be > 5.
            precision   iteration stops when the largest change in light travel
                        time between one iteration and the next falls below this
                        threshold (in seconds).
            limit       the maximum allowed absolute value of the change in
                        light travel time from the nominal range calculated
                        initially. Changes in light travel with absolute values
                        larger than this limit are clipped. Can be used to
                        limit divergence of the solution in some rare cases.

        Return:         Returns the surface event describing the arrival or
                        departure of the photon. The surface normal and velocity
                        field subfields "perp" and "vflat" are always filled in.
                        The subfields (arr, arr_lt) or (dep, dep_lt) are filled
                        in for arriving or departing photons, respectively.
        """

        # Interpret the sign
        signed_c = sign * constants.C
        if sign < 0.:
            surface_key = "dep"
            event_key = "arr"
        else:
            event_key = "dep"
            surface_key = "arr" 

        # Define the surface path and frame relative to the SSB in J2000
        origin_wrt_ssb  = registry.connect_paths(self.origin_id, "SSB", "J2000")
        frame_wrt_j2000 = registry.connect_frames(self.frame_id, "J2000")

        # Define the observer and line of sight in the SSB frame
        event_wrt_ssb = event.wrt_ssb(quick)
        obs_wrt_ssb = event_wrt_ssb.pos
        los_wrt_ssb = event_wrt_ssb.subfields[event_key].unit() * constants.C

        # Make an initial guess at the light travel time using the range to the
        # surface's origin
        lt = (obs_wrt_ssb -
              origin_wrt_ssb.event_at_time(event.time).pos).norm() / signed_c
        lt_min = lt.min() - limit
        lt_max = lt.max() + limit

        # Interpret the quick parameters
        if quick is not False:
            loop_quick = {"path_extension": limit,
                          "frame_extension": limit}
            if type(quick) == type({}):
                loop_quick = dict(quick, **loop_quick)

        # Iterate a fixed number of times. Convergence is rapid because all
        # speeds are non-relativistic.
        max_dlt = np.inf
        for iter in range(iters):

            # Evaluate the current time
            surface_time = event.time + lt

            # Quicken when needed
            origin_wrt_ssb = origin_wrt_ssb.quick_path(surface_time,
                                                       quick=loop_quick)
            frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time,
                                                          quick=loop_quick)

            # Locate the photons relative to the current origin in SSB/J2000
            pos_in_j2000 = (obs_wrt_ssb + lt * los_wrt_ssb
                            - origin_wrt_ssb.event_at_time(surface_time).pos)

            # Rotate into the surface-fixed frame
            transform = frame_wrt_j2000.transform_at_time(surface_time)
            pos_in_frame = transform.rotate(pos_in_j2000)
            los_in_frame = transform.rotate(los_wrt_ssb)

            # Update the intercept times; save the intercept positions
            (intercept, dlt) = self.intercept(pos_in_frame, los_in_frame)
            new_lt = (lt + dlt).clip(lt_min, lt_max)
            dlt = new_lt - lt
            lt = new_lt

            # Test for convergence
            prev_max_dlt = max_dlt
            max_dlt = abs(dlt).max()

            if LOGGING.surface_iterations: print iter, max_dlt

            if max_dlt <= precision or max_dlt >= prev_max_dlt: break

        # Update the mask on light time to hide intercepts behind the observer
        # or outside the defined limit
        lt.mask = (lt.mask | (lt.vals * sign < 0.) |
                   (lt.vals == lt_min) | (lt.vals == lt_max))
        if not np.any(lt.mask): lt.mask = False

        # Create the surface Event object
        surface_time = event.time + lt
        surface_event = Event(surface_time, intercept, (0,0,0),
                              self.origin_id, self.frame_id)
        surface_event.insert_subfield("perp",  self.normal(intercept))
        surface_event.insert_subfield("vflat", self.velocity(intercept))

        surface_event.insert_subfield(surface_key, -lt * los_in_frame)
        surface_event.insert_subfield(surface_key + "_lt", -lt)

        # Insert the derivative aubarrays if necessary
        if derivs:

            # Re-start geometry from the origin
            pos_in_j2000 = (obs_wrt_ssb
                            - origin_wrt_ssb.event_at_time(surface_time).pos)

            transform = frame_wrt_j2000.transform_at_time(surface_time)
            pos_in_frame = transform.rotate(pos_in_j2000)
            los_in_frame = transform.rotate(los_wrt_ssb)

            (intercept, dlt) = self.intercept(pos_in_frame, los_in_frame,
                                              derivs=True)

            # Rotate derivatives WRT origin
            dtime_dpos = sign * dlt.d_dobs
            dtime_dlos = sign * dlt.d_dlos
            dpos_dpos = intercept.d_dobs
            dpos_dlos = intercept.d_dlos

            surface_wrt_j2000_mat = (
                frame_wrt_j2000.transform_at_time(surface_time, quick).matrix)

            event_inv_frame = registry.connect_frames("J2000", event.frame_id)
            j2000_wrt_event_mat = (
                event_inv_frame.transform_at_time(event.time, quick).matrix)

            surface_wrt_event_mat = (
                surface_wrt_j2000_mat.rotate_matrix3(j2000_wrt_event_mat))

            dtime_dpos = surface_wrt_event_mat.unrotate(dtime_dpos.T()).T()
            dtime_dlos = surface_wrt_event_mat.unrotate(dtime_dlos.T()).T()
            dpos_dpos  = surface_wrt_event_mat.unrotate(dpos_dpos.T()).T()
            dpos_dlos  = surface_wrt_event_mat.unrotate(dpos_dlos.T()).T()

            # Save the derivatives as subfields
            surface_event.time.insert_subfield("d_dpos", dtime_dpos)
            surface_event.time.insert_subfield("d_dlos", dtime_dlos)
            surface_event.pos.insert_subfield( "d_dpos", dpos_dpos)
            surface_event.pos.insert_subfield( "d_dlos", dpos_dlos)

        return surface_event

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Surface(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
