################################################################################
# oops/surface/surface.py: Abstract class Surface
#
# 2/5/12 Modified (MRS) - Minor updates for style
################################################################################

import numpy as np

from oops.event import Event
from oops.xarray.all import *
import oops.frame.all as frame_
import oops.path.all  as path_
import oops.constants as constants

class Surface(object):
    """Surface is an abstract class describing a 2-D object that moves and
    rotates in space. A surface employs an internal coordinate system, not
    necessarily rectangular, in which two primary coordinates define locations
    on the surface, and an optional third coordinate can define points above or
    below that surface. The shape is always fixed."""

    DEBUG = False

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

    def intercept(self, obs, los):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.

        Return:         a tuple (position, factor)
            position    a unitless Vector3 of intercept points on the surface,
                        in km.
            factor      a unitless Scalar of factors such that:
                            position = obs + factor * los
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

    def gradient(self, position, axis=0, projected=True, **keywords):
        """Returns the gradient vector at a specified position at or near the
        surface. The gradient is defined as the vector pointing in the direction
        of most rapid change in the value of a particular surface coordinate.

        The magnitude of the gradient vector is the rate of change of the
        coordinate value when starting from this point and moving in this
        direction.

        Input:
            position    a Vector3 of positions at or near the surface, with
                        optional units.
            axis        0, 1 or 2, identifying the coordinate axis for which the
                        gradient is sought.
            projected   True to project the gradient vector into the surface.
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

########################################
# No need to override...
########################################

    def event_at(self, time, position):
        """Constructs an event object defined by time and position.

        Input:
            time            a Scalar of event times.
            position        a Vector3 of positions near the surface.

        Return:             an Event object. Note that the optional event
                            attributes "perp" and "vflat" are filled in.
        """

        # Create the absolute and relative Event objects
        return Event(time, position, (0,0,0),
                     self.origin_id, self.frame_id,
                     self.normal(position),
                     Empty(), Empty(),
                     self.velocity(position))

    def coords_at_event(self, event, axes=3):
        """Returns three Scalars of coordinates describing the positions and
        times found in an Event object.

        Input:
            event           an Event object.
            axes            2 or 3, indicating whether to include the third
                            Scalar of coordinates.

        Return:             a tuple containing two or three Scalars of
                            coordinate values.
        """

        event = event.wrt(self.origin_id, self.frame_id)
        return self.as_coords(event.pos, axes)

################################################################################
# Photon Solver
################################################################################

    def photon_from_event(self, event, iters=2, quick=True):
        """Returns the photon arrival event at the body's surface, for photons
        departing earlier from the specified event.
        """

        return self._solve_photon(event, +1, iters, quick)

    def photon_to_event(self, event, iters=2, quick=True):
        """Returns the photon departure event at the body's surface, for photons
        arriving later at the specified event.
        """

        return self._solve_photon(event, -1, iters, quick)

    def _solve_photon(self, event, sign, iters=2, quick=True):
        """Solve for the Event object located on the body's surface that falls
        at the other end of the photon's path to or from another Event.

        Input:
            event       the Event of a photon's arrival or departure. The path
                        of the Event use the same inertial origin and the same
                        inertial frame.

            sign        -1 to return earlier Events, corresponding to photons
                           departing from the surface and arriving at the event.
                        +1 to return later Events, corresponding to photons
                           departing the event and arriving at the surface.

            iters       number of iterations to perform.

            quick       True to use QuickPaths and QuickFrames, where warranted,
                        to speed up the calculation.

        Return:         a tuple containing the absolute and relative events at
                        the surface. The absolute event is relative to the
                        origin of the surface; the relative event originates
                        with the event at the other end of the photon's path.
        """

        signed_c = sign * constants.C

        # Define the path, frame and event relative to the SSB in J2000
        event_wrt_ssb = event.wrt_ssb(quick)

        origin_wrt_ssb = path_.connect(self.origin_id, "SSB", "J2000")
        frame_wrt_j2000 = frame_.connect(self.frame_id, "J2000")

        # Define the origin and line of sight in the SSB frame
        obs_wrt_ssb = event_wrt_ssb.pos
        if sign < 0.:
            vel_wrt_ssb = event_wrt_ssb.arr.unit() * constants.C
        else:
            vel_wrt_ssb = event_wrt_ssb.dep.unit() * constants.C

        # Make an initial guess at the light travel time using the range to the
        # surface's origin
        lt = (obs_wrt_ssb -
              origin_wrt_ssb.event_at_time(event.time).pos).norm() / signed_c
        surface_time = event.time + lt

        # Quicken the path and frame evaluations if requested
        origin_wrt_ssb = origin_wrt_ssb.quick_path(surface_time, quick)
        frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time, quick)

        # Iterate a fixed number of times. Convergence is rapid because all
        # speeds are non-relativistic.
        for iter in range(iters):

            # Un-mask the time, position and velocity for each iteration. This
            # is necessary because a line of sight could miss the surface on one
            # iteration but hit it on the next
            obs_wrt_ssb.mask = False
            vel_wrt_ssb.mask = False
            lt.mask = False

            # Evaluate the current time
            surface_time = event.time + lt

            # Locate the photons relative to the current origin in SSB/J2000
            pos_in_j2000 = (obs_wrt_ssb + lt * vel_wrt_ssb
                            - origin_wrt_ssb.event_at_time(surface_time).pos)

            # Rotate into the surface-fixed frame
            transform = frame_wrt_j2000.transform_at_time(surface_time)
            pos_in_frame = transform.rotate(pos_in_j2000)
            vel_in_frame = transform.rotate(vel_wrt_ssb)

            # Update the intercept times; save the intercept positions
            (intercept, dlt) = self.intercept(pos_in_frame, vel_in_frame)
            lt = lt + dlt

            if Surface.DEBUG:
                print iter
                print dlt

        # Update the mask on light time to hide intercepts behind the observer
        lt.mask = lt.mask | (lt.vals * sign < 0.)
        if not np.any(lt.mask): lt.mask = False

        # Create the absolute and relative Event objects
        absolute_event = self.event_at(event.time + lt, intercept)

        if sign < 0.:
            absolute_event.dep = vel_in_frame
        else:
            absolute_event.arr = vel_in_frame

        relative_event = absolute_event.wrt_event(event, quick)
        relative_event.time = lt    # slighty finer precision

        # Return the intercept event in two frames
        return (absolute_event, relative_event)

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
