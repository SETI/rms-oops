# oops/Surface/__init__.py

import numpy as np
import unittest

import oops

################################################################################
# Abstract Surface class
################################################################################

class Surface(object):
    """An abstract object describing a 2-D object that moves and rotates in
    space. A surface employs an internal coordinate system, not necessarily
    rectangular, in which two primary coordinates define locations on the
    surface, and an optional third coordinate can define points above or below
    that surface. The shape is always fixed."""

    OOPS_CLASS = "Surface"

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

    def as_coords(self, position, axes=2):
        """Converts from position vectors in the internal frame into the surface
        coordinate system.

        Input:
            position        a Vector3 of positions at or near the surface.
            axes            2 or 3, indicating whether to return a tuple of two
                            or 3 Scalar objects.

        Return:             coordinate values packaged as a tuple containing
                            two or three Scalars, one for each coordinate.
        """

        pass

    def as_vector3(self, coord1, coord2, coord3=oops.Scalar(0.)):
        """Converts coordinates in the surface's internal coordinate system into
        position vectors at or near the surface.

        Input:
            coord1          a Scalar of values for the first coordinate.
            coord2          a Scalar of values for the second coordinate.
            coord3          a Scalar of values for the third coordinate; default
                            is Scalar(0.).

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.

        Return:             the corresponding Vector3 object of positions.
        """

        pass

    def intercept(self, obs, los):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs             observer position as a Vector3.
            los             line of sight as a Vector3.

        Return:             a tuple (position, factor)
            position        a Vector3 of intercept points on the surface.
            factor          a Scalar of factors such that
                                position = obs + factor * los
        """

        pass

    def normal(self, position):
        """Returns the normal vector at a position at or near a surface.

        Input:
            position        a Vector3 of positions at or near the surface.

        Return:             a Vector3 containing directions normal to the
                            surface that pass through the position. Lengths are
                            arbitrary.
        """

        pass

    def gradient(self, position, axis=0, projected=True):
        """Returns the gradient vector at a specified position at or near the
        surface. The gradient is defined as the vector pointing in the direction
        of most rapid change in the value of a particular surface coordinate.

        The magnitude of the gradient vector is the rate of change of the
        coordinate value when starting from this point and moving in this
        direction.

        Input:
            position        a Vector3 of positions at or near the surface.

            axis            0, 1 or 2, identifying the coordinate axis for which
                            the gradient is sought.

            projected       True to project the gradient into the surface if
                            necessary. This has no effect on a RingPlane.
        """

        pass

    def velocity(self, position):
        """Returns the local velocity vector at a point within the surface.
        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            position        a Vector3 of positions at or near the surface.
        """

        return oops.Vector3((0,0,0))

    def intercept_with_normal(self, normal):
        """Constructs the intercept point on the surface where the normal vector
        is parallel to the given vector.

        Input:
            normal          a Vector3 of normal vectors.

        Return:             a Vector3 of surface intercept points. Where no
                            solution exists, the components of the returned
                            vector should be np.nan.
        """

        pass

    def intercept_normal_to(self, position):
        """Constructs the intercept point on the surface where a normal vector
        passes through a given position.

        Input:
            position        a Vector3 of positions near the surface.

        Return:             a Vector3 of surface intercept points. Where no
                            solution exists, the components of the returned
                            vector should be np.nan.
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

        Return:             an Event object. Not that the optional event
                            attributes "perp" and "vflat" are filled in.
        """

        # Create the absolute and relative Event objects
        return oops.Event(time,
                          position,
                          oops.Vector3((0.,0.,0.)),
                          self.origin_id, self.frame_id,
                          self.normal(position),
                          oops.Empty(), oops.Empty(),
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

    def photon_from_event(self, event, iters=3, quick_info=None):
        """Returns the photon arrival event at the body's surface, for photons
        departing earlier from the specified event.
        """

        return self._solve_photon(event, +1, iters, quick_info)

    def photon_to_event(self, event, iters=3, quick_info=None):
        """Returns the photon departure event at the body's surface, for photons
        arriving later at the specified event.
        """

        return self._solve_photon(event, -1, iters, quick_info)

    def _solve_photon(self, event, sign, iters=3, quick_info=None):
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

            quick_info  parameters to be passed to quick_path() and
                        quick_frame(), if these mechanisms are to be used to
                        speed up the calculations. None to do things the slow
                        way.

        Return:         the Events on the surface of the body.
        """

        signed_c = sign * oops.C

        # Define the path, frame and event relative to the SSB in J2000
        event_wrt_ssb = event.wrt_ssb()

        origin_wrt_ssb  = oops.Path.connect(self.origin_id, "SSB", "J2000")
        frame_in_j2000 = oops.Frame.connect(self.frame_id, "J2000")

        # Define the origin and line of sight in the SSB frame
        obs_wrt_ssb = event_wrt_ssb.pos
        if sign < 0.:
            vel_wrt_ssb = event_wrt_ssb.arr.unit() * oops.C
        else:
            vel_wrt_ssb = event_wrt_ssb.dep.unit() * oops.C

        # Make an initial guess at the light travel time using the range to the
        # surface's origin
        lt = (obs_wrt_ssb -
              origin_wrt_ssb.event_at_time(event.time).pos).norm() / signed_c
        surface_time = event.time + lt

        # Quickend the path and frame evaluations if requested
        if iters > 1 and quick_info is not None:
            epoch = np.mean(surface_time.vals)
            origin_wrt_ssb = origin_wrt_ssb.quick_path(epoch, quick_info)
            frame_wrt_j2000 = frame_wrt_j2000.quick_frame(epoch, quick_info)

        # Iterate a fixed number of times. Convergence is rapid because all
        # speeds are non-relativistic.
        for iter in range(iters):

            # Evaluate the current time
            surface_time = event.time + lt

            # Locate the photons relative to the current origin in J2000
            pos_in_j2000 = (obs_wrt_ssb + lt * vel_wrt_ssb
                            - origin_wrt_ssb.event_at_time(surface_time).pos)

            # Rotate into the surface-fixed frame
            transform = frame_in_j2000.transform_at_time(surface_time)
            pos_in_frame = transform.rotate(pos_in_j2000)
            vel_in_frame = transform.rotate(vel_wrt_ssb)

            # Update the intercept times; save the intercept positions
            (intercept, dlt) = self.intercept(pos_in_frame, vel_in_frame)
            lt += dlt

            if oops.DEBUG:
                print iter
                print dlt

        # Create the absolute and relative Event objects
        absolute_event = self.event_at(event.time + lt, intercept)

        if sign < 0.:
            absolute_event.dep = vel_in_frame
        else:
            absolute_event.arr = vel_in_frame

        relative_event = absolute_event.wrt_event(event)
        relative_event.time = lt    # slighty finer precision

        # Return the intercept event in two frames
        return (absolute_event, relative_event)

################################################################################
# UNIT TESTS
################################################################################

class Test_Surface(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
