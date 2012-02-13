################################################################################
# Event
#
# 2/2/12 Modified (MRS) - import and class naming hierarchy revised.
################################################################################

import numpy as np
import unittest

from oops.xarray.all import *
import oops.frame.registry as frame_registry
import oops.path.registry as path_registry
import oops.constants as constants

class Event(object):
    """An Event object is defined by a time, position and velocity. Events can
    be either absolute or relative.

    In an absolute path, times are measured in seconds TDB relative to January
    1, 2000 as defined in the SPICE toolkit. Positions and velocities are
    measured in km relative to a named origin Path object and within a named
    Frame.

    In a relative path, times, positions and velocities are measured relative to
    the corresponding values in another event. The event objects need not have
    the same shape; standard rules of broadcasting apply.
    """

    def __init__(self, time, pos, vel, origin, frame=None,
                       perp=Empty(), arr=Empty(), dep=Empty(),
                       vflat=Vector3([0.,0.,0.])):
        """Constructor for the Event class.

        Input:
            time        a Scalar of event time in seconds TDB.
            pos         position vectors as a Vector3 object.
            vel         velocity vectors as a Vector3 object.
            origin      If the origin is a Path object or a path ID, it is
                        interpreted as the path defining the origin of this
                        event. If the origin is an Event object, then the time,
                        position and velocity are intepreted as offsets from
                        the corresponding values in this other event.
            frame       a frame or its ID, identifying the coordinate system.
                        Default is to match the default frame of the origin.

        Optional attributes, defined as Empty() if unused:
            perp        the direction of a normal vector if the event falls on a
                        surface.
            arr         the direction of a photon arriving at the event, NOT
                        corrected for stellar aberration.
            dep         the direction of a photon departing from the event, NOT
                        corrected for stellar aberration.
            vflat       a velocity component within the surface, which can be
                        used to describe winds across a planet or orbital motion
                        within a ring plane.

        These attributes are also maintained internally.
            shape       a list of integers defining the overall shape of the
                        event.

            ssb         a copy of this Event object in J2000 coordinates
                        relative to the Solar System Barycenter. None if not
                        currently available.

        Note that these attributes need not have the same shape, but they must
        all be broadcastable to the same shape.
        """

        self.time = Scalar.as_scalar(time)
        self.pos  = Vector3.as_vector3(pos)
        self.vel  = Vector3.as_vector3(vel)

        # Absolute event case
        if path_registry.is_id(origin) or path_registry.is_path(origin):
            self.origin = None
            self.origin_id = path_registry.as_id(origin)
            self.frame_id = path_registry.as_path(origin).frame_id

        # Relative event case
        else:
            self.origin = origin
            self.origin_id = None
            self.frame_id = self.origin.frame_id

        if frame is not None:
            self.frame_id = frame_registry.as_id(frame)

        self.perp  = perp
        self.arr   = arr
        self.dep   = dep
        self.vflat = vflat

        self.ssb = None

        self.shape = []
        self.update_shape()

    def update_shape(self, *arg):
        """Used to update the shape of this object when any attributes are
        modified. The argument can be any number of arguments as long as each
        has a shape attribute. If no arguments are provided, then the shape of
        the object is completely regenerated.
        """

        oldshape = self.shape

        if arg == ():
            self.shape = Array.broadcast_shape(
                            [self.time, self.pos, self.vel,
                             self.origin,
                             path_registry.as_path(self.origin_id),
                             frame_registry.as_frame(self.frame_id),
                             self.perp, self.arr, self.vel, self.vflat])
        else:
            self.shape = Array.broadcast_shape((self,) + arg)

        if self.shape != oldshape:
            self.ssb = None

        return self

    def is_absolute(self):
        """Returns True if the event is absolute, False if it is relative."""

        return self.origin is None

    def is_relative(self):
        """Returns True if the event is relative, False if it is absolute."""

        return self.origin is not None

    def get_origin_param(self):
        """Returns the origin parameter required by the constructor: an event
        object, a Path object, or a path ID."""

        if self.origin is not None:
            return self.origin

        return self.origin_id

    def absolute_time(self):
        """Returns the absolute time of this event."""

        if self.is_absolute(): return self.time

        # Recursive call...
        return self.time + self.origin.absolute_time()

    def wrt_ssb(self):
        """Returns the event relative to SSB coordinates in the J2000 frame
        while also filling in the internal value if necessary."""

        if self.ssb is None:
            self.ssb = self.wrt("SSB", "J2000")

        return self.ssb

    def copy(self):
        """Returns a copy of the Event object with all attributes themselves
        copied."""

        return Event(self.time.copy(),
                     self.pos.copy(),
                     self.vel.copy(),
                     self.get_origin_param(),
                     self.frame_id,
                     self.perp.copy(),
                     self.arr.copy(),
                     self.dep.copy(),
                     self.vflat.copy())

    @staticmethod
    def null_event(time, origin="SSB", frame="J2000"):
        return Event(time, (0.,0.,0.), (0.,0.,0.), origin, frame)

############################################
# Event arithmetic
############################################

    def absolute(self):
        """Returns an absolute version of this Event object, with the new origin
        and frame defined by the first absolute event encountered."""

        if self.is_absolute():
            return self

        # Recursive call...
        origin = self.origin.absolute()
        origin_ssb = origin.wrt_ssb()

        # Get the offset in J2000 coordinates, with the transform evaluated at
        # the time of the origin event
        event_ssb = self.wrt_frame("J2000")

        # Offset the event
        result_ssb = Event(event_ssb.time + origin_ssb.time,
                           event_ssb.pos  + origin_ssb.pos,
                           event_ssb.vel  + origin_ssb.vel,
                           "SSB", "J2000",
                           event_ssb.perp.copy(),
                           event_ssb.arr.copy(),
                           event_ssb.dep.copy(),
                           event_ssb.vflat.copy())

        # Rotate back to the default path and frame
        result = result_ssb.wrt(origin.origin_id, origin.frame_id)
        result.ssb = result_ssb

        return result.update_shape()

    def wrt_event(self, arg):
        """Returns a revised version of this Event object, with times and
        positions specified relative to another event and using that event's
        frame."""

        # Move into standard coordinates
        origin_ssb = arg.absolute().wrt_ssb()
        event_ssb  = self.absolute().wrt_ssb()

        # Apply the offset
        result_ssb = Event(event_ssb.time - origin_ssb.time,
                           event_ssb.pos  - origin_ssb.pos,
                           event_ssb.vel  - origin_ssb.vel,
                           "SSB", "J2000",
                           event_ssb.perp.copy(),
                           event_ssb.arr.copy(),
                           event_ssb.dep.copy(),
                           event_ssb.vflat.copy())

        # Rotate back to the default frame (or make a copy at least)
        if arg.frame_id == "J2000":
            result = result_ssb.copy()
        else:
            result = result_ssb.wrt_frame(arg.frame_id, origin_ssb.time)

        result.ssb = result_ssb

        # Now redefine the origin
        result.origin_id = None
        result.origin = arg

        return result.update_shape(arg)

    def wrt_surface(self, surface, axes=3):
        """Returns the coordinate values associated with the event when 
        transformed into this surface.

        Input:
            surface         a Surface object.
            axes            2 or 3, indicating whether to include the third
                            Scalar of coordinates.

        Return:             a tuple containing two or three Scalars of
                            coordinate values.
        """

        return surface.coords_at_event(self, axes)

    def wrt(self, path=None, frame=None):
        """Returns a new event specified relative to a new path and/or a
        new coordinate frame.

        Input:
            path        the Path object or ID defining the new origin; None to
                        leave the origin unchanged.
            frame       the Frame object of ID of the new coordinate frame; None
                        to leave the frame unchanged.
        """

        # Convert to the new path if necessary
        event = self.wrt_path(path)

        # Convert to the new frame if necessary
        event = event.wrt_frame(frame)

        return event

    def wrt_path(self, path):
        """Returns the same event, but defined relative to a new origin. The
        frame will be unchanged. Relative events are converted to absolute
        first.

        Input:
            path        the Path object to be subtracted. Its origin must be
                        the event's current origin, and the event returned will
                        have its origin at this path.

                        Alternatively, if this is a path ID, then the returned
                        event will have this path ID as its new origin and will
                        use the default frame in which this path is defined.

                        If the value is None, the event is returned unchanged.
        """

        if path is None: return self

        event = self.absolute()
        path = path_registry.connect(path, event.origin_id, event.frame_id)

        return path.subtract_from_event(event)

    def wrt_frame(self, frame, time=None):
        """Returns the same event after coordinates have been transformed to
        the specified frame. The origin is unchanged. For relative events, the
        frame is evaluated at the time of the origin event.

        Input:
            frame       a Frame object to transform the coordinates. Its
                        reference frame must be the current frame of the event.

                        Alternatively, if this is a Frame ID, then the returned
                        event will be transformed to this frame.

                        If the value is None, the event is returned unchanged.

            time        a Scalar of time values at which to evaluate the
                        coordinate transformation. By default, the event time is
                        used. However, this parameter is needed (at least) when
                        transforming relative events.
        """

        if frame is None: return self

        if frame_registry.is_id(frame):
            frame = frame_registry.connect(frame, self.frame_id)

        if time is None:
            if self.is_relative():
                time = self.origin.absolute_time()
            else:
                time = self.time

        return self.rotate_by_frame(frame, time)

    def rotate_by_frame(self, frame, time=None):
        """Returns the same event after all coordinates have been transformed
        forward into a new frame. The origin is unchanged.

        Input:
            frame       a Frame object to transform the coordinates. Its
                        reference frame must be the current frame of the event.
            time        a Scalar of time values at which to evaluate the
                        coordinate transformation. By default, the event time is
                        used. However, this parameter is needed (at least) when
                        transforming relative events.
        """

        assert self.frame_id == frame.reference_id

        if time is None: time = self.time

        transform = frame.transform_at_time(time)
        return Event(time,
                     transform.rotate_pos(self.pos),
                     transform.rotate_vel(self.vel, self.pos),
                     self.get_origin_param(),
                     frame.frame_id,
                     transform.rotate(self.perp),
                     transform.rotate(self.arr),
                     transform.rotate(self.dep),
                     transform.rotate(self.vflat))

    def unrotate_by_frame(self, frame, time=None):
        """Returns the same event after all coordinates have been transformed
        backward to the parent frame. The origin is unchanged.

        Input:
            frame       a Frame object to to inverse-transform the coordinates.
                        Its target frame must be the current frame of the event.
                        The returned event will use the reference frame instead.
            time        a Scalar of time values at which to evaluate the
                        coordinate transformation. By default, the event time is
                        used. However, this parameter is needed (at least) when
                        transforming relative events.
        """

        assert self.frame_id == frame.frame_id

        if time is None: time = self.time

        transform = frame.transform_at_time(time)
        return Event(time,
                     transform.unrotate_pos(self.pos),
                     transform.unrotate_vel(self.vel, self.pos),
                     self.get_origin_param(),
                     frame.reference_id,
                     transform.unrotate(self.perp),
                     transform.unrotate(self.arr),
                     transform.unrotate(self.dep),
                     transform.unrotate(self.vflat))

    # binary "-" operator
    def __sub__(self, arg):

        if isinstance(arg, Event):
            return self.wrt_event(arg)

        if path_registry.is_id(arg) or path_registry.is_path(arg):
            return self.wrt_path(arg)

        Array.raise_type_mismatch(self, "-", arg)

    # binary "/" operator
    def __div__(self, arg):

        if frame_registry.is_id(arg) or frame_registry.is_frame(arg):
            return self.wrt_frame(arg)

        Array.raise_type_mismatch(self, "/", arg)

    # string operations
    def __str__(self):
        if self.origin_id is None:
            name = "Event()"
        else:
            name = repr(self.origin_id)

        list = ["Event([", repr(self.shape).replace(' ', ''), " - ",
                           name, "]/",
                           repr(self.frame_id)]
        return "".join(list)

    def __repr__(self): return self.__str__()

    # indexing []
    # Probably not a good idea
#     def __getitem__(self, i):
#         (time, pos, vel,
#          perp, arr, dep, vflat) = Array.broadcast_arrays(
#                                     (self.time, self.pos, self.vel,
#                                      self.perp, self.arr, self.dep, self.vflat))
# 
#         return Event(time[i], pos[i], vel[i],
#                      self.get_origin_param(), self.frame_id,
#                      perp[i], arr[i], dep[i], vflat[i])
# 
#     def __getslice__(self, i, j):
#         (time, pos, vel,
#          perp, arr, dep, vflat) = Array.broadcast_arrays(
#                                     (self.time, self.pos, self.vel,
#                                      self.perp, self.arr, self.dep, self.vflat))
# 
#         return Event(time[i:j], pos[i:j], vel[i:j],
#                      self.get_origin_param(), self.frame_id,
#                      perp[i:j], arr[i:j], dep[i:j], vflat[i:j])

################################################################################
# Photon event calculations
################################################################################

    def photon_to_path(self, path, relative=False, iters=3, quick_info=None):
        """Connects a photon departing from this event to its later arrival on a
        specified path. It updates the photon departure attribute of this event
        and returns a new event describing the photon's arrival on the specified
        path.

        Note that the event and the Path objects need not have the same shape,
        but they must be broadcastable to the same shape.

        Input:
            path        the path to which the photon is departing.

            relative    True to return an event relative to the this event;
                        False to return an absolute event defined with respect
                        to the path.

            iters       number of iterations of Newton's method to perform. For
                        iters == 0, the time of the returned event will only be
                        corrected for the light travel time to the path's
                        origin. Full precision is generally achieved in 2-3
                        iterations.

            quick_info  parameters to be passed to quick_path() and
                        quick_frame(), if these mechanisms are to be used to
                        speed up the calculations. None to do things the slow
                        way.

        Return:         a tuple containing(path_event, lt)

            path_event  the associated photon arrival event on the given path.

            lt          the light travel time between the two events.

        Side-Effects:   the photon departure attribute is filled in for this
                        event.
        """

        return path.photon_from_event(self, relative, iters, quick_info)

    def photon_from_path(self, path, relative=False, iters=3, quick_info=None):
        """Connects a photon arriving at this event to its earlier departure
        from a specified path. It updates the photon arrival attribute of this
        event and returns a new event describing the photon's departure on the
        specified path.

        Note that the event and the Path objects need not have the same shape,
        but they must be broadcastable to the same shape.

        Input:
            path        the path from which the photon is arriving.

            relative    True to return an event relative to the this event;
                        False to return an absolute event defined with respect
                        to the path.

            iters       number of iterations of Newton's method to perform. For
                        iters == 0, the time of the returned event will only be
                        corrected for the light travel time to the path's
                        origin. Full precision is generally achieved in 2-3
                        iterations.

            quick_info  parameters to be passed to quick_path() and
                        quick_frame(), if these mechanisms are to be used to
                        speed up the calculations. None to do things the slow
                        way.

        Return:         a tuple containing(path_event, lt)

            path_event  the associated photon departure event on the given path.

            lt          the light travel time between the two events.

        Side-Effects:   the photon arrival attribute is filled in for this
                        event.
        """

        return path.photon_to_event(self, relative, iters, quick_info)

########################################

    def photon_to_surface(self, body, iters=3, quick_info=None):
        """Connects the photon departing from this event to its later arrival at
        the surface of a specified body. It returns a new event describing the
        arrivals.
        """

        return body.photon_from_event(self, iters, quick_info)

    def photon_from_surface(self, body, iters=3, quick_info=None):
        """Connects the photon arriving at this event to its earlier departure
        from the surface of a specified body. It returns a new event describing
        the departures.
        """

        return body.photon_to_event(self, iters, quick_info)

################################################################################
# Geometry procedures
################################################################################

    def aberrated_ray(self, ray):
        """Returns the apparent direction of a photon given its actual
        direction in the SSB/J2000 frame."""

        # This procedure is equivalent to a vector subtraction of the velocity
        # of the observer from the ray, given the ray has length C.

        wrt_ssb = self.wrt_ssb()
        return ray - (wrt_ssb.vel + wrt_ssb.vflat) * ray.norm()/constants.C

    def aberrated_arr(self): return self.aberrated_ray(self.wrt_ssb().arr)

    def aberrated_dep(self): return self.aberrated_ray(self.wrt_ssb().dep)

    def incidence_angle(self, aberration=False):
        """Returns the incidence angle, measured between the surface normal and
        the reversed direction of the arriving photon."""

        return self.wrt_ssb().perp.sep(self.aberrated_arr(), reverse=True)

    def emission_angle(self):
        """Returns the emission angle, measured between the surface normal and
        the direction of the departing photon."""

        return self.wrt_ssb().perp.sep(self.aberrated_dep())

    def phase_angle(self):
        """Returns the phase angle, measured between the direction of the
        arriving photon and the reversed direction of the departing photon."""

        return self.aberrated_arr().sep(self.aberrated_dep(), reversed=True)

    def ra_and_dec(self, aberration=False, frame="J2000"):
        """Returns the J2000 right ascension amd declination in the path and
        frame of the event.

        Input:
            aberration  True to include stellar aberration, thereby returning
                        the apparent direction of the photon relative to the
                        background stars; False to return the purely geometric
                        values, neglecting the motion of the observer.
            frame       The frame in which the values should be returned. The
                        default is J2000, but B1950 might be useful under some
                        circumstances.
        """

        # Locate arrival ray in the SSB/J2000 frame
        if aberration:
            arr = -self.aberrated_arr()
        else:
            arr = -self.wrt_ssb().arr

        # Convert to RA and dec
        buffer = np.empty(arr.shape + [2])
        buffer[...,0] = np.arctan2(arr.vals[...,1],
                                   arr.vals[...,0]) % (2.*np.pi)
        buffer[...,1] = np.arcsin(arr.vals[...,2] / arr.norm().vals)

        return Pair(buffer)

################################################################################
# UNIT TESTS
################################################################################

class Test_Event(unittest.TestCase):

    def runTest(self):

        # Tested thoroughly in other modules
        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
