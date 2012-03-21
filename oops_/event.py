################################################################################
# oops_/event.py: Event class
#
# 2/2/12 Modified (MRS) - import and class naming hierarchy revised.
# 3/12/12 MRS - Implemented support for derivatives.
################################################################################

import numpy as np
import unittest

from oops_.array.all import *
from oops_.config import QUICK, EVENT_COLLAPSE_THRESHOLD, LOGGING
import oops_.registry as registry
import oops_.constants as constants

class Event(object):
    """An Event object is defined by a time, position and velocity. Times are
    measured in seconds TDB relative to noon TDB on January 1, 2000, as defined
    in the SPICE toolkit. Positions and velocities are measured in km relative
    to a named origin Path object and within a named Frame.

    The event objects need not have the same shape; standard rules of
    broadcasting apply.
    """

    def __init__(self, time, pos, vel, origin, frame=None, **subfields):
        """Constructor for the Event class.

        Input:
            time        a Scalar of event times in seconds TDB.
            pos         position vectors as a Vector3 object.
            vel         velocity vectors as a Vector3 object.
            origin_id   the ID of a path defining the origin of this event.
            frame_id    the ID of a frame identifying the coordinate system.
                        Default is to match the frame of the origin.

        These optional attributes define the connection between this event and
        some other event.
            link        an Event object relative to which this one was defined.
                        Any partial derivatives in this Event object are assumed
                        to be relative to changes at the prior event.
            sign        the sign of the direction of the photon between this
                        event and the link. +1 for an outgoing photon to a later
                        event; -1 for an incoming photon from an earlier event;
                        0 for an unlinked event.

        The following attributes are optional. They are inserted using the
        insert_subfield() method, or as extra keyword=value pairs in the
        constructor:
            arr         the direction of a photon arriving at the event, NOT
                        corrected for stellar aberration.
            dep         the direction of a photon departing from the event, NOT
                        corrected for stellar aberration.
            arr_lt      the (negative) light travel time for an arriving photon
                        from its origin.
            dep_lt      the light travel time of a departing photon to its
                        destination.
            perp        the direction of a normal vector if the event falls on a
                        surface.
            vflat       a velocity component within the surface, which can be
                        used to describe winds across a planet or orbital motion
                        within a ring plane.

            subfields   a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.

        These properties are also available.
            shape       a list of integers defining the overall shape of the
                        event, found as a result of broadcasting together the
                        time, position, velocity, arr and dep attributes.

            mask        the overall mask of the Event, constructed as the "or"
                        of the time, pos, vel, arr and dep masks.
        """

        self.time  = Scalar.as_scalar(time)
        self.pos   = Vector3.as_vector3(pos)
        self.vel   = Vector3.as_vector3(vel)

        self.origin_id = registry.as_path_id(origin)
        self.frame_id = registry.as_path(origin).frame_id

        self.link = None
        self.sign = 0

        if frame is not None:
            self.frame_id = registry.as_frame_id(frame)

        self.subfields = {}
        self.insert_subfield("arr", Empty())            # These always exist
        self.insert_subfield("dep", Empty())
        self.insert_subfield("vflat", Vector3([0.,0.,0.])) # default

        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        self.filled_shape = None
        self.filled_mask  = None
        self.filled_ssb   = None
        self.subfield_math_property = True

    @property
    def shape(self):
        if self.filled_shape is None:
            self.filled_shape = Array.broadcast_shape([self.time,
                                            self.pos, self.vel,
                                            registry.as_path(self.origin_id),
                                            registry.as_frame(self.frame_id),
                                            self.arr, self.dep])
        return self.filled_shape

    @property
    def mask(self):
        if self.filled_mask is None:
            self.filled_mask = (self.time.mask | self.pos.mask | self.vel.mask |
                                self.arr.mask | self.dep.mask)

        return self.filled_mask

    # subfield_math pseudo-attribute
    def get_subfield_math(self):
        return self.subfield_math_property

    def set_subfield_math(self, value):
        self.subfield_math_property = value

        self.time.subfield_math = value
        self.pos.subfield_math  = value
        self.vel.subfield_math  = value

        for key in self.subfields.keys():
            self.subfields[key].subfield_math = value

    subfield_math = property(get_subfield_math, set_subfield_math)

    def expand_mask(self):
        """Expands the mask to an array if it is currently just a boolean."""

        if self.mask.shape == ():
            if self.mask:
                self.lt.mask  = np.ones(self.shape, dtype="bool")
                self.pos.mask = np.ones(self.shape, dtype="bool")
                self.vel.mask + np.ones(self.shape, dtype="bool")
            else:
                self.lt.mask  = np.zeros(self.shape, dtype="bool")
                self.pos.mask = np.zeros(self.shape, dtype="bool")
                self.vel.mask + np.zeros(self.shape, dtype="bool")

            self.filled_mask = None
            ignore = self.mask

    def collapse_mask(self):
        """Reduces the mask to a single boolean if possible."""

        if not np.any(self.mask):
            self.lt.mask  = False
            self.pos.mask = False
            self.vel.mask = False
        elif np.all(self.mask):
            self.lt.mask  = True
            self.pos.mask = True
            self.vel.mask = True

        self.filled_mask = None
        ignore = self.mask

    def collapse_time(self, threshold=EVENT_COLLAPSE_THRESHOLD):
        """Replaces the time array by the average of its min and max, provided
        the range of times is less than or equal to the specified threshold."""

        if self.time.shape != []:
            tmin = self.time.min()
            tmax = self.time.max()
            if (tmax - tmin <= threshold):
                self.time = Scalar((tmin + tmax) / 2.)

                # Update the SSB version as well
                if self.filled_ssb is not None:
                    self.filled_ssb.time = self.time

            if LOGGING.event_time_collapse:
                print LOGGING.prefix, "Event.collapse_time()",
                print self.time.vals, tmax - tmin

        return self

    def wrt_ssb(self, quick=QUICK, derivs=False):
        """Returns the event relative to SSB coordinates in the J2000 frame
        while also filling in the internal cached value if necessary. """

        if self.filled_ssb is None:
            self.filled_ssb = self.wrt("SSB", "J2000", quick, derivs=derivs)
            self.filled_ssb.filled_ssb = self.filled_ssb

        has_derivs = ("d_dpos" in self.filled_ssb.pos.subfields.keys() and
                      "d_dt"   in self.filled_ssb.pos.subfields.keys())

        if derivs and not has_derivs:
            self.filled_ssb = self.wrt("SSB", "J2000", quick, derivs=derivs)
            self.filled_ssb.filled_ssb = self.filled_ssb

        self.filled_ssb.subarray_math = derivs
        return self.filled_ssb

    def copy(self, derivs=True):
        """Returns a copy of the Event object with all attributes themselves
        copied. However, if derivs is False; the subfields of all Array
        attributes are not copied."""

        result = Event(self.time.copy(derivs),
                       self.pos.copy(derivs), self.vel.copy(derivs),
                       self.origin_id, self.frame_id)

        for key in self.subfields.keys():
            subfield = self.subfields[key]
            if isinstance(subfield, Array):
                result.insert_subfield(key, subfield.copy(derivs))
            else:
                result.insert_subfield(key, subfield)

        result.filled_shape = self.filled_shape
        result.filled_mask  = self.filled_mask

        if self.filled_ssb is self:
            result.filled_ssb = result
        elif result.filled_ssb is not None:
            result.filled_ssb = self.filled_ssb.copy(derivs)
        else:
            result.filled_ssb = None

        return result

    def plain(self, subfields=True):
        """Returns a shallow copy of self with derivatives and, optionally, the
        subfields removed."""

        result = Event(self.time.plain(),
                       self.pos.plain(), self.vel.plain(),
                       self.origin_id, self.frame_id)

        if subfields:
            for key in self.subfields.keys():
                subfield = self.subfields[key]
                if isinstance(subfield, Array):
                    result.insert_subfield(key, subfield.plain())
                else:
                    result.insert_subfield(key, subfield)

        result.filled_shape = self.filled_shape
        result.filled_mask  = self.filled_mask

        if self.filled_ssb is self:
            result.filled_ssb = result
        else:
            result.filled_ssb = None

        return result

    def unmasked(self, subfields=True):
        """Returns a shallow copy of self with the mask removed."""

        result = Event(self.time.unmasked(),
                       self.pos.unmasked(), self.vel.unmasked(),
                       self.origin_id, self.frame_id)

        if subfields:
            for key in self.subfields.keys():
                subfield = self.subfields[key]
                if isinstance(subfield, Array):
                    result.insert_subfield(key, subfield.unmasked())
                else:
                    result.insert_subfield(key, subfield)

        result.filled_shape = self.filled_shape
        result.filled_mask  = False

        if self.filled_ssb is self:
            result.filled_ssb = result
        else:
            result.filled_ssb = None

        return result

    @staticmethod
    def null_event(time, origin="SSB", frame="J2000"):
        return Event(time, (0.,0.,0.), (0.,0.,0.), origin, frame)

    ####################################################
    # Indexing operators
    ####################################################

    def __getitem__(self, i):

        time = self.time.rebroadcast(self.shape)[i]
        pos  = self.pos.rebroadcast(self.shape)[i]
        vel  = self.vel.rebroadcast(self.shape)[i]

        result = Event(time, pos, vel, self.origin_id, self.frame_id)

        for key in self.subfields.keys():
            subfield = self.subfields[key].rebroadcast(self.shape)[i]
            result.insert_subfield(key, subfield)

        return result

    def __getslice__(self, i, j):

        time = self.time.rebroadcast(self.shape)[i:j]
        pos  = self.pos.rebroadcast(self.shape)[i:j]
        vel  = self.vel.rebroadcast(self.shape)[i:j]

        result = Event(time, pos, vel, self.origin_id, self.frame_id)

        for key in self.subfields.keys():
            subfield = self.subfields[key].rebroadcast(self.shape)[i:j]
            result.insert_subfield(key, subfield)

        return result

    ####################################################
    # Subfield support methods
    ####################################################

    def insert_subfield(self, key, value):
        """Adds a given subfield to the Event."""

        self.subfields[key] = value
        self.__dict__[key] = value      # This makes it an attribute as well

        self.filled_ssb = None          # SSB version is now out of date

    def delete_subfield(self, key):
        """Deletes a subfield, but not arr or dep."""

        if key in ("arr","dep"):
            self.subfields[key] = Empty()
            self.__dict__[key] = self.subfields[key]
        elif key in self.subfields.keys():
            del self.subfields[key]
            del self.__dict__[key]

        if self.filled_ssb is not None:
            try:
                del self.filled_ssb.subfields[key]
                del self.filled_ssb.__dict__[key]
            except KeyError: pass

    def delete_subfields(self):
        """Deletes all subfields."""

        for key in self.subfields.keys():
            if key not in ("arr","dep"):
                del self.subfields[key]
                del self.__dict__[key]

        if self.filled_ssb is not None:
            try:
                del self.filled_ssb.subfields[key]
                del self.filled_ssb.__dict__[key]
            except KeyError: pass

    def delete_derivs(self):
        """Deletes all subfields of Array attributes."""

        for key in self.subfields.keys():
            try:
                self.subfields[key].delete_subfields()
            except: pass

        self.time.delete_subfields()
        self.pos.delete_subfields()
        self.vel.delete_subfields()

        if self.filled_ssb is not None and self.filled_ssb is not self:
            self.filled_ssb.delete_sub_subfields()

############################################
# Event transformations
############################################

    def wrt(self, path=None, frame=None, quick=QUICK, derivs=False):
        """Returns a new event specified relative to a new path and/or a
        new coordinate frame.

        Input:
            path        the Path object or ID defining the new origin; None to
                        leave the origin unchanged.
            frame       the Frame object of ID of the new coordinate frame; None
                        to leave the frame unchanged.
            quick       False to disable QuickPaths; True for the default
                        options; a dictionary to override specific options.
            derivs      if True, any vectors in the event are returned with
                        additional subfields "d_dt" and "d_dpos", representing
                        the partial derivatives in the new frame and path
                        relative to those in the original frame and path.

        If the self event already contains derivatives, and derivs is True, then
        the new derivatives will multiply the originals, yielding derivatives
        that are with respect to the same event as those of the self event.
        """

        # Sort out inputs
        if path is None: path = self.origin_id
        if frame is None: frame = self.frame_id

        path = registry.as_path(path)
        frame = registry.as_frame(frame)

        # Point to the working copy of the Event object
        event = self

        # If the path is shifting...
        if event.origin_id != path.path_id:

            # and the current frame is rotating...
            old_frame = registry.as_frame(event.frame_id)
            if old_frame.origin_id is not None:

                # then rotate to J2000
                event = event.wrt_frame("J2000", quick, derivs=derivs)

        # If the frame is changing...
        if event.frame_id != frame.frame_id:

            # and the new frame is rotating...
            if frame.origin_id is not None:

                # then shift to the origin of the new frame
                event = event.wrt_path(frame.origin_id, quick, derivs=derivs)

        # Now it is safe to rotate to the new frame
        event = event.wrt_frame(frame.frame_id, quick, derivs=derivs)

        # Now it is safe to shift to the new path
        event = event.wrt_path(path.path_id, quick, derivs=derivs)

        event.subarray_math = derivs
        return event

    def wrt_path(self, path, quick=QUICK, derivs=False):
        """Returns an equivalent event, but defined relative to a different
        origin. The frame will be unchanged.

        Input:
            path        the Path object to be used as the new origin. If the
                        value is None, the event is returned unchanged.
            quick       False to disable QuickPaths; True for the default
                        options; a dictionary to override specific options.
            derivs      if True, any vectors in the event are returned with
                        additional subfields "d_dt" and "d_dpos", representing
                        the partial derivatives in the new frame and path
                        relative to those in the original path.
        """

        if path is None: return self

        path = registry.as_path_id(path)
        if path == self.origin_id: return self

        path = registry.connect_paths(path, self.origin_id, self.frame_id)
        path = path.quick_path(self.time, quick)

        return path.subtract_from_event(self, derivs=derivs)

    def wrt_frame(self, frame, quick=QUICK, derivs=False):
        """Returns an equivalent event, but defined relative to a different
        frame. The path is unchanged.

        Input:
            frame       the Frame object to be used as the new reference. If the
                        value is None, the event is returned unchanged.
            quick       False to disable QuickPaths; True for the default
                        options; a dictionary to override specific options.
            derivs      if True, any vectors in the event are returned with
                        additional subfields "d_dt" and "d_dpos", representing
                        the partial derivatives in the new frame and path
                        relative to those in the original frame.
        """

        if frame is None: return self

        frame = registry.as_frame_id(frame)
        if frame == self.frame_id: return self

        frame = registry.connect_frames(frame, self.frame_id)
        frame = frame.quick_frame(self.time, quick)

        return self.rotate_by_frame(frame, quick=False, derivs=derivs)

    def wrt_event(self, event, quick=QUICK, derivs=False):
        """Returns an equivalent event, but defined relative to the frame and
        path of the specified body.

        Input:
            body        a body object or its ID.
            quick       False to disable QuickPaths; True for the default
                        options; a dictionary to override specific options.
            derivs      if True, any vectors in the event are returned with
                        additional subfields "d_dt" and "d_dpos", representing
                        the partial derivatives in the new frame and path
                        relative to those in the original frame and path.
        """

        body = registry.as_body(body)
        return self.wrt(body.path_id, body.frame_id, quick=quick, derivs=derivs)

    def wrt_body(self, body, quick=QUICK, derivs=False):
        """Returns an equivalent event, but defined relative to the frame and
        path of the specified body.

        Input:
            body        a body object or its ID.
            quick       False to disable QuickPaths; True for the default
                        options; a dictionary to override specific options.
            derivs      if True, any vectors in the event are returned with
                        additional subfields "d_dt" and "d_dpos", representing
                        the partial derivatives in the new frame and path
                        relative to those in the original frame and path.
        """

        body = registry.as_body(body)
        return self.wrt(body.path_id, body.frame_id, quick=quick, derivs=derivs)

    def rotate_by_frame(self, frame, quick=QUICK, derivs=False):
        """Returns the same event after all coordinates have been transformed
        forward into a new frame. The origin is unchanged. Coordinate rotation
        is also performed on any subfields that are not Scalars.

        Input:
            frame       a Frame object to transform the coordinates. Its
                        reference frame must be the current frame of the event.
            quick       False to disable QuickPaths; True for the default
                        options; a dictionary to override specific options.
            derivs      if True, any vectors in the event are returned with
                        additional subfields "d_dt" and "d_dpos", representing
                        the partial derivatives in the new frame and path
                        relative to those in the original frame.
        """

        assert self.frame_id == frame.reference_id

        transform = frame.transform_at_time(self.time, quick)
        (pos, vel) = transform.rotate_pos_vel(self.pos, self.vel, derivs=derivs)
        result = Event(self.time, pos, vel, self.origin_id, frame.frame_id)

        for key in self.subfields:
            subfield = self.subfields[key]
            if isinstance(subfield, Array):
                result.insert_subfield(key, transform.rotate(subfield,
                                                             derivs=derivs))
            else:
                result.insert_subfield(key, subfield)

        result.filled_shape = self.filled_shape
        result.filled_mask  = self.filled_mask
        result.filled_ssb   = self.filled_ssb

        return result

    def unrotate_by_frame(self, frame, quick=QUICK, derivs=False):
        """Returns the same event after all coordinates have been transformed
        backward to the parent frame. The origin is unchanged. Subarrays that
        are not Scalars are also transformed.

        Input:
            frame       a Frame object to to inverse-transform the coordinates.
                        Its target frame must be the current frame of the event.
                        The returned event will use the reference frame instead.
            quick       False to disable QuickPaths; True for the default
                        options; a dictionary to override specific options.
            derivs      if True, any vectors in the event are returned with
                        additional subfields "d_dt", representing the rate of
                        change due to rotation of the frame.
        """

        assert self.frame_id == frame.frame_id

        transform = frame.transform_at_time(self.time, quick)
        (pos, vel) = transform.unrotate_pos_vel(self.pos, self.vel,
                                                derivs=derivs)
        result = Event(self.time, pos, vel, self.origin_id, frame.reference_id)

        for key in self.subfields:
            subfield = self.subfields[key]
            if isinstance(subfield, Array):
                result.insert_subfield(key, transform.unrotate(subfield,
                                                               derivs=derivs))
            else:
                result.insert_subfield(key, subfield)

        result.filled_shape = self.filled_shape
        result.filled_mask  = self.filled_mask
        result.filled_ssb   = self.filled_ssb

        return result

################################################################################
# Geometry procedures
################################################################################

    def aberrated_ray_ssb(self, ray_ssb, quick=QUICK):
        """Returns the apparent direction of a photon given its actual
        direction in the SSB/J2000 frame."""

        # This procedure is equivalent to a vector subtraction of the velocity
        # of the observer from the ray, given the ray has length C.

        return ray_ssb - (self.wrt_ssb(quick).vel +
                          self.wrt_ssb(quick).vflat)*ray_ssb.norm()/constants.C

    def aberrated_arr_ssb(self, quick=QUICK):
        """Aberrated arriving ray in the SSB/J2000 frame."""
        return self.aberrated_ray_ssb(self.wrt_ssb(quick).arr)

    def aberrated_arr(self, quick=QUICK):
        """Aberrated arriving ray in the event frame."""
        arr_ssb = self.aberrated_ray_ssb(self.wrt_ssb(quick).arr)

        frame = registry.connect_frames(self.frame_id, "J2000")
        xform = frame.transform_at_time(self.time, quick)
        return xform.rotate(arr_ssb)

    def aberrated_dep_ssb(self, quick=QUICK):
        """Aberrated departing ray in the SSB/J2000 frame."""
        return self.aberrated_ray_ssb(self.wrt_ssb(quick).dep)

    def aberrated_dep(self, quick=QUICK):
        """Aberrated departing ray in the event frame."""
        dep_ssb = self.aberrated_ray_ssb(self.wrt_ssb(quick).dep)

        frame = registry.connect_frames(self.frame_id, "J2000")
        xform = frame.transform_at_time(self.time, quick)
        return xform.rotate(dep_ssb)

    def incidence_angle(self, aberration=False, quick=QUICK):
        """Returns the incidence angle, measured between the surface normal and
        the reversed direction of the arriving photon."""

        return self.wrt_ssb(quick).perp.sep(self.aberrated_arr_ssb(),
                                            reversed=True)

    def emission_angle(self, quick=QUICK):
        """Returns the emission angle, measured between the surface normal and
        the direction of the departing photon."""

        return self.wrt_ssb(quick).perp.sep(self.aberrated_dep_ssb(quick))

    def phase_angle(self, quick=QUICK):
        """Returns the phase angle, measured between the direction of the
        arriving photon and the reversed direction of the departing photon."""

        return self.aberrated_arr_ssb(quick).sep(self.aberrated_dep_ssb(quick),
                                                 reversed=True)

    def ra_and_dec(self, aberration=False, subfield="arr", frame="J2000",
                         quick=QUICK):
        """Returns the J2000 right ascension amd declination in the path and
        frame of the event, as a tuple of two scalars.

        Input:
            aberration  True to include stellar aberration, thereby returning
                        the apparent direction of the photon relative to the
                        background stars; False to return the purely geometric
                        values, neglecting the motion of the observer.
            subfield    The subfield to use for the calculation, either "arr"
                        or "dep". Note that an arriving direction is reversed.
            frame       The frame in which the values should be returned. The
                        default is J2000, but B1950 might be useful under some
                        circumstances.
            quick       False to disable QuickPaths; True for the default
                        options; a dictionary to override specific options.
        """

        # Locate arrival ray in the SSB/J2000 frame
        assert subfield in {"arr", "dep"}

        if aberration:
            if subfield == "arr":
                ray = -self.aberrated_arr_ssb(quick)
            else:
                ray = self.aberrated_dep_ssb(quick)
        else:
            if subfield == "arr":
                ray = -self.wrt_ssb(quick).arr
            else:
                ray = self.wrt_ssb(quick).dep

        # Convert to RA and dec
        (x,y,z) = ray.as_scalars()
        ra = y.arctan2(x) % (2*np.pi)
        dec = (z/ray.norm()).arcsin()

        return (ra, dec)

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
