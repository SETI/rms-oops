################################################################################
# oops_/edelta.py: Event-delta class
#
# 3/2/12 Created (MRS).
################################################################################

import numpy as np
import unittest

from oops_.event import Event
from oops_.array.all import *
import oops_.registry as registry
import oops_.constants as constants

class Edelta(object):
    """An Edelta object is defined by a time, position and velocity as offsets
    relative to another event. Note that because the times of the two events can
    differ.
    """

    def __init__(self, time, pos, vel, event, **subfields):
        """Constructor for the Edelta class.

        Input:
            time        a Scalar of event times offsets in seconds.
            pos         position offset vectors as a Vector3 object.
            vel         velocity offset vectors as a Vector3 object.
            event       the event relative to which this event is defined.

        These attributes are defined by the constructor, based on the origin
        event.
            epoch       the time of the origin event.
            origin_id   the ID of a path defining the origin of this event.
            frame_id    the ID of a frame identifying the coordinate system.
                        Default is to match the frame of the origin.

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

        self.event = event
        self.epoch = event.time
        self.origin_id = event.origin_id
        self.frame_id = event.frame_id

        self.subfields = {}
        self.insert_subfield("arr", Empty())            # These always exist
        self.insert_subfield("dep", Empty())
        self.insert_subfield("vflat", Vector3([0.,0.,0.])) # default

        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        self.filled_shape = None
        self.filled_mask  = None
        self.filled_ssb   = None

    @property
    def shape(self):
        if self.filled_shape is None:
            self.filled_shape = Array.broadcast_shape([self.time,
                                                       self.pos, self.vel])
        return self.filled_shape

    @property
    def mask(self):
        if self.filled_mask is None:
            self.filled_mask = (self.time.mask | self.pos.mask | self.vel.mask |
                                self.arr.mask | self.dep.mask)

        return self.filled_mask

    def wrt_ssb(self, quick=None):
        """Returns an equivalent Event object, defined in absolute terms
        relative to the Solar System Barycenter and the J2000 frame."""

        if self.filled_ssb is None:
            frame = registry.connect_frames(self.frame_id, "J2000")
            transform = frame.transform_at_time(self.epoch, quick)

            (pos, vel) = transform.unrotate_pos_vel(self.pos, self.vel)

            event = self.event.wrt_ssb(quick).copy()
            event.pos  += pos
            event.vel  += vel
            event.time += self.time

            for key in self.subfields.keys():
                subfield = self.subfields[key]
                event.insert_subfield(key, transform.unrotate(subfield))

            self.filled_ssb = event

        return self.filled_ssb

    def copy(self):
        """Returns a copy of the Edelta object with all attributes themselves
        copied."""

        result = Event(self.time.copy(), self.pos.copy(), self.vel.copy(),
                       self.origin_id, self.frame_id)

        for key in self.subfields.keys():
            result.insert_subfield(key, self.subfields[key])

        result.filled_shape = self.filled_shape
        result.filled_mask  = self.filled_mask
        result.filled_ssb   = self.filled_ssb

    ####################################################
    # Subarray support methods
    ####################################################

    def insert_subfield(self, key, value):
        """Adds a given subfield to the Event."""

        self.subfields[key] = value
        self.__dict__[key] = value      # This makes it an attribute as well

    def delete_subfield(self, key):
        """Deletes a subfield, but not arr or dep."""

        if key in ("arr","dep"):
            self.subfields[key] = Empty()
            self.__dict__[key] = self.subfields[key]
        elif key in self.subfields.keys():
            del self.subfields[key]
            del self.__dict__[key]

    def delete_subfields(self):
        """Deletes all subfields."""

        for key in self.subfields.key():
            if key not in ("arr","dep"):
                del self.subfields[key]
                del self.__dict__[key]

    def add_to_subfield(self, key, value):
        """Adds to an existing subfield of the same name, or else inserts a new
        subfield with this value."""

        if key in self.subfields.keys():
            self.subfields[key] = self.subfields[key] + value 
            self.__dict__[key] = self.subfields[key]
            return

        self.insert_subfield(key, value)

    ####################################################
    # Event subtraction
    ####################################################

    @staticmethod
    def sub_events(event, origin, quick=None):
        """Returns the difference between two events as an Edelta object. The
        difference is returned in the frame of the origin."""

        event_ssb = event.wrt_ssb(quick)
        origin_ssb = origin.wrt_ssb(quick)

        frame = registry.connect_frames(origin.frame_id, "J2000")
        transform = frame.transform_at_time(origin.time, quick)

        (pos, vel) = transform.rotate_pos_vel(event_ssb.pos - origin_ssb.pos,
                                              event_ssb.vel - origin_ssb.vel)

        delta = Edelta(event_ssb.time - origin.time, pos, vel, origin)

        for key in event_ssb.subfields.keys():
            subfield = event_ssb.subfields[key]
            delta.insert_subfield(key, transform.rotate(subfield))

        return delta

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
