##########################################################################################
# oops/backplane/pole.py
##########################################################################################

from polymath       import Scalar, Vector3, Matrix3
from oops.backplane import Backplane
from oops.frame     import Frame


def pole_clock_angle(self, event_key):
    """Gridless clock angle of the body's projected pole vector on the sky.

    Measured from north through west; in other words, clockwise on the sky. The value is
    in radians.

    Parameters:
        event_key (str or tuple): Key defining the event at the body's path.
    """

    self.refresh()
    gridless_key = Backplane.gridless_event_key(event_key)

    key = ('pole_clock_angle', gridless_key)
    if key in self._backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(gridless_key)

    # Get the body frame's Z-axis in J2000 coordinates
    frame = Frame.J2000.wrt(event.frame)
    xform = frame.transform_at_time(event.time)
    pole_j2000 = xform.rotate(Vector3.ZAXIS)

    # Define the vector to the observer in the J2000 frame
    dep_j2000 = event.wrt_ssb().dep_ap

    # Construct a rotation matrix from J2000 to a frame in which the Z-axis
    # points along -dep and the J2000 pole is in the X-Z plane. As it
    # appears to the observer, the Z-axis points toward the body, the X-axis
    # points toward celestial north as projected on the sky, and the Y-axis
    # points toward celestial west (not east!).
    rotmat = Matrix3.twovec(-dep_j2000, 2, Vector3.ZAXIS, 0)

    # Rotate the body frame's Z-axis to this frame.
    pole = rotmat * pole_j2000

    # Convert the X and Y components of the rotated pole into an angle
    clock_angle = pole.longitude(recursive=self._ALL_DERIVS)

    return self.register_backplane(key, clock_angle)


def pole_position_angle(self, event_key):
    """The projected angle of a body's pole vector on the sky.

    The angle is measured from celestial north toward celestial east, that is,
    counterclockwise on the sky.

    This is the complement of the clock angle, in radians.

    Parameters:
        event_key (str or tuple): Key defining the event at the body's path.
    """

    self.refresh()
    gridless_key = Backplane.gridless_event_key(event_key)

    key = ('pole_position_angle', gridless_key)
    if key in self._backplanes:
        return self.get_backplane(key)

    clock = self.pole_clock_angle(gridless_key)
    return self.register_backplane(key, Scalar.TWOPI - clock)

##########################################################################################

Backplane._define_backplane_names(globals().copy())

##########################################################################################
