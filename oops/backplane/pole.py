################################################################################
# oops/backplanes/pole.py: Pole angle backplanes
################################################################################

from polymath       import Scalar, Vector3, Matrix3
from oops.backplane import Backplane
from oops.frame     import Frame

def pole_clock_angle(self, event_key):
    """Gridless projected pole vector on the sky, measured from north through
    west.

    In other words, measured clockwise on the sky.
    """

    gridless_key = Backplane.gridless_event_key(event_key)

    key = ('pole_clock_angle', gridless_key)
    if key in self.backplanes:
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
    clock_angle = pole.longitude(recursive=self.ALL_DERIVS)

    return self.register_backplane(key, clock_angle)

#===============================================================================
def pole_position_angle(self, event_key):
    """Projected angle of a body's pole vector on the sky, measured from
    celestial north toward celestial east (i.e., counterclockwise on the sky).
    """

    event_key = Backplane.standardize_event_key(event_key)

    key = ('pole_position_angle', event_key)
    if key in self.backplanes:
        return self.get_backplane(key)

    clock = self.pole_clock_angle(event_key)
    return self.register_backplane(key, Scalar.TWOPI - clock)

################################################################################

Backplane._define_backplane_names(globals().copy())

################################################################################
