################################################################################
# oops/backplanes/pole.py: Pole angle backplanes
################################################################################

from polymath import Scalar, Vector3, Matrix3

from oops.backplane import Backplane
from oops.frame     import Frame

def pole_clock_angle(self, event_key):
    """Gridless projected pole vector on the sky, measured from north through
    west.

    In other words, measured clockwise on the sky.
    """

    gridless_key = self.gridless_event_key(event_key)

    key = ('pole_clock_angle', gridless_key)
    if key in self.backplanes:
        return self.backplanes[key]

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
    coords = pole.to_scalars()
    clock_angle = coords[1].arctan2(coords[0]) % Scalar.TWOPI

    return self.register_backplane(key, clock_angle)

#===============================================================================
def pole_position_angle(self, event_key):
    """Projected angle of a body's pole vector on the sky, measured from
    celestial north toward celestial east (i.e., counterclockwise on the sky).
    """

    event_key = self.standardize_event_key(event_key)

    key = ('pole_position_angle', event_key)
    if key in self.backplanes:
        return self.backplanes[key]

    return self.register_backplane(key, Scalar.TWOPI
                                        - self.pole_clock_angle(event_key))

################################################################################

Backplane._define_backplane_names(globals().copy())

################################################################################
# GOLD MASTER TESTS
################################################################################

from oops.backplane.gold_master import register_test_suite
from oops.constants import DPR

def pole_test_suite(bpt):

    bp = bpt.backplane
    for name in bpt.body_names + bpt.ring_names:
        clock = bp.pole_clock_angle(name) * DPR
        position = bp.pole_position_angle(name) * DPR
        bpt.gmtest(clock,
                   name + ' pole clock angle (deg)',
                   method='mod360', limit=0.001)
        bpt.gmtest(position,
                   name + ' pole position angle (deg)',
                   method='mod360', limit=0.001)
        bpt.compare(clock + position, 0.,
                    name + ' pole clock plus position angle (deg)',
                    method='mod360', limit=1.e-13)

register_test_suite('pole', pole_test_suite)

################################################################################
# UNIT TESTS
################################################################################
import unittest
from oops.constants import DPR
from oops.backplane.unittester_support import show_info

#===============================================================================
def exercise(bp,
             planet=None, moon=None, ring=None,
             undersample=16, use_inventory=False, inventory_border=2,
             **options):
    """generic unit tests for pole.py"""

    if planet is not None:
        test = bp.pole_clock_angle(planet)
        show_info(bp, 'planet pole clock angle (deg)', test*DPR, **options)
        test = bp.pole_position_angle(planet)
        show_info(bp, 'planet pole position angle (deg)', test*DPR, **options)


#*******************************************************************************
class Test_Pole(unittest.TestCase):

    #===========================================================================
    def runTest(self):
        from oops.backplane.unittester_support import Backplane_Settings
        if Backplane_Settings.EXERCISES_ONLY:
            self.skipTest("")
        pass


########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
