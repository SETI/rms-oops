################################################################################
# oops/backplanes/pole.py: Pole angle backplanes
################################################################################

from polymath import Scalar, Vector3, Matrix3

from oops.backplane import Backplane
from oops.frame     import Frame
from oops.constants import TWOPI

#===============================================================================
def pole_clock_angle(self, event_key):
    """Projected pole vector on the sky, measured from north through wes.

    In other words, measured clockwise on the sky.
    """

    event_key = self.standardize_event_key(event_key)

    key = ('pole_clock_angle', event_key)
    if key not in self.backplanes:
        event = self.get_gridless_event(event_key)

        # Get the body frame's Z-axis in J2000 coordinates
        frame = Frame.J2000.wrt(event.frame)
        xform = frame.transform_at_time(event.time)
        pole_j2000 = xform.rotate(Vector3.ZAXIS)

        # Define the vector to the observer in the J2000 frame
        dep_j2000 = event.wrt_ssb().apparent_dep()

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
        clock_angle = coords[1].arctan2(coords[0]) % TWOPI

        self.register_gridless_backplane(key, clock_angle)

    return self.backplanes[key]

#===============================================================================
def pole_position_angle(self, event_key):
    """Projected angle of a body's pole vector on the sky, measured from
    celestial north toward celestial east (i.e., counterclockwise on the sky).
    """

    event_key = self.standardize_event_key(event_key)

    key = ('pole_position_angle', event_key)
    if key not in self.backplanes:
        self.register_gridless_backplane(key,
                        Scalar.TWOPI - self.pole_clock_angle(event_key))

    return self.backplanes[key]

################################################################################

Backplane._define_backplane_names(globals().copy())

################################################################################


################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops.meshgrid     import Meshgrid
from oops.unittester_support import TESTDATA_PARENT_DIRECTORY
from oops.constants    import DPR
from oops.backplane.unittester_support    import show_info

#===============================================================================
def exercise(bp, printing, saving, dir, refdir,
             planet=None, moon=None, ring=None,
             undersample=16, use_inventory=False, inventory_border=2):
    """generic unit tests for pole.py"""

    if planet is not None:
        test = bp.pole_clock_angle(planet)
        show_info(bp, 'planet pole clock angle (deg)', test*DPR,
                  printing=printing, saving=saving, dir=dir, refdir=refdir)

        test = bp.pole_position_angle(planet)
        show_info(bp, 'planet pole position angle (deg)', test*DPR,
                  printing=printing, saving=saving, dir=dir, refdir=refdir)


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
