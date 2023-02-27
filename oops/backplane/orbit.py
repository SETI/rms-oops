################################################################################
# oops/backplanes/orbit.py: Orbit backplanes
################################################################################

from polymath import Vector3, Matrix3

from oops.backplane import Backplane
from oops.body      import Body
from oops.frame     import Frame
from oops.constants import TWOPI

#===============================================================================
def orbit_longitude(self, event_key, reference='obs', planet=None):
    """Longitude on an orbit path relative to the central planet.

    Input:
        event_key       key defining the event on the orbit path.
        reference       defines the location of zero longitude.
                        'aries' for the First point of Aries;
                        'node'  for the J2000 ascending node;
                        'obs'   for the sub-observer longitude;
                        'sun'   for the sub-solar longitude;
                        'oha'   for the anti-observer longitude;
                        'sha'   for the anti-solar longitude, returning the
                                solar hour angle.
        planet          ID of the body at the center of the orbit; None for the
                        default, which is the parent of the targeted body.
    """

    assert reference in {'aries', 'node', 'obs', 'oha', 'sun', 'sha'}

    # Determine/validate the planet
    event_key = self.standardize_event_key(event_key)
    if planet is None:
        (body,_) = self.get_body_and_modifier(event_key)
        planet = body.parent.name

    # Get the event
    if reference in {'sun', 'sha'}:
        orbit_event = self.get_gridless_event_with_arr(event_key)
    else:
        orbit_event = self.get_gridless_event(event_key)

    # Look up under the reference
    key0 = ('orbit_longitude', event_key)
    key = key0 + (reference, planet)
    if key in self.backplanes:
        return self.backplanes[key]

    planet_body = Body.lookup(planet)
    planet_event = planet_body.path.event_at_time(orbit_event.time)
    planet_event = planet_event.wrt_frame(Frame.J2000)
    orbit_event = orbit_event.wrt(planet_body.path, Frame.J2000)

    # Locate reference vector in the J2000 frame
    if reference == 'obs':
        reference_dir = orbit_event.dep_ap.wod
    elif reference == 'oha':
        reference_dir = -orbit_event.dep_ap.wod
    elif reference == 'sun':
        reference_dir = orbit_event.neg_arr_ap.wod
    elif reference == 'sha':
        reference_dir = orbit_event.arr_ap.wod
    elif reference == 'aries':
        reference_dir = Vector3.XAXIS
    else:           # 'node'
        pole = orbit_event.pos.cross(orbit_event.vel)
        reference_dir = Vector3.ZAXIS.cross(pole)

    # This matrix rotates J2000 coordinates to a frame in which the Z-axis
    # is the orbit pole and the moon is instantaneously at the X-axis.
    matrix = Matrix3.twovec(orbit_event.pos, 0, orbit_event.vel, 1)

    # Locate the reference direction in this frame
    reference_wrt_orbit = matrix.rotate(reference_dir)
    x = reference_wrt_orbit.to_scalar(0)
    y = reference_wrt_orbit.to_scalar(1)
    ref_lon_wrt_orbit = y.arctan2(x,recursive=False)

    # Convert to an orbit with respect to the reference direction
    orbit_lon_wrt_ref = (-ref_lon_wrt_orbit) % TWOPI
    self.register_gridless_backplane(key, orbit_lon_wrt_ref)
    return orbit_lon_wrt_ref

################################################################################

Backplane._define_backplane_names(globals().copy())

################################################################################
# UNIT TESTS
################################################################################
import unittest
from oops.constants import DPR
from oops.backplane.unittester_support import show_info

#===============================================================================
def exercise_longitude(bp,
                       planet=None, moon=None, ring=None,
                       undersample=16, use_inventory=False, inventory_border=2,
                       **options):
    """generic unit tests for orbit.py"""

    if moon is not None:
        test = bp.orbit_longitude(moon, reference='obs')
        show_info(bp, 'moon orbit longitude wrt observer (deg)', test*DPR, **options)
        test = bp.orbit_longitude(moon, reference='oha')
        show_info(bp, 'moon orbit longitude wrt OHA (deg)', test*DPR, **options)
        test = bp.orbit_longitude(moon, reference='sun')
        show_info(bp, 'moon orbit longitude wrt Sun (deg)', test*DPR, **options)
        test = bp.orbit_longitude(moon, reference='sha')
        show_info(bp, 'moon orbit longitude wrt SHA (deg)', test*DPR, **options)
        test = bp.orbit_longitude(moon, reference='aries')
        show_info(bp, 'moon orbit longitude wrt Aries (deg)', test*DPR, **options)
        test = bp.orbit_longitude(moon, reference='node')
        show_info(bp, 'moon orbit longitude wrt node (deg)', test*DPR, **options)


#*******************************************************************************
class Test_Orbit(unittest.TestCase):

    #===========================================================================
    def runTest(self):
        from oops.backplane.unittester_support import Backplane_Settings
        if Backplane_Settings.EXERCISES_ONLY:
            self.skipTest("")
        pass


########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
