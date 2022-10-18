################################################################################
# oops/backplanes/ansa_backplanes.py: Ansa backplanes
################################################################################

from polymath import Scalar, Pair, Vector3

from oops.backplane import Backplane
from oops.event     import Event
from oops.path      import AliasPath
from oops.constants import TWOPI

#===============================================================================
def ansa_radius(self, event_key, radius_type='positive', rmax=None,
                      lock_limits=False):
    """Radius of the ring ansa intercept point in the image.

    Input:
        event_key       key defining the ring surface event.
        radius_type     'right' for radii increasing rightward when prograde
                                rotation pole is 'up';
                        'left' for the opposite of 'right';
                        'positive' for all radii using positive values.
        rmax            maximum absolute value of the radius in km; None to
                        allow it to be defined by the event_key.
        lock_limits     if True, the rmax value will be applied to the default
                        event, so that all backplanes generated from this
                        event_key will have the same limits. This option can
                        only be applied the first time this event_key is used.
    """

    # Look up under the desired radius type
    event_key = self.standardize_event_key(event_key)
    key0 = ('ansa_radius', event_key)
    key = key0 + (radius_type, rmax)
    if key in self.backplanes:
        return self.backplanes[key]

    # If not found, look up the default 'right'
    assert radius_type in ('right', 'left', 'positive')

    key_default = key0 + ('right', None)
    if key_default not in self.backplanes:
        self._fill_ansa_intercepts(event_key, rmax, lock_limits)

    key_right = key0 + ('right', rmax)
    backplane = self.backplanes[key_right]

    if radius_type == 'left':
        self.register_backplane(key, -backplane)
    elif radius_type == 'positive':
        self.register_backplane(key, backplane.abs())

    return self.backplanes[key]

#===============================================================================
def ansa_altitude(self, event_key):
    """Elevation of the ring ansa intercept point in the image.

    Input:
        event_key       key defining the ring surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('ansa_altitude', event_key)
    if key not in self.backplanes:
        self._fill_ansa_intercepts(event_key)

    return self.backplanes[key]

#===============================================================================
def ansa_longitude(self, event_key, reference='node'):
    """Longitude of the ansa intercept point in the image.

    Input:
        event_key       key defining the ring surface event.
        reference       defines the location of zero longitude.
                        'aries' for the First point of Aries;
                        'node'  for the J2000 ascending node;
                        'obs'   for the sub-observer longitude;
                        'sun'   for the sub-solar longitude;
                        'oha'   for the anti-observer longitude;
                        'sha'   for the anti-solar longitude, returning the
                                solar hour angle.
    """

    event_key = self.standardize_event_key(event_key)
    assert reference in {'aries', 'node', 'obs', 'oha', 'sun', 'sha'}

    # Look up under the desired reference
    key0 = ('ansa_longitude', event_key)
    key = key0 + (reference,)
    if key in self.backplanes:
        return self.backplanes[key]

    # If it is not found with reference J2000, fill in those backplanes
    key_node = key0 + ('node',)
    if key_node not in self.backplanes:
        self._fill_ansa_longitudes(event_key)

    # Now apply the reference longitude
    if reference == 'node':
        return self.backplanes[key]

    if reference == 'aries':
        ref_lon = self._aries_ring_longitude(event_key)
    elif reference == 'sun':
        ref_lon = self._sub_solar_ansa_longitude(event_key)
    elif reference == 'sha':
        ref_lon = self._sub_solar_ansa_longitude(event_key) - Scalar.PI
    elif reference == 'obs':
        ref_lon = self._sub_observer_ansa_longitude(event_key)
    elif reference == 'oha':
        ref_lon = self._sub_observer_ansa_longitude(event_key) - Scalar.PI

    lon = (self.backplanes[key_node] - ref_lon) % TWOPI
    self.register_backplane(key, lon)

    return self.backplanes[key]

#===============================================================================
def _fill_ansa_intercepts(self, event_key, rmax=None, lock_limits=False):
    """Internal method to fill in the ansa intercept geometry backplanes.

    Input:
        rmax            maximum absolute value of the radius in km; None to
                        allow it to be defined by the event_key.
        lock_limits     if True, the rmax value will be applied to the
                        default event, so that all backplanes generated
                        from this event_key will have the same limits. This
                        option can only be applied the first time this
                        event_key is used.
    """

    # Don't allow lock_limits if the backplane was already generated
    if rmax is None:
        lock_limits = False

    if lock_limits and event_key in self.surface_events:
        raise ValueError('lock_limits option disallowed for pre-existing ' +
                         'ansa event key ' + str(event_key))

    # Get the ansa intercept coordinates
    event = self.get_surface_event(event_key)
    if event.surface.COORDINATE_TYPE != 'cylindrical':
        raise ValueError('ansa intercepts require a "cylindrical" ' +
                         'surface type')

    # Limit the event if necessary
    if lock_limits:

        # Apply the upper limit to the event
        radius = event.coord1.abs()
        self.apply_mask_to_event(event_key, radius > rmax)
        event = self.get_surface_event(event_key)

    # Register the default backplanes
    self.register_backplane(('ansa_radius', event_key, 'right', None),
                            event.coord1)
    self.register_backplane(('ansa_altitude', event_key),
                            event.coord2)

    # Apply a mask just to these backplanes if necessary
    if rmax is not None:
        mask = (event.coord1 > rmax)
        self.register_backplane(('ansa_radius', event_key, 'right', rmax),
                                event.coord1.mask_where(mask))
        self.register_backplane(('ansa_altitude', event_key, rmax),
                                event.coord2.mask_where(mask))

#===============================================================================
def _fill_ansa_longitudes(self, event_key):
    """Internal method to fill in the ansa intercept longitude backplane."""

    # Get the ansa intercept event
    event_key = self.standardize_event_key(event_key)
    event = self.get_surface_event(event_key)
    assert event.surface.COORDINATE_TYPE == 'cylindrical'

    # Get the longitude in the associated ring plane
    lon = event.surface.ringplane.coords_from_vector3(event.pos, axes=2)[1]
    self.register_backplane(('ansa_longitude', event_key, 'node'),
                            lon)

#===============================================================================
def _sub_observer_ansa_longitude(self, event_key):
    """Sub-observer longitude evaluated at the ansa intercept time.

    Used only internally. DEPRECATED.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('_sub_observer_ansa_longitude', event_key)
    if key in self.backplanes:
        return self.backplanes[key]

    # At each intercept time, determine the outgoing direction to the
    # observer from the center of the planet
    event = self.get_surface_event(event_key)
    center_event = Event(event.time, Vector3.ZERO,
                                     event.origin, event.frame)
    center_event = self.obs_event.origin.photon_from_event(center_event)[1]

    surface = self.get_surface(event_key).ringplane
    (r,lon) = surface.coords_from_vector3(center_event.apparent_dep(),
                                          axes=2)

    self.register_gridless_backplane(key, lon)
    return self.backplanes[key]

#===============================================================================
def _sub_solar_ansa_longitude(self, event_key):
    """Sub-solar longitude evaluated at the ansa intercept time.

    Used only internally. DEPRECATED.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('_sub_solar_ansa_longitude', event_key)
    if key in self.backplanes:
        return self.backplanes[key]

    # At each intercept time, determine the incoming direction from the
    # Sun to the center of the planet
    event = self.get_surface_event(event_key)
    center_event = Event(event.time, Vector3.ZERO,
                                     event.origin, event.frame)
    center_event = AliasPath('SUN').photon_to_event(center_event)[1]

    surface = self.get_surface(event_key).ringplane
    (r,lon) = surface.coords_from_vector3(-center_event.apparent_arr(),
                                          axes=2)

    self.register_gridless_backplane(key, lon)
    return self.backplanes[key]

#===============================================================================
def ansa_radial_resolution(self, event_key):
    """Projected radial resolution in km/pixel at the ring ansa intercept.

    Input:
        event_key       key defining the ring surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('ansa_radial_resolution', event_key)
    if key in self.backplanes:
        return self.backplanes[key]

    event = self.get_surface_event_w_derivs(event_key)
    assert event.surface.COORDINATE_TYPE == 'cylindrical'

    r = event.coord1
    dr_duv = r.d_dlos.chain(self.dlos_duv)
    res = dr_duv.join_items(Pair).norm()

    self.register_backplane(key, res)
    return self.backplanes[key]

#===============================================================================
def ansa_vertical_resolution(self, event_key):
    """Projected radial resolution in km/pixel at the ring ansa intercept.

    Input:
        event_key       key defining the ring surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('ansa_vertical_resolution', event_key)
    if key in self.backplanes:
        return self.backplanes[key]

    event = self.get_surface_event_w_derivs(event_key)
    assert event.surface.COORDINATE_TYPE == 'cylindrical'

    z = event.coord2
    dz_duv = z.d_dlos.chain(self.dlos_duv)
    res = dz_duv.join_items(Pair).norm()

    self.register_backplane(key, res)
    return self.backplanes[key]

################################################################################

Backplane._define_backplane_names(globals().copy())

################################################################################




################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops.meshgrid                       import Meshgrid
from oops.unittester_support             import TESTDATA_PARENT_DIRECTORY
from oops.constants                      import DPR
from oops.backplane.unittester_support   import show_info


#===========================================================================
def exercise_resolution(bp, obs, printing, saving, dir, 
                        planet=None, moon=None, ring=None, 
                        undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for orbit.py"""

    if planet != None:
        test = bp.ansa_radial_resolution(planet+':ansa')
        show_info('Ansa radial resolution (km)', test, 
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ansa_vertical_resolution(planet+':ansa')
        show_info('Ansa vertical resolution (km)', test,  
                                    printing=printing, saving=saving, dir=dir)



#===========================================================================
def exercise_geometry(bp, obs, printing, saving, dir, 
                        planet=None, moon=None, ring=None, 
                        undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for orbit.py"""
    
    if planet != None:
        test = bp.ansa_radius(planet+':ansa')
        show_info('Ansa radius (km)', test,  
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ansa_altitude(planet+':ansa')
        show_info('Ansa altitude (km)', test,  
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ansa_longitude(planet+':ansa', 'node')
        show_info('Ansa longitude wrt node (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ansa_longitude(planet+':ansa', 'aries')
        show_info('Ansa longitude wrt Aries (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ansa_longitude(planet+':ansa', 'obs')
        show_info('Ansa longitude wrt observer (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ansa_longitude(planet+':ansa', 'oha')
        show_info('Ansa longitude wrt OHA (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ansa_longitude(planet+':ansa', 'sun')
        show_info('Ansa longitude wrt Sun (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ansa_longitude(planet+':ansa', 'sha')
        show_info('Ansa longitude wrt SHA (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)




#*******************************************************************************
class Test_Ansa(unittest.TestCase):

    #===========================================================================
    def runTest(self):
        from oops.backplane.unittester_support import Backplane_Settings
        if Backplane_Settings.EXERCISES_ONLY: 
            return
        pass


########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
