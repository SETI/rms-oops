from __future__ import print_function

import numpy as np
import pylab
import oops
import oops.inst.cassini.vims as cassini_vims

PRINT = True
DISPLAY = True

#===============================================================================
# show_info
#===============================================================================
def show_info(title, array):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Internal method to print(summary information and display images as)
    desired.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    global PRINT, DISPLAY
    if not PRINT: return
    
    print("")
    print(title)
    
    if isinstance(array, np.ndarray):
        if array.dtype == np.dtype("bool"):
            count = np.sum(array)
            total = np.size(array)
            percent = int(count / float(total) * 100. + 0.5)
            print("   ", (count, total-count), end='')
            print((percent, 100-percent), "(True, False pixels)")
            if DISPLAY:
                ignore = pylab.imshow(array, norm=None, vmin=0, vmax=1)
                ignore = raw_input(title + ": ")
        
        else:
            minval = np.min(array)
            maxval = np.max(array)
            if minval == maxval:
                print("    ", minval)
            else:
                print("    ", (minval, maxval), "(min, max)")
                
                if DISPLAY:
                    ignore = pylab.imshow(array)
                ignore = raw_input(title + ": ")
    
    elif isinstance(array, oops.Array):
        if np.any(array.mask):
            print("    ", np.min(array.vals), end='')
            print(        np.max(array.vals), "(unmasked min, max)")
            print("    ", array.min(), end='')
            print(        array.max(), "(masked min, max)")
#            print("    ", (np.min(array.vals), end='')
#                           np.max(array.vals)), "(unmasked min, max)"
#            print("    ", (array.min(), end='')
#                           array.max()), "(masked min, max)"
            masked = np.sum(array.mask)
            total = np.size(array.mask)
            percent = int(masked / float(total) * 100. + 0.5)
            print("    ", (masked, total-masked), end='')
            print(        (percent, 100-percent), "(masked, unmasked pixels)")
            
            if DISPLAY and array.vals.size > 1:
                ignore = pylab.imshow(array.vals)
                ignore = raw_input(title + ": ")
                background = np.zeros(array.shape, dtype="uint8")
                background[0::2,0::2] = 1
                background[1::2,1::2] = 1
                ignore = pylab.imshow(background)
                ignore = pylab.imshow(array.mvals)
                ignore = raw_input(title + ": ")
        
        else:
            minval = np.min(array.vals)
            maxval = np.max(array.vals)
            if minval == maxval:
                print("    ", minval)
            else:
                print("    ", (minval, maxval), "(min, max)")
                
                if DISPLAY:
                    ignore = pylab.imshow(array.vals)
                    ignore = raw_input(title + ": ")
    
    else:
        print("    ", array)
#===============================================================================



#===============================================================================
# vims_test_suite
#===============================================================================
def vims_test_suite(filespec, derivs, info, display):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Master test suite for a Cassini VIMS image.
        
        Input:
        filespec    file path and name to a Cassini VIMS image file.
        derivs      True to calculate derivatives where needed to derive
        quantities related to spatial resolution; False to omit all
        resolution calculations.
        info        True to print(out geometry information as it progresses.)
        display     True to display each backplane using Pylab, and pause until
        the user hits RETURN.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    global PRINT, DISPLAY
    PRINT = info
    DISPLAY = display
    
    #---------------------------------------
    # Define the bodies we care about
    #---------------------------------------
    ring_body = oops.registry.body_lookup("SATURN_MAIN_RINGS")
    saturn_body = oops.registry.body_lookup("SATURN")
    sun_body = oops.registry.body_lookup("SUN")
    
    #-----------------------------------
    # Create the pushbroom objects
    #-----------------------------------
    pushbrooms = cassini_vims.from_file(filespec)
    
    #------------------------------------------------------
    # Create the pusbroom event for the visual image
    # ... with a grid point at the middle of each pixel
    #------------------------------------------------------
    fov_shape = pushbrooms[0].fov.uv_shape
    
    uv_pair = oops.Pair.cross_scalars(np.arange(fov_shape.vals[0]) + 0.5,
                                      np.arange(fov_shape.vals[1]) + 0.5)
    
    #--------------------------------------------------------------------------
    # los.d_duv is now the [3,2] MatrixN of derivatives dlos/d(u,v), where los
    # is in the frame of the Cassini camera....
    #--------------------------------------------------------------------------
    los = pushbrooms[0].fov.los_from_uv(uv_pair, derivs=derivs)
    
    #--------------------------------------------------------------------------
    # This line swaps the image for proper display using pylab.imshow().
    # Also change sign for incoming photons, and for subfield "d_duv".
    #--------------------------------------------------------------------------
    arrivals = -los.swapaxes(0,1)
    
    #--------------------------------------------------------------------------
    # Define the event as a 1024x1024 array of simultaneous photon arrivals
    # coming from slightly different directions
    #--------------------------------------------------------------------------
    pushbroom_event = oops.Event(pushbrooms[0].midtime, (0.,0.,0.), (0.,0.,0.),
                                 pushbrooms[0].path_id, pushbrooms[0].frame_id,
                                 arr=arrivals)
    
    #----------------------------------------------------
    # For single-point calculations about the geometry
    #----------------------------------------------------
    point_event = oops.Event(pushbrooms[0].midtime, (0.,0.,0.), (0.,0.,0.),
                             pushbrooms[0].path_id, pushbrooms[0].frame_id)
    
    ############################################
    # Sky coordinates
    ############################################
    
    (right_ascension, declination) = pushbroom_event.ra_and_dec()
    #show_info("Right ascension (deg)", right_ascension * oops.DPR)
    #show_info("Declination (deg)", declination * oops.DPR)
    
    pushbrooms[0].insert_subfield("right_ascension", right_ascension)
    pushbrooms[0].insert_subfield("declination", declination)
    
    ############################################
    # Sub-observer ring geometry
    ############################################
    
    #--------------------------------------------------------------------------
    # Define the apparent location of the observer relative to Saturn ring frame
    #--------------------------------------------------------------------------
    ring_center_event = ring_body.path.photon_to_event(point_event)
    ring_center_event = ring_center_event.wrt_frame(ring_body.frame_id)
    
    #-------------------------------------------------
    # Event separation in ring surface coordinates
    #-------------------------------------------------
    obs_wrt_ring_center = oops.Edelta.sub_events(point_event,
                                                 ring_center_event)
    
    obs_wrt_ring_range = obs_wrt_ring_center.pos.norm()
    
    (obs_wrt_ring_radius,
     obs_wrt_ring_longitude,
     obs_wrt_ring_elevation) = ring_body.surface.event_as_coords(obs_wrt_ring_center.event,
                                                                 axes=3)
    
    show_info("Ring range to observer (km)", obs_wrt_ring_range)
    show_info("Ring radius of observer (km)", obs_wrt_ring_radius)
    show_info("Ring longitude of observer(deg)", obs_wrt_ring_longitude *
              oops.DPR)
    show_info("Ring elevation of observer(km)", obs_wrt_ring_elevation)
    
    pushbrooms[0].insert_subfield("obs_wrt_ring_range", obs_wrt_ring_range)
    pushbrooms[0].insert_subfield("obs_wrt_ring_radius", obs_wrt_ring_radius)
    pushbrooms[0].insert_subfield("obs_wrt_ring_longitude", obs_wrt_ring_longitude)
    pushbrooms[0].insert_subfield("obs_wrt_ring_elevation", obs_wrt_ring_elevation)
    
    ############################################
    # Sub-solar ring geometry
    ############################################
    
    #----------------------------------------------------------------------
    # Define the apparent location of the Sun relative to the ring frame
    #----------------------------------------------------------------------
    sun_center_event = sun_body.path.photon_to_event(ring_center_event)
    sun_center_event = sun_center_event.wrt_frame(ring_body.frame_id)
    
    #-------------------------------------------------
    # Event separation in ring surface coordinates
    #-------------------------------------------------
    sun_wrt_ring_center = oops.Edelta.sub_events(sun_center_event,
                                                 ring_center_event)
    
    sun_wrt_ring_range = sun_wrt_ring_center.pos.norm()
    
    (sun_wrt_ring_radius,
     sun_wrt_ring_longitude,
     sun_wrt_ring_elevation) = ring_body.surface.event_as_coords(sun_wrt_ring_center.event,
                                                                 axes=3)
    
    show_info("Ring range to Sun (km)", sun_wrt_ring_range)
    show_info("Ring radius of Sun (km)", sun_wrt_ring_radius)
    show_info("Ring longitude of Sun (deg)", sun_wrt_ring_longitude * oops.DPR)
    show_info("Ring elevation of Sun (km)", sun_wrt_ring_elevation)
    
    pushbrooms[0].insert_subfield("sun_wrt_ring_range", sun_wrt_ring_range)
    pushbrooms[0].insert_subfield("sun_wrt_ring_radius", sun_wrt_ring_radius)
    pushbrooms[0].insert_subfield("sun_wrt_ring_longitude", sun_wrt_ring_longitude)
    pushbrooms[0].insert_subfield("sun_wrt_ring_elevation", sun_wrt_ring_elevation)
    
    ############################################
    # Sub-observer Saturn geometry
    ############################################
    
    #--------------------------------------------------------------------------
    # Define the apparent location of the observer relative to Saturn frame
    #--------------------------------------------------------------------------
    saturn_center_event = saturn_body.path.photon_to_event(point_event)
    saturn_center_event = saturn_center_event.wrt_frame(saturn_body.frame_id)
    
    #----------------------------------------------------
    # Event separation in Saturn surface coordinates
    #----------------------------------------------------
    obs_wrt_saturn_center = oops.Edelta.sub_events(point_event,
                                                   saturn_center_event)
    
    obs_wrt_saturn_range = obs_wrt_saturn_center.pos.norm()
    
    (obs_wrt_saturn_longitude,
     obs_wrt_saturn_latitude,
     obs_wrt_saturn_elevation) = saturn_body.surface.event_as_coords(obs_wrt_saturn_center.event,
                                                                     axes=3)
    #print("obs_wrt_saturn_longitude.shape: ", obs_wrt_saturn_longitude.shape)
    #print(this_doesnt_work)

    show_info("Saturn range to observer (km)", obs_wrt_saturn_range)
    show_info("Saturn longitude of observer (deg)", obs_wrt_saturn_longitude *
              oops.DPR)
    show_info("Saturn latitude of observer (km)", obs_wrt_saturn_latitude *
              oops.DPR)
    show_info("Saturn elevation of observer (km)", obs_wrt_saturn_elevation)
    
    pushbrooms[0].insert_subfield("obs_wrt_saturn_range", obs_wrt_saturn_range)
    pushbrooms[0].insert_subfield("obs_wrt_saturn_longitude",
                             obs_wrt_saturn_longitude)
    pushbrooms[0].insert_subfield("obs_wrt_saturn_latitude", obs_wrt_saturn_latitude)
    pushbrooms[0].insert_subfield("obs_wrt_saturn_elevation",
                             obs_wrt_saturn_elevation)
    
    ############################################
    # Sub-solar Saturn geometry
    ############################################
    
    #--------------------------------------------------------------------------
    # Define the apparent location of the Sun relative to the Saturn frame
    #--------------------------------------------------------------------------
    sun_center_event = sun_body.path.photon_to_event(saturn_center_event)
    sun_center_event = sun_center_event.wrt_frame(saturn_body.frame_id)
    
    #---------------------------------------------------
    # Event separation in Saturn surface coordinates
    #---------------------------------------------------
    sun_wrt_saturn_center = oops.Edelta.sub_events(sun_center_event,
                                                   saturn_center_event)
    
    sun_wrt_saturn_range = sun_wrt_saturn_center.pos.norm()
    
    (sun_wrt_saturn_longitude,
     sun_wrt_saturn_latitude,
     sun_wrt_saturn_elevation) = saturn_body.surface.event_as_coords(sun_wrt_saturn_center.event,
                                                                     axes=3)
    
    show_info("Saturn range to Sun (km)", sun_wrt_saturn_range)
    show_info("Saturn longitude of Sun (deg)", sun_wrt_saturn_longitude *
              oops.DPR)
    show_info("Saturn latitude of Sun (km)", sun_wrt_saturn_latitude *
              oops.DPR)
    show_info("Saturn elevation of Sun (km)", sun_wrt_saturn_elevation)
    
    pushbrooms[0].insert_subfield("sun_wrt_saturn_range", sun_wrt_saturn_range)
    pushbrooms[0].insert_subfield("sun_wrt_saturn_longitude",
                             sun_wrt_saturn_longitude)
    pushbrooms[0].insert_subfield("sun_wrt_saturn_latitude", sun_wrt_saturn_latitude)
    pushbrooms[0].insert_subfield("sun_wrt_saturn_elevation",
                             sun_wrt_saturn_elevation)
    
    ############################################
    # Ring intercept points in pushbroom
    ############################################
    
    #------------------------------------
    # Find the ring intercept events
    #------------------------------------
    ring_event_w_derivs = ring_body.surface.photon_to_event(pushbroom_event,
                                                            derivs=derivs)
    #ring_event = ring_event_w_derivs.copy()
    #ring_event.delete_sub_subfields()
    ring_event = ring_event_w_derivs.plain()
    print("ring_event.pos: ", ring_event.pos)
    
    #------------------------------------------------------
    # This mask is True inside the rings, False outside
    #------------------------------------------------------
    ring_mask = np.logical_not(ring_event.mask)
    
    #------------------
    # Get the range
    #------------------
    ring_range = ring_event.dep.norm()
    
    #-------------------------------------------------------------------
    # Get the radius and inertial longitude; track radial derivatives
    #-------------------------------------------------------------------
   (ring_radius,
     ring_longitude) = ring_body.surface.event_as_coords(ring_event,
                                                         axes=2, derivs=derivs)
    
    ring_emission = ring_event.emission_angle()
    
    show_info("Ring mask", ring_mask)
    show_info("Ring range (km)", ring_range)
    show_info("Ring radius (km)", ring_radius)
    show_info("Ring longitude (deg)", ring_longitude * oops.DPR)
    show_info("Ring emission angle (deg)", ring_emission * oops.DPR)
    
    pushbrooms[0].insert_subfield("ring_mask", ring_mask)
    pushbrooms[0].insert_subfield("ring_range", ring_range)
    pushbrooms[0].insert_subfield("ring_radius", ring_radius)
    pushbrooms[0].insert_subfield("ring_longitude", ring_longitude)
    pushbrooms[0].insert_subfield("ring_emission", ring_emission)
    
    #--------------------------------
    # Get the ring plane resolution
    #--------------------------------
    if derivs:
        dpos_duv = ring_event_w_derivs.pos.d_dlos * los.d_duv
        
        gradient = ring_radius.d_dpos * dpos_duv
        ring_radial_resolution = gradient.as_pair().norm()
        
        gradient = ring_longitude.d_dpos * dpos_duv
        ring_angular_resolution = gradient.as_pair().norm()
        
        show_info("Ring radial resolution (km/pixel)", ring_radial_resolution)
        show_info("Ring angular resolution (deg/pixel)",
                  ring_angular_resolution * oops.DPR)
        
        pushbrooms[0].insert_subfield("ring_radial_resolution",
                                 ring_radial_resolution)
        pushbrooms[0].insert_subfield("ring_angular_resolution",
                                 ring_angular_resolution)
    
    ############################################
    # Saturn intercept points in pushbroom
    ############################################
    
    #----------------------------------
    # Find the ring intercept events
    #----------------------------------
    saturn_event_w_derivs = saturn_body.surface.photon_to_event(pushbroom_event,
                                                                derivs=derivs)
    #saturn_event = saturn_event_w_derivs.copy()
    #saturn_event.delete_sub_subfields()
    saturn_event = saturn_event_w_derivs.plain()
    
    #---------------------------------------------------------
    # This mask is True on the planet, False off the planet
    #---------------------------------------------------------
    saturn_mask = np.logical_not(saturn_event.mask)
    
    #-----------------
    # Get the range
    #-----------------
    saturn_range = saturn_event.dep.norm()
    
    #-------------------------------------------------
    # Get the longitude and three kinds of latitude
    #-------------------------------------------------
    (saturn_longitude,
     saturn_squashed_lat) = saturn_body.surface.event_as_coords(saturn_event,
                                                                axes=2)
    saturn_centric_lat = saturn_body.surface.lat_to_centric(saturn_squashed_lat)
    saturn_graphic_lat = saturn_body.surface.lat_to_graphic(saturn_squashed_lat)
    
    saturn_emission = saturn_event.emission_angle()
    
    show_info("Saturn mask", saturn_mask)
    show_info("Saturn range (km)", saturn_range)
    show_info("Saturn longitude (deg)", saturn_longitude * oops.DPR)
    show_info("Saturn planetocentric latitude (deg)", saturn_centric_lat *
              oops.DPR)
    show_info("Saturn planetographic latitude (deg)", saturn_graphic_lat *
              oops.DPR)
    show_info("Saturn emission angle (deg)", saturn_emission * oops.DPR)
    
    pushbrooms[0].insert_subfield("saturn_mask", saturn_mask)
    pushbrooms[0].insert_subfield("saturn_range", saturn_range)
    pushbrooms[0].insert_subfield("saturn_longitude", saturn_longitude)
    pushbrooms[0].insert_subfield("saturn_centric_lat", saturn_centric_lat)
    pushbrooms[0].insert_subfield("saturn_graphic_lat", saturn_graphic_lat)
    pushbrooms[0].insert_subfield("saturn_emission", saturn_emission)
    
    #-------------------------------------------------------
    # Get the Saturn surface resolution and foreshortening
    #-------------------------------------------------------
    if derivs:
        dpos_duv = saturn_event_w_derivs.pos.d_dlos * los.d_duv
        
        dpos_du = oops.Vector3(dpos_duv.vals[...,0], dpos_duv.mask)
        dpos_dv = oops.Vector3(dpos_duv.vals[...,1], dpos_duv.mask)
        
        pushbrooms[0].insert_subfield("dpos_du", dpos_du)
        pushbrooms[0].insert_subfield("dpos_dv", dpos_dv)
        pushbrooms[0].insert_subfield("perp", saturn_event.perp)
        pushbrooms[0].insert_subfield("pos", saturn_event.pos)
        pushbrooms[0].insert_subfield("time", saturn_event.time)
        
        (saturn_resolution_min,
         saturn_resolution_max) = oops.surface.Surface.resolution(dpos_duv)
        
        show_info("Saturn finest resolution (km/pixel)", saturn_resolution_min)
        show_info("Saturn coarsest resolution (km/pixel)",
                  saturn_resolution_max)
        
        pushbrooms[0].insert_subfield("saturn_resolution_min", saturn_resolution_min)
        pushbrooms[0].insert_subfield("saturn_resolution_max", saturn_resolution_max)
    
    ############################################
    # Which object is in front?
    ############################################
    
    ring_would_be_closer_mask = (saturn_event.dep.norm() >
                                 ring_event.dep.norm()).vals
    
    ring_unobscured_mask = (ring_would_be_closer_mask |
                            np.logical_not(saturn_mask)) & ring_mask
    
    saturn_unobscured_mask = (np.logical_not(ring_would_be_closer_mask) |
                              np.logical_not(ring_mask)) & saturn_mask
    
    show_info("Rings in front mask", ring_unobscured_mask)
    show_info("Saturn in front mask", saturn_unobscured_mask)
    
    pushbrooms[0].insert_subfield("ring_unobscured_mask", ring_unobscured_mask)
    pushbrooms[0].insert_subfield("saturn_unobscured_mask", saturn_unobscured_mask)
    
    ############################################
    # Ring lighting geometry
    ############################################
    
    #----------------------------------------------
    # Find the Sun departure events to the ring
    #----------------------------------------------
    sun_to_ring_event = sun_body.path.photon_to_event(ring_event)
    
    sun_ring_range = ring_event.arr.norm()
    ring_incidence = ring_event.incidence_angle()
    ring_phase = ring_event.phase_angle()
    
    show_info("Sun-Ring range (km)", sun_ring_range)
    show_info("Ring incidence angle (deg)", ring_incidence * oops.DPR)
    show_info("Ring phase angle (deg)", ring_phase * oops.DPR)
    
    pushbrooms[0].insert_subfield("sun_ring_range", sun_ring_range)
    pushbrooms[0].insert_subfield("ring_incidence", ring_incidence)
    pushbrooms[0].insert_subfield("ring_phase", ring_phase)
    
    ############################################
    # Saturn lighting geometry
    ############################################
    
    #---------------------------------------------
    # Find the Sun departure events to Saturn
    #---------------------------------------------
    sun_to_saturn_event = sun_body.path.photon_to_event(saturn_event)
    
    sun_saturn_range = saturn_event.arr.norm()
    saturn_incidence = saturn_event.incidence_angle()
    saturn_phase = saturn_event.phase_angle()
    saturn_lit_side_mask = (saturn_incidence < (90.*oops.RPD)).vals
    
    show_info("Sun-Saturn range (km)", sun_saturn_range)
    show_info("Saturn incidence angle (deg)", saturn_incidence * oops.DPR)
    show_info("Saturn phase angle (deg)", saturn_phase * oops.DPR)
    show_info("Saturn lit side mask)", saturn_lit_side_mask)
    
    pushbrooms[0].insert_subfield("sun_saturn_range", sun_saturn_range)
    pushbrooms[0].insert_subfield("saturn_incidence", saturn_incidence)
    pushbrooms[0].insert_subfield("saturn_phase", saturn_phase)
    pushbrooms[0].insert_subfield("saturn_lit_side_mask", saturn_lit_side_mask)
    
    ############################################
    # Shadow of Saturn on the rings
    ############################################
    
    #--------------------------------------------------------------------------
    # Trace the ring arrival events from the Sun backward into Saturn's surface
    #--------------------------------------------------------------------------
    ring_in_shadow_event = saturn_body.surface.photon_to_event(ring_event)
    
    #---------------------------------------------------------------
    # False (unmasked) in the shadow, True (masked) where sunlit
    #---------------------------------------------------------------
    ring_unshadowed_mask = ring_in_shadow_event.mask
    
    show_info("Rings unshadowed mask", ring_unshadowed_mask)
    pushbrooms[0].insert_subfield("ring_unshadowed_mask", ring_unshadowed_mask)
    
    ############################################
    # Shadow of the rings on Saturn
    ############################################
    
    #--------------------------------------------------------------------------
    # Trace the ring arrival events from Saturn backward into the ring surface
    #--------------------------------------------------------------------------
    saturn_in_shadow_event = ring_body.surface.photon_to_event(saturn_event)
    
    #-------------------------------------------------------------
    # False (unmasked) in the shadow, True (masked) where sunlit
    #-------------------------------------------------------------
    saturn_unshadowed_mask = saturn_in_shadow_event.mask
    
    show_info("Saturn unshadowed mask", saturn_unshadowed_mask)
    pushbrooms[0].insert_subfield("saturn_unshadowed_mask", saturn_unshadowed_mask)
    
    ############################################
    # Composite masks
    ############################################
    
    ring_visible_and_lit_mask = ring_unshadowed_mask & ring_unobscured_mask
    
    saturn_visible_and_lit_mask = (saturn_unshadowed_mask &
                                   saturn_unobscured_mask &
                                   saturn_lit_side_mask)
    
    show_info("Rings visible and lit mask", ring_visible_and_lit_mask)
    show_info("Saturn visible and lit mask", saturn_visible_and_lit_mask)
    
    pushbrooms[0].insert_subfield("ring_visible_and_lit_mask",
                             ring_visible_and_lit_mask)
    
    pushbrooms[0].insert_subfield("saturn_visible_and_lit_mask",
                             saturn_visible_and_lit_mask)
    
    return pushbrooms
#===============================================================================



################################################################################
# UNIT TESTS
################################################################################

import unittest
import os.path

UNITTEST_PRINTING = True
UNITTEST_LOGGING = True
UNITTEST_DERIVS = True

#*******************************************************************************
# Test_Cassini_VIMS_Suite
#*******************************************************************************
class Test_Cassini_VIMS_Suite(unittest.TestCase):
    
    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):
        
        from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY

        if UNITTEST_LOGGING: oops.config.LOGGING.on("        ")
        
        #filespec = "cassini/VIMS/V1546355804_1.QUB"
        filespec = os.path.join(TESTDATA_PARENT_DIRECTORY, "cassini/VIMS/V1463282505_1.QUB")
        pushbrooms = vims_test_suite(filespec, UNITTEST_DERIVS,
                                     UNITTEST_PRINTING, DISPLAY)
        
        oops.config.LOGGING.off()
    #===========================================================================


#*******************************************************************************


############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
