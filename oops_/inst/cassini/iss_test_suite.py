import numpy as np
import pylab
import oops_.all as oops
import oops_.inst.cassini.iss as cassini_iss

PRINT = True
DISPLAY = True

def show_info(title, array):
    """Internal method to print summary information and display images as
    desired."""

    global PRINT, DISPLAY
    if not PRINT: return

    print ""
    print title

    if isinstance(array, np.ndarray):
        if array.dtype == np.dtype("bool"):
            count = np.sum(array)
            total = np.size(array)
            percent = int(count / float(total) * 100. + 0.5)
            print "   ", (count, total-count),
            print (percent, 100-percent), "(True, False pixels)"
            if DISPLAY:
                ignore = pylab.imshow(array, norm=None, vmin=0, vmax=1)
                ignore = raw_input(title + ": ")

        else:
            minval = np.min(array)
            maxval = np.max(array)
            if minval == maxval:
                print "    ", minval
            else:
                print "    ", (minval, maxval), "(min, max)"

                if DISPLAY:
                    ignore = pylab.imshow(array)
                ignore = raw_input(title + ": ")

    elif isinstance(array, oops.Array):
        if np.any(array.mask):
            print "    ", (np.min(array.vals),
                           np.max(array.vals)), "(unmasked min, max)"
            print "    ", (array.min(),
                           array.max()), "(masked min, max)"
            masked = np.sum(array.mask)
            total = np.size(array.mask)
            percent = int(masked / float(total) * 100. + 0.5)
            print "    ", (masked, total-masked),
            print         (percent, 100-percent), "(masked, unmasked pixels)"

            if DISPLAY:
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
                print "    ", minval
            else:
                print "    ", (minval, maxval), "(min, max)"

                if DISPLAY:
                    ignore = pylab.imshow(array.vals)
                    ignore = raw_input(title + ": ")

    else:
        print "    ", array

def iss_test_suite(filespec, derivs, info, display):
    """Master test suite for a Cassini ISS image.

    Input:
        filespec    file path and name to a Cassini ISS image file.
        derivs      True to calculate derivatives where needed to derive
                    quantities related to spatial resolution; False to omit all
                    resolution calculations.
        info        True to print out geometry information as it progresses.
        display     True to display each backplane using Pylab, and pause until
                    the user hits RETURN.
    """

    global PRINT, DISPLAY
    PRINT = info
    DISPLAY = display

    # Define the bodies we care about
    ring_body = oops.SOLAR_SYSTEM["SATURN_MAIN_RINGS"]
    saturn_body = oops.SOLAR_SYSTEM["SATURN"]
    sun_body = oops.SOLAR_SYSTEM["SUN"]

    # Create the snapshot object
    snapshot = cassini_iss.from_file(filespec)

    # Create the snapshot event
    # ... with a grid point at the middle of each pixel
    fov_shape = snapshot.fov.uv_shape
    
    uv_pair = oops.Pair.cross_scalars(
        np.arange(fov_shape.vals[0]) + 0.5,
        np.arange(fov_shape.vals[1]) + 0.5
    )

    los = snapshot.fov.los_from_uv(uv_pair, derivs=derivs)
    # los.d_duv is now the [3,2] MatrixN of derivatives dlos/d(u,v), where los
    # is in the frame of the Cassini camera.

    # This line swaps the image for proper display using pylab.imshow().
    # Also change sign for incoming photons, and for subfield "d_duv".
    arrivals = -los.swapaxes(0,1)

    # Define the event as a 1024x1024 array of simultaneous photon arrivals
    # coming from slightly different directions
    snapshot_event = oops.Event(
        snapshot.midtime, (0.,0.,0.), (0.,0.,0.),
        snapshot.path_id, snapshot.frame_id, arr=arrivals)

    # For single-point calculations about the geometry
    point_event = oops.Event(
        snapshot.midtime, (0.,0.,0.), (0.,0.,0.),
        snapshot.path_id, snapshot.frame_id)
 
    ############################################
    # Sky coordinates
    ############################################

    (right_ascension, declination) = snapshot_event.ra_and_dec()
    show_info("Right ascension (deg)", right_ascension * oops.DPR)
    show_info("Declination (deg)", declination * oops.DPR)

    snapshot.insert_subfield("right_ascension", right_ascension)
    snapshot.insert_subfield("declination", declination)

    ############################################
    # Sub-observer ring geometry
    ############################################

    # Define the apparent location of the observer relative to Saturn ring frame
    ring_center_event = ring_body.path.photon_to_event(point_event)

    # Event separation in ring surface coordinates
    obs_wrt_ring_center = oops.Edelta.sub_events(point_event,
                                                 ring_center_event)

    obs_wrt_ring_range = obs_wrt_ring_center.pos.norm()

    (obs_wrt_ring_radius,
    obs_wrt_ring_longitude,
    obs_wrt_ring_elevation) = ring_body.surface.as_coords(
                                                    obs_wrt_ring_center.pos,
                                                    axes=3)
    
    show_info("Ring range to observer (km)", obs_wrt_ring_range)
    show_info("Ring radius of observer (km)", obs_wrt_ring_radius)
    show_info("Ring longitude of observer(deg)", obs_wrt_ring_longitude *
                                                 oops.DPR)
    show_info("Ring elevation of observer(km)", obs_wrt_ring_elevation)

    snapshot.insert_subfield("obs_wrt_ring_range", obs_wrt_ring_range)
    snapshot.insert_subfield("obs_wrt_ring_radius", obs_wrt_ring_radius)
    snapshot.insert_subfield("obs_wrt_ring_longitude", obs_wrt_ring_longitude)
    snapshot.insert_subfield("obs_wrt_ring_elevation", obs_wrt_ring_elevation)

    ############################################
    # Sub-solar ring geometry
    ############################################

    # Define the apparent location of the Sun relative to the ring frame
    sun_center_event = sun_body.path.photon_from_event(ring_center_event)

    # Event separation in ring surface coordinates
    sun_wrt_ring_center = oops.Edelta.sub_events(point_event,
                                                 sun_center_event)

    sun_wrt_ring_range = sun_wrt_ring_center.pos.norm()

    (sun_wrt_ring_radius,
    sun_wrt_ring_longitude,
    sun_wrt_ring_elevation) = ring_body.surface.as_coords(
                                                    sun_wrt_ring_center.pos,
                                                    axes=3)
    
    show_info("Ring range to Sun (km)", sun_wrt_ring_range)
    show_info("Ring radius of Sun (km)", sun_wrt_ring_radius)
    show_info("Ring longitude of Sun (deg)", sun_wrt_ring_longitude * oops.DPR)
    show_info("Ring elevation of Sun (km)", sun_wrt_ring_elevation)

    snapshot.insert_subfield("sun_wrt_ring_range", sun_wrt_ring_range)
    snapshot.insert_subfield("sun_wrt_ring_radius", sun_wrt_ring_radius)
    snapshot.insert_subfield("sun_wrt_ring_longitude", sun_wrt_ring_longitude)
    snapshot.insert_subfield("sun_wrt_ring_elevation", sun_wrt_ring_elevation)

    ############################################
    # Sub-observer Saturn geometry
    ############################################

    # Define the apparent location of the observer relative to Saturn frame
    saturn_center_event = saturn_body.path.photon_to_event(point_event)

    # Event separation in Saturn surface coordinates
    obs_wrt_saturn_center = oops.Edelta.sub_events(point_event,
                                                   saturn_center_event)

    obs_wrt_saturn_range = obs_wrt_saturn_center.pos.norm()

    (obs_wrt_saturn_longitude,
    obs_wrt_saturn_latitude,
    obs_wrt_saturn_elevation) = saturn_body.surface.as_coords(
                                                    obs_wrt_saturn_center.pos,
                                                    axes=3)
    
    show_info("Saturn range to observer (km)", obs_wrt_saturn_range)
    show_info("Saturn longitude of observer (deg)", obs_wrt_saturn_longitude *
                                                  oops.DPR)
    show_info("Saturn latitude of observer (km)", obs_wrt_saturn_latitude *
                                                  oops.DPR)
    show_info("Saturn elevation of observer (km)", obs_wrt_saturn_elevation)

    snapshot.insert_subfield("obs_wrt_saturn_range", obs_wrt_saturn_range)
    snapshot.insert_subfield("obs_wrt_saturn_longitude",
                                                     obs_wrt_saturn_longitude)
    snapshot.insert_subfield("obs_wrt_saturn_latitude", obs_wrt_saturn_latitude)
    snapshot.insert_subfield("obs_wrt_saturn_elevation",
                                                     obs_wrt_saturn_elevation)

    ############################################
    # Sub-solar Saturn geometry
    ############################################

    # Define the apparent location of the Sun relative to the Saturn frame
    sun_center_event = sun_body.path.photon_from_event(saturn_center_event)

    # Event separation in Saturn surface coordinates
    sun_wrt_saturn_center = oops.Edelta.sub_events(snapshot_event,
                                                   sun_center_event)

    sun_wrt_saturn_range = sun_wrt_saturn_center.pos.norm()

    (sun_wrt_saturn_longitude,
    sun_wrt_saturn_latitude,
    sun_wrt_saturn_elevation) = saturn_body.surface.as_coords(
                                                    sun_wrt_saturn_center.pos,
                                                    axes=3)

    show_info("Saturn range to Sun (km)", sun_wrt_saturn_range)
    show_info("Saturn longitude of Sun (deg)", sun_wrt_saturn_longitude *
                                                    oops.DPR)
    show_info("Saturn latitude of Sun (km)", sun_wrt_saturn_latitude *
                                                  oops.DPR)
    show_info("Saturn elevation of Sun (km)", sun_wrt_saturn_elevation)

    snapshot.insert_subfield("sun_wrt_saturn_range", sun_wrt_saturn_range)
    snapshot.insert_subfield("sun_wrt_saturn_longitude",
                                                     sun_wrt_saturn_longitude)
    snapshot.insert_subfield("sun_wrt_saturn_latitude", sun_wrt_saturn_latitude)
    snapshot.insert_subfield("sun_wrt_saturn_elevation",
                                                     sun_wrt_saturn_elevation)

    ############################################
    # Ring intercept points in snapshot
    ############################################

    # Find the ring intercept events
    ring_event_w_derivs = ring_body.surface.photon_to_event(snapshot_event,
                                                            derivs=derivs)
    ring_event = ring_event_w_derivs.copy()
    ring_event.delete_sub_subfields()

    # This mask is True inside the rings, False outside
    ring_mask = ~ring_event.mask

    # Get the range
    ring_range = ring_event.dep.norm()

    # Get the radius and inertial longitude; track radial derivatives
    (ring_radius,
    ring_longitude) = ring_body.surface.as_coords(ring_event.pos,
                                                  axes=2, derivs=derivs)

    ring_emission = ring_event.emission_angle()

    show_info("Ring mask", ring_mask)
    show_info("Ring range (km)", ring_range)
    show_info("Ring radius (km)", ring_radius)
    show_info("Ring longitude (deg)", ring_longitude * oops.DPR)
    show_info("Ring emission angle (deg)", ring_emission * oops.DPR)

    snapshot.insert_subfield("ring_mask", ring_mask)
    snapshot.insert_subfield("ring_range", ring_range)
    snapshot.insert_subfield("ring_radius", ring_radius)
    snapshot.insert_subfield("ring_longitude", ring_longitude)
    snapshot.insert_subfield("ring_emission", ring_emission)

    # Get the ring plane resolution
    if derivs:
        dpos_duv = ring_event_w_derivs.pos.d_dlos * los.d_duv

        gradient = ring_radius.d_dpos * dpos_duv
        ring_radial_resolution = gradient.as_pair().norm()

        gradient = ring_longitude.d_dpos * dpos_duv
        ring_angular_resolution = gradient.as_pair().norm()

        show_info("Ring radial resolution (km/pixel)", ring_radial_resolution)
        show_info("Ring angular resolution (deg/pixel)",
                                            ring_angular_resolution * oops.DPR)

        snapshot.insert_subfield("ring_radial_resolution",
                                            ring_radial_resolution)
        snapshot.insert_subfield("ring_angular_resolution",
                                            ring_angular_resolution)

    ############################################
    # Saturn intercept points in snapshot
    ############################################

    # Find the ring intercept events
    saturn_event_w_derivs = saturn_body.surface.photon_to_event(snapshot_event,
                                                                derivs=derivs)
    saturn_event = saturn_event_w_derivs.copy()
    saturn_event.delete_sub_subfields()

    # This mask is True on the planet, False off the planet
    saturn_mask = ~saturn_event.mask

    # Get the range
    saturn_range = saturn_event.dep.norm()

    # Get the longitude and three kinds of latitude
    (saturn_longitude,
    saturn_squashed_lat) = saturn_body.surface.as_coords(saturn_event.pos,
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

    snapshot.insert_subfield("saturn_mask", saturn_mask)
    snapshot.insert_subfield("saturn_range", saturn_range)
    snapshot.insert_subfield("saturn_longitude", saturn_longitude)
    snapshot.insert_subfield("saturn_centric_lat", saturn_centric_lat)
    snapshot.insert_subfield("saturn_graphic_lat", saturn_graphic_lat)
    snapshot.insert_subfield("saturn_emission", saturn_emission)

    # Get the Saturn surface resolution and foreshortening
    if derivs:
        dpos_duv = saturn_event_w_derivs.pos.d_dlos * los.d_duv

        dpos_du = oops.Vector3(dpos_duv.vals[...,0], dpos_duv.mask)
        dpos_dv = oops.Vector3(dpos_duv.vals[...,1], dpos_duv.mask)

        snapshot.insert_subfield("dpos_du", dpos_du)
        snapshot.insert_subfield("dpos_dv", dpos_dv)
        snapshot.insert_subfield("perp", saturn_event.perp)
        snapshot.insert_subfield("pos", saturn_event.pos)
        snapshot.insert_subfield("time", saturn_event.time)

        (saturn_resolution_min,
        saturn_resolution_max) = oops.surface.Surface.resolution(dpos_duv)

        show_info("Saturn finest resolution (km/pixel)", saturn_resolution_min)
        show_info("Saturn coarsest resolution (km/pixel)",
                                                         saturn_resolution_max)

        snapshot.insert_subfield("saturn_resolution_min", saturn_resolution_min)
        snapshot.insert_subfield("saturn_resolution_max", saturn_resolution_max)

    ############################################
    # Which object is in front?
    ############################################

    ring_would_be_closer_mask = (saturn_event.dep.norm() >
                                 ring_event.dep.norm()).vals

    ring_unobscured_mask = (ring_would_be_closer_mask |
                            ~saturn_mask) & ring_mask

    saturn_unobscured_mask = (~ring_would_be_closer_mask |
                              ~ring_mask) & saturn_mask

    show_info("Rings in front mask", ring_unobscured_mask)
    show_info("Saturn in front mask", saturn_unobscured_mask)

    snapshot.insert_subfield("ring_unobscured_mask", ring_unobscured_mask)
    snapshot.insert_subfield("saturn_unobscured_mask", saturn_unobscured_mask)

    ############################################
    # Ring lighting geometry
    ############################################

    # Find the Sun departure events to the ring
    sun_to_ring_event = sun_body.path.photon_to_event(ring_event)

    sun_ring_range = ring_event.arr.norm()
    ring_incidence = ring_event.incidence_angle()
    ring_phase = ring_event.phase_angle()

    show_info("Sun-Ring range (km)", sun_ring_range)
    show_info("Ring incidence angle (deg)", ring_incidence * oops.DPR)
    show_info("Ring phase angle (deg)", ring_phase * oops.DPR)

    snapshot.insert_subfield("sun_ring_range", sun_ring_range)
    snapshot.insert_subfield("ring_incidence", ring_incidence)
    snapshot.insert_subfield("ring_phase", ring_phase)

    ############################################
    # Saturn lighting geometry
    ############################################

    # Find the Sun departure events to Saturn
    sun_to_saturn_event = sun_body.path.photon_to_event(saturn_event)

    sun_saturn_range = saturn_event.arr.norm()
    saturn_incidence = saturn_event.incidence_angle()
    saturn_phase = saturn_event.phase_angle()
    saturn_lit_side_mask = (saturn_incidence < (90.*oops.RPD)).vals

    show_info("Sun-Saturn range (km)", sun_saturn_range)
    show_info("Saturn incidence angle (deg)", saturn_incidence * oops.DPR)
    show_info("Saturn phase angle (deg)", saturn_phase * oops.DPR)
    show_info("Saturn lit side mask)", saturn_lit_side_mask)

    snapshot.insert_subfield("sun_saturn_range", sun_saturn_range)
    snapshot.insert_subfield("saturn_incidence", saturn_incidence)
    snapshot.insert_subfield("saturn_phase", saturn_phase)
    snapshot.insert_subfield("saturn_lit_side_mask", saturn_lit_side_mask)

    ############################################
    # Shadow of Saturn on the rings
    ############################################

    # Trace the ring arrival events from the Sun backward into Saturn's surface
    ring_in_shadow_event = saturn_body.surface.photon_to_event(ring_event)

    # False (unmasked) in the shadow, True (masked) where sunlit
    ring_unshadowed_mask = ring_in_shadow_event.mask

    show_info("Rings unshadowed mask", ring_unshadowed_mask)
    snapshot.insert_subfield("ring_unshadowed_mask", ring_unshadowed_mask)

    ############################################
    # Shadow of the rings on Saturn
    ############################################

    # Trace the ring arrival events from Saturn backward into the ring surface
    saturn_in_shadow_event = ring_body.surface.photon_to_event(saturn_event)

    # False (unmasked) in the shadow, True (masked) where sunlit
    saturn_unshadowed_mask = saturn_in_shadow_event.mask

    show_info("Saturn unshadowed mask", saturn_unshadowed_mask)
    snapshot.insert_subfield("saturn_unshadowed_mask", saturn_unshadowed_mask)

    ############################################
    # Composite masks
    ############################################

    ring_visible_and_lit_mask = ring_unshadowed_mask & ring_unobscured_mask

    saturn_visible_and_lit_mask = (saturn_unshadowed_mask &
                                   saturn_unobscured_mask &
                                   saturn_lit_side_mask)

    show_info("Rings visible and lit mask", ring_visible_and_lit_mask)
    show_info("Saturn visible and lit mask", saturn_visible_and_lit_mask)

    snapshot.insert_subfield("ring_visible_and_lit_mask",
                              ring_visible_and_lit_mask)

    snapshot.insert_subfield("saturn_visible_and_lit_mask",
                              saturn_visible_and_lit_mask)

    return snapshot

################################################################################
# UNIT TESTS
################################################################################

import unittest

UNITTEST_PRINTING = False
UNITTEST_LOGGING = False
UNITTEST_DERIVS = False

class Test_Cassini_ISS_Suite(unittest.TestCase):

    def runTest(self):

        if UNITTEST_LOGGING: oops.config.LOGGING.on("        ")

        filespec = "test_data/cassini/ISS/W1573721822_1.IMG"
        snapshot = iss_test_suite(filespec, UNITTEST_DERIVS,
                                  UNITTEST_PRINTING, False)

        oops.config.LOGGING.off()

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
