################################################################################
# oops_/inst/hst/wfc3/uvis_test_suite.py
################################################################################

import numpy as np
import scipy.ndimage.filters as filters
import pylab
import oops
import oops.inst.hst as hst


PRINT = True
DISPLAY = False
PAUSE = False

def show_info(title, array):
    """Internal method to print summary information and display images as
    desired."""
    #
    global PRINT, DISPLAY, PAUSE
    if not PRINT: return
    #
    print ""
    print title
    #
    if isinstance(array, np.ndarray):
        if array.dtype == np.dtype("bool"):
            count = np.sum(array)
            total = np.size(array)
            percent = int(count / float(total) * 100. + 0.5)
            print " ", (count, total-count),
            print (percent, 100-percent), "(True, False pixels)"
            if DISPLAY:
                ignore = pylab.imshow(array, norm=None, vmin=0, vmax=1)
                if PAUSE: ignore = raw_input(title + ": ")
        #
        else:
            minval = np.min(array)
            maxval = np.max(array)
            if minval == maxval:
                print "  ", minval
            else:
                print "  ", (minval, maxval), "(min, max)"
                #
                if DISPLAY:
                    ignore = pylab.imshow(array)
                    if PAUSE: ignore = raw_input(title + ": ")
    #
    elif isinstance(array, oops.Array):
        if np.any(array.mask):
            print "  ", (np.min(array.vals),
                           np.max(array.vals)), "(unmasked min, max)"
            print "  ", (array.min(),
                           array.max()), "(masked min, max)"
            masked = np.sum(array.mask)
            total = np.size(array.mask)
            percent = int(masked / float(total) * 100. + 0.5)
            print "  ", (masked, total-masked),
            print         (percent, 100-percent), "(masked, unmasked pixels)"
            #
            if DISPLAY:
                ignore = pylab.imshow(array.vals)
                ignore = raw_input(title + ": ")
                background = np.zeros(array.shape, dtype="uint8")
                background[0::2,0::2] = 1
                background[1::2,1::2] = 1
                ignore = pylab.imshow(background)
                ignore = pylab.imshow(array.mvals)
                if PAUSE: ignore = raw_input(title + ": ")
        #
        else:
            minval = np.min(array.vals)
            maxval = np.max(array.vals)
            if minval == maxval:
                print "  ", minval
            else:
                print "  ", (minval, maxval), "(min, max)"
                #
                if DISPLAY:
                    ignore = pylab.imshow(array.vals)
                    if PAUSE: ignore = raw_input(title + ": ")
    #
    else:
        print "  ", array

# def uvis_test_suite(filespec, derivs, info, display):
#     """Master test suite for an HST/WFC3/UVIS image.
# 
#     Input:
#         filespec    file path and name to a UVIS image file.
#         derivs      True to calculate derivatives where needed to derive
#                     quantities related to spatial resolution; False to omit all
#                     resolution calculations.
#         info        True to print out geometry information as it progresses.
#         display     True to display each backplane using Pylab, and pause until
#                     the user hits RETURN.
#     """

# filespec = "test_data/hst/ibht02v4q_flt.fits"
filespec = "test_data/hst/ibht02v5q_flt.fits"
# filespec = "test_data/hst/ibht02v6q_flt.fits"

# Define the bodies we care about
ring_body = oops.registry.body_lookup("URANUS_RING_PLANE")
uranus_body = oops.registry.body_lookup("URANUS")
sun_body = oops.registry.body_lookup("SUN")

moon_bodies = uranus_body.select_children(include_all=["SATELLITE", "REGULAR"])
moon_multipath = oops.Body.define_multipath(moon_bodies, id="URANUS_MOONS")

derivs = True

# Create the snapshot objects
snapshot = hst.from_file(filespec)

# Create the snapshot event
# ... with a grid point at the middle of each pixel
fov_shape = snapshot.fov.uv_shape

uv_pair = oops.Pair.cross_scalars(
    np.arange(fov_shape.vals[0]) + 0.5,
    np.arange(fov_shape.vals[1]) + 0.5
)

los = snapshot.fov.los_from_uv(uv_pair, derivs=True)
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

# Define the apparent location of the observer relative to Uranus ring frame
ring_center_event = ring_body.path.photon_to_event(point_event)
ring_center_event = ring_center_event.wrt_body(ring_body)
ring_center_event.insert_subfield("perp", oops.Vector3((0,0,1)))    # Z-axis

# Event separation in ring surface coordinates
obs_wrt_ring_center = oops.Edelta.sub_events(point_event, ring_center_event)
obs_wrt_ring_range = obs_wrt_ring_center.pos.norm()

(obs_wrt_ring_radius,
obs_wrt_ring_longitude) = ring_body.surface.as_coords(
                                                obs_wrt_ring_center.pos,
                                                axes=2)


ring_center_emission = ring_center_event.emission_angle()

show_info("Ring range to observer (km)", obs_wrt_ring_range)
show_info("Ring longitude of observer (deg)", obs_wrt_ring_longitude * oops.DPR)
show_info("Ring center emission angle (deg)", ring_center_emission * oops.DPR)

snapshot.insert_subfield("obs_wrt_ring_range", obs_wrt_ring_range)
snapshot.insert_subfield("obs_wrt_ring_longitude", obs_wrt_ring_longitude)
snapshot.insert_subfield("ring_center_emission", ring_center_emission)

############################################
# Sub-solar ring geometry
############################################

# Define the apparent location of the Sun relative to the ring frame
sun_center_event = sun_body.path.photon_to_event(ring_center_event)
sun_center_event = sun_center_event.wrt_body(ring_body)

# Event separation in ring surface coordinates
sun_wrt_ring_center = oops.Edelta.sub_events(sun_center_event,
                                             ring_center_event)

sun_wrt_ring_range = sun_wrt_ring_center.pos.norm()

(sun_wrt_ring_radius,
sun_wrt_ring_longitude) = ring_body.surface.as_coords(
                                                sun_wrt_ring_center.pos,
                                                axes=2)

show_info("Ring range to Sun (km)", sun_wrt_ring_range)
show_info("Ring longitude of Sun (deg)", sun_wrt_ring_longitude * oops.DPR)

ring_center_incidence = ring_center_event.incidence_angle()
ring_center_phase = ring_center_event.phase_angle()

show_info("Ring range to Sun (km)", sun_wrt_ring_range)
show_info("Ring longitude of Sun (deg)", sun_wrt_ring_longitude * oops.DPR)
show_info("Ring center incidence angle (deg)", ring_center_incidence * oops.DPR)
show_info("Ring center phase angle (deg)", ring_center_phase * oops.DPR)

snapshot.insert_subfield("sun_wrt_ring_range", sun_wrt_ring_range)
snapshot.insert_subfield("sun_wrt_ring_longitude", sun_wrt_ring_longitude)
snapshot.insert_subfield("ring_center_incidence", ring_center_incidence)
snapshot.insert_subfield("ring_center_phase", ring_center_phase)

############################################
# Sub-observer Uranus geometry
############################################

# Define the apparent location of the observer relative to the Uranus frame
uranus_center_event = uranus_body.path.photon_to_event(point_event)
uranus_center_event = uranus_center_event.wrt_body(uranus_body)

# Event separation in Uranus surface coordinates
obs_wrt_uranus_center = oops.Edelta.sub_events(point_event,
                                               uranus_center_event)

(obs_wrt_uranus_longitude,
obs_wrt_uranus_latitude) = uranus_body.surface.as_coords(
                                                obs_wrt_uranus_center.pos,
                                                axes=2)

show_info("Uranus longitude of observer (deg)", obs_wrt_uranus_longitude *
                                              oops.DPR)
show_info("Uranus latitude of observer (km)", obs_wrt_uranus_latitude *
                                              oops.DPR)

snapshot.insert_subfield("obs_wrt_uranus_longitude",
                                                 obs_wrt_uranus_longitude)
snapshot.insert_subfield("obs_wrt_uranus_latitude", obs_wrt_uranus_latitude)

############################################
# Sub-solar Uranus geometry
############################################

# Define the apparent location of the Sun relative to the Uranus frame
sun_center_event = sun_body.path.photon_from_event(uranus_center_event)
sun_center_event = sun_center_event.wrt_body(uranus_body)

# Event separation in Uranus surface coordinates
sun_wrt_uranus_center = oops.Edelta.sub_events(sun_center_event,
                                               uranus_center_event)

(sun_wrt_uranus_longitude,
sun_wrt_uranus_latitude) = uranus_body.surface.as_coords(
                                                sun_wrt_uranus_center.pos,
                                                axes=2)

show_info("Uranus longitude of Sun (deg)", sun_wrt_uranus_longitude * oops.DPR)
show_info("Uranus latitude of Sun (km)", sun_wrt_uranus_latitude * oops.DPR)

snapshot.insert_subfield("sun_wrt_uranus_longitude", sun_wrt_uranus_longitude)
snapshot.insert_subfield("sun_wrt_uranus_latitude", sun_wrt_uranus_latitude)

############################################
# Target bodies in sky coordinates and in fov
############################################

blur = filters.median_filter(snapshot.data, 9)
flat = (snapshot.data - blur)
pylab.imshow(flat.clip(-200,600))
pylab.gray()

uranus_arrival_event = oops.Event(
    snapshot.midtime, (0.,0.,0.), (0.,0.,0.),
    snapshot.path_id, snapshot.frame_id)

ignore = uranus_body.path.photon_to_event(uranus_arrival_event)

(uranus_center_ra, uranus_center_dec) = uranus_arrival_event.ra_and_dec()

uranus_uv = snapshot.fov.uv_from_los(-uranus_arrival_event.arr).as_scalars()
pylab.plot(uranus_uv[0].vals, uranus_uv[1].vals, "go")

moon_arrival_event = oops.Event(
    snapshot.midtime, (0.,0.,0.), (0.,0.,0.),
    snapshot.path_id, snapshot.frame_id)

ignore = moon_multipath.photon_to_event(moon_arrival_event)

# (moon_center_ra, moon_center_dec) = moon_arrival_event.ra_and_dec()
#
# for i in range(len(moon_multipath)):
#     id = moon_multipath[i].path_id
#     show_info(id + " center ra (deg)", moon_center_ra[i] * oops.DPR)
#     show_info(id + " center dec (deg)", moon_center_dec[i] * oops.DPR)

moon_uv = snapshot.fov.uv_from_los(-moon_arrival_event.arr).as_scalars()
clip_u = moon_uv[0].clip(0, snapshot.data.shape[1])
clip_v = moon_uv[1].clip(0, snapshot.data.shape[0])

pylab.plot(clip_u.vals, clip_v.vals, "r+")



