################################################################################
# oops/inst/hst/wfc3/wfc3_test_suite.py
################################################################################

import numpy as np
import scipy.ndimage.filters as filters
import pylab
import os.path

# Load oops
import oops

# Load the HST reader
import oops.inst.hst as hst

# At this point, a Body class object has been created for every planet and moon
# (including Pluto). If you don't believe me...
print oops.registry.BODY_REGISTRY.keys()

from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY

# Below is just a handy routine for illustrative purposes. Cut, paste, ignore.

PRINT = True
DISPLAY = True
PAUSE = False

def show_info(title, array):
    global PRINT, DISPLAY, PAUSE
    if not PRINT: return
    print ""
    print title
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
        if DISPLAY:
            ignore = pylab.imshow(array.vals)
            ignore = raw_input(title + ": ")
            background = np.zeros(array.shape, dtype="uint8")
            background[0::2,0::2] = 1
            background[1::2,1::2] = 1
            ignore = pylab.imshow(background)
            ignore = pylab.imshow(array.mvals)
            if PAUSE: ignore = raw_input(title + ": ")
    elif array.vals.dtype == np.dtype("bool"):
        masked = np.sum(array.vals)
        total = np.size(array.vals)
        percent = int(masked / float(total) * 100. + 0.5)
        print "  ", (masked, total-masked),
        print         (percent, 100-percent), "(masked, unmasked pixels)"
        if DISPLAY and masked != 0 and masked != total:
            ignore = pylab.imshow(array.vals)
            if PAUSE: ignore = raw_input(title + ": ")
    else:
        minval = np.min(array.vals)
        maxval = np.max(array.vals)
        if minval == maxval:
            print "  ", minval
        else:
            print "  ", (minval, maxval), "(min, max)"
            if DISPLAY:
                ignore = pylab.imshow(array.vals)
                if PAUSE: ignore = raw_input(title + ": ")

# Pick a file, any file
# filespec = "test_data/hst/ibht02v4q_flt.fits"
filespec = os.path.join(TESTDATA_PARENT_DIRECTORY, "test_data/hst/ibht02v5q_flt.fits")
# filespec = "test_data/hst/ibht02v6q_flt.fits"

# Create the snapshot object. It contains everything you need to know.
snapshot = hst.from_file(filespec)

# Wanna know the target body?
print snapshot.target.name

# Display the image (upside-down by default in pylab)
pylab.imshow(snapshot.data)

# Create a meshgrid for the field of view. This is just a 2-D array of indices
# into the image array. However, it caches useful information for cases when you
# deal with the same field of view over and over in multiple images. You just
# need to create the meshgrid once.

meshgrid = oops.Meshgrid.for_fov(snapshot.fov, swap=True)
# swap=True because HST image arrays are indexed (vertical,horizontal). The
# alternative would be to swap the axes of the data object. Your choice.

# A backplane object holds all the geometry information we could want
bp = oops.Backplane(snapshot, meshgrid)

# Every backplane you could ever want
# The first argument to each function is the body ID
# Some functions take additional arguments
show_info("Right ascension (deg)", bp.right_ascension() * oops.DPR)

show_info("Declination (deg)", bp.declination() * oops.DPR)

show_info("Ring radius (km)", bp.ring_radius("uranus_ring_plane"))

show_info("Ring longitude WRT J2000 (deg)",
            bp.ring_longitude("uranus_ring_plane", "j2000") * oops.DPR)

show_info("Ring longitude WRT Sun (deg)",
            bp.ring_longitude("uranus_ring_plane", "sun") * oops.DPR)

show_info("Ring longitude WRT SHA (deg)",
            bp.ring_longitude("uranus_ring_plane", "sha") * oops.DPR)

show_info("Ring longitude WRT observer (deg)",
            bp.ring_longitude("uranus_ring_plane", "obs") * oops.DPR)

show_info("Ring longitude WRT OHA (deg)",
            bp.ring_longitude("uranus_ring_plane", "oha") * oops.DPR)

show_info("Ring radial resolution (km/pixel)",
            bp.ring_radial_resolution("uranus_ring_plane"))

show_info("Ring angular resolution (deg/pixel)",
            bp.ring_angular_resolution("uranus_ring_plane") * oops.DPR)

show_info("Ring incidence angle (deg)",
            bp.incidence_angle("uranus_ring_plane") * oops.DPR)

show_info("Ring emission angle (deg)",
            bp.emission_angle("uranus_ring_plane") * oops.DPR)

show_info("Ring phase angle (deg)",
            bp.phase_angle("uranus_ring_plane") * oops.DPR)

show_info("Uranus planetographic latitude (deg)",
            bp.latitude("uranus", "graphic") * oops.DPR)

show_info("Uranus planetocentric latitude (deg)",
            bp.latitude("uranus", "centric") * oops.DPR)

show_info("Uranus longitude WRT IAU (deg)",
            bp.longitude("uranus", "iau", "east", 0) * oops.DPR)

show_info("Uranus longitude WRT Sun (deg)",
            bp.longitude("uranus", "sun", "east", 0) * oops.DPR)

show_info("Uranus longitude WRT Sun, -180 to 180 (deg)",
            bp.longitude("uranus", "sun", "east", -180) * oops.DPR)

show_info("Uranus longitude WRT observer (deg)",
            bp.longitude("uranus", "obs", "east", 0) * oops.DPR)

show_info("Uranus longitude WRT observer, -180 to 180 (deg)",
            bp.longitude("uranus", "obs", "east", -180) * oops.DPR)

show_info("Uranus finest surface resolution (km/pixel)",
            bp.finest_resolution("uranus"))

show_info("Uranus coarsest surface resolution (km/pixel)",
            bp.coarsest_resolution("uranus"))

show_info("Uranus incidence angle (deg)",
            bp.incidence_angle("uranus") * oops.DPR)

show_info("Uranus emission angle (deg)",
            bp.emission_angle("uranus") * oops.DPR)

show_info("Uranus phase angle (deg)",
            bp.phase_angle("uranus") * oops.DPR)

# We can also define masks
show_info("Uranus mask",
            bp.where_intercepted("uranus"))

show_info("Approximate mu ring mask",
            bp.where_between(("ring_radius", "uranus_ring_plane"), 9.e4, 11.e4))

show_info("Defined mu ring mask",
            bp.where_intercepted("mu_ring"))

# We can also outline masks
show_info("Boundary outside Uranus",
            bp.border_outside(("where_intercepted", "uranus")))

show_info("Boundary inside Uranus",
            bp.border_inside(("where_intercepted", "uranus")))

# We can also outline contours of a particular value
show_info("Just above 100,000 km",
            bp.border_above(("ring_radius","uranus_ring_plane"), 100000.))
show_info("Just below 100,000 km",
            bp.border_below(("ring_radius","uranus_ring_plane"), 100000.))
show_info("Roughly centered on 100,000 km",
            bp.border_atop(("ring_radius","uranus_ring_plane"), 100000.))

# The Uranian precessing/inclined rings are defined but have not been fully
# validated

epsilon = bp.ring_radius("epsilon_ring")
centered = bp.ring_radius("uranus_ring_plane")
print (epsilon - centered).min(), (epsilon - centered).max()

# So here is a plot of the nominal epsilon ring

border = bp.border_atop(("ring_radius", "epsilon_ring"), 51149.32)
pylab.imshow(border.vals)

overlay = np.maximum(snapshot.data, snapshot.data.max() * border.vals)
pylab.imshow(overlay)
# You can see an offset in the position of the epsilon ring by about 10 pixels.
# This is the pointing error for the image, and we need to find a way to fix it.

# A quick cleanup of the image shows the ring more clearly
blur = filters.median_filter(snapshot.data, 9)
flat = (snapshot.data - blur)
pylab.imshow(flat.clip(-200,600))

overlay = np.maximum(flat, 600 * border.vals).clip(-200,600)
pylab.imshow(overlay)
# I am thinking we can do some test with the ring itself to find the offset
# automatically. My idea is to use an FFT to find the offset between the image
# and a blurred version of the border mask that maximized the correlation
# between the images. TBD.


# Onward to the moons...

# Select all the regular moons of Uranus
moons = snapshot.target.select_children(include_all=["REGULAR", "SATELLITE"])

# Extract the names in order
names = [moon.name for moon in moons]
print names

# Create a "multipath" that defines the path of each
multipath = oops.Body.define_multipath(moons, id="URANIAN_MOONS")

# Locate the moons in the FOV by solving for the photon paths
image_event = oops.Event(snapshot.midtime, (0.,0.,0.), (0.,0.,0.),
                         snapshot.path_id, snapshot.frame_id)
moon_event = multipath.photon_to_event(image_event)
# At this point, image_event.arr contains the arrival vector of the photons from
# each moon.

# Where are they in the image?

# This is the (u,v) image coordinate pair for each moon
moon_uv = snapshot.fov.uv_from_los(-image_event.arr)
print moon_uv

# Or we can convert them to ra and dec with full sub-pixel precision
(ra,dec) = image_event.ra_and_dec()
radec = oops.Pair.from_scalars(ra, dec) * oops.DPR  # converted to degrees
print radec

