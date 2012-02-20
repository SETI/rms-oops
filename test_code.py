import numpy as np
import pylab
import oops_.all as oops
import oops_.inst.cassini.iss as cassini_iss

# Try:
# >>> execfile("test_code.py")
# to run everything at once. Handy for timing tests

# Create the snapshot object
filespec = "test_data/cassini/ISS/W1573721822_1.IMG"
snapshot = cassini_iss.from_file(filespec)

# Create the snapshot event
# ... with a grid point at the middle of each pixel
fov_shape = snapshot.fov.uv_shape

uv_pair = oops.Pair.cross_scalars(
    np.arange(fov_shape.vals[0]) + 0.5,
    np.arange(fov_shape.vals[1]) + 0.5
)

midtime = (snapshot.t0 + snapshot.t1)/2.

arrivals = -snapshot.fov.los_from_uv(uv_pair)

# This line swaps the image for proper display using pylab.imshow()
arrivals = oops.Vector3(arrivals.swapaxes(0,1))

snapshot_event = oops.Event(
    midtime, (0.,0.,0.), (0.,0.,0.),
    snapshot.path_id, snapshot.frame_id, arr=arrivals)

# Get the right ascension and declination
radec = snapshot_event.ra_and_dec()
(right_ascension, declination) = radec.as_scalars()

pylab.imshow(right_ascension.vals)
print (np.min(right_ascension.vals) * oops.DPR,
       np.max(right_ascension.vals) * oops.DPR)

pylab.imshow(declination.vals)
print (np.min(declination.vals) * oops.DPR,
       np.max(declination.vals) * oops.DPR)

# Find the ring intercept points
ring_surface = oops.SOLAR_SYSTEM["SATURN_MAIN_RINGS"].surface

(ring_event_abs,
 ring_event_rel) = ring_surface.photon_to_event(snapshot_event)

# This mask is True inside the rings, False outside
ring_mask = ~ring_event_abs.pos.mask

pylab.imshow(ring_mask)

# Get the radius and inertial longitude
(ring_radius,
 ring_longitude) = ring_surface.as_coords(ring_event_abs.pos, axes=2)

pylab.imshow(ring_radius.vals)
print np.min(ring_radius.vals), np.max(ring_radius.vals)

pylab.imshow(ring_longitude.vals)
print (np.min(ring_longitude.vals) * oops.DPR,
       np.max(ring_longitude.vals) * oops.DPR)
# Note that 2pi-to-zero discontinuity in longitude passes through the middle

# Get the range from the observer to the rings
ring_range = ring_event_rel.pos.norm()

pylab.imshow(ring_range.vals)
print np.min(ring_range.vals), np.max(ring_range.vals)

# Get the ring emission angle
ring_emission = ring_event_abs.emission_angle()

pylab.imshow(ring_emission.vals)
print (np.min(ring_emission.vals) * oops.DPR,
       np.max(ring_emission.vals) * oops.DPR)

# Backtrack the ring intercept points from the Sun
sun_path = oops.path.Waypoint("SUN")

(sun_ring_event_abs,
 sun_ring_event_rel) = sun_path.photon_to_event(ring_event_abs)

# Get the ring incidence and phase
# The works because the arriving photon was filled in during the call above
ring_phase = ring_event_abs.phase_angle()
ring_incidence = ring_event_abs.incidence_angle()

pylab.imshow(ring_phase.vals)
print (np.min(ring_phase.vals) * oops.DPR,
       np.max(ring_phase.vals) * oops.DPR)

pylab.imshow(ring_incidence.vals)
print (np.min(ring_incidence.vals) * oops.DPR,
       np.max(ring_incidence.vals) * oops.DPR)

# Range to the Sun
ring_sun_range = sun_ring_event_rel.pos.norm()

pylab.imshow(ring_sun_range.vals)
print np.min(ring_sun_range.vals), np.max(ring_sun_range.vals)

# Backtrack the photons from the Sun and see where they intercept Saturn
saturn_surface = oops.SOLAR_SYSTEM["SATURN"].surface

(ring_in_shadow_event_abs,
 ring_in_shadow_event_rel) = saturn_surface.photon_to_event(ring_event_abs)

# The shadow is defined by the set of these points that intercepted Saturn
ring_in_shadow_mask = ~ring_in_shadow_event_abs.pos.mask
pylab.imshow(ring_in_shadow_mask)

ring_in_sunlight_mask = ring_mask & ~ring_in_shadow_mask
pylab.imshow(ring_in_sunlight_mask)

# Look at Saturn directly
(saturn_event_abs,
 saturn_event_rel) = saturn_surface.photon_to_event(snapshot_event)

# Get the Saturn mask
saturn_mask = ~saturn_event_abs.pos.mask
pylab.imshow(saturn_mask)

# Get the range from the observer to Saturn
saturn_range = saturn_event_rel.pos.norm()

pylab.imshow(saturn_range.vals)
print np.min(saturn_range.vals), np.max(saturn_range.vals)

# Get the longitude and latitude
(saturn_longitude,
 saturn_squashed_lat) = saturn_surface.as_coords(saturn_event_abs.pos, axes=2)

pylab.imshow(saturn_longitude.vals)
print (np.min(saturn_longitude.vals) * oops.DPR,
       np.max(saturn_longitude.vals) * oops.DPR)

pylab.imshow(saturn_squashed_lat.vals)
print (np.min(saturn_squashed_lat.vals) * oops.DPR,
       np.max(saturn_squashed_lat.vals) * oops.DPR)

# Convert the latitude to something standard
saturn_centric_latitude = saturn_surface.lat_to_centric(saturn_squashed_lat)
saturn_graphic_latitude = saturn_surface.lat_to_graphic(saturn_squashed_lat)

pylab.imshow(saturn_centric_latitude.vals)
print (np.min(saturn_centric_latitude.vals) * oops.DPR,
       np.max(saturn_centric_latitude.vals) * oops.DPR)

pylab.imshow(saturn_graphic_latitude.vals)
print (np.min(saturn_graphic_latitude.vals) * oops.DPR,
       np.max(saturn_graphic_latitude.vals) * oops.DPR)

# Emission angle
saturn_emission = saturn_event_abs.emission_angle()

pylab.imshow(saturn_emission.vals)
print (np.min(saturn_emission.vals) * oops.DPR,
       np.max(saturn_emission.vals) * oops.DPR)

# Backtrack the Saturn intercept points from the Sun
(sun_saturn_event_abs,
 sun_saturn_event_rel) = sun_path.photon_to_event(saturn_event_abs)

# Get the ring incidence and phase
saturn_phase = saturn_event_abs.phase_angle()
saturn_incidence = saturn_event_abs.incidence_angle()

pylab.imshow(saturn_phase.vals)
print (np.min(saturn_phase.vals) * oops.DPR,
       np.max(saturn_phase.vals) * oops.DPR)

pylab.imshow(saturn_incidence.vals)
print (np.min(saturn_incidence.vals) * oops.DPR,
       np.max(saturn_incidence.vals) * oops.DPR)

# Range to the Sun
saturn_sun_range = sun_saturn_event_rel.pos.norm()

pylab.imshow(saturn_sun_range.vals)
print np.min(saturn_sun_range.vals), np.max(saturn_sun_range.vals)

# Identify the points where Saturn is lit
saturn_sunlit_mask = saturn_mask & (saturn_incidence.vals <= np.pi/2)

pylab.imshow(saturn_sunlit_mask)

# Where are the rings visible?
# There might be a more elegant way to do this
ring_visible_mask = ring_mask.copy()
ring_visible_mask[saturn_mask & (saturn_range.vals < ring_range.vals)] = False

pylab.imshow(ring_visible_mask)

ring_visible_sunlit_mask = ring_visible_mask & ~ring_in_shadow_mask
pylab.imshow(ring_visible_sunlit_mask)

# Where is Saturn visible?
saturn_visible_mask = saturn_mask.copy()
saturn_visible_mask[ring_mask & (ring_range.vals < saturn_range.vals)] = False

pylab.imshow(saturn_visible_mask)

# Backtrack the Saturn photons from the Sun and see where they intercept the
# rings
(saturn_in_shadow_event_abs,
 saturn_in_shadow_event_rel) = ring_surface.photon_to_event(saturn_event_abs)

saturn_in_shadow_mask = ~saturn_in_shadow_event_abs.pos.mask
pylab.imshow(saturn_in_shadow_mask)

saturn_visible_sunlit_mask = saturn_sunlit_mask & ~saturn_in_shadow_mask
pylab.imshow(saturn_visible_sunlit_mask)


