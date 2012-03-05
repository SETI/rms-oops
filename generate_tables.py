import numpy as np
import pylab
import oops_.all as oops
import oops_.inst.cassini.iss as cassini_iss
from math import log, ceil

index_file = "geometry_list.txt"
spacer = '    ,   '

def is_power_of_2(n):
    return log(n, 2) % 1.0 == 0.0

def next_power_of_2(n):
    return (2 ** ceil(log(n, 2)))

def number_of_digits(ivalue, ndigits=0):
    ivalue_10 = ivalue / 10
    if ivalue_10 > 0:
        return number_of_digits(ivalue_10, ndigits + 1)
    return ndigits + 1

def width_str(width, value):
    ivalue = int(value)
    ndigits = number_of_digits(ivalue)
    n_sig_digits = width - ndigits
    fmt_str = '%%%d.%df%%s' % (width, n_sig_digits)
    ret_str = fmt_str % (value, spacer)
    return ret_str

def generate_metadata(snapshot, resolution=16.):
    geom_line = '"S/IMG/CO/ISS/'
    geom_line += snapshot.dict['IMAGE_NUMBER']
    name = snapshot.dict["INSTRUMENT_NAME"]
    if "WIDE" in name:
        camera = '/W",'
    else:
        camera = '/N",'
    geom_line += camera
    
    fov_shape = snapshot.fov.uv_shape

    uv_pair = oops.Pair.cross_scalars(np.arange(fov_shape.vals[0]) + 0.5,
                                      np.arange(fov_shape.vals[1]) + 0.5)

    midtime = (snapshot.t0 + snapshot.t1)/2.

    arrivals = -snapshot.fov.los_from_uv(uv_pair)

    # check if resolution is power of 2
    if not is_power_of_2(resolution):
        resolution = next_power_of_2(resolution)
    uv_shape = snapshot.fov.uv_shape.vals / resolution
    buffer = np.empty((uv_shape[0], uv_shape[1], 2))
    buffer[:,:,1] = np.arange(uv_shape[1]).reshape(uv_shape[1],1)
    buffer[:,:,0] = np.arange(uv_shape[0])
    buffer *= resolution
    indices = oops.Pair(buffer + 0.5)
    
    arrivals = -snapshot.fov.los_from_uv(indices)

    # This line swaps the image for proper display using pylab.imshow()
    #arrivals = oops.Vector3(arrivals.swapaxes(0,1))

    snapshot_event = oops.Event(midtime, (0.,0.,0.), (0.,0.,0.),
                                snapshot.path_id, snapshot.frame_id,
                                arr=arrivals)

    # Get the right ascension and declination
    radec = snapshot_event.ra_and_dec()
    (right_ascension, declination) = radec.as_scalars()

    geom_line += " %.15f," % (np.min(right_ascension.vals) * oops.DPR)
    geom_line += " %.15f," % (np.max(right_ascension.vals) * oops.DPR)
    geom_line += " %.15f," % (np.min(declination.vals) * oops.DPR)
    geom_line += " %.15f," % (np.max(declination.vals) * oops.DPR)
    #geom_line += width_str(16, np.min(right_ascension.vals) * oops.DPR)
    #geom_line += width_str(16, np.max(right_ascension.vals) * oops.DPR)
    #geom_line += width_str(16, np.min(declination.vals) * oops.DPR)
    #geom_line += width_str(16, np.max(declination.vals) * oops.DPR)

    # Find the ring intercept points
    ring_surface = oops.SOLAR_SYSTEM["SATURN_MAIN_RINGS"].surface

    (ring_event_abs,
     ring_event_rel) = ring_surface.photon_to_event(snapshot_event)

    # This mask is True inside the rings, False outside
    ring_mask = ~ring_event_abs.pos.mask

    # Get the radius and inertial longitude
    (ring_radius,
     ring_longitude) = ring_surface.as_coords(ring_event_abs.pos, axes=2)

    radial_res_min = 0.0
    radial_res_max = 0.0
    geom_line += " %.15f," % np.min(ring_radius.vals)
    geom_line += " %.15f," % np.max(ring_radius.vals)
    geom_line += " %.15f," % radial_res_min
    geom_line += " %.15f," % radial_res_max
    #geom_line += width_str(16, np.min(ring_radius.vals))
    #geom_line += width_str(16, np.max(ring_radius.vals))
    #geom_line += width_str(16, radial_res_min)
    #geom_line += width_str(16, radial_res_max)

    # Note that 2pi-to-zero discontinuity in longitude passes through the middle
    """
    print "ring_range shape:"
    print ring_range.shape
    print "ring_range.size(0):"
    print np.size(ring_range,0)
    print "ring_range.size(1):"
    print np.size(ring_range,1)
    range_deriv = np.diff(ring_range, axis=1)
    geom_line += np.str(np.min(range_deriv))
    geom_line += '    ,   '
    geom_line += np.str(np.max(range_deriv))
    geom_line += '    ,   ' """


    # Backtrack the ring intercept points from the Sun
    sun_path = oops.path.Waypoint("SUN")

    (sun_ring_event_abs,
     sun_ring_event_rel) = sun_path.photon_to_event(ring_event_abs)
        
    # Backtrack the photons from the Sun and see where they intercept Saturn
    saturn_surface = oops.SOLAR_SYSTEM["SATURN"].surface
        
    # Look at Saturn directly
    (saturn_event_abs,
     saturn_event_rel) = saturn_surface.photon_to_event(snapshot_event)

    # Get the longitude and latitude
    (saturn_longitude,
     saturn_squashed_lat) = saturn_surface.as_coords(saturn_event_abs.pos, axes=2)
    
    long_obs_min = 0.
    long_obs_max = 0.
    long_sha_min = 0.
    long_sha_max = 0.
    min_long_j2000 = np.min(ring_longitude.vals) * oops.DPR
    max_long_j2000 = np.max(ring_longitude.vals) * oops.DPR
    if (min_long_j2000 < 1.) and (max_long_j2000 > 359.):
        bins = np.arange(361)
        long_j2000 = ring_longitude * oops.DPR
        h,b = np.histogram(long_j2000.vals, bins)
        i = 0
        min_at_zero = 5
        n_at_zero = 0
        start_of_zero = 0
        end_of_zero = 0
        while(i < 360):
            if h[i] == 0:
                n_at_zero += 1
                if n_at_zero == min_at_zero:
                    start_of_zero = i - n_at_zero
                end_of_zero = i
            elif start_of_zero != 0:
                break
            i += 1
        if start_of_zero != 0:
            min_long_j2000 = end_of_zero
            max_long_j2000 = start_of_zero
    geom_line += " %.15f," % min_long_j2000
    geom_line += " %.15f," % max_long_j2000
    geom_line += " %.15f," % long_obs_min
    geom_line += " %.15f," % long_obs_max
    geom_line += " %.15f," % long_sha_min
    geom_line += " %.15f," % long_sha_max

    # Get the ring incidence and phase
    # The works because the arriving photon was filled in during the call above
    ring_phase = ring_event_abs.phase_angle()
    ring_incidence = ring_event_abs.incidence_angle()

    # Get the ring emission angle
    ring_emission = ring_event_abs.emission_angle()
    
    geom_line += " %.15f," % (np.min(ring_phase.vals) * oops.DPR)
    geom_line += " %.15f," % (np.max(ring_phase.vals) * oops.DPR)
    geom_line += " %.15f," % (np.min(ring_incidence.vals) * oops.DPR)
    geom_line += " %.15f," % (np.max(ring_incidence.vals) * oops.DPR)
    geom_line += " %.15f," % (np.min(ring_emission.vals) * oops.DPR)
    geom_line += " %.15f," % (np.max(ring_emission.vals) * oops.DPR)

    # Get the range from the observer to the rings
    ring_range = ring_event_rel.pos.norm()
    geom_line += " %.15f," % np.min(ring_range.vals)
    geom_line += " %.15f," % np.max(ring_range.vals)

    # Range to the Sun
    ring_sun_range = sun_ring_event_rel.pos.norm()

    (ring_in_shadow_event_abs,
     ring_in_shadow_event_rel) = saturn_surface.photon_to_event(ring_event_abs)

    # The shadow is defined by the set of these points that intercepted Saturn
    ring_in_shadow_mask = ~ring_in_shadow_event_abs.pos.mask

    #ring_in_sunlight_mask = ring_mask & ~ring_in_shadow_mask

    # Get the Saturn mask
    saturn_mask = ~saturn_event_abs.pos.mask
    geom_line += " %.15f," % np.float(ring_in_shadow_mask.all())
    geom_line += " %.15f," % np.float(ring_in_shadow_mask.any())
    geom_line += " %.15f," % np.float(saturn_mask.all())
    geom_line += " %.15f," % np.float(saturn_mask.any())

    # we do not have any of the rest of the calculations yet
    geom_line += " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n"

    """
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
    """
    return geom_line

def generate_table_for_index(file_name):
    snapshots = cassini_iss.from_index(file_name)
    output_buf = ''
    output_buf += generate_metadata(snapshots[432], 8.)
    #file_line = generate_metadata(snapshots[504], 8.)
    #print file_line
    #output_buf += file_line
    #for snapshot in snapshots:
    """for i in range(0,500):
        snapshot = snapshots[i]
        print snapshot.dict['FILE_NAME']
        file_line = generate_metadata(snapshot)
        output_buf += file_line"""
    return output_buf

list_file = open(index_file, 'r')
lines = list_file.readlines()
for line in lines:
    line = line.strip('\n')
    print "generate table for index: %s" % line
    output_buf = generate_table_for_index(line)
    output_file = open("test_geom.tab", 'w')
    output_file.write(output_buf)
    output_file.close()
