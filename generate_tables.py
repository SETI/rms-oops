import numpy as np
import pylab
import oops
import oops.inst.cassini.iss as cassini_iss
import oops_.surface.ansa as ansa_
from math import log, ceil
from oops_.meshgrid import Meshgrid

index_file = "geometry_list.txt"
spacer = '    ,   '

PRINT = False
DISPLAY = False

def add_info(array):
    print "array min/max:"
    if np.all(array.mask):
        print ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        output_buf = ", -0.1000000000000000E+31, -0.1000000000000000E+31"
    else:
        print array.min()
        print array.max()
        output_buf = ", %.15f, %.15f" % (array.min(), array.max())
    return output_buf

def write_output(snapshot):
    
    #print the camera field
    output_buf = '"S/IMG/CO/ISS/'
    output_buf += snapshot.index_dict['IMAGE_NUMBER']
    name = snapshot.index_dict["INSTRUMENT_NAME"]
    if "WIDE" in name:
        camera = '/W"'
    else:
        camera = '/N"'
    output_buf += camera

    output_buf += add_info(snapshot.right_ascension)
    output_buf += add_info(snapshot.declination)
    output_buf += add_info(snapshot.ring_radius)
    output_buf += add_info(snapshot.ring_radial_resolution)

    return output_buf

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


def generate_metadata(snapshot, resolution):
    
    #print the camera field
    output_buf = '"S/IMG/CO/ISS/'
    output_buf += snapshot.index_dict['IMAGE_NUMBER']
    name = snapshot.index_dict["INSTRUMENT_NAME"]
    if "WIDE" in name:
        camera = '/W"'
    else:
        camera = '/N"'
    output_buf += camera

    meshgrid = Meshgrid.for_fov(snapshot.fov, undersample=resolution, swap=True)
    bp = oops.Backplane(snapshot, meshgrid)
    test = bp.right_ascension()
    output_buf += add_info(test * oops.DPR)

    test = bp.declination()
    output_buf += add_info(test * oops.DPR)

    test = bp.ring_radius("saturn_main_rings")
    output_buf += add_info(test)

    test = bp.ring_radial_resolution("saturn_main_rings")
    output_buf += add_info(test)

    test = bp.ring_longitude("saturn_main_rings",reference="j2000")
    output_buf += add_info(test * oops.DPR)
    
    test = bp.ring_longitude("saturn_main_rings", reference="obs")
    output_buf += add_info(test * oops.DPR)

    test = bp.ring_longitude("saturn_main_rings", reference="sha")
    output_buf += add_info(test * oops.DPR)
        
    test = bp.phase_angle("saturn_main_rings")
    output_buf += add_info(test * oops.DPR)

    test = bp.incidence_angle("saturn_main_rings")
    output_buf += add_info(test * oops.DPR)
    
    test = bp.emission_angle("saturn_main_rings")
    output_buf += add_info(test * oops.DPR)

    test = bp.range("saturn_main_rings")
    output_buf += add_info(test)

    test = bp.where_outside_shadow("saturn", "saturn_main_rings")
    output_buf += " %.15f," % np.float(np.all(test.vals))
    output_buf += " %.15f," % np.float(np.any(test.vals))

    intercepted = bp.where_intercepted("saturn_main_rings")
    saturn_in_front = bp.where_in_front("saturn", "saturn_main_rings")
    saturn_intercepted = bp.where_intercepted("saturn")
    rings_blocked = saturn_in_front & saturn_intercepted
    rings_in_view = intercepted & (~rings_blocked)
    #test = bp.where_in_back("saturn", "saturn_main_rings")
    output_buf += " %.15f," % np.float(np.all(rings_in_view.vals))
    output_buf += " %.15f," % np.float(np.any(rings_in_view.vals))

    #pylab.imshow(rings_in_view.vals)
    #raw_input("Press Enter to continue...")

    #####################################################
    # ANSA surface values
    #####################################################
    # For single-point calculations about the geometry
    point_event = oops.Event(snapshot.midtime, (0.,0.,0.), (0.,0.,0.),
                             snapshot.path_id, snapshot.frame_id)
    # Define the apparent location of the observer relative to Saturn ring frame
    ring_body = oops.registry.body_lookup("SATURN_MAIN_RINGS")
    ring_center_event = ring_body.path.photon_to_event(point_event)
    ring_center_event = ring_center_event.wrt_frame(ring_body.frame_id)

    # Event separation in ring surface coordinates
    obs_wrt_ring_center = oops.Edelta.sub_events(point_event, ring_center_event)
    fov_shape = snapshot.fov.uv_shape

    if not is_power_of_2(resolution):
        resolution = next_power_of_2(resolution)

    uv_pair = oops.Pair.cross_scalars(np.arange(fov_shape.vals[0]/resolution)*resolution + 0.5,
                                      np.arange(fov_shape.vals[1]/resolution)*resolution + 0.5)

    los = snapshot.fov.los_from_uv(uv_pair, derivs=True)

    ansa_surface = ansa_.Ansa(ring_body.path_id, ring_body.frame_id)
    cept_event = ansa_surface.photon_to_event(point_event)
    #cept = ansa_surface.intercept(obs_wrt_ring_center.pos, los)
    print "cept_event.pos:"
    print cept_event.pos
    r_z_theta = ansa_surface.as_coords(cept_event.pos, obs_wrt_ring_center.pos, 3)
    r_z_theta.mask = rings_in_view

    print "r max:"
    print np.max(r_z_theta.mvals[...,0])

    return output_buf


def generate_table_for_index(file_name):
    snapshots = cassini_iss.from_index(file_name)
    output_buf = ''
        #for i in range(252, 500):
        #for i in range(0,500):
    for i in range(432,433):
        snapshot = snapshots[i]
        print snapshot.index_dict['FILE_NAME']
        file_line = generate_metadata(snapshot, 8)
        output_buf += file_line + '\n'
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
