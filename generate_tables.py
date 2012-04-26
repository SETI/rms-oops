import numpy as np
import sys
print sys.executable
print sys.path
import csv
import datetime

import pylab
import oops
import oops.inst.cassini.iss as cassini_iss
import oops_.surface.ansa as ansa_
from math import log, ceil
from oops_.meshgrid import Meshgrid
import vicar

#index_file = "geometry_list.txt"
list_file = 'geometry_list.csv'
spacer = '    ,   '

PRINT = False
DISPLAY = False
OBJC_DISPLAY = True

def add_info(array, minmax=None):
    if PRINT:
        print "array min/max:"
    if minmax is not None:
        if PRINT:
            print ", %f, %f" % (minmax[0], minmax[1])
        output_buf = ", " + str(minmax[0]) + ", " + str(minmax[1])
    elif np.all(array.mask):
        if PRINT:
            print ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        output_buf = ", -0.1000000000000000E+31, -0.1000000000000000E+31"
    else:
        if PRINT:
            print array.min()
            print array.max()
        output_buf = ", %.15f, %.15f" % (array.min(), array.max())
    return output_buf

def show_info(title, array):
    """Internal method to print summary information and display images as
        desired."""
    
    global PRINT, DISPLAY
    if not PRINT:
        if OBJC_DISPLAY:
            if isinstance(array, np.ndarray):
                if array.dtype == np.dtype("bool"):
                    pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                                 array, vmin=0, vmax=1, cmap=pylab.cm.gray)
                else:
                    minval = np.min(array)
                    maxval = np.max(array)
                    if minval != maxval:
                         pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                                      array, cmap=pylab.cm.gray)
            elif isinstance(array, oops.Array):
                if np.any(array.mask):
                    pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                                 array.vals)
                else:
                    minval = np.min(array.vals)
                    maxval = np.max(array.vals)
                    if minval != maxval:
                            pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                                         array.vals, cmap=pylab.cm.gray)
        return
    
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
            elif OBJC_DISPLAY:
                pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                             array, vmin=0, vmax=1, cmap=pylab.cm.gray)
        
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
                elif OBJC_DISPLAY:
                    pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                                 array, cmap=pylab.cm.gray)
    
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
            elif OBJC_DISPLAY:
                pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                             array.vals, cmap=pylab.cm.gray)
        
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
                elif OBJC_DISPLAY:
                    pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                                 array.vals, cmap=pylab.cm.gray)
    
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

def argmax(items, func, cmp=cmp):
    '''argmax(items, func) -> max_item, max_value'''
    it = iter(items)
    # Initial values
    try:
        item = it.next()
    except StopIteration:
        raise ValueError("can't run over empty sequence")
    val = func(item)
    
    for i in it:
        v = func(i)
        if cmp(v, val) == 1:
            item = i
            val = v
    return item, val

def argmin(items, func, cmp=cmp):
    '''argmin(items, func) -> min_item, min_value'''
    return argmax(items, func, lambda x,y : -cmp(x,y))

def angle_single_limits(angles):
    
    #find the histogram for the angles
    bins = np.arange(361)
    m_arr = angles.mvals
    h,b = np.histogram(m_arr[~m_arr.mask], bins)    # hist doesn't support masks
    
    #find all of the sections without values (where the histogram is zero)
    empty_sections = []
    first_non_zero = -1
    i = 0
    in_zero = False
    num_zero = 0
    min_zero = 5
    current_value_start = 0.
    current_value_stop = 0.
    sorted_angles = m_arr.copy().reshape(m_arr.size)    # to get actual values
    sorted_angles.sort()
    samples = 0
    while i < 360:
        if h[i] == 0 and (not in_zero):
            num_zero += 1
            if num_zero == min_zero:
                current_section_start = i + 1 - num_zero
                if i != 0:
                    current_value_start = sorted_angles[samples-1]
                in_zero = True
        elif h[i] != 0:
            num_zero = 0
            if in_zero:
                current_section = (current_section_start,
                                   i - current_section_start,
                                   current_value_start,
                                   sorted_angles[samples])
                empty_sections.append(current_section)
                in_zero = False
            if first_non_zero == -1:
                first_non_zero = i
        if i == 359:
            if first_non_zero == -1:
                return (angles.min(), angles.max())   # we have no angles
            if in_zero:
                current_section = (current_section_start,
                                   i + 1 - current_section_start,
                                   current_value_start, 0.)
                empty_sections.append(current_section)
        samples += h[i]
        i += 1
    
    # if we have a a gap that starts at zero and ends at 360, we need to combine
    # them
    first_section = empty_sections[0]
    if first_section[0] == 0:
        last_section = empty_sections[-1]
        if (last_section[0] + last_section[1]) == 360:
            new_section = (last_section[0],
                           first_section[1] + last_section[1],
                           last_section[2],
                           first_section[3])
            empty_sections[0] = new_section
            empty_sections.pop()

    
    #find the largest of those
    #(item,val) = argmax(empty_sections, key=lambda x: x[1])
    item = max(empty_sections, key=lambda x: x[1])
    return (item[3], item[2])

def create_overlay_image(overlay_mask, resolution, underlay_file_name,
                         overlay_data=None):
    rel_path = "./test_data/cassini/ISS/" + underlay_file_name
    vic = vicar.VicarImage.from_file(rel_path)
    vic_res = vic.data[0].shape
    ov_res = np.empty([vic_res[0], vic_res[1]])
    ir = int(resolution)
    mask_val = vic.data[0].max()
    for j in range(0,vic_res[1]):
        for i in range(0,vic_res[0]):
            d = overlay_mask.vals[i/ir][j/ir]
            if d == 0.:
                ov_res[i][j] = vic.data[0][i][j]
            elif overlay_data is None:
                ov_res[i][j] = mask_val
            else:
                ov_res[i][j] = overlay_data[i/ir][j/ir]
    pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png", ov_res,
                 cmap=pylab.cm.gray)

def generate_metadata(snapshot, resolution):
    
    #print the camera field
    output_buf = '"S/IMG/CO/ISS/'
    
    output_buf += snapshot.index_dict['IMAGE_NUMBER']
    name = snapshot.index_dict["INSTRUMENT_NAME"]
    #output_buf += str(snapshot.vicar_dict['IMAGE_NUMBER'])
    #name = snapshot.vicar_dict["INSTRUMENT_NAME"]
    if "WIDE" in name:
        camera = '/W"'
    else:
        camera = '/N"'
    output_buf += camera

    meshgrid = Meshgrid.for_fov(snapshot.fov, undersample=resolution, swap=True)
    bp = oops.Backplane(snapshot, meshgrid)

    intercepted = bp.where_intercepted("saturn_main_rings")
    #show_info("Rings:", intercepted)
    saturn_in_front = bp.where_in_front("saturn", "saturn_main_rings")
    saturn_intercepted = bp.where_intercepted("saturn")
    rings_blocked = saturn_in_front & saturn_intercepted
    rings_in_view = intercepted & (~rings_blocked)
    #test = bp.where_in_back("saturn", "saturn_main_rings")

    rings_not_in_shadow = bp.where_outside_shadow("saturn_main_rings", "saturn")
    vis_rings_not_shadow = rings_in_view & rings_not_in_shadow
    #show_info("mask:", ~vis_rings_not_shadow)
    #create_overlay_image(vis_rings_not_shadow, resolution,
    #                     snapshot.index_dict["FILE_NAME"])

#meshgrid.uv.mask = ~vis_rings_not_shadow

#    not_visible_lit_rings = intercepted & rings_blocked & (~rings_not_in_shadow)
#    not_visible_lit_rings = (~rings_in_view) & (~rings_not_in_shadow)
#    create_overlay_image(not_visible_lit_rings, resolution,
#                         snapshot.index_dict["FILE_NAME"])

    test = bp.right_ascension()
    test.mask |= ~vis_rings_not_shadow.vals
    output_buf += add_info(test * oops.DPR)

    test = bp.declination()
    test.mask |= ~vis_rings_not_shadow.vals
    output_buf += add_info(test * oops.DPR)

    test = bp.ring_radius("saturn_main_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    output_buf += add_info(test)

    test = bp.ring_radial_resolution("saturn_main_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    output_buf += add_info(test)

    test = bp.ring_longitude("saturn_main_rings",reference="j2000")
    test.mask |= ~vis_rings_not_shadow.vals
    limits = angle_single_limits(test * oops.DPR)
    output_buf += add_info(test, limits)
    
    test = bp.ring_longitude("saturn_main_rings", reference="obs")
    test.mask |= ~vis_rings_not_shadow.vals
    limits = angle_single_limits(test * oops.DPR)
    output_buf += add_info(test, limits)

    test = bp.ring_longitude("saturn_main_rings", reference="sha")
    test.mask |= ~vis_rings_not_shadow.vals
    limits = angle_single_limits(test * oops.DPR)
    output_buf += add_info(test, limits)
        
    test = bp.phase_angle("saturn_main_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    output_buf += add_info(test * oops.DPR)

    test = bp.incidence_angle("saturn_main_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    output_buf += add_info(test * oops.DPR)
    
    test = bp.emission_angle("saturn_main_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    output_buf += add_info(test * oops.DPR)

    test = bp.range("saturn_main_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    output_buf += add_info(test)

    output_buf += " %.15f," % np.float(np.all(rings_not_in_shadow.vals))
    output_buf += " %.15f," % np.float(np.any(rings_not_in_shadow.vals))
    #show_info("Shadow:", test)

    output_buf += " %.15f," % np.float(np.all(rings_in_view.vals))
    output_buf += " %.15f," % np.float(np.any(rings_in_view.vals))

    test = bp.where_sunward("saturn_main_rings")
    output_buf += " %.15f," % np.float(np.all(test.vals))
    output_buf += " %.15f," % np.float(np.any(test.vals))

    #pylab.imshow(rings_in_view.vals)
    #raw_input("Press Enter to continue...")

    #####################################################
    # ANSA surface values
    #####################################################
    intercepted = bp.where_intercepted("saturn_main_rings:ansa")
    saturn_in_front = bp.where_in_front("saturn:ansa", "saturn_main_rings:ansa")
    saturn_intercepted = bp.where_intercepted("saturn:ansa")
    rings_blocked = saturn_in_front & saturn_intercepted
    rings_in_view = intercepted & (~rings_blocked)

    rings_not_in_shadow = bp.where_outside_shadow("saturn_main_rings:ansa",
                                                  "saturn:ansa")
    vis_rings_not_shadow = rings_in_view & rings_not_in_shadow

    test = bp.ansa_radius("saturn:ansa")
    test.mask |= ~vis_rings_not_shadow.vals
    output_buf += add_info(test)
    #show_info("Saturn:ansa radius (km)", test)
        
    test = bp.ansa_elevation("saturn:ansa")
    test.mask |= ~vis_rings_not_shadow.vals
    output_buf += add_info(test)
    #show_info("Saturn:ansa elevation (km)", test)

    #test = bp.ring_longitude("saturn:ansa",reference="j2000")
    #test.mask |= ~vis_rings_not_shadow.vals
    #output_buf += add_info(test * oops.DPR)
    output_buf += ", -0.1000000000000000E+31, -0.1000000000000000E+31"

    #test = bp.ring_longitude("saturn:ansa", reference="sha")
    #test.mask |= ~vis_rings_not_shadow.vals
    #output_buf += add_info(test * oops.DPR)
    output_buf += ", -0.1000000000000000E+31, -0.1000000000000000E+31"

    test = bp.range("saturn:ansa")
    test.mask |= ~vis_rings_not_shadow.vals
    output_buf += add_info(test)

    test = bp.ansa_radial_resolution("saturn:ansa")
    test.mask |= ~vis_rings_not_shadow.vals
    output_buf += add_info(test)

    return output_buf


def generate_table_for_index(file_name, omit_range, fortran_list):
    print "omit_range = ", omit_range
    output_buf = ''
    snapshots = cassini_iss.from_index(file_name)
    nSnapshots = len(snapshots)
    i = 0
    then = datetime.datetime.now()
    start = 2209
    #for i in range(start,nSnapshots):
    for i in range(start,2210):
        snapshot = snapshots[i]
        #for snapshot in snapshots:
        image_code = snapshot.index_dict['IMAGE_NUMBER']
        name = snapshot.index_dict["INSTRUMENT_NAME"]
        if "WIDE" in name:
            image_code += '/W'
        else:
            image_code += '/N'
    
        if i not in omit_range and image_code in fortran_list:
            print "    %d of %d, %s" % (i, nSnapshots,
                                        snapshot.index_dict['FILE_NAME'])
            file_line = generate_metadata(snapshot, 8)
            output_buf += file_line + '\n'
        else:
            print "        OMITTING %d of %d, %s" % (i, nSnapshots,
                                                     snapshot.index_dict['FILE_NAME'])
        now = datetime.datetime.now()
        time_so_far = now - then
        time_left = time_so_far * (nSnapshots - start) / (i + 1 - start) - time_so_far
        print "        time remaining: (approximately) ", time_left
        if (i % 100) == 99:
            temp_file_name = "./test_data/cassini/ISS/COISS_geom/temp" + str(i) + ".tab"
            f = open(temp_file_name, 'w')
            f.write(output_buf)
            f.close()
        i += 1
    
    return output_buf

def get_fortran_file_list(fortran_file_name):
    reader = csv.reader(open(fortran_file_name, 'rU'), delimiter=',')
    file_list = []
    for row in reader:
        files = str(row[0]).split('S/IMG/CO/ISS/')
        file_plus_letter = files[1]
        file_list.append(file_plus_letter)
    return file_list

volumeReader = csv.reader(open(list_file, 'rU'), delimiter=';')
for row in volumeReader:
    index_file_name = str(row[0])
    fortran_list = get_fortran_file_list(str(row[1]))
    geom_file_name = str(row[2])
    omit_range = []
    len_of_row = len(row)
    for i in range(3,len_of_row):
        omit_range.append(int(row[i]))
    print "Generating geometry table for file: ", index_file_name
    output_buf = generate_table_for_index(index_file_name, omit_range,
                                          fortran_list)
    geom_output_file = open(geom_file_name, 'w')
    geom_output_file.write(output_buf)
    geom_output_file.close()
print "Done."
"""
list_file = open(index_file, 'r')
lines = list_file.readlines()
for line in lines:
    line = line.strip('\n')
    print "generate table for index: %s" % line
    output_buf = generate_table_for_index(line)
    output_file = open("test_geom.tab", 'w')
    output_file.write(output_buf)
    output_file.close()
"""
