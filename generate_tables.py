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



################################################################################

class FileGeometry(object):
    
    def __init__(self, parent=None):
        if parent is None:
            self.clear()
        else:
            self.copy_data(parent)
    
    def clear(self):
        self.image_name = ""
        self.ra = None
        self.dec = None
        self.ring_radius = None
        self.radial_resolution = None
        self.longitude_j2000 = None
        self.longitude_obs = None
        self.longitude_sha = None
        self.phase = None
        self.incidence = None
        self.emission = None
        self.range_to_rings = None
        self.shadow_of_planet = None
        self.planet_behind_rings = None
        self.backlit_rings = None
        self.ansa_radius = None
        self.ansa_elevation = None
        self.ansa_longitude_j2000 = None
        self.ansa_longitude_sha = None
        self.ansa_range = None
        self.ansa_resolution = None

    def copy_data(self, parent):
        self.image_name = parent.image_name
        self.ra = parent.ra.copy()
        self.dec = parent.dec.copy()
        self.ring_radius = parent.ring_radius.copy()
        self.radial_resolution = parent.radial_resolution.copy()
        self.longitude_j2000 = parent.longitude_j2000.copy()
        self.longitude_obs = parent.longitude_obs.copy()
        self.longitude_sha = parent.longitude_sha.copy()
        self.phase = parent.phase.copy()
        self.incidence = parent.incidence.copy()
        self.emission = parent.emission.copy()
        self.range_to_rings = parent.range_to_rings.copy()
        self.shadow_of_planet = parent.shadow_of_planet.copy()
        self.planet_behind_rings = parent.planet_behind_rings.copy()
        self.backlit_rings = parent.backlit_rings.copy()
        self.ansa_radius = parent.ansa_radius.copy()
        self.ansa_elevation = parent.ansa_elevation.copy()
        self.ansa_longitude_j2000 = parent.ansa_longitude_j2000.copy()
        self.ansa_longitude_sha = parent.ansa_longitude_sha.copy()
        self.ansa_range = parent.ansa_range.copy()
        self.ansa_resolution = parent.ansa_resolution.copy()

    def or_mask(self, mask):
        self.ra.mask |= mask
        self.dec.mask |= mask
        self.ring_radius.mask |= mask
        self.radial_resolution.mask |= mask
        self.longitude_j2000.mask |= mask
        self.longitude_obs.mask |= mask
        self.longitude_sha.mask |= mask
        self.phase.mask |= mask
        self.incidence.mask |= mask
        self.emission.mask |= mask
        self.range_to_rings.mask |= mask
        self.shadow_of_planet.mask |= mask
        self.planet_behind_rings.mask |= mask
        self.backlit_rings.mask |= mask
        self.ansa_radius.mask |= mask
        self.ansa_elevation.mask |= mask
        self.ansa_longitude_j2000.mask |= mask
        self.ansa_longitude_sha.mask |= mask
        self.ansa_range.mask |= mask
        self.ansa_resolution.mask |= mask

    def set_image_camera(self, snapshot):
        """output to string the image/camera label"""
        self.image_name = '"S/IMG/CO/ISS/'

        self.image_name += snapshot.index_dict['IMAGE_NUMBER']
        name = snapshot.index_dict["INSTRUMENT_NAME"]
        if "WIDE" in name:
            camera = '/W"'
        else:
            camera = '/N"'
        self.image_name += camera

    def add_single_minmax_info(self, array, minmax=None):
        if minmax is not None:
            output_buf = ", " + str(minmax[0]) + ", " + str(minmax[1])
        elif array is None:
            output_buf = ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        elif np.all(array.mask):
            output_buf = ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        else:
            output_buf = ", %.15f, %.15f" % (array.min(), array.max())
        return output_buf

    def angle_single_limits(self, angles):
        
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
        item = max(empty_sections, key=lambda x: x[1])
        return (item[3], item[2])

    def angle_coverage(self, angles):
        """return a list of min/max pairs defining sections of angles that are
            covered in angles.
        """
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
        return empty_sections

    def output_opus2(self, radii_ranges):
        """output to a string lines in the OPUS2 format representing one image.
            
            Input:
            distance_ranges     list of distances for which we have a separate
                                line in the output file. For N values in
                                radii_ranges, we have N distance ranges to
                                evaluate. 0 should always be the first value,
                                so that range i includes the range
                                [radii_ranges[i], radii_ranges[i+1]) where
                                radii_ranges[N] is taken to be infinite.
        """
        l = len(radii_ranges)
        lines = ""
        for i in range(l):
            start = radii_ranges[i]
            if i < l:
                end = radii_ranges[i+1]
            else:
                end = sys.float_info.max
            radius_mask = self.ring_radius < start or self.ring_radius >= end
            
            # now for each radius range we have a line for each angle span
            # ACTUALLY, it makes no sense to print a separate line for each
            # span as what is filled in for the other values other than this
            # particular angle?  What about other angles than have multiple
            # min/max spans?
            #spans = self.angle_coverage(self.longitude_j2000 * oops.DPR)
            sub_geom = FileGeometry(self)
            sub_geom.or_mask(radius_mask)
            lines += sub_geom.output_single_line(start, end)
            
        return lines
        

    def output_single_line(self, start=None, end=None):
        """output to a string line in the OPUS1 format"""
        line = self.image_name
        if (start is not None) and (end is not None):
            # for Opus2 format
            line += ", %f, %f" % (start, end)
        line += self.add_single_minmax_info(self.ra * oops.DPR)
        line += self.add_single_minmax_info(self.dec * oops.DPR)
        line += self.add_single_minmax_info(self.ring_radius)
        line += self.add_single_minmax_info(self.radial_resolution)
        if not np.all(self.longitude_j2000.mask):
            limits = self.angle_single_limits(self.longitude_j2000 * oops.DPR)
            line += add_info(self.longitude_j2000, limits)
        else:
            line += ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        #line += self.add_single_minmax_info(self.longitude_j2000)
        limits = self.angle_single_limits(self.longitude_obs * oops.DPR)
        line += self.add_single_minmax_info(self.longitude_obs, limits)
        #line += self.add_single_minmax_info(self.longitude_obs)
        limits = self.angle_single_limits(self.longitude_sha * oops.DPR)
        line += self.add_single_minmax_info(self.longitude_sha, limits)
        #line += self.add_single_minmax_info(self.longitude_sha)
        line += self.add_single_minmax_info(self.phase * oops.DPR)
        line += self.add_single_minmax_info(self.incidence * oops.DPR)
        line += self.add_single_minmax_info(self.emission * oops.DPR)
        line += self.add_single_minmax_info(self.range_to_rings)
        line += " %.15f," % np.float(np.all(self.shadow_of_planet.vals))
        line += " %.15f," % np.float(np.any(self.shadow_of_planet.vals))
        #line += self.add_single_minmax_info(self.shadow_of_planet)
        line += " %.15f," % np.float(np.all(self.planet_behind_rings.vals))
        line += " %.15f," % np.float(np.any(self.planet_behind_rings.vals))
        #line += self.add_single_minmax_info(self.planet_behind_rings)
        line += " %.15f," % np.float(np.all(self.backlit_rings.vals))
        line += " %.15f," % np.float(np.any(self.backlit_rings.vals))
        #line += self.add_single_minmax_info(self.backlit_rings)
        line += self.add_single_minmax_info(self.ansa_radius)
        line += self.add_single_minmax_info(self.ansa_elevation)
        line += ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        line += ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        #line += self.add_single_minmax_info(self.ansa_longitude_j2000 * oops.DPR)
        #line += self.add_single_minmax_info(self.ansa_longitude_sha * oops.DPR)
        line += self.add_single_minmax_info(self.ansa_range)
        line += self.add_single_minmax_info(self.ansa_resolution)
        line += '\n'
        return line




list_file = 'geometry_list.csv'
radii_file = 'geometry_ring_ranges.csv'
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
    """output_buf = '"S/IMG/CO/ISS/'
    
    output_buf += snapshot.index_dict['IMAGE_NUMBER']
    name = snapshot.index_dict["INSTRUMENT_NAME"]
    if "WIDE" in name:
        camera = '/W"'
    else:
        camera = '/N"'
    output_buf += camera"""
    geometry = FileGeometry()
    geometry.set_image_camera(snapshot)

    meshgrid = Meshgrid.for_fov(snapshot.fov, undersample=resolution, swap=True)
    bp = oops.Backplane(snapshot, meshgrid)

    intercepted = bp.where_intercepted("saturn_main_rings")
    saturn_in_front = bp.where_in_front("saturn", "saturn_main_rings")
    saturn_intercepted = bp.where_intercepted("saturn")
    rings_blocked = saturn_in_front & saturn_intercepted
    rings_in_view = intercepted & (~rings_blocked)

    rings_not_in_shadow = bp.where_outside_shadow("saturn_main_rings", "saturn")
    vis_rings_not_shadow = rings_in_view & rings_not_in_shadow


    test = bp.right_ascension()
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.ra = test.copy()

    test = bp.declination()
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.dec = test.copy()

    test = bp.ring_radius("saturn_main_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.ring_radius = test.copy()

    test = bp.ring_radial_resolution("saturn_main_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.radial_resolution = test.copy()

    test = bp.ring_longitude("saturn_main_rings",reference="j2000")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.longitude_j2000 = test.copy()
    
    test = bp.ring_longitude("saturn_main_rings", reference="obs")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.longitude_obs = test.copy()

    test = bp.ring_longitude("saturn_main_rings", reference="sha")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.longitude_sha = test.copy()
        
    test = bp.phase_angle("saturn_main_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.phase = test.copy()

    test = bp.incidence_angle("saturn_main_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.incidence = test.copy()
    
    test = bp.emission_angle("saturn_main_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.emission = test.copy()

    test = bp.range("saturn_main_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    range_test = test.rebroadcast(test.mask.shape)
    geometry.range_to_rings = range_test.copy()

    geometry.shadow_of_planet = rings_not_in_shadow.copy()

    geometry.planet_behind_rings = rings_in_view.copy()

    test = bp.where_sunward("saturn_main_rings")
    geometry.backlit_rings = test.copy()

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
    geometry.ansa_radius = test.copy()
        
    test = bp.ansa_elevation("saturn:ansa")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.ansa_elevation = test.copy()

    """
    test = bp.ring_longitude("saturn:ansa",reference="j2000")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.ansa_longitude_j2000 = test.copy()

    test = bp.ring_longitude("saturn:ansa", reference="sha")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.ansa_longitude_sha = test.copy()
    """

    test = bp.range("saturn:ansa")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.ansa_range = test.copy()

    test = bp.ansa_radial_resolution("saturn:ansa")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.ansa_resolution = test.copy()

    return geometry

def output_opus1_file(file_name, geometries):
    f = open(file_name, 'w')
    output_buf = ""
    for geometry in geometries:
        output_buf += geometry.output_single_line()
    f.write(output_buf)
    f.close()

def output_opus2_file(file_name, geometries, radii_ranges):
    f = open(file_name, 'w')
    output_buf = ""
    for geometry in geometries:
        output_buf += geometry.output_opus2(radii_ranges)
    f.write(output_buf)
    f.close()


def generate_table_for_index(file_name, omit_range, fortran_list):
    print "omit_range = ", omit_range
    output_buf = ''
    snapshots = cassini_iss.from_index(file_name)
    nSnapshots = len(snapshots)
    i = 0
    then = datetime.datetime.now()
    start = 0
    geometries = []
    for i in range(start,nSnapshots):
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
            #file_line = generate_metadata(snapshot, 8)
            #output_buf += file_line + '\n'
            geometry = generate_metadata(snapshot, 256)
            geometries.append(geometry)
        else:
            print "        OMITTING %d of %d, %s" % (i, nSnapshots,
                                                     snapshot.index_dict['FILE_NAME'])
        now = datetime.datetime.now()
        time_so_far = now - then
        time_left = time_so_far * (nSnapshots - start) / (i + 1 - start) - time_so_far
        print "        time remaining: (approximately) ", time_left
        """if (i % 100) == 99:
            temp_file_name = "./test_data/cassini/ISS/COISS_geom/temp" + str(i) + ".tab"
            output_opus1_file(temp_file_name, geometries)"""
        #i += 1
    
    #return output_buf
    return geometries

def get_fortran_file_list(fortran_file_name):
    reader = csv.reader(open(fortran_file_name, 'rU'), delimiter=',')
    file_list = []
    for row in reader:
        files = str(row[0]).split('S/IMG/CO/ISS/')
        file_plus_letter = files[1]
        file_list.append(file_plus_letter)
    return file_list

radiiReader = csv.reader(open(radii_file, 'rU'), delimiter=';')
radii_ranges = []
for row in radiiReader:
    for i in range(len(row)):
        radii_ranges.append(float(row[i]))

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
    geometries = generate_table_for_index(index_file_name, omit_range,
                                          fortran_list)
    output_opus1_file(geom_file_name, geometries)
    output_opus2_file(geom_file_name, geometries, radii_ranges)
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
