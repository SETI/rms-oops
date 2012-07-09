import oops
import numpy as np
import sys
import csv
import datetime

import pylab
import oops.inst.cassini.iss as cassini_iss
import oops.inst.cassini.vims as cassini_vims
import oops_.surface.ansa as ansa_
from math import log, ceil
from oops_.meshgrid import Meshgrid
import vicar


ISS_TYPE = "ISS"
VIMS_TYPE = "VIMS"
UVIS_TYPE = "UVIS"

################################################################################
# Hanlde command line arguments for generate_tables.py
################################################################################
list_file = 'geometry_list.csv'
grid_resolution = 8.
start_file_index = 0
stop_file_index = -1
write_frequency = 50
do_output = True

nArguments = len(sys.argv)
for i in range(nArguments):
    if sys.argv[i] == '-lf':
        if i < (nArguments - 1):
            i += 1
            list_file = sys.argv[i]
    elif sys.argv[i] == '-res':
        if i < (nArguments - 1):
            i += 1
            grid_resolution = float(sys.argv[i])
    elif sys.argv[i] == '-start':
        if i < (nArguments - 1):
            i += 1
            start_file_index = int(sys.argv[i])
    elif sys.argv[i] == '-stop':
        if i < (nArguments - 1):
            i += 1
            stop_file_index = int(sys.argv[i])
    elif sys.argv[i] == '-wfreq':
        if i < (nArguments - 1):
            i += 1
            write_frequency = int(sys.argv[i])
    elif sys.argv[i] == '-out':
        if i < (nArguments - 1):
            i += 1
            do_output = int(sys.argv[i])
    elif sys.argv[i] == '-h':
        print "usage: python %s [-lf list_file] [-res grid_resolution] [-start start_file_index] [-stop stop_file_index] [-wfreq write_frequency] [-out do_output]" % sys.argv[0]
        print "default values:"
        print "\tlist_file: ", list_file
        print "\tgrid_resolution: ", grid_resolution
        print "\tstart_file_index: ", start_file_index
        print "\tstop_file_index (-1 means do all): ", stop_file_index
        print "\twrite_frequency: ", write_frequency
        print "\tdo_output: ", int(do_output)
        sys.exit()


def image_code_name(ob, file_type):
    if file_type is ISS_TYPE:
        image_code = ob.index_dict['IMAGE_NUMBER']
        name = ob.index_dict["INSTRUMENT_NAME"]
        if "WIDE" in name:
            image_code += '/W'
        else:
            image_code += '/N'
    else:
        file_name = ob.index_dict['FILE_NAME']
        #strip letter off of front and extension off of end
        number_code = file_name.split('.')[0][1:]
        if "_IR" in ob.path_id:
            wave = 'I'
            if 'NORMAL' in ob.index_dict['IR_SAMPLING_MODE_ID']:
                res = 'N'
            else:
                res = 'L'
        else:
            wave = 'V'
            if 'NORMAL' in ob.index_dict['VIS_SAMPLING_MODE_ID']:
                res = 'N'
            else:
                res = 'L'
        image_code = number_code + '/' + wave + '/' + res
    return image_code


################################################################################
# BodySurface                                                                  #
################################################################################

class BodySurface(object):

    def __init__(self, owner=None, parent=None):
        self.owner = owner
        if parent is None:
            self.clear()
        else:
            self.copy_data(parent)

    
    def clear(self):
        pass


    def copy_data(self, parent):
        pass


    def output_minmax_info(self, array, minmax=None):
        if minmax is not None:
            output_buf = ", " + str(minmax[0]) + ", " + str(minmax[1])
        elif array is None:
            output_buf = ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        elif np.all(array.mask):
            output_buf = ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        else:
            output_buf = ", %.15f, %.15f" % (array.min(), array.max())
        return output_buf


    def angle_wrap_coverage(self, angles):
        
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
                        if samples == 0:
                            current_value_start = 0.
                        else:
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

    def output_line(self):
        """output to a string line"""
        line = self.owner.obs_id
        line += ", SATURN"  # in future have this as a variable
        line += self.output_minmax_info(self.geocentric_latitude * oops.DPR)
        line += self.output_minmax_info(self.geographic_latitude * oops.DPR)
        line += self.output_minmax_info(self.iau_longitude * oops.DPR)
        line += self.output_minmax_info(self.sha_longitude * oops.DPR)
        line += self.output_minmax_info(self.obs_longitude * oops.DPR)
        line += self.output_minmax_info(self.finest_resolution)
        line += self.output_minmax_info(self.coarsest_resolution)
        line += self.output_minmax_info(self.phase * oops.DPR)
        line += self.output_minmax_info(self.incidence * oops.DPR)
        line += self.output_minmax_info(self.emission * oops.DPR)
        line += self.output_minmax_info(self.range_to_body)
        line += " %f," % np.float(np.all(self.night_side_flag.vals))
        line += " %f," % np.float(np.any(self.night_side_flag.vals))
        line += " %f," % np.float(np.all(self.behind_rings_flag.vals))
        line += " %f," % np.float(np.any(self.behind_rings_flag.vals))
        line += " %f," % np.float(np.all(self.in_ring_shadow_flag.vals))
        line += " %f," % np.float(np.any(self.in_ring_shadow_flag.vals))
        
        line += " %f," % self.sub_solar_long.vals
        line += " %f," % self.sub_solar_lat.vals
        line += " %f," % self.solar_dist.vals
        line += " %f," % self.sub_obs_long.vals
        line += " %f," % self.sub_obs_lat.vals
        line += " %f," % self.obs_dist.vals
        line += '\n'
        return line

################################################################################
# BodySurfaceSummary                                                           #
################################################################################

class BodySurfaceSummary(BodySurface):
    
    def __init__(self, owner=None, parent=None):
        self.owner = owner
        if parent is None:
            self.clear()
        else:
            self.copy_data(parent)
    
    def clear(self):
        # these are min/max pairs
        self.geocentric_latitude = None
        self.geographic_latitude = None
        self.iau_longitude = None
        self.sha_longitude = None
        self.obs_longitude = None
        self.finest_resolution = None
        self.coarsest_resolution = None
        self.phase = None
        self.incidence = None
        self.emission = None
        self.range_to_body = None
        self.night_side_flag = None
        self.behind_rings_flag = None
        self.in_ring_shadow_flag = None
        # these are single values
        self.sub_solar_geocentric_latitude = None
        self.sub_solar_geographic_latitude = None
        self.sub_solar_iau_longitude = None
        self.solar_distance_to_body_center = None
        self.sub_obs_geocentric_latitude = None
        self.sub_obs_geographic_latitude = None
        self.sub_obs_iau_longitude = None
        self.obs_distance_to_body_center = None

    def copy_data(self, parent):
        self.file_type = parent.file_type

        self.obs_id = parent.obs_id
        self.body_name = parent.body_name
        
        self.geocentric_latitude = parent.geocentric_latitude
        self.geographic_latitude = parent.geographic_latitude
        self.iau_longitude = parent.iau_longitude
        self.sha_longitude = parent.sha_longitude
        self.obs_longitude = parent.obs_longitude
        self.finest_resolution = parent.finest_resolution
        self.coarsest_resolution = parent.coarsest_resolution
        self.phase = parent.phase
        self.incidence = parent.incidence
        self.emission = parent.emission
        self.range_to_body = parent.range_to_body
        self.night_side_flag = parent.night_side_flag
        self.behind_rings_flag = parent.behind_rings_flag
        self.in_ring_shadow_flag = parent.in_ring_shadow_flag

        self.sub_solar_geocentric_longitude = parent.sub_solar_geocentric_longitude
        self.sub_solar_geographic_longitude = parent.sub_solar_geographic_longitude
        self.sub_solar_iau_longitude = parent.sub_solar_iau_longitude
        self.solar_distance_to_body_center = parent.solar_distance_to_body_center
        self.sub_obs_geocentric_longitude = parent.sub_obs_geocentric_longitude
        self.sub_obs_geographic_longitude = parent.sub_obs_geographic_longitude
        self.sub_obs_iau_longitude = parent.sub_obs_iau_longitude
        self.obs_distance_to_body_center = parent.obs_distance_to_body_center
    
    def output_line(self):
        """output to a string line"""
        line = self.owner.obs_id
        line += ", %s" % self.owner.body_name
        line += self.output_minmax_info(self.geocentric_latitude * oops.DPR)
        line += self.output_minmax_info(self.geographic_latitude * oops.DPR)
        line += self.output_minmax_info(self.iau_longitude * oops.DPR)
        line += self.output_minmax_info(self.sha_longitude * oops.DPR)
        line += self.output_minmax_info(self.obs_longitude * oops.DPR)
        line += self.output_minmax_info(self.finest_resolution)
        line += self.output_minmax_info(self.coarsest_resolution)
        line += self.output_minmax_info(self.phase * oops.DPR)
        line += self.output_minmax_info(self.incidence * oops.DPR)
        line += self.output_minmax_info(self.emission * oops.DPR)
        line += self.output_minmax_info(self.range_to_body)
        line += " %f," % np.float(np.all(self.night_side_flag.vals))
        line += " %f," % np.float(np.any(self.night_side_flag.vals))
        line += " %f," % np.float(np.all(self.behind_rings_flag.vals))
        line += " %f," % np.float(np.any(self.behind_rings_flag.vals))
        line += " %f," % np.float(np.all(self.in_ring_shadow_flag.vals))
        line += " %f," % np.float(np.any(self.in_ring_shadow_flag.vals))
        line += " %f," % self.sub_solar_geocentric_latitude.vals * oops.DPR
        line += " %f," % self.sub_solar_geographic_latitude.vals * oops.DPR
        line += " %f," % self.sub_solar_iau_longitude.vals * oops.DPR
        line += " %f," % self.solar_distance_to_body_center.vals
        line += " %f," % self.sub_obs_geocentric_latitude.vals * oops.DPR
        line += " %f," % self.sub_obs_geographic_latitude.vals * oops.DPR
        line += " %f," % self.sub_obs_iau_longitude.vals * oops.DPR
        line += " %f\n" % self.obs_distance_to_body_center.vals
        return line

################################################################################
# BodySurfaceDetail                                                            #
################################################################################

class BodySurfaceDetail(BodySurface):
    
    def __init__(self, owner=None, parent=None):
        self.owner = owner
        if parent is None:
            self.clear()
        else:
            self.copy_data(parent)
    
    def clear(self):
        self.latitude_band_number = None
        # these are min/max pairs
        self.geocentric_latitude = None
        self.geographic_latitude = None
        self.iau_longitude = None
        self.sha_longitude = None
        self.obs_longitude = None
        self.finest_resolution = None
        self.coarsest_resolution = None
        self.phase = None
        self.incidence = None
        self.emission = None
        self.range_to_body = None
        self.night_side_flag = None
        self.behind_rings_flag = None
        self.in_ring_shadow_flag = None
    
    def copy_data(self, parent):
        self.latitude_band_number = parent.latitude_band_number
        
        self.geocentric_latitude = parent.geocentric_latitude
        self.geographic_latitude = parent.geographic_latitude
        self.iau_longitude = parent.iau_longitude
        self.sha_longitude = parent.sha_longitude
        self.obs_longitude = parent.obs_longitude
        self.finest_resolution = parent.finest_resolution
        self.coarsest_resolution = parent.coarsest_resolution
        self.phase = parent.phase
        self.incidence = parent.incidence
        self.emission = parent.emission
        self.range_to_body = parent.range_to_body
        self.night_side_flag = parent.night_side_flag
        self.behind_rings_flag = parent.behind_rings_flag
        self.in_ring_shadow_flag = parent.in_ring_shadow_flag
    
    def output_line(self):
        """output to a string line"""
        line = self.owner.obs_id
        line += ", %s" % self.owner.body_name
        line += ", %d" % self.latitude_band_number
        line += self.output_minmax_info(self.geocentric_latitude * oops.DPR)
        line += self.output_minmax_info(self.geographic_latitude * oops.DPR)
        line += self.output_minmax_info(self.iau_longitude * oops.DPR)
        line += self.output_minmax_info(self.sha_longitude * oops.DPR)
        line += self.output_minmax_info(self.obs_longitude * oops.DPR)
        line += self.output_minmax_info(self.finest_resolution)
        line += self.output_minmax_info(self.coarsest_resolution)
        line += self.output_minmax_info(self.phase * oops.DPR)
        line += self.output_minmax_info(self.incidence * oops.DPR)
        line += self.output_minmax_info(self.emission * oops.DPR)
        line += self.output_minmax_info(self.range_to_body)
        line += " %f," % np.float(np.all(self.night_side_flag.vals))
        line += " %f," % np.float(np.any(self.night_side_flag.vals))
        line += " %f," % np.float(np.all(self.behind_rings_flag.vals))
        line += " %f," % np.float(np.any(self.behind_rings_flag.vals))
        line += " %f," % np.float(np.all(self.in_ring_shadow_flag.vals))
        line += " %f\n" % np.float(np.any(self.in_ring_shadow_flag.vals))
        return line

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
                        if samples == 0:
                            current_value_start = 0.
                        else:
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
        
        angle_masks = []
        spans = []
        for i in range(len(empty_sections)):
            min_angle = empty_sections[i][2] / oops.DPR
            max_angle = empty_sections[i][3] / oops.DPR
            span = (max_angle, min_angle)
            spans.append(span)
            a = angles.vals > min_angle
            b = angles.vals < max_angle
            if min_angle > max_angle:
                angle_mask = a | b
            else:
                #print "empty_section: ", empty_sections[i]
                #print "min_angle: ", min_angle
                #print "a: ", a
                #print "b: ", b
                angle_mask = a & b
            angle_masks.append(angle_mask)
        return (angle_masks, spans)


################################################################################
# BodyLimb                                                                     #
################################################################################

class BodyLimb(BodySurface):
    
    def __init__(self, owner=None, parent=None):
        self.owner = owner
        if parent is None:
            self.clear()
        else:
            self.copy_data(parent)
    
    def clear(self):
        # these are min/max pairs
        self.elevation = None
        self.geocentric_latitude = None
        self.geographic_latitude = None
        self.resolution = None
        self.phase = None
        self.incidence = None
        self.range_to_limb = None
    
    def copy_data(self, parent):
        self.elevation = parent.elevation
        self.geocentric_latitude = parent.geocentric_latitude
        self.geographic_latitude = parent.geographic_latitude
        self.resolution = parent.resolution
        self.phase = parent.phase
        self.incidence = parent.incidence
        self.range_to_limb = parent.range_to_limb
    
    def output_line(self):
        """output to a string line"""
        line = self.owner.obs_id
        line += ", %s" % self.owner.body_name
        line += self.output_minmax_info(self.elevation)
        line += self.output_minmax_info(self.geocentric_latitude * oops.DPR)
        line += self.output_minmax_info(self.geographic_latitude * oops.DPR)
        line += self.output_minmax_info(self.resolution)
        line += self.output_minmax_info(self.phase * oops.DPR)
        line += self.output_minmax_info(self.incidence * oops.DPR)
        line += self.output_minmax_info(self.range_to_limb)
        line += "\n"
        return line

class FileGeometry(object):
    
    def __init__(self, body_name, parent=None):
        self.body_name = body_name.upper()
        if parent is None:
            self.file_type = ISS_TYPE
            self.clear()
        else:
            self.copy_data(parent)
    
    def clear(self):
        self.obs_id = ""
        self.summary = BodySurfaceSummary(self)
        self.details = []
        self.limb = BodyLimb(self)
    
    def copy_data(self, parent):
        self.file_type = parent.file_type
        
        self.obs_id = parent.obs_id
        self.body_name = parent.body_name

        self.file_type = parent.file_type
        self.summary.copy(parent.summary)
        self.details.empty()
        for body_detail in parent.details:
            new_detail = BodySurfaceDetail(self)
            new_detail.copy(body_detail)
            self.details.append(new_detail)
        self.limb.copy(parent.limb)

    def output(self):
        lines = self.summary.output_line()
        for detail in self.details:
            lines += detail.output_line()
        lines += self.limb.output_line()
        return lines
    
    def set_image_id(self, obs):
        """output to string the image/camera label"""
        if self.file_type is ISS_TYPE:
            self.obs_id = '"S/IMG/CO/ISS/'
            
            self.obs_id += obs.index_dict['IMAGE_NUMBER']
            name = obs.index_dict["INSTRUMENT_NAME"]
            if "WIDE" in name:
                self.obs_id += '/W"'
            else:
                self.obs_id += '/N"'
        else:
            file_name = obs.index_dict['FILE_NAME']
            #strip letter off of front and extension off of end
            number_code = file_name.split('.')[0][1:]
            if "_IR" in obs.path_id:
                wave = 'I'
                if 'NORMAL' in obs.index_dict['IR_SAMPLING_MODE_ID']:
                    res = 'N'
                else:
                    res = 'L'
            else:
                wave = 'V'
                if 'NORMAL' in obs.index_dict['VIS_SAMPLING_MODE_ID']:
                    res = 'N'
                else:
                    res = 'L'
            self.obs_id = number_code + '/' + wave + '/' + res


    def compute_details(self, freq):
        """compute the Body Details list given the frequency.
            Input   freq - in degrees, the frequency of the bands."""
        start_angle = -90.
        end_angle = freq
        band_number = 1
        del self.details[:]
        while start_angle < 90.:
            a = self.summary.geocentric_latitude.mvals < (start_angle * oops.RPD)
            b = self.summary.geocentric_latitude.mvals >= (end_angle * oops.RPD)
            latitude_mask = a | b
            if not np.all(latitude_mask):
                detail = BodySurfaceDetail(self)
                detail.latitude_band_number = band_number
                detail.geocentric_latitude = self.summary.geocentric_latitude.copy()
                detail.geocentric_latitude.mask |= latitude_mask
                detail.geographic_latitude = self.summary.geographic_latitude.copy()
                detail.geographic_latitude.mask |= latitude_mask
                detail.iau_longitude = self.summary.iau_longitude.copy()
                detail.iau_longitude.mask |= latitude_mask
                detail.sha_longitude = self.summary.sha_longitude.copy()
                detail.sha_longitude.mask |= latitude_mask
                detail.obs_longitude = self.summary.obs_longitude.copy()
                detail.obs_longitude.mask |= latitude_mask
                detail.finest_resolution = self.summary.finest_resolution.copy()
                detail.finest_resolution.mask |= latitude_mask
                detail.coarsest_resolution = self.summary.coarsest_resolution.copy()
                detail.coarsest_resolution.mask |= latitude_mask
                detail.phase = self.summary.phase.copy()
                detail.phase.mask |= latitude_mask
                detail.incidence = self.summary.incidence.copy()
                detail.incidence.mask |= latitude_mask
                detail.emission = self.summary.emission.copy()
                detail.emission.mask |= latitude_mask
                detail.range_to_body = self.summary.range_to_body.copy()
                detail.range_to_body.mask |= latitude_mask
                detail.night_side_flag = self.summary.night_side_flag.copy()
                detail.night_side_flag.mask |= latitude_mask
                detail.behind_rings_flag = self.summary.behind_rings_flag.copy()
                detail.behind_rings_flag.mask |= latitude_mask
                detail.in_ring_shadow_flag = self.summary.in_ring_shadow_flag.copy()
                detail.in_ring_shadow_flag.mask |= latitude_mask
                self.details.append(detail)
            band_number += 1
            start_angle += freq
            end_angle += freq
            if end_angle > 90.:
                end_angle = 90.



################################################################################
# Actual generation of the data                                                #
# deal with possible pointing errors - up to 3 pixels in any                   #
# direction for WAC and 30 pixels for NAC                                      #
################################################################################
def get_error_buffer_size(snapshot, file_type):
    error_buffer = 60
    if file_type is ISS_TYPE:
        name = snapshot.index_dict["INSTRUMENT_NAME"]
        if "WIDE" in name:
            error_buffer = 6
    elif file_type is VIMS_TYPE:
        if 'NORMAL' not in snapshot.index_dict['VIS_SAMPLING_MODE_ID']:
            error_buffer = 6
    return error_buffer

################################################################################
# Actual generation of the data                                                #
################################################################################
def generate_metadata(obs, resolution, file_type, body_name):
    
    geometry = FileGeometry(body_name)
    geometry.file_type = file_type
    geometry.set_image_id(obs)

    # deal with possible pointing errors - up to 3 pixels in any
    # direction for WAC and 30 pixels for NAC
    error_buffer = get_error_buffer_size(obs, file_type)
    limit = obs.fov.uv_shape + oops.Pair(np.array([error_buffer,
                                                        error_buffer]))
    
    meshgrid = Meshgrid.for_fov(obs.fov, undersample=resolution,
                                limit=limit, swap=True)

    bp = oops.Backplane(obs, meshgrid)
    
    try:
        intercepted = bp.where_intercepted("saturn_main_rings")
    except:
        return None
    intercept_mask = ~intercepted.vals

    result = bp.latitude(body_name)                      # geocentric latitude
    result.mask |= intercept_mask
    geometry.summary.geocentric_latitude = result.copy()
    
    result = bp.latitude(body_name, "graphic")           # geographic latitude
    result.mask |= intercept_mask
    geometry.summary.geographic_latitude = result.copy()
    
    result = bp.longitude(body_name)                     # iau longitude
    result.mask |= intercept_mask
    geometry.summary.iau_longitude = result.copy()
    
    result = bp.longitude(body_name, "sha")              # sha longitude
    result.mask |= intercept_mask
    geometry.summary.sha_longitude = result.copy()
    
    result = bp.longitude(body_name, "obs")              # obs longitude
    result.mask |= intercept_mask
    geometry.summary.obs_longitude = result.copy()
    
    result = bp.finest_resolution(body_name)             # finest resolution
    result.mask |= intercept_mask
    geometry.summary.finest_resolution = result.copy()
    
    result = bp.coarsest_resolution(body_name)           # coarsest resolution
    result.mask |= intercept_mask
    geometry.summary.coarsest_resolution = result.copy()
    
    result = bp.phase_angle(body_name)                   # phase angle
    result.mask |= intercept_mask
    geometry.summary.phase = result.copy()
    
    result = bp.incidence_angle(body_name)               # incidence angle
    result.mask |= intercept_mask
    geometry.summary.incidence = result.copy()
    
    result = bp.emission_angle(body_name)                # emission angle
    result.mask |= intercept_mask
    geometry.summary.emission = result.copy()
    
    result = bp.distance(body_name)                      # distance to body
    if result.shape != []:
        result.mask |= intercept_mask
    geometry.summary.range_to_body = result.copy()
    
    result = bp.where_sunward(body_name)                 # night side flag
    geometry.summary.night_side_flag = result.copy()
    
    result = bp.where_in_back(body_name, "saturn_main_rings")# where behind rings
    geometry.summary.behind_rings_flag = result.copy()
    
    result = bp.where_inside_shadow(body_name, "saturn_main_rings")  # in shadow
    geometry.summary.in_ring_shadow_flag = result.copy()

    result = bp.sub_solar_latitude(body_name)            # subsolar centric lat
    geometry.summary.sub_solar_geocentric_latitude = result.copy()
    
    result = bp.sub_solar_latitude(body_name, "graphic") # subsolar graphic lat
    geometry.summary.sub_solar_geographic_latitude = result.copy()
    
    result = bp.sub_solar_longitude(body_name)           # subsolar iau longitude
    geometry.summary.sub_solar_iau_longitude = result.copy()
    
    result = bp.solar_distance_to_center(body_name) # solar distance to body
    geometry.summary.solar_distance_to_body_center = result.copy()
    
    result = bp.sub_observer_latitude(body_name)            # obs centric lat
    geometry.summary.sub_obs_geocentric_latitude = result.copy()
    
    result = bp.sub_observer_latitude(body_name, "graphic")  # obs centric lat
    geometry.summary.sub_obs_geographic_latitude = result.copy()
    
    result = bp.sub_observer_longitude(body_name)            # obs centric lat
    geometry.summary.sub_obs_iau_longitude = result.copy()
            
    result = bp.observer_distance_to_center(body_name)
    geometry.summary.obs_distance_to_body_center = result.copy()

    #onto the body detail
    geometry.compute_details(15.)

    #onto the Limb
    limb_body = body_name + ":limb"
    body_intercepted = bp.where_intercepted(limb_body)
    limb_mask = ~body_intercepted.vals

    result = bp.elevation(limb_body)
    result.mask |= limb_mask
    geometry.limb.elevation = result.copy()

    result = bp.latitude(limb_body)
    result.mask |= limb_mask
    geometry.limb.geocentric_latitude = result.copy()
    
    result = bp.latitude(limb_body, "graphic")
    result.mask |= limb_mask
    geometry.limb.geographic_latitude = result.copy()
    
    result = bp.resolution(limb_body)
    result.mask |= limb_mask
    geometry.limb.resolution = result.copy()
    
    result = bp.phase_angle(limb_body)
    result.mask |= limb_mask
    geometry.limb.phase = result.copy()
    
    result = bp.incidence_angle(limb_body)
    result.mask |= limb_mask
    geometry.limb.incidence = result.copy()
    
    result = bp.distance(limb_body)
    result.mask |= limb_mask
    geometry.limb.range_to_limb = result.copy()

    return geometry



################################################################################
# timing class                                                                 #
#                                                                              #
# class to figure out time left whie running through loop with fairly consistent
# amount of time for each step of loop.                                        #
#                                                                              #
#total - (current_index - start)      time_left                                #
#-------------------------------   =  ---------                                #
#total                                total_time                               #
#                                                                              #
#current_index - start     time_so_far                                         #
#---------------------  =  -----------                                         #
#total                     total_time                                          #
################################################################################
class Progress(object):

    def __init__(self, start, end):
        self.beginTime = datetime.datetime.now()
        self.start = start
        self.end = end
        self.total = end - start

    def time_left(self, current_index):
        current_time = datetime.datetime.now()
        time_so_far = current_time - self.beginTime
        index_so_far = current_index + 1 - self.start
        total_time = time_so_far * self.total / index_so_far
        return total_time * (self.total - index_so_far) / self.total


################################################################################
# helper functions                                                             #
################################################################################
def index_file_type(file_name):
    fd = open(file_name)
    for line in fd:
        if "INSTRUMENT_ID" in line:
            if VIMS_TYPE in line:
                fd.close()
                return VIMS_TYPE
        elif "DWELL_TIME" in line:
            fd.close()
            return UVIS_TYPE
    fd.close()
    return ISS_TYPE

def append_to_file(file_name, geometries, start, stop):
    if start < write_frequency:
        f = open(file_name, 'w')
    else:
        f = open(file_name, 'a')
    output_buf = ""
    for i in range(start, stop):
        output_buf += geometries[i].output()
    f.write(output_buf)
    f.close()

################################################################################
# generate the geometries                                                      #
################################################################################

def generate_geometries_for_index(file_name, body_name):
    """Create the geometry objects for all the image files in the list of files
        within the index file file_name.
        
        Input:
            file_name       index file for this volume.
            omit_range      list of images, by index, to omit that are not
                            omitted in the FORTRAN code.
            fortran_list    list of files processed in the FORTRAN code (so we
                            can avoid further segmnentation faults.
        """
    
    # note that "obs" is used for the array of observations, and "ob" is used as
    # a single observation
    file_type = index_file_type(file_name)
    if file_type is ISS_TYPE:
        obs = cassini_iss.from_index(file_name)
    elif file_type is VIMS_TYPE:
        obs = cassini_vims.from_index(file_name)
    else:
        print "unsupported file type for index file."
        return None
    nObs = len(obs)
    if stop_file_index > 0 and stop_file_index < nObs:
        nObs = stop_file_index

    # initialize some variables
    i = 0
    then = datetime.datetime.now()
    start = start_file_index
    geometries = []
    info_str = ""
    actual_i = start
    start_write_index = 0
    progress = Progress(start, nObs)
    omitting = []
    
    for i in range(start, nObs):
        if file_type is ISS_TYPE:
            ob = obs[i]
        else:
            ob = obs[i][1]
    
        info_len = len(info_str)
        i1 = i + 1
        init_info_str = "    " + str(i+1) + " of " + str(nObs)
            #        if i not in omit_range:
        geometry = generate_metadata(ob, grid_resolution, file_type, body_name.lower())
        if geometry is not None:
            geometries.append(geometry)
            actual_i += 1
        else:
            omitting.append(i)
        
        # write time and image number status to stdout
        time_left = progress.time_left(i)
        init_info_str += ", time rem: " + str(time_left)
        info_str = init_info_str.split('.')[0]
        for item in range(0, info_len):
            sys.stdout.write('\b')
        sys.stdout.write(info_str)
        sys.stdout.flush()
            #        if i not in omit_range and (len(fortran_list) == 0) or (image_code in fortran_list):
        if write_frequency > 0 and (actual_i % write_frequency) == 0:
            i0 = actual_i - write_frequency
            if do_output:
                info_str = "appending rows %d to %d to output file %s" % (i0,
                                                                          actual_i,
                                                                          geom_file_name)
                for item in range(0, info_len):
                    sys.stdout.write('\b')
                sys.stdout.write(info_str)
                sys.stdout.flush()
                
                end_write_index = len(geometries)
                append_to_file(geom_file_name, geometries, start_write_index,
                               end_write_index)
                start_write_index = end_write_index
                
                # blank out the longer text of this line
                l = len(info_str)
                for ispace in range(l):
                    sys.stdout.write('\b')
                for ispace in range(l):
                    sys.stdout.write(' ')
    
    if write_frequency > 0 and (actual_i % write_frequency) != 0:
        i0 = (actual_i / write_frequency) * write_frequency
        if do_output:
            print "\nappending rows %d to %d to output file %s" % (i0,
                                                                   actual_i,
                                                                   geom_file_name)
            end_write_index = len(geometries)
            append_to_file(geom_file_name, geometries, start_write_index,
                           end_write_index)
    
    sys.stdout.write('\n')
    if len(omitting) > 0:
        sys.stdout.write( "Omitted image(s) ")
        for i in omitting:
            sys.stdout.write(i)
        sys.stdout.write('\n')
    return geometries


################################################################################
# Main Program                                                                 #
################################################################################

volumeReader = csv.reader(open(list_file, 'rU'), delimiter=';')
for row in volumeReader:
    index_file_name = str(row[0])
    geom_file_name = str(row[1])
    body_name = str(row[2])
    print "Generating geometry table for %s, file: %s" % (body_name,
                                                          index_file_name)
    geometries = generate_geometries_for_index(index_file_name, body_name)
    if write_frequency < 0:
        if do_output:
            print "writing file ", geom_file_name
            output_file(geom_file_name, geometries)
print "Done."
