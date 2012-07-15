import oops
import numpy as np
import sys
import csv
import datetime
import math

import pylab
import oops.inst.cassini.iss as cassini_iss
import oops.inst.cassini.vims as cassini_vims
import oops_.surface.ansa as ansa_
from math import log, ceil
from oops_.meshgrid import Meshgrid
from oops_.inventory import Inventory
import vicar
import oops_.registry as registry
from oops_.event import Event
from oops_.array.all import *
import oops_.path.all as path_
import oops_.frame.all as frame_
import oops_.surface.all as surface_


ISS_TYPE = "ISS"
VIMS_TYPE = "VIMS"
UVIS_TYPE = "UVIS"



################################################################################
# Hanlde command line arguments for generate_tables.py
################################################################################
list_file = 'geometry_list.csv'
radii_file = 'geometry_ring_ranges.csv'
output_file = 'body_list.csv'
grid_resolution = 8.
start_file_index = 0
stop_file_index = -1
write_frequency = 50
do_single_observation = False

nArguments = len(sys.argv)
for i in range(nArguments):
    if sys.argv[i] == '-lf':
        if i < (nArguments - 1):
            list_file = sys.argv[i+1]
            i += 1
    elif sys.argv[i] == '-rf':
        if i < (nArguments - 1):
            radii_file = sys.argv[i+1]
            i += 1
    elif sys.argv[i] == '-res':
        if i < (nArguments - 1):
            grid_resolution = float(sys.argv[i+1])
            i += 1
    elif sys.argv[i] == '-start':
        if i < (nArguments - 1):
            start_file_index = int(sys.argv[i+1])
            i += 1
    elif sys.argv[i] == '-stop':
        if i < (nArguments - 1):
            stop_file_index = int(sys.argv[i+1])
            i += 1
    elif sys.argv[i] == '-wfreq':
        if i < (nArguments - 1):
            write_frequency = int(sys.argv[i+1])
            i += 1
    elif sys.argv[i] == '-single':
        if i < (nArguments - 1):
            do_single_observation = int(sys.argv[i+1])
            i += 1
    elif sys.argv[i] == '-h':
        print "usage: python %s [-lf list_file] [-rf radii_file] [-res grid_resolution] [-start start_file_index] [-stop stop_file_index] [-wfreq write_frequency] [-single do_single_observation]" % sys.argv[0]
        print "default values:"
        print "\tlist_file: ", list_file
        print "\tradii_file: ", radii_file
        print "\tgrid_resolution: ", grid_resolution
        print "\tstart_file_index: ", start_file_index
        print "\tstop_file_index (-1 means do all): ", stop_file_index
        print "\twrite_frequency: ", write_frequency
        print "\tdo_single_observation: ", int(do_single_observation)
        sys.exit()

def image_code_name(snapshot, file_type):
    if file_type is ISS_TYPE:
        image_code = snapshot.index_dict['IMAGE_NUMBER']
        name = snapshot.index_dict["INSTRUMENT_NAME"]
        if "WIDE" in name:
            image_code += '/W'
        else:
            image_code += '/N'
    else:
        file_name = snapshot.index_dict['FILE_NAME']
        #strip letter off of front and extension off of end
        number_code = file_name.split('.')[0][1:]
        if "_IR" in snapshot.path_id:
            wave = 'I'
            if 'NORMAL' in snapshot.index_dict['IR_SAMPLING_MODE_ID']:
                res = 'N'
            else:
                res = 'L'
        else:
            wave = 'V'
            if 'NORMAL' in snapshot.index_dict['VIS_SAMPLING_MODE_ID']:
                res = 'N'
            else:
                res = 'L'
        image_code = number_code + '/' + wave + '/' + res
        print "image_code: ", image_code
    return image_code

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

def obs_file_type(observation):
    try:
        type = observation.dict["INSTRUMENT_ID"]
    except KeyError:
        return UVIS_TYPE;

    if type is VIMS_TYPE:
        return VIMS_TYPE
    return ISS_TYPE

def csv_row(list, delimiter):
    s = ''
    for i in range(len(list)):
        s += delimiter + list[i]
    return s

def csv_rows(dict, delimiter):
    s = ''
    keylist = dict.keys()
    keylist.sort()
    for key in keylist:
        s += key + csv_row(dict[key], delimiter) + '\n'
    return s

def get_error_buffer_size(snapshot, file_type):
    # deal with possible pointing errors - up to 3 pixels in any
    # direction for WAC and 30 pixels for NAC
    error_buffer = 60
    if file_type is ISS_TYPE:
        name = snapshot.index_dict["INSTRUMENT_NAME"]
        if "WIDE" in name:
            error_buffer = 6
    elif file_type is VIMS_TYPE:
        if 'NORMAL' not in snapshot.index_dict['VIS_SAMPLING_MODE_ID']:
            error_buffer = 6
    return error_buffer

def mask_targets(snapshot, file_type):
    """With Saturn (if in list of bodies) and the target body for the snapshot
        (if not Saturn), create a mask and see which of the remaining bodies
        reside outside the mask"""
    error_buffer = get_error_buffer_size(snapshot, file_type)
    limit = snapshot.fov.uv_shape + oops.Pair(np.array([error_buffer,
                                                        error_buffer]))
    
    meshgrid = Meshgrid.for_fov(snapshot.fov, undersample=grid_resolution,
                                limit=limit, swap=True)
    bp = oops.Backplane(snapshot, meshgrid)
    
    intercepted = bp.where_intercepted("saturn")
    target = snapshot.index_dict["TARGET_NAME"].lower()
    print "target = ", target
    if target is not "saturn":
        t_intercepted = bp.where_intercepted(target)
        mask = intercepted | t_intercepted
    else:
        mask = intercepted
    return mask

def get_distance_backplane(snapshot, file_type):
    """With Saturn (if in list of bodies) and the target body for the snapshot
        (if not Saturn), create a mask and see which of the remaining bodies
        reside outside the mask"""
    error_buffer = get_error_buffer_size(snapshot, file_type)
    limit = snapshot.fov.uv_shape + oops.Pair(np.array([error_buffer,
                                                        error_buffer]))
    
    meshgrid = Meshgrid.for_fov(snapshot.fov, undersample=grid_resolution,
                                limit=limit, swap=True)
    bp = oops.Backplane(snapshot, meshgrid)
    distance_backplane = bp.distance("saturn")
    target = snapshot.index_dict["TARGET_NAME"].lower()
    if target is not "saturn":
        t_distance = bp.distance(target)
        occ_distance = np.minimum(distance_backplane.vals, t_distance.vals)
    else:
        occ_distance = distance_backplane.vals
    return occ_distance

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


def single_body_table(file_name, fortran_list, omit_range):
    
    file_type = index_file_type(file_name)
    if file_type is ISS_TYPE:
        snapshot = cassini_iss.obs_from_index(file_name, start_file_index)
    #elif file_type is VIMS_TYPE:
    #    snapshot = cassini_vims.from_index(file_name)
    else:
        print "unsupported file type for index file."
        return None
    
    bodies = [registry.body_lookup("SATURN"),
              registry.body_lookup("MIMAS"),
              registry.body_lookup("ENCELADUS"),
              registry.body_lookup("TETHYS"),
              registry.body_lookup("DIONE"),
              registry.body_lookup("RHEA"),
              registry.body_lookup("TITAN"),
              registry.body_lookup("IAPETUS"),
              registry.body_lookup("PHOEBE"),
              registry.body_lookup("HYPERION"),
              registry.body_lookup("PAN"),
              registry.body_lookup("DAPHNIS"),
              registry.body_lookup("PROMETHEUS"),
              registry.body_lookup("ATLAS"),
              registry.body_lookup("PANDORA"),
              registry.body_lookup("EPIMETHEUS"),
              registry.body_lookup("JANUS"),
              registry.body_lookup("AEGAEON"),
              registry.body_lookup("METHONE"),
              registry.body_lookup("ANTHE"),
              registry.body_lookup("PALLENE"),
              registry.body_lookup("TELESTO"),
              registry.body_lookup("HELENE"),
              registry.body_lookup("POLYDEUCES"),
              registry.body_lookup("HYPERION"),
              registry.body_lookup("CALYPSO")]
    
    snapshots_bodies = {}
    print "processing observation ", start_file_index
    image_code = image_code_name(snapshot, file_type)
    if start_file_index not in omit_range:
        snapshot_bodies = []
        mask = mask_targets(snapshot, file_type)
        for mass_body in bodies:
            if(snapshot.any_part_object_in_view("CASSINI", mass_body, mask)):
                snapshot_bodies.append(mass_body.name)
        snapshots_bodies[image_code] = snapshot_bodies

        fh = open(output_file, 'a')
        fh.write(csv_rows(snapshots_bodies, ','))
        fh.close()

def point_in_frustrum(frustrum_normals, body, pos, snapshot_bodies,
                      body_positions):
    intersects = [False, False, False, False]
    positive_side = np.array([False, False, False, False])
    for i in range(4):
        dist = frustrum_normals[i].dot(pos)
        if dist.vals < (-body.radius):
            return   # entirely outside of plane
        elif math.fabs(dist.vals) < body.radius:
            # intersects plane, thereforem possibly in the viewing frustrum
            # check if intersects OR positive side of perpendicular planes
            j = (i+1)%4
            k = (i+3)%4
            if intersects[j] or intersects[k]:
                snapshot_bodies.append(body)
                body_positions.append(pos.vals)
                return
            elif positive_side[j] and positive_side[k]:
                snapshot_bodies.append(body)
                body_positions.append(pos.vals)
                return
            intersects[i] = True
        else:
            positive_side[i] = True
    # if we have reached here, might be inside frustrum
    if np.all(positive_side):
        snapshot_bodies.append(body)
        body_positions.append(pos.vals)


def generate_body_table(file_name):

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
    start = start_file_index
    if start < 0:
        start = 0

    print "Processing indices %d to %d" % (start_file_index, nObs)

    bodies = [registry.body_lookup("SATURN"),
              registry.body_lookup("MIMAS"),
              registry.body_lookup("ENCELADUS"),
              registry.body_lookup("TETHYS"),
              registry.body_lookup("DIONE"),
              registry.body_lookup("RHEA"),
              registry.body_lookup("TITAN"),
              registry.body_lookup("IAPETUS"),
              registry.body_lookup("PHOEBE"),
              registry.body_lookup("HYPERION"),
              registry.body_lookup("PAN"),
              registry.body_lookup("DAPHNIS"),
              registry.body_lookup("PROMETHEUS"),
              registry.body_lookup("ATLAS"),
              registry.body_lookup("PANDORA"),
              registry.body_lookup("EPIMETHEUS"),
              registry.body_lookup("JANUS"),
              registry.body_lookup("AEGAEON"),
              registry.body_lookup("METHONE"),
              registry.body_lookup("ANTHE"),
              registry.body_lookup("PALLENE"),
              registry.body_lookup("TELESTO"),
              registry.body_lookup("HELENE"),
              registry.body_lookup("POLYDEUCES"),
              registry.body_lookup("CALYPSO")]

    obs_bodies = {}
    progress_str = ""
    progress = Progress(start, nObs)
    for i in range(start_file_index, nObs):
        if file_type is ISS_TYPE:
            ob = obs[i]
        else:
            ob = obs[i][1]
        image_code = image_code_name(ob, file_type)
        error_buffer_size = get_error_buffer_size(ob, file_type)
        inventory = Inventory(ob, bodies)
        confirmed_bodies = inventory.where_not_blocked("CASSINI", "saturn",
                                                       error_buffer_size)
        if confirmed_bodies is not None:
            obs_bodies[image_code] = confirmed_bodies
        
        time_left = progress.time_left(i)
        
        l = len(progress_str)
        time_left_str = "File " + str(i+1) + ": " + image_code + ", time rem: " + str(time_left)
        progress_str = time_left_str.split('.')[0]
        for j in range(l):
            sys.stdout.write('\b')
        sys.stdout.write(progress_str)
        sys.stdout.flush()

    print "\nWriting output file ", output_file
    fh = open(output_file, 'w')
    fh.write(csv_rows(obs_bodies, ','))
    fh.close()


def get_fortran_file_list(fortran_file_name):
    file_list = []
    if fortran_file_name != "":
        reader = csv.reader(open(fortran_file_name, 'rU'), delimiter=',')
        for row in reader:
            files = str(row[0]).split('S/IMG/CO/ISS/')
            file_plus_letter = files[1]
            file_list.append(file_plus_letter)
    return file_list

################################################################################
# main program
################################################################################
volumeReader = csv.reader(open(list_file, 'rU'), delimiter=';')
for row in volumeReader:
    index_file_name = str(row[0])
    output_file = str(row[1])
#    if do_single_observation:
#        single_body_table(index_file_name, fortran_list, omit_range)
#    else:
    generate_body_table(index_file_name)
if not do_single_observation:
    print "Done."
