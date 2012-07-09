import numpy as np
import sys
import csv
import datetime
import math

import pylab
import oops
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


def generate_body_table(file_name, fortran_list, omit_range):
    # this step actually UNREGISTERS everything for some reason.
    # make sure registry is set up
    #registry.initialize_frame_registry()
    #registry.initialize_path_registry()
    #registry.initialize_body_registry()

    file_type = index_file_type(file_name)
    if file_type is ISS_TYPE:
        snapshots = cassini_iss.from_index(file_name)
    elif file_type is VIMS_TYPE:
        snapshots = cassini_vims.from_index(file_name)
    else:
        print "unsupported file type for index file."
        return None
    nSnapshots = len(snapshots)
    if stop_file_index > 0 and stop_file_index < nSnapshots:
        nSnapshots = stop_file_index

    print "Processing indices %d to %d" % (start_file_index, nSnapshots)

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
    then = datetime.datetime.now()
    progress_str = ""
    for i in range(start_file_index, nSnapshots):
        if file_type is ISS_TYPE:
            snapshot = snapshots[i]
        else:
            snapshot = snapshots[i][1]
        image_code = image_code_name(snapshot, file_type)
        if (i not in omit_range) and ((len(fortran_list) == 0) or (image_code in fortran_list)):

            error_buffer_size = get_error_buffer_size(snapshot, file_type)
            inventory = Inventory(snapshot, bodies)
            confirmed_bodies = inventory.where_not_blocked("CASSINI", "saturn",
                                                           error_buffer_size)
            """
            # get image event for this snapshot
            image_event = Event(snapshot.midtime, (0,0,0), (0,0,0), "CASSINI",
                                snapshot.frame_id)

            # get the planes of the viewing frustrum
            uv_shape = snapshot.fov.uv_shape.vals
            uv = np.array([(0,0),
                           (uv_shape[0],0),
                           (uv_shape[0],uv_shape[1]),
                           (0,uv_shape[1])])
            uv_pair = Pair(uv)
            los = snapshot.fov.los_from_uv(uv_pair)
            # scale los to avoid accuracy problems
            los *= 1000000.

            # if the dot product of the normal of each plane and the center of
            # object is less then negative the radius, we lie entirely outside
            # that plane of the frustrum, and therefore no part is in fov.  If
            # between -r and r, we have intersection, therefore part in fov.
            # otherwise, entirely in fov.
            # for the moment use the bounding sphere, but in future use a
            # bounding box which would be more appropriate for objects not
            # sphere-like
            frustrum_normals = []
            for i in range(4):
                #j = (i+1)%4
                #frustrum_normals.append(los[j].cross(los[i]).unit())
                frustrum_normals.append(los[i].cross(los[(i+1)%4]).unit())

            snapshot_bodies = []
            body_positions = []
            #mask = mask_targets(snapshot)
            #print " in generate_body_table, mask = ", mask
            for mass_body in bodies:
                #if(snapshot.surface_in_view_neighborhood(mass_body.surface,
                #                                         sin_angle)):
                path = path_.Path.connect(mass_body.path_id, "CASSINI",
                                          snapshot.frame_id)
                abs_event = path.photon_to_event(image_event)
                rel_event = abs_event.wrt_path("CASSINI")
                # first check that body is in front of camera
                if rel_event.pos.vals[2] < 0.:
                    continue;
                point_in_frustrum(frustrum_normals, mass_body, rel_event.pos,
                                  snapshot_bodies, body_positions)
                
                #if(snapshot.any_part_object_in_view("CASSINI", mass_body)):
                #    snapshot_bodies.append(mass_body.name.strip())

            if len(snapshot_bodies) == 1 and snapshot_bodies[0] == "saturn":
                confirmed_bodies.append("saturn")
            else:
                error_buffer = get_error_buffer_size(snapshot, file_type)
                limit = snapshot.fov.uv_shape + oops.Pair(np.array([error_buffer,
                                                                    error_buffer]))

                meshgrid = Meshgrid.for_fov(snapshot.fov, undersample=grid_resolution,
                                            limit=limit, swap=True)
                bp = oops.Backplane(snapshot, meshgrid)
                saturn_distance = bp.distance("saturn")

                target = snapshot.index_dict["TARGET_NAME"].strip().lower()
                confirmed_bodies = []
                if target != "saturn":
                    t_distance = bp.distance(target)
                    if saturn_distance.shape == []:
                        confirmed_bodies.append(target)
                    else:
                        if np.less(t_distance.vals, saturn_distance.vals).any():
                            # target is closer than saturn somehwere
                            confirmed_bodies.append(target)
                        if np.less(saturn_distance.vals, t_distance.vals).any():
                            # saturn is closwer than target somewhere
                            confirmed_bodies.append("saturn")
                    occ_distance = np.minimum(saturn_distance.vals, t_distance.vals)
                    target_saturn_distance = np.ma.array(occ_distance,
                                                         mask=saturn_distance.mask & t_distance.mask)
                else:
                    if saturn_distance.shape != []:
                        confirmed_bodies.append("saturn")
                    target_saturn_distance = saturn_distance.vals


                body_number = 0
                for potential_body in snapshot_bodies:
                    p_body = potential_body.name.lower()
                    if (p_body != "saturn") and (p_body != target):
                        pos = body_positions[body_number]
                        pix = snapshot.fov.uv_from_los(pos)
                        # now, if the center of the body is off the FOV, then
                        # the closest point is still going to be part of the
                        # same body (if that body is at least partially in the
                        # FOV, which we know it is).
                        if pix.vals[0] < 0:
                            pix.vals[0] = 0
                        elif pix.vals[0] >= target_saturn_distance.shape[0]:
                            pix.vals[0] = target_saturn_distance.shape[0]
                        if pix.vals[1] < 0:
                            pix.vals[1] = 0
                        elif pix.vals[1] >= target_saturn_distance.shape[1]:
                            pix.vals[1] = target_saturn_distance.shape[1]
                        ix = int(pix.vals[0])
                        iy = int(pix.vals[1])
                        
                        if pos[2] < target_saturn_distance[ix][iy]:
                            confirmed_bodies.append(p_body)
                        elif target_saturn_distance.mask[ix][iy]:
                            confirmed_bodies.append(p_body)
                        else:
                            # most time consuming test... test ring of points,
                            # but don't bother if there are no true mask values
                            if np.any(target_saturn_distance.mask.vals):
                                # get radius, in pixels, at the distance of the
                                # body in the FOV
                                radius_los = np.array([potential_body.radius,
                                                       potential_body.radius,
                                                       pos[2]])
                                # we only need to check perpendicular directions
                                # since if they are blocked, all points in
                                # between are blocked.
                                pix_radius = snapshot.fov.uv_from_los(radius_los)
                                ix1 = min(int(ix + pix_radius[0]), target_saturn_distance.shape[0])
                                ix2 = max(int(ix - pix_radius[0]), 0)
                                iy1 = min(int(iy + pix_radius[1]), target_saturn_distance.shape[1])
                                iy2 = max(int(iy - pix_radius[1]), 0)
                                if target_saturn_distance.mask[ix1][iy] or target_saturn_distance.mask[ix2][iy] or target_saturn_distance.mask[ix][iy1] or target_saturn_distance.mask[ix][iy2]:
                                    confirmed_bodies.append(p_body)
                                
                                
                    body_number += 1
                """
    
    
            #print "confirmed_bodies: ", csv_row(confirmed_bodies, ';')
            snapshots_bodies[image_code] = confirmed_bodies

        now = datetime.datetime.now()
        time_so_far = now - then
        time_left = time_so_far * (nSnapshots - start_file_index) / (i + 1 - start_file_index) - time_so_far

        l = len(progress_str)
        time_left_str = "File " + str(i+1) + ": " + image_code + ", time rem: " + str(time_left)
        progress_str = time_left_str.split('.')[0]
        for j in range(l):
            sys.stdout.write('\b')
        sys.stdout.write(progress_str)
        sys.stdout.flush()

    print "\nWriting output file ", output_file
    fh = open(output_file, 'w')
    fh.write(csv_rows(snapshots_bodies, ','))
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
    fortran_list = get_fortran_file_list(str(row[1]))
    output_file = str(row[2])
    omit_range = []
    len_of_row = len(row)
    for i in range(3,len_of_row):
        omit_range.append(int(row[i]))
    if do_single_observation:
        single_body_table(index_file_name, fortran_list, omit_range)
    else:
        generate_body_table(index_file_name, fortran_list, omit_range)
if not do_single_observation:
    print "Done."
