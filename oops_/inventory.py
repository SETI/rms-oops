################################################################################
# oops_/inventory.py: Inventory class
#
# 6/12/12 Created (BSW)
################################################################################
import oops

import numpy as np
import math

import oops_.config as config
import oops_.constants as constants
import oops_.registry as registry
import oops_.surface.all as surface_
import oops_.path.all as path_
from oops_.array.all import *
from oops_.event import Event
from oops_.meshgrid import Meshgrid
from oops_.backplane import Backplane

class Inventory(object):
    """Inventory is a class that allows for finding the inventory of bodies
        within the observation.  Different methods allow for finding bodies
        under different conditions."""
    
    def __init__(self, obs, bodies):
        """The constructor.
            
            Input:
            obs         the Observation object with which this inventory is
                        associated.
            bodies      a list of bodies to check whether within the
                        observaiton.
            """
        
        self.obs = obs
        self.bodies = bodies

    
    def any_part_object_in_view(self, origin_id, mass_body):
        """Determine if any part of mass_body is w/in field-of-view.
            NOTE: This code is for a sphere to approximate.
            
            Input:
            origin_id   id of camera
            mass_body   body to check
            
            Return:     Boolean.  True if any part w/in fov.
            """
        # get position of object relative to view point
        image_event = Event(self.obs.midtime, (0,0,0), (0,0,0), origin_id,
                            self.obs.frame_id)
        surface_path = path_.Path.connect(mass_body.path_id, origin_id,
                                          self.obs.frame_id)
        #rel_event = surface_path.photon_to_event(image_event)[1]
        # photon_to_event no longer returns relative event, so need to make
        # separate call to get it
        abs_event = surface_path.photon_to_event(image_event)
        rel_event = abs_event.wrt_path(origin_id)
        if rel_event.pos.vals[2] < 0.:
            return False    # event is behind the origin (camera)
        
        #now get planes of viewing frustrum
        uv_shape = self.obs.fov.uv_shape.vals
        uv = np.array([(0,0),
                       (uv_shape[0],0),
                       (uv_shape[0],uv_shape[1]),
                       (0,uv_shape[1])])
        uv_pair = Pair(uv)
        los = self.obs.fov.los_from_uv(uv_pair)
        # if the dot product of the normal of each plane and the center of
        # object is less then negative the radius, we lie entirely outside that
        # plane of the frustrum, and therefore no part is in fov.  If between
        # -r and r, we have intersection, therefore part in fov. otherwise,
        # entirely in fov.
        # for the moment use the bounding sphere, but in future use a bounding
        # box which would be more appropriate for objects not sphere-like
        min_r = -mass_body.radius
        max_r = mass_body.radius
        rel = rel_event.pos.vals
        for i in range(0,4):
            normal = los[i].cross(los[(i+1)%4]).unit()
            dist = normal.dot(rel_event.pos)
            d = dist.vals
            if d < min_r:    # entirely outside of plane
                return False
            if (d < max_r) and (-d > min_r):  # intersects plane
                return True
        # must be inside of all the planes
        return True


    def point_in_frustrum(self, frustrum_normals, body, pos, obs_bodies,
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
                    obs_bodies.append(body)
                    body_positions.append(pos.vals)
                    return
                elif positive_side[j] and positive_side[k]:
                    obs_bodies.append(body)
                    body_positions.append(pos.vals)
                    return
                intersects[i] = True
            else:
                positive_side[i] = True
        # if we have reached here, might be inside frustrum
        if np.all(positive_side):
            obs_bodies.append(body)
            body_positions.append(pos.vals)

    def where_not_blocked(self, origin_id, main_planet, error_buffer_size,
                          grid_resolution=1):
        """Determine if any part of each of the bodies is w/in field-of-view and
            not blocked by main planet nor target.
            
            Input:
            origin_id   id of camera
            
            Return:     list of bodies in snapshot.
            """
        # get image event for this snapshot
        image_event = Event(self.obs.midtime, (0,0,0), (0,0,0), origin_id,
                            self.obs.frame_id)
        
        # get the planes of the viewing frustrum
        uv_shape = self.obs.fov.uv_shape.vals
        uv = np.array([(0,0),
                       (uv_shape[0],0),
                       (uv_shape[0],uv_shape[1]),
                       (0,uv_shape[1])])
        uv_pair = Pair(uv)
        los = self.obs.fov.los_from_uv(uv_pair)
        # scale los to avoid accuracy problems
        los *= 1000000.

        # get some pixel maximums
        max_x_pixel = uv_shape[0] + error_buffer_size - 1
        max_y_pixel = uv_shape[1] + error_buffer_size - 1
        
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
            frustrum_normals.append(los[i].cross(los[(i+1)%4]).unit())
        
        limit = self.obs.fov.uv_shape + oops.Pair(np.array([error_buffer_size,
                                                            error_buffer_size]))

        meshgrid = Meshgrid.for_fov(self.obs.fov, undersample=grid_resolution,
                                    limit=limit, swap=True)
        bp = oops.Backplane(self.obs, meshgrid)
        try:
            saturn_distance = bp.distance(main_planet)
        except:
            return None
        target = self.obs.index_dict["TARGET_NAME"].strip().lower()

        potential_bodies = []
        body_positions = []
        for mass_body in self.bodies:
            try:
                path = path_.Path.connect(mass_body.path_id, origin_id,
                                          self.obs.frame_id)
                abs_event = path.photon_to_event(image_event)
                rel_event = abs_event.wrt_path(origin_id)
            except:
                continue
            # first check that body is in front of camera
            if rel_event.pos.vals[2] < 0.:
                continue;
            self.point_in_frustrum(frustrum_normals, mass_body, rel_event.pos,
                                   potential_bodies, body_positions)        
        
        if len(potential_bodies) == 1 and potential_bodies[0] == main_planet:
            confirmed_bodies.append(main_planet)
        else:
            
            confirmed_bodies = []
            if target != main_planet and target != "sky":
                t_distance = bp.distance(target)
                if saturn_distance.shape == [] or saturn_distance.shape == ():
                    confirmed_bodies.append(target)
                else:
                    if np.less(t_distance.vals, saturn_distance.vals).any():
                        # target is closer than saturn somehwere
                        confirmed_bodies.append(target)
                    if np.less(saturn_distance.vals, t_distance.vals).any():
                        # saturn is closwer than target somewhere
                        confirmed_bodies.append(main_planet)
                occ_distance = np.minimum(saturn_distance.vals, t_distance.vals)
                target_saturn_distance = np.ma.array(occ_distance,
                                                     mask=saturn_distance.mask & t_distance.mask)
            else:
                if saturn_distance.shape != []:
                    confirmed_bodies.append(main_planet)
                target_saturn_distance = np.ma.array(saturn_distance.vals,
                                                     mask=saturn_distance.mask)
                #target_saturn_distance = saturn_distance.vals
            
            body_number = 0
            for potential_body in potential_bodies:
                p_body = potential_body.name.lower()
                if (p_body != main_planet) and (p_body != target):
                    if target_saturn_distance.shape == [] or target_saturn_distance.shape == ():
                        confirmed_bodies.append(p_body)
                    else:
                        pos = body_positions[body_number]
                        pix = self.obs.fov.uv_from_los(pos)
                        # now, if the center of the body is off the FOV, then
                        # the closest point is still going to be part of the
                        # same body (if that body is at least partially in the
                        # FOV, which we know it is).
                        if pix.vals[0] < 0:
                            pix.vals[0] = 0
                        elif pix.vals[0] >= max_x_pixel:
                            pix.vals[0] = max_x_pixel
                        if pix.vals[1] < 0:
                            pix.vals[1] = 0
                        elif pix.vals[1] >= max_y_pixel:
                            pix.vals[1] = max_y_pixel
                        ix = int(pix.vals[0])
                        iy = int(pix.vals[1])
                        
                        if pos[2] < target_saturn_distance[ix][iy]:
                            confirmed_bodies.append(p_body)
                        elif target_saturn_distance.mask[ix][iy]:
                            confirmed_bodies.append(p_body)
                        else:
                            # most time consuming test... test ring of points,
                            # but don't bother if there are no true mask values
                            if np.any(target_saturn_distance.mask):
                                # get radius, in pixels, at the distance of the
                                # body in the FOV
                                radius_los = np.array([potential_body.radius,
                                                       potential_body.radius,
                                                       pos[2]])
                                # we only need to check perpendicular directions
                                # since if they are blocked, all points in
                                # between are blocked.
                                pix_radius = self.obs.fov.uv_from_los(radius_los)
                                ix1 = min(int(ix + pix_radius.vals[0]), max_x_pixel)
                                ix2 = max(int(ix - pix_radius.vals[0]), 0)
                                iy1 = min(int(iy + pix_radius.vals[1]), max_y_pixel)
                                iy2 = max(int(iy - pix_radius.vals[1]), 0)
                                if target_saturn_distance.mask[ix1][iy] or target_saturn_distance.mask[ix2][iy] or target_saturn_distance.mask[ix][iy1] or target_saturn_distance.mask[ix][iy2]:
                                    confirmed_bodies.append(p_body)
                
                body_number += 1

        return confirmed_bodies
