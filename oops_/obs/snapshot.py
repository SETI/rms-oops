################################################################################
# oops_/obs/snapshot.py: Subclass Snapshot of class Observation
################################################################################

import numpy as np
import sys
import math

from oops_.obs.observation_ import Observation
from oops_.array.all import *
from oops_.config import QUICK

import oops_.frame.all as frame_
import oops_.path.all  as path_
import oops_.surface.all as surface_
from oops_.event import Event

from oops import inst

dist_tolerance = 1000.

class Snapshot(Observation):
    """A Snapshot is an Observation consisting of a 2-D image made up of pixels
    all exposed at the same time."""

    ZERO_PAIR = Pair((0,0))

    def __init__(self, time, fov, path_id, frame_id, **subfields):
        """Constructor for a Snapshot.

        Input:
            time        a tuple defining the start time and end time of the
                        observation overall, in seconds TDB.
            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y).
            path_id     the registered ID of a path co-located with the
                        instrument.
            frame_id    the registered ID of a coordinate frame fixed to the
                        optics of the instrument. This frame should have its
                        Z-axis pointing outward near the center of the line of
                        sight, with the X-axis pointing rightward and the y-axis
                        pointing downward.
            subfields   a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.
        """

        self.time = time
        self.fov = fov
        self.path_id = path_id
        self.frame_id = frame_id

        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        # Attributes specific to a Snapshot
        self.midtime = (self.time[0] + self.time[1]) / 2.
        self.texp = self.time[1] - self.time[0]

    def times_at_uv(self, uv_pair, extras=()):
        """Returns the start time and stop time associated with the selected
        spatial pixel index (u,v).

        Input:
            uv_pair     a Pair of spatial indices (u,v).
            extras      Scalars of any extra index values needed to define the
                        timing of array elements.
        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.

                        If derivs is True, then each time has a subfield "d_duv"
                        defining the change in time associated with a 1-pixel
                        step along the u and v axes. This is represented by a
                        MatrixN with item shape [1,2].
        """

        return self.time

    def sweep_duv_dt(self, uv_pair, extras=()):
        """Returns the mean local sweep speed of the instrument in the (u,v)
        directions.

        Input:
            uv_pair     a Pair of spatial indices (u,v).
            extras      Scalars of any extra index values needed to define the
                        timing of array elements.

        Return:         a Pair containing the local sweep speed in units of
                        pixels per second in the (u,v) directions.
        """

        return Snapshot.ZERO_PAIR

    def uv_from_path(self, path, quick=QUICK, derivs=False):
        """Solves for the (u,v) indices of an object in the field of view, given
        its path.

        Input:
            path        a Path object.
            quick       defines how to use QuickPaths and QuickFrames.
            derivs      True to include derivatives.

        Return:
            uv_pair     the (u,v) indices of the pixel in which the point was
                        found. The path is evaluated at the mid-time of this
                        pixel.

        For paths that fall outside the field of view, the returned values of
        time and index are masked.
        """

        # Snapshots are easy and require zero iterations
        return Observation.uv_from_path(self, path, (), quick, derivs, iters=0)
    
    def surface_in_view(self, surface):
        """Determine whether a surface exists in the snapshot.
            
            Input:
            surface     A surface (RingPlane, Spheroid, etc) to be checked
            
            Return:     Boolean.  True if exists, else False.
            """
        return self.recursive_surface_in_view(surface, 16.)
    
    def recursive_surface_in_view(self, surface, resolution):
        """Recursively check, with finer and finer resolution, whether surface
            is intersected by ray at center of a pixel.
            
            Input:
            surface     A surface (RingPlane, Spheroid, etc) to be checked.
            resoltuion  must be power of 2.
            
            Return:     Boolean.  True if exists, else False.
            """
        uv_shape = self.fov.uv_shape.vals / resolution
        buffer = np.empty((uv_shape[0], uv_shape[1], 2))
        buffer[:,:,1] = np.arange(uv_shape[1]).reshape(uv_shape[1],1)
        buffer[:,:,0] = np.arange(uv_shape[0])
        buffer *= resolution
        indices = Pair(buffer + 0.5)
        
        rays = self.fov.los_from_uv(indices)
        arrivals = -rays
        image_event = Event(self.midtime, (0,0,0), (0,0,0), "CASSINI",
                            self.frame_id, Empty(), arrivals)
        
        (surface_event, rel_surf_evt) = surface.photon_to_event(image_event, 1)
        
        if not surface_event.pos.mask.all():
            return True
        elif resolution == 1:
            return False
        return self.recursive_surface_in_view(surface, resolution / 2.)
    
    def surface_center_within_view_bounds(self, origin_id, path_id):
        """Determine whether teh center of a surface is somewhere within the
            snapshot bounds, even if subpixel.
            
            Input:
            surface     A surface (RingPlane, Spheroid, etc) to be checked
            
            Return:     Boolean.  True if exists, else False.
            """
        image_event = Event(self.midtime, (0,0,0), (0,0,0), origin_id, self.frame_id)
        surface_path = path_.connect(path_id, origin_id, self.frame_id)
        (abs_event, rel_event) = surface_path.photon_to_event(image_event)
        xy_pair = self.fov.xy_from_los(rel_event.pos)
        uv = self.fov.uv_from_xy(xy_pair)
        u = uv.vals[...,0]
        v = uv.vals[...,1]
        uv_shape = self.fov.uv_shape.vals
        return((u >= 0.) and (u <= uv_shape[0]) and (v >= 0.) and (v <= uv_shape[1]))
    
    def any_part_object_in_view(self, origin_id, mass_body):
        """Determine if any part of object from path_id is w/in field-of-view.
            NOTE: This code is for a sphere and should be put in Spheroid,
            probably.
            
            Input:
            path_id     id of object path
            
            Return:     Boolean.  True if any part w/in fov.
            """
        # get position of object relative to view point
        image_event = Event(self.midtime, (0,0,0), (0,0,0), origin_id, self.frame_id)
        surface_path = path_.Path.connect(mass_body.path_id, origin_id,
                                          self.frame_id)
        #rel_event = surface_path.photon_to_event(image_event)[1]
        # photon_to_event no longer returns relative event, so need to make
        # separate call to get it
        abs_event = surface_path.photon_to_event(image_event)
        rel_event = abs_event.wrt_path(origin_id)
        
        #now get planes of viewing frustrum
        uv_shape = self.fov.uv_shape.vals
        uv = np.array([(0,0),
                       (uv_shape[0],0),
                       (uv_shape[0],uv_shape[1]),
                       (0,uv_shape[1])])
        uv_pair = Pair(uv)
        los = self.fov.los_from_uv(uv_pair)
        # if the dot product of the normal of each plane and the center of
        # object is less then negative the radius, we lie entirely outside that
        # plane of the frustrum, and therefore no part is in fov.  If between
        # -r and r, we have intersection, therefore part in fov. otherwise,
        # entirely in fov.
        #frustrum_planes = []
        # for the moment use the bounding sphere, but in future use a bounding
        # box which would be more appropriate for objects not sphere-like
        min_r = -mass_body.radius
        max_r = mass_body.radius
        for i in range(0,4):
            normal = los[i].cross(los[(i+1)%4])
            dist = normal.dot(rel_event.pos)
            d = dist.vals
            if d < min_r:    # entirely outside of plane
                return False
            if (d < max_r) and (-d > min_r):  # intersects plane
                return True
        # must be inside of all the planes
        return True

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Snapshot(unittest.TestCase):

    def runTest(self):
        
        # Imports are here to avoid conflicts
        import oops_.registry as registry
        import oops.inst.cassini.iss as iss
        
        #registry.initialize_frame_registry()
        #registry.initialize_path_registry()
        
        print '\n'
        paths = ["test_data/cassini/ISS/W1575634136_1.IMG",
                 "test_data/cassini/ISS/W1573721822_1.IMG",
                 "test_data/cassini/ISS/N1649465367_1.IMG",
                 "test_data/cassini/ISS/N1649465412_1.IMG",
                 "test_data/cassini/ISS/N1649465464_1.IMG",
                 "test_data/cassini/ISS/W1572114418_1.IMG",
                 "test_data/cassini/ISS/W1575632515_1.IMG",
                 "test_data/cassini/ISS/W1575633938_1.IMG",
                 "test_data/cassini/ISS/W1575633971_1.IMG",
                 "test_data/cassini/ISS/W1575634037_1.IMG",
                 "test_data/cassini/ISS/W1575634070_1.IMG",
                 "test_data/cassini/ISS/W1575634103_1.IMG",
                 "test_data/cassini/ISS/W1558932373_3.IMG",
                 "test_data/cassini/ISS/N1649465323_1.IMG"]
        saturn_solns = [True, True, True, True, True, True, True, True, True,
                        True,  True, True, True, True]
        mimas_solns = [False, False, False, False, False, True, False, False,
                       False, False, False, False, True, False]
        enceladus_solns = [True, False, True, True, True, False, False, True,
                           True, True, True, True, False, True]
        tethys_solns = [False, False, False, False, False, False, True, False,
                        False, False, False, False, False, False]
        dione_solns = [True, False, False, False, False, False, False, True,
                       True, True, True, True, False, False]
        rhea_solns = [False, False, True, True, True, False, False, False,
                      False, False, False, False, False, True]
        titan_solns = [False, False, False, False, False, True, False, False,
                       False, False, False, False, False, False]
        iapetus_solns = [False, False, False, False, False, False, False, False,
                         False, False, False, False, False, False]
        phoebe_solns = [False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False]
        hyperion_solns = [True, False, False, False, False, False, True, True,
                          True, True, True, True, False, False]
        solns = [saturn_solns, mimas_solns, enceladus_solns, tethys_solns,
                 dione_solns, rhea_solns, titan_solns, iapetus_solns,
                 phoebe_solns, hyperion_solns]
        bodies = [registry.body_lookup("SATURN"),
                  registry.body_lookup("MIMAS"),
                  registry.body_lookup("ENCELADUS"),
                  registry.body_lookup("TETHYS"),
                  registry.body_lookup("DIONE"),
                  registry.body_lookup("RHEA"),
                  registry.body_lookup("TITAN"),
                  registry.body_lookup("IAPETUS"),
                  registry.body_lookup("PHOEBE"),
                  registry.body_lookup("HYPERION")]
        i = 0
        #check_name = "HYPERION"
        #print "Any part of %s in view: " % check_name
        for path in paths:
            snapshot = iss.from_file(path)
            j = 0
            for mass_body in bodies:
                in_view = snapshot.any_part_object_in_view("CASSINI",
                                                           mass_body)
                self.assertTrue(in_view == solns[j][i])
                j += 1
            i += 1

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
