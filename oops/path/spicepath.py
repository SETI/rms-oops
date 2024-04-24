################################################################################
# oops/path_/spicepath.py: Subclass SpicePath of class Path
################################################################################

import numpy as np

import cspyce

from polymath          import Scalar, Vector3
from oops.event        import Event
from oops.frame.frame_ import Frame
from oops.path.path_   import Path, ReversedPath, RotatedPath
import oops.spice_support as spice

class SpicePath(Path):
    """A Path subclass that returns information based on an SPICE SP kernel.

    It represents the geometric position of a single target body with respect to
    a single origin.
    """

    # Set False to confirm that SpicePaths return the same results without
    # shortcuts and with shortcuts
    USE_SPICEPATH_SHORTCUTS = True

    #===========================================================================
    def __init__(self, spice_id, spice_origin="SSB", spice_frame="J2000",
                       path_id=None, shortcut=None, unpickled=False):
        """Constructor for a SpicePath object.

        Input:
            spice_id        the name or integer ID of the target body as used
                            in the SPICE toolkit.
            spice_origin    the name or integer ID of the origin body as
                            used in the SPICE toolkit; "SSB" for the Solar
                            System Barycenter by default. It may also be the
                            registered name of another SpicePath.
            spice_frame     the name or integer ID of the reference frame or of
                            the a body with which the frame is primarily
                            associated, as used in the SPICE toolkit.
            path_id         the name or ID under which the path will be
                            registered. By default, this will be the value of
                            spice_id if that is given as a string; otherwise
                            it will be the name as used by the SPICE toolkit.
            shortcut        If a shortcut is specified, then this is registered
                            as a shortcut definition; the other registered path
                            definitions are unchanged.
            unpickled       True if this object was read from a pickle file. If
                            so, then it will be treated as a duplicate of a
                            pre-existing SpicePath for the same SPICE ID.
        """

        # Preserve the inputs
        self.spice_id = spice_id
        self.spice_origin = spice_origin
        self.spice_frame = spice_frame
        self.shortcut = shortcut

        # Interpret the SPICE IDs
        (self.spice_target_id,
         self.spice_target_name) = spice.body_id_and_name(spice_id)

        (self.spice_origin_id,
         self.spice_origin_name) = spice.body_id_and_name(spice_origin)

        self.spice_frame_name = spice.frame_id_and_name(spice_frame)[1]

        # Fill in the Path ID and save it in the global dictionary
        if path_id is None:
            if isinstance(spice_id, str):
                self.path_id = spice_id
            else:
                self.path_id = self.spice_target_name
        else:
            self.path_id = path_id

        # Only save info in the PATH_TRANSLATION dictionary if it is not already
        # there. We do not want to overwrite original definitions with those
        # just read from pickle files.
        if not shortcut:
            if self.spice_target_id not in spice.PATH_TRANSLATION:
                spice.PATH_TRANSLATION[self.spice_target_id] = self.path_id
            if self.spice_target_name not in spice.PATH_TRANSLATION:
                spice.PATH_TRANSLATION[self.spice_target_name] = self.path_id

        # Fill in the origin waypoint, which should already be in the dictionary
        origin_id = spice.PATH_TRANSLATION[self.spice_origin_id]
        self.origin = Path.as_waypoint(origin_id)

        # Fill in the frame wayframe, which should already be in the dictionary
        frame_id = spice.FRAME_TRANSLATION[self.spice_frame_name]
        self.frame = Frame.as_wayframe(frame_id)

        # No shape, no keys
        self.shape = ()
        self.keys = set()
        self.shortcut = shortcut

        # Register the SpicePath; fill in the waypoint
        self.register(shortcut, unpickled=unpickled)

    def __getstate__(self):
        return (self.spice_target_id, self.spice_origin_id,
                self.spice_frame_name)

    def __setstate__(self, state):

        (spice_target_id, spice_origin_id, spice_frame_name) = state

        # If this is a duplicate of a pre-existing SpicePath, make sure it gets
        # assigned the pre-existing path ID and Waypoint
        path_id = spice.PATH_TRANSLATION.get(spice_target_id, None)
        self.__init__(spice_target_id, spice_origin_id, spice_frame_name,
                      path_id=path_id, unpickled=True)

    #===========================================================================
    def event_at_time(self, time, quick={}):
        """An Event corresponding to a specified Scalar time on this path.

        Input:
            time        a time Scalar at which to evaluate the path.

        Return:         an Event object containing (at least) the time, position
                        and velocity of the path.
        """

        time = Scalar.as_scalar(time).as_float()

        # A fully-masked time can be handled quickly
        if time.mask is True:
            return Event(time, Vector3.ZERO, self.origin, self.frame)

        # A single unmasked time can be handled quickly
        if time.shape == ():
            (state,
             lighttime) = cspyce.spkez(self.spice_target_id,
                                       time.vals,
                                       self.spice_frame_name,
                                       'NONE',
                                       self.spice_origin_id)

            return Event(time, (state[0:3],state[3:6]), self.origin, self.frame)

        # Use a QuickPath if warranted, possibly making a recursive call
        if isinstance(quick, dict):
            return self.quick_path(time, quick).event_at_time(time, False)

        # Fill in the states and light travel times using cspyce
        if np.any(time.mask):
            state = cspyce.spkez_vector(self.spice_target_id,
                                        time.vals[time.antimask],
                                        self.spice_frame_name,
                                        'NONE',
                                        self.spice_origin_id)[0]

            pos = np.zeros(time.shape + (3,))
            vel = np.zeros(time.shape + (3,))
            pos[time.antimask] = state[...,0:3]
            vel[time.antimask] = state[...,3:6]

        else:
            state = cspyce.spkez_vector(self.spice_target_id,
                                        time.vals.ravel(),
                                        self.spice_frame_name,
                                        'NONE',
                                        self.spice_origin_id)[0]
            pos = state[:,0:3].reshape(time.shape + (3,))
            vel = state[:,3:6].reshape(time.shape + (3,))

        # Convert to an Event and return
        return Event(time, (pos,vel), self.origin, self.frame)

    #===========================================================================
    def wrt(self, origin, frame=None):
        """Construct a path pointing from an origin to this target in any frame.

        SpicePath overrides the default method to create quicker "shortcuts"
        between SpicePaths.

        Input:
            origin      an origin Path object or its registered name.
            frame       a frame object or its registered ID. Default is to use
                        the frame of the origin's path.
        """

        # Use the slow method if necessary, for debugging
        if not SpicePath.USE_SPICEPATH_SHORTCUTS:
            return Path.wrt(self, origin, frame)

        # Interpret the origin path
        origin = Path.as_primary_path(origin)
        if origin in (Path.SSB, None):
            spice_origin_id = 0
        elif isinstance(origin, SpicePath):
            spice_origin_id = origin.spice_target_id
        else:
            # If the origin is not a SpicePath, seek from the other direction
            return ReversedPath(origin.wrt(self, frame))

        origin_id = spice.PATH_TRANSLATION[spice_origin_id]

        # Interpret the frame
        frame = Frame.as_primary_frame(frame)
        if frame in (Frame.J2000, None):
            spice_frame_name = 'J2000'
            uses_spiceframe = True
        elif type(frame).__name__ == 'SpiceFrame':  # avoids a circular load
            spice_frame_name = frame.spice_frame_name
            uses_spiceframe = True
        else:
            uses_spiceframe = False     # not a SpiceFrame
            spice_frame_name = 'J2000'

        if uses_spiceframe:
            frame_id = spice.FRAME_TRANSLATION[spice_frame_name]
        else:
            frame_id = 'J2000'

        shortcut = ('SPICE_SHORTCUT[' + str(self.path_id) + ',' +
                                        str(origin_id)    + ',' +
                                        str(frame_id)     + ']')

        result = SpicePath(self.spice_target_id, spice_origin_id,
                           spice_frame_name, self.path_id, shortcut)

        # If the path uses a non-spice frame, add a rotated version
        if not uses_spiceframe:
            shortcut = ('SHORTCUT_' + str(self.path_id) + '_' +
                                      str(origin_id)    + '_' +
                                      str(frame.frame_id))
            result = RotatedPath(result, frame)
            result.register(shortcut)

        return result

################################################################################
