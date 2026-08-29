##########################################################################################
# oops/observation/pixel.py: Subclass Pixel of class Observation
##########################################################################################

import numpy as np

from polymath         import Scalar, Pair, Vector3
from oops.observation import Observation
from oops.event       import Event
from oops.frame       import Frame
from oops.path        import Path


class Pixel(Observation):
    """A subclass of Observation consisting of one or more measurements obtained from a
    single rectangular pixel.

    Generalization to other FOV shapes is TODO.
    """

    def __init__(self, axes, cadence, fov, path, frame, **subfields):
        """Constructor for a Pixel observation.
        """

        # Basic properties
        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        # FOV
        self.fov = fov
        if self.fov.uv_shape != (1,1):
          raise ValueError('Pixel observation FOV must have shape (1,1)')

        self.uv_shape = (1,1)

        # Axes
        self.axes = list(axes)
        self.u_axis = -1
        self.v_axis = -1
        self.swap_uv = False
        if 't' in self.axes:
            self.t_axis = self.axes.index('t')
        else:
            self.t_axis = -1

        # Cadence
        self.cadence = cadence
        if len(self.cadence.shape) != 1:
            raise ValueError('Pixel observation requires a 1-D cadence')

        samples = self.cadence.shape[0]

        # Shape / Size
        shape_list = len(axes) * [0]
        if self.t_axis >= 0:
            shape_list[self.t_axis] = samples
        self.shape = tuple(shape_list)

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

    def __getstate__(self):
        self.refresh()
        return (self.axes, self.cadence, self.fov, self.path, self.frame,
                self.subfields)

    def __setstate__(self, state):
        self.__init__(*state[:-1], **state[-1])
        self.freeze()

    def uvt(self, indices, remask=False, derivs=True):
        """Coordinates (u,v) and time t for indices into the data array.

        This method supports non-integer index values.

        Parameters:
            indices (Scalar): Or Vector of array indices.
            remask (bool, optional): True to mask values outside the field of view.
            derivs (bool, optional): True to include derivatives in the returned values.

        Returns:
            (tuple): (uv, time), where:

            * `uv` (Pair): Defining the values of (u,v) within the FOV that are associated
              with the array indices.
            * `time` (Scalar): Defining the time in seconds TDB associated with the array
              indices.
        """

        # Works for a 1-D index or a multi-D index
        tstep = Observation.scalar_from_indices(indices, self.t_axis,
                                                derivs=derivs)

        if tstep is None:       # if t_axis < 0
            uv = Pair.filled(indices.shape, 0.5)
            return (uv, Scalar(self.cadence.midtime))

        time = self.cadence.time_at_tstep(tstep, remask=remask)
        uv = Pair.filled(time.shape, 0.5, mask=time.mask)
        return (uv, time)

    def uvt_range(self, indices, remask=False):
        """Ranges of (u,v) spatial coordinates and time for integer array indices.

        Parameters:
            indices (Vector): A Vector of array indices.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            (tuple): (uv_min, uv_max, time_min, time_max), where:

            * `uv_min` (Pair): Defining the minimum values of (u,v) associated the pixel.
            * `uv_max` (Pair): Defining the maximum values of (u,v).
            * `time_min` (Scalar): Defining the minimum time associated with the pixel. It
              is given in seconds TDB.
            * `time_max` (Scalar): Defining the maximum time value.
        """

        if self.t_axis < 0:
            return (Pair.INT00, Pair(self.fov.uv_shape),
                    Scalar(self.cadence.time[0]), Scalar(self.cadence.time[1]))

        # Works for a 1-D index or a multi-D index
        tstep = Observation.scalar_from_indices(indices, self.t_axis)
        (time_min,
         time_max) = self.cadence.time_range_at_tstep(tstep, remask=remask)

        # uv pair
        uv_min = Pair.zeros(indices.shape, dtype='int', mask=time_min.mask)

        return (uv_min, uv_min + self.fov.uv_shape, time_min, time_max)

    def time_range_at_uv(self, uv_pair, remask=False):
        """The start and stop times of the specified spatial pixel (u,v).

        The Pixel observation subclass has no spatial axes, so the inputs here are
        generally ignored, although they are expected to fall between 0 and 1 inclusive.

        Parameters:
            uv_pair (Pair): Spatial (u,v) data array coordinates, truncated to integers if
                necessary.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            (tuple): Scalars of the start time and stop time of each (u,v) pair, as
                seconds TDB.
        """

        return self.time_range_at_uv_0d(uv_pair, remask=remask)

    def uv_range_at_time(self, time, remask=False):
        """The (u,v) range of spatial pixels observed at the specified time.

        For the Pixel observation subclass, the (u,v) ranges are always (0,1). The time is
        largely ignored, although it is expected to fall within the time limits of the
        observation and will be masked if remask == True.

        Parameters:
            time (Scalar): Time values in seconds TDB.
            remask (bool, optional): True to mask values outside the time limits.

        Returns:
            (tuple): (uv_min, uv_max), where:

            * `uv_min` (Pair): The lower (u,v) corner Pair of the area observed at the
              specified time.
            * `uv_max` (Pair): The upper (u,v) corner Pair of the area observed at the
              specified time.
        """

        return Observation.uv_range_at_time_0d(self, time,
                                               uv_shape=self.uv_shape,
                                               remask=remask)

    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Parameters:
            dtime (float): The time offset to apply to the observation, in units of
                seconds. A positive value shifts the observation later.

        Returns:
            A (shallow) copy of the object with a new time.
        """

        obs = Pixel(axes=self.axes, cadence=self.cadence.time_shift(dtime),
                    fov=self.fov, path=self.path, frame=self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

    ######################################################################################
    # Overrides of Observation class methods
    ######################################################################################

    def event_at_grid(self, meshgrid, tfrac=0.5, time=None):
        """An event object describing the arrival of a photon at a set of locations
        defined by the given meshgrid. This version overrides the default definition to
        apply the timing for each pixel of a time-sequence by default.

        Parameters:
            meshgrid (Meshgrid): Object describing the sampling of the field of view.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where tfrac=0 at the beginning and 1 at the end. Default is 0.5.
            time (Scalar, optional): Optional Scalar of absolute time in seconds. Only one
                of tfrac and time can be specified.

        Returns:
            The corresponding event.
        """

        if time is None:
            tstep = np.arange(self.cadence.shape[0]) + tfrac
            time = self.cadence.time_at_tstep(tstep)
            time = time.append_axes(len(meshgrid.shape))

        event = Event(time, Vector3.ZERO, self.path, self.frame)

        # Insert the arrival directions
        event.neg_arr_ap = meshgrid.los

        return event

    def gridless_event(self, meshgrid, tfrac=0.5, time=None,
                             shapeless=False):
        """An event object describing the arrival of a photon at a set of locations
        defined by the given meshgrid. This version overrides the default definition to
        apply the timing for each pixel of a time-sequence by default.

        Parameters:
            meshgrid (Meshgrid): Object describing the sampling of the field of view.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where tfrac=0 at the beginning and 1 at the end. Default is 0.5.
            time (Scalar, optional): Optional Scalar of absolute time in seconds. Only one
                of tfrac and time can be specified; the other must be None.
            shapeless (bool, optional): True to return a shapeless event, referring to the
                mean of all the times.

        Returns:
            The corresponding event.
        """

        if tfrac is not None:
            if time is not None:
                raise ValueError('tfrac and time cannot both be defined')

            tstep = np.arange(self.cadence.shape[0]) + tfrac
            time = self.cadence.time_at_tstep(tstep)
            time = time.append_axes(len(meshgrid.shape))

        if shapeless:
            time = time.mean()

        event = Event(time, Vector3.ZERO, self.path, self.frame)

        return event

##########################################################################################
