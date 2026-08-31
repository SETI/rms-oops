##########################################################################################
# oops/observation/pixel.py
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

        Parameters:
            axes (list or tuple): Strings, with one value for each axis in the associated
                data array. A value of 't' should appear at the location of the array's
                time axis, if any.
            cadence (Cadence): A 1-D Cadence object defining the start time and duration
                of each consecutive measurement.
            fov (FOV): (field-of-view) object, which describes the field of view including
                any spatial distortion. It maps between spatial coordinates (u,v) and
                instrument coordinates (x,y). For a Pixel object, the FOV must have shape
                (1,1).
            path (Path): The path waypoint co-located with the instrument.
            frame (Frame): The wayframe of a coordinate frame fixed to the optics of the
                instrument. This frame should have its Z-axis pointing outward near the
                center of the line of sight, with the X-axis pointing rightward and the
                Y-axis pointing downward.
            subfields (dict): All of the optional attributes. Additional subfields may be
                included as needed.

        Raises:
            ValueError: If the FOV does not have shape (1,1) or if the cadence is not 1-D.
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
        self._axes = list(axes)
        self.u_axis = -1
        self.v_axis = -1
        self.swap_uv = False
        if 't' in self._axes:
            self.t_axis = self._axes.index('t')
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
        return (self._axes, self.cadence, self.fov, self.path, self.frame,
                self.subfields)

    def __setstate__(self, state):
        self.__init__(*state[:-1], **state[-1])
        self.freeze()

    def uvt(self, indices, *, remask=False, derivs=True):
        """Coordinates `(u,v)` and time `t` for indices into the data array.

        This method supports non-integer index values.

        Parameters:
            indices (Scalar or Vector): Array indices.
            remask (bool, optional): True to mask values outside the field of view.
            derivs (bool, optional): True to include derivatives in the returned values.

        Returns:
            tuple[Pair, Scalar]: `(uv, time)`, where `uv` defines the values of `(u,v)`
            within the FOV that are associated with the array indices and `time` defines
            the time in seconds TDB associated with the array indices.
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

    def uvt_range(self, indices, *, remask=False):
        """Ranges of `(u,v)` spatial coordinates and time for integer array indices.

        Parameters:
            indices (Scalar or Vector): Array indices.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Pair, Pair, Scalar, Scalar]: `(uv_min, uv_max, time_min, time_max)`,
            where:

            * `uv_min`: The minimum values of (u,v) associated with the pixel.
            * `uv_max`: The maximum values of (u,v).
            * `time_min`: The minimum time associated with the pixel, in seconds TDB.
            * `time_max`: The maximum time value.
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

    def time_range_at_uv(self, uv_pair, *, remask=False):
        """The start and stop times of the specified spatial pixel `(u,v)`.

        A Pixel observation has no spatial axes, so the input is largely ignored, although
        it is expected to fall between 0 and 1 inclusive.

        Parameters:
            uv_pair (Pair): Spatial (u,v) data array coordinates, truncated to integers if
                necessary.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Scalar, Scalar]: Scalars of the start time and stop time of each `(u,v)`
            pair, as seconds TDB.
        """

        return self.time_range_at_uv_0d(uv_pair, remask=remask)

    def uv_range_at_time(self, time, *, remask=False):
        """The `(u,v)` range of spatial pixels in the data array observed at the specified
        time.

        For a Pixel observation, the `(u,v)` range is always (0,0) to (1,1). The time is
        largely ignored, although it is expected to fall within the time limits of the
        observation and is masked if `remask` is True.

        Parameters:
            time (Scalar): Time values in seconds TDB.
            remask (bool, optional): True to mask values outside the time limits.

        Returns:
            tuple[Pair, Pair]: `(uv_min, uv_max)`, where `uv_min` is the lower corner of
            the `(u,v)` rectangle observed and `uv_max` is the upper corner.
        """

        return Observation.uv_range_at_time_0d(self, time, uv_shape=self.uv_shape,
                                               remask=remask)

    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Parameters:
            dtime (float): The time offset to apply to the observation, in units of
                seconds. A positive value shifts the observation later.

        Returns:
            Observation: A (shallow) copy of the object with a new time.
        """

        obs = Pixel(axes=self._axes, cadence=self.cadence.time_shift(dtime),
                    fov=self.fov, path=self.path, frame=self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

    ######################################################################################
    # Overrides of Observation class methods
    ######################################################################################

    def event_at_grid(self, meshgrid=None, *, tfrac=0.5, time=None):
        """A photon arrival event from directions defined by a meshgrid.

        This overrides the default definition to apply the timing of each sample of the
        time sequence by default.

        Parameters:
            meshgrid (Meshgrid, optional): Object describing the sampling of the field of
                view.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where tfrac=0 at the beginning and 1 at the end. Default is 0.5. Ignored
                if `time` is specified.
            time (Scalar, optional): Optional Scalar of absolute time in seconds.

        Returns:
            Event: The corresponding Event.
        """

        if time is None:
            tstep = np.arange(self.cadence.shape[0]) + tfrac
            time = self.cadence.time_at_tstep(tstep)
            time = time.reshape(time.shape + len(meshgrid.shape) * (1,))

        event = Event(time, Vector3.ZERO, self.path, self.frame)

        # Insert the arrival directions
        event.neg_arr_ap = meshgrid.los(time)

        return event

    def gridless_event(self, meshgrid=None, *, tfrac=0.5, time=None, shapeless=False):
        """A photon arrival event irrespective of the direction.

        This overrides the default definition to apply the timing of each sample of the
        time sequence by default.

        Parameters:
            meshgrid (Meshgrid, optional): Object describing the sampling of the field of
                view. Here, it is only used to define the shape of the returned event.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where tfrac=0 at the beginning and 1 at the end. Default is 0.5.
            time (Scalar, optional): Optional Scalar of absolute time in seconds. If
                specified, `tfrac` is ignored.
            shapeless (bool, optional): True to return a shapeless event, referring to the
                mean of all the times.

        Returns:
            Event: The corresponding Event.
        """

        if time is None:
            tstep = np.arange(self.cadence.shape[0]) + tfrac
            time = self.cadence.time_at_tstep(tstep)
            time = time.reshape(time.shape + len(meshgrid.shape) * (1,))

        if shapeless:
            time = time.mean()

        event = Event(time, Vector3.ZERO, self.path, self.frame)

        return event

##########################################################################################
