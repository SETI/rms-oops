##########################################################################################
# oops/frame/spicetype1frame.py
##########################################################################################

import numpy as np
import cspyce

from polymath              import Matrix3, Scalar, Vector3
from oops.cache            import Cache
from oops.frame            import Frame, J2000Frame, LinkedFrame
from oops.frame.spiceframe import SpiceFrame
from oops.transform        import Transform


class SpiceType1Frame(SpiceFrame):
    """A Frame subclass defined within the SPICE toolkit as a Type 1 (discrete) C kernel.
    """

    _FRAME_LOOKUP = {}  # (name, reference name, tick_tolerance, cache_size) -> SpiceFrame

    def __init__(self, spice_frame, tick_tolerance, reference=None, *, frame_id=None,
                 cache_size=100):
        """Constructor for a SpiceType1Frame.

        Parameters:
            spice_frame (str or int): The name, frame code, or frame name as used in the
                SPICE toolkit.
            tick_tolerance (float, int, or str): A number or string defining the time
                tolerance in spacecraft clock ticks for the Frame returned.
            reference (SpiceFrame or str, optional): The Frame or ID of the Frame relative
                to which this frame is defined. This must be a SpiceFrame or else, by
                default, J2000.
            frame_id (str, optional): The ID under which to register this Frame. If not
                specified, the name as defined in the SPICE Toolkit is used. Note that
                SpiceType1Frames are always registered.
            cache_size (int, optional): The number of transforms to cache. This can be
                useful because it avoids unnecessary SPICE calls when the Frame is being
                used repeatedly at a finite set of times.

        Raises:
            LookupError: If `spice_frame` is not a recognized frame name or frame code
                within the SPICE Toolkit.
            ValueError: If `reference` is not a SpiceFrame or J2000.
        """

        self._fill_spice_info(spice_frame, reference)

        # Fill in the time tolerance
        if isinstance(tick_tolerance, str):
            self._tick_tolerance = cspyce.sctiks(self._spice_origin_code, tick_tolerance)
        else:
            self._tick_tolerance = tick_tolerance

        self._time_tolerance = None             # filled in on first use

        self._cache_size = cache_size or 100

        # If the reference is not J2000, construct the primary version first
        if self._reference != Frame.J2000:
            wrt_j2000 = SpiceType1Frame.get(self._spice_frame_name, self._tick_tolerance,
                                            Frame.J2000, frame_id=frame_id,
                                            cache_size=self._cache_size)
            # Cache but don't register under this frame ID
            self._register(frame_id=None)
            self._wayframe = wrt_j2000._wayframe
            self._frame_id = wrt_j2000._frame_id
        else:
            # If the reference is J2000, register as normal
            _ = SpiceFrame._FOR_NAME.setdefault(self._spice_frame_name, self)
            self._register(frame_id or self._spice_frame_name.replace(' ', '_'))

        self._refresh()

        # Save for use by get()
        for key_cache_size in (self._cache_size, None):
            key = (self._spice_frame_name, self._spice_reference_name,
                   self._tick_tolerance, key_cache_size)
            _ = SpiceType1Frame._FRAME_LOOKUP.setdefault(key, self)

    def _refresh(self):
        self._cache = Cache(self._cache_size)   # saves result for multiple single times

        self._cached_transform = None           # saves result for one shaped time
        self._cached_time = None
        self._cached_shape = None

        if hasattr(self, '_quickframes'):
            self._quickframes.clear()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        return (self._spice_frame_name, self._reference, self._tick_tolerance,
                self.stripped_id, self._cache_size, self._get_quickframes())

    def __setstate__(self, state):
        (frame_name, reference, tick_tolerance, frame_id, cache_size, quickframes) = state
        self.__init__(frame_name, tick_tolerance, reference, frame_id=frame_id,
                      cache_size=cache_size)
        if quickframes:
            self._quickframes = quickframes

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, *, quick=None):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Transform: Rotates vectors from the reference frame to this frame at the
            specified time.

        Raises:
            OSError: If any individual time is out of range for the currently furnished C
                kernels. The SPICE Toolkit reports this as SPICE(CKINSUFFDATA).
        """

        # Fill in the time tolerance in seconds
        if self._time_tolerance is None:
            time = Scalar.as_scalar(time)
            # One representative time; sce2c takes a single value, and the tick rate does
            # not vary appreciably across the times of one call
            sample = float(np.ravel(time.vals)[0])
            ticks = cspyce.sce2c(self._spice_origin_code, sample)
            ticks_per_sec = cspyce.sce2c(self._spice_origin_code, sample + 1.) - ticks
            self._time_tolerance = self._tick_tolerance / ticks_per_sec

        # A single input time can be handled quickly
        time = Scalar.as_scalar(time)
        if time.shape == ():
            # Check cache first
            xform = self._cache[time.vals]
            if xform:
                return xform

            ticks = cspyce.sce2c(self._spice_origin_code, time.vals)
            (matrix3, true_ticks) = cspyce.ckgp(self._spice_frame_code, ticks,
                                                self._tick_tolerance,
                                                self._spice_reference_name)
            xform = Transform(matrix3, Vector3.ZERO, self, self._reference)
            self._cache[time.vals] = xform
            return xform

        # Check to see if the latest shaped transform is adequate
        if np.shape(time.vals) == self._cached_shape:
            diff = np.abs(time.vals - self._cached_time)
            if np.all(diff < self._time_tolerance):
                return self._cached_transform

        # If all the times are close, we can return more quickly
        time_min = time.vals.min()
        time_max = time.vals.max()
        if (time_max - time_min) < self._time_tolerance:
            tick = cspyce.sce2c(self._spice_origin_code, (time_min + time_max)/2.)
            (matrix3, true_tick) = cspyce.ckgp(self._spice_frame_code, tick,
                                               self._tick_tolerance,
                                               self._spice_reference_name)
            true_time = cspyce.sct2e(self._spice_origin_code, true_tick)

            self._cached_shape = time.shape
            self._cached_time = true_time
            self._cached_transform = Transform(matrix3, Vector3.ZERO, self,
                                               self._reference)
            return self._cached_transform

        # Otherwise, process the array...
        ticks = cspyce.sce2c_vector(self._spice_origin_code, time.vals.ravel())
        matrix3, true_ticks = cspyce.ckgp_vector(self._spice_frame_code, ticks,
                                                 self._tick_tolerance,
                                                 self._spice_reference_name)
        matrix3 = Matrix3.as_matrix3(matrix3).reshape(time.shape)
        true_times = cspyce.sct2e_vector(self._spice_origin_code, true_ticks)

        self._cached_shape = time.shape
        self._cached_time = true_times
        self._cached_transform = Transform(matrix3, Vector3.ZERO, self, self._reference)
        return self._cached_transform

    def transform_at_time_if_possible(self, time, *, quick=None):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Unlike method `transform_at_time`, this variant tolerates times that raise cspyce
        errors. If `time` is 1-D, this method returns a new time Scalar along with the new
        Transform, where both objects skip over the times at which the transform could not
        be evaluated. If `time ` has more than one dimension, the cspyce error is still
        raised.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            tuple[Scalar, Transform]: (`valid_time`, `transform`):

            * `valid_time` identifies the time(s) at which `transform` has been provided;
              this may be a subset of the input times, because it omits the times at which
              the Transform could not be evaluated.
            * `transform` is the Transform defined at `valid_time`. It rotates vectors
              from the reference frame to this frame.

        Raises:
            OSError: If `time` is multidimensional and any single time is out of range for
                the currently furnished C kernels, or if every time is out of range. A
                one-dimensional `time` with some times in range returns those. The SPICE
                Toolkit reports this as SPICE(CKINSUFFDATA).
        """

        # Fill in the time tolerance in seconds
        if self._time_tolerance is None:
            time = Scalar.as_scalar(time)
            # One representative time; sce2c takes a single value, and the tick rate does
            # not vary appreciably across the times of one call
            sample = float(np.ravel(time.vals)[0])
            ticks = cspyce.sce2c(self._spice_origin_code, sample)
            ticks_per_sec = cspyce.sce2c(self._spice_origin_code, sample + 1.) - ticks
            self._time_tolerance = self._tick_tolerance / ticks_per_sec

        # A single input time can be handled quickly (with RuntimeError on failure)
        time = Scalar.as_scalar(time)
        if time.shape == ():
            # Check cache first
            xform = self._cache[time.vals]
            if xform:
                return (time, xform)

            ticks = cspyce.sce2c(self._spice_origin_code, time.vals)
            (matrix3, true_ticks) = cspyce.ckgp(self._spice_frame_code, ticks,
                                                self._tick_tolerance,
                                                self._spice_reference_name)
            xform = Transform(matrix3, Vector3.ZERO, self, self._reference)
            self._cache[time.vals] = xform
            return (time, xform)

        # Check to see if the latest shaped transform is adequate
        if np.shape(time.vals) == self._cached_shape:
            diff = np.abs(time.vals - self._cached_time)
            if np.all(diff < self._time_tolerance):
                return (time, self._cached_transform)

        # If all the times are close, we can return more quickly
        time_min = time.vals.min()
        time_max = time.vals.max()
        if (time_max - time_min) < self._time_tolerance:
            tick = cspyce.sce2c(self._spice_origin_code, (time_min + time_max)/2.)
            (matrix3, true_tick) = cspyce.ckgp(self._spice_frame_code, tick,
                                               self._tick_tolerance,
                                               self._spice_reference_name)
            true_time = cspyce.sct2e(self._spice_origin_code, true_tick)
            # If all times fail, raise RuntimeError

            self._cached_shape = time.shape
            self._cached_time = true_time
            self._cached_transform = Transform(matrix3, Vector3.ZERO, self,
                                               self._reference)
            return (time, self._cached_transform)

        # Otherwise, process the array...
        ticks = cspyce.sce2c_vector(self._spice_origin_code, time.vals.ravel())

        # Try all at once
        all_at_once = True
        try:
            matrix3, true_ticks = cspyce.ckgp_vector(self._spice_frame_code, ticks,
                                                     self._tick_tolerance,
                                                     self._spice_reference_name)
        except (RuntimeError, ValueError, IOError) as e:
            if len(time.shape) > 1:
                raise e
            all_at_once = False
        else:
            matrix3 = Matrix3.as_matrix3(matrix3).reshape(time.shape)
            true_times = cspyce.sct2e_vector(self._spice_origin_code, true_ticks)
            valid_times = time

        # Try one at a time for 1-D result
        if not all_at_once:
            matrices = []
            true_ticks = []
            valid_times = []
            error_found = None
            for k, tick in enumerate(ticks):
                if np.shape(time.mask) and time.mask[k]:
                    continue
                try:
                    matrix, true_tick = cspyce.ckgp(self._spice_frame_code, tick,
                                                    self._tick_tolerance,
                                                    self._spice_reference_name)
                except (RuntimeError, ValueError, IOError) as e:
                    error_found = e
                    continue

                matrices.append(matrix)
                true_ticks.append(true_tick)
                valid_times.append(time[k])

            if not matrices:    # if every time failed
                raise error_found

            matrix3 = Matrix3.as_matrix3(matrices)
            true_times = cspyce.sct2e_vector(self._spice_origin_code,
                                             np.array(true_ticks))
            valid_times = Scalar(valid_times)

        xform = Transform(matrix3, Vector3.ZERO, self, self._reference)

        # Cache the result only when it covers every input time. A partial result is
        # defined at fewer times than were asked for, so a later call matching on the
        # input shape would be handed a transform of the wrong size.
        if np.shape(valid_times.vals) == np.shape(time.vals):
            self._cached_shape = time.shape
            self._cached_time = true_times
            self._cached_transform = xform
        else:
            self._cached_shape = None
            self._cached_time = None
            self._cached_transform = None

        return (valid_times, xform)

    ######################################################################################
    # SpiceFrame API
    ######################################################################################

    @staticmethod
    def get(spice_frame, tick_tolerance, reference=None, *, frame_id=None,
            cache_size=None):
        """The SpiceType1Frame defined by the given parameters.

        If a matching SpiceType1Frame already exists, it is returned; otherwise, a new one
        is constructed and returned.

        Parameters:
            spice_frame (str or int): The name, frame code, or frame name as used in the
                SPICE toolkit. Alternatively, an existing SpiceType1Frame (which might use
                the wrong reference frame).
            tick_tolerance (float, int, or str): A number or string defining the time
                tolerance in spacecraft clock ticks for the Frame returned.
            reference (SpiceFrame or str, optional): The Frame or ID of the Frame relative
                to which this frame is defined. This must be a SpiceFrame or else, by
                default, J2000.
            frame_id (str, optional): The ID under which to register this Frame. If not
                specified, the name as defined in the SPICE Toolkit is used. Note that
                SpiceType1Frames are always registered. This input is used only if a new
                SpiceType1Frame is constructed; otherwise, the pre-existing ID is
                retained.
            cache_size (int, optional): The number of transforms to cache. This can be
                useful because it avoids unnecessary SPICE calls when the Frame is being
                used repeatedly at a finite set of times. If not specified, an existing
                SpiceType1Frame with any `cache_size` is returned.

        Returns:
            SpiceType1Frame: The SpiceType1Frame, newly constructed if necessary.

        Raises:
            LookupError: If `spice_frame` is not a recognized frame name or frame code
                within the SPICE Toolkit.
            ValueError: If `reference` is not a SpiceFrame or J2000.
        """

        reference = Frame.as_wayframe(reference)

        # Handle a SpiceType1Frame input; use it if it matches
        if isinstance(spice_frame, SpiceType1Frame):
            if (reference == spice_frame._reference
                    and tick_tolerance == spice_frame._tick_tolerance
                    and cache_size in (spice_frame._cache_size, None)):
                return spice_frame
            # Otherwise, identify the name and continue
            name = spice_frame._spice_frame_name
        else:
            (_, name) = SpiceFrame._frame_code_and_name(spice_frame)

        # The reference must be usable by the constructor; fail here with the same error
        # rather than on a missing attribute below
        reference_name = SpiceType1Frame._reference_spice_info(reference)[1]

        # The constructor converts a tolerance given as a string into ticks, so the key
        # has to be built from the converted value in order to match what it stored.
        if isinstance(tick_tolerance, str):
            tick_tolerance = cspyce.sctiks(cspyce.frinfo(name)[0], tick_tolerance)

        # See if a pre-existing Frame matches the request (including ticks and cache size)
        key = (name, reference_name, tick_tolerance, cache_size)
        if key in SpiceType1Frame._FRAME_LOOKUP:
            return SpiceType1Frame._FRAME_LOOKUP[key]

        # Otherwise, we need a new SpiceType1Frame
        return SpiceType1Frame(name, tick_tolerance, reference,
                               frame_id=frame_id, cache_size=cache_size)

    def _get_shortcut(self, reference):
        """A Frame that directly transforms from the given reference to this Frame.

        This is an override of the default method, needed because the SPICE Toolkit can
        handle the connections between SpiceFrames very efficiently.

        Parameters:
            reference (Frame): The reference Frame, which must be a valid wayframe.

        Returns:
            Frame: This Frame relative to `reference`, connected through the nearest
            SpiceFrame ancestor of `reference`.
        """

        # Find the first SpiceFrame (or J2000) that's an ancestor of the reference
        ancestor = reference
        while not isinstance(ancestor, (SpiceFrame, J2000Frame)):
            ancestor = ancestor._reference

        # Get the SpiceType1Frame to the selected ancestor
        spice_frame = SpiceType1Frame.get(self, self._tick_tolerance, ancestor,
                                          cache_size=self._cache_size)

        # Maybe we're done
        if ancestor == reference:
            return spice_frame

        # Get the "remainder" frame from the ancestor to the reference, then link
        remainder = ancestor._wrt(reference, use_shortcuts=False)
        return LinkedFrame(spice_frame, remainder)

##########################################################################################

Frame._FRAME_SUBCLASSES.append(SpiceType1Frame)

##########################################################################################
