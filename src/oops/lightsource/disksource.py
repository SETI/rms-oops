##########################################################################################
# oops/lightsource/disksource.py
##########################################################################################
"""DiskSource class, which spreads another light source over a uniform, circular disk."""

import numpy as np

from polymath         import Scalar, Vector3, Matrix3
from oops._exceptions import OopsTypeError, OopsValueError
from oops.body        import Body
from oops.constants   import C, RPS
from oops.event       import Event
from oops.lightsource import LightSource


class DiskSource(LightSource):
    """An extended, circular disk with uniform illumination, centered on another source.

    A subclass of LightSource that wraps a single, unshaped LightSource and spreads its
    arrival direction over a disk of lines of sight.

    Attributes:
        lightsource (LightSource): The source at the center of the disk.
        xy_grid (Vector3): The lines of sight across the disk, as offsets in the X/Y
            plane within the unit circle, before they are scaled to the radius of the
            disk and rotated to its center.
        radius (float): The radius of the disk, in km for a moving source or in radians
            for a fixed direction.
    """

    def __init__(self, name, source, radius, *, size=11, compress=False):
        """Constructor for a DiskSource.

        Parameters:
            name (str): Name of this source. It is registered in the Body dictionary
                under its key, as given by :meth:`~oops.lightsource.LightSource.as_key`.
            source (LightSource): The source at the center of the disk. It must be
                unshaped, and it remains registered under its own key.
            radius (float): Radius of the disk, in km if `source` moves along a path or
                in arcseconds if it is a fixed direction.
            size (int, optional): Number of pixels along the side of the square array
                defining the lines of sight; default 11. Use an odd number so that the
                central pixel falls at the center of the source.
            compress (bool, optional): True to strip away the masked pixels outside the
                circle, which leaves about 20% fewer lines of sight to calculate and
                suits a case where the disk is only averaged over at the end. False, the
                default, keeps the source as a 2-D image for later use.

        Raises:
            OopsTypeError: If `name` is not a string or `source` is not a LightSource.
            OopsValueError: If `name` is already the name of a Body, or if `source` is
                shaped.
        """

        if not isinstance(source, LightSource):
            raise OopsTypeError(f'DiskSource source must be a LightSource: {source!r}')

        if source.shape != ():
            raise OopsValueError('DiskSource source must have shape (), not '
                                 + str(source.shape))

        self._set_name(name)
        self.lightsource = source
        self.source = source.source
        self.source_is_moving = source.source_is_moving

        # Create a masked, circular array of vectors in the X/Y plane, inside
        # unit radius
        array = np.zeros((size,size,3))
        xy = (np.arange(size) - (size-1)/2.) / (size/2.)
        array[:,:,0] = xy[np.newaxis]
        array[:,:,1] = xy[:,np.newaxis]
        mask = (array[:,:,0]**2 + array[:,:,1]**2) > 1.

        if compress:
            self.xy_grid = Vector3(array[~mask,:])
            mask = np.zeros(self.xy_grid.shape, dtype='bool')
        else:
            self.xy_grid = Vector3(array, mask=mask)

        self.shape = self.xy_grid.shape
        self.radius = radius if self.source_is_moving else radius * RPS

        # Define the weights
        weight = 1. - np.asarray(mask, dtype=np.float64)
        self.weight = Scalar(weight / np.sum(weight))

        # Keep the constructor inputs so that pickling can rebuild the disk
        self._init_args = (radius, size, compress)

        # Re-register as a Body
        Body.BODY_REGISTRY[self.key] = self

    def __getstate__(self):
        return (self.name, self.lightsource) + self._init_args

    def __setstate__(self, state):
        (name, source, radius, size, compress) = state
        self.__init__(name, source, radius, size=size, compress=compress)

    def photon_to_event(self, event, *, derivs=False, guess=None, antimask=None,
                        quick=None, converge=None):
        """Solve for a photon arrival event from this lightsource.

        Input parameters are identical to the Path method of the same name. The wrapped
        source supplies the direction to the center of the disk, and the disk spreads it
        across an array of lines of sight, so no single departure event corresponds to
        them and the departure is always None.

        Parameters:
            event (Event): The event of the observation.
            derivs (bool, optional): True to propagate derivatives of the event position
                into the returned event. The time derivative is always retained.
            guess (ScalarLike, optional): An initial guess to use as the event time along
                the path; otherwise None. Should only be used if the event time was
                already returned from a similar calculation.
            antimask (numpy.ndarray | bool, optional): If not None, this is a boolean
                array to be applied to event times and positions. Only the indices where
                antimask=True will be used in the solution.
            quick (dict | bool, optional): A dictionary of parameter values to use as
                overrides to the configured default :class:`~oops.path.QuickPath` and
                :class:`~oops.frame.QuickFrame` parameters. Use False to disable the use
                of QuickPaths and QuickFrames. The default quick dictionary is defined in
                config.py.
            converge (dict, optional): A dictionary of parameters to override the
                configured default path convergence parameters. The default configuration
                is defined in config.py. Convergence parameters are as follows:

                * `max_iterations` (int): The maximum number of iterations of Newton's
                  method to perform. It should almost never need to be > 6.
                * `dlt_precision` (float): Iteration stops when the largest change in
                  light travel time between one iteration and the next falls below this
                  threshold (in seconds).
                * `dlt_limit` (float): The maximum allowed absolute value of the change in
                  light travel time from the nominal range calculated initially. Changes
                  in light travel with absolute values larger than this limit are clipped.
                  This prevents the divergence of the solution in some cases.

        Returns:
            tuple[None, Event]: `(departure_event, arrival_event)`.

            * `departure_event`: Always None, because the arrival directions are spread
              across the disk rather than originating from a single point.
            * `arrival_event`: A copy of the given `event`, carrying the arriving
              photon's line of sight spread across the disk.
        """

        arrival = self.lightsource.photon_to_event(event, derivs=derivs, guess=guess,
                                                   antimask=antimask, quick=quick,
                                                   converge=converge)[1]

        # A disk around a path has a radius in km, so its angular size depends on range
        if self.source_is_moving:
            radius = self.radius / (C * arrival.arr_lt)
            center = arrival.neg_arr_ap_j2000
        else:
            radius = self.radius
            center = arrival.neg_arr_j2000

        # The twovec matrix rotates the center onto the Z-axis, so its transpose carries
        # the grid around the Z-axis to the center
        matrix = Matrix3.twovec(center, 2, Vector3.YAXIS, 1)
        new_los = matrix.T * (Vector3.ZAXIS + radius * self.xy_grid)

        if self.source_is_moving:
            new_event = Event(event.time, event.state, event.origin, event.frame)
            new_event.neg_arr_ap_j2000 = new_los
        else:
            new_event = event.copy() if derivs else event.wod.copy()
            new_event.neg_arr_j2000 = new_los

        return (None, new_event)

##########################################################################################
