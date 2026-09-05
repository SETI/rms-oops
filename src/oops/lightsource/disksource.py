##########################################################################################
# oops/lightsource/disksource.py
##########################################################################################

import numpy as np

from polymath         import Scalar, Vector3, Matrix3
from oops.body        import Body
from oops.constants   import C, RPS
from oops.event       import Event
from oops.lightsource import LightSource


class DiskSource(LightSource):
    """An extended, circular disk with uniform illumination.

    A subclass of LightSource.
    """

    def __init__(self, name, source, radius, size=11, compress=False):
        """Constructor for a DiskSource.

        Parameters:
            name (str): Name to register in the Body dictionary.
            source (Path, tuple, or Vector3): A Path, or a fixed direction in J2000
                coordinates given either as a (right ascension, declination) pair in
                degrees or as a single line of sight.
            radius (float): Radius of the source, in km for a path or in arcseconds for a
                fixed J2000 source.
            size (int, optional): Number of pixels along the side of the square array
                defining the lines of sight; default 11. Use an odd number so that the
                central pixel falls at the center of the source.
            compress (bool, optional): True to strip away the masked pixels outside the
                circle, which leaves about 20% fewer lines of sight to calculate and
                suits a case where the disk is only averaged over at the end. False, the
                default, keeps the source as a 2-D image for later use.

        Raises:
            ValueError: If `source` does not describe a single, unshaped direction.
        """

        # Start with the default LightSource
        lightsource = LightSource(name, source)

        # Make sure this one is un-shaped
        if lightsource.shape != ():
            del Body.BODY_REGISTRY[lightsource.name]
            raise ValueError('DiskSource source must have shape (), not ' +
                             str(lightsource.shape))

        # At this point, the LightSource internals are filled in and valid
        self.name = lightsource.name
        self.source = lightsource.source
        self.source_is_moving = lightsource.source_is_moving

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
        self.radius = radius

        # For a fixed line of sight, rotate and scale the vectors now
        if not self.source_is_moving:
            self.radius *= RPS
            matrix = Matrix3.twovec(lightsource.source, 2, Vector3.YAXIS, 1)
            self.xy_grid = matrix * (Vector3.ZAXIS + self.radius * self.xy_grid)

        # Define the weights
        weight = 1. - np.asarray(mask, dtype=np.float64)
        self.weight = Scalar(weight / np.sum(weight))

        # Re-register as a Body
        Body.BODY_REGISTRY[self.name] = self

    def photon_to_event(self, event, derivs=False, guess=None, antimask=None, quick=None,
                        converge=None):
        """Solve for a photon arrival event from this lightsource.

        Input parameters are identical to the Path method of the same name. The disk
        spreads the arrival direction across an array of lines of sight, so no single
        departure event corresponds to them and the departure is always None.

        Parameters:
            event (Event): The event of the observation.
            derivs (bool, optional): True to propagate derivatives of the event position
                into the returned event. The time derivative is always retained.
            guess (Scalar, optional): An initial guess to use as the event time along the
                path; otherwise None. Should only be used if the event time was already
                returned from a similar calculation.
            antimask (numpy.ndarray or bool, optional): If not None, this is a boolean
                array to be applied to event times and positions. Only the indices where
                antimask=True will be used in the solution.
            quick (dict or bool, optional): A dictionary of parameter values to use as
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

        if self.source_is_moving:
            arrival = self.source.photon_to_event(event, derivs=derivs, guess=guess,
                                                  antimask=antimask, quick=quick,
                                                  converge=converge)[1]

            rad = self.radius / (C * arrival.arr_lt)
            los = arrival.neg_arr_ap_j2000
            matrix = Matrix3.twovec(los, 2, Vector3.YAXIS, 1)
            new_los = matrix * (Vector3.ZAXIS + rad * self.xy_grid)
            new_event = Event(event.time, event.state, event.origin, event.frame)
            new_event.neg_arr_ap_j2000 = new_los
            return (None, new_event)

        if derivs:
            new_event = event.copy()
        else:
            new_event = event.wod.copy()

        new_event.neg_arr_j2000 = self.xy_grid
        return (None, new_event)

##########################################################################################
