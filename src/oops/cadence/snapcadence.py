##########################################################################################
# oops/cadence/snapcadence.py
##########################################################################################

from oops.cadence import Metronome


class SnapCadence(Metronome):
    """A Cadence subclass with a single time step."""

    def __init__(self, tstart, texp, *, clip=True):
        """Constructor for a SnapCadence.

        Parameters:
            tstart (float): The start time of the observation in seconds TDB.
            texp (float): The exposure time in seconds.
            clip (bool, optional): If True (the default), times and index values are
                always clipped into the valid range.
        """

        Metronome.__init__(self, tstart, texp, texp, 1, clip=clip)

    def __getstate__(self):
        self.refresh()
        return (self._tstart, self._texp, self._clip)

    def __setstate__(self, state):
        (tstart, texp, clip) = state
        self.__init__(tstart, texp, clip=clip)
        self.freeze()

##########################################################################################
