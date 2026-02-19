################################################################################
# oops/cadence/snapcadence.py: SnapCadence subclass of class Metronome
################################################################################

from oops.cadence import Metronome

class SnapCadence(Metronome):
    """A shapeless Cadence subclass with a single start and stop."""

    def __init__(self, tstart, texp, clip=True):
        """Constructor for a SnapCadence.

        Input:
            tstart      the start time of the observation in seconds TDB.
            texp        the exposure time in seconds.
            clip        if True (the default), times and index values are always
                        clipped into the valid range.
        """

        Metronome.__init__(self, tstart, texp, texp, 1, clip=clip)

    def __getstate__(self):
        return (self.tstart, self.texp, self.clip)

    def __setstate__(self, state):
        self.__init__(*state)

################################################################################
