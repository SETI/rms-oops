################################################################################
# oops/inst/instrument.py: Abstract Instrument class
################################################################################

import numpy as np

class Instrument(object):
    """Instrument is an abstract class that interprets a given data file and
    returns the associated Observation object. Instrument classes have no
    instances but use inheritance to define the relationships between different
    missions and instruments.
    """

    pass

################################################################################
