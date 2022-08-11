################################################################################
# oops/obs_/insitu.py: Subclass InSitu of class Observation
################################################################################

import numpy as np
from polymath import *

from oops.obs_.observation  import Observation
from oops.path_.path        import Path
from oops.frame_.frame      import Frame
from oops.cadence_.cadence  import Cadence
from oops.cadence_.instant  import Instant
from oops.fov_.nullfov      import NullFOV

#*******************************************************************************
# InSitu
#*******************************************************************************
class InSitu(Observation):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    InSitu is a subclass of Observation that has timing and path information,
    but no attributes related to pointing or incoming photon direction. It can
    be useful for describing in situ measurements.

    It can also be used to obtain information from gridless backplanes, which
    do not require directional information.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    PACKRAT_ARGS = ['cadence', 'path', 'subfields']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, cadence, path, **subfields):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for an InSitu observation.

        Input:
            cadence     a Cadence object defining the time and duration of each
                        "measurement". Note that the shape of the cadence
                        defines the dimensions of the observation. As a special
                        case, a Scalar value is converted to a Cadence of
                        subclass Instant, making this observation suitable for
                        evaluating any gridless backplane. The shape of this
                        cadence defines the shape of the observation.

            path        the path waypoint co-located with the observer.

            subfields   a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #--------------------------------------------------
        # Basic properties
        #--------------------------------------------------
        self.path = Path.as_waypoint(path)
        self.frame = Frame.J2000

        #--------------------------------------------------
        # FOV
        #--------------------------------------------------
        self.fov = NullFOV()

        #--------------------------------------------------
        # Cadence
        #--------------------------------------------------
        if isinstance(cadence, Cadence):
            self.cadence = cadence
        elif isinstance(cadence, Scalar):
            self.cadence = Instant(cadence)
        else:
            raise TypeError('Invalid cadence class: ' + type(cadence).__name__)

        #--------------------------------------------------
        # Axes / Shape / Size
        #--------------------------------------------------
        self.u_axis = -1
        self.v_axis = -1
        self.swap_uv = False

        self.uv_shape = (1,1)

        self.shape = self.cadence.shape
        self.t_axis = list(np.range(len(self.shape)))

        #--------------------------------------------------
        # Shape / Size
        #--------------------------------------------------
        self.shape = self.cadence.shape
        self.uv_shape = (1,1)

        #--------------------------------------------------
        # Timing
        #--------------------------------------------------
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        #--------------------------------------------------
        # Optional subfields
        #--------------------------------------------------
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        return
    #===========================================================================



#*******************************************************************************

################################################################################
# UNIT TESTS
################################################################################

import unittest

#*******************************************************************************
# Test_InSitu
#*******************************************************************************
class Test_InSitu(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        # No tests here - this is just an abstract superclass

        pass
    #===========================================================================


#*******************************************************************************

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
