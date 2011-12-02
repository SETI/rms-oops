import numpy as np
import pylab
import unittest

import oops

################################################################################
# Observation Class
################################################################################

class Observation(object):
    """An Observation is an abstract class that defines the timing and pointing
    of the samples that comprise a data product.

    A data product is always indexed in "standard order" (u,v,t,...). If the
    indices are not in this order at the outset, then the array must be revised
    using rollaxis(), swapaxes(), etc. to take on this shape.

    Spatial locations are indexed by (u,v), where u is horizontal and v is
    vertical within the field of view. The u-axis always points toward the
    right, but the v-axis direction can point either upward or downward. Time is
    indexed by t. Additional indices might be needed to identify other axes of
    a data product, such as the wavelength within a spectrum.

    The shape of a product is described by the number of samples along each
    axis. For example, a movie consisting of ten images, each 1000x1000, would
    have shape (1000,1000,10). A single 2-D image will have length = 1 along the
    t-axis. Similarly, a single occultation profile with 10000 samples will have
    shape (1,1,10000). A spectrum with 100 wavelength samples would have shape
    (1,1,1,100), where the additional axis is used to index wavelength.

    Indices can have non-integer values. When they do, the integer part
    identifies one "corner" of the sample, and the fractional part locates a
    point within the sample, i.e., part way from the start time to the end time
    of an integration, or a location inside the boundaries of a spatial pixel.
    Half-integer indices falls at the midpoint of each sample.

    These attributes are used to describe the field of view:
        data            the data array, reshaped for indexing using standard
                        order.

        axes            a list containing the codes that describe the axes:
                            "u" = horizontal, always rightward;
                            "v" = vertical, up if fov.uv_scale[1] is negative;
                            "t" = time
                        Other axis names are allowed as needed.

        (t0, t1)        overall time limits of the observation.

        fov             a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and a direction in
                        space.

        path_id         the registered ID of a path co-located with the
                        instrument.

        frame_id        the registered ID of a coordinate frame fixed to the
                        optics of the instrument. This frame should have its
                        Z-axis pointing outward near the center of the line of
                        sight, with the X-axis pointing rightward and the y-axis
                        pointing downward.

    """

    OOPS_CLASS = "Observation"

    # Works for a standard 2-D image
    def __init__(self, data, axes, time, fov, path_id, frame_id):

        self.data  = data
        self.path_id = path_id
        self.frame_id = frame_id
        self.t0 = time[0]
        self.t1 = time[1]
        self.fov = fov

        self.u_axis = axes.index("u")
        self.v_axis = axes.index("v")
        self.t_axis = None

########################################
# UNIT TESTS
########################################

class Test_Observation(unittest.TestCase):

    def runTest(self):

        pass

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################