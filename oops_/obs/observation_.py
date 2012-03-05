################################################################################
# oops_/obs/baseclass.py: Abstract class Observation
#
# 2/11/12 Modified (MRS) - updated for style
################################################################################

import numpy as np

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

    At minimum, attributes are used to describe the observation:
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

        subfields       a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.

    """

    def __init__(self):

        pass

    ####################################################
    # Subarray support methods
    ####################################################

    def insert_subfield(self, key, value):
        """Adds a given subfield to the Event."""

        self.subfields[key] = value
        self.__dict__[key] = value      # This makes it an attribute as well

    def delete_subfield(self, key):
        """Deletes a subfield, but not arr or dep."""

        if key in ("arr","dep"):
            self.subfields[key] = Empty()
            self.__dict__[key] = self.subfields[key]
        elif key in self.subfields.keys():
            del self.subfields[key]
            del self.__dict__[key]

    def delete_subfields(self):
        """Deletes all subfields."""

        for key in self.subfields.keys():
            if key not in ("arr","dep"):
                del self.subfields[key]
                del self.__dict__[key]

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Observation(unittest.TestCase):

    def runTest(self):

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
