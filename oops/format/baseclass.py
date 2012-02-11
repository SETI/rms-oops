################################################################################
# oops/format/baseclass.py: Abstract class Format
#
# 1/24/12 (MRS) - Drafted.
################################################################################

class Format(object):
    """Format is a generic class that defines a mechanism for converting a
    numeric value to a string and a string to a numeric value.
    """

    OOPS_CLASS = "Format"

    def __init__(self):
        """The constructor for a Format object"""

        pass

    def str(self, value):
        """Returns a character string indicating the value of a numeric quantity
        such as a coordinate.
        """

        pass

    def parse(self, string):
        """Returns a numeric value interpreted from a character string."""

        pass

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Format(unittest.TestCase):
    
    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
