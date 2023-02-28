################################################################################
# oops/format_/format.py: Abstract class Format
################################################################################

class Format(object):
    """A generic class for converting numeric values to/from strings."""

    OOPS_CLASS = "Format"

    def __init__(self):
        """The constructor for a Format object"""
        pass

    def str(self, value):
        """Returns a character string indicating the value of a numeric
        quantity.
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
        # No tests here - this is just an abstract superclass
        pass

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
