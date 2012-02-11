################################################################################
# PythonFormat
#
# 1/24/12 (MRS) - Drafted.
################################################################################

import oops

class PythonFormat(oops.Format):
    """A PythonFormat is a format string defined using the default formatting
    mechanism in Python.
    """

    def __init__(self, string):
        """The constructor for a Format object"""

        self.format = string

    def str(value):
        """Returns a character string indicating the value of a numeric quantity
        such as a coordinate.
        """

        return self.format % value

    def parse(string):
        """Returns a numeric value derived by parsing a character string."""

        return float(string)

################################################################################
