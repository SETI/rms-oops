################################################################################
# HmsFormat
#
# 1/24/12 (MRS) - Drafted.
################################################################################

import oops

class HmsFormat(oops.Format):
    """An HmsFormat is a Format object that handles a numeric value in
    hour-minute-second format.
    """

    def __init__(self, hchar="h ", mchar="m ", schar="s", digits=0, pos=" "):
        """The constructor for an HmsFormat object.

        Input:
            hchar       the character or characters that trails the hour/degree
                        field.
            mchar       the character or characters that trails the minute
                        field, defined as 1/60th of an hour/degree.
            schar       the character or characters the trails the seconds
                        field, defined as 1/60th of a minute.
            digits      the number of digits to attach to the seconds field.
            pos         the leading character or characters to include if the
                        sign of the value is positive. A value of "-" always
                        leads if the value is negative.
        """

        self.hchar = hchar
        self.mchar = mchar
        self.schar = schar
        self.digits = digits
        self.pos = pos

    def str(value):
        """Returns a character string indicating the value of a numeric quantity
        such as a coordinate.

        Example: The default HmsFormat for a value (1 + 2/60. + 3/3600.) is:
            "01h 02m 03s"

        Example: For HmsFormat(":", ":", "", 3), the value of (1 + 2/60. +
        3.45678/3600. would be:
            "01:02:03.457"

        Example: For HmsFormat(":", ":", "", 3), the value of (-1 - 2/60. -
        3.45678/3600. would be:
            "-01:02:03.457"
        """

        # TBD
        pass

    def parse(string):
        """Returns a numeric value derived by parsing a character string."""

        # TBD
        pass

# Random note: Be very careful about the leading sign. The most common error in
# such a routine is to lose the sign if the hours value is zero.
################################################################################
