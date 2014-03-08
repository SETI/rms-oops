################################################################################
# oops/format_/hms.py: HMS subclass of class Format
################################################################################

from oops.format_.format import Format

class HMS(Format):
    """A Format object that handles a numeric value in hour-minute-second format.
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

    def str(self, value):
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

        hours = int(value)
        fminutes = (value - hours) * 60.
        minutes = int(fminutes)
        fseconds = (fminutes - minutes) * 60.
        seconds_pad = ''
        if fseconds < 10.:
            seconds_pad = '0'
        leading_s = self.pos
        if value < 0.:
            leading_s = '-'
        f = "%s%2d%s%02d%s%s%f%s" % (leading_s, hours, self.hchar, minutes,
                                     self.mchar, seconds_pad, fseconds,
                                     self.schar)
        return f

    def parse(self, string):
        """Returns a numeric value derived by parsing a character string."""

        hs = string.split(self.hchar)
        hours = int(hs[0])
        # the hchar and mchar may or may not be the same, so check
        if self.hchar == self.mchar:
            minutes = int(hs[1])
            if len(self.schar) > 0:
                ss = hs[2].split(self.schar)[0]
            else:
                ss = hs[2]
        else:
            ms = string.split(self.mchar)
            ms1 = ms[0].split(self.hchar)
            minutes = int(ms1[1])
            if len(self.schar) > 0:
                ss = ms[1].split(self.schar)[0]
            else:
                ss = ms[1]

        seconds = float(ss)
        time = hours + minutes / 60. + seconds / 3600.
        return time

    def int_from_component(self, component):
        """Returns an int for the string, dealing with leading zeroes properly."""
        s = component
        if len(component) > 1 and component[0] == '0':
            s = component[1:]
        return int(s)

# Random note: Be very careful about the leading sign. The most common error in
# such a routine is to lose the sign if the hours value is zero.

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_HMS(unittest.TestCase):
    
    def runTest(self):
        
        t1 = 2.384141
        fmt = HMS(':',':','')
        s1 = fmt.str(2.384141)
        self.assertEqual(s1, "  2:23:02.907600")
        t1a = fmt.parse(s1)
        self.assertTrue(t1 == t1a)
                        
        fmt = HMS('h ','m ','s')
        s2 = fmt.str(2.384141)
        self.assertEqual(s2, "  2h 23m 02.907600s")
        t2a = fmt.parse(s2)
        self.assertTrue(t1 == t2a)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
