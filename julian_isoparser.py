################################################################################
# julian_isoparser.py
#
# This is a set of routines for parsing ISO-formatted dates and times. It is a
# component of the Julian Library and is not intended to be invoked separately,
# although it can be.
#
# Date grammars:
#
#   ISO_DATE        parses a date string in the form yyyy-ddd or yyyy-mm-dd.
#
#   ISO_TIME        parses a time string in the form hh:mm:ss[.sss][Z]
#
#   ISO_DATETIME    parses a date and time separated by a "T" or a single blank.
#
# The parsers return a list of 2-item lists, where the first item is the name
# of the component and the second is its value. The names are "YEAR", "MONTH",
# "DAY" for dates, "HOUR", "MINUTE" and "SECOND" for times. The value of seconds
# can be integer or float; other values are floats. If they format yyyy-mmm is
# used for the date, then the month is assigned a value of zero.
#
# Mark R. Showalter
# PDS Rings Node
# August 2011
#
# Revised December 23, 2011 (BSW) - changed DOY parsing to allow 3rd digit to be
#                                   greater than 5 for values >= 300
################################################################################

from pyparsing import *
import unittest

################################################################################
# BEGIN GRAMMAR
################################################################################

# Whitespace is ignored
ParserElement.setDefaultWhitespaceChars("")

# Useful definitions...
DASH            = Suppress(Literal("-"))
COLON           = Suppress(Literal(":"))
T               = Suppress(Literal("T"))
SPACE           = Suppress(Literal(" "))
Z               = Suppress(Literal("Z"))

################################################################################
# Date and time components
################################################################################

YEAR        = Word(nums,exact=4)
YEAR.setParseAction(lambda s,l,t: [["YEAR", int(t[0])]])

MONTH       = Word("01",nums,exact=2)
MONTH.setParseAction(lambda s,l,t: [["MONTH", int(t[0])]])

DAY         = Word("0123",nums,exact=2)
DAY.setParseAction(lambda s,l,t: [["DAY", int(t[0])]])

DOY         = ( Word("012",nums,exact=3)
              | Combine("3"  + Word("012345",nums,exact=2))
              | Combine("36" + Word("0123456",exact=1))
              )
DOY.setParseAction(lambda s,l,t: [["DAY", int(t[0])]])

HOUR        = Word("012",nums,exact=2)
HOUR.setParseAction(lambda s,l,t: [["HOUR", int(t[0])]])

MINUTE      = Word("012345",nums,exact=2)
MINUTE.setParseAction(lambda s,l,t: [["MINUTE", int(t[0])]])

SECOND_INT   = Combine(Word("0123456",nums,exact=2))
SECOND_INT.setParseAction(lambda s,l,t: [["SECOND", int(t[0])]])

SECOND_FLOAT = Combine(Word("0123456",nums,exact=2)
                + "." + Optional(Word(nums)))
SECOND_FLOAT.setParseAction(lambda s,l,t: [["SECOND", float(t[0])]])

SECOND      = SECOND_FLOAT | SECOND_INT

################################################################################
# Parsers
################################################################################

ISO_DATE    = YEAR + DASH + ((MONTH + DASH + DAY) | DOY)

ISO_TIME    = HOUR + COLON + MINUTE + COLON + SECOND + Optional(Z)

ISO_DATETIME = ISO_DATE + (T | SPACE) + ISO_TIME

########################################
# UNIT TESTS
########################################

class Test_ISO_DATE(unittest.TestCase):

    def runTest(self):

        # ISO_DATE matches...
        parser = ISO_DATE + StringEnd()

        self.assertEqual(parser.parseString("1776-07-04").asList(),
                [["YEAR",1776],["MONTH",7],["DAY",4]])
        self.assertEqual(parser.parseString("1776-004").asList(),
                [["YEAR",1776],["DAY",4]])

        # Doesn't recognize...
        self.assertRaises(ParseException, parser.parseString, " 1776-07-04")
        self.assertRaises(ParseException, parser.parseString, "1776 -07-04")
        self.assertRaises(ParseException, parser.parseString, "1776- 07-04")
        self.assertRaises(ParseException, parser.parseString, "1776-07 -04")
        self.assertRaises(ParseException, parser.parseString, "1776-07- 04")
        self.assertRaises(ParseException, parser.parseString, "1776-07-044")
        self.assertRaises(ParseException, parser.parseString, "1776-07-04T")
        self.assertRaises(ParseException, parser.parseString, "1776-07-004")
        self.assertRaises(ParseException, parser.parseString, "1776-004T")
        self.assertRaises(ParseException, parser.parseString, "1776- 004")
        self.assertRaises(ParseException, parser.parseString, "1776 -004")
        self.assertRaises(ParseException, parser.parseString, "1776-367")

class Test_ISO_TIME(unittest.TestCase):

    def runTest(self):

        # ISO_TIME matches...
        parser = ISO_TIME + StringEnd()

        self.assertEqual(parser.parseString("12:34:56").asList(),
                        [["HOUR", 12], ["MINUTE", 34], ["SECOND", 56]])
        self.assertEqual(parser.parseString("23:45:67Z").asList(),
                        [["HOUR", 23], ["MINUTE", 45], ["SECOND", 67]])
        self.assertEqual(parser.parseString("12:34:56.").asList(),
                        [["HOUR", 12], ["MINUTE", 34], ["SECOND", 56.]])
        self.assertEqual(parser.parseString("23:45:67.Z").asList(),
                        [["HOUR", 23], ["MINUTE", 45], ["SECOND", 67.]])
        self.assertEqual(parser.parseString("12:34:56.").asList(),
                        [["HOUR", 12], ["MINUTE", 34], ["SECOND", 56.]])
        self.assertEqual(parser.parseString("12:34:56.789").asList(),
                        [["HOUR", 12], ["MINUTE", 34], ["SECOND", 56.789]])
        self.assertEqual(parser.parseString("23:45:67.89Z").asList(),
                        [["HOUR", 23], ["MINUTE", 45], ["SECOND", 67.89]])

        # Doesn't recognize...
        self.assertRaises(ParseException, parser.parseString, " 12:34:56")
        self.assertRaises(ParseException, parser.parseString, "12 :34:56")
        self.assertRaises(ParseException, parser.parseString, "12: 34:56")
        self.assertRaises(ParseException, parser.parseString, "12:34 :56")
        self.assertRaises(ParseException, parser.parseString, "12:34: 56")
        self.assertRaises(ParseException, parser.parseString, "12:34: 56.")
        self.assertRaises(ParseException, parser.parseString, "12:34:56 .789")
        self.assertRaises(ParseException, parser.parseString, "12:34:56.7 89")
        self.assertRaises(ParseException, parser.parseString, "12:34: 56")
        self.assertRaises(ParseException, parser.parseString, "12:34: 56")
        self.assertRaises(ParseException, parser.parseString, "12:34:56 Z")
        self.assertRaises(ParseException, parser.parseString, "12:34:56a")

class Test_ISO_DATETIME(unittest.TestCase):

    def runTest(self):

        # ISO_DATETIME matches...
        parser = ISO_DATETIME + StringEnd()
        self.assertEqual(parser.parseString("1776-07-04 12:34:56").asList(),
                [["YEAR",1776],["MONTH",7],["DAY",4],
                 ["HOUR", 12], ["MINUTE", 34], ["SECOND", 56]])
        self.assertEqual(parser.parseString("1776-07-04T12:34:56Z").asList(),
                [["YEAR",1776],["MONTH",7],["DAY",4],
                 ["HOUR", 12], ["MINUTE", 34], ["SECOND", 56]])
        self.assertEqual(parser.parseString("1776-004 12:34:56").asList(),
                [["YEAR",1776],["DAY",4],
                 ["HOUR", 12], ["MINUTE", 34], ["SECOND", 56]])
        self.assertEqual(parser.parseString("1776-004T12:34:56Z").asList(),
                [["YEAR",1776],["DAY",4],
                 ["HOUR", 12], ["MINUTE", 34], ["SECOND", 56]])

        # Doesn't recognize...
        self.assertRaises(ParseException,
                          parser.parseString, " 1776-07-04 12:34:56")
        self.assertRaises(ParseException,
                          parser.parseString, "1776-07-04  12:34:56")
        self.assertRaises(ParseException,
                          parser.parseString, "1776-07-04 T12:34:56")
        self.assertRaises(ParseException,
                          parser.parseString, "1776-07-04T 12:34:56")
        self.assertRaises(ParseException,
                          parser.parseString, "1776-07-04T 12:34:567")
        self.assertRaises(ParseException,
                          parser.parseString, " 1776-004 12:34:56")
        self.assertRaises(ParseException,
                          parser.parseString, "1776-004  12:34:56")
        self.assertRaises(ParseException,
                          parser.parseString, "1776-004 T12:34:56")
        self.assertRaises(ParseException,
                          parser.parseString, "1776-004T 12:34:56")
        self.assertRaises(ParseException,
                          parser.parseString, "1776-004T 12:34:567")

################################################################################
# Perform unit testing if executed from the command line
################################################################################

if __name__ == "__main__":
    unittest.main()

################################################################################

