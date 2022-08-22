import sqlite3
import unittest

################################################################################
# Low-level database IO using SQLite 3
#
# Deukkwon Yoon & Mark Showalter
# PDS Rings Node, SETI Institute
# December 2011
################################################################################

global CONNECTION, CURSOR
CONNECTION = None
CURSOR = None

#===============================================================================
# open
#===============================================================================
def open(filepath):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Opens the database.

    Input:
        filepath        The file path and name of the database file.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    global CONNECTION, CURSOR

    CONNECTION = sqlite3.connect(filepath)
    CURSOR = CONNECTION.cursor()

#===============================================================================



#===============================================================================
# close
#===============================================================================
def close():
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Closes the database.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    global CONNECTION, CURSOR

    CURSOR.close()
    CONNECTION = None
    CURSOR = None
#===============================================================================



#===============================================================================
# query
#===============================================================================
def query(sql_string):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Executes a SQL query.

    Input:
        sql_string      A string containing the complete SQL query.

    Output:
        table           A list of lists containing the rows and columns of
                        results returned by the query.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if CURSOR is None:
        raise RuntimeError("open database file first")

    #------------------------------------
    # Execute and return the results
    #------------------------------------
    CURSOR.execute(sql_string)

    #-----------------------------------------------
    # Convert to a list of KernelInfo objects...
    #-----------------------------------------------
    table = []
    for row in CURSOR:
        columns = []
        for item in row:

            # convert items to Python type if necessary
            if type(item) == type(0):               # Item is an integer
                value = item

            elif type(item) == type(0.0):           # Item is a float
                value = item

            elif type(item) == type(u"unicode"):    # Item is an unicode
                value = str(item)

            elif type(item) == type(None):          # Item is a None type
                value = None

            else:
                value = item

            columns.append(value)

        table.append(columns)

    return table
#===============================================================================



########################################
# UNIT TESTS
########################################

#*******************************************************************************
# test_sqlite_db
#*******************************************************************************
class test_sqlite_db(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        self.assertTrue(CONNECTION is None)
        self.assertTrue(CURSOR is None)

        open("test_data/SPICE.db")

        self.assertTrue(CONNECTION is not None)
        self.assertTrue(CURSOR is not None)

        result = query("select name from sqlite_master")
        self.assertEqual(result, [["SPICEDB"]])

        string = query("select sql from sqlite_master")[0][0]
        self.assertTrue("KERNEL_NAME text NOT NULL" in string)
        self.assertTrue("KERNEL_TYPE text NOT NULL" in string)
        self.assertTrue("FILESPEC text" in string)
        self.assertTrue("START_TIME text" in string)
        self.assertTrue("STOP_TIME text" in string)
        self.assertTrue("RELEASE_DATE text" in string)
        self.assertTrue("SPICE_ID integer" in string)
        self.assertTrue("LOAD_PRIORITY integer" in string)

        close()

        self.assertTrue(CONNECTION is None)
        self.assertTrue(CURSOR is None)
    #===========================================================================


#*******************************************************************************



################################################################################
# Perform unit testing if executed from the command line
################################################################################

if __name__ == '__main__':
    unittest.main()

################################################################################
