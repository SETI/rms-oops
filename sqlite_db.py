import sqlite3
import unittest

################################################################################
# Low-level database IO using SQLite 3
################################################################################

global DATABASE, CONNECTION, CURSOR

DATABASE = None
CONNECTION = None
CURSOR = None

def open(filepath):
    """Opens the database.

    Input:
        filepath        The file path and name of the database file.
    """
    global DATABASE, CONNECTION, CURSOR

    CONNECTION = sqlite3.connect(DATABASE)
    CURSOR = CONNECTION.cursor()

############################################

def close():
    """Closes the database."""

    global DATABASE, CONNECTION, CURSOR

    CURSOR.close()
    DATABASE = None
    CONNECTION = None
    CURSOR = None

############################################

def query(sql_string):
    """Executes a SQL query.

    Input:
        sql_string      A string containing the complete SQL query.

    Output:
        table           A list of lists containing the rows and columns of
                        results returned by the query.
    """

    # Execute and return the results
    CURSOR.execute(query_string)

    # Convert to a list of KernelInfo objects...
    table = []
    for row in cursor:
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

################################################################################
# UNIT TESTS
################################################################################

class test_database(unittest.TestCase):

    def runTest(self):

        self.assertTrue(DATABASE == None)
        self.assertTrue(CONNECTION == None)
        self.assertTrue(CURSOR == None)

        open()

        self.assertTrue(DATABASE != None)
        self.assertTrue(CONNECTION != None)
        self.assertTrue(CURSOR != None)

        close()

        self.assertTrue(DATABASE == None)
        self.assertTrue(CONNECTION == None)
        self.assertTrue(CURSOR == None)

################################################################################
# Perform unit testing if executed from the command line
################################################################################

if __name__ == '__main__':
    unittest.main()

################################################################################
