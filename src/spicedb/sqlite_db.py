import sqlite3

##########################################################################################
# Low-level database IO using SQLite 3
##########################################################################################

global CONNECTION, CURSOR
CONNECTION = None
CURSOR = None

def open(filepath):
    """Opens the database.

    Parameters:
        filepath (str or FCPath): The file path and name of the database file.
    """

    global CONNECTION, CURSOR

    CONNECTION = sqlite3.connect(filepath)
    CURSOR = CONNECTION.cursor()

def close():
    """Closes the database."""

    global CONNECTION, CURSOR

    CURSOR.close()
    CONNECTION = None
    CURSOR = None

def query(sql_string):
    """Executes a SQL query.

    Parameters:
        sql_string (str): A string containing the complete SQL query.

    Returns:
        tuple: A tuple, where:

        * `table` (list): A list of lists containing the rows and columns of results
          returned by the query.
    """

    if CURSOR is None:
        raise RuntimeError("open database file first")

    # Execute and return the results
    CURSOR.execute(sql_string)

    # Convert to a list of KernelInfo objects...
    table = []
    for row in CURSOR:
        columns = []
        for item in row:

            # convert items to Python type if necessary
            if isinstance(item, (int, float)):      # Item is an int or float
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

##########################################################################################
