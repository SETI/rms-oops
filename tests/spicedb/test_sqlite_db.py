##########################################################################################
# tests/spicedb/test_sqlite_db.py
##########################################################################################

import pytest

import spicedb.sqlite_db as db


@pytest.mark.skip(reason='needs the minimal test_data/SPICE.db fixture, which is '
                         'not part of the OOPS_RESOURCES tree')
def test_sqlite_db():

    assert db.CONNECTION is None
    assert db.CURSOR is None

    db.open("test_data/SPICE.db")

    assert db.CONNECTION is not None
    assert db.CURSOR is not None

    result = db.query("select name from sqlite_master")
    assert result == [["SPICEDB"]]

    string = db.query("select sql from sqlite_master")[0][0]
    assert "KERNEL_NAME text NOT NULL" in string
    assert "KERNEL_TYPE text NOT NULL" in string
    assert "FILESPEC text" in string
    assert "START_TIME text" in string
    assert "STOP_TIME text" in string
    assert "RELEASE_DATE text" in string
    assert "SPICE_ID integer" in string
    assert "LOAD_PRIORITY integer" in string

    db.close()

    assert db.CONNECTION is None
    assert db.CURSOR is None

##########################################################################################
