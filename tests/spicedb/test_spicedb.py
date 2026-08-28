################################################################################
# tests/spicedb/test_spicedb.py
################################################################################

from pathlib import Path

import pytest

import spicedb

# For reference...
# ['KERNEL_NAME','KERNEL_VERSION', 'KERNEL_TYPE', 'FILESPEC',
#  'START_TIME', 'STOP_TIME', 'RELEASE_DATE', 'SPICE_ID', 'LOAD_PRIORITY']
def test_kernelinfo():
    # Sort based on kernel type
    T0 = '2000-01-01T00:00:00'
    T1 = '2001-01-01T00:00:00'
    T2 = '2002-01-01T00:00:00'
    T3 = '2003-01-01T00:00:00'
    T4 = '2004-01-01T00:00:00'
    T5 = '2005-01-01T00:00:00'
    T6 = '2006-01-01T00:00:00'
    T7 = '2007-01-01T00:00:00'
    T8 = '2008-01-01T00:00:00'
    T9 = '2009-01-01T00:00:00'

    lsk  = spicedb.KernelInfo(['LSK',  '1', 'LSK',  'file', T0, T1, T2, 0, 1])
    lsk2 = spicedb.KernelInfo(['LSK',  '1', 'LSK',  'file', T0, T1, T2, 0, 1])
    sclk = spicedb.KernelInfo(['SCLK', '1', 'SCLK', 'file', T0, T1, T2, 0, 1])
    fk   = spicedb.KernelInfo(['FK',   '1', 'FK',   'file', T0, T1, T2, 0, 1])
    ik   = spicedb.KernelInfo(['IK',   '1', 'IK',   'file', T0, T1, T2, 0, 1])
    ck   = spicedb.KernelInfo(['CK',   '1', 'CK',   'file', T0, T1, T2, 0, 1])
    spk  = spicedb.KernelInfo(['SPK',  '1', 'SPK',  'file', T0, T1, T2, 0, 1])

    assert lsk == lsk2
    assert lsk <= lsk2
    assert lsk >= lsk2
    assert not lsk < lsk2
    assert not lsk > lsk2
    assert not lsk != lsk2

    assert lsk < sclk
    assert sclk < ck
    assert fk < ck
    assert fk < ik
    assert ik < spk

    kernels = [spk, ck, ik, fk, sclk, lsk2, lsk]
    kernels.sort()
    assert kernels == [lsk, lsk2, sclk, fk, ik, spk, ck]

    # Sort based on load priority
    spk1 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 1])
    spk2 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 2])
    spk3 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 3])
    spk4 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 4])
    spk5 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 5])
    spk6 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 6])

    kernels = [spk6, spk5, spk4, spk3, spk2, spk1]
    kernels.sort()
    assert kernels == [spk1, spk2, spk3, spk4, spk5, spk6]

    # Sort including release dates
    lsk1 = spicedb.KernelInfo(['LSK', '1', 'LSK', 'file', T0, T1, T9, 0, 9])
    spk0 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T0, 0, 9])
    spk2 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T2, 0, 2])
    spk3 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T3, 0, 3])
    spk4 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T4, 0, 4])
    spk5 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T0, T1, T5, 0, 5])

    # note--spk0 has the highest load priority
    kernels = [spk0, spk5, spk4, spk3, spk2, lsk1]
    kernels.sort()
    assert kernels == [lsk1, spk2, spk3, spk4, spk5, spk0]

    # Sort by name and version
    spk0 = spicedb.KernelInfo(['AA', '1', 'SPK', 'file', T0, T1, T9, 0, 1])
    spk1 = spicedb.KernelInfo(['AA', '2', 'SPK', 'file', T0, T1, T9, 0, 1])
    spk2 = spicedb.KernelInfo(['AA', '3', 'SPK', 'file', T0, T1, T9, 0, 1])
    spk3 = spicedb.KernelInfo(['BB', '3', 'SPK', 'file', T0, T1, T9, 0, 1])
    spk4 = spicedb.KernelInfo(['BB', '4', 'SPK', 'file', T0, T1, T9, 0, 1])
    spk5 = spicedb.KernelInfo(['CC', '4', 'SPK', 'file', T0, T1, T9, 0, 1])

    kernels = [spk5, spk4, spk3, spk2, spk1, spk0]
    kernels.sort()
    assert kernels == [spk0, spk1, spk2, spk3, spk4, spk5]

    # Sort by time ranges
    spk0 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T4, T7, T9, 0, 1])
    spk1 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T6, T7, T9, 0, 1])
    spk2 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T0, T4, T9, 0, 1])
    spk3 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T1, T4, T9, 0, 1])
    spk4 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T2, T4, T9, 0, 1])
    spk5 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file', T3, T4, T9, 0, 1])

    kernels = [spk0, spk1, spk2, spk3, spk4, spk5]
    kernels.sort()
    assert kernels == [spk5, spk4, spk3, spk2, spk1, spk0]

    # Sort by file name
    spk0 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file0', T0, T9, T9, 0, 1])
    spk1 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file1', T0, T9, T9, 0, 1])
    spk2 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file2', T0, T9, T9, 0, 1])
    spk3 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file3', T1, T9, T9, 0, 1])
    spk4 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file4', T0, T9, T9, 0, 1])
    spk5 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file5', T0, T9, T9, 0, 1])

    kernels = [spk5, spk4, spk3, spk2, spk1, spk0]
    kernels.sort()
    assert kernels == [spk3, spk0, spk1, spk2, spk4, spk5]

    # Sort by body ID
    spk0 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file1', T0, T9, T9, 0, 1])
    spk1 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file1', T0, T9, T9, 1, 1])
    spk2 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file1', T0, T9, T9, 2, 1])
    spk3 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file0', T0, T9, T9, 3, 1])
    spk4 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file0', T0, T9, T9, 4, 1])
    spk5 = spicedb.KernelInfo(['SPK', '1', 'SPK', 'file0', T0, T9, T9, 5, 1])

    kernels = [spk5, spk4, spk3, spk2, spk1, spk0]
    kernels.sort()
    assert kernels == [spk3, spk4, spk5, spk0, spk1, spk2]

    # Test full names
    spk = spicedb.KernelInfo(['VG1-JUP', '+230', 'SPK', 'file', T0, T1, T2, 0, 1])
    assert spk.full_name == 'VG1-JUP230'

    spk = spicedb.KernelInfo(['VG1', 'JUP230', 'SPK', 'file', T0, T1, T2, 0, 1])
    assert spk.full_name == 'VG1-JUP230'

    spk = spicedb.KernelInfo(['VG1-JUP230', None, 'SPK', 'file', T0, T1, T2, 0, 1])
    assert spk.full_name == 'VG1-JUP230'

################################################################################
# UNIT TESTS for queries
################################################################################

def test_spicedb():
    ############################################################################
    # _sort_kernels()
    ############################################################################

    # Leapseconds should always come first
    lsk0 = spicedb.KernelInfo(['LEAPSECONDS', '1', 'LSK', 'File0.tls',
                       '2000-01-01', '2000-01-02', '2000-01-03', None, 100])

    # Spacecraft clock should always come second
    # These kernels are ordered alphabetically
    sclk0 = spicedb.KernelInfo(['SCLK82', '1', 'SCLK', 'sclk-82.tsc',
                        '2000-01-01', '2000-01-02', '2003-01-03', -82, 100])

    sclk1 = spicedb.KernelInfo(['SCLK99', '1', 'SCLK', 'sclk-99.tsc',
                        '2000-01-01', '2000-01-02', '2003-01-03', -99, 100])

    # CKs come next alphabetically
    # Lowest load priority comes first, even with later release date
    ck0 = spicedb.KernelInfo(['CK-PREDICTED', '1', 'CK', 'File2.ck',
                      '2001-01-01', '2099-01-01', '2005-01-01', -82, 50])

    # Others are loaded in order of increasing end date
    ck1 = spicedb.KernelInfo(['CK-RECONSTRUCTED', '1', 'CK', 'File3.ck',
                      '2001-01-01', '2002-01-01', '2003-01-01', -82, 100])

    ck2 = spicedb.KernelInfo(['CK-RECONSTRUCTED', '1', 'CK', 'File4.ck',
                      '2002-01-01', '2003-01-01', '2004-01-01', -82, 100])

    random = [ck2, lsk0, ck1, sclk1, ck0, sclk0]

    sorted = [lsk0, sclk0, sclk1, ck0, ck1, ck2]
    assert spicedb._sort_kernels(random) == sorted

    # Frame and PC kernels
    # Ordered by priority, release date, version; with duplicates removed
    fk1 = spicedb.KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5a.fk',
                      None, None, '2004-01-01', 1, 100])
    fk2 = spicedb.KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5b.fk',
                      None, None, '2004-01-01', 2, 100])
    fk3 = spicedb.KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5c.fk',
                      None, None, '2004-01-01', 3, 100])

    # later release date, but only body 1
    fk4 = spicedb.KernelInfo(['FRAMES', 'BBBB', 'FK', 'File6a.fk',
                      None, None, '2005-01-01', 1, 100])

    random = [fk1, fk2, fk3, fk4]
    sorted = [fk2, fk3, fk4]
    assert spicedb._sort_kernels(random) == sorted

    # three bodies in one file
    fk1 = spicedb.KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5.fk',
                      None, None, '2004-01-01', 1, 100])
    fk2 = spicedb.KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5.fk',
                      None, None, '2004-01-01', 2, 100])
    fk3 = spicedb.KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5.fk',
                      None, None, '2004-01-01', 3, 100])

    # later release date, but only body 1
    fk4 = spicedb.KernelInfo(['FRAMES', 'BBBB', 'FK', 'File6a.fk',
                     None, None, '2005-01-01', 1, 100])

    random = [fk1, fk2, fk3, fk4]
    sorted = [fk3, fk4]
    assert spicedb._sort_kernels(random) == sorted

    # higher load priority
    fk1 = spicedb.KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5.fk',
                      None, None, '2004-01-01', 1, 150])
    fk2 = spicedb.KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5.fk',
                      None, None, '2004-01-01', 2, 150])
    fk3 = spicedb.KernelInfo(['FRAMES', 'CCCC', 'FK', 'File5.fk',
                      None, None, '2004-01-01', 3, 150])

    # later release date, but only body 1
    fk4 = spicedb.KernelInfo(['FRAMES', 'BBBB', 'FK', 'File6a.fk',
                      None, None, '2005-01-01', 1, 100])

    random = [fk1, fk2, fk3, fk4]
    sorted = [fk3]
    assert spicedb._sort_kernels(random) == sorted

    # SP Kernels
    # A low-priority predict kernel comes first
    spk1 = spicedb.KernelInfo(['SPK_PREDICTED', '1', 'SPK', 'predict.spk',
                 '2000-01-02', '2020-12-31', '2003-01-03', -82, 50])

    # These are duplicates and all but the last will be skipped
    spk2 = spicedb.KernelInfo(['SPK-RECONSTRUCTED', '1', 'SPK', 'recon.spk',
                 '2002-01-01', '2005-01-01', '2003-01-03', -82, 100])

    spk2a = spicedb.KernelInfo(['SPK-RECONSTRUCTED', '1', 'SPK', 'recon.spk',
                  '2002-01-01', '2005-01-01', '2003-01-03', 6, 100])

    spk2b = spicedb.KernelInfo(['SPK-RECONSTRUCTED', '1', 'SPK', 'recon.spk',
                  '2002-01-01', '2005-01-01', '2003-01-03', 601, 100])

    spk2c = spicedb.KernelInfo(['SPK-RECONSTRUCTED', '1', 'SPK', 'recon.spk',
                  '2002-01-01', '2005-01-01', '2003-01-03', 602, 100])

    spk2d = spicedb.KernelInfo(['SPK-RECONSTRUCTED', '1', 'SPK', 'recon.spk',
                  '2002-01-01', '2005-01-01', '2003-01-03', 699, 100])

    spk2e = spicedb.KernelInfo(['SPK-RECONSTRUCTED', '1', 'SPK', 'recon.spk',
                  '2002-01-01', '2005-01-01', '2003-01-03', 699, 100])

    # Another SPK, duplicated for three moons, alphabetically earlier
    spk3 = spicedb.KernelInfo(['SAT123','1',  'SPK', 'sat123.spk',
                 '1950-01-01', '2050-01-02', '2003-01-03', 619, 100])

    spk3a = spicedb.KernelInfo(['SAT123', '1', 'SPK', 'sat123.spk',
                  '1950-01-01', '2050-01-02', '2003-01-03', 635, 100])

    spk3b = spicedb.KernelInfo(['SAT123', '1', 'SPK', 'sat123.spk',
                  '1950-01-01', '2050-01-02', '2003-01-03', 636, 100])

    random = [spk1, spk2, spk2a, spk2b, spk2c, spk2d, spk2e, spk3, spk3a, spk3b]
    sorted = [spk1, spk3b, spk2e]
    assert spicedb._sort_kernels(random) == sorted

    # Put them all together in a random order
    random = [spk3, ck2, fk3, spk2d, ck0, spk2a, spk2b, fk1, fk2, spk2c, ck1,
              lsk0, spk2e, sclk1, sclk0, spk1, fk4, spk2, spk3b, spk3a]
    sorted = [lsk0, sclk0, sclk1, fk3, spk1, spk3b, spk2e, ck0, ck1, ck2]
    assert spicedb._sort_kernels(random) == sorted

    ############################################################################
    # _remove_overlaps()
    ############################################################################

    start_time = '2000-01-01T00:00:00'
    stop_time  = '2010-01-01T00:00:00'

    info0 = spicedb.KernelInfo(['SPK0', 'V1', 'SPK', '0000.spk',
                  '1950-01-01', '2050-01-01', '2003-01-01', 6, 100])

    info1 = spicedb.KernelInfo(['SPK1', 'V1', 'SPK', '1111.spk',
                  '1950-01-01', '2002-01-01', '2003-01-02', 6, 100])

    info2 = spicedb.KernelInfo(['SPK2', 'V1', 'SPK', '2222.spk',
                  '1950-01-01', '2003-01-01', '2003-02-01', 6, 100])

    info3 = spicedb.KernelInfo(['SPK3', 'V1', 'SPK', '3333.spk',
                  '2002-07-01', '2004-01-01', '2003-03-01', 6, 100])

    info4 = spicedb.KernelInfo(['SPK4', 'V1', 'SPK', '4444.spk',
                  '1950-01-01', '2002-07-01', '2003-04-01', 6, 100])

    info5 = spicedb.KernelInfo(['SPK5', 'V1', 'SPK', '5555.spk',
                  '2004-01-01', '2050-01-01', '2003-05-01', 6, 100])

    info6 = spicedb.KernelInfo(['SPK6', 'V1', 'SPK', '6666.spk',
                  '2004-01-01', '2004-07-01', '2003-06-01', 6, 100])

    info7 = spicedb.KernelInfo(['SPK7', 'V1', 'SPK', '7777.spk',
                  '1950-01-01', '2050-01-01', '2003-07-01', 6, 100])

    info8 = spicedb.KernelInfo(['SPK8', 'V1', 'SPK', '8888.spk',
                  '2004-07-01', '2050-01-01', '2003-08-01', 6, 100])

    spks = [info0, info1, info2, info3, info4, info5, info6, info7, info8]
    assert spicedb._remove_overlaps(spks, start_time, stop_time) == [info7, info8]

    spks = [info0, info1, info2, info3, info4, info5, info6, info7]
    assert spicedb._remove_overlaps(spks, start_time, stop_time) == [info7]

    spks = [info0, info1, info2, info3, info4, info5, info6, info8]
    assert (spicedb._remove_overlaps(spks, start_time, stop_time)
            == [info3, info4, info6, info8])

    spks = [info0, info1, info2, info3, info4, info5, info6]
    assert (spicedb._remove_overlaps(spks, start_time, stop_time)
            == [info3, info4, info5, info6])

    spks = [info0, info1, info2, info3, info4, info5]
    assert spicedb._remove_overlaps(spks, start_time, stop_time) == [info3, info4, info5]

    spks = [info0, info1, info2, info3, info4]
    assert spicedb._remove_overlaps(spks, start_time, stop_time) == [info0, info3, info4]

    spks = [info0, info1, info2, info3]
    assert spicedb._remove_overlaps(spks, start_time, stop_time) == [info0, info2, info3]

    spks = [info0, info1, info2]
    assert spicedb._remove_overlaps(spks, start_time, stop_time) == [info0, info2]

    spks = [info0, info1]
    assert spicedb._remove_overlaps(spks, start_time, stop_time) == [info0, info1]

    spks = [info0]
    assert spicedb._remove_overlaps(spks, start_time, stop_time) == [info0]

    ############################################################################
    ############################################################################
    # SPICE.db tests
    ############################################################################
    ############################################################################

    try:
        spicedb.get_spice_path()    # Find SPICE.db in the usual place
        spicedb.open_db()

        spicedb.DEBUG = True        # Avoid attempting to load kernels
        spicedb.ABSPATH_LIST = []

        ########################################################################
        # _query_kernels()
        ########################################################################

        kernels = spicedb._query_kernels('LSK')
        assert len(kernels) == 1

        kernels = spicedb._query_kernels('LSK', asof='2014-03-09')
        assert kernels[0].full_name == 'NAIF-LSK-0010'

        kernels = spicedb._query_kernels('LSK', asof=(14*365.25*86400))
        assert kernels[0].full_name == 'NAIF-LSK-0010'

        kernels = spicedb._query_kernels('LSK', asof='2010-01-01')
        assert kernels[0].full_name == 'NAIF-LSK-0009'

        with pytest.raises(ValueError):
            spicedb._query_kernels('LSK', asof='1950')

        with pytest.raises(ValueError):
            spicedb._query_kernels('LSK', after='3000')

        kernels = spicedb._query_kernels('LSK', asof='1950-01-01', redo=True)
        assert len(kernels) == 1
        assert kernels[0].full_name.startswith('NAIF-LSK-')

        kernels = spicedb._query_kernels('LSK', after='3000-01-01', redo=True)
        assert len(kernels) == 1
        assert kernels[0].full_name.startswith('NAIF-LSK-')

        kernels = spicedb._query_kernels('PCK', name='NAIF%')
        assert len(kernels) == 1
        assert kernels[0].full_name.startswith('NAIF-PCK-')

        kernels = spicedb._query_kernels('PCK', body=2)
        assert len(kernels) == 1           # Only NAIF PCKs have Venus
        assert kernels[0].full_name.startswith('NAIF-PCK-')

        kernels = spicedb._query_kernels('PCK', body=2, asof='2014')
        assert kernels[0].full_name == 'NAIF-PCK-00010'

        kernels = spicedb._query_kernels('PCK', body=(1,2,3), asof='2014')
        assert kernels[0].full_name == 'NAIF-PCK-00010'

        # Cassini CK tests
        kernels = spicedb._query_kernels('CK', body=-82, asof='2014',
                                        time=('2008-01-01','2008-02-01'),
                                        limit=False)
        for kernel in kernels[:-2]:
            assert kernel.full_name == 'CAS-CK-RECONSTRUCTED-V01'

        for kernel in kernels[-2:]:
            assert kernel.full_name == 'CAS-CK-PREDICTED-V01'

        assert len(kernels) == 9
        assert kernels[ 0].filespec.endswith('07362_08002ra.bc')
        assert kernels[-1].filespec.endswith('08022_08047pg_live.bc')

        # Cassini SPK tests
        kernels = spicedb._query_kernels('SPK', body=-82, asof='2014',
                                        time=('2008-01-01','2009-01-01'),
                                        limit=False)
        for kernel in kernels:
            assert kernel.full_name == 'CAS-SPK-RECONSTRUCTED-V01'

        assert len(kernels) == 13
        assert kernels[ 0].filespec.endswith( '080327R_SCPSE_07365_08045.bsp')
        assert kernels[-1].filespec.endswith( '090225R_SCPSE_08350_09028.bsp')

        ########################################################################
        # furnish_lsk(asof=None, after=None, redo=True)
        ########################################################################

        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_lsk(asof='2014')
        assert kernels == ['NAIF-LSK-0010']
        assert len(spicedb.ABSPATH_LIST) == 1
        assert spicedb.ABSPATH_LIST[0].name == 'naif0010.tls'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        k1 = kernels
        kernels = spicedb.furnish_lsk(asof=(14*365.25*86400))
        assert kernels == ['NAIF-LSK-0010']
        assert spicedb.ABSPATH_LIST[0].name == 'naif0010.tls'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_lsk(asof='2010-01-01')
        assert kernels == ['NAIF-LSK-0009']
        assert spicedb.ABSPATH_LIST[0].name == 'naif0009.tls'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        with pytest.raises(ValueError):
            spicedb.furnish_lsk(asof='1950', redo=False)

        with pytest.raises(ValueError):
            spicedb.furnish_lsk(after='3000', redo=False)

        latest = spicedb.furnish_lsk()

        kernels = spicedb.furnish_lsk(asof='1950-01-01', redo=True)
        assert kernels == latest

        kernels = spicedb.furnish_lsk(after='3000-01-01', redo=True)
        assert kernels == latest

        ########################################################################
        # furnish_pck(bodies, asof=None, after=None, redo=True)
        ########################################################################

        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_pck()
        naif_pck_mars = -1
        naif_pck = -1
        for (i,kernel) in enumerate(kernels):
            if kernel.startswith('NAIF-PCK-MARS-'):
                naif_pck_mars = i
            if kernel.startswith('NAIF-PCK-00'):
                naif_pck = i

        assert naif_pck >= 0
        assert naif_pck_mars >= 0

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_pck(bodies=spicedb.lrange(1,11) +
                                     spicedb.lrange(199,1000,100) + [301] +
                                     spicedb.lrange(401,403) + spicedb.lrange(501,517) +
                                     spicedb.lrange(601,654) + spicedb.lrange(701,716) +
                                     spicedb.lrange(801,808) + spicedb.lrange(901,906) +
                                     [814,65035,65040,65041,65045,65048,65050],
                              asof='2014-03-10')

        assert (kernels
                == ['NAIF-PCK-MARS-IAU2000-V0', 'CAS-FK-ROCKS-V18', 'CAS-PCK-ROCKS-2011-01-21', 'CAS-PCK-2014-02-19', 'NAIF-PCK-00010-EDIT-V01'])

        assert spicedb.ABSPATH_LIST[0].name == 'mars_iau2000_v0.tpc'
        assert spicedb.ABSPATH_LIST[1].name == 'cas_rocks_v18.tf'
        assert spicedb.ABSPATH_LIST[2].name == 'cpck_rock_21Jan2011_merged.tpc'
        assert spicedb.ABSPATH_LIST[3].name == 'cpck19Feb2014.tpc'
        assert spicedb.ABSPATH_LIST[4].name == 'pck00010_edit_v01.tpc'

        ########################################################################
        # furnish_spk(bodies, time=None, asof=None, after=None, redo=True)
        ########################################################################

        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_spk([1,2,3,4,5,6,7,8,9], asof='2014-03-10')
        assert kernels == ['DE430']
        assert len(spicedb.ABSPATH_LIST) == 1
        assert spicedb.ABSPATH_LIST[0].name == 'de430.bsp'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_spk([699], asof='2014-03-10')
        assert kernels == ['SAT363', 'DE430'] #####
        assert len(spicedb.ABSPATH_LIST) == 2
        assert spicedb.ABSPATH_LIST[0].name == 'sat363.bsp'
        assert spicedb.ABSPATH_LIST[1].name == 'de430.bsp'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_spk(range(601,654), asof='2014-03-10')
        assert kernels == ['SAT357', 'SAT360', 'SAT362', 'SAT363', 'DE430']

        # Only SAT357-rocks is loaded the first time, not SAT357
        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels

        for file in F1:
            assert file in spicedb.ABSPATH_LIST

        ########
        with pytest.raises(ValueError):
            spicedb.furnish_spk([601], after='3000-01-01', redo=False)

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_spk([601], after='3000-01-01', redo=True)
        assert kernels[0][:3] == 'SAT'
        assert kernels[0][3:] >= '360'
        assert kernels[1][:2] == 'DE'
        assert kernels[1][2:] >= '430'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_spk([-82], asof='2013-12-13', time=None)
        assert kernels == ['CAS-SPK-RECONSTRUCTED-V01[1-133]']
        assert len(spicedb.ABSPATH_LIST) == 133
        assert spicedb.ABSPATH_LIST[ 0].name == '000331R_SK_LP0_V1P32.bsp'
        assert spicedb.ABSPATH_LIST[-1].name == '131212R_SCPSE_13273_13314.bsp'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_spk([-82], asof='2014-03-10',
                                     time=(12*365.25*86400., '2012-04-01'))
        assert kernels == ['CAS-SPK-RECONSTRUCTED-V01[114-117]']
        assert len(spicedb.ABSPATH_LIST) == 4
        assert spicedb.ABSPATH_LIST[0].name == '120227R_SCPSE_11357_12016.bsp'
        assert spicedb.ABSPATH_LIST[1].name == '120312R_SCPSE_12016_12042.bsp'
        assert spicedb.ABSPATH_LIST[2].name == '120416R_SCPSE_12042_12077.bsp'
        assert spicedb.ABSPATH_LIST[3].name == '120426R_SCPSE_12077_12098.bsp'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1, time=(12*365.25*86400., '2012-04-01'))
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_spk([-82], asof='2012-04-01',
                                     time=(12*365.25*86400., '2012-04-01'))
        assert (kernels
                == ['CAS-SPK-PREDICTED-2011-08-18', 'CAS-SPK-RECONSTRUCTED-V01[114,115]'])
        assert len(spicedb.ABSPATH_LIST) == 3
        assert spicedb.ABSPATH_LIST[0].name == '110818AP_SCPSE_11175_17265.bsp'
        assert spicedb.ABSPATH_LIST[1].name == '120227R_SCPSE_11357_12016.bsp'
        assert spicedb.ABSPATH_LIST[2].name == '120312R_SCPSE_12016_12042.bsp'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1, time=(12*365.25*86400., '2012-04-01'))
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_spk([-82], asof='2011-09-01',
                                     time=(12*365.25*86400., '2012-04-01'))
        assert kernels == ['CAS-SPK-PREDICTED-2011-08-18']
        assert len(spicedb.ABSPATH_LIST) == 1
        assert spicedb.ABSPATH_LIST[0].name == '110818AP_SCPSE_11175_17265.bsp'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1, time=(12*365.25*86400., '2012-04-01'))
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_spk([-82], asof='2011-08-01',
                                     time=(12*365.25*86400., '2012-04-01'))
        assert kernels == ['CAS-SPK-PREDICTED-2009-10-05']
        assert len(spicedb.ABSPATH_LIST) == 1
        assert spicedb.ABSPATH_LIST[0].name == '091005AP_SCPSE_09248_17265.bsp'

        with pytest.raises(ValueError):
            spicedb.furnish_spk([-82], after='3000-01-01', redo=False)

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1, time=(12*365.25*86400., '2012-04-01'))
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_spk([-32,601,699], asof='2014-08-01',
                              time=('1981-08-14', '1981-08-24'))
        assert kernels == ['SAT360', 'SAT363', 'VG2-SPK-SAT337', 'DE432']
        assert len(spicedb.ABSPATH_LIST) == 5
        assert spicedb.ABSPATH_LIST[-3].name == 'sat337.bsp'
        assert spicedb.ABSPATH_LIST[-2].name == 'vgr2_sat337.bsp'
        assert 'de432' in str(spicedb.ABSPATH_LIST[-1])

        with pytest.raises(ValueError):
            spicedb.furnish_spk([-82], after='3000-01-01', redo=False)

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1, time=('1981-08-14', '1981-08-24'))
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        sl9 = spicedb.lrange(1000181,1000189) + [1000190,1000191] + \
              spicedb.lrange(1000193,1000204)
        kernels = spicedb.furnish_spk(sl9, asof='2014-08-01')
        assert kernels == ['SL9-SPK-DE403']
        assert len(spicedb.ABSPATH_LIST) == len(sl9) + 1
        assert spicedb.ABSPATH_LIST[-1].name == 'de403.bsp'
        for file in spicedb.ABSPATH_LIST[:-1]:
            assert file.name.endswith('_1992-1994.gst.DE403.bsp')

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########################################################################
        # furnish_inst(ids, inst=None, asof=None, after=None, redo=True)
        ########################################################################

        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_inst(-82, inst=[], asof='2017-06-01')
        assert kernels == ['CAS-SCLK-00171', 'CAS-FK-V04']
        assert len(spicedb.ABSPATH_LIST) == 3
        assert spicedb.ABSPATH_LIST[0].name == 'cas00171.tsc'
        assert spicedb.ABSPATH_LIST[1].name == 'cas_v40.tf'
        assert spicedb.ABSPATH_LIST[2].name == 'cas_status_v04.tf'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_inst(-82, inst='ISS', asof='2017-06-01')
        assert kernels == ['CAS-SCLK-00171', 'CAS-FK-V04', 'CAS-IK-ISS-V10']
        assert len(spicedb.ABSPATH_LIST) == 4
        assert spicedb.ABSPATH_LIST[0].name == 'cas00171.tsc'
        assert spicedb.ABSPATH_LIST[1].name == 'cas_v40.tf'
        assert spicedb.ABSPATH_LIST[2].name == 'cas_status_v04.tf'
        assert spicedb.ABSPATH_LIST[3].name == 'cas_iss_v10.ti'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_inst(-82, inst=None, asof='2017-06-01')
        for file in spicedb.ABSPATH_LIST[3:]:      # skip over one .tsc and two .tf files
            assert file.name.endswith('.ti')

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_inst(-31, inst='ISS', asof='2017-06-01')
        assert (kernels
                == ['VG1-SCLK-00019', 'VG1-FK-V02', 'VG1-IK-ISSNA-V02', 'VG1-IK-ISSWA-V01'])

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########################################################################
        # furnish_ck(ids, name=None, time=None, asof=None, after=None, redo=True)
        ########################################################################

        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_ck(-82, name="%PREDICTED%")
        assert kernels[0] == 'CAS-CK-PREDICTED-V01[1]'
        assert kernels[1], 'CAS-CK-PREDICTED-V02[1-104]'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_ck(-82, name="%PREDICTED%",
                                  time=('2004-02-01','2018-01-01'))
        assert kernels[0], 'CAS-CK-PREDICTED-V02[1-104]'

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_ck(-82, asof='2014-02-18', name="%PREDICTED%")
        assert kernels == ['CAS-CK-PREDICTED-V01[1-81]']
        assert len(spicedb.ABSPATH_LIST) == 81

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_ck(-82, time=('2005-01-01','2005-02-01'),
                                  asof='2014-01-01')
        assert kernels == ['CAS-CK-RECONSTRUCTED-V01[69-75]']
        assert len(spicedb.ABSPATH_LIST) == 7

        filenames = list(spicedb.ABSPATH_LIST)
        filenames.sort()
        assert filenames[0].name == '04361_05002ra.bc'
        assert filenames[1].name == '05002_05007ra.bc'
        assert filenames[2].name == '05007_05012ra.bc'
        assert filenames[3].name == '05012_05017ra.bc'
        assert filenames[4].name == '05017_05022ra.bc'
        assert filenames[5].name == '05022_05027ra.bc'
        assert filenames[6].name == '05027_05032ra.bc'

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_ck(-82, time=('2005-01-01','2005-02-01'),
                                  asof='2014-01-01', name="%PREDICTED%")
        assert kernels == ['CAS-CK-PREDICTED-V01[10,11]']
        assert len(spicedb.ABSPATH_LIST) == 2
        assert spicedb.ABSPATH_LIST[0].name == '04351_05022ph_fsiv.bc'
        assert spicedb.ABSPATH_LIST[1].name == '05022_05058pj_fsiv.bc'

        F1 = spicedb.ABSPATH_LIST
        k1 = ['CAS-CK-PREDICTED-V01[9-20]']
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1, time=('2005-01-01','2005-02-01'))
        assert F1 == spicedb.ABSPATH_LIST

        assert len(spicedb.ABSPATH_LIST) == 2
        assert spicedb.ABSPATH_LIST[0].name == '04351_05022ph_fsiv.bc'
        assert spicedb.ABSPATH_LIST[1].name == '05022_05058pj_fsiv.bc'

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_ck(-32, asof='2014-01-01')
        assert 'VG2-CK-ISS-JUP-V01' in kernels
        assert 'VG2-CK-ISS-SAT-V01' in kernels
        assert 'VG2-CK-ISS-URA-V01' in kernels
        assert 'VG2-CK-ISS-NEP-V01' in kernels

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_ck(-98, asof='2015-06-01')
        assert kernels == ['NH-CK-RECONSTRUCTED-V01[1-9]']

        F1 = spicedb.ABSPATH_LIST
        k1 = kernels
        spicedb.ABSPATH_LIST = []
        kernels = spicedb.furnish_by_name(k1)
        assert k1 == kernels
        assert F1 == spicedb.ABSPATH_LIST

        ########################################################################
        # DEBUG mode off...
        ########################################################################

        spicedb.DEBUG = False

        ########################################################################
        # furnish_solar_system(start_time, stop_time, asof=None)
        # unload_by_name(names)
        # unload_by_type(types=None)
        ########################################################################

        kernels = spicedb.furnish_solar_system('2000-01-01', '2020-01-01',
                                       asof='2014-03-10')

        assert 'NAIF-LSK-0010' in kernels[0:1]
        assert 'NAIF-PCK-MARS-IAU2000-V0' in kernels[1:6]
        assert 'NAIF-PCK-00010-EDIT-V01' in kernels[1:6]
        assert 'CAS-FK-ROCKS-V18' in kernels[1:6]
        assert 'CAS-PCK-ROCKS-2011-01-21' in kernels[1:6]
        assert 'CAS-PCK-2014-02-19' in kernels[1:6]
        assert 'MAR097' in kernels[6:-1]
        assert 'JUP300' in kernels[6:-1]
        assert 'JUP310' in kernels[6:-1]
        assert 'SAT357' in kernels[6:-1]
        assert 'SAT360' in kernels[6:-1]
        assert 'SAT362' in kernels[6:-1]
        assert 'SAT363' in kernels[6:-1]
        assert 'URA091' in kernels[6:-1]
        assert 'URA111' in kernels[6:-1]
        assert 'URA112' in kernels[6:-1]
        assert 'NEP077' in kernels[6:-1]
        assert 'NEP081' in kernels[6:-1]
        assert 'NEP086' in kernels[6:-1]
        assert 'NEP087' in kernels[6:-1]
        assert 'PLU043' in kernels[6:-1]
        assert 'DE430' in kernels[-1:]

        assert len(kernels) == 22

        assert kernels.index('JUP300') < kernels.index('JUP310')
        assert kernels.index('SAT357') < kernels.index('SAT360')
        assert kernels.index('SAT360') < kernels.index('SAT362')
        assert kernels.index('SAT362') < kernels.index('SAT363')
        assert kernels.index('URA091') < kernels.index('URA111')
        assert kernels.index('URA111') < kernels.index('URA112')
        assert kernels.index('NEP077') < kernels.index('NEP081')
        assert kernels.index('NEP081') < kernels.index('NEP087')
        assert kernels.index('NEP087') < kernels.index('NEP086')
        # NEP087 < NEP086 because the latter has the later creation date

        assert len(spicedb.FURNISHED_ABSPATHS['LSK']) == 1
        assert len(spicedb.FURNISHED_ABSPATHS['PCK']) == 5
        assert len(spicedb.FURNISHED_ABSPATHS['SPK']) == 16
        assert len(spicedb.FURNISHED_NAMES['LSK']) == 1
        assert len(spicedb.FURNISHED_NAMES['PCK']) == 5
        assert len(spicedb.FURNISHED_NAMES['SPK']) == 16

        spicedb.unload_by_name(kernels[:6])
        assert len(spicedb.FURNISHED_ABSPATHS['LSK']) == 0
        assert len(spicedb.FURNISHED_ABSPATHS['PCK']) == 0
        assert len(spicedb.FURNISHED_ABSPATHS['SPK']) == 16
        assert len(spicedb.FURNISHED_NAMES['LSK']) == 0
        assert len(spicedb.FURNISHED_NAMES['PCK']) == 0
        assert len(spicedb.FURNISHED_NAMES['SPK']) == 16

        spicedb.unload_by_type('SPK')
        assert len(spicedb.FURNISHED_ABSPATHS['SPK']) == 0
        assert len(spicedb.FURNISHED_NAMES['SPK']) == 0

        ########
        kernels1 = spicedb.furnish_solar_system(asof='2014-03-10')  # no time limits
        assert kernels[:6] == kernels1[:6]     # Non-SPKs are the same
        assert kernels[-1] == 'DE430'

        assert kernels.index('JUP300') < kernels.index('JUP310')
        assert kernels.index('SAT357') < kernels.index('SAT360')
        assert kernels.index('SAT360') < kernels.index('SAT362')
        assert kernels.index('SAT362') < kernels.index('SAT363')
        assert kernels.index('URA091') < kernels.index('URA111')
        assert kernels.index('URA111') < kernels.index('URA112')
        assert kernels.index('NEP077') < kernels.index('NEP081')
        assert kernels.index('NEP081') < kernels.index('NEP087')
        assert kernels.index('NEP087') < kernels.index('NEP086')

        ########################################################################
        # furnish_cassini_kernels(start_time, stop_time, instrument=None,
        #                         asof=None)
        # unload_by_name(names)
        # unload_by_type(types=None)
        # furnished_names(types=None)
        ########################################################################

# Function disabled 2/22/2020
#         unload_by_type()
#         kernels = furnish_cassini_kernels('2010-01-01', '2010-04-01',
#                                           instrument='ISS', asof='2014-03-10')
#
#         self.assertIn('NAIF-LSK-0010', kernels[0:1])
#         self.assertIn('CAS-SCLK-00171', kernels[1:2])
#         self.assertIn('CAS-FK-V04', kernels[2:4])
#         self.assertIn('CAS-IK-ISS-V10', kernels[2:4])
#         self.assertIn('CAS-FK-ROCKS-V18', kernels[4:8])
#         self.assertIn('CAS-PCK-ROCKS-2011-01-21', kernels[4:8])
#         self.assertIn('CAS-PCK-2014-02-19', kernels[4:8])
#         self.assertIn('NAIF-PCK-00010-EDIT-V01', kernels[4:8])
#         self.assertIn('SAT357', kernels[8:12])
#         self.assertIn('SAT360', kernels[8:12])
#         self.assertIn('SAT362', kernels[8:12])
#         self.assertIn('SAT363', kernels[8:12])
#         self.assertIn('CAS-SPK-RECONSTRUCTED-V01[90-94]', kernels[12:13])
#         self.assertIn('DE430', kernels[13:14])
#         self.assertIn('CAS-CK-RECONSTRUCTED-V01[438-456]', kernels[-1:])
#
#         self.assertEqual(FURNISHED_NAMES['LSK'], ['NAIF-LSK-0010'])
#         self.assertEqual(FURNISHED_NAMES['SCLK'], ['CAS-SCLK-00171'])
#         self.assertEqual(FURNISHED_NAMES['FK'], ['CAS-FK-V04'])
#         self.assertEqual(FURNISHED_NAMES['IK'], ['CAS-IK-ISS-V10'])
#
#         self.assertIn('CAS-FK-ROCKS-V18', FURNISHED_NAMES['PCK'])
#         self.assertIn('CAS-PCK-ROCKS-2011-01-21', FURNISHED_NAMES['PCK'])
#         self.assertIn('CAS-PCK-2014-02-19', FURNISHED_NAMES['PCK'])
#         self.assertIn('NAIF-PCK-00010-EDIT-V01', FURNISHED_NAMES['PCK'])
#         self.assertEqual(len(FURNISHED_ABSPATHS['PCK']), 4)
#
#         self.assertIn('SAT357', FURNISHED_NAMES['SPK'][:4])
#         self.assertIn('SAT360', FURNISHED_NAMES['SPK'][:4])
#         self.assertIn('SAT362', FURNISHED_NAMES['SPK'][:4])
#         self.assertIn('SAT363', FURNISHED_NAMES['SPK'][:4])
#         self.assertIn('CAS-SPK-RECONSTRUCTED-V01', FURNISHED_NAMES['SPK'][4:5])
#         self.assertIn('DE430', FURNISHED_NAMES['SPK'][5:6])
#         self.assertEqual(FURNISHED_FILENOS['CAS-SPK-RECONSTRUCTED-V01'],
#                          range(90,95))
#         self.assertEqual(len(FURNISHED_ABSPATHS['SPK']), 5 + 5)
#
#         self.assertEqual(FURNISHED_NAMES['CK'], ['CAS-CK-PREDICTED-V01',
#                                                  'CAS-CK-RECONSTRUCTED-V01'])
#         self.assertEqual(FURNISHED_FILENOS['CAS-CK-PREDICTED-V01'],
#                          range(59,62))
#         self.assertEqual(FURNISHED_FILENOS['CAS-CK-RECONSTRUCTED-V01'],
#                          range(438,457))
#         self.assertEqual(len(FURNISHED_ABSPATHS['CK']), 457 - 438 + 62 - 59)
#
#         ########
#         kernels1 = furnish_cassini_kernels('2010-03-01', '2010-06-01',
#                                           instrument='VIMS', asof='2014-03-10')
#
#         self.assertIn('NAIF-LSK-0010', kernels1[0:1])
#         self.assertIn('CAS-SCLK-00171', kernels1[1:2])
#         self.assertIn('CAS-FK-V04', kernels1[2:4])
#         self.assertIn('CAS-IK-VIMS-V06', kernels1[2:4])
#         self.assertIn('CAS-FK-ROCKS-V18', kernels1[4:8])
#         self.assertIn('CAS-PCK-ROCKS-2011-01-21', kernels1[4:8])
#         self.assertIn('CAS-PCK-2014-02-19', kernels1[4:8])
#         self.assertIn('NAIF-PCK-00010-EDIT-V01', kernels1[4:8])
#         self.assertIn('SAT357', kernels1[8:12])
#         self.assertIn('SAT360', kernels1[8:12])
#         self.assertIn('SAT362', kernels1[8:12])
#         self.assertIn('SAT363', kernels1[8:12])
#         self.assertIn('CAS-SPK-RECONSTRUCTED-V01[93-97]', kernels1[12:13])
#         self.assertIn('DE430', kernels1[13:14])
#         self.assertIn('CAS-CK-RECONSTRUCTED-V01[450-468]', kernels1[-1:])
#
#         self.assertEqual(FURNISHED_NAMES['LSK'], ['NAIF-LSK-0010'])
#         self.assertEqual(FURNISHED_NAMES['SCLK'], ['CAS-SCLK-00171'])
#         self.assertEqual(FURNISHED_NAMES['FK'], ['CAS-FK-V04'])
#         self.assertEqual(FURNISHED_NAMES['IK'], ['CAS-IK-ISS-V10',
#                                                  'CAS-IK-VIMS-V06'])
#
#         self.assertIn('CAS-FK-ROCKS-V18', FURNISHED_NAMES['PCK'])
#         self.assertIn('CAS-PCK-ROCKS-2011-01-21', FURNISHED_NAMES['PCK'])
#         self.assertIn('CAS-PCK-2014-02-19', FURNISHED_NAMES['PCK'])
#         self.assertIn('NAIF-PCK-00010-EDIT-V01', FURNISHED_NAMES['PCK'])
#         self.assertEqual(len(FURNISHED_ABSPATHS['PCK']), 4)
#
#         self.assertIn('SAT357', FURNISHED_NAMES['SPK'][:4])
#         self.assertIn('SAT360', FURNISHED_NAMES['SPK'][:4])
#         self.assertIn('SAT362', FURNISHED_NAMES['SPK'][:4])
#         self.assertIn('SAT363', FURNISHED_NAMES['SPK'][:4])
#         self.assertIn('CAS-SPK-RECONSTRUCTED-V01', FURNISHED_NAMES['SPK'][4:5])
#         self.assertIn('DE430', FURNISHED_NAMES['SPK'][5:6])
#         self.assertEqual(FURNISHED_FILENOS['CAS-SPK-RECONSTRUCTED-V01'],
#                          range(90,98))
#         self.assertEqual(len(FURNISHED_ABSPATHS['SPK']), 8 + 5)
#
#         self.assertEqual(FURNISHED_NAMES['CK'], ['CAS-CK-PREDICTED-V01',
#                                                  'CAS-CK-RECONSTRUCTED-V01'])
#         self.assertEqual(FURNISHED_FILENOS['CAS-CK-PREDICTED-V01'],
#                          range(59,64))
#         self.assertEqual(FURNISHED_FILENOS['CAS-CK-RECONSTRUCTED-V01'],
#                          range(438,469))
#         self.assertEqual(len(FURNISHED_ABSPATHS['CK']), 469 - 438 + 64 - 59)
#         # SPK and CK file_no lists get merged
#
#         ########
#         self.assertEqual(furnished_names('CK'),
#                          ['CAS-CK-PREDICTED-V01[59-63]',
#                           'CAS-CK-RECONSTRUCTED-V01[438-468]'])
#
#         unload_by_name('CAS-CK-RECONSTRUCTED-V01[440]')
#
#         self.assertEqual(furnished_names('CK'),
#                          ['CAS-CK-PREDICTED-V01[59-63]',
#                           'CAS-CK-RECONSTRUCTED-V01[438,439,441-468]'])
#         self.assertEqual(FURNISHED_FILENOS['CAS-CK-RECONSTRUCTED-V01'],
#                          [438,439] + range(441,469))
#
#         unload_by_name('CAS-CK-RECONSTRUCTED-V01[1-438]')
#
#         self.assertEqual(furnished_names('CK'),
#                          ['CAS-CK-PREDICTED-V01[59-63]',
#                           'CAS-CK-RECONSTRUCTED-V01[439,441-468]'])
#         self.assertEqual(FURNISHED_FILENOS['CAS-CK-RECONSTRUCTED-V01'],
#                          [439] + range(441,469))
#
#         unload_by_name('CAS-CK-RECONSTRUCTED-V01[439,442-465]')
#
#         self.assertEqual(furnished_names('CK'),
#                          ['CAS-CK-PREDICTED-V01[59-63]',
#                           'CAS-CK-RECONSTRUCTED-V01[441,466-468]'])
#         self.assertEqual(FURNISHED_FILENOS['CAS-CK-RECONSTRUCTED-V01'],
#                          [441] + range(466,469))
#
#         unload_by_name('CAS-CK-RECONSTRUCTED-V01[441-468]')
#
#         self.assertEqual(furnished_names('CK'), ['CAS-CK-PREDICTED-V01[59-63]'])
#         self.assertNotIn('CAS-CK-RECONSTRUCTED-V01', FURNISHED_FILENOS)
#
#         self.assertEqual(furnished_names('SPK'),
#                          ['SAT357', 'SAT360', 'SAT362', 'SAT363',
#                           'CAS-SPK-RECONSTRUCTED-V01[90-97]', 'DE430'])
#
#         self.assertEqual(furnished_names(['IK','FK','LSK','SCLK']),
#                          ['CAS-IK-ISS-V10', 'CAS-IK-VIMS-V06',
#                           'CAS-FK-V04', 'NAIF-LSK-0010', 'CAS-SCLK-00171'])

        ########################################################################
        # Test translator
        ########################################################################

        spicedb.unload_by_type()

        spicedb.DEBUG = True

        # Function to translate Cassini SPKs, adding "_testing" before suffix
        # and replacing the "reconstructed" directory with "my_testing"
        def translator(filepath):
            filepath = Path(filepath)
            s_filepath = str(filepath)
            if s_filepath.endswith('.bsp') and 'RECONSTRUCTED' in s_filepath.upper():
                parts = filepath.parts
                parts = [p if 'RECONSTRUCTED' not in p.upper() else 'my_testing'
                         for p in parts]
                new_parts = list(parts[:-1]) + [parts[-1][:-4] + '_testing.bsp']
                ret = Path(*new_parts)
                return ret
            return filepath

        # Translator will not affect solar system kernels
        spicedb.ABSPATH_LIST = []
        kernels1 = spicedb.furnish_solar_system('2000-01-01', '2020-01-01',
                                        asof='2014-03-10')
        abspaths1 = set(spicedb.ABSPATH_LIST)
        spicedb.unload_by_type()

        spicedb.set_translator(translator)
        spicedb.ABSPATH_LIST = []
        kernels2 = spicedb.furnish_solar_system('2000-01-01', '2020-01-01',
                                        asof='2014-03-10')
        abspaths2 = set(spicedb.ABSPATH_LIST)
        spicedb.unload_by_type()

        assert kernels1 == kernels2
        assert abspaths1 == abspaths2

        # Translator will change Cassini SPKs
        spicedb.set_translator(None)
        spicedb.ABSPATH_LIST = []
        kernels1 = spicedb.furnish_cassini_kernels('2010-01-01', '2010-04-01',
                                           instrument='ISS', asof='2014-03-10')
        abspaths1 = set(spicedb.ABSPATH_LIST)
        spicedb.unload_by_type()

        spicedb.set_translator(translator)
        spicedb.ABSPATH_LIST = []
        kernels2 = spicedb.furnish_cassini_kernels('2010-01-01', '2010-04-01',
                                           instrument='ISS', asof='2014-03-10')
        abspaths2 = set(spicedb.ABSPATH_LIST)
        spicedb.unload_by_type()

        assert kernels1 == kernels2

        translated = abspaths2 - abspaths1
        originals  = abspaths1 - abspaths2

        remainder1 = abspaths1 - originals
        remainder2 = abspaths2 - translated
        assert remainder1 == remainder2

        for abspath in originals:
            assert abspath.name.endswith('.bsp')
            assert 'CASSINI' in str(abspath).upper()

            newpath = translator(abspath)
            assert str(newpath) in [str(x) for x in translated]

        assert len(translated) == len(originals)

        # Function to replace all files "*.bc" with a blank string
        def translator2(filepath):
            if filepath.endswith('.bc') :
                return ''
            return filepath

        # Translator will eliminate all C kernels from list
        spicedb.set_translator(translator2)
        spicedb.ABSPATH_LIST = []
        kernels2 = spicedb.furnish_cassini_kernels('2010-01-01', '2010-04-01',
                                           instrument='ISS', asof='2014-03-10')
        abspaths2 = set(spicedb.ABSPATH_LIST)
        spicedb.unload_by_type()

        for abspath in abspaths2:
            if abspath.name.endswith('.bc'):
                assert abspath not in abspaths1
            else:
                assert abspath in abspaths1

    ############################################################################
    # Clean up...
    ############################################################################

    finally:
        spicedb.unload_by_type()

        spicedb.DEBUG = False
        spicedb.close_db()
################################################################################
