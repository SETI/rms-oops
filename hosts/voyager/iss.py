################################################################################
# hosts/voyager/iss.py
################################################################################

import cspyce
import spicedb
import oops
import julian
import vicar
import pdstable
import pdsparser
import numpy as np
import os
import warnings

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, astrometry=False, action='error', parameters={}):
    """A general, static method to return a Snapshot object based on a given
    Voyager ISS image file or its label.

    Input:
        filespec        name of the image file or its PDS3 label.
        astrometry      True to omit loading the image data.
        action          What to do for a missing C kernel entry, via the Python
                        warnings interface: 'error', 'ignore', 'always',
                        'default', 'module', 'once'.
    """
    ISS.initialize()    # Define everything the first time through

    # Load the PDS label if available
    if filespec.upper().endswith('.LBL'):
        label_dict = pdsparser.PdsLabel.from_file(filespec).as_dict()
        imagefile = label_dict['^IMAGE'][0]
        imagespec = os.path.join(os.path.split(filespec)[0], imagefile)
    else:
        (body,ext) = os.path.splitext(filespec)
        if ext == ext.upper():
            labelspec = body + '.LBL'
        else:
            labelspec = body + '.lbl'

        if os.path.exists(labelspec):
            label_dict = pdsparser.PdsLabel.from_file(labelspec).as_dict()
        else:
            label_dict = None

        imagespec = filespec

    # Load the VICAR file
    vicar_dict = label_dict
    if not astrometry:
        vic = vicar.VicarImage.from_file(imagespec)
        vicar_dict = vic.as_dict()

    # Get key information, preferably from the PDS label
    if label_dict is not None:
        stop_time = label_dict['STOP_TIME']
        texp = max(1.e-6, label_dict['EXPOSURE_DURATION'])

        try:
            vgr = label_dict['SPACECRAFT_NAME'][-1]
            label_dict['INSTRUMENT_HOST_NAME'] = label_dict['SPACECRAFT_NAME']
        except KeyError:
            vgr = label_dict['INSTRUMENT_HOST_NAME'][-1]
            label_dict['SPACECRAFT_NAME'] = label_dict['INSTRUMENT_HOST_NAME']

        ivgr = int(vgr)

        spacecraft = 'VOYAGER' + vgr
        camera = label_dict['INSTRUMENT_ID'][-1] + 'AC' # WAC or NAC
        planet = label_dict['MISSION_PHASE_NAME'][:-len(' ENCOUNTER')]
        target = label_dict['TARGET_NAME']
        filter = label_dict['FILTER_NAME']
        factor = label_dict['IMAGE']['REFLECTANCE_SCALING_FACTOR']
    else:
        lab02 = vicar_dict['LAB02']
        lab03 = vicar_dict['LAB03']
        stop_time = '19%s-%sT%s' % (lab02[47:49],lab02[50:53],lab02[54:62])
        texp = max(1.e-3, float(lab03[14:24])) / 1000.
        vgr = lab02[4]
        ivgr = int(lab02[4])
        spacecraft = 'VOYAGER' + vgr
        camera = lab03[0] + 'AC'

        if stop_time < '1980':
            planet = 'JUPITER'
        elif stop_time < '1983':
            planet = 'SATURN'
        elif stop_time < '1987':
            planet = 'URANUS'
        else:
            planet = 'NEPTUNE'

        target = vicar_dict['LAB05'][31:43].rstrip()
        target = target.replace('_', ' ')

        filter = lab03[37:43].rstrip()
        factor = None

    # Interpret the GEOMED parameter
    if 'GEOMA' in vic.get_values('TASK'):
        assert vic.data_2d.shape == (1000,1000)
        fovs = {
            'NAC': ISS.fovs['NAC_GEOMED'],
            'WAC': ISS.fovs['WAC_GEOMED'],
        }
    else:
        fovs = ISS.fovs

    # Get image time
    tai = julian.tai_from_iso(stop_time) - texp
    tstart = julian.tdb_from_tai(tai)

    # Get spacecraft clock ticks
    scid = -(30 + ivgr)
    start_ticks = cspyce.sce2t(scid, tstart)
    mid_ticks   = cspyce.sce2t(scid, tstart + texp/2.)
    stop_ticks  = cspyce.sce2t(scid, tstart + texp)

    # Construct the image coordinate frame
    scan_platform_id = scid * 1000 - 100
    tol_ticks = 800 + texp/48.

    with warnings.catch_warnings():
        warnings.simplefilter(action)

        try:
            (j2000_to_platform,
             found_ticks) = cspyce.ckgp(scan_platform_id, mid_ticks,
                                                          tol_ticks, 'J2000')
            platform_to_camera = cspyce.pxform('VG' + vgr + '_SCAN_PLATFORM',
                                               'VG' + vgr + '_ISS' + camera[:2],
                                               0.)
            image_frame = oops.frame.Cmatrix(oops.Matrix3(platform_to_camera) *
                                             oops.Matrix3(j2000_to_platform))

        except LookupError:
            warnings.warn('C kernel is unavailable for ' +
                          label_dict['PRODUCT_ID'], RuntimeWarning)
            image_frame = spacecraft + '_ISS_' + camera

    # Wrap the frame in a Navigation if necessary
    # - Use navigation=True in from_file() to enable a navigation frame without
    #   specifying the rotation angles initially.
    # - Use navigation=angles with a tuple of two or three angles to specify the
    #   initial pointing offset in radians.
    # - Use offset=(du,dv) to specify a pointing offset in units of pixels.
    offset = parameters.get('offset', None)
    if offset:
        navigation = fovs[camera].offset_angles_from_duv(offset)
    else:
        navigation = parameters.get('navigation', None)

    if navigation:
        if navigation is True:
            navigation = (0., 0.)
        image_frame = oops.frame.Navigation(navigation, image_frame,
                                            frame_id=image_frame.frame_id,
                                            override=True)

    # Create a Snapshot
    result = oops.obs.Snapshot(('v','u'), tstart, texp, fovs[camera],
                               path = spacecraft,
                               frame = image_frame,
                               dict = vicar_dict,           # Add the VICAR dict
                               data = vic.data_2d,          # Add the data array
                               instrument = 'ISS',
                               detector = camera,
                               filter = filter,
                               planet = planet,
                               target = target)

    if factor is not None:
        result.insert_subfield('extended_calib',
                               oops.calib.ExtendedSource('I/F', factor))

    return result

#===============================================================================
def from_index(filespec, geomed=False, action='ignore', omit=True,
               parameters={}):
    """A static method to return a list of Snapshot objects, one for each row
    in an ISS index file. The filespec refers to the label of the index file.

    Input:
        filespec        name of the image file or its PDS3 label.
        geomed          assume the image is geomed (1000x1000).
        action          What to do for a missing C kernel entry or a missing
                        time, via the Python warnings interface: 'error',
                        'ignore', 'always', 'default', 'module', 'once'.
        omit            True to remove any images with missing C kernels or
                        missing times from the returned list; False to include
                        them. If time is missing, tstart = 0.
    """
    ISS.initialize()    # Define everything the first time through

    # Read the index file
    COLUMNS = []                # Return all columns
    table = pdstable.PdsTable(filespec, columns=COLUMNS)
    row_dicts = table.dicts_by_row()

    # Interpret GEOMED parameter
    if geomed:
        fovs = {
            'NAC': ISS.fovs['NAC_GEOMED'],
            'WAC': ISS.fovs['WAC_GEOMED'],
        }
    else:
        fovs = ISS.fovs

    # Create a list of Snapshot objects
    with warnings.catch_warnings():
      warnings.simplefilter(action)

      snapshots = []
      for label_dict in row_dicts:

        try:
            vgr = label_dict['SPACECRAFT_NAME'][-1]
        except KeyError:
            vgr = label_dict['INSTRUMENT_HOST_NAME'][-1]

        ivgr = int(vgr)

        spacecraft = 'VOYAGER' + vgr
        planet = label_dict['MISSION_PHASE_NAME'][:-len(' ENCOUNTER')]
        target = label_dict['TARGET_NAME']

        if 'WIDE' in label_dict['INSTRUMENT_NAME']:
            camera = 'WAC'
        else:
            camera = 'NAC'

        # Get image time
        texp = label_dict['EXPOSURE_DURATION']
        if texp <= 1.e-6:
            texp = 1.e-6

        timestring = label_dict['IMAGE_TIME']
        if timestring == 'UNK':
            warnings.warn('Image time is unavailable for ' +
                          label_dict['PRODUCT_ID'], RuntimeWarning)
            if omit:
                continue

            tstart = 0.
        else:
            tai = julian.tai_from_iso(label_dict['IMAGE_TIME']) - texp
            tstart = julian.tdb_from_tai(tai)

        # Get spacecraft clock ticks
        scid = -(30 + ivgr)
        start_ticks = cspyce.sce2t(scid, tstart)
        mid_ticks   = cspyce.sce2t(scid, tstart + texp/2.)
        stop_ticks  = cspyce.sce2t(scid, tstart + texp)

        # Construct the image coordinate frame
        scan_platform_id = scid * 1000 - 100
        tol_ticks = 800 + texp/48.

        try:
            (j2000_to_platform,
             found_ticks) = cspyce.ckgp(scan_platform_id, mid_ticks,
                                                          tol_ticks, 'J2000')
            platform_to_camera = cspyce.pxform('VG' + vgr + '_SCAN_PLATFORM',
                                               'VG' + vgr + '_ISS' + camera[:2],
                                               0.)
            image_frame = oops.frame.Cmatrix(oops.Matrix3(platform_to_camera) *
                                             oops.Matrix3(j2000_to_platform))

        except (LookupError, IOError):
            warnings.warn('C kernel is unavailable for ' +
                          label_dict['PRODUCT_ID'], RuntimeWarning)
            if omit:
                continue

            image_frame = spacecraft + '_ISS_' + camera

        item = oops.obs.Snapshot(('v','u'), tstart, texp, fovs[camera],
                                 spacecraft,
                                 image_frame,
                                 dict = label_dict,     # Add index dictionary
                                 instrument = 'ISS',
                                 detector = camera,
                                 filter = filter,
                                 planet = planet,
                                 target = target)

        snapshots.append(item)

    return snapshots

################################################################################

class ISS(object):
    """An instance-free class to hold Voyager ISS instrument parameters."""

    fovs = {}
    frames = {}
    initialized = False

    @staticmethod
    def initialize(asof=None):
        """Fill in key information about the WAC and NAC.

        Must be called first.
        """

#         TOL_TICKS = 800.
        TOL_TICKS = 80000.  # needed to deal with very long exposures because
                            # C kernel defines end-time but frame is evaluated
                            # at mid-time.

        # Quick exit after first call
        if ISS.initialized:
            return

        # Check the formatting of the "as of" date
        if asof is not None:
            (day, sec) = julian.day_sec_from_iso(asof)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        # Define some important paths and frames
        oops.Body.define_solar_system('1979-01-06', '1989-10-02', asof=asof)

        # Check the formatting of the "as of" date
        if asof is not None:
            (day, sec) = julian.day_sec_from_iso(asof)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        # Furnish instruments and frames
        spicedb.open_db()

        _ = spicedb.furnish_inst(-31, asof=asof,)
        _ = spicedb.furnish_inst(-32, asof=asof)

        _ = spicedb.furnish_ck(-31, asof=asof)
        _ = spicedb.furnish_ck(-32, asof=asof)

        _ = spicedb.furnish_spk(-31, name='%JUP%', asof=asof)
        _ = spicedb.furnish_spk(-31, name='%SAT%', asof=asof)
        _ = spicedb.furnish_spk(-32, name='%JUP%', asof=asof)
        _ = spicedb.furnish_spk(-32, name='%SAT%', asof=asof)
        _ = spicedb.furnish_spk(-32, name='%URA%', asof=asof)
        _ = spicedb.furnish_spk(-32, name='%NEP%', asof=asof)

        spicedb.close_db()

        # Construct a flat FOV for the narrow angle camera, raw
        xfov = 0.003700098  # radians
        uscale = np.arctan(np.tan(xfov) / 400.)
        vscale = np.arctan(np.tan(xfov) / 400.)
        ISS.fovs['NAC'] = oops.fov.FlatFOV((uscale,vscale), (800,800))

        # Construct a flat FOV for the narrow angle camera, geomed
        xfov = 0.4493 / oops.DPR / 2.   # radians
        uscale = np.arctan(np.tan(xfov) / 500.)
        vscale = np.arctan(np.tan(xfov) / 500.)
        ISS.fovs['NAC_GEOMED'] = oops.fov.FlatFOV((uscale,vscale), (1000,1000))

        # Construct a flat FOV for the wide angle camera, raw
        xfov = 0.02765      # radians
        uscale = np.arctan(np.tan(xfov) / 400.)
        vscale = np.arctan(np.tan(xfov) / 400.)
        ISS.fovs['WAC'] = oops.fov.FlatFOV((uscale,vscale), (800,800))

        # Construct a flat FOV for the wide angle camera, raw
        xfov = 3.364 / oops.DPR / 2.    # radians
        uscale = np.arctan(np.tan(xfov) / 500.)
        vscale = np.arctan(np.tan(xfov) / 500.)
        ISS.fovs['WAC_GEOMED'] = oops.fov.FlatFOV((uscale,vscale), (1000,1000))

        # Construct the Voyager paths
        ignore = oops.path.SpicePath('VOYAGER 1', path_id='VOYAGER1')
        ignore = oops.path.SpicePath('VOYAGER 2', path_id='VOYAGER2')

        # Construct a SpiceType1Frame for each scan platform
        _ = oops.frame.SpiceType1Frame('VG1_SCAN_PLATFORM', -31, TOL_TICKS,
                                       frame_id='VOYAGER1_SCAN_PLATFORM')

        _ = oops.frame.SpiceType1Frame('VG2_SCAN_PLATFORM', -32, TOL_TICKS,
                                       frame_id='VOYAGER2_SCAN_PLATFORM')

        # Construct additional rotations for each camera
        matrix = cspyce.pxform('VG1_SCAN_PLATFORM', 'VG1_ISSNA', 0.)
        _ = oops.frame.Cmatrix(matrix, 'VOYAGER1_SCAN_PLATFORM',
                                       frame_id='VOYAGER1_ISS_NAC')

        matrix = cspyce.pxform('VG1_SCAN_PLATFORM', 'VG1_ISSWA', 0.)
        _ = oops.frame.Cmatrix(matrix, 'VOYAGER1_SCAN_PLATFORM',
                                       frame_id='VOYAGER1_ISS_WAC')

        matrix = cspyce.pxform('VG2_SCAN_PLATFORM', 'VG2_ISSNA', 0.)
        _ = oops.frame.Cmatrix(matrix, 'VOYAGER2_SCAN_PLATFORM',
                                       frame_id='VOYAGER2_ISS_NAC')

        matrix = cspyce.pxform('VG2_SCAN_PLATFORM', 'VG2_ISSWA', 0.)
        _ = oops.frame.Cmatrix(matrix, 'VOYAGER2_SCAN_PLATFORM',
                                       frame_id='VOYAGER2_ISS_WAC')

        ISS.initialized = True

################################################################################
# Initialize at load time
################################################################################

ISS.initialize()







################################################################################
# UNIT TESTS
################################################################################
#import unittest
#import os.path
#import oops.backplane.gold_master as gm
#
#from oops.unittester_support            import TESTDATA_PARENT_DIRECTORY
#
#
##===============================================================================
#class Test_VGR_ISS(unittest.TestCase):
#
#    #===========================================================================
#    def runTest(self):
#        pass
#
#
##===============================================================================
#class Test_VGR_ISS_GoldMaster_C3450201_GEOMED(unittest.TestCase):
#
#    #===========================================================================
#    def runTest(self):
#        """
#        C3450201_GEOMED Compare w Gold Masters
#
#        To preview and regenerate gold masters (from pds-oops/oops/backplane/):
#            python gold_master.py \
#                ~/Dropbox-SETI/OOPS-Resources/test_data/voyager/ISS/VGISS_6109/C34502XX/C3450201_GEOMED.img \
#                --module hosts.voyager.iss \
#                --planet SATURN \
#                --ring SATURN_MAIN_RINGS \
#                --no-inventory \
#                --preview
#
#            python gold_master.py \
#                ~/Dropbox-SETI/OOPS-Resources/test_data/voyager/ISS/VGISS_6109/C34502XX/C3450201_GEOMED.img \
#                --module hosts.voyager.iss \
#                --planet SATURN \
#                --ring SATURN_MAIN_RINGS \
#                --no-inventory \
#                --adopt
#        """
#        gm.execute_as_unittest(self,
#                obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
#                                       'voyager/ISS/VGISS_6109/C34502XX/'
#                                       'C3450201_GEOMED.img'),
#                index   = None,
#                module  = 'hosts.voyager.iss',
#                planet  = 'SATURN',
#                moon    = '',
#                ring    = 'SATURN_MAIN_RINGS',
#                inventory=False, border=10)
#
#
###############################################
#if __name__ == '__main__':
#    unittest.main(verbosity=2)







# import unittest
# import os.path
#
# from oops.unittester_support            import TESTDATA_PARENT_DIRECTORY
# from oops.backplane.exercise_backplanes import exercise_backplanes
# from oops.backplane.unittester_support  import Backplane_Settings
#
#
# #*******************************************************************************
# class Test_Voyager_ISS_Backplane_Exercises(unittest.TestCase):
#
#     def runTest(self):
#
#         if Backplane_Settings.NO_EXERCISES:
#             self.skipTest('')
#
#         root = os.path.join(TESTDATA_PARENT_DIRECTORY, 'voyager/ISS')
#         file = os.path.join(root, 'c3440346.gem')
#         obs = from_file(file)
#         exercise_backplanes(obs, use_inventory=True, inventory_border=4,
#                                  planet_key='SATURN')
#
# ##############################################
# from oops.backplane.unittester_support import backplane_unittester_args
#
# if __name__ == '__main__':
#     backplane_unittester_args()
#     unittest.main(verbosity=2)
################################################################################
