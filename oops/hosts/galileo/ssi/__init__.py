################################################################################
# oops/hosts/galileo/ssi/__init__.py
################################################################################
import sys
import os
import numpy as np
import julian
import cspyce
import vicar
import pdstable
import pdsparser
import oops

from oops.hosts         import pds3
from oops.hosts.galileo import Galileo

from filecache import FCPath


################################################################################
# Standard class methods
################################################################################
def from_file(filespec,
              return_all_planets=False, full_fov=False, **parameters):
    """A general, static method to return a Snapshot object based on a given
    Galileo SSI image file.  By default, only the valid image region is
    returned.

    Inputs:
        filespec            The full path to a Galileo SSI file or its PDS label.

        return_all_planets  Include kernels for all planets not just
                            Jupiter or Saturn.

        full_fov:           If True, the full image is returned with a mask
                            describing the regions with no data.
    """

    SSI.initialize()    # Define everything the first time through; use defaults
                        # unless initialize() is called explicitly.

    filespec = FCPath(filespec)

    # Load the PDS label
    label = pds3.get_label(filespec)

    # Load the data array
    local_path = filespec.retrieve()
    vic = vicar.VicarImage.from_file(local_path)
    vicar_dict = vic.as_dict()

    # Get image metadata
    meta = Metadata(label)

    # Define the field of view
    FOV = meta.fov(full_fov=full_fov)

    # Trim the image
    data = meta.trim(vic.data_2d, full_fov=full_fov)

    # Create a Snapshot
    result = oops.obs.Snapshot(('v','u'), meta.tstart, meta.exposure,
                               FOV,
                               path = 'GLL',
                               frame = 'GLL_SCAN_PLATFORM',
                               dict = vicar_dict,       # Add the VICAR dict
                               data = data,             # Add the data array
                               instrument = 'SSI',
                               filter = meta.filter,
                               filespec = filespec,
                               basename = filespec.name)

    result.insert_subfield('spice_kernels',
                           Galileo.used_kernels(result.time, 'ssi',
                                                return_all_planets))

    return result

#===============================================================================
def from_index(filespec, supplemental_filespec=None, full_fov=False, **parameters):
    """A static method to return a list of Snapshot objects.

    One object for each row in an SSI index file. The filespec refers to the
    label of the index file.
    """
    SSI.initialize()    # Define everything the first time through

    filespec = FCPath(filespec)

    # Read the index file
    COLUMNS = []        # Return all columns
    local_path = filespec.retrieve(filespec)
    table = pdstable.Pds3Table(local_path, columns=COLUMNS)
    row_dicts = table.dicts_by_row()

    # Read the supplemental index file
    if supplemental_filespec is not None:
        supplemental_filespec = FCPath(supplemental_filespec)
        supplemental_local_path = supplemental_filespec.retrieve()
        table = pdstable.PdsTable(supplemental_local_path)
        supplemental_row_dicts = table.dicts_by_row()

#        # Sort supplemental rows to match index file
#        specs = [os.path.splitext(row_dict['FILE_SPECIFICATION_NAME'])[0] for row_dict in row_dicts]
#        supplemental_specs = \
#            [os.path.splitext(supplemental_row_dict['FILE_SPECIFICATION_NAME'])[0] \
#             for supplemental_row_dict in supplemental_row_dicts]

#        indices = np.argsort(specs)

#        row_dicts_sorted = [None]*len(row_dicts)
#        for i in range(len(row_dicts)):
#            row_dicts_sorted[i] = row_dicts[indices[i]]

#        supplemental_indices = np.argsort(supplemental_specs)
#        supplemental_row_dicts_sorted = [None]*len(supplemental_row_dicts)
#        for i in range(len(supplemental_row_dicts)):
#            supplemental_row_dicts_sorted[i] = supplemental_row_dicts[supplemental_indices[i]]

        # Append supplemental columns to index file
        for row_dict, supplemental_row_dict in zip(row_dicts, supplemental_row_dicts):
            row_dict.update(supplemental_row_dict)

    # Create a list of Snapshot objects
    snapshots = []
    for row_dict in row_dicts:
        file = row_dict['FILE_SPECIFICATION_NAME']

        # Get image metadata; do not return observations with zero exposures
        meta = Metadata(row_dict)

        if meta.exposure == 0:
            continue

        # Define the field of view
        FOV = meta.fov(full_fov=full_fov)

        # Create a Snapshot
        basename = os.path.basename(file)
        item = oops.obs.Snapshot(('v','u'), meta.tstart, meta.exposure,
                                 FOV,
                                 path = 'GLL',
                                 frame = 'GLL_SCAN_PLATFORM',
                                 dict = row_dict,         # Add the index dict
                                 instrument = 'SSI',
                                 filter = meta.filter,
                                 filespec = file,
                                 basename=basename)

        item.spice_kernels = Galileo.used_kernels(item.time, 'ssi')

        item.filespec = (row_dict['VOLUME_ID'] + '/' +
                         row_dict['FILE_SPECIFICATION_NAME'])
        item.basename = basename

        snapshots.append(item)

    return snapshots

#===============================================================================
def initialize(planets=None, asof=None,
               mst_pck=True, irregulars=True):
    """Initialize key information about the SSI instrument.

    Must be called first. After the first call, later calls to this function
    are ignored.

    Input:
        planets     A list of planets to pass to define_solar_system. None or
                    0 means all.
        asof        Only use SPICE kernels that existed before this date; None
                    to ignore.
        mst_pck     True to include MST PCKs, which update the rotation models
                    for some of the small moons.
        irregulars  True to include the irregular satellites;
                    False otherwise.
    """
    SSI.initialize(planets=planets, asof=asof,
                   mst_pck=mst_pck, irregulars=irregulars)


#===============================================================================
class Metadata(object):

    #===========================================================================
    def __init__(self, meta_dict):
        """Use the label or index dict to assemble the image metadata."""

        info = SSI.instrument_kernel['INS'][-77036]

        # Image dimensions
        self.nlines = info['MAX_LINE']
        self.nsamples = info['MAX_SAMPLE']

        # Exposure time
        exposure_ms = meta_dict['EXPOSURE_DURATION']
        self.exposure = exposure_ms/1000.

        # Filters
        self.filter = meta_dict['FILTER_NAME']

        #TODO: determine whether IMAGE_TIME is the start time or the mid time..
        if meta_dict['IMAGE_TIME'] == 'UNK':
            self.tstart = self.tstop = sys.float_info.min
        else:
            self.tstart = julian.tdb_from_tai(
                            julian.tai_from_iso(meta_dict['IMAGE_TIME']))
            self.tstop = self.tstart + self.exposure

        # Target
        self.target = meta_dict['TARGET_NAME']

        # Telemetry mode
        self.mode = meta_dict['TELEMETRY_FORMAT_ID']

        # Window
        self.window = None
        if 'CUT_OUT_WINDOW' in meta_dict:
            window = np.array(meta_dict['CUT_OUT_WINDOW'])

            # check for [-1,-1,-1,-1].  This is the value written in the
            # supplemental index when there is no CUT_OUT_WINDOW in the label.
            if window.tolist() != [-1,-1,-1,-1]:
                self.window = window
                self.window_origin = self.window[0:2]-1
                self.window_shape = self.window[2:]
                self.window_uv_origin = np.flip(self.window_origin)
                self.window_uv_shape = np.flip(self.window_shape)

    #===========================================================================
    def trim(self, data, full_fov=False):
        """Trim image to label window.

        Input:
            data            Numpy array containing the image data.

            full_fov        If True, the image is not trimmed.

        Output:
            Data array trimmed to the data window.
        """
        if full_fov:
            return data

        if self.window is None:
            return data

        origin = self.window_origin
        shape = self.window_shape

        return data[origin[0]:origin[0]+shape[0],
                    origin[1]:origin[1]+shape[1]]

    #===========================================================================
    def fov(self, full_fov=False):
        """Construct the field of view based on the metadata.

        Input:
            full_fov        If False, the FOV is cropped to the dimensions
                            given by the cutout window.

        Attributes:
            FOV object.
        """

        # Get FOV
        fov = SSI.fovs[self.mode]

        # Apply cutout window if full fov not requested
        if not full_fov and self.window is not None:
            uv_origin = self.window_uv_origin
            uv_shape = self.window_uv_shape
            fov = oops.fov.SliceFOV(fov, uv_origin, uv_shape)

        return fov


#===============================================================================
class SSI(object):
    """An instance-free class to hold Galileo SSI instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False

    #===========================================================================
    @staticmethod
    def initialize(planets=None, asof=None,
                   mst_pck=True, irregulars=True):
        """Initialize key information about the SSI instrument.

        Fills in key information about the camera.  Must be called first.
        After the first call, later calls to this function are ignored.

        Input:
            planets     A list of planets to pass to define_solar_system. None
                        or 0 means all.
            asof        Only use SPICE kernels that existed before this date;
                        None to ignore.
            mst_pck     True to include MST PCKs, which update the rotation
                        models for some of the small moons.
            irregulars  True to include the irregular satellites;
                        False otherwise.
        """

        # Quick exit after first call
        if SSI.initialized:
            return

        # Initialize Galileo
        Galileo.initialize(planets=planets, asof=asof,
                           mst_pck=mst_pck, irregulars=irregulars)
        Galileo.load_instruments(asof=asof)

        # Load the instrument kernel
        SSI.instrument_kernel = Galileo.spice_instrument_kernel('SSI')[0]

        # Construct the FOVs
        info = SSI.instrument_kernel['INS'][-77036]

        cf_var = 'INS-77036_DISTORTION_COEFF'
        fo_var = 'INS-77036_FOCAL_LENGTH'
        px_var = 'INS-77036_PIXEL_SIZE'
        cxy_var = 'INS-77036_FOV_CENTER'

        cf = cspyce.gdpool(cf_var, 0)[0]
        fo = cspyce.gdpool(fo_var, 0)[0]
        px = cspyce.gdpool(px_var, 0)[0]
        cxy = cspyce.gdpool(cxy_var, 0)

        scale = px/fo
        distortion_coeff = [1, 0, cf]

        # Construct FOVs
        assert info['MAX_SAMPLE'] == 800
        assert info['MAX_LINE'] == 800

        fov_full = oops.fov.BarrelFOV(scale,
                                      (info['MAX_SAMPLE'], info['MAX_LINE']),
                                      coefft_uv_from_xy=distortion_coeff,
                                      uv_los=(cxy[0], cxy[1]))
        fov_summed = oops.fov.SubsampledFOV(fov_full, 2)
#        fov_his =
#        fov_hma =
#        fov_hca = oops.fov.GapFOV(oops.fov.SubsampledFOV(fov_full, (1,4)),
#                                  (1,0.25))
#               ... maybe need SparseFOV or SkipFOV class
#        fov_him =

        # Construct FOV dictionary
        SSI.fovs['FULL'] = fov_full

        # Phase-2 Telemetry Formats
        SSI.fovs['HIS'] = fov_summed
        SSI.fovs['HMA'] = fov_full
        SSI.fovs['HCA'] = fov_full
        SSI.fovs['HIM'] = fov_full
        SSI.fovs['IM8'] = fov_full
        SSI.fovs['AI8'] = fov_summed
        SSI.fovs['IM4'] = fov_full

        # Phase-1 Telemetry Formats
        SSI.fovs['XCM'] = fov_full
#        SSI.fovs['XED'] = fov_full
        SSI.fovs['HCJ'] = fov_full      # Inference based on inspection
        SSI.fovs['HCM'] = fov_full      # Inference based on inspection
                                        # hmmm, actually C0248807700R.img is 800x200
                                        # maybe this is just a cropped full fov

        # Construct the SpiceFrame
        _ = oops.frame.SpiceFrame("GLL_SCAN_PLATFORM")

        # Load kernels
        Galileo.load_kernels()

        SSI.initialized = True
        return

    #===========================================================================
    @staticmethod
    def reset():
        """Reset the internal Galileo SSI parameters.

        Can be useful for debugging.
        """

        SSI.instrument_kernel = None
        SSI.fovs = {}
        SSI.initialized = False

        Galileo.reset()

################################################################################
