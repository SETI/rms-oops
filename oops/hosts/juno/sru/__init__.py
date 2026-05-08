################################################################################
# oops/hosts/juno/sru/__init__.py
################################################################################
import os
import sys

import julian
import numpy as np
import oops
import pdsparser
import pdstable
try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits

from filecache import FCPath

from oops.hosts.juno import Juno


################################################################################
# Standard class methods
################################################################################
def from_file(filespec, return_all_planets=False, full_fov=False,
              method='strict', **parameters):
    """A static method to return a Snapshot from a Juno SRU FITS file."""

    SRU.initialize()    # Use defaults unless explicitly initialized.

    filespec = FCPath(filespec)

    # Load FITS payload.
    local_path = filespec.retrieve()
    with pyfits.open(local_path) as hdul:
        hdu = hdul[0]
        if hdu.data is None:
            for item in hdul:
                if item.data is not None:
                    hdu = item
                    break

        data_2d = hdu.data
        header_dict = dict(hdu.header)

    # Read detached PDS3 label if available.
    label = _read_label_dict(filespec, method=method)

    meta_source = {}
    if isinstance(label, dict):
        meta_source.update(label)
    meta_source.update(header_dict)

    # Build metadata and FOV.
    meta = Metadata(meta_source)
    fov = meta.fov(full_fov=full_fov)

    data = np.asarray(data_2d)
    if data.ndim != 2:
        data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError('SRU FITS image must be 2-D')

    # Ensure time-dependent kernels are available.
    if meta.tstart > sys.float_info.min:
        Juno.load_cks(meta.tstart, meta.tstop)
        Juno.load_spks(meta.tstart, meta.tstop)

    result = oops.obs.Snapshot(('v', 'u'), meta.tstart, meta.exposure,
                               fov,
                               path='JUNO',
                               frame='JUNO_SRU',
                               dict=header_dict,
                               data=data,
                               mask=(data == 0),
                               instrument='SRU',
                               filespec=filespec,
                               basename=filespec.name)

    try:
        kernels = Juno.used_kernels(result.time, 'sru', return_all_planets)
        result.insert_subfield('spice_kernels', kernels)
    except Exception:
        # Keep observation construction robust even if kernel tracing changes.
        pass

    return result


#===============================================================================
def _read_label_dict(filespec, method='strict'):
    """Read SRU detached PDS label, returning an empty dict if unavailable."""

    candidates = [filespec]
    if filespec.suffix.upper() != '.LBL':
        candidates += [
            filespec.with_suffix('.LBL'),
            filespec.with_suffix('.lbl')
        ]

    for candidate in candidates:
        try:
            return pdsparser.Pds3Label(candidate, method=method).as_dict()
        except Exception:
            continue

    return {}


#===============================================================================
def from_index(filespec, supplemental_filespec=None, full_fov=False, **parameters):
    """A static method to return a list of Snapshot objects from an index file."""

    SRU.initialize()
    filespec = FCPath(filespec)

    table = pdstable.PdsTable(filespec, columns=[])
    row_dicts = table.dicts_by_row()

    if supplemental_filespec is not None:
        supplemental_filespec = FCPath(supplemental_filespec)
        supplemental_table = pdstable.PdsTable(supplemental_filespec)
        supplemental_row_dicts = supplemental_table.dicts_by_row()
        for row_dict, supplemental_row in zip(row_dicts, supplemental_row_dicts):
            row_dict.update(supplemental_row)

    snapshots = []
    for row_dict in row_dicts:
        meta = Metadata(row_dict)
        if meta.exposure == 0:
            continue

        fov = meta.fov(full_fov=full_fov)
        file_spec_name = row_dict.get('FILE_SPECIFICATION_NAME', '')
        basename = os.path.basename(file_spec_name)

        item = oops.obs.Snapshot(('v', 'u'), meta.tstart, meta.exposure,
                                 fov,
                                 path='JUNO',
                                 frame='JUNO_SRU',
                                 dict=row_dict,
                                 instrument='SRU',
                                 filespec=file_spec_name,
                                 basename=basename)

        try:
            item.spice_kernels = Juno.used_kernels(item.time, 'sru')
        except Exception:
            pass

        if 'VOLUME_ID' in row_dict and 'FILE_SPECIFICATION_NAME' in row_dict:
            item.filespec = (row_dict['VOLUME_ID'] + '/' +
                             row_dict['FILE_SPECIFICATION_NAME'])
            item.basename = basename

        snapshots.append(item)

    return snapshots


#===============================================================================
def initialize(ck='reconstructed', planets=None, asof=None,
               spk='reconstructed', gapfill=True,
               mst_pck=True, irregulars=True):
    """Initialize key information about the SRU instrument."""

    SRU.initialize(ck=ck, planets=planets, asof=asof,
                   spk=spk, gapfill=gapfill,
                   mst_pck=mst_pck, irregulars=irregulars)


################################################################################
class Metadata(object):

    #===========================================================================
    def __init__(self, meta_dict):
        """Assemble image metadata from a label or index row dictionary."""

        def _float_value(*keys, default=0.0):
            value = _value(*keys, default=default)
            if isinstance(value, (list, tuple)):
                value = value[0]
            try:
                return float(value)
            except Exception:
                return float(default)

        def _value(*keys, default=None):
            for key in keys:
                if key in meta_dict and meta_dict[key] not in (None, ''):
                    return meta_dict[key]
            return default

        self.nlines = int(_value('LINES', 'NAXIS2', default=SRU.LINES))
        self.nsamples = int(_value('LINE_SAMPLES', 'NAXIS1', default=SRU.SAMPLES))

        # Exposure duration in seconds.
        exposure_s = _value('EXPOSURE_DURATION', 'EXPTIME', 'EXPOSURE', default=None)
        if exposure_s is not None:
            self.exposure = _float_value('EXPOSURE_DURATION', 'EXPTIME', 'EXPOSURE',
                                         default=0.0)
        else:
            self.exposure = 0.0

        start_time = _value('START_TIME', 'DATE-OBS', 'IMAGE_TIME', default='UNK')
        stop_time = _value('STOP_TIME', default='UNK')
        if start_time == 'UNK':
            self.tstart = self.tstop = sys.float_info.min
        else:
            self.tstart = julian.tdb_from_tai(julian.tai_from_iso(start_time))
            if stop_time != 'UNK':
                self.tstop = julian.tdb_from_tai(julian.tai_from_iso(stop_time))
                self.exposure = max(0.0, self.tstop - self.tstart)
            else:
                self.tstop = self.tstart + self.exposure

        self.target = _value('TARGET', 'TARGET_NAME', default='')
        self.filter = ''
        self.mode = 'FULL'

    #===========================================================================
    def fov(self, full_fov=False):
        """Construct an SRU FOV."""

        return SRU.fovs['FULL']


#===============================================================================
class SRU(object):
    """An instance-free class to hold Juno SRU instrument parameters."""

    LINES = 512
    SAMPLES = 512
    UV_LOS = (255.5, 255.5)
    FOCAL_LENGTH_PX = 1760.21137

    instrument_kernel = None
    instrument_code = None
    fov_info = {}
    fovs = {}
    initialized = False

    #===========================================================================
    @staticmethod
    def initialize(ck='reconstructed', planets=None, asof=None,
                   spk='reconstructed', gapfill=True,
                   mst_pck=True, irregulars=True):
        """Initialize key information about the SRU instrument."""

        if SRU.initialized:
            return

        Juno.initialize(ck=ck, planets=planets, asof=asof, spk=spk,
                        gapfill=gapfill,
                        mst_pck=mst_pck, irregulars=irregulars)
        Juno.load_instruments(instruments=['SRU'], asof=asof)

        try:
            SRU.instrument_kernel = Juno.spice_instrument_kernel('SRU')[0]
        except Exception:
            SRU.instrument_kernel = None

        SRU.fovs = SRU._build_fovs()

        try:
            _ = oops.frame.SpiceFrame('JUNO_SRU')
        except Exception:
            pass

        SRU.initialized = True

    #===========================================================================
    @staticmethod
    def _build_fovs():
        # SIS: 512x512, boresight at (255.5, 255.5), focal length 1760.21137 px.
        scale = 1.0 / SRU.FOCAL_LENGTH_PX
        fov_full = oops.fov.FlatFOV((scale, scale),
                                    (SRU.SAMPLES, SRU.LINES),
                                    uv_los=SRU.UV_LOS)
        return {
            'FULL': fov_full,
            'NONE': fov_full
        }

    #===========================================================================
    @staticmethod
    def reset():
        """Reset the internal SRU parameters."""

        SRU.instrument_kernel = None
        SRU.instrument_code = None
        SRU.fov_info = {}
        SRU.fovs = {}
        SRU.initialized = False

        Juno.reset()

