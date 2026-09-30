##########################################################################################
# hosts/cassini/iss.py
##########################################################################################

import numpy as np
import julian
import pdstable

from filecache import FCPath
from pdsparser import Pds3Label
from vicar     import VicarImage

import oops
from . import _Cassini
from hosts import Host
from hosts._pds3_support import _read_pds3_image_array

# There are two C-matrix conventions here, related by _CMATRIX_ROTATION (a 180-degree spin
# about the boresight):
#   * spice-frame: the pointing straight from SPICE (a CK or cspyce.pxform), i.e.
#     the J2000 -> CASSINI_ISS_<camera> rotation. Axes: z along the line of
#     sight, x to the left, y up.
#   * oops-frame: the oops observation frame that the FOV uses. Axes follow the
#     oops convention: z along the line of sight, x to the right, y down.
# oops-frame = _CMATRIX_ROTATION * spice-frame. The instrument's internal coordinate
# system matches the oops-frame, so a recorded (spice-frame) C-matrix is rotated by
# _CMATRIX_ROTATION to build the observation frame. The generic
# Snapshot.get_spice_cmatrix() uses this matrix as attribute `spice_to_frame`.
_CMATRIX_ROTATION = oops.Matrix3([[-1,0,0],[0,-1,0],[0,0,1]])

# Create a master version of the NAC and WAC distortion models from
#   Owen Jr., W.M., 2003. Cassini ISS Geometric Calibration of April 2003.
#   JPL IOM 312.E-2003.
# These polynomials convert from X,Y (radians) to U,V (pixels).

_NAC_F = 2002.703    # mm
_NAC_E2 = 8.28e-6    # / mm2
_NAC_E5 = 5.45e-6    # / mm
_NAC_E6 = -19.67e-6  # / mm
_NAC_KX = 83.33333   # samples/mm
_NAC_KY = 83.3428    # lines/mm

_NAC_COEFF = np.zeros((4,4,2))
_NAC_COEFF[1,0,0] = _NAC_KX         * _NAC_F
_NAC_COEFF[3,0,0] = _NAC_KX*_NAC_E2 * _NAC_F**3
_NAC_COEFF[1,2,0] = _NAC_KX*_NAC_E2 * _NAC_F**3
_NAC_COEFF[1,1,0] = _NAC_KX*_NAC_E5 * _NAC_F**2
_NAC_COEFF[2,0,0] = _NAC_KX*_NAC_E6 * _NAC_F**2

_NAC_COEFF[0,1,1] = _NAC_KY         * _NAC_F
_NAC_COEFF[2,1,1] = _NAC_KY*_NAC_E2 * _NAC_F**3
_NAC_COEFF[0,3,1] = _NAC_KY*_NAC_E2 * _NAC_F**3
_NAC_COEFF[0,2,1] = _NAC_KY*_NAC_E5 * _NAC_F**2
_NAC_COEFF[1,1,1] = _NAC_KY*_NAC_E6 * _NAC_F**2

_WAC_F = 200.7761    # mm
_WAC_E2 = 60.89e-6   # / mm2
_WAC_E5 = 4.93e-6    # / mm
_WAC_E6 = -72.28e-6  # / mm
_WAC_KX = 83.33333   # samples/mm
_WAC_KY = 83.34114   # lines/mm

_WAC_COEFF = np.zeros((4,4,2))
_WAC_COEFF[1,0,0] = _WAC_KX         * _WAC_F
_WAC_COEFF[3,0,0] = _WAC_KX*_WAC_E2 * _WAC_F**3
_WAC_COEFF[1,2,0] = _WAC_KX*_WAC_E2 * _WAC_F**3
_WAC_COEFF[1,1,0] = _WAC_KX*_WAC_E5 * _WAC_F**2
_WAC_COEFF[2,0,0] = _WAC_KX*_WAC_E6 * _WAC_F**2

_WAC_COEFF[0,1,1] = _WAC_KY         * _WAC_F
_WAC_COEFF[2,1,1] = _WAC_KY*_WAC_E2 * _WAC_F**3
_WAC_COEFF[0,3,1] = _WAC_KY*_WAC_E2 * _WAC_F**3
_WAC_COEFF[0,2,1] = _WAC_KY*_WAC_E5 * _WAC_F**2
_WAC_COEFF[1,1,1] = _WAC_KY*_WAC_E6 * _WAC_F**2

# For testing the numerical inverse fitting
#    _WAC_COEFF = np.zeros((4,4,2))
#    _WAC_COEFF[0,1,0] = 1.5
#    _WAC_COEFF[1,0,0] = 0.5
#    _WAC_COEFF[0,1,1] = 0.75
#    _WAC_COEFF[1,0,1] = 2.0

_DISTORTION_COEFF_XY_TO_UV = {'NAC': _NAC_COEFF, 'WAC': _WAC_COEFF}

# Create a master version of the inverse distortion model.
# These coefficients were computed by numerically solving the above polynomials. These
# polynomials convert from U,V (pixels) to X,Y (radians). Maximum errors from applying the
# original distortion model and then inverting:
# NAC
#   X DIFF MIN MAX -7.01489382641e-10 1.15568299657e-10
#   Y DIFF MIN MAX -7.7440587623e-10 8.81658628916e-10
#   U DIFF MIN MAX -0.00138077186011 1.94695478513e-05
#   V DIFF MIN MAX -0.000474712833352 0.000563339044731
# WAC
#   X DIFF MIN MAX -3.29122915029e-07 5.42305755497e-08
#   Y DIFF MIN MAX -3.92109405705e-07 3.98905653939e-07
#   U DIFF MIN MAX -0.0535697887647 0.000916852346108
#   V DIFF MIN MAX -0.0182888189072 0.0187174796013

_NAC_INV_COEFF = np.zeros((4,4,2))
_NAC_INV_COEFF[:,:,0] = [
    [ -1.14799845e-10,  7.80494024e-14,  1.73312704e-15, -5.95242349e-19],
    [  5.99190197e-06, -3.91823615e-13, -7.12054264e-15,  0.00000000e+00],
    [  1.42455664e-12, -2.15242793e-18,  0.00000000e+00,  0.00000000e+00],
    [ -7.11199158e-15,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
_NAC_INV_COEFF[:,:,1] = [
    [ -4.86573092e-12,  5.99122009e-06, -3.91358768e-13, -7.12774967e-15],
    [  1.03832114e-13,  1.41728538e-12, -1.55778300e-18,  0.00000000e+00],
    [  2.03725690e-16, -7.11687723e-15,  0.00000000e+00,  0.00000000e+00],
    [  1.67086926e-21,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]

_WAC_INV_COEFF = np.zeros((4,4,2))
_WAC_INV_COEFF[:,:,0] = [
    [ -5.42748411e-08,  5.06916433e-12,  8.16888725e-13, -3.85258837e-17],
    [  5.97679707e-05, -3.53472340e-12, -5.13400740e-13,  0.00000000e+00],
    [  5.66784001e-11, -1.27038656e-16,  0.00000000e+00,  0.00000000e+00],
    [ -5.09498280e-13,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
_WAC_INV_COEFF[:,:,1] = [
    [ -3.31749330e-10,  5.97618831e-05, -3.50726141e-12, -5.17011679e-13],
    [  6.70738287e-12,  5.34696724e-11, -8.85892191e-17,  0.00000000e+00],
    [  1.35389219e-14, -5.11900149e-13,  0.00000000e+00,  0.00000000e+00],
    [  7.39668979e-19,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]

# For testing the numerical inverse fitting
#    _WAC_INV_COEFF[:,:,0] = [
#        [  1.14377216e-09,  5.71428571e-01, -5.17435487e-16,  1.39385624e-17],
#        [ -2.85714286e-01, -3.18326095e-14, -1.04402502e-17,  0.00000000e+00],
#        [ -6.02467295e-14, -1.36440914e-16,  0.00000000e+00,  0.00000000e+00],
#        [ -2.77233884e-17,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
#    _WAC_INV_COEFF[:,:,1] = [
#        [ -2.86046684e-09, -1.90476190e-01,  1.72236899e-14, -3.66611741e-18],
#        [  7.61904762e-01,  4.33089979e-14,  1.25388484e-16,  0.00000000e+00],
#        [  7.96957986e-14,  1.53755115e-16,  0.00000000e+00,  0.00000000e+00],
#        [  4.82114213e-17,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]

_DISTORTION_COEFF_UV_TO_XY = {'NAC': _NAC_INV_COEFF, 'WAC': _WAC_INV_COEFF}

# Target name as found in PDS label -> correct name
_TARGET_NAME_REPAIRS = {
    'ERRIAPO'  : 'ERRIAPUS',
    'K07S4'    : 'AEGAEON',
    'S8_2004'  : 'FORNJOT',
    'S12_2004' : 'S/2004_S_12',
    'S13_2004' : 'S/2004_S_13',
    'S14_2004' : 'HATI',
    'S18_2004' : 'BESTLA',
    'SKADI'    : 'SKATHI',
    'SUTTUNG'  : 'SUTTUNGR',
    'THRYM'    : 'THRYMR',
    'DARK SKY' : 'NONE',
    'SKY'      : 'NONE',
    'UNK'      : 'NONE',
}

# These are star targets and need to be defined as LightSources
_TARGET_STARS = {'FOMALHAUT', 'SPICA'}


class ISS(Host):

    NAME = 'Cassini ISS'

    _INSTRUMENT_KERNEL = None
    _FOVS = {}
    _initialized = False

    @staticmethod
    def from_file(fileinfo, *, fast_distortion=True, return_all_planets=False,
                  pds3_method='fast', astrometry=False, target=None, lightsource=None,
                  path=None, frame=None, fov=None, calibrations=[], timeshift=None,
                  navigation=None, parallel=None, tracker=None, **kwargs):
        """A Snapshot object based on a given Cassini ISS image file.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | PdsLabel | VicarLabel | VicarImage):
                The file path or its parsed content as a PdsLabel, VicarLabel, or
                VicarImage.
            fast_distortion (bool | None, optional): True to use a pre-inverted
                polynomial; False to use a dynamically solved polynomial; None to use a
                :class:`~oops.fov.FlatFOV`.
            return_all_planets (bool, optional): Include kernels for all planets, not just
                Jupiter or Saturn.

            pds3_method (str, optional): The method of parsing a PDS3 label. One of:

                * "strict" performs strict parsing, which requires that the label conform
                  to the full PDS3 standard.
                * "loose" is similar to the above, but tolerates some common syntax
                  errors.
                * "compound" is similar to "loose", but it parses a "compound" label,
                  i.e., one that might contain more than one "END" statement. This option
                  is not supported for attached labels.
                * "fast": uses s a different parser, which executes ~ 30x fast than the
                  above and handles all the most common aspects of the PDS3 standard.
                  However, it is not guaranteed to provide an accurate parsing under all
                  circumstances.

            astrometry (bool, optional):
                True to specify that the returned Observations will only be used for
                timing and astrometry. In this case, no data arrays are read and returned
                in the Observation.
            target (str | oops.Body, optional): Override for the default target Body of
                the observation. Use "NONE" for inertial pointing, indicating that no
                Solar System body was tracked.
            lightsource (oops.Lightsource, optional): Override for the default Lightsource
                of the observation.
            path (str | oops.Path, optional): Override for the Path of the observer.
            frame (str | oops.Frame, optional): Override for the Frame of the observing
                instrument.
            fov (oops.Frame, optional): Override for the default FOV of the observing
                instrument.
            calibrations (oops.Calibration | list[oops.Calibration], optional): Override
                for the calibration or list of calibrations.
            timeshift (tuple[float, str], optional): Assign a Fittable time shift to one
                or more attributes of the Observation. The first item is the initial time
                shift in seconds. The second item is a string indicating which attributes
                are to be time-shifted: "T" for the start/stop time, "P" for the path,
                and/or "F" for the frame. If multiple letters are given, the time shifts
                are coupled to whichever attribute is listed first.
            navigation (tuple[float, ...], optional): Two or three rotation angles in
                radians to apply to the frame in order to align the calculated geometry
                with the observed geometry. In this case, a fittable Navigation frame
                wraps the default frame. Specify three angles to include a rotation about
                the optic axis or two for a pointing offset without rotation. Use (0,0) or
                (0,0,0) as the input if you have no starting guess.
            parallel (Observation, optional): A parallel Observation (same origin and
                time, different frame and FOV) relative to which this Observation has a
                fixed offset.
            tracker (str, optional): Assign a TrackerFrame to the Observation, to ensure
                that the target body remains at a fixed position within the FOV. Use one
                of "start", "midtime", and "end", indicating the time within the
                Observation at which the target body's calculated position within the FOV
                is accurate. Note that no more than one of `parallel` and `tracker` can be
                specified.
            **kwargs: Additional keyword arguments; they are accepted and ignored.

        Returns:
            Snapshot: The observation, with subfields `spice_kernels`, `filepath`,
            `basename`, `spice_to_frame`, `spice_frame_name`, `spice_frame_id`, `abspath`
            and `image_url` inserted.
        """

        ISS._initialize()  # define everything the first time through; use defaults
                           # unless _initialize() was called explicitly.

        fileinfo = ISS._read_fileinfo(fileinfo, formats='PV', astrometry=astrometry,
                                      pds3_method=pds3_method)

        data = None
        if isinstance(fileinfo, Pds3Label):
            if not astrometry:
                data = _read_pds3_image_array(fileinfo)
            dict_ = fileinfo.dict
        elif isinstance(fileinfo, VicarImage):
            data = fileinfo.data_2d
            dict_ = fileinfo.label
        else:   # VicarLabel, with astrometry == True
            dict_ = fileinfo

        filepath = FCPath(fileinfo.filepath)

        snapshot = ISS._make_snapshot(dict_, filepath=filepath, data=data,
                                      fast_distortion=fast_distortion,
                                      return_all_planets=return_all_planets,
                                      target=target, lightsource=lightsource, path=path,
                                      frame=frame, fov=fov, calibrations=calibrations,
                                      timeshift=timeshift, navigation=navigation,
                                      parallel=parallel, tracker=tracker)
        return snapshot

    @staticmethod
    def from_index(filepath, *, fast_distortion=True, return_all_planets=False,
                   calibrations=[], timeshift=None, navigation=None):
        """A list of Snapshot objects, one for each row of a Cassini ISS index file.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to a Cassini ISS index
                file or its PDS label.
            fast_distortion (bool | None, optional): True to use a pre-inverted
                polynomial; False to use a dynamically solved polynomial; None to use a
                :class:`~oops.fov.FlatFOV`.
            return_all_planets (bool, optional): Include kernels for all planets, not just
                Jupiter or Saturn.
            calibrations (oops.Calibration | list[oops.Calibration], optional): Override
                for the calibration or list of calibrations.
            timeshift (tuple[float, str], optional): Assign a Fittable time shift to one
                or more attributes of the Observation. The first item is the initial time
                shift in seconds. The second item is a string indicating which attributes
                are to be time-shifted: "T" for the start/stop time, "P" for the path,
                and/or "F" for the frame. If multiple letters are given, the time shifts
                are coupled to whichever attribute is listed first.
            navigation (tuple[float, ...], optional): Two or three rotation angles in
                radians to apply to the frame in order to align the calculated geometry
                with the observed geometry. In this case, a fittable Navigation frame
                wraps the default frame. Specify three angles to include a rotation about
                the optic axis or two for a pointing offset without rotation. Use (0,0) or
                (0,0,0) as the input if you have no starting guess.

        Returns:
            list[Snapshot]: One observation per row of the index, each with subfields
            `spice_kernels`, `filepath`, `basename`, `spice_to_frame`, `spice_frame_name`
            and `spice_frame_id` inserted.
        """

        ISS._initialize()
        ISS._define_camera_frames()

        filepath = FCPath(filepath)

        # Read the index file
        table = pdstable.PdsTable(filepath, columns=[])
        row_dicts = table.dicts_by_row()

        # Create the list of Snapshot objects
        snapshots = []
        for row_dict in row_dicts:
            filepath = row_dict['VOLUME_ID'] + '/' + row_dict['FILE_SPECIFICATION_NAME']
            obs = ISS._make_snapshot(row_dict, filepath=filepath, data=None,
                                     fast_distortion=fast_distortion,
                                     return_all_planets=return_all_planets,
                                     calibrations=calibrations, timeshift=timeshift,
                                     navigation=navigation)
            snapshots.append(obs)

        return snapshots

    @staticmethod
    def _make_snapshot(dict_, filepath, data=None, *, fast_distortion=True,
                       return_all_planets=False, target=None, lightsource=None, path=None,
                       frame=None, fov=None, calibrations=[], timeshift=None,
                       navigation=None, parallel=None, tracker=None):

        # Note that VICAR and PDS3 dictionaries use the same names.
        tstart = julian.tdb_from_iso(dict_['START_TIME'])
        texp   = dict_['EXPOSURE_DURATION'] / 1000.
        mode   = dict_['INSTRUMENT_MODE_ID']        # "FULL", "SUM2", or "SUM4"
        camera = 'WAC' if 'WIDE' in dict_['INSTRUMENT_NAME'] else 'NAC'

        # Merge filter names, ignoring "CL1" and "CL2"
        if 'FILTER_NAME' in dict_:
            filter1, filter2 = dict_['FILTER_NAME']
        else:
            filter1 = dict_['FILTER1_NAME']
            filter2 = dict_['FILTER2_NAME']

        if filter1[:2] == 'CL':
            if filter2[:2] == 'CL':
                filter_ = 'CLEAR'
            else:
                filter_ = filter2
        else:
            if filter2[:2] == 'CL':
                filter_ = filter1
            else:
                filter_ = '+'.join(sorted((filter1, filter2)))

        gain_mode = None
        if dict_['GAIN_MODE_ID'][:3] == '215':
            gain_mode = 0
        elif dict_['GAIN_MODE_ID'][:2] == '95':
            gain_mode = 1
        elif dict_['GAIN_MODE_ID'][:2] == '29':
            gain_mode = 2
        elif dict_['GAIN_MODE_ID'][:2] == '12':
            gain_mode = 3

        label_target = _TARGET_NAME_REPAIRS.get(dict_['TARGET_NAME'],
                                                dict_['TARGET_NAME'])
        if label_target in _TARGET_STARS:
            label_lightsource = oops.lightsource.star_lookup(label_target)
            label_target = 'NONE'
        else:
            label_target = ISS._fix_cassini_iss_target(label_target, dict_, tstart)
            label_lightsource = 'SUN'

        # Make sure the SPICE kernels are loaded; construct the frame
        _Cassini.load_spks(tstart, tstart + texp)
        using_cks = frame is None
        if using_cks:
            _Cassini.load_cks(tstart, tstart + texp)
            ISS._define_camera_frames()

        # Define the Snapshot parameters
        params = {
            'cadence'     : oops.cadence.SnapCadence(tstart, texp),
            'fov'         : ISS._FOVS[camera, mode, fast_distortion],
            'path'        : 'CASSINI',
            'frame'       : 'CASSINI_ISS_' + camera,
            'target'      : label_target,
            'lightsource' : label_lightsource,
            'calibrations': [],         # TBD
        }

        # Apply standard overrides
        Host._apply_overrides(params, target=target, lightsource=lightsource, path=path,
                              frame=frame, fov=fov, calibrations=calibrations,
                              timeshift=timeshift, navigation=navigation,
                              parallel=parallel, tracker=tracker)

        # For a Snapshot, `tstart` is the parameter name when using a Cadence as input
        tstart = params['cadence']
        del params['cadence']

        obs = oops.observation.Snapshot(('v','u'), tstart, texp, **params)
        if data is not None:
            obs.insert_subfield('data', data)

        obs.insert_subfield('dict', dict_)
        obs.insert_subfield('instrument', 'ISS')
        obs.insert_subfield('detector', camera)
        obs.insert_subfield('sampling', mode)
        obs.insert_subfield('filter', filter_)
        obs.insert_subfield('filter1', filter1)
        obs.insert_subfield('filter2', filter2)
        obs.insert_subfield('gain_mode', gain_mode)
        obs.insert_subfield('label_target', dict_['TARGET_NAME'])

        # With a custom frame, pointing never came from a CK, so any CK that happens to be
        # furnished (e.g. the gapfill CKs loaded unconditionally by _Cassini.initialize())
        # is unrelated to this observation and must not be reported as used.
        kernels = _Cassini.used_kernels(obs.time, 'iss',
                                        return_all_planets=return_all_planets,
                                        ck=using_cks)
        obs.insert_subfield('spice_kernels', kernels)
        obs.insert_subfield('spice_to_frame', _CMATRIX_ROTATION)
        obs.insert_subfield('spice_frame_name', 'CASSINI_ISS_' + camera)
        obs.insert_subfield('spice_frame_id', -82360 if camera == 'NAC' else -82361)

        if isinstance(filepath, str):   # for from_index()
            obs.insert_subfield('filepath', filepath)
            obs.insert_subfield('basename', filepath.rpartition('/')[-1])
        else:
            obs.insert_subfield('filepath', filepath)
            obs.insert_subfield('basename', filepath.name)
            obs.insert_subfield('abspath', filepath.get_local_path().resolve())
            obs.insert_subfield('image_url', filepath.absolute().as_posix())

        return obs

    @staticmethod
    def _detect_in_pds3(label):
        """True if the given parsed PDS3 label describes this host's data."""

        return (label.get('INSTRUMENT_HOST_NAME', '').startswith('CASSINI')
                and label.get('INSTRUMENT_ID', '').startswith('ISS'))

    @staticmethod
    def _detect_in_vicar(label):
        """True if the given VicarLabel describes this host's data."""

        return ISS._detect_in_pds3(label)   # PDS3 and VICAR use the same names

    @staticmethod
    def _fix_cassini_iss_target(target, dict_, tstart):
        """The target of a ring observation, or the given target otherwise.

        The observation is identified as a ring observation by the target code in its
        OBSERVATION_ID, which has the form "ISS_<orbit><code>_<activity>_<suffix>" (e.g.,
        "ISS_053RI_PHOTOMDRK002_PRIME"); the ring codes are RI and RA through RG.

        Parameters:
            target (str): The target name from the label, after repairs.
            dict_ (dict): The label or index row, from which OBSERVATION_ID is read.
            tstart (float): The observation start time in seconds TDB, which determines
                whether a ring observation is of Jupiter's rings or Saturn's.

        Returns:
            str: "JUPITER_RING_PLANE" for a ring observation before the Saturn tour,
            "SATURN_RING_PLANE" for one during it, and `target` for any other observation
            or if OBSERVATION_ID is absent or not of the expected form.
        """

        parts = dict_.get('OBSERVATION_ID', '').split('_')
        if len(parts) < 3:
            return target
        if parts[1][-2:] in {'RI', 'RA', 'RB', 'RC', 'RD', 'RE', 'RF', 'RG'}:
            if tstart < _Cassini.TOUR:
                return 'JUPITER_RING_PLANE'
            return 'SATURN_RING_PLANE'

        return target

    @staticmethod
    def _initialize(*, ck='reconstructed', planets=None, asof=None, spk='reconstructed',
                    gapfill=True, mst_pck=True, irregulars=True):
        """Initialize key information about the ISS instrument.

        Fills in key information about the WAC and NAC. Must be called first. After the
        first call, later calls to this function are ignored.

        Parameters:
            ck (str, optional): The set of C kernels to load, 'reconstructed' or
                'predicted' (case-insensitive); 'none' to load no C kernels automatically,
                leaving their handling to the caller.
            planets (list, optional): A list of planets to pass to
                :meth:`~oops.Body.define_solar_system`. None or 0 means all.
            asof (str, optional): Only use SPICE kernels that existed before this date;
                None to ignore.
            spk (str, optional): The set of SP kernels to load, 'reconstructed' or
                'predicted' (case-insensitive); 'none' to load no SP kernels
                automatically, leaving their handling to the caller.
            gapfill (bool, optional): True to include gapfill CKs. False otherwise.
            mst_pck (bool, optional): True to include MST PCKs, which update the rotation
                models for some of the small moons.
            irregulars (bool, optional): True to include the irregular satellites; False
                otherwise.
        """

        # Quick exit after first call
        if ISS._initialized:
            return

        _Cassini.initialize(ck=ck, planets=planets, asof=asof, spk=spk, gapfill=gapfill,
                            mst_pck=mst_pck, irregulars=irregulars)
        _Cassini.load_instruments(asof=asof)

        # Load the instrument kernel
        ISS._INSTRUMENT_KERNEL = _Cassini.spice_instrument_kernel('ISS')[0]

        # Construct a Polynomial FOV for each camera
        fovs = {}
        for detector in ['NAC', 'WAC']:
            info = ISS._INSTRUMENT_KERNEL['INS']['CASSINI_ISS_' + detector]

            # Full field of view
            lines = info['PIXEL_LINES']
            samples = info['PIXEL_SAMPLES']

            xfov = info['FOV_REF_ANGLE']
            yfov = info['FOV_CROSS_ANGLE']
            assert info['FOV_ANGLE_UNITS'] == 'DEGREES'

            uscale = np.arctan(np.tan(xfov * oops.RPD) / (samples/2.))
            vscale = np.arctan(np.tan(yfov * oops.RPD) / (lines/2.))

            # Display directions: [u,v] = [right,down]
            full_fov = oops.fov.PolynomialFOV(
                                (samples,lines),
                                coefft_uv_from_xy=_DISTORTION_COEFF_XY_TO_UV[detector],
                                coefft_xy_from_uv=None)
            full_fov_fast = oops.fov.PolynomialFOV(
                                (samples,lines),
                                coefft_uv_from_xy=_DISTORTION_COEFF_XY_TO_UV[detector],
                                coefft_xy_from_uv= _DISTORTION_COEFF_UV_TO_XY[detector])
            full_fov_none = oops.fov.FlatFOV((uscale,vscale), (samples,lines))

            # Load the dictionary, include the subsampling modes
            fovs[detector, 'FULL', False] = full_fov
            fovs[detector, 'SUM2', False] = oops.fov.SubsampledFOV(full_fov, 2)
            fovs[detector, 'SUM4', False] = oops.fov.SubsampledFOV(full_fov, 4)
            fovs[detector, 'FULL', True] = full_fov_fast
            fovs[detector, 'SUM2', True] = oops.fov.SubsampledFOV(full_fov_fast, 2)
            fovs[detector, 'SUM4', True] = oops.fov.SubsampledFOV(full_fov_fast, 4)
            fovs[detector, 'FULL', None] = full_fov_none
            fovs[detector, 'SUM2', None] = oops.fov.SubsampledFOV(full_fov_none, 2)
            fovs[detector, 'SUM4', None] = oops.fov.SubsampledFOV(full_fov_none, 4)

            fovs[detector, 'FULL'] = full_fov_none
            fovs[detector, 'SUM2'] = oops.fov.SubsampledFOV(full_fov_none, 2)
            fovs[detector, 'SUM4'] = oops.fov.SubsampledFOV(full_fov_none, 4)

        ISS._FOVS = fovs
        ISS._initialized = True

    @staticmethod
    def _define_camera_frames():
        """Register the SPICE-derived CASSINI_ISS_NAC and CASSINI_ISS_WAC frames.

        :meth:`initialize` must have been called first. Built lazily (and only once) so
        that observations using a custom C-matrix never construct or depend on the SPICE
        camera frames.
        """

        # Each camera is tested individually against the Frame registry rather than
        # against a flag on this class. Registering a second frame under an ID that
        # already exists does not fail; it is renamed CASSINI_ISS__NAC_2 and never found
        # again, so the test has to ask the registry what it actually holds.
        for camera in ('NAC', 'WAC'):
            frame_id = 'CASSINI_ISS_' + camera
            if oops.Frame.frame_id_exists(frame_id):
                continue

            # The SpiceFrame takes the "flipped" ID, leaving the camera's own ID for the
            # Cmatrix that rotates it into the oops-frame convention
            spice_frame = oops.frame.SpiceFrame(frame_id, frame_id=frame_id + '_FLIPPED')
            oops.frame.Cmatrix(_CMATRIX_ROTATION, spice_frame, frame_id=frame_id)

    @staticmethod
    def _reset():
        """Reset the internal Cassini ISS parameters.

        Can be useful for debugging.
        """

        ISS._INSTRUMENT_KERNEL = None
        ISS._FOVS = {}
        ISS._initialized = False
        _Cassini.reset()


ISS._register()

##########################################################################################
