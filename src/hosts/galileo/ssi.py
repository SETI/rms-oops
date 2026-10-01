##########################################################################################
# hosts/galileo/ssi.py
##########################################################################################

import re

import numpy as np
import julian
import cspyce
import pdstable

from filecache import FCPath
from pdsparser import Pds3Label, PdsError
from vicar     import VicarImage

import oops
from . import _Galileo
from hosts import Host, PdsHostError
from hosts._pds3_support import _read_pds3_image_array

# The SSI camera frame is the scan platform frame, whose C kernels are sampled at discrete
# spacecraft clock ticks. SSI images are spaced as closely as 1 unit in the file name,
# which corresponds to 80 clock ticks. Therefore, we use a tolerance of +/-40.
_FRAME_ID = 'GLL_SCAN_PLATFORM'
_SPICE_FRAME_ID = -77001
_TICK_TOLERANCE = 40

# The NAIF ID of the SSI instrument, as used in its instrument kernel
_NAIF_ID = -77036

# The CUT_OUT_WINDOW value written in the supplemental index when the label has none
_NO_WINDOW = [-1, -1, -1, -1]

# Target name as found in PDS label -> correct name
_TARGET_NAME_REPAIRS = {
    'BLACK_SKY'  : 'NONE',
    'NON_SCIENCE': 'NONE',
    'STAR'       : 'NONE',
    'HD151288'   : 'HD 151288',
    'J RINGS'    : 'JUPITER_RING_PLANE',
}

# These are star targets and need to be defined as LightSources
# "BET_ARI", "DEL_PSC", "GAM_ORI", "HD 151288", "PLEIADES", "SIRIUS", "TAU_CET"
_TARGET_STARS = re.compile(r'(HD 151288|PLEIADES|SIRIUS|[A-Z]{3}_[A-Z]{3})$')


class SSI(Host):

    NAME = 'Galileo SSI'

    _INSTRUMENT_KERNEL = None
    _FOVS = {}
    _initialized = False

    @staticmethod
    def from_file(fileinfo, *, full_fov=False, return_all_planets=False,
                  pds3_method='fast', astrometry=False, target=None, lightsource=None,
                  path=None, frame=None, fov=None, calibrations=[], timeshift=None,
                  navigation=None, parallel=None, tracker=None, **kwargs):
        """A Snapshot object based on a given Galileo SSI image file.

        By default, only the valid image region is returned.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | PdsLabel | VicarLabel | VicarImage):
                The file path or its parsed content as a PdsLabel, VicarLabel, or
                VicarImage. A VICAR file must be accompanied by its detached PDS3 label.
            full_fov (bool, optional): If True, the full image is returned with a mask
                describing the regions with no data; otherwise, the image and FOV are
                trimmed to the label's cutout window.
            return_all_planets (bool, optional): Include kernels for all planets, not just
                the target of the mission phase.

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
            fov (oops.FOV, optional): Override for the default FOV of the observing
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
            **kwargs: Additional keyword arguments, ignored here.

        Returns:
            Snapshot: The observation, with subfields `spice_kernels`, `filepath`,
            `basename`, `spice_to_frame`, `spice_frame_name`, `spice_frame_id`, `abspath`
            and `image_url` inserted.

        Raises:
            PdsHostError: If the PDS3 label is missing or cannot be parsed.
        """

        SSI._initialize()  # define everything the first time through; use defaults
                           # unless _initialize() was called explicitly.

        fileinfo = SSI._read_fileinfo(fileinfo, formats='PV', astrometry=astrometry,
                                      pds3_method=pds3_method)

        # The metadata always come from the PDS3 label; a VICAR file's own header does
        # not contain the same keywords.
        data = None
        if isinstance(fileinfo, Pds3Label):
            label = fileinfo
            if not astrometry:
                data = _read_pds3_image_array(label)
        else:
            if isinstance(fileinfo, VicarImage):
                data = fileinfo.data_2d
            try:
                label = Pds3Label(fileinfo.filepath, method=pds3_method)
            except PdsError as err:
                raise PdsHostError(str(err)) from err

        # Unlike Pds3Label.dict, which returns a set for a value such as CUT_OUT_WINDOW
        # enclosed in braces, as_dict() preserves the order of its elements.
        dict_ = label.as_dict()
        filepath = FCPath(fileinfo.filepath)

        snapshot = SSI._make_snapshot(dict_, filepath=filepath, data=data,
                                      full_fov=full_fov,
                                      return_all_planets=return_all_planets,
                                      target=target, lightsource=lightsource, path=path,
                                      frame=frame, fov=fov, calibrations=calibrations,
                                      timeshift=timeshift, navigation=navigation,
                                      parallel=parallel, tracker=tracker)
        return snapshot

    @staticmethod
    def from_index(filepath, supplemental_filepath=None, *, full_fov=False,
                   return_all_planets=False, target=None, lightsource=None, path=None,
                   frame=None, fov=None, calibrations=[], timeshift=None, navigation=None,
                   tracker=None, **kwargs):
        """A list of Snapshot objects, one for each row in an SSI index file.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to the label of the
                index file.
            supplemental_filepath (str | pathlib.Path | FCPath, optional): The full path
                to the label of a supplemental index file whose columns, row by row,
                augment those of the index file.
            full_fov (bool, optional): If True, each field of view covers the full image
                rather than the cutout window.
            return_all_planets (bool, optional): Include kernels for all planets, not just
                the target of the mission phase.
            target (str | oops.Body, optional): Override for the default target Body of
                every observation. Use "NONE" for inertial pointing, indicating that no
                Solar System body was tracked.
            lightsource (oops.Lightsource, optional): Override for the default Lightsource
                of every observation.
            path (str | oops.Path, optional): Override for the Path of the observer.
            frame (str | oops.Frame, optional): Override for the Frame of the observing
                instrument.
            fov (oops.FOV, optional): Override for the FOV of every observation,
                replacing the FOV that each row's telemetry format and cutout window would
                otherwise define.
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
            tracker (str, optional): Assign a TrackerFrame to the Observation, to ensure
                that the target body remains at a fixed position within the FOV. Use one
                of "start", "midtime", and "end", indicating the time within the
                Observation at which the target body's calculated position within the FOV
                is accurate.
            **kwargs: Additional keyword arguments, ignored here except as noted below.

        Returns:
            list[Snapshot]: One observation per row of the index, each with subfields
            `spice_kernels`, `filepath`, `basename`, `spice_to_frame`, `spice_frame_name`
            and `spice_frame_id` inserted.

        Raises:
            ValueError: If `parallel` is given; it cannot apply to every row.
        """

        # Check for unsupported options
        for key in ('parallel',):
            if kwargs.get(key) is not None:
                raise ValueError(f'disallowed Galileo SSI.from_index() option {key}')

        SSI._initialize()
        SSI._define_camera_frame()

        # Read the index file
        table = pdstable.PdsTable(FCPath(filepath), columns=[])
        row_dicts = table.dicts_by_row()

        # Append the supplemental columns to the index rows
        if supplemental_filepath is not None:
            table = pdstable.PdsTable(FCPath(supplemental_filepath))
            for row_dict, supplemental_row_dict in zip(row_dicts, table.dicts_by_row()):
                row_dict.update(supplemental_row_dict)

        # Create the list of Snapshot objects
        snapshots = []
        for row_dict in row_dicts:
            fpath = row_dict['VOLUME_ID'] + '/' + row_dict['FILE_SPECIFICATION_NAME']
            obs = SSI._make_snapshot(row_dict, filepath=fpath, data=None,
                                     full_fov=full_fov,
                                     return_all_planets=return_all_planets,
                                     target=target, lightsource=lightsource,
                                     path=path, frame=frame, fov=fov,
                                     calibrations=calibrations, timeshift=timeshift,
                                     navigation=navigation, tracker=tracker)
            snapshots.append(obs)

        return snapshots

    @staticmethod
    def _make_snapshot(dict_, filepath, data=None, *, full_fov=False,
                       return_all_planets=False, target=None, lightsource=None, path=None,
                       frame=None, fov=None, calibrations=[], timeshift=None,
                       navigation=None, parallel=None, tracker=None):

        # Note that PDS3 labels and index rows use the same names.
        texp = dict_['EXPOSURE_DURATION'] / 1000.

        #TODO: determine whether IMAGE_TIME is the start time or the mid time..
        if dict_['IMAGE_TIME'] == 'UNK':
            # Some RAW_CAL frames in GO_0002 and GO_0003 have no IMAGE_TIME, but every
            # label carries the spacecraft clock count, which the SCLK kernel converts to
            # within a few seconds of IMAGE_TIME wherever both are given. Never fall back
            # to a placeholder time: it silently places the frame at the J2000 epoch.
            tstart = _Galileo.tdb_from_sclk(dict_['SPACECRAFT_CLOCK_START_COUNT'])
            time_from_sclk = True
        else:
            tstart = julian.tdb_from_tai(julian.tai_from_iso(dict_['IMAGE_TIME']))
            time_from_sclk = False

        mode = dict_.get('TELEMETRY_FORMAT_ID', 'NONE')
        filter_ = dict_['FILTER_NAME']

        label_target = dict_['TARGET_NAME']
        label_target = _TARGET_NAME_REPAIRS.get(label_target, label_target)
        if _TARGET_STARS.match(label_target):
            label_lightsource = oops.lightsource.star_lookup(label_target)
            label_target = 'NONE'
        else:
            label_lightsource = 'SUN'

        # The cutout window is (line, sample, lines, samples), with a one-based origin
        window = None
        if 'CUT_OUT_WINDOW' in dict_:
            window = list(dict_['CUT_OUT_WINDOW'])
            if window == _NO_WINDOW:
                window = None

        # Make sure the SPICE frame is defined unless a custom frame was given
        using_cks = frame is None
        if using_cks:
            SSI._define_camera_frame()

        # Trim the FOV and data array to the cutout window
        default_fov = SSI._FOVS[mode]
        if window is not None and not full_fov:
            origin = np.array(window[:2]) - 1
            shape = np.array(window[2:])
            default_fov = oops.fov.SliceFOV(default_fov, np.flip(origin), np.flip(shape))
            if data is not None:
                data = data[origin[0]:origin[0] + shape[0],
                            origin[1]:origin[1] + shape[1]]

        # Define the Snapshot parameters
        params = {
            'cadence'     : oops.cadence.SnapCadence(tstart, texp),
            'fov'         : default_fov,
            'path'        : 'GLL',
            'frame'       : _FRAME_ID,
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
        obs.insert_subfield('instrument', 'SSI')
        obs.insert_subfield('filter', filter_)
        obs.insert_subfield('time_from_sclk', time_from_sclk)
        obs.insert_subfield('label_target', dict_['TARGET_NAME'])

        # With a custom frame, pointing never came from a CK, so any CK that happens to be
        # furnished is unrelated to this observation and must not be reported as used.
        kernels = _Galileo.used_kernels(obs.time, 'ssi',
                                        return_all_planets=return_all_planets,
                                        ck=using_cks)
        obs.insert_subfield('spice_kernels', kernels)
        obs.insert_subfield('spice_to_frame', oops.Matrix3.IDENTITY)
        obs.insert_subfield('spice_frame_name', _FRAME_ID)
        obs.insert_subfield('spice_frame_id', _SPICE_FRAME_ID)

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

        # Labels spell the INSTRUMENT_NAME as "SOLID STATE IMAGING SYSTEM" or
        # "SOLID_STATE_IMAGING"
        return (label.get('SPACECRAFT_NAME', '').startswith('GALILEO')
                and label.get('INSTRUMENT_NAME', '').startswith('SOLID'))

    @staticmethod
    def _detect_in_vicar(label):
        """True if the given VicarLabel describes this host's data."""

        return label.get('MISSION', '') == 'GALILEO' and label.get('SENSOR', '') == 'SSI'

    @staticmethod
    def _initialize(*, planets=None, asof=None, mst_pck=True, irregulars=True):
        """Initialize key information about the SSI instrument.

        Fills in key information about the camera. Must be called first. After the first
        call, later calls to this function are ignored.

        Parameters:
            planets (list, optional): A list of planets to pass to
                :meth:`~oops.Body.define_solar_system`. None or 0 means all.
            asof (str, optional): Only use SPICE kernels that existed before this date;
                None to ignore.
            mst_pck (bool, optional): True to include MST PCKs, which update the rotation
                models for some of the small moons.
            irregulars (bool, optional): True to include the irregular satellites; False
                otherwise.
        """

        # Quick exit after first call
        if SSI._initialized:
            return

        _Galileo.initialize(planets=planets, asof=asof, mst_pck=mst_pck,
                            irregulars=irregulars)
        _Galileo.load_instruments(asof=asof)

        # Load the instrument kernel
        SSI._INSTRUMENT_KERNEL = _Galileo.spice_instrument_kernel('SSI')[0]

        # Construct the FOVs
        info = SSI._INSTRUMENT_KERNEL['INS'][_NAIF_ID]

        cf = cspyce.gdpool(f'INS{_NAIF_ID}_DISTORTION_COEFF', 0)[0]
        fo = cspyce.gdpool(f'INS{_NAIF_ID}_FOCAL_LENGTH', 0)[0]
        px = cspyce.gdpool(f'INS{_NAIF_ID}_PIXEL_SIZE', 0)[0]
        cxy = cspyce.gdpool(f'INS{_NAIF_ID}_FOV_CENTER', 0)

        scale = px/fo
        distortion_coeff = [1, 0, cf]

        assert info['MAX_SAMPLE'] == 800
        assert info['MAX_LINE'] == 800

        fov_full = oops.fov.BarrelFOV(scale, (info['MAX_SAMPLE'], info['MAX_LINE']),
                                      coefft_uv_from_xy=distortion_coeff,
                                      uv_los=(cxy[0], cxy[1]))
        fov_summed = oops.fov.SubsampledFOV(fov_full, 2)
#        fov_his =
#        fov_hma =
#        fov_hca = oops.fov.GapFOV(oops.fov.SubsampledFOV(fov_full, (1,4)),
#                                  (1,0.25))
#               ... maybe need SparseFOV or SkipFOV class
#        fov_him =

        # Construct FOV dictionary, keyed by the telemetry format ID
        fovs = {}
        fovs['FULL'] = fov_full

        # Phase-2 Telemetry Formats
        fovs['HIS'] = fov_summed
        fovs['HMA'] = fov_full
        fovs['HCA'] = fov_full
        fovs['HIM'] = fov_full
        fovs['IM8'] = fov_full
        fovs['AI8'] = fov_summed
        fovs['IM4'] = fov_full

        # Phase-1 Telemetry Formats
        fovs['XCM'] = fov_full
#        fovs['XED'] = fov_full
        fovs['HCJ'] = fov_full      # Inference based on inspection
        fovs['HCM'] = fov_full      # Inference based on inspection
                                    # hmmm, actually C0248807700R.img is 800x200
                                    # maybe this is just a cropped full fov
        fovs['NONE'] = fov_full     # Inference based on inspection

        SSI._FOVS = fovs

        # Load kernels
        _Galileo.load_kernels()

        # Define star HD 151288, a target not in the default catalog
        ra = 15*(16 + 45/60 + 6.3505410601/3600)    # 16 45 06.3505410601 in Simbad
        dec = 33 + 30/60 + 33.213816014/3600        # +33 30 33.213816014
        oops.lightsource.add_star('HD 151288', ra, dec, 8.11, 'K7.5Ve')

        SSI._initialized = True

    @staticmethod
    def _define_camera_frame():
        """Register the SPICE-derived GLL_SCAN_PLATFORM frame.

        :meth:`_initialize` must have been called first. Built lazily (and only once) so
        that observations using a custom frame never construct or depend on the SPICE
        camera frame.
        """

        # The frame is tested against the Frame registry rather than against a flag on
        # this class. Registering a second frame under an ID that already exists does not
        # fail; it is renamed GLL_SCAN_PLATFORM_2 and never found again, so the test has
        # to ask the registry what it actually holds.
        if not oops.Frame.frame_id_exists(_FRAME_ID):
            _ = oops.frame.SpiceType1Frame(_FRAME_ID, _TICK_TOLERANCE)

    @staticmethod
    def _reset():
        """Reset the internal Galileo SSI parameters.

        Can be useful for debugging.
        """

        SSI._INSTRUMENT_KERNEL = None
        SSI._FOVS = {}
        SSI._initialized = False
        _Galileo.reset()


SSI._register()

##########################################################################################
