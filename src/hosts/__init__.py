##########################################################################################
# host/__init__.py
##########################################################################################

from pathlib import Path
from typing import TypeAlias

import numpy as np
import oops
from astropy.io import fits as pyfits
from filecache  import FCPath
from pdsparser  import Pds3Label, PdsError
from vicar      import VicarError, VicarImage, VicarLabel

ObsArray: TypeAlias = (np.ndarray[tuple[int, ...], np.dtype[np.object_]]
                       | tuple[oops.Observation, ...])


class HostError(OSError):
    """Exception arising from the content of a data file or label."""
    pass


class UnknownHostError(HostError):
    """If a data file is not associated with any registered Host subclass."""
    pass


class HostFormatError(HostError):
    """If a data file is incorrectly formatted."""
    pass


class PdsHostError(HostFormatError, PdsError):
    """A subclass of both HostFormatError and PdsError."""
    pass


class VicarHostError(HostFormatError, VicarError):
    """A subclass of both HostFormatError and VicarError."""
    pass


class Host:
    """Superclass of all host modules."""

    _HOSTS = []     # A list of all registered Host subclasses.
    _LOOKUP = {}    # name -> subclass
    NAME = 'none'   # Each subclass must define

    @staticmethod
    def from_file(filepath, *, pds3_method='fast', select=None, astrometry=False,
                  target=None, lightsource=None, path=None, frame=None, fov=None,
                  calibrations=[], timeshift=None, navigation=None, parallel=None,
                  tracker=None, **kwargs):
        """One or more Observation objects derived from a given data file.

        Parameters:
            filepath (str | Path | FCPath): The path to the data file or its  detached
                abel in PDS3 or PDS4 format.
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

            select (int | slice | str | tuple[int | slice | str, ...]): An index, slice,
                or tuple of indices and slices to apply to the returned array of
                Observation objects. For example, if the file contains two Observations
                but `select=1`, only the second Observation will be returned.
                Alternatively, specify one or more host-specific names of the data arrays.
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
            **kwargs:
                Any other instrument-specific parameters, as needed.

        Returns:
            oops.Observation | ObsArray: The Observation or array of Observations
            retrieved from the file.

        Raises:
            HostError: If `filepath` is not associated with any registered Host subclass.
            HostFormatError: If `filepath` has invalid syntax for format. Note that
                subclass :class:`~HostPdsError` is returned for PDS-formatted files and
                :class:`~HostVicarError` is returned for VICAR files.
            OSError: If `filepath` cannot be read; one of FileNotFoundError,
                IsADirectoryError, PermissionError, etc.
            ValueError: If the value of `pds3_method` is invalid.
        """

        host, fileinfo = Host._identify_host(filepath, astrometry=astrometry,
                                             pds3_method=pds3_method)
        return host.from_file(fileinfo, pds3_method=pds3_method, astrometry=astrometry,
                              target=target, lightsource=lightsource, path=path,
                              frame=frame, fov=fov, calibrations=calibrations,
                              timeshift=timeshift, navigation=navigation,
                              parallel=parallel, tracker=tracker, **kwargs)

    @staticmethod
    def _detect_in_pds3(label):
        """True if the given parsed PDS3 label describes this host's data.

        The developer must override this method for any host that might use PDS3-labeled
        data files. It must return True if the label indicates that the file was obtained
        from this instrument.

        Parameters:
            label (Pds3Label): A parsed PDS3 label.

        Returns:
            bool: True if `label` describes this host's data; False otherwise.
        """

        return False

    @staticmethod
    def _detect_in_vicar(label):
        """True if the given VicarLabel describes this host's data.

        The developer must override this method for any host that might use VICAR format
        data files. It must return True if the VICAR header indicates that the file was
        obtained from this instrument.

        Parameters:
            label (vicar.VicarLabel): A parsed VICAR label.

        Returns:
            bool: True if `label` describes this host's data; False otherwise.
        """

        return False

    @staticmethod
    def _detect_in_fits(hdulist):
        """True if the given FITS HDUList describes this host's data.

        The developer must override this method for any host that might use FITS-formatted
        data files. It must return True if the FITS header indicates that the file was
        obtained from this instrument.

        Parameters:
            hdulist (HDUList): A FITS HDUList.

        Returns:
            bool: True if `hdulist` contains this host's data; False otherwise.
        """

        return False

    @staticmethod
    def _detect_in_file(filepath):
        """True if the given parsed PDS4 label describes this host's data.

        The developer must override this method for any host that might be recognized by
        the name or if the file is not in one of PDS3, VICAR, or FITS format. It must
        return True if the file was obtained from this instrument.

        Parameters:
            filepath (FCPath): Path to the data file.

        Returns:
            bool: True if `filepath` contains this host's data; False otherwise.
        """

        return False

    @staticmethod
    def _identify_host(filepath, *, astrometry=False, pds3_method='fast'):
        """Identify the host module associated with a given data file or label.

        Also return the parsed file content for PDS3, VICAR, and FITS formats.

        Parameters:
            filepath (str | Path | FCPath): The path to the data file or its detached
                label in PDS3 or PDS4 format.
            astrometry (bool, optional): True to specify that the returned Observations
                will only be used for timing and astrometry. In this case, no data arrays
                are read and returned in the Observation.
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

        Returns:
            tuple[type, (Pds3Label | VicarLabel | VicarImage | HDUList | FCPath)]:
            The Host subclass associated with the `filepath`, followed by the parsed
            content of its label, if any. If the file is not in a standard format, the
            second returned item is None.

        Raises:
            HostError: If `filepath` is not associated with any registered Host subclass.
            HostFormatError: If `filepath` has invalid syntax for format. Note that
                subclass :class:`~HostPdsError` is returned for PDS-formatted files and
                :class:`~HostVicarError` is returned for VICAR files.
            OSError: If `filepath` cannot be read; one of FileNotFoundError,
                IsADirectoryError, PermissionError, etc.
            ValueError: If the value of `pds3_method` is invalid.
        """

        # Check every registered host and every defined file format
        return Host._identify_and_parse(filepath, hosts=Host._HOSTS, formats='PVFO',
                                        pds3_method=pds3_method, astrometry=astrometry)

    @classmethod
    def _read_fileinfo(cls, fileinfo, *, formats, astrometry=False, pds3_method='fast'):
        """Read and parse the file content for PDS3, VICAR, or FITS format.

        Parameters:
            fileinfo (str | Path | FCPath | Pds3Label | VicarLabel | HDUList): The path
                to the data file or its label or its parsed content.
            formats (str, optional): A string identifying the file formats to consider:
                'P' for PDS3; 'V' for VICAR; 'F' for FITS; 'O' for other.
            astrometry (bool, optional):
                True to specify that the returned Observations will only be used for
                timing and astrometry. In this case, no data arrays are read and returned
                in the Observation.
            pds3_method (str, optional):
                The method of parsing a PDS3 label. One of:

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

        Returns:
            tuple[type, (Pds3Label | VicarLabel | VicarImage | HDUList | FCPath)]:
            The Host subclass associated with the `filepath`, followed by the parsed
            content of its label, if any. If the file is not in a standard format, the
            second returned item is None.

        Raises:
            HostError: If `filepath` is not associated with the given Host subclass.
            HostFormatError: If `filepath` has invalid syntax for format. Note that
                subclass :class:`~HostPdsError` is returned for PDS-formatted files and
                :class:`~HostVicarError` is returned for VICAR files.
            OSError: If the file cannot be read; one of FileNotFoundError,
                IsADirectoryError, PermissionError, etc.
            ValueError: If the value of `pds3_method` is invalid.
        """

        # If the file was already parsed, just return this info
        if not isinstance(fileinfo, (str, Path, FCPath)):
            return fileinfo

        # Parse the file if it's in a standard format
        _, parsed = Host._identify_and_parse(fileinfo, hosts=[cls], formats=formats,
                                             astrometry=astrometry,
                                             pds3_method=pds3_method)
        return parsed

    @staticmethod
    def _identify_and_parse(filepath, hosts, *, formats='PVFO', astrometry=False,
                            pds3_method='fast'):
        """Interpret the filepath and identify the associated host module.

        Also return the parsed file content for PDS3, VICAR, and FITS formats.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The path to the data file or its
                detached label in PDS3 or PDS4 format.
            hosts (list[type]): A list of Host subclasses to consider.
            formats (str, optional): A string identifying the file formats to consider:
                'P' for PDS3; 'V' for VICAR; 'F' for FITS; 'O' for other.
            astrometry (bool, optional): True to specify that the returned Observations
                will only be used for timing and astrometry. In this case, no data arrays
                are read and returned in the Observation.
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

        Returns:
            tuple[type, (Pds3Label | VicarLabel | VicarImage | HDUList | FCPath)]:
            The Host subclass associated with the `filepath`, followed by the parsed
            content of its label, if any. If the file is not in a standard format, the
            second returned item is None.

        Raises:
            HostError: If `filepath` is not associated with any listed Host subclass.
            HostFormatError: If `filepath` has invalid syntax for format. Note that
                subclass :class:`~HostPdsError` is returned for PDS-formatted files and
                :class:`~HostVicarError` is returned for VICAR files.
            OSError: If `filepath` cannot be read; one of FileNotFoundError,
                IsADirectoryError, PermissionError, etc.
            ValueError: If the value of `pds3_method` is invalid.
        """

        filepath = FCPath(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f'No such file or directory: {filepath}')
        if not filepath.is_file():
            raise IsADirectoryError(f'File is a directory: {filepath}')

        suffix = filepath.suffix.lower()

        # Handle a detached PDS3 label
        if 'P' in formats and suffix == '.lbl':
            try:
                label = Pds3Label(filepath, method=pds3_method)
            except PdsError as err:
                raise PdsHostError(str(err)) from err
            else:
                for host in hosts:
                    if host._detect_in_pds3(label):
                        return host, label
                raise UnknownHostError(f'unrecognized host in PDS3 label: {filepath}')

        # Handle a FITS file
        if 'F' in formats and suffix in ('.fits', '.fit'):
            try:
                hdulist = pyfits.open(filepath.retrieve())
            except OSError as err:
                raise HostFormatError(f'not a valid FITS file: {filepath}') from err
            else:
                for host in hosts:
                    if host._detect_in_fits(hdulist):
                        return host, hdulist
                raise UnknownHostError(f'unrecognized host in FITS file: {filepath}')

        # Handle a VICAR file
        if 'V' in formats and VicarLabel.is_vicar_file(filepath):
            try:
                if astrometry:
                    vic = VicarLabel(filepath, strict=False)
                else:
                    vic = VicarImage.from_file(filepath, strict=False)
                    vic.filepath = filepath     # from_file does not record the path
            except VicarError as err:
                raise VicarHostError(str(err)) from err
            else:
                for host in hosts:
                    if host._detect_in_vicar(vic):
                        return host, vic
                raise UnknownHostError(f'unrecognized host in VICAR header: {filepath}')

        # Handle an attached PDS3 label
        if 'P' in formats:
            try:
                label = Pds3Label(filepath, method=pds3_method)
            except PdsError as err:
                raise PdsHostError(str(err)) from err
            else:
                for host in hosts:
                    if host._detect_in_pds3(label):
                        return host, label
                raise UnknownHostError(f'unrecognized host in PDS3 file: {filepath}')

        # Handle another file format
        if 'O' in formats:
            for host in hosts:
                if host._detect_in_file(filepath):
                    return host, filepath

        if len(hosts) == 1:
            raise HostError(f'unrecognized file format for {hosts[0].NAME}: '
                            f'{filepath}')
        raise UnknownHostError(f'unrecognized host in file: {filepath}')

    _TIMESHIFT_CLASS = {'T': oops.cadence.TimeShift,
                        'P': oops.path.PathShift,
                        'F': oops.frame.FrameShift}

    _TIMESHIFT_ATTR = {'T': 'cadence',
                       'P': 'path',
                       'F': 'frame'}

    @staticmethod
    def _apply_overrides(params, *, timeshift=None, navigation=None, parallel=None,
                         tracker=None, **kwargs):
        """Apply overrides to the Observation constructor's input parameters.

        The subclass's constructor should call this method to handle the standard override
        parameters. Other input parameters are ignored.

        Parameters:
            params (dict): The dictionary of inputs to the Observation constructor before
                any overrides have been applied.
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
                is accurate. Note that only one of `parallel` and `tracker` can be
                specified.
            **kwargs: Any additional host-specific parameters, ignored here.

        Returns:
            dict: The `params` dictionary, modified in place.
        """

        # Override certain standard attributes
        for key in ('target', 'lightsource', 'path', 'frame', 'fov', 'calibrations'):
            if kwargs.get(key) is not None:
                params[key] = kwargs[key]

        # Apply an optional time shift
        if timeshift is not None:
            dt, letters = timeshift
            letters = letters.upper()
            if not letters:
                raise ValueError('the timeshift must specify or more of T/P/F')
            if set(letters) - set('TPF'):
                raise ValueError(f'invalid timeshift letters: {letters}')
            arg = None
            for letter in letters:
                attr = Host._TIMESHIFT_ATTR[letter]
                if arg is None:     # the first time-shifted item is independent
                    arg = Host._TIMESHIFT_CLASS[letter](dt, params[attr])
                    params[attr] = arg
                else:               # subsequent time shifts are linked to the first
                    params[attr] = Host._TIMESHIFT_CLASS[letter](arg, params[attr])

        # Apply frame adjustments
        if parallel is not None:
            if tracker is not None:
                raise ValueError('parallel and tracker cannot both be specified')

            frame = oops.Frame.as_frame(params['frame']).wrt(parallel.frame)
            xform = frame.transform_at_time(params['cadence'].midtime)
            params['frame'] = oops.frame.Cmatrix(xform.matrix, reference=parallel.frame)

        elif tracker is not None:
            try:
                tfrac = {'start': 0., 'midtime': 0.5, 'end': 1.}[tracker]
            except KeyError:
                raise ValueError('invalid tracker, should be "start", "midtime", or '
                                 f'"end": {tracker!r}')
            times = params['cadence'].time
            epoch = times[0] + tfrac * (times[1] - times[0])
            params['frame'] = oops.frame.TrackerFrame(params['frame'],
                                                      target=params['target'],
                                                      observer=params['path'],
                                                      epoch=epoch)

        if navigation is not None:
            params['frame'] = oops.frame.Navigation(navigation, params['frame'])

        return params

    @classmethod
    def _register(cls):
        """Register this Host subclass with the Hosts module.

        The developer must call ``module._register()`` immediately after defining
        ``module`` as a host subclass. This ensures that ``Host.from_file(...)`` will
        identify the correct host and call the correct constructor.
        """

        if not issubclass(cls, Host):
            raise TypeError(f'not a Host subclass: {cls}')

        if cls not in Host._HOSTS:
            Host._HOSTS.append(cls)
            Host._LOOKUP[cls.NAME] = cls

##########################################################################################
