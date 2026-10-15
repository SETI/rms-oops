##########################################################################################
# oops/host/host_.py
##########################################################################################

import numbers
import pathlib
from typing import TypeAlias

import numpy as np
from astropy.io import fits as pyfits
from filecache  import FCPath
from pdsparser  import Pds3Label, PdsError, is_pds3_file
from pdstable   import PdsTable
from vicar      import VicarError, VicarImage, VicarLabel

from oops.body        import Body
from oops.cadence     import TimeShift
from oops.frame       import Cmatrix, Frame, FrameShift, Navigation, TrackerFrame
from oops.observation import Observation
from oops.path        import PathShift

from ._errors import (HostError, HostFormatError, PdsHostError, UnknownHostError,
                      VicarHostError)

ObsArray: TypeAlias = (np.ndarray[tuple[int, ...], np.dtype[np.object_]]
                       | tuple[Observation, ...])


class Host:
    """Superclass of all host modules."""

    _HOSTS = []     # A list of all registered Host subclasses.
    _LOOKUP = {}    # name -> subclass
    NAME = 'none'   # Each subclass must define

    @staticmethod
    def from_file(filepath, *, pds3_options=None, select=None, astrometry=False,
                  target=None, lightsource=None, path=None, frame=None, cadence=None,
                  fov=None, calibrations=[], timeshift=None, navigation=None,
                  tracker=None, parallel=None, **kwargs):
        """One or more Observation objects derived from a given data file.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The path to the data file or its
                detached label in PDS3 or PDS4 format.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.
            select (int | slice | str | tuple[int | slice | str, ...]): An index, slice,
                or tuple of indices and slices to apply to the returned array of
                Observation objects. For example, if the file contains two Observations
                but `select=1`, only the second Observation will be returned.
                Alternatively, specify one or more host-specific names of the data arrays.
            astrometry (bool, optional):
                True to specify that the returned Observations will only be used for
                timing and astrometry. In this case, no data arrays are read and returned
                in the Observation.
            target (str | Body, optional): Override for the default target Body of the
                observation. Use "NONE" for inertial pointing, indicating that no Solar
                System body was tracked.
            lightsource (Lightsource, optional): Override for the default Lightsource of
                the observation.
            path (str | oops.Path, optional): Override for the Path of the observer.
            frame (str | Frame, optional): Override for the Frame of the observing
                instrument.
            cadence (Cadence, optional): Override for the Cadence of the observation.
            fov (FOV, optional): Override for the default FOV of the observing instrument.
            calibrations (Calibration | list[Calibration], optional): Override for the
                calibration or list of calibrations.
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
            parallel (Observation, optional): A parallel Observation (same origin and
                time, different frame and FOV) relative to which this Observation has a
                fixed offset. Note that no more than one of `parallel` and `tracker` can
                be specified.
            **kwargs:
                Any other instrument-specific parameters, as needed.

        Returns:
            Observation | ObsArray: The Observation or array of Observations retrieved
            from the file.

        Raises:
            HostError: If `filepath` is not associated with any registered Host subclass.
            HostFormatError: If `filepath` has invalid syntax for format. Note that
                subclass :class:`~PdsHostError` is returned for PDS-formatted files and
                :class:`~VicarHostError` is returned for VICAR files.
            OSError: If `filepath` cannot be read; one of FileNotFoundError,
                IsADirectoryError, PermissionError, etc.
            ValueError: If a value in `pds3_options` is invalid.
        """

        # Set the default Pds3Label options to method='fast', sources=True
        pds3_options = Host._pds3_options_with_defaults(pds3_options)

        host, fileinfo = Host._identify_host(filepath, astrometry=astrometry,
                                             pds3_options=pds3_options)
        return host.from_file(fileinfo, pds3_options=pds3_options, select=select,
                              astrometry=astrometry, target=target,
                              lightsource=lightsource, path=path,
                              frame=frame, cadence=cadence, fov=fov,
                              calibrations=calibrations, timeshift=timeshift,
                              navigation=navigation, tracker=tracker, parallel=parallel,
                              **kwargs)

    @staticmethod
    def from_index(filepath, *, supplement=None, label=None, row_dicts=None,
                   pds3_options=None, filter=None, select=None, target=None,
                   lightsource=None, path=None, frame=None, cadence=None, fov=None,
                   calibrations=None, timeshift=None, navigation=None, tracker=None,
                   parallel=None, **kwargs):
        """A list of Observations, one for each row of a data file index.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to the PDS label of a
                data file index.
            supplement (str | pathlib.Path | FCPath, optional): The path to the PDS label
                of a supplemental index file. If provided, this file must have the same
                records in the same order as the table described by `filepath`.
            label (PdsLabel, optional): The PDS label of `filepath` if it has already been
                read.
            row_dicts (list[dict[str, Any]], optional): A list of dictionaries, one per
                row of the index, if `filepath` and `supplement` have already been read.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.
            filter (dict[str, Any], optional): A dictionary of index column names and
                their values or a set or tuple of values. If provided, only rows of the
                index in which each named column equals the value, falls in the set of
                values, or falls in a range defined by a tuple of two values, will be
                included in the returned list.
            select (int | slice | str | tuple[int | slice | str, ...]): An index, slice,
                or tuple of indices and slices to apply to the returned array of
                Observation objects. For example, if the file contains two Observations
                but `select=1`, only the second Observation will be returned.
                Alternatively, specify one or more host-specific names of the data arrays.
            target (str | oops.Body, optional): Override for the default target Body of
                every observation. Use "NONE" for inertial pointing, indicating that no
                Solar System body was tracked.
            lightsource (oops.Lightsource, optional): Override for the default Lightsource
                of every observation.
            path (str | oops.Path, optional): Override for the Path of the observer.
            frame (str | oops.Frame, optional): Override for the Frame of the observing
                instrument.
            cadence (Cadence, optional): Override for the Cadence of the observation.
            fov (FOV, optional): Override for the default FOV of the observing instrument.
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
            parallel (Observation, optional): A parallel Observation (same origin and
                time, different frame and FOV) relative to which this Observation has a
                fixed offset. Note that no more than one of `parallel` and `tracker` can
                be specified.
            **kwargs: Additional keyword arguments, ignored here except as noted below.

        Returns:
            list[Observation | ObsArray]: The list of Observations or arrays of
            Observations, one for each filtered row of the index file.

        Raises:
            HostError: If `filepath` is not associated with any registered Host subclass.
            PdsHostError: If `filepath` has invalid PDS label syntax.
            OSError: If `filepath` or `supplement` cannot be read; one of
                FileNotFoundError, IsADirectoryError, PermissionError, etc.
            ValueError: If a value in `pds3_options` is invalid.
        """

        filepath = FCPath(filepath)

        # Set default PDS3 options to method='fast', sources=True
        pds3_options = Host._pds3_options_with_defaults(pds3_options)

        if label is None:
            try:
                label = Pds3Label(filepath, **pds3_options)
            except PdsError as err:
                raise PdsHostError(str(err)) from err

        # Read and filter the rows
        if row_dicts is None:
            row_dicts = Host._read_index_rows(filepath, supplement=supplement,
                                              pds3_options=pds3_options, label=label)
        row_dicts = Host._filter_index_rows(row_dicts, filter)

        # Determine which host is represented by the index. A host whose detector returns
        # None can only be identified from the content of a row.
        for host in Host._HOSTS:
            status = host._detect_in_index(label)
            if status is None:
                if row_dicts and host._detect_in_row(row_dicts[0]):
                    break
            elif status:
                break
        else:
            raise HostError(f'index is not associated with any known host: {filepath}')

        # Process the host-specific index
        return host.from_index(filepath, supplement=None, row_dicts=row_dicts,
                               select=select, target=target, lightsource=lightsource,
                               path=path, frame=frame, cadence=cadence, fov=fov,
                               calibrations=calibrations, timeshift=timeshift,
                               navigation=navigation, tracker=tracker, parallel=parallel,
                               **kwargs)

    @staticmethod
    def _read_index_rows(filepath, *, supplement=None, pds3_options=None, label=None):
        """Read the index rows and return one dictionary per row, with supplementary index
        content merged.

        Parameters:
            filepath (str | pathlib.Path | FCPath): Path to the index label.
            supplement (str | pathlib.Path | FCPath): Path to the supplemental index
                label.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.
            label (PdsLabel, optional): The PDS label of `filepath` if it has already been
                read.

        Returns:
            list[dict[str, Any]]: One dictionary per row.

        Raises:
            ValueError: If `supplement` does not have the same records in the same order
                as `filepath`.
        """

        filepath = FCPath(filepath)
        pds3_method = (pds3_options or {}).get('method', 'fast')
        row_dicts = PdsTable(filepath, label_method=pds3_method).dicts_by_row()

        # Merge supplement if any
        if not supplement:
            return row_dicts

        supplement = FCPath(supplement)
        supp_dicts = PdsTable(supplement, label_method=pds3_method).dicts_by_row()
        if len(supp_dicts) != len(row_dicts):
            raise ValueError('index files have different record counts: '
                             f'{filepath}, {supplement}')

        # Make sure records are in the same order
        common_keys = []
        if row_dicts:
            common_keys = list(set(row_dicts[0].keys()) & set(supp_dicts[0].keys()))
        if common_keys:
            for i, row_dict in enumerate(row_dicts):
                supp_dict = supp_dicts[i]
                supp_vals = [supp_dict[key] for key in common_keys]
                row_vals = [row_dict[key] for key in common_keys]
                if row_vals != supp_vals:
                    raise ValueError(f'index files mismatch at record [{i}]: '
                                     f'{filepath}, {supplement}')

        # Merge dictionaries
        for i, row_dict in enumerate(row_dicts):
            row_dict = row_dict.copy()
            row_dict.update(supp_dicts[i])
            row_dicts[i] = row_dict

        return row_dicts

    @staticmethod
    def _filter_index_rows(row_dicts, filter):
        """Filter the rows of the index.

        Parameters:
            row_dicts (list[dict[str, Any]]): The index content, one dictionary per row.
            filter (dict[str, Any], optional): A dictionary of index column names and
                their values or a set or tuple of values. If provided, only rows of the
                index in which each named column equals the value, falls in the set of
                values, or falls in a range defined by a tuple of two values, will be
                included in the returned list.

        Returns:
            list[dict[str, Any]]: The rows of `row_dicts` that pass the filter; a new list
            unless `filter` is empty, in which case `row_dicts` itself.
        """

        # Apply filter if any
        if not filter:
            return row_dicts

        filtered = []
        for row_dict in row_dicts:
            include = True
            for key, values in filter.items():
                value = row_dict[key]
                if isinstance(values, tuple) and len(values) == 2:
                    if value < values[0] or value > values[1]:
                        include = False
                        break
                elif isinstance(values, (set, frozenset, list, tuple)):
                    if value not in values:
                        include = False
                        break

                # A single value; "in" would do a substring match on strings
                elif value != values:
                    include = False
                    break

            if include:
                filtered.append(row_dict)

        return filtered

    ######################################################################################
    # Detectors requiring overrides
    ######################################################################################

    @staticmethod
    def _detect_in_pds3(label):
        """True if the given parsed PDS3 label describes data from this host or
        instrument.

        The developer must override this method for any data set that might use
        PDS3-labeled data files. It must return True if `label` indicates that the file
        was obtained from this host/instrument.

        Parameters:
            label (Pds3Label): A parsed PDS3 label.

        Returns:
            bool: True if `label` describes this host's data; False otherwise.
        """

        return False

    @staticmethod
    def _detect_in_vicar(label):
        """True if the given VicarLabel describes data from this host or instrument.

        The developer must override this method for any data set that might use VICAR
        format data files. It must return True if `label` indicates that the file
        was obtained from this host or instrument.

        Parameters:
            label (vicar.VicarLabel): A parsed VICAR label.

        Returns:
            bool: True if `label` describes this host's data; False otherwise.
        """

        return False

    @staticmethod
    def _detect_in_fits(hdulist):
        """True if the given FITS HDUList describes data from this host or instrument.

        The developer must override this method for any data set that might use FITS
        format data files. It must return True if `label` indicates that the file was
        obtained from this host or instrument.

        Parameters:
            hdulist (HDUList): A FITS HDUList.

        Returns:
            bool: True if `hdulist` contains this host's data; False otherwise.
        """

        return False

    @staticmethod
    def _detect_in_file(filepath):
        """True if the given file describes data from this host or instrument.

        The developer must override this method for any data set that might be recognized
        by the name or if the file is not in one of PDS3, VICAR, or FITS format. It must
        return True if the file was obtained from this host or instrument.

        Parameters:
            filepath (str | pathlib.Path | FCPath): Path to the data file.

        Returns:
            bool: True if `filepath` contains this host's data; False otherwise.
        """

        return False

    @staticmethod
    def _detect_in_index(label):
        """True if the given PDS3 label describes an index table for this host or
        instrument.

        Return None if the host or instrument cannot be inferred from the label, only from
        individual records.

        The developer must override this method for any data set in which method
        :meth:`~from_index` is defined. It must return True if the the index describes
        data from the host or instrument.

        Parameters:
            label (Pds3Label): The parsed PDS label of an index file.

        Returns:
            bool | None: True if `label` describes an index of data from the host or
            instrument; False if it definitely does not; None if the host or instrument
            cannot be inferred from the label.
        """

        return None

    @staticmethod
    def _detect_in_row(row_dict):
        """True if the given row of an index file refers to this host or instrument.

        The developer must override this method for any data set in which
        :meth:`~_detect_in_index` returns None. It must return True if the information in
        a row of the index describes data from the host or instrument.

        Parameters:
            row_dict (dict): A dictionary containing the content of one row of an index
                file.

        Returns:
            bool: True if this label describes data from the host/instrument; False
            otherwise.
        """

        return False

    ######################################################################################
    # Support methods not to be overridden
    ######################################################################################

    @staticmethod
    def _identify_host(filepath, *, astrometry=False, pds3_options=None):
        """Identify the host module associated with a given data file or label.

        Also return the parsed file content for PDS3, VICAR, and FITS formats.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The path to the data file or its
                detached label in PDS3 or PDS4 format.
            astrometry (bool, optional): True to specify that the returned Observations
                will only be used for timing and astrometry. In this case, no data arrays
                are read and returned in the Observation.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.

        Returns:
            tuple[type, (Pds3Label | VicarLabel | VicarImage | HDUList | FCPath)]:
            The Host subclass associated with the `filepath`, followed by the parsed
            content of its label, if any. If the file is not in a standard format, the
            second returned item is None.

        Raises:
            HostError: If `filepath` is not associated with any registered Host subclass.
            HostFormatError: If `filepath` has invalid syntax for format. Note that
                subclass :class:`~PdsHostError` is returned for PDS-formatted files and
                :class:`~VicarHostError` is returned for VICAR files.
            OSError: If `filepath` cannot be read; one of FileNotFoundError,
                IsADirectoryError, PermissionError, etc.
            ValueError: If a value in `pds3_options` is invalid.
        """

        # Check every registered host and every defined file format
        return Host._identify_and_parse(filepath, hosts=Host._HOSTS, formats='PVFO',
                                        pds3_options=pds3_options, astrometry=astrometry)

    @staticmethod
    def _read_fileinfo(fileinfo, *, formats, astrometry=False, pds3_options=None):
        """Read and parse the file content for PDS3, VICAR, or FITS format.

        This is a support method for the `from_file` methods of the Host subclasses.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | Pds3Label | VicarLabel | HDUList): The
                path to the data file or its label or its parsed content.
            formats (str, optional): A string identifying the file formats to consider:
                'P' for PDS3; 'V' for VICAR; 'F' for FITS; 'O' for other.
            astrometry (bool, optional):
                True to specify that the returned Observations will only be used for
                timing and astrometry. In this case, no data arrays are read and returned
                in the Observation.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.

        Returns:
            Pds3Label | VicarLabel | VicarImage | HDUList | FCPath: The parsed file
            content if it can be obtained from `fileinfo`, either because it is already a
            parsed format or if it refers to a file in a parsed format. If it is file that
            is not in a standard format, the FCPath is returned.

        Raises:
            UnknownHostError: If `fileinfo` is not associated with any registered Host
                subclass.
            HostFormatError: If `fileinfo` has invalid syntax for format. Note that
                subclass :class:`~PdsHostError` is returned for PDS-formatted files and
                :class:`~VicarHostError` is returned for VICAR files.
            OSError: If the file cannot be read; one of FileNotFoundError,
                IsADirectoryError, PermissionError, etc.
            ValueError: If the value in `pds3_options` is invalid.
        """

        # If the file was already parsed, just return this info
        if not isinstance(fileinfo, (str, pathlib.Path, FCPath)):
            return fileinfo

        # Parse the file if it's in a standard format
        _, parsed = Host._identify_and_parse(fileinfo, hosts=Host._HOSTS,
                                             formats=formats, astrometry=astrometry,
                                             pds3_options=pds3_options)
        return parsed

    @staticmethod
    def _identify_and_parse(filepath, hosts, *, formats='PVFO', astrometry=False,
                            pds3_options=None):
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
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.

        Returns:
            tuple[type, (Pds3Label | VicarLabel | VicarImage | HDUList | FCPath)]:
            The Host subclass associated with the `filepath`, followed by the parsed
            content of its label, if any. If the file is not in a standard format, the
            second returned item is the FCPath.

        Raises:
            HostError: If `filepath` is not associated with any listed Host subclass.
            HostFormatError: If `filepath` has invalid syntax for format. Note that
                subclass :class:`~PdsHostError` is returned for PDS-formatted files and
                :class:`~VicarHostError` is returned for VICAR files.
            OSError: If `filepath` cannot be read; one of FileNotFoundError,
                IsADirectoryError, PermissionError, etc.
            ValueError: If the value in `pds3_options` is invalid.
        """

        filepath = FCPath(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f'No such file or directory: {filepath}')
        if not filepath.is_file():
            raise IsADirectoryError(f'File is a directory: {filepath}')

        suffix = filepath.suffix.lower()

        pds3_options = pds3_options or {}

        # Handle a detached PDS3 label
        if 'P' in formats and suffix == '.lbl':
            try:
                label = Pds3Label(filepath, **pds3_options)
            except PdsError as err:
                raise PdsHostError(str(err)) from err
            else:
                for host in hosts:
                    if hasattr(host, '_detect_in_pds3') and host._detect_in_pds3(label):
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
                    if hasattr(host, '_detect_in_fits') and host._detect_in_fits(hdulist):
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
                    if hasattr(host, '_detect_in_vicar') and host._detect_in_vicar(vic):
                        return host, vic
                raise UnknownHostError(f'unrecognized host in VICAR header: {filepath}')

        # Handle an attached PDS3 label
        if 'P' in formats and is_pds3_file(filepath):
            try:
                label = Pds3Label(filepath, **pds3_options)
            except PdsError as err:
                raise PdsHostError(str(err)) from err
            else:
                for host in hosts:
                    if hasattr(host, '_detect_in_pds3') and host._detect_in_pds3(label):
                        return host, label
                raise UnknownHostError(f'unrecognized host in PDS3 file: {filepath}')

        # Handle another file format
        if 'O' in formats:
            for host in hosts:
                if hasattr(host, '_detect_in_file') and host._detect_in_file(filepath):
                    return host, filepath

        if len(hosts) == 1:
            raise HostError(f'unrecognized file format for {hosts[0].NAME}: '
                            f'{filepath}')
        raise UnknownHostError(f'unrecognized host in file: {filepath}')

    _TIMESHIFT_CLASS = {'T': TimeShift, 'P': PathShift, 'F': FrameShift}
    _TIMESHIFT_ATTR = {'T': 'cadence', 'P': 'path', 'F': 'frame'}

    @staticmethod
    def _apply_overrides(params, filepath, *, timeshift=None, navigation=None,
                         tracker=None, parallel=None, **kwargs):
        """Apply overrides to the Observation constructor's input parameters.

        The subclass's constructor should call this method to handle the standard override
        parameters. Other input parameters are ignored.

        Parameters:
            params (dict): The dictionary of inputs to the Observation constructor before
                any overrides have been applied.
            filepath (str | pathlib.Path | FCPath): Path to the file.
            target (str | Body, optional): Override for the default target Body of the
                observation. Use "NONE" for inertial pointing, indicating that no Solar
                System body was tracked.
            lightsource (Lightsource, optional): Override for the default Lightsource of
                the observation.
            path (str | oops.Path, optional): Override for the Path of the observer.
            frame (str | Frame, optional): Override for the Frame of the observing
                instrument.
            cadence (Cadence, optional): Override for the Cadence of the observation.
            fov (FOV, optional): Override for the default FOV of the observing instrument.
            calibrations (Calibration | list[Calibration], optional): Override for the
                calibration or list of calibrations.
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
            parallel (Observation, optional): A parallel Observation (same origin and
                time, different frame and FOV) relative to which this Observation has a
                fixed offset. Note that no more than one of `parallel` and `tracker` can
                be specified.
            **kwargs: Any additional host-specific parameters, ignored here.

        Returns:
            dict: The `params` dictionary, modified in place.
        """

        # Fill in the file/URL info
        params['filepath'] = filepath
        if isinstance(filepath, str):   # for from_index()
            params['basename'] = filepath.rpartition('/')[-1]
        else:
            params['basename'] = filepath.name
            params['abspath'] = filepath.get_local_path().resolve()
            params['image_url'] = filepath.absolute().as_posix()

        # Override certain standard attributes
        for key in ('target', 'lightsource', 'path', 'frame', 'cadence', 'fov',
                    'calibrations'):
            if kwargs.get(key) is not None:
                params[key] = kwargs[key]

        # Apply an optional time shift
        if timeshift is not None:
            dt, letters = timeshift
            letters = letters.upper()
            if not letters:
                raise ValueError('the timeshift must specify one or more of T/P/F')
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

            frame = Frame.as_frame(params['frame']).wrt(parallel.frame)
            xform = frame.transform_at_time(params['cadence'].midtime)
            params['frame'] = Cmatrix(xform.matrix, reference=parallel.frame)

        elif tracker is not None:
            try:
                tfrac = {'start': 0., 'midtime': 0.5, 'end': 1.}[tracker]
            except KeyError:
                raise ValueError('invalid tracker, should be "start", "midtime", or '
                                 f'"end": {tracker!r}')
            times = params['cadence'].time
            epoch = times[0] + tfrac * (times[1] - times[0])

            # Convert body to path or None
            target = params['target']
            if isinstance(target, str) and target.upper() == 'NONE':
                target_path = None
            else:
                target_path = Body.as_body(target).path

            params['frame'] = TrackerFrame(params['frame'],
                                           target=target_path,
                                           observer=params['path'],
                                           epoch=epoch)

        if navigation is not None:
            params['frame'] = Navigation(navigation, params['frame'])

        return params

    @staticmethod
    def _apply_select(observations, select, names=None):
        """The observation(s) chosen by the `select` option of a host's from_file().

        This support method requires that all Observation objects be constructed
        beforehand. Depending on the host/instrument properties, this may be an
        inefficient approach, especially if `astrometry` is True.

        Parameters:
            observations (tuple[Observation | None, ...]): The observations from one file.
            select (int | slice | str | tuple[int | slice | str, ...] | None): An index, a
                slice, or the name of one observation, or a tuple of these; None to choose
                them all.
            names (dict[str, int], optional): The host-specific names of the observations,
                in upper case, each mapped to its index in `observations`. Names in
                `select` are matched without regard to case.

        Returns:
            Observation | None | tuple[Observation | None, ...] | ObsArray: `observations`
            itself if `select` is None; the one observation chosen by an index or a name;
            or a tuple of the observations chosen by a slice or by a tuple, in the order
            given, with each slice in a tuple contributing all of its observations.

        Raises:
            IndexError: If an index is out of range.
            ValueError: If a name is not one of `names`.
            TypeError: If `select`, or an item of a tuple, is not an int, slice or str.
        """

        if select is None:
            return observations

        if isinstance(select, tuple):
            chosen = []
            for item in select:
                if isinstance(item, tuple):
                    raise TypeError(f'invalid select item: {item!r}')
                if isinstance(item, slice):
                    chosen += observations[item]
                else:
                    chosen.append(Host._apply_select(observations, item, names))
            return tuple(chosen)

        if isinstance(select, slice):
            return observations[select]

        if isinstance(select, str):
            names = names or {}
            if select.upper() not in names:
                raise ValueError(f'unrecognized select name: {select!r}')
            return observations[names[select.upper()]]

        if isinstance(select, numbers.Integral) and not isinstance(select, bool):
            return observations[select]

        raise TypeError(f'invalid select: {select!r}')

    @staticmethod
    def _pds3_options_with_defaults(options):
        """The PDS3 option dictionary with Host-wide defaults."""

        options = {} if options is None else options.copy()
        options.setdefault('method', 'fast')
        options.setdefault('sources', True)
        return options

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
