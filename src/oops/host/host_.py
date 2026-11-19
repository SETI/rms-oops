##########################################################################################
# oops/host/host_.py
##########################################################################################

import numbers
import pathlib
from collections import Counter
from typing import TypeAlias

import numpy as np
from astropy.io import fits as pyfits
from filecache  import FCPath
from pdsparser  import Pds3Label, PdsError, is_pds3_file
from pdstable   import PdsTable
from vicar      import VicarError, VicarImage, VicarLabel

from oops._exceptions import OopsIndexError, OopsTypeError, OopsValueError
from oops.body        import Body
from oops.cadence     import TimeShift
from oops.frame       import Cmatrix, Frame, FrameShift, Navigation, TrackerFrame
from oops.observation import Observation
from oops.path        import PathShift

from ._errors import (HostFormatError, HostIndexError, PdsHostError, UnknownHostError,
                      VicarHostError, WrongHostError)

# The number of unpaired or repeated paths that an error message about pairing the rows of
# an index and its supplement lists
_PATHS_IN_MESSAGE = 3

ObsArray: TypeAlias = (np.ndarray[tuple[int, ...], np.dtype[np.object_]]
                       | tuple[Observation | None, ...])


class Host:
    """Superclass of all host modules.

    When a host builds an observation whose pointing comes from SPICE, it checks at load
    time, in its ``from_file``, that the SPICE kernels can supply that pointing at the
    time of the observation. If they cannot, because no C-kernel record covers the time or
    no frame connection can be made, the host raises :class:`~oops.OopsSpiceError`, built
    with :meth:`~oops.OopsSpiceError.from_spice_error` from the exception that cspyce
    raised and raised ``from`` it. That is the only exception a host raises for missing
    SPICE pointing: it never lets the cspyce exception escape, and it never raises
    :class:`~oops.OopsSpiceError` for a failure that is not about SPICE data.

    A host does not check SPICE, and does not raise :class:`~oops.OopsSpiceError`, when
    the caller supplies the pointing: when ``from_file`` is given a `frame` override,
    whether a :class:`~oops.frame.Cmatrix` or any other :class:`~oops.Frame`, or when the
    caller chooses a pointing source that is not SPICE, through an option of the host.

    A host's ``from_index`` returns an iterator that yields one entry per filtered row of
    the index, in index order, and builds each row's observation only when the iteration
    reaches that row, so an error in building one row is raised there. Everything else is
    done by the call itself, so that a bad call fails at once: the checks of the
    arguments, the reading and checking of the index label, its table and any supplement,
    and the filtering of the rows. ``from_index`` is therefore an ordinary function that
    does that work and then returns a generator, never a generator function.
    """

    _HOSTS = []     # A list of all registered Host subclasses.
    _LOOKUP = {}    # name -> subclass
    NAME = 'none'   # Each subclass must define

    SUPPLEMENT = object()  # Sentinel so from_index() searches for a supplemental index

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
            lightsource (LightSource, optional): Override for the default LightSource of
                the observation.
            path (str | Path, optional): Override for the Path of the observer.
            frame (str | Frame, optional): Override for the Frame of the observing
                instrument. When it is given, the host does not check that SPICE can
                supply the pointing.
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
            Observation | ObsArray | None: An Observation or a tuple or array of
            Observations retrieved from the file. None if a selected Observation is
            absent.

        Raises:
            HostFormatError: If `filepath` has invalid syntax or uses an unsupported
                option. Note that subclass :class:`~PdsHostError` is raised for a PDS3
                syntax error and :class:`~VicarHostError` is raised for a VICAR syntax
                error.
            OSError: If `filepath` does not exist or cannot be read.
            UnknownHostError: If `filepath` is not associated with any registered Host
                subclass.
            OopsValueError: If a value in `pds3_options` is invalid.
            OopsSpiceError: If the observation's pointing comes from SPICE and the SPICE
                kernels cannot supply it at the time of the observation. Never raised when
                `frame` is given or when the pointing does not come from SPICE.
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
    def from_index(filepath, *, supplement=SUPPLEMENT, label=None, row_dicts=None,
                   pds3_options=None, filter=None, select=None, target=None,
                   lightsource=None, path=None, frame=None, cadence=None, fov=None,
                   calibrations=None, timeshift=None, navigation=None, tracker=None,
                   parallel=None, **kwargs):
        """An iterator over the Observations of a data file index, one for each row.

        The host is identified when this method is called, from the index label or else
        from the first row of the index, and the host's own ``from_index`` then reads,
        pairs, and filters the rows of the index and its supplement; each row's
        observation is built only when the iteration reaches it. Besides the exceptions
        listed here, the call raises any that the host's ``from_index`` documents.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to the PDS label of a
                data file index.
            supplement (str | pathlib.Path | FCPath | object, optional): The path to the
                PDS label of a supplemental index file, if any, whose columns are merged
                into each row. Each index row is merged with the supplement row for the
                same file: by position when the two tables list the same files in the
                same order, and otherwise by the file that each row names. Use the
                sentinel value `Host.SUPPLEMENT` to read the file in the directory of
                `filepath` whose name matches except that "_index" is replaced by
                "_supplemental_index" in upper or lower case. The host determines
                whether the supplement must exist and how a pair of rows is checked.
            label (Pds3Label, optional): The PDS label of `filepath` if it has already
                been read.
            row_dicts (list[dict[str, Any]], optional): A list of dictionaries, one per
                row of the index, if `filepath` and `supplement` have already been read.
                If provided, they are passed to the host, which does not read
                `supplement`.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.
            filter (dict[str, Any], optional): A dictionary of index column names and
                their values or a set or tuple of values. If provided, only rows of the
                index in which each named column equals the value, falls in the set of
                values, or falls in a range defined by a tuple of two values, will be
                included.
            select (int | slice | str | tuple[int | slice | str, ...]): An index, slice,
                or tuple of indices and slices to apply to the returned array of
                Observation objects. For example, if the file contains two Observations
                but `select=1`, only the second Observation will be returned.
                Alternatively, specify one or more host-specific names of the data arrays.
            target (str | Body, optional): Override for the default target Body of every
                observation. Use "NONE" for inertial pointing, indicating that no Solar
                System body was tracked.
            lightsource (LightSource, optional): Override for the default LightSource of
                every observation.
            path (str | Path, optional): Override for the Path of the observer.
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
            **kwargs: Additional keyword arguments, passed to the host's ``from_index``.

        Returns:
            collections.abc.Iterator[Observation | ObsArray | None]: The host's
            iterator, which yields the Observation, or the tuple or array of Observations,
            for each filtered row of the index file, in index order. It yields None where
            a selected Observation is absent from that row. Wrap it in ``list()`` to build
            every row at once.

        Raises:
            OopsValueError: If a value in `pds3_options` is invalid.
            HostIndexError: If `filepath` or `supplement` cannot be interpreted as an
                index table, or if the rows of the two tables cannot be paired: an index
                row that no supplement row names, a supplement row that names no index
                row, a file listed twice in either table while the rows are paired by
                file, or a pair of rows that fails the host's check.
            OSError: If `filepath` or `supplement` does not exist or cannot be read.
            PdsHostError: If `filepath` uses invalid PDS label syntax.
            UnknownHostError: If `filepath` is not associated with any registered Host
                subclass.
        """

        filepath = FCPath(filepath)

        if label is None:
            try:
                label = Pds3Label(filepath,
                                  **Host._pds3_options_with_defaults(pds3_options))
            except PdsError as err:
                raise PdsHostError(str(err)) from err
            except (TypeError, ValueError) as err:
                raise OopsValueError(str(err)) from err

        # Determine which host is represented by the index. A host whose detector returns
        # None can only be identified from the content of a row; the index is read for
        # that, without its supplement, at most once.
        detection_rows = row_dicts
        for host in Host._HOSTS:
            status = host._detect_in_index(label)
            if status is None:
                if detection_rows is None:
                    detection_rows = Host._read_index_rows(filepath,
                                                           pds3_options=pds3_options,
                                                           label=label)
                if detection_rows and host._detect_in_row(detection_rows[0]):
                    break
            elif status:
                break
        else:
            raise UnknownHostError(f'unknown host in index {filepath}')

        # The host reads, pairs and filters the rows itself, because only the host knows
        # how its supplement is matched to its index
        return host.from_index(filepath, supplement=supplement, label=label,
                               row_dicts=row_dicts, pds3_options=pds3_options,
                               filter=filter, select=select, target=target,
                               lightsource=lightsource, path=path, frame=frame,
                               cadence=cadence, fov=fov, calibrations=calibrations,
                               timeshift=timeshift, navigation=navigation,
                               tracker=tracker, parallel=parallel, **kwargs)

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
            UnknownHostError: If `filepath` is not associated with any registered Host
                subclass.
            HostFormatError: If `filepath` has invalid syntax for format. Note that
                subclass :class:`~PdsHostError` is returned for PDS-formatted files and
                :class:`~VicarHostError` is returned for VICAR files.
            OSError: If `filepath` does not exist or cannot be read.
            OopsValueError: If a value in `pds3_options` is invalid.
        """

        # Check every registered host and every defined file format
        return Host._identify_and_parse(filepath, hosts=Host._HOSTS, formats='PVFO',
                                        pds3_options=pds3_options, astrometry=astrometry)

    @classmethod
    def _read_fileinfo(cls, fileinfo, *, formats, astrometry=False, pds3_options=None):
        """Read and parse the file content for this class in PDS3, VICAR, or FITS format.

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
            HostFormatError: If `fileinfo` has invalid syntax for format. Note that
                subclass :class:`~PdsHostError` is returned for PDS-formatted files and
                :class:`~VicarHostError` is returned for VICAR files.
            OopsValueError: If the value in `pds3_options` is invalid.
            OSError: If the file does not exist or cannot be read.
            WrongHostError: If `fileinfo` is not associated with this Host subclass.
        """

        # If the file was already parsed, just return this info
        if not isinstance(fileinfo, (str, pathlib.Path, FCPath)):
            return fileinfo

        # Parse the file if it's in a standard format
        _, parsed = Host._identify_and_parse(fileinfo, hosts=[cls],
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
            HostFormatError: If `filepath` has invalid syntax for its format. Note that
                subclass :class:`~PdsHostError` is returned for PDS-formatted files and
                :class:`~VicarHostError` is returned for VICAR files.
            OopsValueError: If the value in `pds3_options` is invalid.
            OSError: If `filepath` does not exist or cannot be read.
            UnknownHostError: If `filepath` is not associated with any of several listed
                Host subclasses.
            WrongHostError: If `filepath` is not associated with a single specified Host
                subclass.
        """

        filepath = FCPath(filepath)
        suffix = filepath.suffix.lower()
        pds3_options = pds3_options or {}

        # Handle a detached PDS3 label
        if 'P' in formats and suffix == '.lbl':
            try:
                label = Pds3Label(filepath, **pds3_options)
            except PdsError as err:
                raise PdsHostError(str(err)) from err
            except (TypeError, ValueError) as err:
                raise OopsValueError(str(err)) from err
            else:
                for host in hosts:
                    if hasattr(host, '_detect_in_pds3') and host._detect_in_pds3(label):
                        return host, label
                raise Host._no_host_error(hosts, 'PDS3 label', filepath)

        # Handle a FITS file
        if 'F' in formats and suffix in ('.fits', '.fit'):
            local_path = filepath.retrieve()    # a missing file is not a format error
            try:
                hdulist = pyfits.open(local_path)
            except OSError as err:
                raise HostFormatError(f'not a valid FITS file: {filepath}') from err
            else:
                for host in hosts:
                    if hasattr(host, '_detect_in_fits') and host._detect_in_fits(hdulist):
                        return host, hdulist
                raise Host._no_host_error(hosts, 'FITS file', filepath)

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
                raise Host._no_host_error(hosts, 'VICAR file', filepath)

        # Handle an attached PDS3 label
        if 'P' in formats and is_pds3_file(filepath):
            try:
                label = Pds3Label(filepath, **pds3_options)
            except PdsError as err:
                raise PdsHostError(str(err)) from err
            except (TypeError, ValueError) as err:
                raise OopsValueError(str(err)) from err
            else:
                for host in hosts:
                    if hasattr(host, '_detect_in_pds3') and host._detect_in_pds3(label):
                        return host, label
                raise Host._no_host_error(hosts, 'PDS3 file', filepath)

        # Handle another file format
        if 'O' in formats:
            for host in hosts:
                if hasattr(host, '_detect_in_file') and host._detect_in_file(filepath):
                    return host, filepath

        raise Host._no_host_error(hosts, 'file', filepath)

    @staticmethod
    def _no_host_error(hosts, kind, filepath):
        """The exception for a file that none of the listed Host subclasses recognizes.

        Parameters:
            hosts (list[type]): The Host subclasses that were considered.
            kind (str): A description of the file or label, such as "PDS3 label".
            filepath (FCPath): The path to the file.

        Returns:
            UnknownHostError | WrongHostError: WrongHostError if `hosts` contains a single
            Host subclass; otherwise, UnknownHostError.
        """

        if len(hosts) == 1:
            return WrongHostError(f'not a {hosts[0].NAME} {kind}: {filepath}')
        return UnknownHostError(f'unrecognized host in {kind}: {filepath}')

    _TIMESHIFT_CLASS = {'T': TimeShift, 'P': PathShift, 'F': FrameShift}
    _TIMESHIFT_ATTR = {'T': 'cadence', 'P': 'path', 'F': 'frame'}

    @classmethod
    def _read_index_rows(cls, filepath, *, supplement=None, require_supplement=True,
                         validate=None, file_key=None, supplement_key=None,
                         prefer_supplement=False, pds3_options=None, label=None):
        """Read the index rows and return one dictionary per row, with supplementary index
        content merged.

        Each row of the index is merged with the row of the supplement that describes the
        same file. When the two tables list the same files in the same order, the rows are
        paired by position. Otherwise, each index row is paired with the supplement row
        that names the same file, and then every index row must have exactly one such
        supplement row, every supplement row must belong to an index row, and no file may
        be listed twice in either table. Paths are compared without regard to case, to
        empty or "." components (as in a leading "/" or "./" or a doubled slash), or to
        the extension of the file name, so that a row naming a product's label and a row
        naming its data file match. Without a file key, as when the tables do not both
        have the default columns, the rows are paired by position.

        Parameters:
            filepath (str | pathlib.Path | FCPath): Path to the index label.
            supplement (str | pathlib.Path | FCPath | object, optional): The path to the
                PDS label of a supplemental index file, if any. Use the sentinel value
                `Host.SUPPLEMENT` to read the file in the directory of `filepath` whose
                name matches except that "_index" is replaced by "_supplemental_index" in
                upper or lower case; that file must exist.
            require_supplement (bool, optional): If True and `supplement` is specified,
                the supplement file must exist. If False, a supplement file that is
                missing, or that cannot be named because the name of `filepath` contains
                no "_index", is ignored.
            validate (str | list[str] | collections.abc.Callable, optional): The
                mechanism to use to verify that each pair of rows from `filepath` and
                `supplement` match, once they have been paired. If a string or list of
                strings, it identifies column names that the two row dictionaries must
                share. (Tests of strings are case-insensitive and they need not match
                exactly, provided one string falls inside the other.) Alternatively, a
                function with the signature ``(dict[str, Any], dict[str, Any]) -> bool``
                that receives the row dictionaries from each file and returns True if the
                rows match and either returns False or raises an exception if they do
                not match. If unspecified, the column names the two tables share are used,
                except those of the file key, which the pairing compares as paths.
            file_key (str | tuple[str, ...] | collections.abc.Callable, optional): How
                a row of the index names its file: a column name; a tuple of column
                names, whose values are joined with "/", such as ``("VOLUME_ID",
                "PATH_NAME", "FILE_NAME")``; or a function with the signature
                ``(dict[str, Any]) -> str`` that returns the path from a row dictionary.
                If unspecified, the key is FILE_SPECIFICATION_NAME, preceded by VOLUME_ID
                or else VOLUME_NAME if both tables have that column, provided that both
                tables have FILE_SPECIFICATION_NAME.
            supplement_key (str | tuple[str, ...] | collections.abc.Callable, optional):
                How a row of the supplement names its file, in the same forms as
                `file_key`. If unspecified, `file_key` applies to both tables.
            prefer_supplement (bool, optional): True to keep the supplement's value of a
                column that both tables have; False to keep the index's.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.
            label (Pds3Label, optional): The PDS label of `filepath` if it has already
                been read.

        Returns:
            list[dict[str, Any]]: One dictionary per row of the index, in its order.

        Raises:
            HostIndexError: If the two tables cannot be paired: rows paired by position
                in tables with different record counts; an index row that no supplement
                row names; a supplement row that names no index row; a file that either
                table lists more than once while the rows are paired by file; or a key
                column that a table does not have. The message names the first few
                cases and both files. Also if a pair of rows fails `validate`, if no
                columns can be used for cross-validation of the index and supplement, if
                `supplement` is `Host.SUPPLEMENT`, `require_supplement` is True, and the
                name of `filepath` contains no "_index", or if either file cannot be
                interpreted as an index table.
            OopsValueError: If a value in `pds3_options` is invalid.
            OSError: If `filepath` or `supplement` does not exist or cannot be read.
            PdsHostError: If `filepath` or `supplement` uses invalid PDS label syntax.
            WrongHostError: If `filepath` is not associated with this Host subclass.
        """

        def read_table(filepath, label_contents=None, *, pds3_method='fast'):
            """Read a PDS3 table, converting its errors to Host errors."""

            try:
                table = PdsTable(filepath, label_contents=label_contents,
                                 label_method=pds3_method)
            except PdsError as err:
                raise PdsHostError(str(err)) from err
            except (TypeError, ValueError) as err:
                # The options were already validated when the label was parsed, so the
                # problem lies in the table itself
                raise HostIndexError(f'invalid index table {filepath}: {err}') from err
            return table

        def column_validator_func(names):
            """Return a function that tests the rows based on a list of column names."""

            if not names:
                raise HostIndexError('a list of column names is needed for validation')

            def column_validator(indx_dict, supp_dict):
                for name in names:
                    indx_val = indx_dict[name]
                    supp_val = supp_dict[name]
                    if indx_val == supp_val:
                        continue
                    if isinstance(indx_val, str) and isinstance(supp_val, str):
                        indx_upper = indx_val.upper()
                        supp_upper = supp_val.upper()
                        if supp_upper in indx_upper or indx_upper in supp_upper:
                            continue
                    raise HostIndexError(f'{name} value mismatch: {indx_val!r}, '
                                         f'{supp_val!r}')
                return True
            return column_validator

        # Read the index
        filepath = FCPath(filepath)
        pds3_options = Host._pds3_options_with_defaults(pds3_options)
        pds3_method = pds3_options['method']
        if label is None:
            try:
                label = Pds3Label(filepath, **pds3_options)
            except PdsError as err:
                raise PdsHostError(str(err)) from err
            except (TypeError, ValueError) as err:
                raise OopsValueError(str(err)) from err

        row_table = read_table(filepath, label_contents=label, pds3_method=pds3_method)
        row_dicts = row_table.dicts_by_row()

        # Check the host; Host itself accepts any index, because Host.from_index()
        # identifies the host after reading the rows
        if cls is not Host:
            status = cls._detect_in_index(label)
            if status is None:
                status = bool(row_dicts) and cls._detect_in_row(row_dicts[0])
            if not status:
                raise WrongHostError(f'not a {cls.NAME} index: {filepath}')

        if not supplement:
            return row_dicts

        # Look for supplemental index
        if supplement is Host.SUPPLEMENT:
            supp_name = (filepath.name.replace('_index', '_supplemental_index')
                                      .replace('_INDEX', '_SUPPLEMENTAL_INDEX'))
            if supp_name == filepath.name:
                if require_supplement:
                    raise HostIndexError(f'supplemental index required for {filepath}')
                return row_dicts
            supplement = filepath.with_name(supp_name)
        else:
            supplement = FCPath(supplement)

        # Read the supplemental index
        try:
            supp_table = read_table(supplement, pds3_method=pds3_method)
        except FileNotFoundError:
            if require_supplement:
                raise
            return row_dicts

        # Pair each index row with its supplement row
        supp_dicts = supp_table.dicts_by_row()
        files = f'{filepath}, {supplement}'
        if file_key is None:
            file_key = Host._default_file_key(row_table.get_keys(),
                                              supp_table.get_keys())
        if supplement_key is None:
            supplement_key = file_key

        if file_key is None:
            if len(supp_dicts) != len(row_dicts):
                raise HostIndexError(f'index files have different record counts: {files}')
        else:
            row_paths = Host._file_paths(row_dicts, file_key, 'index', filepath)
            supp_paths = Host._file_paths(supp_dicts, supplement_key, 'supplement',
                                          supplement)
            if ([Host._path_key(path) for path in row_paths]
                    != [Host._path_key(path) for path in supp_paths]):
                supp_dicts = Host._pair_by_file(row_paths, supp_paths, supp_dicts,
                                                files=files)

        # Check each pair. By default, the columns that name the file are left out,
        # because the pairing has compared them already, as paths, and the two tables may
        # name the same product by its label and by its data file.
        if validate is None:
            shared = set(row_table.get_keys()) & set(supp_table.get_keys())
            if not shared:
                raise HostIndexError(f'index files have no columns in common: {files}')
            validate = sorted(shared - Host._key_columns(file_key, supplement_key))
            if not validate:
                return Host._merge_rows(row_dicts, supp_dicts,
                                        prefer_supplement=prefer_supplement)
        elif isinstance(validate, str):
            validate = [validate]
        if isinstance(validate, (list, tuple)):
            validate = column_validator_func(validate)

        for i, row_dict in enumerate(row_dicts):
            try:
                status = validate(row_dict, supp_dicts[i])
            except Exception as err:
                raise HostIndexError(str(err)
                                     + f'\nindex file mismatch at record [{i}]: '
                                       f'{files}') from err
            if not status:
                raise HostIndexError(f'index file mismatch at record [{i}]: {files}')

        return Host._merge_rows(row_dicts, supp_dicts,
                                prefer_supplement=prefer_supplement)

    @staticmethod
    def _key_columns(file_key, supplement_key):
        """The names of the columns that make up the file keys of two tables.

        Parameters:
            file_key (str | tuple[str, ...] | collections.abc.Callable | None): The key of
                the index.
            supplement_key (str | tuple[str, ...] | collections.abc.Callable | None): The
                key of the supplement.

        Returns:
            set[str]: Every column named by either key; a function names none.
        """

        names = set()
        for key in (file_key, supplement_key):
            if isinstance(key, str):
                names.add(key)
            elif isinstance(key, (list, tuple)):
                names.update(key)
        return names

    @staticmethod
    def _merge_rows(row_dicts, supp_dicts, *, prefer_supplement):
        """Each index row merged with its paired supplement row.

        Parameters:
            row_dicts (list[dict[str, Any]]): The index rows.
            supp_dicts (list[dict[str, Any]]): The supplement row for each index row.
            prefer_supplement (bool): True to keep the supplement's value of a column that
                both tables have; False to keep the index's.

        Returns:
            list[dict[str, Any]]: The merged rows, in index order.
        """

        if prefer_supplement:
            return [row_dict | supp_dict
                    for (row_dict, supp_dict) in zip(row_dicts, supp_dicts, strict=True)]
        return [supp_dict | row_dict
                for (row_dict, supp_dict) in zip(row_dicts, supp_dicts, strict=True)]

    @staticmethod
    def _default_file_key(row_names, supp_names):
        """The default columns that name the file of a row of an index and a supplement.

        Parameters:
            row_names (list[str]): The column names of the index.
            supp_names (list[str]): The column names of the supplement.

        Returns:
            tuple[str, ...] | None: FILE_SPECIFICATION_NAME, preceded by VOLUME_ID or else
            VOLUME_NAME if both tables have that column; None if the tables do not both
            have FILE_SPECIFICATION_NAME.
        """

        shared = set(row_names) & set(supp_names)
        if 'FILE_SPECIFICATION_NAME' not in shared:
            return None
        for volume in ('VOLUME_ID', 'VOLUME_NAME'):
            if volume in shared:
                return (volume, 'FILE_SPECIFICATION_NAME')
        return ('FILE_SPECIFICATION_NAME',)

    @staticmethod
    def _file_paths(row_dicts, file_key, kind, filepath):
        """The path of the file that each row of a table names, as the table gives it.

        Parameters:
            row_dicts (list[dict[str, Any]]): The rows of the table.
            file_key (str | tuple[str, ...] | collections.abc.Callable): A column name,
                a tuple of column names whose values are joined with "/", or a function
                that returns the path from a row dictionary.
            kind (str): "index" or "supplement", for error messages.
            filepath (FCPath): The path to the table, for error messages.

        Returns:
            list[str]: The path for each row.

        Raises:
            HostIndexError: If the table does not have a key column.
        """

        if callable(file_key):
            return [file_key(row_dict) for row_dict in row_dicts]

        names = (file_key,) if isinstance(file_key, str) else tuple(file_key)
        if row_dicts:
            missing = [name for name in names if name not in row_dicts[0]]
            if missing:
                raise HostIndexError(f'{kind} has no column {missing[0]}: {filepath}')
        return ['/'.join(str(row_dict[name]) for name in names) for row_dict in row_dicts]

    @staticmethod
    def _path_key(path):
        """The form of a path in which an index and its supplement are compared.

        Parameters:
            path (str): A path as a table gives it.

        Returns:
            str: The path in upper case, without empty or "." components or the extension
            of the file name.
        """

        parts = [part for part in path.strip().upper().split('/')
                 if part not in ('', '.')]
        if parts:
            parts[-1] = parts[-1].rpartition('.')[0] or parts[-1]
        return '/'.join(parts)

    @staticmethod
    def _pair_by_file(row_paths, supp_paths, supp_dicts, *, files):
        """The supplement rows reordered to match the index rows, file by file.

        Parameters:
            row_paths (list[str]): The path named by each index row.
            supp_paths (list[str]): The path named by each supplement row.
            supp_dicts (list[dict[str, Any]]): The supplement rows.
            files (str): The paths to the index and the supplement, for error messages.

        Returns:
            list[dict[str, Any]]: The supplement row for each index row, in index order.

        Raises:
            HostIndexError: If a file is listed more than once in either table, if an
                index row has no supplement row, or if a supplement row names no index
                row. The message lists the first few such paths, as the table gives them.
        """

        def report(problem, paths):
            """The message for a problem with the given paths."""

            return f'{problem} ({len(paths)}): {paths[:_PATHS_IN_MESSAGE]}; {files}'

        row_keys = [Host._path_key(path) for path in row_paths]
        supp_keys = [Host._path_key(path) for path in supp_paths]

        for (kind, keys, paths) in (('index', row_keys, row_paths),
                                    ('supplement', supp_keys, supp_paths)):
            counts = Counter(keys)
            repeated = list({key: path for (key, path) in zip(keys, paths)
                             if counts[key] > 1}.values())
            if repeated:
                raise HostIndexError(report(f'{kind} lists a file more than once',
                                            repeated))

        supp_by_key = dict(zip(supp_keys, supp_dicts))
        unpaired = [path for (key, path) in zip(row_keys, row_paths)
                    if key not in supp_by_key]
        if unpaired:
            raise HostIndexError(report('no supplement row for an index row', unpaired))

        row_key_set = set(row_keys)
        extra = [path for (key, path) in zip(supp_keys, supp_paths)
                 if key not in row_key_set]
        if extra:
            raise HostIndexError(report('no index row for a supplement row', extra))

        return [supp_by_key[key] for key in row_keys]

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
            lightsource (LightSource, optional): Override for the default LightSource of
                the observation.
            path (str | Path, optional): Override for the Path of the observer.
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

        Raises:
            OopsValueError: If an input parameter is invalid.
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
                raise OopsValueError('the timeshift must specify one or more of T/P/F')
            if set(letters) - set('TPF'):
                raise OopsValueError(f'invalid timeshift letters: {letters}')
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
                raise OopsValueError('parallel and tracker cannot both be specified')

            frame = Frame.as_frame(params['frame']).wrt(parallel.frame)
            xform = frame.transform_at_time(params['cadence'].midtime)
            params['frame'] = Cmatrix(xform.matrix, reference=parallel.frame)

        elif tracker is not None:
            try:
                tfrac = {'start': 0., 'midtime': 0.5, 'end': 1.}[tracker]
            except KeyError:
                raise OopsValueError('invalid tracker, should be "start", "midtime", or '
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
            observations (ObsArray): The observations from one file.
            select (int | slice | str | tuple[int | slice | str, ...] | None): An index, a
                slice, or the name of one observation, or a tuple of these; None to choose
                all the `observations`. A tuple indexes the observations as NumPy indexes
                an array, one item per axis, so `(0, 0)` chooses the first observation of
                a 2x2 array and `(slice(None), 0)` chooses its first column.
            names (dict[str, int], optional): The host-specific names of the observations,
                in upper case, each mapped to its index in `observations`. Names in
                `select` are matched without regard to case. In a tuple, a name may stand
                for the index on any axis.

        Returns:
            Observation | ObsArray | None: `observations` itself if `select` is None; the
            one observation or row chosen by an index or a name; the tuple or array of
            the observations chosen by a slice; or, for a tuple, the observation or the
            array of observations it indexes, which is a tuple if `observations` is one.

        Raises:
            OopsIndexError: If an index is out of range, or if a tuple has more items than
                `observations` has axes.
            OopsValueError: If a name is not one of `names`.
            OopsTypeError: If `select`, or an item of a tuple, is not an int, slice or
                str.
        """

        def name_index(name):
            if name.upper() not in (names or {}):
                raise OopsValueError(f'unrecognized select name: {name!r}')
            return names[name.upper()]

        if select is None:
            return observations

        if isinstance(select, tuple):
            indices = []
            for item in select:
                if isinstance(item, str):
                    indices.append(name_index(item))
                elif (isinstance(item, slice)
                      or (isinstance(item, numbers.Integral)
                          and not isinstance(item, bool))):
                    indices.append(item)
                else:
                    raise OopsTypeError(f'invalid select item: {item!r}')

            # A tuple of observations is indexed as a 1-D object array, and any array it
            # yields is returned as a tuple again
            array = observations
            if not isinstance(array, np.ndarray):
                array = np.empty(len(observations), dtype=np.object_)
                for i, obs in enumerate(observations):
                    array[i] = obs
            try:
                chosen = array[tuple(indices)]
            except IndexError as err:
                raise OopsIndexError(f'invalid select {select!r}: {err}') from err
            if isinstance(chosen, np.ndarray) and not isinstance(observations,
                                                                 np.ndarray):
                return tuple(chosen)
            return chosen

        if isinstance(select, slice):
            return observations[select]

        if isinstance(select, str):
            return observations[name_index(select)]

        if isinstance(select, numbers.Integral) and not isinstance(select, bool):
            try:
                return observations[select]
            except IndexError as err:
                raise OopsIndexError(f'select index out of range: {select}') from err

        raise OopsTypeError(f'invalid select: {select!r}')

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
