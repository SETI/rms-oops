##########################################################################################
# oops/host/host_.py
##########################################################################################

import itertools
import numbers
import pathlib
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from astropy.io import fits as pyfits
from filecache  import FCPath
from pdsparser  import (Pds3Label, PdsError, is_pds3_file, read_label,
                        read_vax_binary_label)
from vicar      import VicarError, VicarImage, VicarLabel

from oops._exceptions import OopsIndexError, OopsTypeError, OopsValueError
from oops.body        import Body
from oops.cadence     import TimeShift
from oops.frame       import Cmatrix, Frame, FrameShift, Navigation, TrackerFrame
from oops.observation import Observation
from oops.path        import PathShift

from .        import _index_rows
from ._errors import (HostError, HostFormatError, HostIndexError, PdsHostError,
                      UnknownHostError, VicarHostError, WrongHostError)

ObsArray: TypeAlias = (np.ndarray[tuple[int, ...], np.dtype[np.object_]]
                       | tuple[Observation | None, ...])


class Host:
    """Superclass of all host modules.

    A host subclass reads the data files of one instrument. Its ``from_file`` reads one
    file and returns its observations; its ``from_index`` reads a PDS3 index and returns
    an iterator that yields the observations of one row at a time, building each when the
    iteration reaches that row. The support methods that a host overrides or calls are
    private and are described on the documentation page of :mod:`oops.host`.
    """

    _HOSTS = []     # A list of all registered Host subclasses.
    _LOOKUP = {}    # name -> subclass

    #: The name of the host, such as "Cassini ISS", for messages. Each subclass defines
    #: it.
    NAME = 'none'

    #: How the rows of an index and of its supplement are paired. False pairs each index
    #: row with the supplement row in the same position; True pairs it with the
    #: supplement row that gives the same file, as :attr:`INDEX_FILE_KEY` and
    #: :attr:`SUPPLEMENT_FILE_KEY` give it, wherever that row is. A host sets it once,
    #: for every index of its instrument.
    MERGE_SUPPLEMENT_BY_FILE = False

    #: The names of the index columns whose values, joined with "/", give the file of an
    #: index row. A host whose index gives the file in other columns names them here.
    INDEX_FILE_KEY = ('VOLUME_ID', 'FILE_SPECIFICATION_NAME')

    #: The names of the supplement columns whose values, joined with "/", give the file
    #: of a supplement row. A host whose supplement gives the file in other columns names
    #: them here.
    SUPPLEMENT_FILE_KEY = ('VOLUME_ID', 'FILE_SPECIFICATION_NAME')

    #: The default value of the `supplement` argument of :meth:`from_index`, which
    #: stands for the supplement that ``_supplement_path`` names beside the index.
    SUPPLEMENT = object()

    @staticmethod
    def from_file(filepath, *, pds3_options=None, select=None, astrometry=False,
                  target=None, lightsource=None, path=None, frame=None, cadence=None,
                  fov=None, calibrations=None, timeshift=None, navigation=None,
                  tracker=None, parallel=None, **kwargs):
        """One or more Observation objects derived from a given data file.

        Every registered host is asked whether it recognizes the file, from its PDS3
        label, VICAR header, FITS header, or name, and exactly one must claim it. A PDS3
        label is parsed for this as it is written, with the "strict" method unless
        `pds3_options` selects another. If it cannot be parsed, whether for invalid
        syntax or for a value the parser cannot convert, it is offered to the
        correction of label text of each registered host that has one: a host claims
        the label when its correction makes the label readable and it then recognizes
        the corrected label.

        The host that claims the file then reads it with its own ``from_file``, which is
        given every argument as the caller gave it, `pds3_options` included. A PDS3
        label is passed to it as its path, so that the host parses the label with its
        own method and correction, and so is a FITS file, which is opened only to read its
        header and is closed again before the host is given the path; a VICAR file is
        passed as the content that was read to identify it.

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
                instrument.
            cadence (Cadence, optional): Override for the Cadence of the observation.
            fov (FOV, optional): Override for the default FOV of the observing instrument.
            calibrations (Calibration | list[Calibration], optional): Override for the
                calibration or list of calibrations. The default None keeps the
                calibrations that the host attaches; an empty list removes them.
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
            HostError: If more than one registered Host subclass claims `filepath`. The
                message names each of them.
            HostFormatError: If `filepath` has invalid syntax or uses an unsupported
                option. Note that subclass :class:`~PdsHostError` is raised for a PDS3
                syntax error and :class:`~VicarHostError` is raised for a VICAR syntax
                error, or for a truncated or unreadable VICAR file, naming the file. A
                PDS3 label that cannot be parsed raises :class:`~PdsHostError`, or
                :class:`~oops.OopsValueError` for a value that cannot be converted,
                only if no host's correction of label text makes it readable and
                recognizable.
            OSError: If `filepath` does not exist or cannot be read.
            UnknownHostError: If `filepath` is not associated with any registered Host
                subclass.
            OopsValueError: If a value in `pds3_options` is invalid.
        """

        # The host is identified with the default PDS3 options, method="strict" and
        # sources=True, unless the caller gives others
        host, fileinfo = Host._identify_host(
            filepath, astrometry=astrometry,
            pds3_options=Host._pds3_options_with_defaults(pds3_options))

        # The host parses a PDS3 label itself, with its own method and correction, and
        # reads a FITS file itself, so the file is closed after it identified the host
        if isinstance(fileinfo, Pds3Label):
            fileinfo = filepath
        elif isinstance(fileinfo, pyfits.HDUList):
            fileinfo.close()
            fileinfo = filepath

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

        The host is identified when this method is called. The index label is parsed as
        it is written, with the "strict" method unless `pds3_options` selects another, and
        every registered host is asked whether it recognizes the label. If no host does,
        the first row of the index is read, and only that row, and offered to each host
        that could not tell from the label. Exactly one host must claim the index.

        The host's own ``from_index`` is then given the path and every argument as the
        caller gave it, `label` and `pds3_options` included; the label parsed here is not
        passed on. It reads the rows of the index and its supplement itself, a few at a
        time as the iteration proceeds, and builds each row's observation when the
        iteration reaches that row. Besides the exceptions listed here, the call and the
        iteration raise any that the host's ``from_index`` documents.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to the PDS label of a
                data file index.
            supplement (str | pathlib.Path | FCPath | object | None, optional): The path
                to the PDS label of a supplemental index, whose columns are merged into
                each row; None for no supplement. The default, `Host.SUPPLEMENT`, stands
                for the file in the directory of `filepath` that the host names, by
                default the one whose name matches except that "_index" is replaced by
                "_supplemental_index" in upper or lower case. The host determines whether
                a supplement must exist and how its rows are paired with the index rows.
            label (Pds3Label, optional): The PDS label of `filepath` if it has already
                been read.
            row_dicts (collections.abc.Iterable[dict[str, Any]], optional): The rows of
                the index, one dictionary per row, already merged with the supplement. If
                provided, the tables are not read and `supplement` is ignored. If the host
                must be identified from a row, the first one is read for that and still
                passed on.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.
            filter (dict[str, Any], optional): A dictionary of index column names and
                their values or a set or tuple of values. If provided, only rows of the
                index in which each named column equals the value, falls in the set of
                values, or falls in a range defined by a tuple of two values, bounds
                included, are yielded. Each row is tested when the iteration reaches it.
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
            collections.abc.Iterator[Observation | ObsArray | None]: The host's iterator,
            which yields the Observation, or the tuple or array of Observations, for each
            filtered row of the index, in index order; None where a selected Observation
            is absent from that row. A caller that needs ``len()`` or indexing wraps the
            call in ``list()``.

        Raises:
            HostError: If more than one registered Host subclass claims the index, from
                its label or from its first row. The message names each of them.
            HostIndexError: If the label of `filepath` does not describe a table that can
                be read, when the first row is needed to identify the host.
            OopsValueError: If a value in `pds3_options` is invalid, or if the parser
                cannot convert a value in the index label.
            OSError: If `filepath` does not exist or cannot be read.
            PdsHostError: If `filepath` uses invalid PDS label syntax.
            UnknownHostError: If `filepath` is not associated with any registered Host
                subclass.
        """

        filepath = FCPath(filepath)
        parsed = label
        if parsed is None:
            parsed = Host._parse_pds3(filepath,
                                      Host._pds3_options_with_defaults(pds3_options))

        # Every host is asked about the label. A host whose detector returns None can
        # only be identified from the content of a row, so if no host claims the label,
        # the first row, and that row alone, is offered to each of those hosts.
        status = {host: host._detect_in_index(parsed) for host in Host._HOSTS}
        host = Host._one_host([host for host in Host._HOSTS if status[host]],
                              f'index {filepath}')
        if host is None:
            undecided = [host for host in Host._HOSTS if status[host] is None]
            if undecided:
                if row_dicts is None:
                    table = _index_rows.open_table(filepath, label=parsed)
                    first = _index_rows.first_row(table, filepath)
                else:
                    (first, row_dicts) = Host._peek(row_dicts)
                if first is not None:
                    host = Host._one_host([host for host in undecided
                                           if host._detect_in_row(first)],
                                          f'first row of the index {filepath}')
            if host is None:
                raise UnknownHostError(f'unknown host in index {filepath}')

        # The host reads, pairs and filters the rows itself, because only the host knows
        # how its supplement is paired with its index
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

    @classmethod
    def _correct_label_text(cls, text, filepath):
        """The text of a PDS3 product label of this host, corrected so that it can be
        parsed.

        A host whose product labels hold text that the PDS3 parser cannot read, or reads
        wrongly, overrides this class method to correct that text, so that the oops
        readers stay free of any one data set's corrections. It is called with the text
        of the label before the parser runs, whenever :meth:`_read_fileinfo` parses a
        detached or attached PDS3 label for this host. The corrected text is parsed in
        place of the file's, and the parsed label keeps the path of the file, so that its
        pointers are found beside it. The labels of an index and of its supplement are
        parsed as they are written, without this method.

        :meth:`from_file` also offers this method the labels of other hosts: when it
        cannot parse a PDS3 label as written, it tries the correction of every registered
        host that overrides this method, and the host whose correction makes the label
        readable and that then recognizes the corrected label claims the file. A
        correction must therefore change only the text that has the defect it corrects,
        and return any other text as it is.

        The default returns `text` unchanged, and a host that keeps it has its labels
        parsed from their files.

        Parameters:
            text (str): The text of the label, up to its END statement for an attached
                label, with <LF> line terminators.
            filepath (FCPath): The path to the file that holds the label, so that a host
                can decide for each file.

        Returns:
            str: The corrected text; `text` itself if nothing needs to change.
        """

        return text

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
                :class:`~VicarHostError` is returned for VICAR files, including a
                truncated or unreadable one, naming the file.
            OSError: If `filepath` does not exist or cannot be read.
            OopsValueError: If a value in `pds3_options` is invalid.
        """

        # Check every registered host and every defined file format
        return Host._identify_and_parse(filepath, hosts=Host._HOSTS, formats='PVFO',
                                        pds3_options=pds3_options, astrometry=astrometry)

    @classmethod
    def _read_fileinfo(cls, fileinfo, *, formats, astrometry=False, pds3_options=None):
        """Read and parse the file content for this class in PDS3, VICAR, or FITS format.

        This is a support method for the `from_file` methods of the Host subclasses. The
        text of a PDS3 label is corrected by this host's :meth:`_correct_label_text`
        before it is parsed.

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
                :class:`~VicarHostError` is returned for VICAR files, including a
                truncated or unreadable one, naming the file.
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
                                             pds3_options=pds3_options,
                                             correct=cls._label_corrector())
        return parsed

    @staticmethod
    def _identify_and_parse(filepath, hosts, *, formats='PVFO', astrometry=False,
                            pds3_options=None, correct=None):
        """Interpret the filepath and identify the associated host module.

        Also return the parsed file content for PDS3, VICAR, and FITS formats. With
        `correct`, the text of a PDS3 label is corrected before it is parsed. Without it,
        the label is parsed as it is written, and if it cannot be parsed, it is parsed
        again with the correction of label text of each host in `hosts` that has one;
        each host that recognizes the label as its correction gives it claims the file.

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
            correct (collections.abc.Callable, optional): A function with the signature
                ``(str, FCPath) -> str`` that corrects the text of a PDS3 label, such as
                a host's :meth:`_correct_label_text`; None to parse the label as it is
                written.

        Returns:
            tuple[type, (Pds3Label | VicarLabel | VicarImage | HDUList | FCPath)]:
            The Host subclass associated with the `filepath`, followed by the parsed
            content of its label, if any. If the file is not in a standard format, the
            second returned item is the FCPath. For a PDS3 label that a host's
            correction made readable, the label is the corrected one.

        Raises:
            HostError: If more than one of the listed Host subclasses claims `filepath`.
                The message names each of them.
            HostFormatError: If `filepath` has invalid syntax for its format. Note that
                subclass :class:`~PdsHostError` is returned for PDS-formatted files and
                :class:`~VicarHostError` is returned for VICAR files, including a
                truncated or unreadable one, naming the file.
            OopsValueError: If the value in `pds3_options` is invalid.
            OSError: If `filepath` does not exist or cannot be read.
            UnknownHostError: If `filepath` is not associated with any of several listed
                Host subclasses.
            WrongHostError: If `filepath` is not associated with a single specified Host
                subclass.
        """

        def identify_pds3(kind):
            """The host and the parsed label of a PDS3 file."""

            try:
                label = Host._parse_pds3(filepath, pds3_options, correct=correct)
            except (PdsHostError, OopsValueError):
                if correct is not None:
                    raise

                # Offer the label to the correction of each host that has one
                found = []
                for host in hosts:
                    corrector = host._label_corrector()
                    if corrector is None:
                        continue
                    try:
                        corrected = Host._parse_pds3(filepath, pds3_options,
                                                     correct=corrector)
                    except (PdsHostError, OopsValueError):
                        continue
                    if host._detect_in_pds3(corrected):
                        found.append((host, corrected))
                if not found:
                    raise
                Host._one_host([host for (host, _) in found], f'{kind} {filepath}')
                return found[0]

            host = Host._one_host([host for host in hosts if host._detect_in_pds3(label)],
                                  f'{kind} {filepath}')
            if host is None:
                raise Host._no_host_error(hosts, kind, filepath)
            return host, label

        filepath = FCPath(filepath)
        suffix = filepath.suffix.lower()
        pds3_options = pds3_options or {}

        # Handle a detached PDS3 label
        if 'P' in formats and suffix == '.lbl':
            return identify_pds3('PDS3 label')

        # Handle a FITS file
        if 'F' in formats and suffix in ('.fits', '.fit'):
            local_path = filepath.retrieve()    # a missing file is not a format error
            try:
                hdulist = pyfits.open(local_path)
            except OSError as err:
                raise HostFormatError(f'not a valid FITS file: {filepath}') from err
            try:
                host = Host._one_host(
                    [host for host in hosts if host._detect_in_fits(hdulist)],
                    f'FITS file {filepath}')
                if host is None:
                    raise Host._no_host_error(hosts, 'FITS file', filepath)
            except BaseException:
                hdulist.close()
                raise
            return host, hdulist

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
            except ValueError as err:   # as for a truncated array
                raise VicarHostError(f'cannot read VICAR file {filepath}: {err}') from err
            host = Host._one_host([host for host in hosts if host._detect_in_vicar(vic)],
                                  f'VICAR file {filepath}')
            if host is None:
                raise Host._no_host_error(hosts, 'VICAR file', filepath)
            return host, vic

        # Handle an attached PDS3 label
        if 'P' in formats and is_pds3_file(filepath):
            return identify_pds3('PDS3 file')

        # Handle another file format
        if 'O' in formats:
            host = Host._one_host([host for host in hosts
                                   if host._detect_in_file(filepath)],
                                  f'file {filepath}')
            if host is not None:
                return host, filepath

        raise Host._no_host_error(hosts, 'file', filepath)

    @staticmethod
    def _parse_pds3(filepath, pds3_options, *, correct=None):
        """The parsed PDS3 label of a file.

        Parameters:
            filepath (FCPath): The path to the label or to a file with an attached label.
            pds3_options (dict[str, Any]): The keyword arguments of the
                :class:`~pdsparser.Pds3Label` parser.
            correct (collections.abc.Callable, optional): A function with the signature
                ``(str, FCPath) -> str`` that corrects the text of the label before it is
                parsed, such as a host's :meth:`_correct_label_text`; None to parse the
                file as it is written.

        Returns:
            Pds3Label: The label, whose `filepath` is `filepath`.

        Raises:
            OopsValueError: If `pds3_options` holds an invalid value, or if the parser
                cannot convert a value in the label.
            OSError: If `filepath` does not exist or cannot be read.
            PdsHostError: If the label, as corrected, uses invalid PDS3 syntax, or if its
                text cannot be read for `correct`. The message has the form "cannot parse
                the PDS3 label <filepath>: <parser's message>".
        """

        filepath = FCPath(filepath)
        (source, options) = Host._pds3_source(filepath, pds3_options, correct)
        try:
            return Host._pds3_label(source, filepath, options)
        except PdsError as err:
            raise PdsHostError(f'cannot parse the PDS3 label {filepath}: {err}') from err
        except (TypeError, ValueError) as err:
            raise OopsValueError(str(err)) from err

    @classmethod
    def _label_corrector(cls):
        """This host's correction of label text, if it has one.

        Returns:
            collections.abc.Callable | None: The host's :meth:`_correct_label_text`; None
            if the host keeps the default, whose labels are then parsed from their files.
        """

        if cls._correct_label_text.__func__ is Host._correct_label_text.__func__:
            return None
        return cls._correct_label_text

    @staticmethod
    def _pds3_source(filepath, pds3_options, correct):
        """What the PDS3 parser reads for a label, with the options to read it.

        Parameters:
            filepath (FCPath): The path to the label or to a file with an attached label.
            pds3_options (dict[str, Any]): The keyword arguments of the
                :class:`~pdsparser.Pds3Label` parser.
            correct (collections.abc.Callable | None): The correction of the label's text;
                None for none.

        Returns:
            tuple[FCPath | str, dict[str, Any]]: `filepath` and `pds3_options` if
            `correct` is None; otherwise the corrected text, and the options with the
            directory of `filepath` first among the directories searched for a ".FMT"
            file, as the parser searches it for a label read from a file. The text is
            read as the parser reads it from the file.

        Raises:
            OSError: If the file does not exist or cannot be read.
            PdsHostError: If the text of the label cannot be read, as for an attached
                label that lacks its END statement.
        """

        if correct is None:
            return (filepath, pds3_options)

        try:
            if pds3_options.get('method') == 'compound':
                text = filepath.read_text(encoding='latin-1')
            elif pds3_options.get('vax'):
                text = read_vax_binary_label(filepath)
            else:
                text = read_label(filepath)
        except PdsError as err:
            raise PdsHostError(f'cannot parse the PDS3 label {filepath}: {err}') from err
        corrected = correct(text, filepath)

        # The parser takes a string that has no line terminator for the path of a file
        if '\n' not in corrected:
            corrected += '\n'

        fmt_dirs = pds3_options.get('fmt_dirs', [])
        fmt_dirs = ([fmt_dirs] if isinstance(fmt_dirs, (str, pathlib.Path, FCPath))
                    else list(fmt_dirs))
        options = pds3_options | {'fmt_dirs': [filepath.parent, *fmt_dirs]}
        options.pop('vax', None)        # the text has been read already
        return (corrected, options)

    @staticmethod
    def _pds3_label(source, filepath, pds3_options):
        """A PDS3 label parsed from its file or from corrected text.

        Parameters:
            source (FCPath | str): The file, or the corrected text, as
                :meth:`_pds3_source` gives them.
            filepath (FCPath): The path to the file that holds the label.
            pds3_options (dict[str, Any]): The keyword arguments of the
                :class:`~pdsparser.Pds3Label` parser.

        Returns:
            Pds3Label: The label, whose `filepath` is `filepath` in either case, so that
            the files its pointers refer to are found beside it.
        """

        label = Pds3Label(source, **pds3_options)
        if label.filepath is None:
            label.filepath = filepath
            label._filepath = filepath      # the parser's other name for it
        return label

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

    @staticmethod
    def _one_host(claimants, what):
        """The one Host subclass that claims a file, if any does.

        Parameters:
            claimants (list[type]): The Host subclasses that claim the file, in the order
                in which they were registered.
            what (str): The file, for the message, such as "PDS3 label <filepath>".

        Returns:
            type | None: The one Host subclass in `claimants`; None if it is empty.

        Raises:
            HostError: If `claimants` holds more than one Host subclass. The message reads
                "<what> is claimed by more than one host: <name>, <name>", with the
                :attr:`NAME` of each.
        """

        if len(claimants) > 1:
            names = ', '.join(host.NAME for host in claimants)
            raise HostError(f'{what} is claimed by more than one host: {names}')
        return claimants[0] if claimants else None

    _TIMESHIFT_CLASS = {'T': TimeShift, 'P': PathShift, 'F': FrameShift}
    _TIMESHIFT_ATTR = {'T': 'cadence', 'P': 'path', 'F': 'frame'}

    @classmethod
    def _index_rows(cls, filepath, *, supplement=SUPPLEMENT, require_supplement=True,
                    label=None, row_dicts=None, pds3_options=None, filter=None,
                    validate=None):
        """An iterator over the rows of an index, each merged with its supplement row and
        filtered, which reads the rows as the iteration proceeds.

        This is the support method with which a host's ``from_index`` reads its index;
        the host builds one observation from each row that the iterator yields.

        The call itself reads the index label, unless `label` is given, and checks that
        the index belongs to this host; if the index label cannot tell, the first row is
        read for that, and only that row. It also reads the label of the supplement. The
        iteration then reads the index and the supplement 100 rows at a time. Each row is
        a dictionary as ``PdsTable.iter_dicts`` gives it: every value as the
        table gives it, declared null values included, with the "_mask" entry of each
        column.

        How each index row is paired with its supplement row is fixed by the host's
        :attr:`MERGE_SUPPLEMENT_BY_FILE`:

        - False, pairing by position: the supplement is read in step with the index, and
          at each row, the file that the index row gives must be the file that the
          supplement row gives.
        - True, pairing by file: the supplement is read ahead only as far as the row that
          gives the same file as the current index row. A supplement row read before its
          partner is held until the partner is reached, so a supplement whose rows are k
          places out of order holds at most k rows. An index row whose file the
          supplement does not list is found only when the whole supplement has been
          read.

        The file of a row is the values of the columns that :attr:`INDEX_FILE_KEY` or
        :attr:`SUPPLEMENT_FILE_KEY` names, joined with "/". Files are compared without
        regard to case, to empty or "." components (as in a leading "/" or "./"), or to
        the extension of the file name, so that a row that gives a product's label and a
        row that gives its data file match. Each pair is then checked with `validate`,
        merged, and tested against `filter`. Where both rows have a column, the merged
        row has the index row's value. Rows are counted from 0 in the messages of errors.

        Parameters:
            filepath (str | pathlib.Path | FCPath): Path to the index label.
            supplement (str | pathlib.Path | FCPath | object | None, optional): The path
                to the PDS label of the supplemental index, in any directory; None to read
                the index alone. The default, `Host.SUPPLEMENT`, reads the file that
                :meth:`_supplement_path` returns.
            require_supplement (bool, optional): True if the supplement file must exist;
                False to read the index alone if the supplement file is missing, or if
                `supplement` is `Host.SUPPLEMENT` and :meth:`_supplement_path` returns
                None. A host that requires a supplement only when the caller names one
                passes ``require_supplement=supplement is not Host.SUPPLEMENT``.
            label (Pds3Label | dict[str, Any], optional): The content of the label of
                `filepath`, if it has already been read.
            row_dicts (collections.abc.Iterable[dict[str, Any]], optional): The rows of
                the index, already merged with the supplement. If they are given, the
                tables are not read, and the iterator yields each of them that passes
                `filter`.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser; the
                method is "strict" unless they give another. The index label is parsed
                with all of them, and the supplement's label with their method alone.
            filter (dict[str, Any], optional): Column names, each with a value, a set,
                list, or tuple of values, or a tuple of two values that bound a range. A
                row is yielded only if every column it lists equals the value, is one of
                the values, or lies within the range, bounds included.
            validate (str | list[str] | tuple[str, ...] | collections.abc.Callable, optional):
                The check of each pair of rows. A column name, or a list or tuple of
                them, gives columns whose values the two rows must share; two strings are
                the same if either one, in upper case, contains the other. An empty list
                or tuple skips the comparison of columns, so the only check is the
                pairing's comparison of the files. A function with the signature
                ``(dict[str, Any], dict[str, Any]) -> bool`` receives the index row and
                the supplement row and returns True if they match; it either returns
                False or raises an exception if they do not. If unspecified, the columns
                that the two tables share are compared, except those of the file keys,
                which the pairing compares as files.

        Returns:
            collections.abc.Iterator[dict[str, Any]]: One dictionary for each row of the
            index that passes `filter`, in index order, with the columns of its
            supplement row.

        Raises:
            HostIndexError: At call time: if either label does not describe a table that
                can be read; if `supplement` is `Host.SUPPLEMENT`, `require_supplement` is
                True, and :meth:`_supplement_path` returns None; or if `validate` is
                unspecified and the two tables do not share any column. During the
                iteration, at the row concerned: an index row whose file is not the file
                of the supplement row in the same position (pairing by position); an index
                row for which the supplement lacks a row, which with pairing by file is
                raised once the whole supplement has been read; a supplement row that
                repeats the file of a row that is still held (pairing by file); a pair of
                rows that fails `validate`; the end of a table file that holds fewer rows
                than its label says, after every row it holds; a row longer than the
                label's row length, raised at the first of the 100 rows read with it.
                After the last index row: a supplement row that was not paired with an
                index row, or a table file that holds more rows than its label says.
            KeyError: During the iteration, at a row that lacks a column that `filter` or
                a file key names.
            OopsValueError: At call time, if a value in `pds3_options` is invalid, or if
                the parser cannot convert a value in a label.
            OSError: At call time, if a label does not exist or cannot be read, unless it
                is the supplement's and `require_supplement` is False. When the
                iteration starts, if a table file does not exist or cannot be read.
            PdsHostError: At call time, if a label uses invalid PDS3 syntax.
            WrongHostError: At call time, if the index does not belong to this host.
        """

        if row_dicts is not None:
            return _index_rows.filtered(row_dicts, filter)

        # Read the index label and check the host
        filepath = FCPath(filepath)
        options = Host._pds3_options_with_defaults(pds3_options)
        if label is None:
            label = Host._parse_pds3(filepath, options)
        index = _index_rows.open_table(filepath, label=label)
        if cls is not Host:
            status = cls._detect_in_index(label)
            if status is None:
                row = _index_rows.first_row(index, filepath)
                status = row is not None and cls._detect_in_row(row)
            if not status:
                raise WrongHostError(f'not a {cls.NAME} index: {filepath}')
        index_rows = _index_rows.stream_rows(index, filepath)

        # Read the supplement label
        supplement_path = None
        if supplement is Host.SUPPLEMENT:
            supplement_path = cls._supplement_path(filepath)
            if supplement_path is None and require_supplement:
                raise HostIndexError(f'supplemental index required for {filepath}')
        elif supplement:
            supplement_path = FCPath(supplement)

        supp_table = None
        if supplement_path is not None:
            try:
                supp_table = _index_rows.open_table(supplement_path,
                                                    method=options['method'])
            except FileNotFoundError:
                if require_supplement:
                    raise

        if supp_table is None:
            return _index_rows.filtered(index_rows, filter)

        # By default, the columns of the file keys are left out of the check, because the
        # pairing compares them already, as files
        files = f'{filepath}, {supplement_path}'
        if validate is None:
            shared = set(index.get_keys()) & set(supp_table.get_keys())
            if not shared:
                raise HostIndexError(f'index files have no columns in common: {files}')
            columns = sorted(shared - set(cls.INDEX_FILE_KEY)
                             - set(cls.SUPPLEMENT_FILE_KEY))
            validate = _index_rows.column_validator(columns) if columns else None
        elif isinstance(validate, str):
            validate = _index_rows.column_validator([validate])
        elif isinstance(validate, (list, tuple)):
            validate = _index_rows.column_validator(validate) if validate else None

        pair = (_index_rows.pair_by_file if cls.MERGE_SUPPLEMENT_BY_FILE
                else _index_rows.pair_by_position)
        rows = pair(index_rows, _index_rows.stream_rows(supp_table, supplement_path),
                    index_key=cls.INDEX_FILE_KEY, supplement_key=cls.SUPPLEMENT_FILE_KEY,
                    validate=validate, files=files)
        return _index_rows.filtered(rows, filter)

    @classmethod
    def _supplement_path(cls, index_path):
        """The supplemental index that belongs beside an index.

        A host whose supplements are named in another way overrides this class method.

        Parameters:
            index_path (FCPath): The path to the index label.

        Returns:
            FCPath | None: The file in the directory of `index_path` whose name matches
            except that "_index" is replaced by "_supplemental_index", and "_INDEX" by
            "_SUPPLEMENTAL_INDEX"; None if the name of `index_path` contains neither
            "_index" nor "_INDEX".
        """

        name = index_path.name
        supp_name = (name.replace('_index', '_supplemental_index')
                         .replace('_INDEX', '_SUPPLEMENTAL_INDEX'))
        if supp_name == name:
            return None
        return index_path.with_name(supp_name)

    @staticmethod
    def _peek(rows):
        """The first of some rows, and the rows with that one still first.

        Parameters:
            rows (collections.abc.Iterable[dict[str, Any]]): The rows.

        Returns:
            tuple[dict[str, Any] | None, collections.abc.Iterable[dict[str, Any]]]: The
            first row, or None if there are none; and the rows: `rows` itself if it is a
            sequence, otherwise an iterator that yields the first row and then the rest.
        """

        if isinstance(rows, Sequence):
            return (rows[0] if rows else None, rows)
        rows = iter(rows)
        first = next(rows, None)
        if first is None:
            return (None, rows)
        return (first, itertools.chain([first], rows))

    @staticmethod
    def _set_path_attributes(obs, image_path, label_path):
        """Set the subfields of an Observation that name its files.

        A host calls this method for each observation it builds, after it has read its
        files and, for a file with an image array, after it has inserted the ``data``
        subfield.

        For an observation from a file, `image_path` is the path of the image file, as
        an FCPath or a :class:`pathlib.Path`, whichever of the two files the caller
        passed, and `label_path` is the path of the detached PDS3 label if the host read
        it, and None otherwise. These subfields are set:

        - ``filepath``: `image_path` as an FCPath.
        - ``basename``: the name of the image file.
        - ``image_url``: the absolute path or URL of the image file.
        - ``image_abspath``: the path of the image file on the local disk, as a resolved
          :class:`pathlib.Path`, if the host read the image file: that is, if
          `label_path` is None, because the host then read the image file itself; if
          `obs` has a ``header_dict`` subfield, because that is present exactly when the
          host read the label of the image file itself, and reading the header alone
          counts as reading the file; or if `obs` has a ``data`` subfield, because the
          host then read the array. An observation built from a detached label alone,
          for astrometry, has no ``image_abspath``.
        - ``label_url`` and ``label_abspath``: the absolute path or URL of the label, and
          its path on the local disk, if `label_path` is given.

        For an observation from a row of an index, `image_path` is the relative path of
        the file as the index gives it, such as "VOLUME/FILE_SPECIFICATION_NAME", given
        as a str, and `label_path` is None. Only ``filepath``, that str, and
        ``basename``, its last component, are set. A caller that wants the file joins
        the path with its own root directory.

        Parameters:
            obs (Observation): The observation.
            image_path (str | pathlib.Path | FCPath): The path of the image file; a str
                for an index row.
            label_path (str | pathlib.Path | FCPath | None): The path of the detached
                label that the host read; None if it read none.
        """

        if isinstance(image_path, str):
            obs.insert_subfield('filepath', image_path)
            obs.insert_subfield('basename', image_path.rpartition('/')[-1])
            return

        image_path = FCPath(image_path)
        obs.insert_subfield('filepath', image_path)
        obs.insert_subfield('basename', image_path.name)
        obs.insert_subfield('image_url', image_path.absolute().as_posix())
        if (label_path is None or 'data' in obs.subfields
                or 'header_dict' in obs.subfields):
            obs.insert_subfield('image_abspath', image_path.get_local_path().resolve())
        if label_path is not None:
            label_path = FCPath(label_path)
            obs.insert_subfield('label_url', label_path.absolute().as_posix())
            obs.insert_subfield('label_abspath', label_path.get_local_path().resolve())

    @staticmethod
    def _apply_overrides(params, *, timeshift=None, navigation=None, tracker=None,
                         parallel=None, **kwargs):
        """Apply overrides to the Observation constructor's input parameters.

        The subclass's constructor should call this method to handle the standard override
        parameters. Other input parameters are ignored. The subfields that name the
        observation's files are set by :meth:`_set_path_attributes` instead.

        Parameters:
            params (dict): The dictionary of inputs to the Observation constructor before
                any overrides have been applied.
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
        options.setdefault('method', 'strict')
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
