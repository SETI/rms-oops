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

from oops._exceptions import OopsIndexError, OopsSpiceError, OopsTypeError, OopsValueError
from oops.body        import Body
from oops.cadence     import TimeShift
from oops.frame       import (Cmatrix, Frame, FrameShift, Navigation, SpiceErrorFrame,
                              TrackerFrame)
from oops.observation import Observation
from oops.path        import PathShift

from .        import _index_rows
from ._errors import (AmbiguousHostError, HostFormatError, HostIndexError, PdsHostError,
                      UnknownHostError, VicarHostError, WrongHostError)

# The methods of the PDS3 label parser, in the order in which Host.from_file() tries them
# to identify the host of a label: "fast" reads most labels quickly, "strict" reads every
# valid label, and "loose" also reads some invalid ones
_PDS3_METHODS = ('fast', 'strict', 'loose')

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

    In a host's ``from_index``, every row of the index gives an observation, whether or
    not SPICE can supply its pointing. Where SPICE cannot, the observation gets a
    :class:`~oops.frame.SpiceErrorFrame` in place of its pointing, and that frame raises
    :class:`~oops.OopsSpiceError` whenever it is evaluated. The support method
    ``_check_spice_frame`` serves both cases: with ``defer=False`` it raises the error,
    for ``from_file``; with ``defer=True`` it returns the stand-in frame, for
    ``from_index``. It checks the frame as the observation will evaluate it once the
    overrides are applied, at the time and relative to the frame that they give.

    A host does not check SPICE, and does not raise :class:`~oops.OopsSpiceError`, when
    the caller supplies the pointing: when ``from_file`` is given a `frame` override,
    whether a :class:`~oops.frame.Cmatrix` or any other :class:`~oops.Frame`, or when the
    caller chooses a pointing source that is not SPICE, through an option of the host.

    A host's ``from_index`` returns an iterator that yields one entry per filtered row of
    the index, in index order. The call itself checks the arguments and reads the labels
    of the index and its supplement, so that a bad call fails at once. The rows are read
    as the iteration proceeds, a few at a time, the supplement along with the index, and
    each row's observation is built when the iteration reaches that row. An error in a
    row is therefore raised when the iteration reaches it. The support method
    ``_index_rows`` does the reading, pairing, and filtering, so ``from_index`` is an
    ordinary function that calls it and returns a generator over its rows.

    How a host's index and supplement are paired is fixed by the host, through the class
    attribute :attr:`~oops.host.Host.MERGE_SUPPLEMENT_BY_FILE`, from a survey of that
    instrument's indexes. Nothing at run time chooses the mode.
    """

    _HOSTS = []     # A list of all registered Host subclasses.
    _LOOKUP = {}    # name -> subclass

    #: The name of the host, such as "Cassini ISS", for messages. Each subclass defines
    #: it.
    NAME = 'none'

    #: False if every index of the host lists the same files in the same order as its
    #: supplement, so that their rows are paired by position; True if a supplement may
    #: list them in another order, so that each index row is paired with the supplement
    #: row for the same file. Each host sets it from a survey of its instrument's indexes.
    MERGE_SUPPLEMENT_BY_FILE = False

    SUPPLEMENT = object()  # Sentinel so from_index() searches for a supplemental index

    @staticmethod
    def from_file(filepath, *, pds3_options=None, select=None, astrometry=False,
                  target=None, lightsource=None, path=None, frame=None, cadence=None,
                  fov=None, calibrations=None, timeshift=None, navigation=None,
                  tracker=None, parallel=None, **kwargs):
        """One or more Observation objects derived from a given data file.

        The host is identified when this method is called, from the file's PDS3 label,
        VICAR header, FITS header, or name. The host's own ``from_file`` then reads the
        file from `filepath`, with the caller's `pds3_options` and overrides, so the
        result is the one the host class gives when it is called directly. To identify
        the host, a PDS3 label is parsed as it is in the file, with the method that
        `pds3_options` selects, "fast" by default, and then with each later method in the
        order "fast", "strict", and "loose". A method that reports a syntax error, or a
        value it cannot convert, such as a time with an hour of 25, is followed by the
        next. If none of them can parse the label, it is offered to each registered
        host's correction of label text (its ``_correct_label_text``), and the host whose
        correction makes it readable and that recognizes it is chosen. If there is no
        such host, each registered host is asked whether it recognizes the file by its
        path, as for a file that is not in a standard format. A VICAR file is identified
        from its header alone, without reading its array. Content that has already been
        parsed, a :class:`~pdsparser.Pds3Label`, a :class:`~vicar.VicarLabel` or
        :class:`~vicar.VicarImage`, or an :class:`~astropy.io.fits.HDUList`, is
        identified from that content and handed to the host as it is. Every registered
        host is asked at each step, and a file that more than one host claims at the same
        step is refused. Besides the exceptions listed here, the call raises any that the
        host's ``from_file`` documents.

        Parameters:
            filepath (str | pathlib.Path | FCPath | Pds3Label | VicarLabel | VicarImage |
                HDUList): The path to the data file or its detached label in PDS3 or PDS4
                format, or its parsed content.
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
                calibration or list of calibrations. If it is not given, the host's own
                calibration applies.
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
                Any other instrument-specific parameters, passed to the host's
                ``from_file``. A keyword that the host's ``from_file`` does not take
                raises TypeError.

        Returns:
            Observation | ObsArray | None: An Observation or a tuple or array of
            Observations retrieved from the file. None if a selected Observation is
            absent.

        Raises:
            AmbiguousHostError: If more than one registered Host subclass claims
                `filepath`. The message names each of them.
            HostFormatError: If `filepath` has invalid syntax or uses an unsupported
                option. Note that subclass :class:`~PdsHostError` is raised for a PDS3
                syntax error and :class:`~VicarHostError` is raised for a VICAR syntax
                error.
            OSError: If `filepath` does not exist or cannot be read.
            UnknownHostError: If `filepath` is not associated with any registered Host
                subclass.
            OopsTypeError: If `filepath` is neither a path nor one of the parsed forms
                listed above.
            OopsValueError: If a value in `pds3_options` is invalid.
            OopsSpiceError: If the observation's pointing comes from SPICE and the SPICE
                kernels cannot supply it at the time of the observation. Never raised when
                `frame` is given or when the pointing does not come from SPICE.
        """

        # A FITS file that is opened to identify it is closed again; the host reads it
        if isinstance(filepath, (str, pathlib.Path, FCPath)):
            (host, parsed) = Host._identify_and_parse(
                filepath, hosts=Host._HOSTS, formats='PVFO', astrometry=True,
                pds3_options=Host._pds3_options_with_defaults(pds3_options),
                pds3_fallback=True, dispatch=True)
            if isinstance(parsed, pyfits.HDUList):
                parsed.close()
        else:
            host = Host._identify_parsed(filepath)
        return host.from_file(filepath, pds3_options=pds3_options, select=select,
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

        The host is identified when this method is called. The index label is parsed
        with the method that `pds3_options` selects, "fast" by default, and every
        registered host is asked whether it recognizes the label. If none does, the first
        row of the index is read, and only that row, and offered to each host that could
        not tell from the label. If the parser cannot read the label, the label is
        offered to each host's correction of label text (its ``_correct_label_text``),
        and the host whose correction makes it readable and that recognizes the corrected
        label gets the call. If no host's correction helps,
        :class:`~oops.host.PdsHostError` is raised. An index that more than one host
        claims, from its label, from its first row, or through their corrections, is
        refused.

        The host's own ``from_index`` then reads the rows of the index and its supplement
        as the iteration proceeds, a few rows at a time, and builds each row's
        observation when the iteration reaches it. Every argument, the overrides
        included, is passed to the host's ``from_index`` as given, so the result is the
        one the host class gives when it is called directly. The label that was parsed
        here is passed on to a host that has no correction of label text, so that the
        label is parsed only once. Besides the exceptions listed here, the call raises
        any that the host's ``from_index`` documents.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to the PDS label of a
                data file index.
            supplement (str | pathlib.Path | FCPath | object, optional): The path to the
                PDS label of a supplemental index file, if any, whose columns are merged
                into each row. Each index row is merged with the supplement row for the
                same file, paired by position or by file as the host's
                :attr:`~oops.host.Host.MERGE_SUPPLEMENT_BY_FILE` says. Use the sentinel
                value `Host.SUPPLEMENT` to read the file in the directory of `filepath`
                whose name matches except that "_index" is replaced by
                "_supplemental_index" in upper or lower case. The host determines whether
                the supplement must exist and how a pair of rows is checked.
            label (Pds3Label, optional): The PDS label of `filepath` if it has already
                been read.
            row_dicts (collections.abc.Iterable[dict[str, Any]], optional): The rows of
                the index, one dictionary per row, if `filepath` and `supplement` have
                already been read. If provided, they are passed to the host, which
                iterates over them in place of reading the tables. If the host must be
                identified from a row, the first one is read for that and still passed
                on.
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
                instrument. When it is given, the host does not check that SPICE can
                supply the pointing.
            cadence (Cadence, optional): Override for the Cadence of the observation.
            fov (FOV, optional): Override for the default FOV of the observing instrument.
            calibrations (Calibration | list[Calibration], optional): Override for the
                calibration or list of calibrations. If it is not given, the host's own
                calibration applies.
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
                A keyword that the host's ``from_index`` does not take raises TypeError.

        Returns:
            collections.abc.Iterator[Observation | ObsArray | None]: The host's
            iterator, which yields the Observation, or the tuple or array of Observations,
            for each filtered row of the index file, in index order. It yields None where
            a selected Observation is absent from that row. Wrap it in ``list()`` to build
            every row at once.

        Raises:
            OopsValueError: If a value in `pds3_options` is invalid, if the parser cannot
                convert a value in the index label, such as day 366 of a year that has
                365 days, or if `filter` uses a column that the index does not have. With
                `row_dicts` given, `filter` is checked against the first row: at call time
                if `row_dicts` is a sequence, and when the iteration reads the first row
                if it is an iterator.
            OopsTypeError: When the iteration reaches a row whose value cannot be compared
                with a range in `filter`.
            HostIndexError: If the label of `filepath` or `supplement` does not describe a
                table that can be read, as when it lacks the object that its table pointer
                refers to. Also raised during the iteration, at the row where the two
                tables cannot be paired, where a table file ends before its label says,
                or at the first of the 100 rows read with a row longer than the label's
                row length, as listed for the support method ``_index_rows``. A value
                that cannot be read as its column's number is not an error of the table;
                see ``_index_rows``.
            OSError: If the label of `filepath` or `supplement` does not exist or cannot
                be read; when the iteration starts, if a table file does not exist or
                cannot be read.
            PdsHostError: If `filepath` uses invalid PDS label syntax and no host's
                correction of label text makes it readable. The message gives the path of
                the label.
            UnknownHostError: If `filepath` is not associated with any registered Host
                subclass.
            AmbiguousHostError: If more than one registered Host subclass claims the
                index, from its label, from its first row, or through its correction of
                label text. The message names each of them.
        """

        filepath = FCPath(filepath)
        options = Host._pds3_options_with_defaults(pds3_options)
        given_label = label

        # If the label cannot be parsed, it is offered to each host's correction of label
        # text. The first host whose correction makes it readable and that recognizes it
        # gets the call, and parses the label again with its own correction. If there is
        # no such host, the parser's error is raised.
        host = None
        if label is None:
            try:
                label = Host._parse_index_label(filepath, options)
            except PdsHostError:
                found = Host._host_from_correction(
                    Host._HOSTS, '_detect_in_index',
                    lambda correct: Host._parse_index_label(filepath, options,
                                                            correct=correct),
                    f'index {filepath}')
                if found is None:
                    raise
                (host, label) = (found[0], None)

        # Determine which host is represented by the index. Every host is asked about the
        # label first, so that a host that recognizes it is found without reading a row.
        # A host whose detector returns None can only be identified from the content of
        # a row; if no host claims the label, the first row, and that row alone, is read
        # once and offered to each of those hosts.
        if host is None:
            status = {host: host._detect_in_index(label) for host in Host._HOSTS}
            host = Host._one_host([host for host in Host._HOSTS if status[host]],
                                  f'index {filepath}')

        if host is None:
            undecided = [host for host in Host._HOSTS if status[host] is None]
            if not undecided:
                raise UnknownHostError(f'unknown host in index {filepath}')
            if row_dicts is None:
                table = _index_rows.open_table(filepath, label=label,
                                               pds3_method=options['method'])
                first = _index_rows.first_row(table, filepath)
            else:
                (first, row_dicts) = Host._peek(row_dicts)
            host = Host._one_host([host for host in undecided
                                   if first is not None and host._detect_in_row(first)],
                                  f'first row of the index {filepath}')
            if host is None:
                raise UnknownHostError(f'unknown host in index {filepath}')

        # A host with a correction of label text parses the label itself, so that it
        # reads the label as its own from_index would
        if host._label_corrector() is not None:
            label = given_label

        # The host reads, pairs and filters the rows itself, through _index_rows(),
        # because only the host knows how its supplement is matched to its index
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

    @classmethod
    def _correct_label_text(cls, text, filepath):
        """The text of a PDS3 label of this host, corrected so that it can be parsed.

        A host whose labels or index labels hold text that the PDS3 parser cannot read,
        or reads wrongly, overrides this method to repair that text, so that the oops
        readers stay free of any one data set's repairs. Every oops method that parses a
        label for this host calls it with the label's text before any parser method
        runs: :meth:`~oops.host.Host._read_fileinfo` for a detached or attached product
        label, and :meth:`~oops.host.Host._read_index_label` and
        :meth:`~oops.host.Host._index_rows` for the labels of an index and its
        supplement. :meth:`~oops.host.Host.from_index` passes a host that overrides this
        method no label of its own, so the host parses the label with its correction.
        The corrected text is parsed in place of the file's, and the parsed label keeps
        the path of the file, so that its pointers are found beside it. An index table
        is still read from its own file.

        The method also receives labels of other hosts. When
        :meth:`~oops.host.Host.from_file` or :meth:`~oops.host.Host.from_index` meets a
        label that no parser method can read, it offers the label to each registered
        host's correction in turn, and gives the file to the host whose correction makes
        the label readable and that then recognizes it. A correction must
        therefore change only the text that has the defect it repairs, and return any
        other text as it is. It should not raise. In that offer, a correction that raises
        an exception counts as no help, and the next host is tried. When the host reads
        its own file, the exception propagates to the caller.

        The default returns `text` unchanged, and a host that keeps it has its labels
        parsed from their files.

        Parameters:
            text (str): The text of the label, up to its END statement for an attached
                label, with <LF> line terminators.
            filepath (FCPath): The path to the file that holds the label, so that a
                host can decide for each file.

        Returns:
            str: The corrected text; `text` itself if nothing needs to change.
        """

        return text

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
    def _host_from_correction(hosts, detector, parse, what):
        """The host whose correction of label text makes a label readable and that then
        recognizes the corrected label.

        Each host's :meth:`~oops.host.Host._correct_label_text` is tried; a host that
        keeps the default is skipped. If a correction raises an exception, or the text it
        returns cannot be parsed, that host is passed over, so that one host's correction
        cannot keep another host from being found.

        Parameters:
            hosts (list[type]): The Host subclasses to consider, in order.
            detector (str): The name of the method that recognizes the label, either
                "_detect_in_pds3" or "_detect_in_index".
            parse (collections.abc.Callable): A function with the signature
                ``(collections.abc.Callable) -> Pds3Label`` that parses the label with the
                correction it is given.
            what (str): The label, for the message of the error, such as "index
                <filepath>".

        Returns:
            tuple[type, Pds3Label] | None: The Host subclass and the label as its
            correction gives it; None if no host's correction helps.

        Raises:
            AmbiguousHostError: If the corrections of more than one host make the label
                readable and those hosts recognize it.
        """

        found = []
        for host in hosts:
            correct = host._label_corrector()
            if correct is None:
                continue
            try:
                label = parse(correct)
            except Exception:   # nosec B112 -- any failure means this host did not help
                continue
            if getattr(host, detector)(label):
                found.append((host, label))

        if Host._one_host([host for (host, _) in found], what) is None:
            return None
        return found[0]

    @staticmethod
    def _identify_parsed(content):
        """Identify the host module associated with the parsed content of a file.

        Parameters:
            content (Pds3Label | VicarLabel | VicarImage | HDUList): The parsed content.

        Returns:
            type: The Host subclass whose detector recognizes `content`.

        Raises:
            OopsTypeError: If `content` is not one of the parsed forms listed above.
            UnknownHostError: If no registered Host subclass recognizes `content`.
            AmbiguousHostError: If more than one registered Host subclass recognizes
                `content`.
        """

        if isinstance(content, Pds3Label):
            (kind, detector) = ('PDS3 label', '_detect_in_pds3')
        elif isinstance(content, (VicarLabel, VicarImage)):
            (kind, detector) = ('VICAR file', '_detect_in_vicar')
        elif isinstance(content, pyfits.HDUList):
            (kind, detector) = ('FITS file', '_detect_in_fits')
        else:
            raise OopsTypeError(f'Host.from_file needs a path (str, pathlib.Path, or '
                                f'FCPath) or the parsed content of a file (Pds3Label, '
                                f'VicarLabel, VicarImage, or HDUList), not '
                                f'{type(content).__name__}')

        host = Host._one_host([host for host in Host._HOSTS
                               if getattr(host, detector)(content)], f'parsed {kind}')
        if host is None:
            raise UnknownHostError(f'unrecognized host in parsed {kind}')
        return host

    @classmethod
    def _read_fileinfo(cls, fileinfo, *, formats, astrometry=False, pds3_options=None,
                       pds3_fallback=False):
        """Read and parse the file content for this class in PDS3, VICAR, or FITS format.

        This is a support method for the `from_file` methods of the Host subclasses. With
        `pds3_fallback` True, a PDS3 label that the requested method of the parser cannot
        read is parsed again with each later method of "fast", "strict", and "loose", as
        described for :meth:`~oops.host.Host._parse_pds3`, so one call takes the place of
        a host's own loop over the methods. If `pds3_options` gives no method, every
        method is tried, starting with "fast". The label text is first corrected by this
        host's :meth:`~oops.host.Host._correct_label_text`.

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
            pds3_fallback (bool, optional): True to try each later method of the PDS3
                parser when the requested one cannot read a label; False to use the
                requested method alone.

        Returns:
            Pds3Label | VicarLabel | VicarImage | HDUList | FCPath: The parsed file
            content if it can be obtained from `fileinfo`, either because it is already a
            parsed format or if it refers to a file in a parsed format. If it is file that
            is not in a standard format, the FCPath is returned.

        Raises:
            HostFormatError: If `fileinfo` has invalid syntax for format. Note that
                subclass :class:`~PdsHostError` is returned for PDS-formatted files and
                :class:`~VicarHostError` is returned for VICAR files. The message of a
                :class:`~PdsHostError` gives the path of the label and, with
                `pds3_fallback`, the error of each method tried.
            OopsValueError: If the value in `pds3_options` is invalid, as described for
                :meth:`~oops.host.Host._parse_pds3`.
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
                                             pds3_fallback=pds3_fallback)
        return parsed

    @staticmethod
    def _identify_and_parse(filepath, hosts, *, formats='PVFO', astrometry=False,
                            pds3_options=None, pds3_fallback=False, dispatch=False):
        """Interpret the filepath and identify the associated host module.

        Also return the parsed file content for PDS3, VICAR, and FITS formats. A FITS file
        that none of the hosts recognizes is closed before the error is raised; one that a
        host recognizes is returned open, and the caller closes it.

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
            pds3_fallback (bool, optional): True to parse a PDS3 label with each method
                after the one that `pds3_options` selects, in the order "fast", "strict",
                "loose", when that one fails, and, if `formats` includes 'O' and no method
                can parse it, to ask each host whether it recognizes the file by its path;
                False to parse it with the selected method alone.
            dispatch (bool, optional): True when `hosts` are the registered hosts among
                which the host of the file is to be found: the label is parsed as it is,
                and a label that cannot be parsed is offered to each host's
                :meth:`~oops.host.Host._correct_label_text` in turn. False when `hosts`
                holds the one host that reads the file: its correction is applied to the
                label text before the label is parsed.

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
            AmbiguousHostError: If more than one of the listed Host subclasses claims
                `filepath`. A FITS file is closed before it is raised.
        """

        def identify_pds3(kind):
            """The host and parsed label of a PDS3 file, or the host and the path of a
            label that no method can parse but a host recognizes by its path."""

            # A host that reads its own file corrects the label text before parsing it.
            # Identification among the registered hosts parses the label as it is, and
            # offers a label that cannot be parsed to each host's correction in turn.
            correct = None if dispatch else hosts[0]._label_corrector()
            try:
                label = Host._parse_pds3(filepath, pds3_options, fallback=pds3_fallback,
                                         correct=correct)
            except PdsHostError:
                if dispatch:
                    found = Host._host_from_correction(
                        hosts, '_detect_in_pds3',
                        lambda correct: Host._parse_pds3(filepath, pds3_options,
                                                         fallback=pds3_fallback,
                                                         correct=correct),
                        f'{kind} {filepath}')
                    if found:
                        return found
                if pds3_fallback and 'O' in formats:
                    host = Host._one_host([host for host in hosts
                                           if host._detect_in_file(filepath)],
                                          f'{kind} {filepath}')
                    if host is not None:
                        return host, filepath
                raise
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
            claimants = [host for host in hosts if host._detect_in_fits(hdulist)]
            if len(claimants) != 1:
                hdulist.close()
            host = Host._one_host(claimants, f'FITS file {filepath}')
            if host is None:
                raise Host._no_host_error(hosts, 'FITS file', filepath)
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
    def _parse_pds3(filepath, pds3_options, *, fallback, correct=None):
        """The parsed PDS3 label of a file.

        With `fallback` True and a method of "fast", "strict", or "loose" in
        `pds3_options`, that method is tried first and each later one in turn; with no
        method in `pds3_options`, every one is tried, starting with "fast". A method
        fails if it raises a syntax error, or a ValueError for a value it cannot convert,
        as the fast method does for a time with an hour of 25 or for day 366 of a year
        that has 365 days; the next method is then tried. A TypeError, which is how the
        parser rejects a keyword it does not accept, ends the attempts at once. With any
        other method, or with `fallback` False, the label is parsed once, and a
        ValueError means an invalid option.

        With `correct`, the label's text is read and corrected once, before any method
        runs, and every method parses the corrected text, as described for
        :meth:`~oops.host.Host._correct_label_text`. If the text cannot be read, as for
        an attached label that lacks its END statement, PdsHostError is raised.

        Parameters:
            filepath (FCPath): The path to the label or to a file with an attached label.
            pds3_options (dict[str, Any]): The keyword arguments of the
                :class:`~pdsparser.Pds3Label` parser.
            fallback (bool): True to try each method after the one that `pds3_options`
                names, in the order "fast", "strict", "loose", when that one fails; False
                to parse the label with `pds3_options` alone.
            correct (collections.abc.Callable, optional): A function with the signature
                ``(str, FCPath) -> str`` that corrects the text of the label, such as a
                host's :meth:`~oops.host.Host._correct_label_text`; None to parse the file
                as it is.

        Returns:
            Pds3Label: The parsed label.

        Raises:
            OopsValueError: If `pds3_options` holds a keyword that the parser does not
                accept; or, when the label is parsed only once, if a value in
                `pds3_options` is invalid or the parser raises a ValueError for a value in
                the label.
            OSError: If `filepath` does not exist or cannot be read.
            PdsHostError: If no method that is tried can parse the label, or if the text
                of the label cannot be read for `correct`. The message has the form
                "cannot parse the PDS3 label <filepath>: <error>" for one attempt, and
                gives each method's error, as in "method fast: <error>; method strict:
                <error>; method loose: <error>", for several. The exception is raised
                from the error of the first method tried.
        """

        method = pds3_options.get('method')
        if fallback and method is None:
            method = _PDS3_METHODS[0]
            pds3_options = pds3_options | {'method': method}
        fallback = fallback and method in _PDS3_METHODS
        (source, pds3_options) = Host._pds3_source(filepath, pds3_options, correct)
        attempts = [pds3_options]
        if fallback:
            attempts = [pds3_options | {'method': later}
                        for later in _PDS3_METHODS[_PDS3_METHODS.index(method):]]

        errors = []         # (method, exception) of each method that failed
        for options in attempts:
            try:
                return Host._pds3_label(source, filepath, options)
            except PdsError as err:
                errors.append((options.get('method'), err))
            except TypeError as err:
                raise OopsValueError(str(err)) from err
            except ValueError as err:
                if not fallback:
                    raise OopsValueError(str(err)) from err
                errors.append((options['method'], err))

        if len(errors) == 1:
            reasons = str(errors[0][1])
        else:
            reasons = '; '.join(f'method {name}: {err}' for (name, err) in errors)
        raise PdsHostError(f'cannot parse the PDS3 label {filepath}: '
                           f'{reasons}') from errors[0][1]

    @classmethod
    def _label_corrector(cls):
        """This host's correction of label text, if it has one.

        Returns:
            collections.abc.Callable | None: The host's
            :meth:`~oops.host.Host._correct_label_text`; None if the host keeps the
            default, whose labels are then parsed from their files directly.
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
            read as the parser reads it from the file and is parsed in place of the
            file even when the correction leaves it unchanged, so the file is read once.

        Raises:
            OSError: If the file does not exist or cannot be read.
            PdsHostError: If the text of the label cannot be read, as for an attached
                label that lacks its END statement, or if `correct` raises a PDS3 parser
                error.
        """

        if correct is None:
            return (filepath, pds3_options)

        try:
            if pds3_options.get('method') == 'compound':
                text = FCPath(filepath).read_text(encoding='latin-1')
            elif pds3_options.get('vax'):
                text = read_vax_binary_label(filepath)
            else:
                text = read_label(filepath)
            corrected = correct(text, filepath)
        except PdsError as err:
            raise PdsHostError(f'cannot parse the PDS3 label {filepath}: {err}') from err

        # The parser takes a string that has no line terminator for the path of a file
        if '\n' not in corrected:
            corrected += '\n'

        fmt_dirs = pds3_options.get('fmt_dirs', [])
        fmt_dirs = [fmt_dirs] if isinstance(fmt_dirs, (str, pathlib.Path, FCPath)) \
            else list(fmt_dirs)
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
            label.filepath = FCPath(filepath)
            label._filepath = label.filepath      # the parser's older name for it
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
            AmbiguousHostError: If `claimants` holds more than one Host subclass. The
                message reads "<what> is claimed by more than one host: <name>, <name>",
                with the :attr:`~oops.host.Host.NAME` of each.
        """

        if len(claimants) > 1:
            names = ', '.join(host.NAME for host in claimants)
            raise AmbiguousHostError(f'{what} is claimed by more than one host: {names}')
        return claimants[0] if claimants else None

    _TIMESHIFT_CLASS = {'T': TimeShift, 'P': PathShift, 'F': FrameShift}
    _TIMESHIFT_ATTR = {'T': 'cadence', 'P': 'path', 'F': 'frame'}

    @classmethod
    def _index_rows(cls, filepath, *, supplement=SUPPLEMENT, require_supplement=True,
                    label=None, row_dicts=None, pds3_options=None, filter=None,
                    validate=None, file_key=None, supplement_key=None,
                    prefer_supplement=False):
        """An iterator over the rows of an index, each merged with its supplement row and
        filtered, which reads the rows as the iteration proceeds.

        This is the support method with which a host's ``from_index`` reads its index;
        the host builds one observation from each row that the iterator yields.

        The call itself reads only labels. It reads the labels of the index and the
        supplement, checks that the index belongs to this host, and checks the columns
        that `filter`, `validate`, and the file keys use against the labels. If the index
        label cannot tell whether the index belongs to this host, the first row is read
        for that, and only that row.

        The iteration then reads the index 100 rows at a time. How each index row is
        paired with its supplement row is fixed by the host's
        :attr:`~oops.host.Host.MERGE_SUPPLEMENT_BY_FILE`:

        - False, pairing by position: the supplement is read in step with the index. At
          each row, if both tables have a file key, the file that the index row gives
          must be the file that the supplement row gives.
        - True, pairing by file: the supplement is read ahead only as far as the partner
          of the current index row, the row that gives the same file. A supplement row
          read before its partner is held until the partner is reached and then
          released, so a supplement whose rows are k places out of order holds at most k
          rows. An index row whose file the supplement does not list is found only when
          the whole supplement has been read.

        Paths are compared without regard to case, to empty or "." components (as in a
        leading "/" or "./" or a doubled slash), or to the extension of the file name,
        so that a row that gives a product's label and a row that gives its data file
        match. Each pair is then checked with `validate`, merged, and tested against
        `filter`. Rows are counted from 0 in the messages of errors.

        The tables are not checked beyond the errors listed below. A value that cannot be
        read as the number its column describes makes ``PdsTable`` issue a UserWarning and
        give the value as a string, with its "_mask" entry True, and the row is yielded
        with it; the host may then raise a ValueError or TypeError when it converts the
        value, or a pairing error if the value is part of a file key. A row shorter than
        the label's row length is read as it is.

        Parameters:
            filepath (str | pathlib.Path | FCPath): Path to the index label.
            supplement (str | pathlib.Path | FCPath | object | None, optional): The path
                to the PDS label of the supplemental index; None to read the index alone.
                Use the sentinel value `Host.SUPPLEMENT` to read the file that
                :meth:`~oops.host.Host._supplement_path` returns.
            require_supplement (bool, optional): True if the supplement file must exist;
                False to read the index alone if the supplement file is missing, or if
                `supplement` is `Host.SUPPLEMENT` and
                :meth:`~oops.host.Host._supplement_path` returns None.
            label (Pds3Label | dict[str, Any], optional): The content of the label of
                `filepath`, if it has already been read; a host may pass a dictionary
                that it has changed, such as one with wider columns.
            row_dicts (collections.abc.Iterable[dict[str, Any]], optional): The rows of
                the index already merged with the supplement. If they are given, the
                iteration filters each one as it reaches it, and the tables are not read.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser for the
                index label. The supplement's label is parsed with their method alone.
            filter (dict[str, Any], optional): Column names, each with a value, a set,
                list, or tuple of values, or a tuple of two values that bound a range. A
                row is yielded only if every column it lists equals the value, is one of
                the values, or lies within the range, bounds included. A row that lacks
                one of the columns is left out.
            validate (str | list[str] | collections.abc.Callable, optional): The check of
                each pair of rows. A column name or a list or tuple of them gives columns
                whose values the two rows must share. Two strings are the same if either
                one, in upper case, contains the other. An empty list or tuple, such as
                ``validate=()``, skips the comparison of columns, so the only check is the
                pairing's comparison of the files. A function with the signature
                ``(dict[str, Any], dict[str, Any]) -> bool`` receives the index row and
                the supplement row and returns True if they match; it either returns
                False or raises an exception if they do not. If unspecified, the columns
                that the two tables share are compared, except those of the file keys,
                which the pairing compares as paths.
            file_key (str | tuple[str, ...] | collections.abc.Callable, optional): How
                a row of the index gives its file: a column name; a tuple of column
                names, whose values are joined with "/", such as ``("VOLUME_ID",
                "PATH_NAME", "FILE_NAME")``; or a function with the signature
                ``(dict[str, Any]) -> str`` that returns the path from a row dictionary.
                If unspecified, the key is FILE_SPECIFICATION_NAME, provided that both
                tables have it. If they do not, rows paired by position are paired
                without comparing their files. A host whose tables can list one file in
                more than one volume, as a cumulative index can, and that pairs them by
                file, gives a key that includes the volume, such as ``("VOLUME_ID",
                "FILE_SPECIFICATION_NAME")``.
            supplement_key (str | tuple[str, ...] | collections.abc.Callable, optional):
                How a row of the supplement gives its file, in the same forms as
                `file_key`. If unspecified, `file_key` applies to both tables.
            prefer_supplement (bool, optional): True to keep the supplement's value of a
                column that both tables have; False to keep the index's.

        Returns:
            collections.abc.Iterator[dict[str, Any]]: One dictionary for each row of the
            index that passes `filter`, in index order, with the columns of its
            supplement row.

        Raises:
            HostIndexError: At call time: if either label does not describe a table that
                can be read; if `supplement` is `Host.SUPPLEMENT`, `require_supplement` is
                True, and :meth:`~oops.host.Host._supplement_path` returns None; if a
                table lacks a column of its file key or of `validate`; if `validate` is
                unspecified and the two tables do not share any column; or if
                :attr:`~oops.host.Host.MERGE_SUPPLEMENT_BY_FILE` is True, `file_key` is
                unspecified, and the tables lack FILE_SPECIFICATION_NAME. During the
                iteration, at the row concerned: an index row whose file is not the file
                of the supplement row in the same position (pairing by position); an index
                row for which the supplement lacks a row, which with pairing by file is
                raised once the whole supplement has been read; a supplement row that
                repeats the file of a row that is still held (pairing by file); a pair of
                rows that fails `validate`; the end of a table file that holds fewer rows
                than its label says, after every row it holds; a row longer than the
                label's row length, raised at the first of the 100 rows read with it, none
                of which is yielded. After the last index row: a supplement row that was
                not paired with an index row, or a table file that holds more rows than
                its label says. With pairing by file, a file that the index lists more
                times than the supplement is reported as an index row for which the
                supplement lacks a row, and a file that the supplement lists more times
                than the index as a supplement row that was not paired; only the rows
                read ahead are kept, so the memory used does not grow with the length of
                the index.
            OopsValueError: At call time, if a value in `pds3_options` is invalid, or if
                `filter` uses a column that neither table has. If `row_dicts` is given,
                `filter` is checked against the first row, at call time if `row_dicts`
                is a sequence and otherwise when the first row is read.
            OopsTypeError: When the iteration reaches a row whose value cannot be compared
                with a range in `filter`.
            OSError: At call time, if a label does not exist or cannot be read, or if the
                supplement's label is missing and `require_supplement` is True. When the
                iteration starts, if a table file does not exist or cannot be read.
            PdsHostError: At call time, if a label uses invalid PDS3 syntax.
            WrongHostError: At call time, if the index does not belong to this host.
        """

        filepath = FCPath(filepath)

        # Rows given by the caller are filtered as they are reached
        if row_dicts is not None:
            names = None
            if filter and isinstance(row_dicts, Sequence) and row_dicts:
                names = row_dicts[0]
                _index_rows.check_filter(filter, names)
            return _index_rows.filtered(row_dicts, filter, names)

        # Read the index label and check the host
        options = Host._pds3_options_with_defaults(pds3_options)
        method = options['method']
        label = cls._read_index_label(filepath, label=label, pds3_options=options)
        index = _index_rows.open_table(filepath, label=label, pds3_method=method)
        index_names = _index_rows.column_names(index)
        if cls is not Host:
            status = cls._detect_in_index(label)
            if status is None:
                row = _index_rows.first_row(index, filepath)
                status = row is not None and cls._detect_in_row(row)
            if not status:
                raise WrongHostError(f'not a {cls.NAME} index: {filepath}')

        # Read the supplement label
        supplement_path = None
        if supplement is Host.SUPPLEMENT:
            supplement_path = cls._supplement_path(filepath)
            if supplement_path is None and require_supplement:
                raise HostIndexError(f'supplemental index required for {filepath}')
        elif supplement:
            supplement_path = FCPath(supplement)

        # The supplement's label is parsed with the parser method alone, after the host's
        # correction of label text if it has one
        supp_table = None
        if supplement_path is not None:
            correct = cls._label_corrector()
            try:
                supp_label = (None if correct is None else
                              Host._parse_index_label(supplement_path,
                                                      {'method': method},
                                                      correct=correct))
                supp_table = _index_rows.open_table(supplement_path, label=supp_label,
                                                    pds3_method=method)
            except FileNotFoundError:
                if require_supplement:
                    raise

        if supp_table is None:
            _index_rows.check_filter(filter, index_names)
            return _index_rows.filtered(_index_rows.stream_rows(index, filepath), filter,
                                        index_names)

        # Check the columns of the file keys, the validation and the filter
        supp_names = _index_rows.column_names(supp_table)
        shared = set(index.get_keys()) & set(supp_table.get_keys())
        files = f'{filepath}, {supplement_path}'
        if file_key is None and 'FILE_SPECIFICATION_NAME' in shared:
            file_key = 'FILE_SPECIFICATION_NAME'
        if supplement_key is None:
            supplement_key = file_key
        if file_key is None:
            if cls.MERGE_SUPPLEMENT_BY_FILE:
                raise HostIndexError(f'the rows of a {cls.NAME} index and its supplement '
                                     f'are paired by file, but the tables lack '
                                     f'FILE_SPECIFICATION_NAME: {files}')
        else:
            for (kind, key, names, path) in (('index', file_key, index_names, filepath),
                                             ('supplement', supplement_key, supp_names,
                                              supplement_path)):
                for name in sorted(_index_rows.key_columns(key)):
                    if name not in names:
                        raise HostIndexError(f'the {kind} lacks the column {name}: '
                                             f'{path}')

        # By default, the columns of the file keys are left out of the check, because the
        # pairing compares them already, as paths, and the two tables may give the same
        # product by its label and by its data file
        if validate is None:
            if not shared:
                raise HostIndexError(f'the index and the supplement do not share any '
                                     f'column: {files}')
            columns = sorted(shared - _index_rows.key_columns(file_key, supplement_key))
            validate = _index_rows.column_validator(columns) if columns else None
        elif isinstance(validate, (list, tuple)) and not validate:
            validate = None         # the pairing's comparison of the files is the check
        elif isinstance(validate, (str, list, tuple)):
            columns = [validate] if isinstance(validate, str) else list(validate)
            for (kind, names, path) in (('index', index_names, filepath),
                                        ('supplement', supp_names, supplement_path)):
                for column in columns:
                    if column not in names:
                        raise HostIndexError(f'the {kind} lacks the column {column} to '
                                             f'compare: {path}')
            validate = _index_rows.column_validator(columns)

        names = index_names | supp_names
        _index_rows.check_filter(filter, names)

        pair = (_index_rows.pair_by_file if cls.MERGE_SUPPLEMENT_BY_FILE
                else _index_rows.pair_by_position)
        rows = pair(_index_rows.stream_rows(index, filepath),
                    _index_rows.stream_rows(supp_table, supplement_path),
                    file_key=file_key, supplement_key=supplement_key, validate=validate,
                    prefer_supplement=prefer_supplement, files=files)
        return _index_rows.filtered(rows, filter, names)

    @classmethod
    def _supplement_path(cls, filepath):
        """The supplemental index that belongs beside an index.

        A host whose supplements are named in another way overrides this method.

        Parameters:
            filepath (FCPath): The path to the index label.

        Returns:
            FCPath | None: The file in the directory of `filepath` whose name matches
            except that "_index" is replaced by "_supplemental_index", in lower or upper
            case; None if the name of `filepath` contains neither "_index" nor "_INDEX".
        """

        name = filepath.name
        supp_name = (name.replace('_index', '_supplemental_index')
                         .replace('_INDEX', '_SUPPLEMENTAL_INDEX'))
        if supp_name == name:
            return None
        return filepath.with_name(supp_name)

    @classmethod
    def _read_index_label(cls, filepath, *, label=None, pds3_options=None):
        """The label of an index that this host reads, parsed once.

        A host's ``from_index`` that needs the index label, for example to decide how to
        read the table, calls this method first and passes the label it returns to
        :meth:`~oops.host.Host._index_rows` as `label`, so that the label is parsed only
        once. The label is parsed with one method of the parser, after this host's
        correction of label text; no later method is tried if that one fails.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The path to the index label.
            label (Pds3Label | dict[str, Any], optional): The label, if it has already
                been read; it is returned as it is.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser; the
                method is "fast" unless they give another.

        Returns:
            Pds3Label | dict[str, Any]: `label` if it is given; otherwise, the parsed
            label of `filepath`.

        Raises:
            OopsValueError: If a value in `pds3_options` is invalid, or if the parser
                cannot convert a value in the label, such as day 366 of a year that has
                365 days.
            OSError: If the label does not exist or cannot be read.
            PdsHostError: If the label uses invalid PDS3 syntax. The message gives the
                path of the label.
        """

        filepath = FCPath(filepath)
        if label is not None:
            return label
        return Host._parse_index_label(filepath,
                                       Host._pds3_options_with_defaults(pds3_options),
                                       correct=cls._label_corrector())

    @staticmethod
    def _parse_index_label(filepath, pds3_options, *, correct=None):
        """The parsed label of an index or a supplement.

        Parameters:
            filepath (FCPath): The path to the label.
            pds3_options (dict[str, Any]): The keyword arguments of the
                :class:`~pdsparser.Pds3Label` parser.
            correct (collections.abc.Callable, optional): A function with the signature
                ``(str, FCPath) -> str`` that corrects the text of the label before it is
                parsed, such as a host's :meth:`~oops.host.Host._correct_label_text`;
                None to parse the file as it is.

        Returns:
            Pds3Label: The label, whose `filepath` is `filepath`.

        Raises:
            OopsValueError: If a value in `pds3_options` is invalid, or if the parser
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
    def _volumes_directory(index_path):
        """The directory that holds the volume directories whose files an index lists.

        Parameters:
            index_path (str | pathlib.Path | FCPath): The index file or its PDS3 label.

        Returns:
            FCPath: For an index in a Ring-Moon Systems Node metadata tree, such as
            ".../metadata/GO_0xxx/GO_0017/GO_0017_index.lbl", the matching volume series
            directory, ".../volumes/GO_0xxx"; for an index anywhere else, the index's own
            directory. The name "metadata" is matched in either case.
        """

        directory = FCPath(index_path).parent
        if directory.parent.parent.name.lower() == 'metadata':
            return directory.parent.parent.parent / 'volumes' / directory.parent.name
        return directory

    @staticmethod
    def _check_select(select, *, names=None, unit=None):
        """Raise the error that a `select` argument would raise, before anything is read.

        Parameters:
            select (Any): The `select` argument of a ``from_file`` or ``from_index``.
            names (dict[str, int], optional): The names that `select` may use, in upper
                case.
            unit (str, optional): For a host whose file or index row is always one
                observation, what that observation comes from, such as "row of a Cassini
                ISS index", for the message; `select` must then be None.

        Raises:
            OopsTypeError: If `select`, or an item of a tuple, is not an int, a slice, or
                a str.
            OopsValueError: If `unit` is given and `select` is not None, or if a string in
                `select` is not one of `names`.
        """

        if select is None:
            return
        if unit is not None:
            raise OopsValueError(f'each {unit} is one observation; select must be None, '
                                 f'not {select!r}')
        for item in (select if isinstance(select, tuple) else (select,)):
            if isinstance(item, str):
                if item.upper() not in (names or {}):
                    raise OopsValueError(f'unrecognized select name: {item!r}')
            elif (isinstance(item, bool)
                  or not isinstance(item, (numbers.Integral, slice))):
                raise OopsTypeError(f'invalid select: {select!r}')

    @staticmethod
    def _apply_overrides(params, filepath, *, abspath=None, timeshift=None,
                         navigation=None, tracker=None, parallel=None, **kwargs):
        """Apply overrides to the Observation constructor's input parameters.

        The subclass's constructor should call this method to handle the standard override
        parameters. Other input parameters are ignored.

        It also puts the file's path subfields in `params`. For a pathlib.Path or FCPath,
        `filepath` is the path as given, `basename` is its file name, and `image_url` is
        its absolute form. For a string, which is taken to be a path as an index row gives
        it, `filepath` and `basename` alone are set. For None, those subfields are left
        out. The file system is not consulted, so nothing is created or retrieved.

        `abspath` is set only from the `abspath` argument. A remote file that exists on
        its server has no local path until it has been retrieved, and a local cache can
        be cleared at any time, so only a host that has retrieved the image file in this
        call, to read its array, knows a local path that exists. Such a host passes the
        path that :meth:`filecache.FCPath.retrieve` returned, or the path of the local
        file that it read. A read that does not retrieve the image file, such as one for
        astrometry only or the build of an observation from an index row, passes none,
        and the observation then has no `abspath`.

        Parameters:
            params (dict): The dictionary of inputs to the Observation constructor before
                any overrides have been applied.
            filepath (str | pathlib.Path | FCPath | None): Path to the file; None if no
                path is known, as for content that was parsed from text.
            abspath (str | pathlib.Path, optional): The local path of the image file, if
                it was retrieved in this call; it is stored resolved as `abspath`.
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

        # Fill in the file/URL info. A string is a path as an index row gives it, relative
        # to a volume that cannot be found from here.
        if filepath is not None:
            params['filepath'] = filepath
            if isinstance(filepath, str):
                params['basename'] = filepath.rpartition('/')[-1]
            else:
                fcpath = FCPath(filepath)
                params['basename'] = fcpath.name
                params['image_url'] = fcpath.absolute().as_posix()
        if abspath is not None:
            params['abspath'] = pathlib.Path(abspath).resolve()

        # Override certain standard attributes
        for key in ('target', 'lightsource', 'path', 'frame', 'cadence', 'fov',
                    'calibrations'):
            if kwargs.get(key) is not None:
                params[key] = kwargs[key]

        # Apply an optional time shift
        if timeshift is not None:
            dt, letters = Host._timeshift_letters(timeshift)
            arg = None
            for letter in letters:
                attr = Host._TIMESHIFT_ATTR[letter]
                if arg is None:     # the first time-shifted item is independent
                    arg = Host._TIMESHIFT_CLASS[letter](dt, params[attr])
                    params[attr] = arg
                else:               # subsequent time shifts are linked to the first
                    params[attr] = Host._TIMESHIFT_CLASS[letter](arg, params[attr])

        # Apply frame adjustments. The parallel and tracker overrides evaluate the frame
        # once, here. A frame that the SPICE kernels cannot supply, such as the
        # SpiceErrorFrame of an index row, is left as it is, so that it raises
        # OopsSpiceError when the observation uses it, as it would without the override.
        Host._check_frame_overrides(tracker, parallel)
        if parallel is not None:
            time = params['cadence'].midtime
            frame = Frame.as_frame(params['frame'])
            try:
                xform = frame.wrt(parallel.frame).transform_at_time(time)
            except OopsSpiceError:
                if Host._evaluates(frame, time):
                    raise                   # the parallel observation's frame failed
            else:
                params['frame'] = Cmatrix(xform.matrix, reference=parallel.frame)

        elif tracker is not None:
            epoch = Host._tracker_epoch(params['cadence'], tracker)

            # Convert body to path or None
            target = params['target']
            if isinstance(target, str) and target.upper() == 'NONE':
                target_path = None
            else:
                target_path = Body.as_body(target).path

            # Only the frame is evaluated when a TrackerFrame is built
            try:
                params['frame'] = TrackerFrame(params['frame'],
                                               target=target_path,
                                               observer=params['path'],
                                               epoch=epoch)
            except OopsSpiceError:
                pass

        if navigation is not None:
            params['frame'] = Navigation(navigation, params['frame'])

        return params

    @staticmethod
    def _tracker_epoch(cadence, tracker):
        """The time at which a `tracker` override holds the target's position fixed.

        Parameters:
            cadence (Cadence): The cadence of the observation, after any `cadence` and
                `timeshift` overrides.
            tracker (str): "start", "midtime", or "end".

        Returns:
            float | Scalar: The start, midtime, or end of `cadence`.

        Raises:
            OopsValueError: If `tracker` is not one of the three values.
        """

        try:
            tfrac = {'start': 0., 'midtime': 0.5, 'end': 1.}[tracker]
        except (KeyError, TypeError):
            raise OopsValueError('invalid tracker, should be "start", "midtime", or '
                                 f'"end": {tracker!r}') from None
        times = cadence.time
        return times[0] + tfrac * (times[1] - times[0])

    @staticmethod
    def _evaluates(frame, time):
        """False if the SPICE kernels cannot supply a Frame at a time.

        Parameters:
            frame (Frame): The Frame.
            time (float): The time in seconds TDB.

        Returns:
            bool: False if evaluating `frame` relative to J2000 at `time` raises
            :class:`~oops.OopsSpiceError`; True otherwise.
        """

        try:
            frame.wrt(Frame.J2000).transform_at_time(time)
        except OopsSpiceError:
            return False
        return True

    @staticmethod
    def _timeshift_letters(timeshift):
        """The time shift and the letters of a `timeshift` override, checked.

        Parameters:
            timeshift (tuple[float, str]): The override: the time shift in seconds and
                one or more of the letters "T", "P", and "F", in either case.

        Returns:
            tuple[float, str]: The time shift and the letters in upper case.

        Raises:
            OopsValueError: If the letters are empty or include any other letter.
        """

        (dt, letters) = timeshift
        letters = letters.upper()
        if not letters:
            raise OopsValueError('the timeshift must specify one or more of T/P/F')
        if set(letters) - set('TPF'):
            raise OopsValueError(f'invalid timeshift letters: {letters}')
        return (dt, letters)

    @staticmethod
    def _check_frame_overrides(tracker, parallel):
        """Raise OopsValueError if both the `tracker` and `parallel` overrides are given.

        Parameters:
            tracker (str | None): The `tracker` override.
            parallel (Observation | None): The `parallel` override.

        Raises:
            OopsValueError: If neither is None.
        """

        if parallel is not None and tracker is not None:
            raise OopsValueError('parallel and tracker cannot both be specified')

    @staticmethod
    def _check_spice_frame(frame, cadence, overrides, *, description, filepath=None,
                           defer=False):
        """Check that the SPICE kernels can supply a Frame as an observation will
        evaluate it once the overrides are applied.

        A host calls this method for an observation whose pointing comes from SPICE,
        before it calls :meth:`~oops.host.Host._apply_overrides`. Its ``from_file`` calls
        it with `defer` False, so that missing pointing raises at load time; its
        ``from_index`` calls it with `defer` True and gives the observation the Frame
        returned, so that the row is still built and the error is raised when the
        pointing is used. A :class:`~oops.frame.SpiceErrorFrame` returned stands in for
        `frame` itself, and the host gives it to
        :meth:`~oops.host.Host._apply_overrides` in place of `frame`.

        The overrides can wrap the Frame and move the time at which it is evaluated, so
        the check builds the Frame and the time as the overrides will:

        - The time is the midtime of the `cadence` override if one is given, otherwise of
          `cadence`, moved by a `timeshift` whose letters include "T".
        - A `timeshift` whose letters include "F" wraps the Frame in a
          :class:`~oops.frame.FrameShift`. A FrameShift shifts the time at which its own
          Frame is evaluated relative to that Frame's reference; the chain from that
          reference to J2000 is evaluated at the observation's time. So for a SPICE frame
          whose reference is J2000, the C kernels are read at the shifted time, and for
          a fixed rotation over a SPICE frame, such as a camera frame that flips the axes
          of the SPICE frame, they are read at the observation's time.
        - With a `tracker` override, the Frame is evaluated relative to J2000 once, when
          the tracking frame is built, at the start, midtime, or end that the override
          selects.
        - With a `parallel` override, the Frame is evaluated relative to the parallel
          observation's frame once, at the midtime, so a SPICE frame that both frames
          share is never evaluated for this observation. If that evaluation fails but
          the Frame can be evaluated relative to J2000, the failure belongs to the
          parallel observation's frame, which the override raises when it is applied,
          and `frame` is returned.
        - In every other case, the Frame is evaluated relative to J2000 at the time, so
          every Frame in its chain of references is checked, as for a Cmatrix whose
          reference is a SPICE frame.

        An exception counts as missing SPICE data only if it is of a class that cspyce
        raises (LookupError, OSError, RuntimeError, or ValueError) and its text contains
        a SPICE short error such as "SPICE(CKINSUFFDATA)", the rule of
        :meth:`~oops.OopsSpiceError.is_spice_error`; any other exception propagates
        unchanged.

        Parameters:
            frame (Frame | str): The Frame, or the ID of a registered Frame, that the
                observation uses before the overrides.
            cadence (Cadence): The cadence of the observation before the overrides.
            overrides (dict[str, Any]): The standard overrides by name, as a host's
                ``from_file`` or ``from_index`` receives them. Only "cadence",
                "timeshift", "tracker", and "parallel" are read; a missing key counts as
                None.
            description (str): What could not be obtained, such as "no Cassini ISS NAC
                pointing", for the message of the error.
            filepath (str | pathlib.Path | FCPath, optional): The file of the observation,
                for the message of the error.
            defer (bool, optional): True to return a
                :class:`~oops.frame.SpiceErrorFrame` if the SPICE kernels cannot supply
                the Frame; False to raise :class:`~oops.OopsSpiceError`.

        Returns:
            Frame | str: `frame` itself if the observation can evaluate it; otherwise, if
            `defer` is True, a :class:`~oops.frame.SpiceErrorFrame` that raises
            :class:`~oops.OopsSpiceError` whenever it is evaluated.

        Raises:
            KeyError: If `frame` is the ID of a Frame that is not registered.
            OopsSpiceError: If `defer` is False and the SPICE kernels cannot supply
                `frame` as the observation will evaluate it. It is built with
                :meth:`~oops.OopsSpiceError.from_spice_error`, so its message reads
                "<description> at <time> for <filepath>: <SPICE code>", in which the time
                is the observation's time at which the Frame is evaluated, without the
                seconds of an "F" time shift, and its cause is the exception that cspyce
                raised.
            OopsValueError: If the `timeshift` letters are invalid, the `tracker` override
                is not "start", "midtime", or "end", or both `tracker` and `parallel` are
                given; :meth:`~oops.host.Host._apply_overrides` raises the same errors.
        """

        # A Frame ID that is not registered is the caller's error, not missing SPICE data
        (shifted, reference, time) = Host._pointing_probe(Frame.as_frame(frame), cadence,
                                                          overrides)
        err = Host._spice_failure(shifted, reference, time)
        if err is not None and reference is not Frame.J2000:
            err = Host._spice_failure(shifted, Frame.J2000, time)

        if err is None:
            return frame
        if defer:
            return SpiceErrorFrame(err, description=description, filepath=filepath)
        raise OopsSpiceError.from_spice_error(err, description=description, time=time,
                                              filepath=filepath) from err

    @staticmethod
    def _pointing_probe(frame, cadence, overrides):
        """The Frame, reference, and time at which an observation will evaluate its
        pointing once the overrides are applied.

        Parameters:
            frame (Frame): The Frame of the observation before the overrides.
            cadence (Cadence): The cadence of the observation before the overrides.
            overrides (dict[str, Any]): The standard overrides by name; see
                :meth:`~oops.host.Host._check_spice_frame`.

        Returns:
            tuple[Frame, Frame, float | Scalar]: `frame`, wrapped in a
            :class:`~oops.frame.FrameShift` for a `timeshift` whose letters include "F";
            the Frame relative to which the observation evaluates it, which is the
            `parallel` observation's frame if that override is given and J2000
            otherwise; and the time in seconds TDB.

        Raises:
            OopsValueError: If the `timeshift` letters are invalid, the `tracker` override
                is not "start", "midtime", or "end", or both `tracker` and `parallel` are
                given.
        """

        # The cadence and the Frame are wrapped as _apply_overrides wraps them
        cadence = overrides.get('cadence') or cadence
        timeshift = overrides.get('timeshift')
        if timeshift is not None:
            (dt, letters) = Host._timeshift_letters(timeshift)
            if 'T' in letters:
                cadence = TimeShift(dt, cadence)
            if 'F' in letters:
                frame = FrameShift(dt, frame)

        tracker = overrides.get('tracker')
        parallel = overrides.get('parallel')
        Host._check_frame_overrides(tracker, parallel)
        if parallel is not None:
            return (frame, Frame.as_frame(parallel.frame), cadence.midtime)
        if tracker is not None:
            return (frame, Frame.J2000, Host._tracker_epoch(cadence, tracker))
        return (frame, Frame.J2000, cadence.midtime)

    @staticmethod
    def _spice_failure(frame, reference, time):
        """The exception that reports missing SPICE data when a Frame is evaluated.

        Parameters:
            frame (Frame): The Frame.
            reference (Frame): The Frame relative to which `frame` is evaluated.
            time (float | Scalar): The time in seconds TDB.

        Returns:
            Exception | None: The exception raised by evaluating `frame` relative to
            `reference` at `time`, if :meth:`~oops.OopsSpiceError.is_spice_error` counts
            it as missing SPICE data; None if the evaluation succeeds.

        Raises:
            Exception: Any exception of the evaluation that is not about SPICE data,
                unchanged.
        """

        try:
            frame.wrt(reference).transform_at_time(time)
        except OopsSpiceError.CSPYCE_ERRORS as err:
            if not OopsSpiceError.is_spice_error(err):
                raise
            return err
        return None

    @staticmethod
    def _index_file_path(volumes, row_dict, *, volume_column='VOLUME_ID',
                         file_column='FILE_SPECIFICATION_NAME'):
        """The path to the file that a row of an index lists.

        The file is given relative to the root directory of its volume, and the volume by
        its name. Leading and trailing blanks of both values are ignored.

        Parameters:
            volumes (FCPath): The directory that holds the volume directories, as
                :meth:`~oops.host.Host._volumes_directory` gives it.
            row_dict (dict[str, Any]): The row.
            volume_column (str, optional): The column that gives the volume's name.
            file_column (str, optional): The column that gives the file's path within its
                volume.

        Returns:
            FCPath: `volumes` / the volume name / the file's path.
        """

        return (volumes / str(row_dict[volume_column]).strip()
                / str(row_dict[file_column]).strip())

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
