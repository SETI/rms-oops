################################################################################
# oops/cadence/timeshift.py: TimeShift subclass of class Cadence
################################################################################

import numpy as np

from polymath import Scalar
from oops.cadence import Metronome, TimeShift


def test_timeshift():
    DT = 10.

    # 100-110, 110-120, 120-130, 130-140
    cadence = Metronome(100., 10., 10., 4)
    shifted = TimeShift(DT, cadence)
    assert shifted.dt == DT
    assert shifted.link is None

    # The shift is held privately and published as a property, as in PathShift and
    # FrameShift; everything the Cadence contract requires stays a public attribute,
    # as in the other Cadence subclasses
    assert isinstance(type(shifted).dt, property)
    assert isinstance(type(shifted).link, property)
    assert sorted(a for a in vars(shifted) if a.startswith('_')) == ['_dt', '_link']
    for name in ('cadence', 'time', 'midtime', 'lasttime', 'shape',
                 'is_continuous', 'is_unique', 'min_tstride', 'max_tstride'):
        assert name in vars(shifted)

    # Every time is offset by dt, and the shape is unchanged
    assert shifted.shape == cadence.shape
    assert shifted.time == (cadence.time[0] + DT, cadence.time[1] + DT)
    assert shifted.midtime == cadence.midtime + DT

    tstep = Scalar(np.arange(0., 4., 0.25))
    assert shifted.time_at_tstep(tstep) == cadence.time_at_tstep(tstep) + DT
    assert shifted.tstep_at_time(cadence.time_at_tstep(tstep) + DT) == tstep

    # A linked TimeShift tracks the offset of the object it is linked to
    linked = TimeShift(shifted, cadence)
    assert linked.link is shifted
    assert linked.dt == DT

    shifted.set_params(np.array([2. * DT]))
    linked._refresh()
    assert linked.dt == 2. * DT
    assert linked.time_at_tstep(tstep) == cadence.time_at_tstep(tstep) + 2. * DT
################################################################################
