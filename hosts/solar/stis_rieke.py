################################################################################
# hosts/solar/stis_rieke.py: STIS and Rieke models merged.
################################################################################

import numpy as np
import os

import hosts.solar       as solar
import hosts.solar.stis  as stis
import hosts.solar.rieke as rieke

import tabulation as tab

if 'FLUX_DENSITY' not in globals(): # pragma: no cover

    assert stis.UNITS  == rieke.UNITS
    assert stis.XUNITS == rieke.XUNITS

    tab1 = stis.FLUX_DENSITY
    tab2 = rieke.FLUX_DENSITY

    mask = (tab2.x > np.max(tab1.x))

    merged_x = np.hstack((tab1.x, tab2.x[mask]))
    merged_y = np.hstack((tab1.y, tab2.y[mask]))

    FLUX_DENSITY = tab.Tabulation(merged_x, merged_y)
    UNITS = stis.UNITS
    XUNITS = stis.XUNITS

################################################################################
