from spice_stcx01 import stcl01, stcf01, stcg01

class SPICE_StarCat(object):
    """Python interface to the SPICE star catalog system.

    Example:
        >>> from spice_starcat import SPICE_StarCat as StarCat
        >>> tycho2_catalog = StarCat("path/tycho2.bdb")
        >>> stars = tycho2_catalog.search(2.,2.1,.4,.5)
        >>> print len(stars)
        1705
        >>> print stars[0]
        (2.0000488236546516, 0.4878883436322212, 3.278255600123402e-07,
        3.5762788076370253e-07, 190721, '040', 11.460000038146973)
        # (ra, dec, ra_uncertainty, dec_uncertainty,
        #  catalog_number, spectral_type, v_magnitude)
    """

    #===========================================================================
    def __init__(self, filename):
        """Create a star catalog object by reading a specified SPICE binary
        star catalog file "*.bdb".

        Input:
            filename        name of the star catalog object.
        """

        self.filename = filename
        self.catalog = stcl01(filename)

    #===========================================================================
    def search(self, west_ra, east_ra, south_dec, north_dec):
        """A list of stars found. Each star is described by a tuple containing
        (RA, dec, RA uncertainty, dec uncertainty, catalog number, spectral
        type, V magnitude).

        Input:
            west_ra         lower limit on RA (radians).
            east_ra         upper limit on RA (radians). This value should be
                            less than west_ra for searches that cross RA=0.
            south_dec       lower limit on dec (radians).
            north_dec       upper limit on dec (radians).
        """

        nstars = stcf01(self.catalog, west_ra, east_ra, south_dec, north_dec)

        results = []
        for i in range(1, nstars+1):
            results.append(tuple(stcg01(i)))

        return results

#*******************************************************************************

