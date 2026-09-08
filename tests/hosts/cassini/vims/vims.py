##########################################################################################
# tests/hosts/cassini/vims.py
##########################################################################################
import programs.gold_master as gm


def test_Cassini_VIMS_GoldMaster_v1690952775():
    """
    *** fails because vims needs updating ***

    v1690952775 Compare w Gold Masters

    To preview and regenerate gold masters (from pds-oops/oops/backplane/):
        python gold_master.py \
            ~/Dropbox-SETI/OOPS-Resources/test_data/cassini/VIMS/v1793917030_1.qub \
            --module hosts.cassini.vims \
            --planet SATURN \
            --no-inventory \
            --preview

        python gold_master.py \
            ~/Dropbox-SETI/OOPS-Resources/test_data/cassini/VIMS/v1793917030_1.qub \
            --module hosts.cassini.vims \
            --planet SATURN \
            --no-inventory \
            --adopt
    """

    gm.execute_as_pytest(
        obspath = 'cassini/VIMS/v1793917030_1.qub',
        index   = None,
        module  = 'oops.hosts.cassini.vims',
        planet  = 'SATURN',
        moon    = '',
        ring    = '',
        inventory=False, border=10)

##########################################################################################
