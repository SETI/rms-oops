##########################################################################################
# tests/hosts/cassini/vims.py
##########################################################################################
import unittest
import os.path
import oops.backplane.gold_master as gm


class Test_Cassini_VIMS_GoldMaster_v1690952775(unittest.TestCase):

    def runTest(self):
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

        gm.execute_as_unittest(self,
                obspath = 'cassini/VIMS/v1793917030_1.qub',
                index   = None,
                module  = 'oops.hosts.cassini.vims',
                planet  = 'SATURN',
                moon    = '',
                ring    = '',
                inventory=False, border=10)

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
##########################################################################################
