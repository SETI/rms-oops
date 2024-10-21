##########################################################################################
# tests/hosts/cassini/uvis.py
##########################################################################################
import unittest
import os.path
import oops.backplane.gold_master as gm


#===============================================================================
class Test_Cassini_UVIS_GoldMaster_HSP2014_197_21_29(unittest.TestCase):

    #===========================================================================
    def runTest(self):
        """
        **** fails because uvis needs updating ****

        HSP2014_197_21_29 Compare w Gold Masters

        To preview and regenerate gold masters (from pds-oops/oops/backplane/):
            python gold_master.py \
                ~/Dropbox-SETI/OOPS-Resources/test_data/cassini/UVIS/HSP2014_197_21_29.DAT \
                --module hosts.cassini.uvis \
                --ring SATURN_MAIN_RINGS \
                --no-inventory \
                --preview

            python gold_master.py \
                ~/Dropbox-SETI/OOPS-Resources/test_data/cassini/UVIS/HSP2014_197_21_29.DAT \
                --module hosts.cassini.uvis \
                --ring SATURN_MAIN_RINGS \
                --no-inventory \
                --adopt
        """

        gm.execute_as_unittest(self,
                obspath = 'cassini/UVIS/HSP2014_197_21_29.DAT',
                index   = None,
                module  = 'oops.hosts.cassini.uvis',
                planet  = '',
                moon    = '',
                ring    = 'SATURN_MAIN_RINGS',
                inventory=False, border=10)


############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
