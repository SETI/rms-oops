################################################################################
# oops/hosts/voyager/iss.py
################################################################################

################################################################################
# UNIT TESTS
################################################################################
#import unittest
#import os.path
#import oops.backplane.gold_master as gm
#
#from oops.unittester_support            import TEST_DATA_PREFIX
#
#
##===============================================================================
#class Test_VGR_ISS(unittest.TestCase):
#
#    #===========================================================================
#    def runTest(self):
#        pass
#
#
##===============================================================================
#class Test_VGR_ISS_GoldMaster_C3450201_GEOMED(unittest.TestCase):
#
#    #===========================================================================
#    def runTest(self):
#        """
#        C3450201_GEOMED Compare w Gold Masters
#
#        To preview and regenerate gold masters (from pds-oops/oops/backplane/):
#            python gold_master.py \
#                ~/Dropbox-SETI/OOPS-Resources/test_data/voyager/ISS/VGISS_6109/C34502XX/C3450201_GEOMED.img \
#                --module hosts.voyager.iss \
#                --planet SATURN \
#                --ring SATURN_MAIN_RINGS \
#                --no-inventory \
#                --preview
#
#            python gold_master.py \
#                ~/Dropbox-SETI/OOPS-Resources/test_data/voyager/ISS/VGISS_6109/C34502XX/C3450201_GEOMED.img \
#                --module hosts.voyager.iss \
#                --planet SATURN \
#                --ring SATURN_MAIN_RINGS \
#                --no-inventory \
#                --adopt
#        """
#        gm.execute_as_unittest(self,
#                obspath = 'voyager/ISS/VGISS_6109/C34502XX/C3450201_GEOMED.img',
#                index   = None,
#                module  = 'oops.hosts.voyager.iss',
#                planet  = 'SATURN',
#                moon    = '',
#                ring    = 'SATURN_MAIN_RINGS',
#                inventory=False, border=10)
#
#
###############################################
#if __name__ == '__main__':
#    unittest.main(verbosity=2)







# import unittest
# import os.path
#
# from oops.unittester_support            import TEST_DATA_PREFIX
# from oops.backplane.exercise_backplanes import exercise_backplanes
# from oops.backplane.unittester_support  import Backplane_Settings
#
#
# #*******************************************************************************
# class Test_Voyager_ISS_Backplane_Exercises(unittest.TestCase):
#
#     def runTest(self):
#
#         if Backplane_Settings.NO_EXERCISES:
#             self.skipTest('')
#
#         file = TEST_DATA_PREFIX.retrieve('voyager/ISS/c3440346.gem')
#         obs = from_file(file)
#         exercise_backplanes(obs, use_inventory=True, inventory_border=4,
#                                  planet_key='SATURN')
#
# ##############################################
# from oops.backplane.unittester_support import backplane_unittester_args
#
# if __name__ == '__main__':
#     backplane_unittester_args()
#     unittest.main(verbosity=2)
################################################################################
