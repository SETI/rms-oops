################################################################################
# tests/hosts/cassini/iss/__init__.py
################################################################################
#
#import os.path
#import unittest
#import oops.backplane.gold_master as gm
#
#from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY
#
#
##===============================================================================
#class Test_Cassini_ISS(unittest.TestCase):
#
#    #===========================================================================
#    def runTest(self):
#
#        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY
#
#        snapshots = from_index(os.path.join(TESTDATA_PARENT_DIRECTORY,
#                                            'cassini/ISS/index.lbl'))
#        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY,
#                                          'cassini/ISS/W1575634136_1.IMG'))
#        snapshot3940 = snapshots[3940]  #should be same as snapshot
#
#        self.assertTrue(abs(snapshot.time[0] - snapshot3940.time[0]) < 1.e-3)
#        self.assertTrue(abs(snapshot.time[1] - snapshot3940.time[1]) < 1.e-3)
#
#
##===============================================================================
#class Test_Cassini_ISS_GoldMaster_N1460072401(unittest.TestCase):
#
#    #===========================================================================
#    def runTest(self):
#        """
#        N1460072401 Compare w Gold Masters
#
#        To preview and regenerate gold masters (from pds-oops/oops/backplane/):
#            python gold_master.py \
#                ~/Dropbox-SETI/OOPS-Resources/test_data/cassini/ISS/N1460072401_1.IMG \
#                --module hosts.cassini.iss \
#                --planet SATURN \
#                --ring SATURN_MAIN_RINGS \
#                --no-inventory \
#                --preview
#
#            python gold_master.py \
#                ~/Dropbox-SETI/OOPS-Resources/test_data/cassini/ISS/N1460072401_1.IMG \
#                --module hosts.cassini.iss \
#                --planet SATURN \
#                --ring SATURN_MAIN_RINGS \
#                --no-inventory \
#                --adopt
#        """
#        gm.execute_as_unittest(self,
#                obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
#                                       'cassini/ISS/N1460072401_1.IMG'),
#                index   = None,
#                module  = 'oops.hosts.cassini.iss',
#                planet  = 'SATURN',
#                moon    = '',
#                ring    = 'SATURN_MAIN_RINGS',
#                inventory=False, border=10)
#
#
##===============================================================================
#class Test_Cassini_ISS_GoldMaster_W1573721822_1(unittest.TestCase):
#
#    #===========================================================================
#    def runTest(self):
#        """
#        W1573721822 Compare w Gold Masters
#
#        To preview and regenerate gold masters (from pds-oops/oops/backplane/):
#            python gold_master.py \
#                ~/Dropbox-SETI/OOPS-Resources/test_data/cassini/ISS/W1573721822_1.IMG \
#                --module hosts.cassini.iss \
#                --planet SATURN \
#                --moon EPIMETHEUS \
#                --ring SATURN_MAIN_RINGS \
#                --no-inventory \
#                --preview
#
#            python gold_master.py \
#                ~/Dropbox-SETI/OOPS-Resources/test_data/cassini/ISS/W1573721822_1.IMG \
#                --module hosts.cassini.iss \
#                --planet SATURN \
#                --moon EPIMETHEUS \
#                --ring SATURN_MAIN_RINGS \
#                --no-inventory \
#                --adopt
#        """
#        gm.execute_as_unittest(self,
#                obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
#                                       'cassini/ISS/W1573721822_1.IMG'),
#                index   = None,
#                module  = 'oops.hosts.cassini.iss',
#                planet  = 'SATURN',
#                moon    = 'EPIMETHEUS',
#                ring    = 'SATURN_MAIN_RINGS',
#                inventory=False, border=10)
#
#
#############################################
#if __name__ == '__main__':
#    unittest.main(verbosity=2)







#
# import unittest
# import os.path
#
# from oops.unittester_support            import TESTDATA_PARENT_DIRECTORY
# from oops.backplane.exercise_backplanes import exercise_backplanes
# from oops.backplane.unittester_support  import Backplane_Settings
#
#
# #*******************************************************************************
# class Test_Cassini_ISS(unittest.TestCase):
#
#     def runTest(self):
#
#         from oops.unittester_support import TESTDATA_PARENT_DIRECTORY
#
#         snapshots = from_index(os.path.join(TESTDATA_PARENT_DIRECTORY,
#                                             'cassini/ISS/index.lbl'))
#         snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY,
#                                           'cassini/ISS/W1575634136_1.IMG'))
#         snapshot3940 = snapshots[3940]  #should be same as snapshot
#
#         self.assertTrue(abs(snapshot.time[0] - snapshot3940.time[0]) < 1.e-3)
#         self.assertTrue(abs(snapshot.time[1] - snapshot3940.time[1]) < 1.e-3)
#
#
# #*******************************************************************************
# class Test_Cassini_ISS_Backplane_Exercises(unittest.TestCase):
#
#     #===========================================================================
#     def runTest(self):
#
#         if Backplane_Settings.NO_EXERCISES:
#             self.skipTest('')
#
#         root = os.path.join(TESTDATA_PARENT_DIRECTORY, 'cassini/ISS')
#         file = os.path.join(root, 'N1460072401_1.IMG')
#         obs = from_file(file)
#         exercise_backplanes(obs, use_inventory=True, inventory_border=4,
#                                  planet_key='SATURN')
#
#
# ############################################
# from oops.backplane.unittester_support import backplane_unittester_args
#
# if __name__ == '__main__':
#     backplane_unittester_args()
#     unittest.main(verbosity=2)
################################################################################
