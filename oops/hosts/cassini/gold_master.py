################################################################################
# oops/hosts/cassini/gold_master.py: Backplane gold master tester for
# Cassini ISS
################################################################################
import oops.backplane.gold_master as gm
from oops.hosts.cassini import standard_obs

if __name__ == '__main__':
    gm.execute_as_command()

################################################################################
