################################################################################
# hosts/juno/jiram/gold_master.py: Backplane gold master tester for Juno JIRAM
################################################################################
import oops.backplane.gold_master as gm
from hosts.juno.jiram import standard_obs

if __name__ == '__main__':
    gm.execute_as_command()

################################################################################
