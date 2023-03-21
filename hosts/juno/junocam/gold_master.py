################################################################################
# hosts/juno/junocam/gold_master.py: Backplane gold master tester for JunoCam
################################################################################
import standard_obs
import oops.backplane.gold_master as gm

# Because JunoCam has such a large, distorted FOV, we need to assign the
# backplanes an especially large inventory border: border=10 seems to work.
# However, inventory=False is safer still.
gm.set_default_args(inventory=False, border=10)

if __name__ == '__main__':
    gm.execute_standard_command(exclude='default')

################################################################################









