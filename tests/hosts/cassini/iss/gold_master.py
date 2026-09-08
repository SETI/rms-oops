##########################################################################################
# tests/hosts/cassini/iss/gold_master.py
##########################################################################################

import programs.gold_master as gm

# Imported for its side effect: it defines the standard observations and sets the
# module whose from_file method reads them.
from tests.hosts.cassini.iss import standard_obs      # noqa: F401

if __name__ == '__main__':
    gm.execute_as_command()

##########################################################################################
