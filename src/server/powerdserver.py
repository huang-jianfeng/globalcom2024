
from functools import cmp_to_key
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRO_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from other.counter import SimpleCounter
from server.streamavgserver import StreamFedAvgServer
class PowerdServer(StreamFedAvgServer):
    def __init__(self, args, clients, **kwargs):
        super().__init__(args, clients, **kwargs)
    
    def sampling_for_clients(self):

        sampling_ratio = self.args['sampling_ratio']
        sampled_num = max(int(sampling_ratio*len(self.clients)),1)
        def cli_cmp(c1,c2):
            return c2.lastloss-c1.lastloss
        clis = sorted(self.clients,key=cmp_to_key(cli_cmp))
        # sorted(self.clients,key=cmp_to_key(cli_cmp))
        return clis[0:sampled_num]