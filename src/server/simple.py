
from loguru import logger
import numpy as np
from server.fedavgserver import FedAvgServer

def get_label_count(test1):
    cnt = [0]*10
    for _,y1 in test1:
        cnt[y1]+=1
    return cnt
def distribution_distance(c1,c2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    c1 = c1/sum(c1)
    c2 = c2/sum(c2)
    return np.linalg.norm(c1-c2)
class Simple(FedAvgServer):
    def __init__(self, args=None,clients = None , **kwargs):
        super().__init__(args,clients, **kwargs)
      

    def sampling_for_clients(self):
        li=[
            [0,1],
            [0,2],
            [0,3],
            [1,2],
            [1,3],
            [2,3]
            ]
        x = li[self.args['seq_num']]
        return [self.clients[x[0]],
                self.clients[x[1]]]
    
    def find_two_client(self):
        score ={}
        global_cnt = get_label_count(self.global_testdataset)
    
        for i in range(0,len(self.clients)):
            for j in range(i+1,len(self.clients)):
                c1 = get_label_count(self.clients[i].train_dataset)
                c2 = get_label_count(self.clients[j].train_dataset)
                c = np.array(c2)+np.array(c1)

                distance = distribution_distance(c,global_cnt)
                score[(i,j)] = (distance,sum(c1)+sum(c2))
        logger.info(f"score:{score}")
          
    def before_fed_training(self):
        super().before_fed_training()
        self.find_two_client()
    
        