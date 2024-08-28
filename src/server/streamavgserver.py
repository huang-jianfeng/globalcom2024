
import os
import sys
from loguru import logger
import numpy as np
import cvxpy as cp
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRO_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from other.counter import SimpleCounter
from server.fedavgserver import FedAvgServer


class StreamFedAvgServer(FedAvgServer):
    def __init__(self, args,clients,**kwargs):
        super().__init__(args,clients,**kwargs)

        # 获取 target counter：
    #     self.counter = SimpleCounter(10,lambda x : x)
    #     for client in self.clients:
    #         self.counter.merge(client.counter)
        
    # @property
    # def counter(self):
    #     return self.counter
    
    # @counter.setter
    # def counter(self,counter):
    #     self.counter = counter
    
    def sampling_for_clients(self):
        if self.args['sample'] == "random":
            logger.info("random sampling")
            sampling_ratio = self.args['sampling_ratio']
            sampled_num = max(int(sampling_ratio*len(self.clients)),1)
            sampled = np.random.choice(self.clients,sampled_num,replace=False)
        elif self.args['sample'] == "counter":
            logger.info("counter sampling")
            counters = None
            for client in self.clients:
                if counters is None:
                    counters = np.array(client.counter.get_counter())
                else:
                    counters = np.vstack([counters,np.array(client.counter.get_counter())])
            
            # total_counter = np.sum(counters,axis=0)
            # # total_counter = np.ones(10)
            # total_counter = total_counter/sum(total_counter)
            
            total_counter = self.global_dis
            
            # 根据各个client的counter构建一个矩阵
            matrix = None
            for i in range(counters.shape[0]):
                if matrix is None:
                    matrix = counters[i,:]/sum(counters[i,:])
                else:
                    matrix = np.vstack([matrix,counters[i,:]/sum(counters[i,:])])
    
            matrix = matrix.transpose()

            # 定义变量，限制条件
            x = cp.Variable(matrix.shape[1])
            constrains = [0<=x,x<=1]
            # 定义目标
            objective = cp.Minimize(cp.sum_squares(matrix@x-total_counter))
            # 问题
            prob = cp.Problem(objective=objective,constraints=constrains)
            prob.solve()
            # 求概率
            probablity =  x.value
            distance = prob.value
            for i,p in enumerate(probablity):
                if p <=0:
                    probablity[i] = 1e-6
            probablity = probablity/sum(probablity)
            logger.info(f"distance={distance:<6.4f},prob = {probablity}")
            sampled_num = max(1,int(self.args['sampling_ratio']*len(self.clients)))
            sampled = np.random.choice(self.clients,sampled_num,replace=False,p=probablity)
        logger.info(f"sampled clients:{sampled}")
        return sampled
    
    def aggreagte(self, models, weights):
        return super().aggreagte(models, weights)
    
    def prep_for_one_round(self):
        super().prep_for_one_round()
        if (self.round+1) % self.args['data_inc_gap']==0:
            logger.trace("数据增长:")
            for client in self.clients:
               client.stream_coming(self.args['data_inc_ratio'])
