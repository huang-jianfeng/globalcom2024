import os
import sys
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from server.baseserver import BaseServer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRO_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from util.utils import avg_model, evaluate


class FedAvgServer(BaseServer):

    def __init__(self, args=None,clients=None,**kwargs):
        super().__init__(**kwargs)
        self.__args = args
        self.__global_testdataset  = None
        self.clients = clients
        self.global_dis = None
    
    @property
    def global_testdataset(self):
        return self.__global_testdataset
    
    @global_testdataset.setter
    def global_testdataset(self,testdataset):
        self.__global_testdataset = testdataset
    
    @property
    def args(self):
        return self.__args
    
    @args.setter
    def args(self,args):
        self.__args = args

    def prep_for_one_round(self):
        logger.info(f"starting traing for round [{self.round:}]:")
        if (self.round+1) % self.args['decay_gap']==0:
            self.args['local_lr'] *= self.args['decay_cofficient']
    
    def sampling_for_clients(self):
        logger.trace(f"{self}:random sampling:")
        sampling_ratio = self.args['sampling_ratio']
        sampled_num = max(int(sampling_ratio*len(self.clients)),1)
        sampled = np.random.choice(self.clients,sampled_num)
        logger.trace(f"{sampled}")
        return sampled
    
    def aggreagte(self,models,weights):
        if self.args['agg'] == 'datasize':
            global_dict = avg_model(models,weights)
        elif self.args['agg'] == 'avg':
            global_dict = avg_model(models,[1 for i in range(len(models))])
        else:
            raise Exception("how to agg models?")
        self.model.load_state_dict(global_dict)
        
    def test_on_global_testdataset(self,model=None):
        """在global 测试集上测试

        Returns:
            总的loss 总的样本 总的正确的样本
        """        
        logger.debug(f"{self}:test on test dataset[{len(self.global_testdataset)}].....")
        batch_size = self.args['test_batch_size']
        dataloader = DataLoader(self.global_testdataset,batch_size=batch_size)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        if model is None:
            model = self.model
        return evaluate(model,dataloader,criterion)
    def test_on_all_client(self,on_train=False):
        """ 
        测试gloabl model

        on train : 
            True:在训练集上测试; 
            False: 在client的测试集上测试
        """
        corrects=0
        loss = 0
        cnt = 0

        for cli in self.clients:
            l,num,num_correct =0,0,0
            if on_train:
                num,num_correct = cli.test_on_train()
            else:
               l,num,num_correct = cli.test_on_test()
            loss += l
            cnt += num
            corrects += num_correct
        return loss,cnt,corrects
    def before_fed_training(self):
        logger.debug("start federated training")
        class_num =10
        if self.args['dataset'] == 'cifar100':
            class_num = 100
        counter = [0]*class_num
        for _,y in self.global_testdataset:
            counter[y%class_num] += 1
        counter = np.array(counter)
        counter = counter/sum(counter)
        self.global_dis = counter
    
        
        

