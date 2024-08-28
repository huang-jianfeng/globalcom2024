from loguru import logger
import torch
from torch.utils.data import DataLoader
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR )
from util.utils import evaluate
from client.baseclient import BaseClient

class FedAvgClient(BaseClient):
    def __init__(self,train_dataset=None,test_dataset=None,model=None,args=None,id=None) -> None:
        self.__train_dataset = train_dataset
        self.__test_dataset = test_dataset
        self.model = model
        self.__args  = args
        self.id=id
    @property
    def args(self):
        return self.__args
    @args.setter
    def args(self,args):
        self.__args = args

        
    def get_train_sample_num(self):
        return len(self.train_dataset)
    @property
    def train_dataset(self):
        # logger.info("get train dataset")
        return self.__train_dataset
    @train_dataset.setter
    def train_dataset(self,train_dataset):
        self.__train_dataset = train_dataset
    
    @property
    def test_dataset(self):
        return self.__test_dataset
    
    @test_dataset.setter
    def test_dataset(self,test_dataset):
        self.__test_dataset = test_dataset
    

    def get_model_dict(self):
        return self.model.state_dict()

    def train(self,model=None):
        logger.debug(f"{self}:traing...train_set[{len(self.train_dataset)}]")

        lr = self.args['local_lr']
        train_batch_size = self.args['train_batch_size']
        momentum = self.args['momentum']
        local_epoch = self.args['local_epoch']
        dataloader = DataLoader(self.train_dataset,train_batch_size,True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(),lr=lr,momentum=momentum)
        
        if model is None:
            model = self.model
        model.train()
        cnt = 0
        total_loss = 0
        total_sample = 0
        total_correct = 0
        iterator = iter(dataloader)
        for epoch in range(local_epoch):
            try:
                x,y = next(iterator)
            except:
                iterator = iter(dataloader)
                x,y = next(iterator)
    
            model.zero_grad()
            output = model(x)
            loss = criterion(output,y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_sample += len(y)
               
        return total_loss,total_sample

    @torch.inference_mode()
    def test_on_test(self,model=None):
        # 测试集测试模型
        logger.debug(f"{self}:test on test dataset[{len(self.test_dataset)}]")
        batch_size = self.args['test_batch_size']
        dataloader = DataLoader(self.test_dataset,batch_size=batch_size)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        if model is None:
            model = self.model
        return evaluate(model,dataloader,criterion)
    
    @torch.inference_mode()
    def test_on_train(self,model=None):
        # 训练集测试模型
        logger.debug(f"{self}:test on train dataset[{len(self.train_dataset)}]")
        batch_size = self.args['test_batch_size']
        dataloader = DataLoader(self.train_dataset,batch_size=batch_size)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        if model is None:
            model = self.model 
        return evaluate(model,dataloader,criterion)
    
    def __repr__(self) -> str:
        return f"cli-{self.id}[{len(self.train_dataset)}-{len(self.test_dataset)}]"
