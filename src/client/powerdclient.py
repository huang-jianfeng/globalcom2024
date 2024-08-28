from functools import cmp_to_key
import os
import sys

from loguru import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRO_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from util.utils import evaluate
from other.counter import SimpleCounter, get_counter
from client.fedavgclient import FedAvgClient
import torch
from torch.utils.data import DataLoader,Subset
from client.streamfedavgclient import StreamFedAvgClient

class PowerdClient(StreamFedAvgClient):
    def __init__(self, train_dataset=None, test_dataset=None,
                  model=None, args=None,id=None) -> None:
        
        super().__init__(train_dataset, test_dataset, model, args,id)
        logger.trace("PowerdClient init.")
        self.lastloss = float('inf')


    def train(self,model=None):
        
        logger.debug(f"{self}:traing...")
        lr = self.args['local_lr']
        train_batch_size = self.args['train_batch_size']
        momentum = self.args['momentum']
        local_epoch = self.args['local_epoch']
        dataset = Subset(self.train_dataset,
                         self.total_data_indices[self.start:self.cur_data_pos])
        dataloader = DataLoader(dataset,train_batch_size,True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(),lr=lr,momentum=momentum)
        
        if model is None:
            model = self.model
        model.train()
        total_loss = 0
        total_samples = 0
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
            total_samples += len(y)

        self.lastloss = total_loss/total_samples
        return total_loss,total_samples
    @torch.inference_mode()
    def test_on_train(self,model=None):
        # 训练集测试模型
        batch_size = self.args['test_batch_size']
        dataset = Subset(self.train_dataset,
                         self.total_data_indices[self.start:self.cur_data_pos])
        dataloader = DataLoader(dataset,batch_size=batch_size)
        logger.debug(f"{self}:test on train dataset[{len(dataset)}]")
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        if model is None:
            model = self.model 
        return evaluate(model,dataloader,criterion)
    def __repr__(self) -> str:
        return f"stream-cli-{self.id}[{len(self.train_dataset)}-{len(self.test_dataset)}]-{self.start}:{self.cur_data_pos}"