from copy import deepcopy
from loguru import logger
import torch
from torch.utils.data import DataLoader
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR )
from util.utils import evaluate
from client.baseclient import BaseClient
from client.fedavgclient import FedAvgClient
from util.tools import trainable_params

class FedProxClient(FedAvgClient):
    def __init__(self, train_dataset=None, test_dataset=None, model=None, args=None, id=None) -> None:
        super().__init__(train_dataset, test_dataset, model, args, id)

    def train(self, model=None):
        logger.debug(f"{self}:traing...train_set[{len(self.train_dataset)}]")
        lr = self.args['local_lr']
        train_batch_size = self.args['train_batch_size']
        momentum = self.args['momentum']
        local_epoch = self.args['local_epoch']
        dataloader = DataLoader(self.train_dataset,train_batch_size,True)
        criterion = torch.nn.CrossEntropyLoss()
        if model is None:
            model = self.model
        global_model = deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
        
        global_params = trainable_params(global_model) 
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

 # logger.info(')_______fedprox....')
            for w, w_t in zip(trainable_params(model), global_params):
                            # logger.info(f'fedprox....{self.args["mu"]}')
       
                            w.grad.data += self.args['mu'] * (w.data - w_t.data)
       

            optimizer.step()
            total_loss += loss.item()
            total_sample += len(y)
               
        return total_loss,total_sample
    
    def __repr__(self) -> str:
        return f"fedprox-cli-{self.id}[{len(self.train_dataset)}-{len(self.test_dataset)}]"