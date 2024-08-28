from copy import deepcopy
from loguru import logger

from functools import cmp_to_key
import os
import sys

from loguru import logger
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRO_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from client.streamfedavgclient import StreamFedAvgClient
from other.counter import SimpleCounter, get_counter
from client.fedavgclient import FedAvgClient
import torch
from torch.utils.data import DataLoader,Subset
from util.tools import trainable_params

class StreamFedProxClient(StreamFedAvgClient):
    def __init__(self, train_dataset=None, test_dataset=None, model=None, args=None, id=None) -> None:
        super().__init__(train_dataset, test_dataset, model, args, id)
        
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
        
        if model is None:
            model = self.model
        global_model = deepcopy(model)
        global_params = trainable_params(global_model, detach=True)
        optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
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

            # logger.info(')_______fedprox....')
            for w, w_t in zip(trainable_params(model), global_params):
                            # logger.info(f'fedprox....{self.args["mu"]}')
                            w.grad.data += self.args['mu'] * (w.data - w_t.data)

            optimizer.step()
            total_loss += loss.item()
            total_samples += len(y)
           
        return total_loss,total_samples