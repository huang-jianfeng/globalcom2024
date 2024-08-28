from functools import cmp_to_key
import os
import sys

from loguru import logger
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRO_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from other.counter import SimpleCounter, get_counter
from client.fedavgclient import FedAvgClient
import torch
from torch.utils.data import DataLoader,Subset

class StreamFedAvgClient(FedAvgClient):
    def __init__(self, train_dataset=None, test_dataset=None,
                  model=None, args=None,id=None) -> None:
        
        super().__init__(train_dataset, test_dataset, model, args)
        logger.trace("streamfedavgclient init.")
        self.cur_data_pos = 0
        self.total_data_indices = [i for i in range(len(self.train_dataset))]
        self.id = id

        self.__counter = get_counter(self.args)
        self.start = 0
        self.capacity = self.args['capacity']

        class_num = 10
        if args['dataset'] == 'cifar100':
            class_num = 100

        # **按照标签将data_indices 排序
        def lable_compare(x,y):
            _,label1 = self.train_dataset[x]
            _,label2 = self.train_dataset[y]
            if label1 > label2:
                return 1
            elif label1 < label2:
                return -1
            else:
                return 0
        if self.args['resort']:
            self.total_data_indices = sorted(self.total_data_indices,key=cmp_to_key(lable_compare))

            split = (len(self.total_data_indices)//class_num)*(self.id % class_num )
        
            self.total_data_indices = self.total_data_indices[split:-1]+self.total_data_indices[0:split]
        self.stream_coming(self.args['data_init_ratio'])
    @property
    def counter(self):
        return self.__counter
    
    @counter.setter
    def counter(self,counter):
        self.__counter = counter

    def stream_coming(self,ratio):
        """
        增加 ratio的数据
        """
        assert( ratio >=0 and ratio <= 1)
        length = len(self.train_dataset)
  
        inc = int(ratio*length)
        if inc <1:
            inc = 1
        new_pos = min(length,self.cur_data_pos+inc)
        
        logger.trace(f"{self}:data inc :{self.cur_data_pos,new_pos}")
        
        self.process_new_data(self.total_data_indices[self.cur_data_pos:new_pos])
       


    def process_new_data(self,indices):
        for indice in indices:
            _,y =self.train_dataset[indice]
            self.counter.add_element(y)
            self.cur_data_pos  += 1
            if self.args['limit'] is True:
                if (self.cur_data_pos - self.start) > self.capacity:
                    _,y = self.train_dataset[self.total_data_indices[self.start]]
                    self.counter.delete(y)
                    self.start += 1

    def train(self,model=None):
        
        logger.debug(f"{self}:traing...")
        lr = self.args['local_lr']
        train_batch_size = self.args['train_batch_size']
        momentum = self.args['momentum']
        local_epoch = self.args['local_epoch']
        dataset = Subset(self.train_dataset,
                         self.total_data_indices[self.start:self.cur_data_pos])
        if len(dataset)==0:
            return 0,0
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
           
        return total_loss,total_samples
    
    def __repr__(self) -> str:
        return f"stream-cli-{self.id}[{len(self.train_dataset)}-{len(self.test_dataset)}]-{self.start}:{self.cur_data_pos}"