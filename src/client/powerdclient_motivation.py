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
from util.utils import evaluate

from torch.utils.data import DataLoader,Subset
from client.streamfedavgclient import StreamFedAvgClient
class PowerdClientMotivation(StreamFedAvgClient):
    def __init__(self, train_dataset=None, test_dataset=None,
                  model=None, args=None,id=None) -> None:
        
        super().__init__(train_dataset, test_dataset, model, args,id)
        
        logger.trace("streamfedavgclient init.")
        self.lastloss = float('inf')
        self.cur_data_pos = 0
        self.total_data_indices = [i for i in range(len(self.train_dataset))]
        self.id = id

        self.__counter = get_counter(self.args)
        self.start = 0
        self.capacity = self.args['capacity']

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

            split = (len(self.total_data_indices)//10)*(self.id % 10 )
        
            self.total_data_indices = self.total_data_indices[split:-1]+self.total_data_indices[0:split]
        data_indices =[[] for _ in range(10)]
        for index in self.total_data_indices:
            data_indices[self.train_dataset[index][1]].append(index)
        sorted_data_indices=[]
        start= [0 for _ in range(10)]
        cnt = 0
        while cnt < 10:
            for i in range(10):
                if len(data_indices[i]) > start[i]:
                    sorted_data_indices.extend(data_indices[i][start[i]:start[i]+20])
                    start[i] += 20
                    if len(data_indices[i]) <= start[i]:
                        cnt+=1
                elif len(data_indices[i])<=start[i]:
                        cnt +=1
                    
        self.stream_coming(self.args['data_init_ratio'] )
    def stream_coming_num(self,num):
        """
        增加 num个数据
        """
      
        length = len(self.train_dataset)
  
        inc = num
        if inc <1:
            inc = 1
        new_pos = min(length,self.cur_data_pos+inc)
        
        logger.trace(f"{self}:data inc num:{self.cur_data_pos,new_pos}")
        
        self.process_new_data(self.total_data_indices[self.cur_data_pos:new_pos])
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