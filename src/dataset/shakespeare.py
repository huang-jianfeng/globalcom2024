import json
from loguru import logger
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRO_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from torch.utils.data import Dataset
import torch
from dataset.basedataset import BaseDataset
from util.language_utils import ALL_LETTERS, word_to_indices
class Shakespeare(BaseDataset):

    def __init__(self,data,target):
        t1 = []
        t2 = []
        for x,y in zip(data,target):
            vec = torch.IntTensor(word_to_indices(x))
            t1.append(vec)
            t2.append(ALL_LETTERS.find(y))
        self.__data = t1
        self.__target = t2

    def __getitem__(self, idx):  # 默认是item，但常改为idx，是index的缩写
        """
        return x,y
        """
      
        return self.__data[idx],self.__target[idx]
    
    def __len__(self):
        return len(self.__target)
    @staticmethod
    def get_split_info():
        return {}
    @staticmethod
    def get_split_datasets(args):
        """
        得到各个client的数据集
        """
        with open(os.path.join(PRO_DIR +r'\data\shakespeare\data\test\all_data_niid_1_keep_0_test_9.json')) as f:
            testdata = json.loads(f.readline())
        with open(os.path.join(PRO_DIR,r'data\shakespeare\data\train\all_data_niid_1_keep_0_train_9.json')) as f:
            traindata = json.loads(f.readline())
        test = testdata['user_data']
        train = traindata['user_data']
        ret=[]
        total_train = 0
        total_test = 0

        for k,v in test.items():
            r = dict(train=Shakespeare(data=train[k]['x'],target=train[k]['y'])
                        ,test=Shakespeare(data=v['x'],target=v['y']))
            ret.append(r)
            total_train += len(r['train'])
            total_test += len(r['test'])
        logger.info(f"shakespeare:train-{total_train},test-{total_test},total_users-{len(ret)}")
        return ret