import json
import os
import pickle
import random
import sys
from loguru import logger
from torch.utils.data import ConcatDataset,Subset
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRO_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from util.data_utils import MEAN, STD, dirichlet
from dataset.basedataset import BaseDataset


class CIFAR100(BaseDataset):
    def __init__(
        self, 
        dataset,
        targets,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None
    ):
        super().__init__()
        self.classes = 100
        self.__dataset = dataset
    
    def __getitem__(self, index):

        return self.__dataset[index]

    def __len__(self):
        return len(self.__dataset)
    @staticmethod
    def get_split_datasets(args):
        """
        得到各个client的数据集
        return [{'train':trainset,'test':testset},...,...]
        """
        logger.info("use cifar100")
        root = PRO_DIR+'/data/cifar100'

        general_data_transform = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize(MEAN['cifar100'], STD['cifar100'])]
        )
        train_part = torchvision.datasets.CIFAR100(root, True, download=True,transform=general_data_transform)
        test_part = torchvision.datasets.CIFAR100(root, False, download=True,transform=general_data_transform)
        lablel_num = len(train_part.classes)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        ret = []
        if args['split_type'] == 'independent':
            dataset = train_part
            targets = train_targets
        elif args['split_type'] == 'mix':
            dataset = ConcatDataset([train_part,test_part])
            targets = torch.cat([train_targets, test_targets])
        else:
            raise Exception(f"split type[{args['split_type']} is unkonwed..")
    
        logger.info("do split cifar100")
        logger.info(f"dirichelet alpha={args['alpha']},train_test_ratio={args['train_test_ratio']},cli_num={args['client_num']}")
        data_indices,stats = dirichlet(targets,args['client_num'],args['alpha'],lablel_num,args)
        ret =[]
        for indice in data_indices:
            random.shuffle(indice)
            if args['split_type'] == 'mix':
                split = int(args['train_test_ratio']*len(indice))
                train,test = indice[0:split],indice[split:]
                ret.append(dict(train=Subset(dataset,train),
                                test=Subset(dataset,test)))
            else:        
                ret.append({'train':Subset(dataset,indice),'test':test_part})
        logger.info("dataset split completed")

        # self.general_data_transform = general_data_transform
        # self.general_target_transform = general_target_transform
        # self.train_data_transform = train_data_transform
        # self.train_target_transform = train_target_transform
        logger.info("cifar100 load success!")
        return ret