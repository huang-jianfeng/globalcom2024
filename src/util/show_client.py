import sys
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
from utils import load_config, set_seed
from pathlib import Path
from torch.utils.data import ConcatDataset
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRO_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from dataset.cifar10 import CIFAR10
from collections import defaultdict
def inspect_heterogeous():
    args = load_config(r"D:\Users\hjf\pythoncode\federated\config.yaml")
    set_seed(args['rand_seed'])
    split_datasets = CIFAR10.get_split_datasets(args)
    test_cnt = 0
    train_cnt = 0
    client_num = len(split_datasets)
    statics = defaultdict()

    class_num=10
    label_distribution = [[] for _ in range(class_num)]
    for i,dataset in enumerate(split_datasets):
        for x,y in dataset['train']:
            label_distribution[y].append(i)


    logger.info(f"DATASET:{args['dataset']}")
    logger.info(f"total:{client_num} clients| {train_cnt} train_samples, {test_cnt} test_samples")

    plt.rcParams.update({'font.size': 21,'font.family':'Arial'})
    plt.figure()

    plt.hist(label_distribution, stacked=True,
                bins=np.arange(-0.5, client_num + 1.5, 1),
                label=[i for i in range(class_num)], rwidth=0.5,density=False)
    plt.xticks(np.arange(client_num), ["%d" %
                                        c_id for c_id in range(client_num)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    # plt.title("Display Label Distribution on Different Clients")
    plt.legend(loc='upper right',prop = {'size':9})
    # plt.savefig(f"alpha_{args['alpha']}_train.png")
    plt.tight_layout()
    plt.show()
def show_lable_distribution(dataset,file):
    length = 10
    targets = []
    for _,y in dataset:
        targets.append(y)
    
    plt.figure(figsize=(12,8))
    plt.hist(targets, stacked=True,
                bins=np.arange(-0.5, length + 1.5, 1),
                label=[i for i in range(length)], rwidth=0.5,density=False)
    plt.xticks(np.arange(length), ["Client %d" %
                                        c_id for c_id in range(length)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    # plt.title("Display Label Distribution on Different Clients")
    plt.legend()
    # plt.savefig(file)
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    args = load_config(r"D:\OneDrive - smail.swufe.edu.cn\federated\config.yaml")
    set_seed(args['rand_seed'])
    # split_datasets = CIFAR10.get_split_datasets(args)
    # test = split_datasets[0]['test']
    # train = split_datasets[0]['train']
    inspect_heterogeous()
    file = f"./distribution/test_global_distr.png"
    # show_lable_distribution(train,file)