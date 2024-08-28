
import argparse
import json
import os
import sys
from time import sleep

from loguru import logger
import numpy as np
from sympy import Subs
import yaml

from client.fedavgclient import FedAvgClient
from torch.utils.data import Subset
from dataset.shakespeare import Shakespeare
from model.charlstm import CharLSTM
from server.fedavgserver import FedAvgServer
from dataset.cifar10 import CIFAR10
from dataset.cifar100 import CIFAR100
from model.two_cnn import TwoCNN
from server.simple import Simple
from server.streamavgserver import StreamFedAvgServer
from client.streamfedavgclient import StreamFedAvgClient
from client.streamfedprox import StreamFedProxClient
from dataset.minist import MINIST
from dataset.fminist import FMINIST
from model.resnet import Resnet
from util.utils import avg_model, load_config, set_seed
from torch.utils.data import ConcatDataset
from client.fedproxclient import FedProxClient
from client.powerdclient import PowerdClient
from server.powerdserver import PowerdServer
logger.remove()


def get_model(args):
        class_num = 10
        if args['dataset'] =='cifar100':
             class_num  = 100
        if args['model'] == 'charlstm':
            model= CharLSTM()
        elif args['model'] == "resnet":
             model =Resnet(class_num)
        elif args['model'] == 'twocnn':
               if args['dataset'] == 'cifar10':
                    model = TwoCNN(3,200,10)
               elif args['dataset'] == 'cifar100':
                    model = TwoCNN(3,200,100)
               elif args['dataset'] == 'fmnist' or args['dataset'] == 'mnist':
                    model = TwoCNN(1,200,10)
               
     
        else:
            raise Exception(f"unsupported model{args['model']}")
        return model

def init_client(args):

     split_datasets = []
     clients = []

     if args['dataset']=='shakespeare':
        split_datasets = Shakespeare.get_split_datasets(args)
     elif args['dataset'] == 'cifar10':
         split_datasets = CIFAR10.get_split_datasets(args)
     elif args['dataset'] == 'mnist':
         split_datasets = MINIST.get_split_datasets(args)
     elif args['dataset'] == 'fmnist':
         split_datasets = FMINIST.get_split_datasets(args)
     elif args['dataset'] =='cifar100':
          split_datasets = CIFAR100.get_split_datasets(args)

     else:
         raise Exception(f"unsupported dataset{args['dataset']}")

     test_cnt = 0
     train_cnt = 0
     for i,dataset in enumerate(split_datasets):
          cli = get_client(args,dataset['train'],dataset['test'],id=i)
          cli.model = get_model(args)
          cli.id = i
          cli.args = args
          train_cnt += len(cli.train_dataset)
          test_cnt += len(cli.test_dataset)
          clients.append(cli)
     logger.info(f"DATASET:{args['dataset']}")
     logger.info(f"total:{len(clients)} clients| {train_cnt} train_samples, {test_cnt} test_samples")
     logger.info(clients)
          
     return clients

def get_client(args,train_dataset,test_dataset,id):
     if args['algorithm'] == 'fedavg' or args['algorithm'] == 'simple':
          return FedAvgClient(train_dataset,test_dataset,args=args,id=id)
     elif args['algorithm'] == 'streamfedavg':
          return StreamFedAvgClient(train_dataset,test_dataset,args=args,id=id)
     elif args['algorithm'] == 'fedproxstream':
          return StreamFedProxClient(train_dataset,test_dataset,args=args,id=id)
     elif args['algorithm'] == 'fedprox':
          return FedProxClient(train_dataset,test_dataset,args=args,id=id)
     elif args['algorithm'] == 'powerd':
          return PowerdClient(train_dataset,test_dataset,args=args,id=id)
     else:
          raise Exception(f"unkonw algorithm[{args['algorithm']}]")
     
def init_server(args, clients):
     if args['algorithm'] == 'fedavg':
        server = FedAvgServer(args,clients)
     elif args['algorithm'] == 'streamfedavg':
         server = StreamFedAvgServer(args,clients)
     elif args['algorithm'] == 'fedproxstream':
         server = StreamFedAvgServer(args,clients)
     elif args['algorithm'] == 'fedprox':
         server = FedAvgServer(args,clients)
     elif args['algorithm'] == 'powerd':
         server = PowerdServer(args,clients)
     elif args['algorithm'] == 'simple':
          server = Simple(args,clients)
     server.clients = clients
     server.model = get_model(args)
     if args['split_type'] == 'mix':
          server.global_testdataset = ConcatDataset([cli.test_dataset for cli in server.clients])
     else:
          server.global_testdataset = clients[0].test_dataset

     return server

def init_for_one_round(server,args,clients):
     server.prep_for_one_round()

def get_acc_cnt(cli,class_num):
     cnt = class_num*[0]
     dataset = Subset(cli.train_dataset,
                         cli.total_data_indices[cli.start:cli.cur_data_pos])
     for _,y in dataset:
          cnt[y]+=1
     return cnt
def compute_distance(sampled:StreamFedAvgClient,args):
     class_num = 10
     if args['dataset'] == 'cifar100':
          class_num = 100
     
     distribution = np.zeros(class_num)
     for cli in sampled:
          distribution += np.array(get_acc_cnt(cli,class_num))
     distribution = distribution/sum(distribution)
     target_distribution = np.full([class_num],0.1)
     x_norm = np.linalg.norm(distribution-target_distribution,ord=2)
     return x_norm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    #******#
    #****加载配置
    parser.add_argument("--config_file",help="the config file",type=str,
                        default=r"D:\Users\hjf\pythoncode\federated\config.yaml")

    parser.add_argument('--client_num',help="client number",type=int)
    parser.add_argument('--alpha',help="dirichlet param",type=float)
    parser.add_argument('--rounds',help="number of federated traing communication ",type=int)
    parser.add_argument('--seq_num',help=" for simple server experiment",type=int)
    parser.add_argument("--sample",help="the method of sampling clients",type=str)
    parser.add_argument("--agg",help="the method of avg client model ",type=str)
    parser.add_argument("--unbalance",help=" unbalance global test ",type=float)
    parser.add_argument("--counter_type",help="  cmsketch ,array ",type=str)
    parser.add_argument("--rand_seed",help=" random seed ",type=int)
    parser.add_argument("--resort",help=" resort data samples by label ",type=bool)
    parser.add_argument("--algorithm",help='fedavg,fedprox....',type=str)
    
    parser.add_argument("--mu",help=" fedprox mu",type=float)
    parser.add_argument("--local_lr",help=" local_lr",type=float)
    parser.add_argument("--sampling_ratio",help=" sampling_ratio",type=float)
    parser.add_argument("--dataset",help=" dataset",type=str)
    parser.add_argument("--local_epoch",help=" dataset",type=int)
    parser.add_argument("--capacity",help=" capacity",type=int)


    command = parser.parse_args()
    args = load_config(command.config_file)
    logger.add(sys.stdout,level=args['loglevel'])

    for k in vars(command):
         if getattr(command,k) is not None:
              args[k] = getattr(command,k)
    logger.info(args)


    set_seed(args['rand_seed'])
    filename =""
    for k in args['show_args']:
         filename += f"{k}_{args[k]}_"
    filename = filename[0:-1]
    # filename = f"{args['algorithm']}_ratio_{args['sampling_ratio']}_{args['dataset']}_Dirich_{args['alpha']}"
    if not os.path.exists(args['result_dir']):
         os.makedirs(args['result_dir'])

    with open(f"{args['result_dir']}/{filename}_config.json",'w') as f:
         json.dump(args,f)
         
    resultfile = open(f"{args['result_dir']}/{filename}.csv",'w')
    resultfile.write("test_correct,test_loss,train_loss\n")

    distance_file = open(f"{args['result_dir']}/{filename}_dist.csv",'w')
    distance_file.write("distancel2\n")

    freq = open(f"{args['result_dir']}/{filename}_freq.csv",'w')
#     freq.write("distancel2\n")
    #******初始化client server
    
    clients  = init_client(args)

    server = init_server(args,clients)
#     server.clients = clients

    #*****训练
    Rounds = args['rounds']

    #***do something
    server.before_fed_training()
    for R in range(Rounds):
        
        server.round = R
        # 每轮训练前
        init_for_one_round(server,args,clients)

        # 选择client 
        sampled = server.sampling_for_clients()
        freq.write(f"{[cli.id for cli in sampled]}\n")
        clis = []
        models = []
        weights = []
        
        train_loss = []
        train_sample = []
        distance = compute_distance(sampled,args)
        distance_file.write(f'{distance}\n')
        distance_file.flush()
        for client in sampled:

            #server send model to clients
            client.model.load_state_dict(server.model.state_dict())

            #client training...
            total_loss,total_sample = client.train()

            train_loss.append(total_loss)
            train_sample.append(total_sample)
     

            #clients send model to servers
            models.append(client.get_model_dict())
            weights.append(client.get_train_sample_num())

        #server aggregate models
        server.aggreagte(models,weights)

        if ( R + 1 )% args['test_gap'] == 0:
                loss,total,correct = server.test_on_global_testdataset()
                logger.info(f"round-{R}:acc={correct/total:<6.4}|loss={loss/total:<10f}|train_loss={sum(train_loss)/len(train_loss):<10f}")
                resultfile.write(f"{correct/total},{loss/total},{sum(train_loss)/len(train_loss)}\n")
                resultfile.flush()

    resultfile.close()