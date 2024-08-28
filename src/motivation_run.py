
import argparse
import json
import os
import sys
from time import sleep
import time

from loguru import logger
import numpy as np
import yaml

from client.fedavgclient import FedAvgClient

from dataset.shakespeare import Shakespeare
from model.charlstm import CharLSTM
from server.fedavgserver import FedAvgServer
from dataset.cifar10 import CIFAR10
from model.two_cnn import TwoCNN
from server.simple import Simple
from server.streamavgserver import StreamFedAvgServer
from client.streamfedavgclient import StreamFedAvgClient
from client.streamfedprox import StreamFedProxClient
from dataset.minist import MINIST
from dataset.fminist import FMINIST
from client.powerdclient_motivation import PowerdClientMotivation
from util.utils import avg_model, load_config, set_seed
from torch.utils.data import ConcatDataset
from client.fedproxclient import FedProxClient
from client.powerdclient import PowerdClient
from server.powerdserver import PowerdServer
logger.remove()


def get_model(args):
        if args['model'] == 'charlstm':
            model= CharLSTM()
        elif args['model'] == 'twocnn':
               if args['dataset'] == 'cifar10':
                    model = TwoCNN(3,200,10)
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
     elif args['algorithm'] == 'motivation_run':
          return PowerdClientMotivation(train_dataset,test_dataset,args=args,id=id)
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
     elif args['algorithm'] == 'motivation_run':
        server = StreamFedAvgServer(args,clients)
         
     server.clients = clients
     server.model = get_model(args)
     if args['split_type'] == 'mix':
          server.global_testdataset = ConcatDataset([cli.test_dataset for cli in server.clients])
     else:
          server.global_testdataset = clients[0].test_dataset

     return server

def init_for_one_round(server,args,clients):
     server.prep_for_one_round()
     


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
    parser.add_argument("--data_inc_num",help=" data_inc_num mu",type=int)

    
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
    movtivation_file = open(f"{args['result_dir']}/{filename}_motivation1.csv",'w')
    movtivation_file.write("loss1,loss2\n")
    time_file = open(f"{args['result_dir']}/{filename}_time.csv",'w')

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
     #    init_for_one_round(server,args,clients)
          
        # 选择client 
        sampled = server.sampling_for_clients()
        loss1 = 0
        loss2 = 0
        totalcnt1= 0
        totalcnt2=0
        total_abs=0
        for client in  clients:
          client.model.load_state_dict(server.model.state_dict())
          logger.trace("数据增长:")
          #   for client in self.clients:
          totalloss1,cnt1,correct1 = client.test_on_train()
          # client.stream_coming(client.args['data_inc_ratio'])
          client.stream_coming_num(client.args['data_inc_num'])
          # if client.id == 2:
          # if client.id == 2:
          totalloss2,cnt2,correct2 = client.test_on_train()
          loss1 += totalloss1
          totalcnt1 += cnt1
          loss2+= totalloss2
          totalcnt2 += cnt2

          logger.info(f"totalloss1:{totalloss1/cnt1}")
          logger.info(f"totalloss2:{totalloss2/cnt2}")
          # movtivation_file.write(f'{loss1/totalcnt1},{loss2/totalcnt2}\n')

          total_abs += abs( loss1/totalcnt1-loss2/totalcnt2)

     
        movtivation_file.write(f'{total_abs},{loss2/totalcnt2}\n')
        movtivation_file.flush()

        clis = []
        models = []
        weights = []
        
        train_loss = []
        train_sample = []
        times = []

        for client in sampled:

            #server send model to clients
            client.model.load_state_dict(server.model.state_dict())

            #client training...
            start = time.time()
            total_loss,total_sample = client.train()
            end = time.time()

            times.append(end-start)
            train_loss.append(total_loss)
            train_sample.append(total_sample)
     

            #clients send model to servers
            models.append(client.get_model_dict())
            weights.append(client.get_train_sample_num())
 
        for t in times:
             time_file.write(f'{t},')
        time_file.write("\n")
        time_file.flush()
        #server aggregate models
        server.aggreagte(models,weights)

        if ( R + 1 )% args['test_gap'] == 0:
                loss,total,correct = server.test_on_global_testdataset()
                logger.info(f"round-{R}:acc={correct/total:<6.4}|loss={loss/total:<10f}|train_loss={sum(train_loss)/len(train_loss):<10f}")
                resultfile.write(f"{correct/total},{loss/total},{sum(train_loss)/len(train_loss)}\n")
                resultfile.flush()
                

    resultfile.close()