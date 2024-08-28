from copy import deepcopy
import torch
import random
import numpy as np
from loguru import logger
import os

import yaml

def load_config(file):
    with open(file,encoding='utf-8') as f:
        args = yaml.load(f,Loader=yaml.FullLoader)

    return args

########
# Seed #
########
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f'[SEED] ...seed is set: {seed}!')

@torch.inference_mode()
def evaluate(model,dataloader,criterion):
    """测试模型

    Args:
        model 
        dataloader 
        criterion 

    Returns:
        总的loss 总的样本 总的正确的样本
    """    
    totalloss=0
    cnt =0
    correct = 0
    model.eval()
    for x,y in dataloader:
            pred = model(x)
            loss = criterion(pred,y)
            totalloss+= loss.item()
            cnt += len(y)
            try:
                y_pred = pred.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum().item()
            except:
                logger.info("can`t compute acc")

    return totalloss,cnt,correct

# 聚合model dict
def avg_model(model_dicts,weight):
    
    weight = np.array(weight)
    weight = weight/sum(weight)
    model_dict = deepcopy(model_dicts[0])
    for k in model_dict.keys():
        model_dict[k] = 0
        for i, model in enumerate(model_dicts):
            model_dict[k] += model[k]*weight[i]
    return model_dict

