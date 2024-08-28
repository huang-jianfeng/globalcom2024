from collections import Counter
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple,List,Dict

def dirichlet(
    targets, client_num: int, alpha: float,label_num:int,args
    ) -> Tuple[List[List[int]], Dict]:
    
    stats = {}
    targets_numpy = np.array(targets, dtype=np.int32)
    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0] for i in range(label_num)
    ]


    new_indices =[]
    ratio = args['unbalance']
    cnt = 0
    for data_idx in data_idx_for_each_label:
        new_indices.append(data_idx[0:int((len(data_idx)*(ratio**cnt)))])
        cnt += 1
    # self.global_testdataset = Subset(self.global_testdataset,np.concatenate(new_indices))
    data_idx_for_each_label = new_indices
    data_indices = [[] for _ in range(client_num)]
    for k in range(label_num):
        np.random.shuffle(data_idx_for_each_label[k])
        distrib = np.random.dirichlet(np.repeat(alpha, client_num))
        distrib = distrib / distrib.sum()
        distrib = (np.cumsum(distrib) * len(data_idx_for_each_label[k])).astype(
            int
        )[:-1]
        data_indices = [
            np.concatenate((idx_j, idx.tolist())).astype(np.int64)
            for idx_j, idx in zip(
                data_indices, np.split(data_idx_for_each_label[k], distrib)
            )
        ]

    for i in range(client_num):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(targets_numpy[data_indices[i]])
        stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    data_indices

    return data_indices, stats


MEAN = {
    "mnist": [0.1307],
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4865, 0.4409],
    "emnist": [0.1736],
    "fmnist": [0.286],
    "femnist": [0.9637],
    "medmnist": [124.9587],
    "medmnistA": [118.7546],
    "medmnistC": [124.424],
    "covid19": [125.0866, 125.1043, 125.1088],
    "celeba": [128.7247, 108.0617, 97.2517],
    "synthetic": [0.0],
    "svhn": [0.4377, 0.4438, 0.4728],
    "tiny_imagenet": [122.5119, 114.2915, 101.388],
    "cinic10": [0.47889522, 0.47227842, 0.43047404],
    "domain": [0.485, 0.456, 0.406],
}

STD = {
    "mnist": [0.3015],
    "cifar10": [0.2023, 0.1994, 0.201],
    "cifar100": [0.2009, 0.1984, 0.2023],
    "emnist": [0.3248],
    "fmnist": [0.3205],
    "femnist": [0.155],
    "medmnist": [57.5856],
    "medmnistA": [62.3489],
    "medmnistC": [58.8092],
    "covid19": [56.6888, 56.6933, 56.6979],
    "celeba": [67.6496, 62.2519, 61.163],
    "synthetic": [1.0],
    "svhn": [0.1201, 0.1231, 0.1052],
    "tiny_imagenet": [58.7048, 57.7551, 57.6717],
    "cinic10": [0.24205776, 0.23828046, 0.25874835],
    "domain": [0.229, 0.224, 0.225],
}
