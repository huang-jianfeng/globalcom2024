from torch.utils.data import Dataset
from abc import *

class BaseDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
    
