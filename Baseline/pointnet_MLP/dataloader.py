from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from model import pointnetbaseline
import numpy as np  
class TreeData(Dataset):
    def __init__(self, data_folder='./Data_pointnet.pickle'):
        self.data_folder = data_folder
        self.boxset = pickle.load(open(data_folder, "rb" ))
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        node_xyabtheta = self.boxset[idx]
        return np.moveaxis(node_xyabtheta,1,0) 
    def __len__(self):
        return len(self.boxset)



