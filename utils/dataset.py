import torch
from torch.utils.data import Dataset
import pickle


# data path
data_path = '/mnt/POD/NNDL_gd/GW_data/datasets/'

class GW_dataset(Dataset):

    def __init__(self, dataset_type, path=data_path, n_classes=3, std=False):
        """
        dataset_type: training, validation, testing
        """
        if n_classes == 3 and std:
            fin_str = '_data_std.pkl'
        elif n_classes == 3 and not std:
            fin_str = '_data.pkl'
        elif n_classes == 2:
            fin_str = '_data_01.pkl'
        dataset_file = path+dataset_type+fin_str
        with open(dataset_file, 'rb') as file:
            dataset = pickle.load(file)
        dataset = dataset.reshape(dataset.shape[0], 1, -1)
        self.x = dataset[:, :, 0:-1]
        self.y = dataset[:, 0, -1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        x = torch.tensor(x, dtype=torch.float64)
        return x, y