import torch
from torch.utils.data import Dataset, DataLoader
from torch

# data path
data_path = '/mnt/NNDL_gd/GW_data/datasets/'

# data transformation
transform = transforms.ToTensor().float64()

class GW_dataset(Dataset):

    def __init__(self, dataset_type, path=data_path, transform=transform):
        """
        dataset_type: training, validation, testing
        """
        dataset_file = path+dataset_type+'_data.pkl' 
        with open(dataset_file, 'rb') as file:
            dataset = pickle.load(file)
        self.x = dataset[0:-1]
        self.y = dataset[:,-1]
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y