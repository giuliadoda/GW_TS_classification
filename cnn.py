import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader

from utils import dataset, plots, metrics


seed = 0
batch_size = 32
LR = 1e-04
nw = 4
n_epochs = 50
n_classes = 3  
std = True

# class weights
if n_classes==2:
    weights = [1,1.2]
elif n_classes==3:
    weights = [1,1,5]


# model
class CNN(nn.Module):

    def __init__(self):
        super().__init__() 

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, 
                      out_channels=32, 
                      kernel_size=4, 
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride = 1), # default stride: kernel_size 
            nn.Dropout(0.1),
            nn.ReLU(True))
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=4, 
                      stride=2),
            nn.MaxPool1d(kernel_size=2, stride = 1),
            nn.Dropout(0.1),
            nn.ReLU(True))
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=8, 
                      stride=2),
            nn.MaxPool1d(kernel_size=4, stride = 1),
            nn.Dropout(0.25),
            nn.ReLU(True))

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=128, 
                      out_channels=256, 
                      kernel_size=16, 
                      stride=4),
            nn.MaxPool1d(kernel_size=4, stride = 1),
            nn.Dropout(0.25),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.1)
        )
        
        self.lin = nn.Sequential(
            nn.Linear(in_features=256, 
                      out_features=64),
            nn.Linear(in_features=64, 
                      out_features=n_classes))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # apply conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.squeeze(-1) # remove last dim that is 1
        # apply linear layer
        out = self.lin(x)
        # apply softmax
        out = self.softmax(out)
        return out 
    

# load data
train_set = dataset.GW_dataset('training', n_classes=n_classes, std=std)
valid_set = dataset.GW_dataset('validation', n_classes=n_classes, std=std)
test_set = dataset.GW_dataset('test', n_classes=n_classes, std=std)

train_DL = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)
valid_DL = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=nw)
test_DL = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=nw)

# check if GPU is available
if torch.cuda.is_available():
    print('GPU availble')
    # define the device
    device = torch.device("cuda")
else:
    print('GPU not availble')
    device = torch.device("cpu")

print(f"SELECTED DEVICE: {device}")

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

cnn_net = CNN().to(device)

# loss
weights = torch.tensor(weights).float().to(device)
loss_function = nn.CrossEntropyLoss(weight=weights)

# optimizer
cnn_opt = optim.Adam(cnn_net.parameters(), lr=LR)

# metrics
train_loss_log, train_acc_log = [], []
val_loss_log, val_acc_log = [], []

acc_metric = MulticlassAccuracy(num_classes=n_classes, average=None).to(device)

# training/validation loop
for epoch in range(n_epochs):
    print(f"\n-----------------\nEpoch {epoch+1}/{n_epochs}\n-----------------")

    cnn_net.train()
    epoch_losses = []
    acc_metric.reset()

    # training
    for sample in train_DL:
        # data to device
        # sample shape: (B, 1, signal_length)
        xb = sample[0].float().to(device)
        yb = sample[1].long().to(device)

        # forward pass
        out = cnn_net(xb)

        # loss
        loss = loss_function(out, yb)

        # backpropagation
        cnn_opt.zero_grad()
        loss.backward()

        # weight update
        cnn_opt.step()

        # log loss and accuracy
        epoch_losses.append(loss.item())
        acc_metric.update(out, yb)

    # average loss and accuracy
    avg_loss = np.mean(epoch_losses)
    class_acc = acc_metric.compute().detach().cpu().numpy()


    train_loss_log.append(avg_loss)
    train_acc_log.append({c: class_acc[c] for c in range(n_classes)})

    print(f"Train loss: {avg_loss:.4f}")
    for c, acc in enumerate(class_acc):
        print(f" Class {c} accuracy: {acc:.4f}")

    # validation
    val_loss = []
    cnn_net.eval()
    acc_metric.reset()
    with torch.no_grad():
        for sample in valid_DL:
            # data to device
            xb = sample[0].float().to(device)
            yb = sample[1].long().to(device)

            # forward 
            out = cnn_net(xb)

            # loss
            l = loss_function(out, yb)

            # save
            val_loss.append(l.item())
            acc_metric.update(out, yb)
        
    avg_val_loss = np.mean(val_loss)
    class_val_acc = acc_metric.compute().detach().cpu().numpy()

    val_loss_log.append(avg_val_loss)
    val_acc_log.append({c: class_val_acc[c] for c in range(n_classes)})

    print(f"Validation loss: {avg_val_loss:.4f}")
    for c, acc in enumerate(class_val_acc):
        print(f" Class {c} accuracy: {acc:.4f}")


plots.plot_loss_acc('CNN', train_loss_log, val_loss_log, train_acc_log, val_acc_log)