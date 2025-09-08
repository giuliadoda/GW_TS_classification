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
nw = 4
n_epochs = 30
LR = 1e-03 # default
n_classes = 2
std = False

# class weights
if n_classes==2:
    weights = [1,1.2]
elif n_classes==3:
    weights = [1,1,5]


# model
class InceptionModule(nn.Module):

    def __init__(self,
                 in_channels=1,
                 nb_filters=32, # number of conv filters in each branch of the IM
                 kernel_sizes=(10, 20, 40),
                 bottleneck_size=32):

        super().__init__()

        ### 1: bottleneck + conv branch ###
        # bottleneck
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)

        # parallel convolutions (3 branches, note that there is no bias)
        self.conv_list = nn.ModuleList()
        for k in kernel_sizes:
            self.conv_list.append(
                nn.Conv1d(bottleneck_size, nb_filters, kernel_size=k, padding='same', bias=False)
            )

        ### 2: maxpool branch: maxpool then convolution ###
        self.mp_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)
        )

        # output layer (to be applied to concatenated branches output)
        # concatenation applies along channel dimension
        # so we have to compute how many channels we have in total:
        #   - parallel convolution contribution: len(kernel_sizes) * nb_filters channels
        #   - maxpool branch contribution: nb_filters channels
        out_channels = nb_filters * (len(kernel_sizes) + 1)
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(out_channels), # output shape is same as input
            nn.ReLU()
        )

    def forward(self, x):

        # 1: bottleneck + conv branch
        x_b = self.bottleneck(x)
        conv_outs = [ conv(x_b) for conv in self.conv_list]

        # 2: maxpool branch
        mp = self.mp_branch(x)

        # concatenate branches outputs
        conv_outs.append(mp)
        out = torch.cat(conv_outs, dim=1)  # concat on channel dim
        # output
        out = self.output_layer(out)
        return out
    
class InceptionTime(nn.Module):
    def __init__(self,
                 in_channels=1,
                 depth=6,
                 nb_filters=32,
                 kernel_sizes=(10,20,40),
                 bottleneck_size=32,
                 num_classes=3
                 ):
        super().__init__()

        self.depth = depth
        self.nb_filters = nb_filters
        self.kernel_sizes = kernel_sizes

        # first inception
        self.inception_1 = InceptionModule(in_channels=in_channels,
                                           nb_filters=nb_filters,
                                           kernel_sizes=kernel_sizes,
                                           bottleneck_size=bottleneck_size)

        # channel size after each inception
        self.out_channels = nb_filters * (len(kernel_sizes) + 1)

        # shortcut layer
        self.shortcut = nn.Sequential(
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU()
        )

        # next inception module(s)
        self.inception = InceptionModule(in_channels=self.out_channels,
                                         nb_filters=nb_filters,
                                         kernel_sizes=kernel_sizes,
                                         bottleneck_size=bottleneck_size)

        # final classifier after residual blocks
        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), # since avg pool gives one dimension more
            nn.Linear(self.out_channels, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.inception_1(x)
        for d in range(self.depth-1):
            out = self.inception(out)
            if d % 3 == 2:  # residual connection every 3 modules
                out = self.shortcut(out)
        out = self.final_layer(out)
        return out
    
# load data
train_set = dataset.GW_dataset('training', n_classes=n_classes, std=std)
valid_set = dataset.GW_dataset('validation', n_classes=n_classes, std=std)
test_set = dataset.GW_dataset('test', n_classes=n_classes, std=std)

train_DL = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)
valid_DL = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=nw)
test_DL = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=nw)

# check if a cuda GPU is available
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

# net
it_net = InceptionTime().to(device)

# loss
weights = torch.tensor(weights).float().to(device)
loss_function = nn.CrossEntropyLoss(weight=weights)

# optimizer
it_opt = optim.Adam(it_net.parameters(), lr=LR)

# metrics
train_loss_log, train_acc_log = [], []
val_loss_log, val_acc_log = [], []

acc_metric = MulticlassAccuracy(num_classes=n_classes, average=None).to(device)

# training/validation loop
for epoch in range(n_epochs):
    print(f"\n-----------------\nEpoch {epoch+1}/{n_epochs}\n-----------------")

    it_net.train()
    epoch_losses = []
    acc_metric.reset()

    # training
    for sample in train_DL:
        xb = sample[0].float().to(device)
        yb = sample[1].long().to(device)

        out = it_net(xb)

        loss = loss_function(out, yb)

        it_opt.zero_grad()
        loss.backward()

        it_opt.step()

        epoch_losses.append(loss.item())
        acc_metric.update(out, yb)
    
    avg_loss = np.mean(epoch_losses)
    class_acc = acc_metric.compute().detach().cpu().numpy()

    train_loss_log.append(avg_loss)
    train_acc_log.append({c: class_acc[c] for c in range(n_classes)})

    print(f"Train loss: {avg_loss:.4f}")
    for c, acc in enumerate(class_acc):
        print(f" Class {c} accuracy: {acc:.4f}")

    # validation
    val_loss = []
    it_net.eval()
    acc_metric.reset()
    with torch.no_grad():
        for sample in valid_DL:
            xb = sample[0].float().to(device)
            yb = sample[1].long().to(device)

            out = it_net(xb)

            l = loss_function(out, yb)

            val_loss.append(l.item())
            acc_metric.update(out, yb)

    avg_val_loss = np.mean(val_loss)
    class_val_acc = acc_metric.compute().detach().cpu().numpy()

    val_loss_log.append(avg_val_loss)
    val_acc_log.append({c: class_val_acc[c] for c in range(n_classes)})

    print(f"Validation loss: {avg_val_loss:.4f}")
    for c, acc in enumerate(class_val_acc):
        print(f" Class {c} accuracy: {acc:.4f}")

plots.plot_loss_acc('IT', train_loss_log, val_loss_log, train_acc_log, val_acc_log, n_classes=n_classes)