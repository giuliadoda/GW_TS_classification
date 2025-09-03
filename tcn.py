import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader

from pytorch_tcn import TCN

from utils import dataset, plots, metrics


seed = 0
batch_size = 32
nw = 4
n_epochs = 50
n_classes = 3

# class weights
weights = [1,1,5]


# model
class TempConvNet(nn.Module):
  def __init__(self,
               n_layers=8,          # number of dilated convolutional layers
               n_filters=32,        # number of filters in each conv layer
               filter_size=16,      # kernel size
               num_classes=n_classes        # number of classes to perform classification
               ):
    super().__init__()

    self.n_layers = n_layers
    self.n_filters = n_filters
    self.kernel_size = filter_size

    channels = np.full(n_layers,n_filters)

    self.tcn = TCN(
        num_inputs=1,
        num_channels=channels,
        kernel_size=filter_size,
        dropout=0.1
    )

    self.final_layer = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(n_filters, num_classes),
        nn.Softmax(dim=1)
    )

  def forward(self, x):
    out = self.tcn(x)
    out = out[:,:,-1]
    out = self.final_layer(out)
    return out

  def compute_R(self):
    d_tot = 0
    for d in range(self.n_layers):
      d_tot += 2**(d+1)
    R = 1 + 2 * d_tot * (self.kernel_size-1)
    return R


# load data
train_set = dataset.GW_dataset('training')
valid_set = dataset.GW_dataset('validation')
test_set = dataset.GW_dataset('test')

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

tcn_net = TempConvNet().to(device)

# loss
weights = torch.tensor(weights).float().to(device)
loss_function = nn.CrossEntropyLoss(weight=weights)

# optimizer
tcn_opt = optim.Adam(tcn_net.parameters())

# metrics
train_loss_log, train_acc_log = [], []
val_loss_log, val_acc_log = [], []

acc_metric = MulticlassAccuracy(num_classes=n_classes, average=None).to(device)

# training/validation loop
for epoch in range(n_epochs):
    print(f"\n-----------------\nEpoch {epoch+1}/{n_epochs}\n-----------------")

    tcn_net.train()
    epoch_losses = []
    acc_metric.reset()

    # training
    for sample in train_DL:
        xb = sample[0].float().to(device)
        yb = sample[1].long().to(device)

        out = tcn_net(xb)

        loss = loss_function(out, yb)

        tcn_opt.zero_grad()
        loss.backward()

        tcn_opt.step()

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
    tcn_net.eval()
    acc_metric.reset()
    with torch.no_grad():
        for sample in valid_DL:
            xb = sample[0].float().to(device)
            yb = sample[1].long().to(device)

            out = tcn_net(xb)

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

plots.plot_loss_acc('TCN', train_loss_log, val_loss_log, train_acc_log, val_acc_log)    