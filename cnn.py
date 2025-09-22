import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader

from codecarbon import EmissionsTracker

import time

from utils import dataset, plots, metrics

model_name = 'CNN'

seed = 0
batch_size = 32
LR = 1e-04
nw = 4
# n_epochs = 50
n_epochs = 3
n_classes = 2
std = False

# class weights
if n_classes==2:
    weights = [1,1.2]
elif n_classes==3:
    weights = [1,1,5]

# emission tracker
tracker = EmissionsTracker(project_name="IT_test")
tracker.start()

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

train_DL = DataLoader(train_set, batch_size=batch_size, num_workers=nw)
valid_DL = DataLoader(valid_set, batch_size=batch_size, num_workers=nw)
test_DL = DataLoader(test_set, batch_size=batch_size, num_workers=nw)

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
train_loss_log, train_acc_log, train_overall_acc_log = [], [], []
val_loss_log, val_acc_log = [], []

acc_metric = MulticlassAccuracy(num_classes=n_classes, average=None).to(device)
micro_acc_metric = MulticlassAccuracy(num_classes=n_classes, average='micro').to(device)

# training/validation loop
# start timer
start_time = time.time()
for epoch in range(n_epochs):
    print(f"\n-----------------\nEpoch {epoch+1}/{n_epochs}\n-----------------")
    
    cnn_net.train()
    epoch_losses = []
    acc_metric.reset()
    micro_acc_metric.reset()

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
        micro_acc_metric.update(out, yb)

    # average loss and accuracy
    avg_loss = np.mean(epoch_losses)
    class_acc = acc_metric.compute().detach().cpu().numpy()
    overall_acc = micro_acc_metric.compute().item()

    train_loss_log.append(avg_loss)
    train_acc_log.append({c: class_acc[c] for c in range(n_classes)})
    train_overall_acc_log.append(overall_acc)

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

# end timer
end_time = time.time()
total_time = end_time - start_time
avg_epoch_time = total_time / n_epochs

print(f"\nTotal training time: {total_time:.2f} seconds")
print(f"Average time per epoch: {avg_epoch_time:.2f} seconds")

# save times to file
with open(model_name+'_times.txt', 'w') as f:
    f.write(str(total_time))
    f.write(str(avg_epoch_time))

plots.plot_loss_acc(model_name, train_loss_log, val_loss_log, train_acc_log, val_acc_log, n_classes=n_classes)

plots.plot_model_params(cnn_net, save_path=model_name+"_params_hist.png")

# save trained model
model_path = model_name + "_trained.pth"
torch.save(cnn_net.state_dict(), model_path)

# save training emissions
emissions = tracker.stop()
print(f"\nTotal CO2 emissions: {float(emissions):.6f} kg")

with open(model_name+'_CO2.txt', 'w') as file:
    file.write(str(emissions))

# analyze activations
plots.plot_model_activations(cnn_net, test_DL, device, max_batches=1)

# test
print('\n-- Testing  \n')
cnn_net.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for sample in test_DL:
        xb = sample[0].float().to(device)
        yb = sample[1].long().to(device)

        out = cnn_net(xb)

        preds = torch.argmax(out, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())
        all_probs.append(out.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_probs = np.concatenate(all_probs)

# ROC
classes = [f"Class {i}" for i in range(n_classes)]
fpr, tpr, roc_auc = metrics.compute_roc(all_probs, all_labels, num_classes=n_classes)

# plot ROC
plots.plot_ROC(model_name, fpr, tpr, roc_auc)

# save AUC values + FPR and TPR points
with open(model_name+"_ROC.txt", "w") as f:
    for i, cls in enumerate(classes):
        f.write(f"{cls} AUC: {roc_auc[i]:.4f}\n")
        f.write(f"{cls} FPR: {','.join([f'{x:.6f}' for x in fpr[i]])}\n")
        f.write(f"{cls} TPR: {','.join([f'{x:.6f}' for x in tpr[i]])}\n\n")
    
    # micro-average
    f.write(f"Micro-average AUC: {roc_auc['micro']:.4f}\n")
    f.write(f"Micro-average FPR: {','.join([f'{x:.6f}' for x in fpr['micro']])}\n")
    f.write(f"Micro-average TPR: {','.join([f'{x:.6f}' for x in tpr['micro']])}\n")

# confusion matrix
plots.plot_CM(model_name, all_preds, all_labels)

# plot predicted signal probabilities
plots.plot_PS(model_name, all_probs, all_labels, n_classes=n_classes)