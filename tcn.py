import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader

from pytorch_tcn import TCN

from codecarbon import EmissionsTracker

import time

from utils import dataset, plots, metrics

model_name = 'TCN'

seed = 0
batch_size = 32
nw = 4
n_epochs = 30
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
train_set = dataset.GW_dataset('training', n_classes=n_classes, std = std)
valid_set = dataset.GW_dataset('validation', n_classes=n_classes, std = std)
test_set = dataset.GW_dataset('test', n_classes=n_classes, std = std)

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
train_loss_log, train_acc_log, train_overall_acc_log = [], [], []
val_loss_log, val_acc_log, val_overall_acc_log = [], [], []

acc_metric = MulticlassAccuracy(num_classes=n_classes, average=None).to(device)
micro_acc_metric = MulticlassAccuracy(num_classes=n_classes, average='micro').to(device)

# training/validation loop
# start timer
start_time = time.time()
for epoch in range(n_epochs):
    print(f"\n-----------------\nEpoch {epoch+1}/{n_epochs}\n-----------------")

    tcn_net.train()
    epoch_losses = []
    acc_metric.reset()
    micro_acc_metric.reset()

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
        micro_acc_metric.update(out, yb)
    
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
    tcn_net.eval()
    acc_metric.reset()
    micro_acc_metric.reset()
    with torch.no_grad():
        for sample in valid_DL:
            xb = sample[0].float().to(device)
            yb = sample[1].long().to(device)

            out = tcn_net(xb)

            l = loss_function(out, yb)

            val_loss.append(l.item())
            acc_metric.update(out, yb)
            micro_acc_metric.update(out, yb)

        
    avg_val_loss = np.mean(val_loss)
    class_val_acc = acc_metric.compute().detach().cpu().numpy()
    overall_val_acc = micro_acc_metric.compute().item()

    val_loss_log.append(avg_val_loss)
    val_acc_log.append({c: class_val_acc[c] for c in range(n_classes)})
    val_overall_acc_log.append(overall_val_acc)

    print(f"Validation loss: {avg_val_loss:.4f}")
    for c, acc in enumerate(class_val_acc):
        print(f" Class {c} accuracy: {acc:.4f}")

# end timer
end_time = time.time()
total_time = end_time - start_time
avg_epoch_time = total_time / n_epochs

# save times to file
time_file = './model_info/' + model_name+'_times.txt'
with open(time_file, 'w') as f:
    f.write(str(total_time))
    f.write('\n')
    f.write(str(avg_epoch_time))

plots.plot_loss_acc(model_name, train_loss_log, val_loss_log, train_acc_log, val_acc_log, train_overall_acc_log, val_overall_acc_log, n_classes=n_classes)   

plots.plot_model_params(tcn_net, save_path='./plots/'+model_name+"_params_hist.png")

# save trained model
model_path = './models/'+ model_name + "_trained.torch"
torch.save(tcn_net.state_dict(), model_path)

# save training emissions
emissions = tracker.stop()
print(f"\nTotal CO2 emissions: {float(emissions):.6f} kg")
co2_file = './model_info/' + model_name+'_CO2.txt'
with open(co2_file, 'w') as file:
    file.write(str(emissions))

# analyze activations
plots.plot_model_activations(tcn_net, test_DL, device, max_batches=1, save_path='./plots/'+model_name+'_act.png')

# test
print('\n-- Testing  \n')
tcn_net.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for sample in test_DL:
        xb = sample[0].float().to(device)
        yb = sample[1].long().to(device)

        out = tcn_net(xb)

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
roc_file = './model_info/'+model_name+"_ROC.txt"
with open(roc_file, "w") as f:
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