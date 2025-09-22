import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


path = '/mnt/POD/NNDL_gd/code/GW_TS_classification/plots/'
classes = ['Noise', 'Signal']


# plot signals
def plot_TS(ts):

    time = np.linspace(0,1,num=2048)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(28,16))

    for i in range(3):
        ax[i].grid(alpha=0.4)
        ax[i].plot(time, ts[i])
        ax[i].set_title(classes[i]+' example')
        ax[i].set_xlabel('time (s)')
        ax[i].set_ylabel('strain (a.u.)')

    fig.savefig('./plots/timeseries.png', bbox_inches='tight', pad_inches=0.6, dpi=400)
    plt.show()
    plt.close(fig)


# plot losses and accuracies (training and validation)
def plot_loss_acc(model, train_loss_log, val_loss_log, train_acc_log, val_acc_log,
                  train_overall_acc_log=None, val_overall_acc_log=None, n_classes=2):

    epochs = np.arange(1, len(train_loss_log) + 1)

    # --- Losses ---
    fig, ax = plt.subplots(figsize=(10,6))
    ax.grid(alpha=0.4)
    ax.set_title(model + ' train and validation losses')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.plot(epochs, train_loss_log, label='Training')
    ax.plot(epochs, val_loss_log, label='Validation')
    ax.legend()
    fig.savefig(path + model + '_loss.png', bbox_inches='tight', pad_inches=0.6, dpi=400)
    plt.close(fig)

    # --- Accuracies ---
    fig, ax = plt.subplots(figsize=(10,6))
    ax.grid(alpha=0.4)
    ax.set_title(model + ' train and validation accuracies')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')

    # overall classifier accuracy
    if train_overall_acc_log is not None and val_overall_acc_log is not None:
        ax.plot(epochs, train_overall_acc_log, label='Overall (training)', color='black', linewidth=2)
        ax.plot(epochs, val_overall_acc_log, label='Overall (validation)', color='black', linewidth=2, linestyle='dashed')

    # per-class accuracies
    for c in range(n_classes):
        train_class_acc = [acc[c] for acc in train_acc_log]
        val_class_acc = [acc[c] for acc in val_acc_log]
        ax.plot(epochs, train_class_acc, label=classes[c]+' (training)', color='C'+str(c))
        ax.plot(epochs, val_class_acc, label=classes[c]+' (validation)', color='C'+str(c), linestyle='dashed')

    ax.legend()
    fig.savefig(path + model + '_acc.png', bbox_inches='tight', pad_inches=0.6, dpi=400)
    plt.close(fig)


# plot ROC 
def plot_ROC(model, fpr, tpr, roc_auc):

    fig, ax = plt.subplots(figsize=(10,6))
    ax.grid(alpha = 0.4)

    ax.set_title(model + ' ROC curves')
    ax.set_ylabel('true positive rate')
    ax.set_xlabel('false positive rate')
    ax.plot([0,1], [0,1], label = 'Random classifier', linestyle = 'dashed', color = 'grey')

    # per-class
    for i, cls in enumerate(classes):
        ax.plot(fpr[i], tpr[i], label=f"{cls} (AUC = {roc_auc[i]:.2f})")

    # micro-average ROC
    ax.plot(fpr["micro"], tpr["micro"], 
            label=f"Micro-average (AUC = {roc_auc['micro']:.2f})", 
            linewidth=2, color='black')

    ax.legend()

    fig.savefig(path+model+'_ROC.png', bbox_inches='tight', pad_inches=0.6, dpi=400)
    plt.close(fig)


# plot confusion matrix 
def plot_CM(model, preds, labels):

    # model (string): CNN, TCN, IT

    cm = confusion_matrix(labels, preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(6,6))

    disp.plot(ax = ax, cmap='Blues', values_format='.1f')

    ax.set_title(model + ' confusion matrix')

    fig.savefig(path+model+'_CM.png', bbox_inches='tight', pad_inches=0.6, dpi=400)
    plt.close(fig)


# plot signal probability distribution
def plot_PS(model, probs, labels, n_classes=2):

    fig, ax = plt.subplots(figsize=(10,6))

    for c in range(n_classes):
        mask = (labels == c)
        ax.hist(probs[mask, 1], alpha=0.6, label=classes[c], density=True)

    ax.set_xlabel('predicted signal probability')
    ax.set_ylabel('normalized counts')
    ax.set_title('Predicted signal probability distribution')
    ax.legend(title='True class')

    fig.savefig(path+model+'_PS.png', bbox_inches='tight', pad_inches=0.6, dpi=400)
    plt.close(fig)


# plot weights histogram
def plot_model_params(model, save_path=None):

    # collect all parameters (weights and biases) with names
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    
    n_layers = len(params)
    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
    
    if n_layers == 1:
        axes = [axes]  # make iterable if only one layer
    
    for ax, (name, p) in zip(axes, params):
        values = p.detach().cpu().numpy().flatten()
        ax.hist(values, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(f"{name}\nmean={values.mean():.4f}, std={values.std():.4f}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


# plot activations
def plot_model_activations(model, dataloader, device, max_batches=1, bins=None, save_path=None):

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy().flatten()
        return hook

    # Register hooks for Conv and Linear layers
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Linear)):
            hooks.append(layer.register_forward_hook(get_activation(name)))

    # Run forward pass on a few batches
    model.eval()
    with torch.no_grad():
        for i, (xb, yb) in enumerate(dataloader):
            xb = xb.float().to(device)
            _ = model(xb)  # forward pass to collect activations
            if i + 1 >= max_batches:
                break

    # Remove hooks
    for h in hooks:
        h.remove()

    # Plot histograms
    n_layers = len(activations)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))

    if n_layers == 1:
        axes = [axes]

    for ax, (name, act) in zip(axes, activations.items()):
        ax.hist(act, bins=bins if bins is not None else 'auto', 
                color='coral', alpha=0.7, edgecolor='black')
        ax.set_title(f"{name}\nmean={act.mean():.4f}, std={act.std():.4f}")
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()