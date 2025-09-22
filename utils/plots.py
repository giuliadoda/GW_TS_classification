import numpy as np
import matplotlib.pyplot as plt
import torch
import math

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


path = '/mnt/POD/NNDL_gd/code/GW_TS_classification_copy/plots/'
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
    ax.grid(alpha=0.3)

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
def plot_model_params(model, save_path=None, normalize=True):

    # collect all parameters (weights and biases) with names
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    # group by layer name (everything before the last ".")
    layers = {}
    for name, p in params:
        layer_name, param_type = name.rsplit('.', 1)
        layers.setdefault(layer_name, {})[param_type] = p

    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 4*n_layers))

    if n_layers == 1:
        axes = axes.reshape(1, 2)  # ensure 2D array even for single layer

    for row, (layer_name, param_dict) in enumerate(layers.items()):
        for col, param_type in enumerate(["weight", "bias"]):
            ax = axes[row, col]
            if param_type in param_dict:
                values = param_dict[param_type].detach().cpu().numpy().flatten()
                ax.hist(
                    values,
                    density=normalize,  
                    color="steelblue",
                    alpha=0.7,
                    edgecolor="black"
                )
                ax.set_title(
                    f"{layer_name}.{param_type}\n"
                    f"mean={values.mean():.4f}, std={values.std():.4f}"
                )
                ax.set_xlabel("Value")
                ax.set_ylabel("Density" if normalize else "Frequency")
            else:
                ax.axis("off")  # blank subplot if no bias

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

    # Register hooks
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Linear)):
            hooks.append(layer.register_forward_hook(get_activation(name)))

    # Forward pass
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, (xb, yb) in enumerate(dataloader):
            xb = xb.float().to(device)
            _ = model(xb)
            if i + 1 >= max_batches:
                break

    # Remove hooks
    for h in hooks:
        h.remove()

    if not activations:
        print("No activations collected.")
        return

    # Determine layout: 2 columns
    n_layers = len(activations)
    n_cols = 2
    n_rows = math.ceil(n_layers / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()  # flatten in case of multiple rows

    for ax, (name, act) in zip(axes, activations.items()):
        ax.hist(act, bins=bins if bins is not None else 'auto',
                color='coral', alpha=0.7, edgecolor='black')
        ax.set_title(f"{name}\nmean={act.mean():.4f}, std={act.std():.4f}")
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Frequency")

    # Turn off unused axes
    for ax in axes[n_layers:]:
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()