import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


path = '/mnt/POD/NNDL_gd/code/GW_TS_classification/plots/'
classes = ['Noise', 'Signal', 'Glitch']


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
def plot_loss_acc(model, train_loss_log, val_loss_log, train_acc_log, val_acc_log, n_classes = 3):

    epochs = np.arange(1, len(train_loss_log) + 1)

    # losses
    fig, ax = plt.subplots(figsize=(10,6))
    ax.grid(alpha=0.4)

    ax.set_title(model + ' train and validation losses')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')

    ax.plot(epochs, train_loss_log, label='Training')
    ax.plot(epochs, val_loss_log, label='Validation')

    ax.legend()

    fig.savefig(path+model+'_loss.png', bbox_inches='tight', pad_inches=0.6, dpi=400)
    plt.close(fig)

    # accuracies
    fig, ax = plt.subplots(figsize=(10,6))
    ax.grid(alpha=0.4)

    ax.set_title(model + ' train and validation accuracies')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')

    for c in range(n_classes):
        train_class_acc = [acc[c] for acc in train_acc_log]
        val_class_acc = [acc[c] for acc in val_acc_log]
        ax.plot(epochs, train_class_acc, label=classes[c]+' (training)', color='C'+str(c))
        ax.plot(epochs, val_class_acc, label=classes[c]+' (validation)', color='C'+str(c), linestyle='dashed')

    ax.legend()

    fig.savefig(path+model+'_acc.png', bbox_inches='tight', pad_inches=0.6, dpi=400)
    plt.close(fig)


# plot ROC --> ADD OVERALL CLASSIFIER!!
def plot_ROC(model, fpr, tpr, roc_auc, n_classes=3):

    fig, ax = plt.subplots(figsize=(10,6))
    ax.grid(alpha = 0.4)

    ax.set_title(model + ' ROC curves')
    ax.set_ylabel('true positive rate')
    ax.set_xlabel('false positive rate')
    ax.plot(fpr[0], fpr[0], label = 'Random classifier', linestyle = 'dashed', color = 'grey')

    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label = classes[i] + f'(AUC = {roc_auc[i]:.2f})' )
    
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
def plot_PS(model, probs, labels, n_classes=3):

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