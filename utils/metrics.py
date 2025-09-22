import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

# compute overall accuracy and accuracy for each class
def compute_accuracy(preds, labels, num_classes=3):
    preds_cls = torch.argmax(preds, dim=1)

    # overall accuracy
    overall = (preds_cls == labels).sum().item() / len(labels)

    # per-class accuracy
    per_class = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum().item() > 0:
            acc = (preds_cls[mask] == labels[mask]).sum().item() / mask.sum().item()
        else:
            acc = float("nan")
        per_class[c] = acc

    return overall, per_class


# ROC
def compute_roc(probs, labels, num_classes=3):

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    # one-hot encode labels
    y_bin = label_binarize(labels, classes=list(range(num_classes)))

    fpr, tpr, roc_auc = {}, {}, {}

    # ROC per class
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc