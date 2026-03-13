import numpy as np
import torch
from src.data_module.compute_metrics import register_metrics
from sklearn.metrics import (
    top_k_accuracy_score, accuracy_score, f1_score, fbeta_score, precision_score, recall_score,
    roc_auc_score, cohen_kappa_score, average_precision_score
)


@register_metrics('classification')
def classification_compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions[1], eval_pred.predictions[0]
    labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
    probs = torch.softmax(torch.tensor(logits), dim=-1)
    preds = torch.argmax(probs, dim=-1)

    # Convert to numpy for sklearn calculations
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    preds_np = preds.numpy() if isinstance(preds, torch.Tensor) else preds
    probs_np = probs.numpy() if isinstance(probs, torch.Tensor) else probs

    # Basic classification metrics
    top1_accuracy = accuracy_score(labels_np, preds_np)
    precision = precision_score(labels_np, preds_np, average='weighted', zero_division=0)
    recall = recall_score(labels_np, preds_np, average='weighted', zero_division=0)
    f1 = f1_score(labels_np, preds_np, average='weighted', zero_division=0)
    f2 = fbeta_score(labels_np, preds_np, beta=2, average='weighted', zero_division=0)

    # Top-k accuracy
    num_classes = logits.shape[1]
    top3_accuracy = top_k_accuracy_score(labels_np, probs_np, k=min(3, num_classes),
                                         labels=np.arange(num_classes))
    top5_accuracy = top_k_accuracy_score(labels_np, probs_np, k=min(5, num_classes),
                                         labels=np.arange(num_classes))

    # AUROC (Area Under ROC Curve)
    try:
        if num_classes == 2:
            # Binary classification case
            auroc = roc_auc_score(labels_np, probs_np[:, 1])
        else:
            # Multi-class case - using ovr (one-vs-rest) strategy
            auroc = roc_auc_score(labels_np, probs_np, multi_class='ovr', average='weighted')
    except ValueError:
        # Handle case with only one class
        auroc = 0.0

    # Cohen's Kappa
    cohen_kappa = cohen_kappa_score(labels_np, preds_np)

    # Accuracy per class
    unique_labels = torch.unique(labels)
    class_accuracies = {}

    for label in unique_labels:
        true_positive = torch.sum((preds == label) & (labels == label)).item()
        total_samples = torch.sum(labels == label).item()
        if total_samples > 0:
            class_accuracies[f'acc_class_{label.item()}'] = true_positive / total_samples
        else:
            class_accuracies[f'acc_class_{label.item()}'] = 0.0

    return {
        # Basic classification metrics
        'top1_accuracy': top1_accuracy,
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        'precision'    : precision,
        'recall'       : recall,
        'f1'           : f1,
        'f2'           : f2,
        'auroc'        : auroc,
        'cohen_kappa'  : cohen_kappa,
        **class_accuracies
    }
