# src/utils/evaluation.py

import numpy as np
from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(preds, labels, num_classes=3):
    """
    Computes the confusion matrix for multi-class segmentation.
    Args:
        preds: list or np.ndarray of predicted masks (flattened)
        labels: list or np.ndarray of ground truth masks (flattened)
        num_classes: number of classes
    Returns:
        cm: confusion matrix of shape (num_classes, num_classes)
    """
    preds = np.concatenate([p.flatten() for p in preds])
    labels = np.concatenate([l.flatten() for l in labels])
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    return cm

def compute_metrics(preds, labels, num_classes=3):
    """
    Computes IoU, Precision, Recall, F1 Score, and Accuracy for multi-class segmentation.
    Args:
        preds: list of predicted masks (numpy arrays)
        labels: list of ground truth masks (numpy arrays)
        num_classes: number of classes
    Returns:
        metrics: dict with keys 'iou', 'precision', 'recall', 'f1_score', 'accuracy'
    """
    cm = compute_confusion_matrix(preds, labels, num_classes)
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (FP + FN + TP)

    # Per-class metrics
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    accuracy = np.sum(TP) / np.sum(cm)

    # Mean metrics (excluding background if desired)
    metrics = {
        'iou': np.mean(iou),
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1_score': np.mean(f1_score),
        'accuracy': accuracy,
        'per_class': {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': TP / (TP + FP + FN + TN + 1e-8)
        }
    }
    return metrics

def print_metrics(metrics, class_names=["NOCLOUD", "CLOUD", "SHADOW"]):
    print("Overall Metrics:")
    print(f"  IoU:       {metrics['iou']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1_score']:.3f}")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print("\nPer-class Metrics:")
    for idx, name in enumerate(class_names):
        print(f"  {name}: IoU={metrics['per_class']['iou'][idx]:.3f}, "
              f"Precision={metrics['per_class']['precision'][idx]:.3f}, "
              f"Recall={metrics['per_class']['recall'][idx]:.3f}, "
              f"F1={metrics['per_class']['f1_score'][idx]:.3f}")

# Example usage
if __name__ == "__main__":
    # Simulate predictions and labels for 2 samples (batch size = 2)
    preds = [np.random.randint(0, 3, (256, 256)), np.random.randint(0, 3, (256, 256))]
    labels = [np.random.randint(0, 3, (256, 256)), np.random.randint(0, 3, (256, 256))]
    metrics = compute_metrics(preds, labels, num_classes=3)
    print_metrics(metrics)
