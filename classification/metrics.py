import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score

def evaluate_classification(confusion_matrix, labels):
    """
    Takes a confusion matrix and outputs various classification metrics.

    Parameters:
    confusion_matrix (np.ndarray): Confusion matrix of shape (n_classes, n_classes)
    labels (list): List of class labels

    Returns:
    dict: Dictionary containing various classification metrics
    """
    if not isinstance(confusion_matrix, np.ndarray):
        raise ValueError("Confusion matrix must be a numpy array")

    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("Confusion matrix must be square")

    # Derive true and predicted labels from confusion matrix
    y_true = []
    y_pred = []
    for true_label in range(len(confusion_matrix)):
        for pred_label in range(len(confusion_matrix)):
            y_true.extend([true_label] * confusion_matrix[true_label][pred_label])
            y_pred.extend([pred_label] * confusion_matrix[true_label][pred_label])

    # Compute metrics
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)

    # Format and return results
    results = {
        'classification_report': report,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'cohen_kappa': kappa
    }
    return results

# Example usage
if __name__ == "__main__":
    cm = np.array([[9, 33], [17, 118]])
    labels = ['Class 0', 'Class 1']
    metrics = evaluate_classification(cm, labels)
    for key, value in metrics.items():
        print(f"{key}: {value}\n")
