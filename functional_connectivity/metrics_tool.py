# metrics_tool.py
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pickle

class MetricsLogger:
    def __init__(self):
        self.results = {}

    def log_result(self, name, y_true, y_prob, history=None):
        """
        Parameters
        ----------
        name : str
            Model name (e.g., anesthetic/drug name).
        y_true : array-like
            Ground-truth labels (0 or 1).
        y_prob : array-like
            Predicted probabilities (0 to 1).
        history : dict, optional
            XGBoost `evals_result` (optional).
        """
        self.results[name] = {
            "y_true": y_true,
            "y_prob": y_prob,
            "history": history
        }

    def save_data(self, filepath):
        """Save the raw data used for metric computation to avoid re-training."""
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Metrics data saved to {filepath}")

    def load_data(self, filepath):
        """Load previously saved metric data."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.results = pickle.load(f)
            return True
        return False

def plot_all_curves(metrics_data, save_dir, group_name):
    """
    Plot all performance curves.

    Parameters
    ----------
    metrics_data : dict
        A dictionary where keys are model/anesthetic names and values are dicts
        containing 'y_true', 'y_prob', and optionally 'history'.
    save_dir : str
        Base directory to save figures.
    group_name : str
        Group name used to create the output folder and figure titles.
    """
    output_dir = os.path.join(save_dir, group_name, "Performance_Curves")
    os.makedirs(output_dir, exist_ok=True)

    # Set plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_data)))

    # ==========================================
    # 1. ROC Curve (AUC-ROC)
    # ==========================================
    plt.figure(figsize=(10, 8))
    for (name, data), color in zip(metrics_data.items(), colors):
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {group_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'ROC_Curve.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'ROC_Curve.pdf'))
    plt.close()

    # ==========================================
    # 2. Precision-Recall Curve
    # ==========================================
    plt.figure(figsize=(10, 8))
    for (name, data), color in zip(metrics_data.items(), colors):
        precision, recall, _ = precision_recall_curve(data['y_true'], data['y_prob'])
        avg_pre = average_precision_score(data['y_true'], data['y_prob'])
        plt.plot(recall, precision, color=color, lw=2, label=f'{name} (AP = {avg_pre:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {group_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'PR_Curve.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'PR_Curve.pdf'))
    plt.close()

    # ==========================================
    # 3. Training Loss / Accuracy Curves (XGBoost only)
    # ==========================================
    # Check whether any history data exists
    has_history = any(data.get('history') is not None for data in metrics_data.values())

    if has_history:
        # Loss Curve
        plt.figure(figsize=(10, 8))
        for (name, data), color in zip(metrics_data.items(), colors):
            history = data.get('history')
            if history:
                # XGBoost typically returns 'validation_0' (train) and 'validation_1' (test)
                # Here we plot the test-set loss
                results = history['validation_1']['logloss']
                plt.plot(results, color=color, lw=2, label=f'{name}')

        plt.xlabel('Iterations')
        plt.ylabel('Log Loss')
        plt.title(f'Validation Loss over Iterations - {group_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'Validation_Loss_Curve.png'), dpi=300)
        plt.close()

        # Accuracy Curve (converted from error: Accuracy = 1 - error)
        plt.figure(figsize=(10, 8))
        for (name, data), color in zip(metrics_data.items(), colors):
            history = data.get('history')
            if history and 'error' in history['validation_1']:
                errors = history['validation_1']['error']
                accuracies = [1 - x for x in errors]
                plt.plot(accuracies, color=color, lw=2, label=f'{name}')

        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title(f'Validation Accuracy over Iterations - {group_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'Validation_Accuracy_Curve.png'), dpi=300)
        plt.close()