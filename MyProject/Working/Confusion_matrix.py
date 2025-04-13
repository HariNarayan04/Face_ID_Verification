import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay

# Load stored similarity scores
def load_scores():
    genuine_scores = pd.read_csv("testagegenuine_scores.csv")["Similarity"].values
    impostor_scores = pd.read_csv("balanced_impostor_scores.csv")["Similarity"].values
    return genuine_scores, impostor_scores

# Evaluate metrics at different thresholds
def evaluate_metrics(thresholds):
    genuine_scores, impostor_scores = load_scores()

    # Create ground truth labels: 1 for genuine, 0 for impostor
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])

    # Initialize lists to store metric results
    precisions, recalls, f1_scores, accuracies = [], [], [], []
    y_pred_list = []

    for threshold in thresholds:
        # Predicted labels based on threshold
        y_pred = np.concatenate([
            (genuine_scores >= threshold).astype(int),  # Genuine classified as genuine (1) or impostor (0)
            (impostor_scores >= threshold).astype(int)  # Impostor classified as genuine (1) or impostor (0)
        ])
        y_pred_list.append(y_pred)

        # Compute metrics
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
        accuracies.append(accuracy_score(y_true, y_pred))

    return precisions, recalls, f1_scores, accuracies, y_pred_list

# Plot confusion matrix for a given threshold
def plot_confusion_matrix(y_true, y_pred, threshold):
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix at Threshold {threshold:.2f}:\n{cm}\n")

# Plot the metrics
def plot_metrics(thresholds, precisions, recalls, f1_scores, accuracies):
    plt.figure(figsize=(12, 8))
    
    plt.plot(thresholds, precisions, label="Precision", marker='o', color="blue")
    plt.plot(thresholds, recalls, label="Recall", marker='o', color="green")
    plt.plot(thresholds, f1_scores, label="F1 Score", marker='o', color="orange")
    plt.plot(thresholds, accuracies, label="Accuracy", marker='o', color="red")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Performance Metrics vs. Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run evaluation and plot results
if __name__ == "__main__":
    thresholds = [0.14, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26]
    precisions, recalls, f1_scores, accuracies, y_pred_list = evaluate_metrics(thresholds)
    plot_metrics(thresholds, precisions, recalls, f1_scores, accuracies)
    
    # Load true labels
    genuine_scores, impostor_scores = load_scores()
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    
    # Plot confusion matrix for each threshold
    for threshold, y_pred in zip(thresholds, y_pred_list):
        plot_confusion_matrix(y_true, y_pred, threshold)

