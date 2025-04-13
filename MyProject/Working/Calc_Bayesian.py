import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load similarity scores from CSV
def load_scores():
    genuine_scores = pd.read_csv("testagegenuine_scores.csv")["Similarity"].values
    impostor_scores = pd.read_csv("balanced_impostor_scores.csv")["Similarity"].values
    return genuine_scores, impostor_scores

# Compute Bayesian threshold and plot
def compute_bayesian_threshold():
    genuine_scores, impostor_scores = load_scores()

    # Fit Gaussian distributions
    genuine_mean, genuine_std = np.mean(genuine_scores), np.std(genuine_scores)
    impostor_mean, impostor_std = np.mean(impostor_scores), np.std(impostor_scores)

    genuine_dist = stats.norm(genuine_mean, genuine_std)
    impostor_dist = stats.norm(impostor_mean, impostor_std)

    # Find threshold where P(S|G) = P(S|I)
    x_values = np.linspace(-0.3, 1, 1000)  # Adjusted to range from -1 to 1
    genuine_pdf = genuine_dist.pdf(x_values)
    impostor_pdf = impostor_dist.pdf(x_values)

    valid_indices = np.where(x_values > 0)[0]
    threshold_index = valid_indices[np.argmin(np.abs(genuine_pdf[valid_indices] - impostor_pdf[valid_indices]))]
    optimal_threshold = x_values[threshold_index]

    # Plot Distributions
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, genuine_pdf, label="Genuine Distribution", color="green")
    plt.plot(x_values, impostor_pdf, label="Impostor Distribution", color="red")
    plt.axvline(optimal_threshold, color="green", linestyle="--", label=f"Threshold = {optimal_threshold:.2f}")
    plt.legend()
    plt.xlabel("Similarity Score")
    plt.ylabel("Probability Density")
    plt.title("Bayesian Threshold for Face Similarity")
    plt.show()

    print(f"Optimal Bayesian threshold: {optimal_threshold:.2f}")

# Run the function
if __name__ == "__main__":
    compute_bayesian_threshold()
