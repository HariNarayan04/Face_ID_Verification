import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Load stored similarity scores
def load_scores():
    genuine_scores = pd.read_csv("genuine_scores.csv")["Similarity"].values
    impostor_scores = pd.read_csv("impostor_scores.csv")["Similarity"].values
    return genuine_scores, impostor_scores

# Plot normal distribution
def plot_distributions(genuine_scores, impostor_scores):
    plt.figure(figsize=(10, 6))

    # Plot genuine score distribution
    sns.histplot(genuine_scores, kde=True, stat="density", color="green", label="Genuine", bins=50)
    mu_g, std_g = np.mean(genuine_scores), np.std(genuine_scores)
    x_g = np.linspace(mu_g - 3*std_g, mu_g + 3*std_g, 100)
    plt.plot(x_g, norm.pdf(x_g, mu_g, std_g), color="green", lw=2)

    # Plot impostor score distribution
    sns.histplot(impostor_scores, kde=True, stat="density", color="red", label="Impostor", bins=50)
    mu_i, std_i = np.mean(impostor_scores), np.std(impostor_scores)
    x_i = np.linspace(mu_i - 3*std_i, mu_i + 3*std_i, 100)
    plt.plot(x_i, norm.pdf(x_i, mu_i, std_i), color="red", lw=2)

    plt.title("Normal Distribution of Genuine and Impostor Scores")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print statistics
    print(f"Genuine Scores: Mean = {mu_g:.4f}, Std Dev = {std_g:.4f}")
    print(f"Impostor Scores: Mean = {mu_i:.4f}, Std Dev = {std_i:.4f}")

# Run distribution plot
if __name__ == "__main__":
    genuine_scores, impostor_scores = load_scores()
    plot_distributions(genuine_scores, impostor_scores)

