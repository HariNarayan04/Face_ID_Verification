
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load stored similarity scores
def load_scores():
    genuine_scores = pd.read_csv("genuine_scores.csv")["Similarity"].values
    impostor_scores = pd.read_csv("impostor_scores.csv")["Similarity"].values
    return genuine_scores, impostor_scores

# Plot interactive histograms with hover info
def plot_interactive_distributions(genuine_scores, impostor_scores, bins=50):
    # Create histograms
    hist_genuine, bin_edges_genuine = np.histogram(genuine_scores, bins=bins, density=False)
    hist_impostor, bin_edges_impostor = np.histogram(impostor_scores, bins=bins, density=False)

    # Create bar plots for genuine and impostor
    fig = go.Figure()

    # Genuine scores histogram
    fig.add_trace(go.Bar(
        x=bin_edges_genuine[:-1],  # Use the bin start as the x-axis
        y=hist_genuine,
        name="Genuine",
        marker_color="green",
        hovertemplate="Score: %{x}<br>Genuine Count: %{y}<extra></extra>"
    ))

    # Impostor scores histogram
    fig.add_trace(go.Bar(
        x=bin_edges_impostor[:-1],
        y=hist_impostor,
        name="Impostor",
        marker_color="red",
        hovertemplate="Score: %{x}<br>Impostor Count: %{y}<extra></extra>"
    ))

    # Layout adjustments
    fig.update_layout(
        title="Distribution of Genuine and Impostor Scores with Counts",
        xaxis_title="Cosine Similarity Score",
        yaxis_title="Count",
        barmode="overlay",  # Overlay histograms for comparison
        bargap=0.1,
        hovermode="x unified"
    )

    fig.show()

# Run interactive plot
if __name__ == "__main__":
    genuine_scores, impostor_scores = load_scores()
    plot_interactive_distributions(genuine_scores, impostor_scores, bins=50)