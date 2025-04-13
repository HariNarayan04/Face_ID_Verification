import pandas as pd

# Load genuine and impostor scores
genuine_scores_path = "testagegenuine_scores.csv"
impostor_scores_path = "testageimpostor_scores.csv"

# Load CSVs
genuine_df = pd.read_csv(genuine_scores_path)
impostor_df = pd.read_csv(impostor_scores_path)

# Get number of rows in genuine data
num_genuine_rows = len(genuine_df)

# Randomly sample impostor data to match genuine rows
balanced_impostor_df = impostor_df.sample(n=num_genuine_rows)

# Save the balanced impostor data to a new CSV file
balanced_impostor_path = "balanced_impostor_scores.csv"
balanced_impostor_df.to_csv(balanced_impostor_path, index=False)

print(f"Balanced impostor data saved to '{balanced_impostor_path}' with {num_genuine_rows} rows.")