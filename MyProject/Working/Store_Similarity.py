import sqlite3
import numpy as np
import csv
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings from database
def load_embeddings():
    conn = sqlite3.connect("TestAGEcelebrity_embeddings.db")
    cursor = conn.cursor()
    cursor.execute("SELECT roll_no, embedding FROM embeddings")
    data = cursor.fetchall()
    conn.close()

    embeddings_dict = {}
    
    for roll_no, embedding in data:
        embedding_array = np.frombuffer(embedding, dtype=np.float32)
        if roll_no in embeddings_dict:
            embeddings_dict[roll_no].append(embedding_array)
        else:
            embeddings_dict[roll_no] = [embedding_array]

    return embeddings_dict

# Compute similarity scores and save to CSV
def save_similarity_scores():
    embeddings_dict = load_embeddings()
    genuine_scores = []
    impostor_scores = []

    roll_numbers = list(embeddings_dict.keys())
    
    print("Start")

    # Compute genuine similarity scores
    for roll_no, embeddings in embeddings_dict.items():
        for emb1, emb2 in combinations(embeddings, 2):
            sim = cosine_similarity([emb1], [emb2])[0][0]
            genuine_scores.append(sim)

    print("done half way")

    # Compute impostor similarity scores
    for roll_no1, roll_no2 in combinations(roll_numbers, 2):
        for emb1 in embeddings_dict[roll_no1]:
            for emb2 in embeddings_dict[roll_no2]:
                sim = cosine_similarity([emb1], [emb2])[0][0]
                impostor_scores.append(sim)

    print("End")
    # Save to CSV files
    with open("testagegenuine_scores.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Similarity"])
        for score in genuine_scores:
            writer.writerow([score])

    print("half writing done")

    with open("testageimpostor_scores.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Similarity"])
        for score in impostor_scores:
            writer.writerow([score])

    print("Similarity scores saved to CSV files.")

# Run the function
if __name__ == "__main__":
    save_similarity_scores()
