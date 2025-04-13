import sqlite3
import numpy as np

DB_PATH = "Backend_Test/app/ml/User.db"

def inspect_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check total number of entries
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    total_entries = cursor.fetchone()[0]
    print(f"Total Entries: {total_entries}\n")

    # Fetch all stored data
    cursor.execute("SELECT id, roll_no, LENGTH(embedding), LENGTH(cropped_face) FROM embeddings")
    rows = cursor.fetchall()

    print(f"{'ID':<5} {'Roll No':<15} {'Embedding Size (bytes)':<20} {'Face Size (bytes)':<20}")
    print("-" * 65)

    for row in rows:
        db_id, roll_no, embedding_size, face_size = row
        print(f"{db_id:<5} {roll_no:<15} {embedding_size:<20} {face_size:<20}")

    conn.close()

if __name__ == "__main__":
    inspect_database()