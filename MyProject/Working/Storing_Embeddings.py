import sqlite3
import numpy as np
import cv2
import insightface
import os
import re

def initialize_db():
    # conn = sqlite3.connect("TestAGEcelebrity_embeddings.db")
    conn = sqlite3.connect("User.db")
    cursor = conn.cursor()
    
    # Create the table with an auto-increment ID, allowing multiple embeddings per roll_no
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique ID for each entry
            roll_no TEXT,  -- Allows duplicate roll numbers
            embedding BLOB
        )
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def extract_face_embedding(image_path, app):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Failed to read {image_path}")
        return None
    
    faces = app.get(img)
    if faces:
        return faces[0].embedding
    else:
        print(f"Warning: No face detected in {image_path}")
        return None

def store_embedding(roll_no, embedding):
    if embedding is None:
        return
    
    # conn = sqlite3.connect("TestAGEcelebrity_embeddings.db")
    conn = sqlite3.connect("User.db")
    cursor = conn.cursor()
    
    # Store each embedding in a new row
    cursor.execute("INSERT INTO embeddings (roll_no, embedding) VALUES (?, ?)",
                   (roll_no, embedding.tobytes()))
    
    conn.commit()
    conn.close()
    print(f"Stored embedding for roll no: {roll_no}")


def process_images(folder_path):
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0)

    initialize_db()
    
    for file_name in os.listdir(folder_path):
        match = re.match(r"(.+)_\d{4}\.jpg", file_name)
        if match:
            roll_no = match.group(1).replace(" ", "_")  # Corrected to use only the name part
            image_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_name}...")

            embedding = extract_face_embedding(image_path, app)
            store_embedding(roll_no, embedding)
            print(f"Stored for roll no. : {roll_no}...")

folder_path = "MyProject/Working/Datasets/TestPoject"  # Change this to your actual dataset path
process_images(folder_path)
