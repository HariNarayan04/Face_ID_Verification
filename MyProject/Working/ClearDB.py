import sqlite3

def reset_database():
    conn = sqlite3.connect("TestAGEcelebrity_embeddings.db")
    cursor = conn.cursor()
    
    # Drop the table if it already exists (removes UNIQUE constraint)
    cursor.execute("DROP TABLE IF EXISTS embeddings")
    
    conn.commit()
    conn.close()
    print("Database reset. Run initialize_db() again to create a new table.")

# reset_database()



# Uncommment to run the code
