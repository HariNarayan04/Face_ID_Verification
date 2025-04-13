import sqlite3
import numpy as np
from collections import defaultdict

# Connect to the database
# conn = sqlite3.connect("TestAGEcelebrity_embeddings.db")
conn = sqlite3.connect("User.db")
cursor = conn.cursor()

# Retrieve stored embeddings
cursor.execute("SELECT roll_no, embedding FROM embeddings")
rows = cursor.fetchall()
conn.close()

# Dictionary to count embeddings per roll number
embedding_count = defaultdict(int)

for row in rows:
    roll_no = row[0]
    embedding_count[roll_no] += 1  # Count occurrences

# Print insights
print("Database Insights:")
print(f"Total unique roll numbers: {len(embedding_count)}")
for roll_no, count in embedding_count.items():
    print(f"Roll No: {roll_no}, Number of embeddings: {count}")


# import sqlite3

# def clear_database():
#     conn = sqlite3.connect("celebrity_embeddings.db")
#     cursor = conn.cursor()
    
#     # Delete all records from the table
#     cursor.execute("DELETE FROM embeddings")
    
#     # # Optional: Reset the auto-increment ID (if applicable)
#     # cursor.execute("DELETE FROM sqlite_sequence WHERE name='embeddings'")  

#     conn.commit()
#     conn.close()
#     print("Database cleared successfully.")

# # Run the function
# clear_database()






# 1 siddharth malhotra
# 2 Sahrukh Khan
# 3 IDK
# 4 Varun Dhawan
# 5 Suhant singh
# 6 Ritik Rohan 
# 7 Aamir khan
# 8 Ramcharan
# 9 Shahid kapoor
# 10 Ranveer kapoor
