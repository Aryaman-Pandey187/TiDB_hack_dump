# import json
# from sentence_transformers import SentenceTransformer

# # Initialize SentenceTransformer model (same as preprocessing)
# model = SentenceTransformer('all-MiniLM-L6-v2')

# user_query = "diabetes symptoms fatigue"

# # Generate embedding for query
# query_embedding = model.encode(user_query).tolist()  # List of 384 floats
# query_embedding_json = json.dumps(query_embedding)  # For SQL

# print(query_embedding_json)

import pymysql
import json
from sentence_transformers import SentenceTransformer
import mysql.connector

# Initialize SentenceTransformer model (same as preprocessing)
model = SentenceTransformer('all-MiniLM-L6-v2')

conn = mysql.connector.connect(
  host = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
  port = 4000,
  user = "4CSiRcRT3Bg3MT1.root",
  password = "bDQIdJIpxYPgD0tV",
  database = "test",
  ssl_ca = "isrgrootx1.pem",
  ssl_verify_cert = True,
  ssl_verify_identity = True
)

cursor = conn.cursor()

# User query (example; later from frontend)
user_query = "diabetes symptoms fatigue"

# Generate embedding for query
query_embedding = model.encode(user_query).tolist()  # List of 384 floats
query_embedding_json = json.dumps(query_embedding)  # For SQL

# Vector search: Use cosine distance to find similar trials

sql = """
    SELECT nct_number, study_title, conditions, brief_summary,
           VEC_COSINE_DISTANCE(embedding, %s) AS distance
    FROM clinical_trials
    ORDER BY distance ASC
    LIMIT 5
"""
cursor.execute(sql, (query_embedding_json,))  # Pass single parameter as tuple
results = cursor.fetchall()

# Display results
for row in results:
    print(f"Trial: {row[0]} - {row[1]}")
    print(f"Conditions: {row[2]}")
    print(f"Summary: {row[3]}")
    print(f"Distance: {row[4]}\n")

# Close connection
cursor.close()
conn.close()