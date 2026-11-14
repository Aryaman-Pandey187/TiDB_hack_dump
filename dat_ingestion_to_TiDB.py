import pandas as pd
# import pymysql
import json  # For parsing embeddings

# Load preprocessed CSV
df = pd.read_csv('preprocessed_trials.csv')

# TiDB connection (replace with your details)
# conn = pymysql.connect(
#     host='gateway01.ap-southeast-1.prod.aws.tidbcloud.com',  # e.g., 'gateway-01.aws.tidbcloud.com'
#     port=4000,
#     user='4CSiRcRT3Bg3MT1.root',  # e.g., 'root'
#     password='bDQIdJIpxYPgD0tV',
#     database='test'  # Your DB name
# )

import mysql.connector

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

# Create table (no FULLTEXT in column defs; add indexes separately)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS clinical_trials_latest (
        nct_number VARCHAR(20) PRIMARY KEY,
        study_title TEXT,
        study_url TEXT,
        acronym VARCHAR(50),
        study_status VARCHAR(50),
        brief_summary TEXT,
        study_results VARCHAR(10),  -- e.g., 'YES'/'NO'
        conditions TEXT,
        interventions TEXT,
        primary_outcome_measures TEXT,
        secondary_outcome_measures TEXT,
        other_outcome_measures TEXT,
        sponsor TEXT,
        collaborators TEXT,
        sex VARCHAR(50),
        age TEXT,
        phases TEXT,
        enrollment FLOAT,
        funder_type VARCHAR(50),
        study_type VARCHAR(50),
        study_design TEXT,
        other_ids TEXT,
        start_date DATE,
        primary_completion_date DATE,
        completion_date DATE,
        first_posted DATE,
        results_first_posted DATE,
        last_update_posted DATE,
        locations TEXT,
        study_documents TEXT,
        text_for_embedding TEXT,
        embedding VECTOR(384)
    );
""")

# Add vector index with on-demand replica
cursor.execute("ALTER TABLE clinical_trials_latest ADD VECTOR INDEX vec_idx ((VEC_COSINE_DISTANCE(embedding))) ADD_COLUMNAR_REPLICA_ON_DEMAND;")

# Add fulltext index (no replica needed)
# cursor.execute("CREATE FULLTEXT INDEX ft_idx ON clinical_trials(brief_summary, conditions, primary_outcome_measures);")
conn.commit()

# Batch insert (adjust batch_size if needed)
batch_size = 200
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i + batch_size]
    values = []
    for _, row in batch.iterrows():
        # Escape text fields to avoid SQL injection; handle NaN as None
        def safe_text(val):
            return val.replace("'", "''") if pd.notna(val) else None
        
        values.append((
            row['NCT Number'],  # VARCHAR
            safe_text(row['Study Title']),  # TEXT
            safe_text(row['Study URL']),  # TEXT
            safe_text(row['Acronym']),  # VARCHAR
            safe_text(row['Study Status']),  # VARCHAR
            safe_text(row['Brief Summary']),  # TEXT FULLTEXT
            safe_text(row['Study Results']),  # VARCHAR
            safe_text(row['Conditions']),  # TEXT FULLTEXT
            safe_text(row['Interventions']),  # TEXT
            safe_text(row['Primary Outcome Measures']),  # TEXT
            safe_text(row['Secondary Outcome Measures']),  # TEXT
            safe_text(row['Other Outcome Measures']),  # TEXT
            safe_text(row['Sponsor']),  # TEXT
            safe_text(row['Collaborators']),  # TEXT
            safe_text(row['Sex']),  # VARCHAR
            safe_text(row['Age']),  # TEXT
            safe_text(row['Phases']),  # TEXT
            row['Enrollment'] if pd.notna(row['Enrollment']) else None,  # FLOAT
            safe_text(row['Funder Type']),  # VARCHAR
            safe_text(row['Study Type']),  # VARCHAR
            safe_text(row['Study Design']),  # TEXT
            safe_text(row['Other IDs']),  # TEXT
            row['Start Date'] if pd.notna(row['Start Date']) else None,  # DATE (string ok)
            row['Primary Completion Date'] if pd.notna(row['Primary Completion Date']) else None,  # DATE
            row['Completion Date'] if pd.notna(row['Completion Date']) else None,  # DATE
            row['First Posted'] if pd.notna(row['First Posted']) else None,  # DATE
            row['Results First Posted'] if pd.notna(row['Results First Posted']) else None,  # DATE
            row['Last Update Posted'] if pd.notna(row['Last Update Posted']) else None,  # DATE
            safe_text(row['Locations']),  # TEXT
            safe_text(row['Study Documents']),  # TEXT
            safe_text(row['text_for_embedding']),  # TEXT
            row['embedding']  # VECTOR (JSON string)
        ))

    sql = """
        INSERT IGNORE INTO clinical_trials_latest 
        (nct_number, study_title, study_url, acronym, study_status, brief_summary, study_results, conditions, interventions, 
         primary_outcome_measures, secondary_outcome_measures, other_outcome_measures, sponsor, collaborators, sex, age, 
         phases, enrollment, funder_type, study_type, study_design, other_ids, start_date, primary_completion_date, 
         completion_date, first_posted, results_first_posted, last_update_posted, locations, study_documents, 
         text_for_embedding, embedding) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.executemany(sql, values)
    conn.commit()
    print(f"Inserted batch {i//batch_size + 1} of {len(df)//batch_size + 1}")

# Close connection
cursor.close()
conn.close()
print("Ingestion complete!")