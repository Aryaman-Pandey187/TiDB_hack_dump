import pandas as pd
from sentence_transformers import SentenceTransformer
import json  # For storing embeddings as JSON strings

# Load your CSV (now with ~1000 records)
df = pd.read_csv("C:\\Users\\pande\\Downloads\\TiDB_Hackathon\\ctg-studies.csv", low_memory=False)

# Print columns to confirm names (run once, then comment out)
# print(df.columns.tolist())

# Clean: Drop duplicates and missing key columns (adjusted for actual column names)
df = df.drop_duplicates(subset=['NCT Number'])  # Unique by trial ID
df = df.dropna(subset=['Conditions', 'Study Status', 'Brief Summary', 'Locations'])  # Key fields including Location

# Generate embeddings on combined text (e.g., Conditions + Brief Summary)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, lightweight model
df['text_for_embedding'] = df['Conditions'] + ' ' + df['Brief Summary']
df['embedding'] = df['text_for_embedding'].apply(lambda x: json.dumps(model.encode(x).tolist()))  # Vector as JSON string for TiDB

# Save preprocessed CSV (ready for TiDB ingestion)
df.to_csv('preprocessed_trials.csv', index=False)
print(f"Preprocessed {len(df)} diverse trials.")