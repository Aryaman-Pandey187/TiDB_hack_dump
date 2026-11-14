import streamlit as st
import pymysql
import json
import requests
from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logging.basicConfig(
    filename='/home/ubuntu/app.log',  # EC2 path (local: 'app.log')
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Streamlit UI
st.title("Clinical Trial Matching Demo")
st.write("Enter your health profile to find matching trials. This is a demo, not medical advice.")

# Input form
user_input = st.text_area("Symptoms/Conditions (e.g., 'diabetes fatigue')", "diabetes symptoms fatigue")
age_group = st.selectbox("Age Group", ["ADULT", "OLDER_ADULT", "CHILD", "ALL"])
sex = st.selectbox("Sex", ["ALL", "MALE", "FEMALE"])
if st.button("Find Trials"):
    try:
        if not user_input.strip():
            raise ValueError("Please enter symptoms or conditions.")
        
        logging.info(f"Processing query: {user_input}, Age: {age_group}, Sex: {sex}")
        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate query embedding
        query_embedding = model.encode(user_input).tolist()
        if len(query_embedding) != 384:  # Ensure correct dimension
            raise ValueError("Embedding dimension mismatch.")
    
        query_embedding_json = json.dumps(query_embedding)
        logging.info("Generated query embedding.")

        # TiDB connection
        conn = pymysql.connect(
        host = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
        port = 4000,
        user = "4CSiRcRT3Bg3MT1.root",
        password = "bDQIdJIpxYPgD0tV",
        database = "test",
        ssl={'ca': '/home/ubuntu/certs/isrgrootx1.pem'}
        )
        cursor = conn.cursor()

        # Vector search with filters
        sql = """
            SELECT nct_number, study_title, conditions, brief_summary,
                   VEC_COSINE_DISTANCE(embedding, %s) AS distance
            FROM clinical_trials
            WHERE sex = %s AND age LIKE %s
            ORDER BY distance ASC
            LIMIT 5
        """
        cursor.execute(sql, (query_embedding_json, sex, f'%{age_group}%'))
        results = cursor.fetchall()

        if not results:
            raise ValueError("No matching trials found.")
        
        cursor.close()
        conn.close()
        logging.info(f"Found {len(results)} trial matches.")

        # Kimi LLM chaining
        kimi_api_key = ''
        kimi_url = 'https://api.moonshot.ai/v1/chat/completions'    
        headers = {
            'Authorization': f'Bearer {kimi_api_key}',
            'Content-Type': 'application/json'
        }

        # First call: Rank/aggregate results
        results_json = json.dumps([{
            'nct_number': row[0],
            'title': row[1],
            'conditions': row[2],
            'summary': row[3],
            'distance': row[4]
        } for row in results])
        prompt1 = (
            "You are Kimi, an AI assistant by Moonshot AI. Rank these clinical trials based on relevance to "
            f"'{user_input}' for a {age_group} {sex} patient: {results_json}"
        )
        response1 = requests.post(kimi_url, json={
            'model': 'kimi-k2-0711-preview',
            'messages': [{'role': 'system', 'content': 'You are Kimi, an AI assistant provided by Moonshot AI.'}, 
                        {'role': 'user', 'content': prompt1}],
            'temperature': 0.6
        }, headers=headers).json()
        ranked_results = response1['choices'][0]['message']['content']

        # Second call: Generate explanation
        prompt2 = (
            "You are Kimi, an AI assistant by Moonshot AI. Explain why these ranked trials "
            f"({ranked_results}) match the query '{user_input}' for a {age_group} {sex} patient. "
            "Keep it concise, user-friendly, and avoid medical advice."
        )
        response2 = requests.post(kimi_url, json={
            'model': 'kimi-k2-0711-preview',
            'messages': [{'role': 'system', 'content': 'You are Kimi, an AI assistant provided by Moonshot AI.'}, 
                        {'role': 'user', 'content': prompt2}],
            'temperature': 0.6
        }, headers=headers).json()
        explanation = response2['choices'][0]['message']['content']
        logging.info("Kimi generated explanation.")

        # Display results
        st.subheader("Matching Trials")
        for row in results:
            st.write(f"**Trial ID**: {row[0]}")
            st.write(f"**Title**: {row[1]}")
            st.write(f"**Conditions**: {row[2]}")
            st.write(f"**Summary**: {row[3]}")
            st.write(f"**Similarity Score**: {row[4]:.4f}")
            st.write("---")
        st.subheader("AI Analysis")
        st.write(f"**Ranked Results**: {ranked_results}")
        st.write(f"**Why These Match**: {explanation}")
        st.write("**Note**: This is a demo, not medical advice. Consult a doctor.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        logging.error(f"Error processing query: {str(e)}")

st.write("Future Enhancement: Blockchain for secure data sharing.")