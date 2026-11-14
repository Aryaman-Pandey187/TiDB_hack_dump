import streamlit as st
import pymysql
import mysql.connector
import json
from sentence_transformers import SentenceTransformer

# Streamlit UI
st.title("Clinical Trial Matching Demo")
st.write("Enter your health profile to find matching trials. This is a demo, not medical advice.")

# Input form
user_input = st.text_area("Symptoms/Conditions (e.g., 'diabetes fatigue')", "diabetes symptoms fatigue")
age_group = st.selectbox("Age Group", ["ADULT", "OLDER_ADULT", "CHILD", "ALL"])
sex = st.selectbox("Sex", ["ALL", "MALE", "FEMALE"])
if st.button("Find Trials"):

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate query embedding
    query_embedding = model.encode(user_input).tolist()
    query_embedding_json = json.dumps(query_embedding)

    # TiDB connection
    # conn = pymysql.connect(
    #     host='gateway01.ap-southeast-1.prod.aws.tidbcloud.com',
    #     port=4000,
    #     user='your_user',
    #     password='your_password',
    #     database='trial_matching',
    #     ssl={'ca': r'C:\Users\pande\Downloads\TiDB_Hackathon\isrgrootx1.pem'}
    # )

    conn = mysql.connector.connect(
    host = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port = 4000,
    user = "4CSiRcRT3Bg3MT1.root",
    password = "",
    database = "test",
    ssl_ca = "isrgrootx1.pem",
    ssl_verify_cert = True,
    ssl_verify_identity = True
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

    # Display results
    st.subheader("Matching Trials")
    for row in results:
        st.write(f"**Trial ID**: {row[0]}")
        st.write(f"**Title**: {row[1]}")
        st.write(f"**Conditions**: {row[2]}")
        st.write(f"**Summary**: {row[3]}")
        st.write(f"**Similarity Score**: {row[4]:.4f}")
        st.write("---")

    cursor.close()
    conn.close()

st.write("Future Enhancement: Blockchain for secure data sharing.")