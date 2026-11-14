import streamlit as st
import pymysql
import json
import requests
from sentence_transformers import SentenceTransformer
import logging
import boto3
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import urllib.parse
import pandas as pd

# Setup logging
logging.basicConfig(
    filename='/home/ec2-user/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Streamlit UI styling
st.markdown("""
    <style>
        .main {background-color: #f0f2f6;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
        .stTextInput>div>input {border: 2px solid #2c3e50; border-radius: 5px;}
        h1 {color: #2c3e50;}
        h2, h3 {color: #34495e;}
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Clinical Trial Matching Demo 🩺")
st.write("Enter your health profile to find matching clinical trials. Demo only, not medical advice.")

# Sidebar
st.sidebar.title("About")
st.sidebar.write("🧬 AI-powered trial matching using TiDB, Kimi AI, and AWS.")
st.sidebar.write("📍 Maps powered by OpenStreetMap.")
st.sidebar.write("🔮 Future: Blockchain for secure data sharing.")

# Input form
user_input = st.text_area("Symptoms/Conditions (e.g., 'diabetes fatigue') 💊", "diabetes symptoms fatigue")
age_group = st.selectbox("Age Group 🎂", ["ADULT", "OLDER_ADULT", "CHILD", "ALL"])
sex = st.selectbox("Sex 🚻", ["ALL", "MALE", "FEMALE"])
email = st.text_input("Email (optional, for report) 📧", "")
if st.button("Find Trials 🔍"):
    try:
        if not user_input.strip():
            raise ValueError("Please enter symptoms or conditions.")
        
        logging.info(f"Processing query: {user_input}, Age: {age_group}, Sex: {sex}, Email: {email}")

        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate query embedding
        query_embedding = model.encode(user_input).tolist()
        if len(query_embedding) != 384:
            raise ValueError("Embedding dimension mismatch.")
        query_embedding_json = json.dumps(query_embedding)
        logging.info("Generated query embedding.")

        # TiDB connection
        conn = pymysql.connect(
            host='gateway01.ap-southeast-1.prod.aws.tidbcloud.com',
            port=4000,
            user='your_user',  # Replace
            password='your_password',  # Replace
            database='trial_matching',
            ssl={'ca': '/home/ec2-user/certs/isrgrootx1.pem'}
        )
        cursor = conn.cursor()

        # Vector search with filters
        sql = """
            SELECT nct_number, study_title, conditions, brief_summary, locations,
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
        kimi_api_key = ''  # Replace
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
            'distance': row[5]
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
        }, headers=headers)
        response1.raise_for_status()
        ranked_results = response1.json()['choices'][0]['message']['content']
        logging.info("Kimi ranked results.")

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
        }, headers=headers)
        response2.raise_for_status()
        explanation = response2.json()['choices'][0]['message']['content']
        logging.info("Kimi generated explanation.")

        # OpenStreetMap: Geocode and display locations (top match for simplicity)
        static_map_url = ""
        try:
            locations = results[0][4] if results and pd.notna(results[0][4]) else ""
            if locations:
                # Clean and encode location string
                locations = urllib.parse.quote(locations.strip().replace('|', ','))
                geocode_url = f"https://nominatim.openstreetmap.org/search?q={locations}&format=json&limit=1"
                headers_osm = {'User-Agent': 'TrialMatchingDemo/1.0 (your_email@example.com)'}  # Replace
                geocode_response = requests.get(geocode_url, headers=headers_osm, timeout=5).json()
                if geocode_response:
                    lat = float(geocode_response[0]['lat'])
                    lng = float(geocode_response[0]['lon'])
                    # Interactive map with fallback
                    m = folium.Map(location=[lat, lng], zoom_start=10, tiles='OpenStreetMap')
                    MarkerCluster().add_to(m)
                    folium.Marker([lat, lng], popup=results[0][1][:100]).add_to(m)
                    st.subheader("Trial Locations Map 📍")
                    folium_static(m, width=700, height=400)
                    # Static map for email
                    static_map_url = f"https://tile.openstreetmap.de/{lat},{lng},10/600x300.png"
                else:
                    st.warning("Could not geocode trial location.")
                    m = folium.Map(location=[0, 0], zoom_start=2, tiles='OpenStreetMap')  # Fallback global map
                    st.subheader("Trial Locations Map 📍")
                    folium_static(m, width=700, height=400)
            else:
                st.warning("No valid location data for trial.")
                m = folium.Map(location=[0, 0], zoom_start=2, tiles='OpenStreetMap')
                st.subheader("Trial Locations Map 📍")
                folium_static(m, width=700, height=400)
        except Exception as e:
            st.warning(f"Could not map trial locations: {str(e)}")
            logging.error(f"Geocoding error: {str(e)}")
            m = folium.Map(location=[0, 0], zoom_start=2, tiles='OpenStreetMap')
            st.subheader("Trial Locations Map 📍")
            folium_static(m, width=700, height=400)

        # Invoke Lambda if email provided
        if email:
            lambda_client = boto3.client(
                'lambda',
                region_name='us-east-1',
                aws_access_key_id='your_aws_access_key',  # Replace
                aws_secret_access_key='your_aws_secret_key'  # Replace
            )
            payload = json.dumps({
                'to_email': email,
                'user_input': user_input,
                'age_group': age_group,
                'sex': sex,
                'ranked_results': ranked_results,
                'explanation': explanation,
                'trials': [{
                    'nct_number': row[0],
                    'title': row[1],
                    'conditions': row[2],
                    'summary': row[3],
                    'distance': row[5]
                } for row in results],
                'static_map_url': static_map_url
            })
            response = lambda_client.invoke(
                FunctionName='send-trial-email',
                InvocationType='RequestResponse',
                Payload=payload
            )
            try:
                result = json.loads(response['Payload'].read().decode('utf-8'))
                if result.get('statusCode') == 200:
                    st.success(f"Formatted report emailed to {email} 📧")
                    logging.info(f"Emailed report to {email}")
                else:
                    st.error(f"Email failed: {result.get('body', 'Unknown error')}")
                    logging.error(f"Lambda invoke failed: {result.get('body', 'Unknown error')}")
            except json.JSONDecodeError:
                st.error("Lambda response invalid.")
                logging.error("Invalid Lambda response JSON.")

        # Display results
        st.subheader("Matching Trials 🧪")
        for row in results:
            st.write(f"**Trial ID**: {row[0]}")
            st.write(f"**Title**: {row[1]}")
            st.write(f"**Conditions**: {row[2]}")
            st.write(f"**Summary**: {row[3]}")
            st.write(f"**Similarity Score**: {row[5]:.4f}")
            st.write("---")
        st.subheader("AI Analysis 🤖")
        st.write(f"**Ranked Results**: {ranked_results}")
        st.write(f"**Why These Match**: {explanation}")
        st.write("**Note**: This is a demo, not medical advice. Consult a doctor. 🚨")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        logging.error(f"Error processing query: {str(e)}")

st.write("Future Enhancement: Blockchain for secure data sharing (e.g., Ethereum smart contracts). 🔗")