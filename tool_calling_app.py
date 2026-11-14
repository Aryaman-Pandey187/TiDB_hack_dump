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
import api_tools
from openai import OpenAI
import time

# Setup logging
logging.basicConfig(
    filename='/home/ubuntu/app.log',
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
        h2, h3 {color: #34495e;background-color: transparent !important;}
        .log-message {background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin: 5px 0; color: #2c3e50;}
        .success {color: #27ae60;}
        .warning {color: #e67e22;}
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
#st.title("Clinical Trial Matching Demo 🩺")
#st.write("Enter your health profile to find matching trials. This is a demo, not medical advice.")

# Sidebar
st.sidebar.title("About")
st.sidebar.write("🧬 AI-powered trial matching using TiDB, Kimi AI, and AWS.")
st.sidebar.write("📍 Maps powered by OpenStreetMap.")
st.sidebar.markdown("**External Tools**: PubMed, RxNorm, MeSH, OpenFDA")
st.sidebar.write("🔮 Future: Blockchain for secure data sharing.")

st.sidebar.title("Navigation 📋")
st.sidebar.markdown("""
    <style>
        .sidebar .stButton>button {
            width: 100%;
            margin: 5px 0;
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            text-align: left;
            padding: 10px;
        }
        .sidebar .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar buttons for navigation
st.sidebar.button("Home 🏠", on_click=lambda: st.session_state.update({'page': 'home'}))
st.sidebar.button("How to Use 😎", on_click=lambda: st.session_state.update({'page': 'howto'}))
st.sidebar.button("Tech Behind It 👩‍💻", on_click=lambda: st.session_state.update({'page': 'tech'}))
st.sidebar.button("Contact Me 🤳", on_click=lambda: st.session_state.update({'page': 'contact'}))

# Page content based on button click
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

if st.session_state['page'] == 'home':
    st.title("MedMatch: Unleashing AI to Connect You with Life-Changing Clinical Trials! 🩺")
    st.write("Enter your health profile to find matching clinical trials. Demo only, not medical advice.")
    # Input form (keep existing form code here)
elif st.session_state['page'] == 'howto':
    st.title("How to Use This Tool 📝")
    st.write("Welcome! This tool helps you find clinical trials that match your health needs. Here’s a simple guide:")
    st.write("- **Step 1**: Type your symptoms (e.g., 'diabetes fatigue') in the 'Symptoms/Conditions' box.")
    st.write("- **Step 2**: Pick your age group and sex from the dropdowns.")
    st.write("- **Step 3**: (Optional) Add your email to get a detailed report.")
    st.write("- **Step 4**: Click 'Find Trials' and wait for results to load with progress updates.")
    st.write("- **Step 5**: Check the trial list, map, and explanation. Explore the map or email for more!")
    st.write("It’s easy and safe—try it with any health concern!")
elif st.session_state['page'] == 'tech':
    st.title("Tech Behind the Tool 💻")
    st.write("This project uses cool tech to help you. Here’s a simple breakdown:")
    st.write("- **TiDB**: A fast database that stores trial info and finds matches using smart math.")
    st.write("- **Kimi AI**: A smart assistant that ranks trials and explains them in easy words.")
    st.write("- **AWS**: A cloud service that runs the email system and keeps everything online.")
    st.write("- **External Tools**: Kimi uses online helpers like PubMed (research articles), RxNorm (drug info), MeSH (medical terms), and OpenFDA (safety checks) to add more details.")
    st.write("- **How They Work Together**: TiDB finds trials, Kimi smartly picks the best ones with help from online tools, AWS sends emails, and a map shows locations—all working as a team!")
elif st.session_state['page'] == 'contact':
    st.title("Contact Me 📧")
    st.write("I’d love to hear from you! Here’s how to reach me:")
    st.write("- **LinkedIn**: [Let's Connect](https://www.linkedin.com/in/aryaman-pandey/)")
    st.write("- **GitHub**: [View my work](https://github.com/Aryaman-Pandey187)")
    st.write("- **Email**: pandeyaryaman187@gmail.com")
    st.write("Feel free to connect for questions or feedback!")

# Log area
log_area = st.empty()

# Input form
user_input = st.text_area("Symptoms/Conditions (e.g., 'diabetes fatigue') 💊", "diabetes symptoms fatigue")
age_group = st.selectbox("Age Group 🎂", ["ADULT", "OLDER_ADULT", "CHILD", "ALL"])
sex = st.selectbox("Sex 🚻", ["ALL", "MALE", "FEMALE"])
email = st.text_input("Email (optional, to receive results directly in your inbox; this process may take up to a minute to complete) 📧", "")  # Optional email
if st.button("Find Trials 🔍"):
    try:
        if not user_input.strip():
            raise ValueError("Please enter symptoms or conditions.")

        logging.info(f"Processing query: {user_input}, Age: {age_group}, Sex: {sex}, Email: {email}")

        # Log start
        log_area.markdown(f'<div class="log-message"><span class="success">🚀 Starting trial search at {time.strftime("%H:%M:%S")}</span></div>', unsafe_allow_html=True)
        time.sleep(1)  # Simulate delay for demo

        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        log_area.markdown(f'<div class="log-message"><span class="success">🧠 Loading AI model...</span></div>', unsafe_allow_html=True)
        time.sleep(1)

        # Generate query embedding
        query_embedding = model.encode(user_input).tolist()
        if len(query_embedding) != 384:
            raise ValueError("Embedding dimension mismatch.")
        query_embedding_json = json.dumps(query_embedding)

        log_area.markdown(f'<div class="log-message"><span class="success">📊 Generating query embedding...</span></div>', unsafe_allow_html=True)
        time.sleep(1)

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
        log_area.markdown(f'<div class="log-message"><span class="success">🔍 Searching TiDB database...</span></div>', unsafe_allow_html=True)
        time.sleep(1)

        # Vector search with filters
        sql = """
            SELECT nct_number, study_title, conditions, brief_summary, locations, interventions,
            VEC_COSINE_DISTANCE(embedding, %s) AS distance
            FROM clinical_trials_latest
            WHERE sex = %s AND age LIKE %s AND VEC_COSINE_DISTANCE(embedding, %s) < 0.5
            ORDER BY distance ASC
            LIMIT 5
        """
        cursor.execute(sql, (query_embedding_json, sex, f'%{age_group}%', query_embedding_json))
        results = cursor.fetchall()

        if not results:
            log_area.markdown(f'<div class="log-message"><span class="warning">⚠ No matching trials found for "{user_input}". Try different symptoms.</span></div>', unsafe_allow_html=True)
            st.stop()  # Halt execution if no results
            # raise ValueError("No matching trials found.")
        cursor.close()
        conn.close()

        log_area.markdown(f'<div class="log-message"><span class="success">✅ Found {len(results)} trials!</span></div>', unsafe_allow_html=True)
        time.sleep(1)

        # logging.info(f"Found {len(results)} trial matches.")

        # Kimi LLM chaining with tool-calling
        client = OpenAI(
            api_key='sk-',  # Replace
            base_url='https://api.moonshot.ai/v1'
        )

        # Define tools for Kimi
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'search_pubmed',
                    'description': 'Search PubMed for articles on a query.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string', 'description': 'Search query.'},
                            'num_results': {'type': 'integer', 'description': 'Number of results.', 'default': 5}
                        },
                        'required': ['query']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'search_rxnorm',
                    'description': 'Map drug to RxNorm codes and therapeutic classes.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'drug_name': {'type': 'string', 'description': 'Drug name to map.'}
                        },
                        'required': ['drug_name']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'search_mesh',
                    'description': 'Link term to MeSH headings and count matches.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'term': {'type': 'string', 'description': 'Medical term to search.'}
                        },
                        'required': ['term']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'search_openfda',
                    'description': 'Fetch OpenFDA adverse events and stats for a drug.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'drug_name': {'type': 'string', 'description': 'Drug name for adverse events.'},
                            'limit': {'type': 'integer', 'description': 'Number of events.', 'default': 5}
                        },
                        'required': ['drug_name']
                    }
                }
            }
        ]

        results_json = json.dumps([{
            'nct_number': row[0],
            'title': row[1],
            'conditions': row[2],
            'summary': row[3],
            'interventions': row[5],
            'distance': row[6]
        } for row in results])

        # Kimi call: Generate API queries and rank trials
        log_area.markdown(f'<div class="log-message"><span class="success">🤖 Contacting Kimi AI for ranking...</span></div>', unsafe_allow_html=True)
        time.sleep(1)

        # Reset messages for each run
        messages = [
            {'role': 'system', 'content': 'You are Kimi, an AI assistant provided by Moonshot AI.'},
            {'role': 'user', 'content': (
                f"For query '{user_input}' for a {age_group} {sex} patient, generate search parameters for PubMed, RxNorm, MeSH, and OpenFDA APIs to enrich trial data. "
                f"Trials: {results_json}. "
                "Provide one query per API in JSON format (e.g., {'api': 'search_pubmed', 'query': '...', 'num_results': 5}). "
                "Then rank trials based on relevance, using API data if available. Return ranking and queries."
            )}
        ]


        completion = client.chat.completions.create(
            model='kimi-k2-0905-preview',
            messages=messages,
            temperature=0.6
        )

        response_data = completion.choices[0].message
        ranked_results = ""
        api_results = {}
        if response_data.content:
            ranked_results += response_data.content + "\n"

        try:
            # Parse Kimi response for API queries
            content = json.loads(response_data.content) if response_data.content else {}
            queries = content.get('queries', [])
            ranked_results = content.get('ranking', "No ranking provided.")
            log_area.markdown(f'<div class="log-message"><span class="success">📝 Kimi generated ranking and API queries...</span></div>', unsafe_allow_html=True)
            logging.info(f"Kimi generated API queries: {json.dumps(queries)}")
        except json.JSONDecodeError:
            queries = []
            ranked_results = response_data.content or "No ranking provided."
            log_area.markdown(f'<div class="log-message"><span class="warning">⚠ Failed to parse Kimi response...</span></div>', unsafe_allow_html=True)
            logging.error("Failed to parse Kimi response as JSON.")

        # Execute API calls
        for query in queries:
            api_name = query.get('api')
            log_area.markdown(f'<div class="log-message"><span class="success">🌐 Fetching data from {api_name}...</span></div>', unsafe_allow_html=True)
            time.sleep(1)
            try:
                if api_name == 'search_pubmed':
                    result = api_tools.search_pubmed(query.get('query', user_input), query.get('num_results', 5))
                elif api_name == 'search_rxnorm':
                    result = api_tools.search_rxnorm(query.get('drug_name', ''))
                elif api_name == 'search_mesh':
                    result = api_tools.search_mesh(query.get('term', user_input))
                elif api_name == 'search_openfda':
                    result = api_tools.search_openfda(query.get('drug_name', ''), query.get('limit', 5))
                else:
                    result = {'error': f'Unknown API: {api_name}'}
                api_results[api_name] = result
                log_area.markdown(f'<div class="log-message"><span class="success">✅ {api_name} data retrieved!</span></div>', unsafe_allow_html=True)

                logging.info(f"API call {api_name}: {json.dumps(result)}")
            except Exception as e:
                api_results[api_name] = {'error': str(e)}
                log_area.markdown(f'<div class="log-message"><span class="warning">⚠ {api_name} failed: {str(e)}</span></div>', unsafe_allow_html=True)
                logging.error(f"API call error for {api_name}: {str(e)}")

        # Kimi call: Rank with API results
        log_area.markdown(f'<div class="log-message"><span class="success">🤖 Finalizing ranking with API data...</span></div>', unsafe_allow_html=True)
        time.sleep(1)
        messages = [
            {'role': 'system', 'content': 'You are Kimi, an AI assistant provided by Moonshot AI.'},
            {'role': 'user', 'content': (
                f"Rank trials for '{user_input}' ({age_group} {sex}) using: {results_json}. "
                f"API results: {json.dumps(api_results)}. "
                "Incorporate PubMed research, RxNorm drug mappings, MeSH terms, and OpenFDA safety data."
            )}
        ]

        # Re-call Kimi with tool results for final ranking
        completion = client.chat.completions.create(
            model='kimi-k2-0905-preview',
            messages=messages,
            temperature=0.6
        )

        ranked_results = completion.choices[0].message.content or "No ranking provided after API calls."
        log_area.markdown(f'<div class="log-message"><span class="success">🎉 Ranking complete!</span></div>', unsafe_allow_html=True)
        logging.info(f"Kimi ranked results with API data: {ranked_results}")

        log_area.markdown(f'<div class="log-message"><span class="success">🤖 Generating explanation...</span></div>', unsafe_allow_html=True)
        time.sleep(1)

        messages = [  # Reset for explanation
            {'role': 'system', 'content': 'You are Kimi, an AI assistant provided by Moonshot AI.'},
            {'role': 'user', 'content': (
                "Explain why these ranked trials "
                f"({ranked_results}) match the query '{user_input}' for a {age_group} {sex} patient. "
                "Incorporate tool data (PubMed, RxNorm, MeSH, OpenFDA). "
                "Use simple, everyday language that a non-expert patient can understand. Avoid medical jargon (e.g., T2DM, MeSH, RxNorm, OpenFDA) and technical terms (e.g., r = 0.62, cytokine-release). Focus on clear benefits, risks, and why the trial fits their needs. Keep it short and friendly.If matches are poor, say 'No good matches found'"
            )}
        ]
        completion = client.chat.completions.create(
            model='kimi-k2-0905-preview',
            messages=messages,
            temperature=0.6
        )

        explanation = completion.choices[0].message.content or "No explanation provided."
        log_area.markdown(f'<div class="log-message"><span class="success">✅ Explanation ready!</span></div>', unsafe_allow_html=True)
        logging.info(f"Kimi generated explanation: {explanation}")

        # Clear logs and display results
        log_area.empty()
        #st.subheader("Trial Results 🧪")
        st.markdown("<h2 style='color: #ffffff;'><u>Matching Trials</u> 🧪</h2>", unsafe_allow_html=True)
        for row in results:
            st.markdown(f"<p style='margin: 10px 0;'><b>Trial ID:</b> {row[0]}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin: 10px 0;'><b>Title:</b> {row[1]}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin: 10px 0;'><b>Conditions:</b> {row[2]}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin: 10px 0;'><b>Summary:</b> {row[3]}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin: 10px 0;'><b>Similarity Score:</b> {row[6]:.4f}</p>", unsafe_allow_html=True)
            st.markdown("<hr style='border: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #ffffff;'><u>Trial Location </u>📍</h2>", unsafe_allow_html=True)
        static_map_url = ""
        try:
            locations = results[0][4] if results and pd.notna(results[0][4]) else ""
            if locations:
                locations = urllib.parse.quote(locations.strip().replace('|', ','))
                geocode_url = f"https://nominatim.openstreetmap.org/search?q={locations}&format=json&limit=1"
                headers_osm = {'User-Agent': 'TrialMatchingDemo/1.0 (your_email@example.com)'}  # Replace
                geocode_response = requests.get(geocode_url, headers=headers_osm, timeout=5).json()
                if geocode_response:
                    lat = float(geocode_response[0]['lat'])
                    lng = float(geocode_response[0]['lon'])
                    m = folium.Map(location=[lat, lng], zoom_start=10, tiles='OpenStreetMap')
                    MarkerCluster().add_to(m)
                    folium.Marker([lat, lng], popup=results[0][1][:100]).add_to(m)
                    folium_static(m, width=700, height=400)
                    static_map_url = f"https://tile.openstreetmap.de/{lat},{lng},10/600x300.png"
                else:
                    st.warning("Could not geocode trial location.")
                    m = folium.Map(location=[0, 0], zoom_start=2, tiles='OpenStreetMap')
                    folium_static(m, width=700, height=400)
            else:
                st.warning("No valid location data for trial.")
                m = folium.Map(location=[0, 0], zoom_start=2, tiles='OpenStreetMap')
                folium_static(m, width=700, height=400)
        except Exception as e:
            st.warning(f"Could not map trial locations: {str(e)}")
            logging.error(f"Geocoding error: {str(e)}")
            m = folium.Map(location=[0, 0], zoom_start=2, tiles='OpenStreetMap')
            folium_static(m, width=700, height=400)
        st.markdown("<h2 style='color: #ffffff;'><u>AI Analysis </u> 🤖</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin: 10px 0;'><b>Ranked Results:</b> {ranked_results}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin: 10px 0;'><b>Why These Match:</b> {explanation}</p>", unsafe_allow_html=True)
        st.markdown("<p style='margin: 10px 0; color: #e74c3c;'><i>**Note**: This is a demo, not medical advice. Consult a doctor.</i></p>", unsafe_allow_html=True)

        # Invoke Lambda if email provided
        if email:
            lambda_client = boto3.client(
                'lambda',
                region_name='us-east-1',  # Replace if different
                aws_access_key_id='',  # Replace
                aws_secret_access_key='f'  # Replace
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
                    'distance': row[6]
                } for row in results],
                'static_map_url': static_map_url
            })
            log_area.markdown(f'<div class="log-message"><span class="success">📧 Sending email report...</span></div>', unsafe_allow_html=True)
            time.sleep(1)
            response = lambda_client.invoke(
                FunctionName='TiDB_hackathon_Email_trigger_lambda',
                InvocationType='RequestResponse',
                Payload=payload
            )
            try:
                result = json.loads(response['Payload'].read().decode('utf-8'))
                if result:
                    log_area.markdown(f'<div class="log-message"><span class="success">✅ Email sent to {email}!</span></div>', unsafe_allow_html=True)
                    st.success(f"Formatted report emailed to {email}📧")
                    logging.info(f"Emailed report to {email}")
                else:
                    st.error(f"Email failed: {result.get('body', 'Unknown error')}")
                    logging.error(f"Lambda invoke failed: {result.get('body', 'Unknown error')}")
            except json.JSONDecodeError:
                st.error("Lambda response invalid.")
                logging.error("Invalid Lambda response JSON.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        logging.error(f"Error processing query: {str(e)}")

st.write("Future Enhancement: Blockchain for secure data sharing (e.g., Ethereum smart contracts). 🔗")
