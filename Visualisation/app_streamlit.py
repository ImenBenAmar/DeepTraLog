import streamlit as st
import pandas as pd
import requests
import json
import base64
from io import BytesIO
import datetime

# Configure page layout and title
st.set_page_config(
    page_title="systéme de surveillance - Détection d'Anomalies",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main { padding: 20px; }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stFileUploader {
        border: 2px dashed #007bff;
        border-radius: 5px;
        padding: 10px;
    }
    .anomaly-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
    }
    .metric-table th {
        background-color: #007bff;
        color: white;
        padding: 8px;
    }
    .metric-table td {
        padding: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🧠systéme de surveillance - Détection d'Anomalies")
st.markdown("""
Bienvenue dans l'outil de détection d'anomalies pour systèmes microservices.  
Chargez vos fichiers JSONS (graphes) et CSV (métriques) pour analyser les anomalies potentielles.  
Les résultats incluent des scores, des visualisations et des explications générées par IA.
""")

# Sidebar for instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. Chargez un fichier **JSONS** contenant les données des graphes (une ligne par graphe).
    2. Chargez un fichier **CSV** contenant les métriques système (avec colonnes timestamp, cpu_r, load_1, etc.).
    3. Cliquez sur **Lancer la détection** pour analyser les données.
    4. Consultez les résultats avec scores, graphiques et explications.
    """)
    st.markdown("---")
    st.info("Assurez-vous que l'API Flask est en cours d'exécution sur `http://127.0.0.1:5000`.")

# --- Upload files ---
st.header("1️⃣ Charger les fichiers d'entrée")
col1, col2 = st.columns(2)

with col1:
    jsons_file = st.file_uploader("📄 Fichier JSONS (graphes)", type=["jsons"], key="jsons_uploader")
with col2:
    csv_file = st.file_uploader("📈 Fichier CSV (métriques)", type=["csv"], key="csv_uploader")

# Check if both files are uploaded
if jsons_file and csv_file:
    st.success("✅ Les deux fichiers ont été chargés avec succès.")
    
    # Button to trigger anomaly detection
    if st.button("🚀 Lancer la détection d'anomalies", key="detect_button"):
        with st.spinner("🕒 Traitement en cours..."):
            try:
                # Read graph data from JSONS file
                graph_data_list = []
                for line in jsons_file:
                    try:
                        graph_json = json.loads(line.decode("utf-8"))
                        graph_data_list.append(graph_json)
                    except json.JSONDecodeError as e:
                        st.error(f"Erreur de lecture du fichier JSONS : {str(e)}")
                        st.stop()

                # Read metrics data from CSV file
                try:
                    df_metrics = pd.read_csv(csv_file)
                    required_columns = ['timestamp', 'cpu_r', 'load_1', 'load_5', 'mem_u', 'disk_q', 
                                     'disk_r', 'disk_w', 'disk_u', 'eth1_fi', 'eth1_fo', 'tcp_timeouts']
                    if not all(col in df_metrics.columns for col in required_columns):
                        st.error(f"Le fichier CSV doit contenir les colonnes : {', '.join(required_columns)}")
                        st.stop()
                    metrics_data = df_metrics.to_dict(orient="records")
                except Exception as e:
                    st.error(f"Erreur de lecture du fichier CSV : {str(e)}")
                    st.stop()

                # Prepare API request
                api_input = {
                    "graph_data_list": graph_data_list,
                    "metrics_data": metrics_data
                }

                # Send request to Flask API
                try:
                    response = requests.post("http://127.0.0.1:5000/detect_fused_anomaly", json=api_input, timeout=30)
                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur de connexion à l'API Flask : {str(e)}")
                    st.stop()

                # Process API response
                if response.status_code == 200:
                    anomalies = response.json().get("anomalies", [])
                    
                    if not anomalies:
                        st.info("✅ Aucune anomalie détectée dans les données fournies.")
                    else:
                        st.header("2️⃣ Résultats des anomalies détectées")
                        st.markdown(f"**Nombre d'anomalies détectées : {len(anomalies)}**")

                        # Display anomalies in a structured format
                        for anomaly in anomalies:
                            with st.container():
                                st.markdown('<div class="anomaly-card">', unsafe_allow_html=True)
                                
                                # Layout haut : 2 colonnes
                                col_left, col_right = st.columns([1, 2])
                                
                                with col_left:
                                    st.subheader(f"Trace ID: {anomaly['trace_id']}")
                                    timestamp = datetime.datetime.fromtimestamp(anomaly['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                                    st.markdown(f"""
                                    **⏱️ Timestamp** : {timestamp}  
                                    **📛 Service** : {anomaly['service_name']}  
                                    **📊 Score trace** : {anomaly['score_trace']:.4f}  
                                    **📊 Score métrique** : {anomaly['score_metric']:.4f}  
                                    **📊 Score fusionné** : {anomaly['fused_score']:.4f}  
                                    """)
                                    
                                    # Metrics table
                                    metrics_dict = anomaly.get('explanation_metrics', anomaly.get('metrics_row', {}))
                                    if metrics_dict:
                                        metrics_df = pd.DataFrame([metrics_dict]).round(4)
                                        st.markdown("**Métriques système :**")
                                        st.table(metrics_df)

                                with col_right:
                                    img_data = anomaly['plot_base64']
                                    st.image(f"data:image/png;base64,{img_data}", use_column_width=True, caption="Visualisation de l'anomalie")
                                
                                # ✅ Partie basse en pleine largeur : explication
                                st.markdown("**💡 Explication de l'anomalie :**")
                                st.write(anomaly['explanation'])

                                st.markdown('</div>', unsafe_allow_html=True)
                                st.markdown("---")

                else:
                    st.error(f"Erreur API : {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erreur lors du traitement : {str(e)}")

else:
    st.warning("⚠️ Veuillez charger les deux fichiers (JSONS et CSV) pour continuer.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
Développé avec Streamlit pour la détection d'anomalies multimodale.  
© 2025 - Projet DeepTraLog
</div>
""", unsafe_allow_html=True)