import streamlit as st
import pandas as pd
import requests
import json
import base64
import datetime

# Config page
st.set_page_config(
    page_title="Système de surveillance - Détection d'Anomalies",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style
st.markdown("""
<style>
.anomaly-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 20px;
    background-color: #f9f9f9;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Système de surveillance - Détection d'Anomalies")
st.markdown("""
Bienvenue dans l'outil de détection d'anomalies pour systèmes microservices.  
Chargez vos fichiers JSONS (graphes) et CSV (métriques) pour analyser les anomalies potentielles.  
Les résultats incluent des scores, des visualisations et des explications générées par IA.
""")

feature_names = ['disk_u','cpu_r', 'load_1', 'load_5', 'mem_u', 'disk_q', 'disk_r',
                 'disk_w', 'eth1_fi', 'eth1_fo', 'tcp_timeouts']

with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. Chargez un fichier **JSONS** contenant les données des graphes (une ligne par graphe).  
    2. Chargez un fichier **CSV** contenant les métriques système (avec colonnes timestamp, cpu_r, load_1, etc.).  
    3. Choisissez la métrique à prévoir.  
    4. Cliquez sur **Lancer la détection** pour analyser les données.  
    5. Consultez les résultats avec scores, graphiques et explications.
    """)
    st.markdown("---")
    st.info("Assurez-vous que l'API Flask est en cours d'exécution sur `http://127.0.0.1:5000`.")
    st.header("⚙️ Paramètres de prévision")
    metric_to_forecast = st.selectbox("Choisir la métrique à prévoir :", feature_names, index=0)

st.header("1️⃣ Charger les fichiers d'entrée")
col1, col2 = st.columns(2)

with col1:
    jsons_file = st.file_uploader("📄 Fichier JSONS (graphes)", type=["jsons"])
with col2:
    csv_file = st.file_uploader("📈 Fichier CSV (métriques)", type=["csv"])

if jsons_file and csv_file:
    st.success("✅ Les deux fichiers ont été chargés avec succès.")
    if st.button("🚀 Lancer la détection d'anomalies"):
        with st.spinner("🕒 Traitement en cours..."):
            try:
                # Lecture JSONS
                graph_data_list = []
                for line in jsons_file:
                    graph_json = json.loads(line.decode("utf-8"))
                    graph_data_list.append(graph_json)

                # Lecture CSV
                df_metrics = pd.read_csv(csv_file)
                required_columns = ['timestamp'] + feature_names
                if not all(col in df_metrics.columns for col in required_columns):
                    st.error(f"Le fichier CSV doit contenir les colonnes : {', '.join(required_columns)}")
                    st.stop()
                metrics_data = df_metrics.to_dict(orient="records")

                # Appel API Flask sans forecast
                api_input = {
                    "graph_data_list": graph_data_list,
                    "metrics_data": metrics_data,
                    "metric_to_forecast": metric_to_forecast,
                    "include_forecast": False
                }

                response = requests.post("http://127.0.0.1:5000/detect_fused_anomaly", json=api_input, timeout=30)
                if response.status_code == 200:
                    anomalies = response.json().get("anomalies", [])
                    st.session_state.anomalies = anomalies
                    st.session_state.metrics_data = metrics_data
                    if not anomalies:
                        st.info("✅ Aucune anomalie détectée.")
                else:
                    st.error(f"Erreur API : {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Erreur lors du traitement : {str(e)}")

if 'anomalies' in st.session_state and st.session_state.anomalies:
    st.header("2️⃣ Résultats des anomalies détectées")
    st.markdown(f"**Nombre d'anomalies détectées : {len(st.session_state.anomalies)}**")
    for anomaly in st.session_state.anomalies:
        with st.container():
            st.markdown('<div class="anomaly-card">', unsafe_allow_html=True)
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

            with col_right:
                st.image(f"data:image/png;base64,{anomaly['plot_base64']}", use_column_width=True, caption="Visualisation de l'anomalie")

            st.markdown("**💡 Explication de l'anomalie :**")
            st.write(anomaly['explanation'])

            # Générer et afficher le forecast séparément
            try:
                forecast_input = {
                    "timestamp": anomaly['timestamp'],
                    "metric_to_forecast": metric_to_forecast,
                    "metrics_data": st.session_state.metrics_data,
                    "trace_id": anomaly['trace_id']
                }
                response_forecast = requests.post("http://127.0.0.1:5000/generate_forecast_plot", json=forecast_input, timeout=30)
                if response_forecast.status_code == 200:
                    forecast_data = response_forecast.json()
                    forecast_img = forecast_data.get("forecast_plot_base64")
                    if forecast_img:
                        st.markdown(f"**📈 Prévision pour la métrique : {forecast_data['forecast']['metric']}**")
                        st.image(f"data:image/png;base64,{forecast_img}", use_column_width=True)
            except Exception as e:
                st.warning(f"Erreur lors de la génération de la prévision : {str(e)}")

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

elif 'anomalies' in st.session_state and not st.session_state.anomalies:
    st.info("✅ Aucune anomalie détectée.")
else:
    st.warning("⚠️ Veuillez charger les deux fichiers (JSONS et CSV) pour continuer.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
Développé avec Streamlit pour la détection d'anomalies multimodale.  
© 2025 - Projet DeepTraLog
</div>
""", unsafe_allow_html=True)