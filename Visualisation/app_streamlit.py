
import streamlit as st
import pandas as pd
import requests
import json
import base64
from io import StringIO

st.set_page_config(layout="wide")
st.title("🧠 Fusion Multimodale - Détection d'anomalies")

# --- Upload fichiers ---
st.header("1️⃣ Charger les fichiers d'entrée")

jsons_file = st.file_uploader("📄 Fichier JSONS (graphes)", type=["jsons"])
csv_file = st.file_uploader("📈 Fichier CSV (métriques)", type=["csv"])

if jsons_file and csv_file:
    st.success("Les deux fichiers ont été chargés.")

    if st.button("🚀 Lancer la détection d'anomalies"):
        with st.spinner("Traitement en cours..."):

            # Lire les graphes
            graph_data_list = []
            for line in jsons_file:
                graph_json = json.loads(line.decode("utf-8"))
                graph_data_list.append(graph_json)

            # Lire les métriques
            df_metrics = pd.read_csv(csv_file)
            metrics_data = df_metrics.to_dict(orient="records")

            # Construire la requête
            api_input = {
                "graph_data_list": graph_data_list,
                "metrics_data": metrics_data
            }

            # Envoyer à l'API Flask (modifie si API est sur un serveur)
            response = requests.post("http://127.0.0.1:5000/detect_fused_anomaly", json=api_input)

            if response.status_code == 200:
                anomalies = response.json()["anomalies"]

                if len(anomalies) == 0:
                    st.info("✅ Aucune anomalie détectée.")
                else:
                    st.subheader("2️⃣ Résultats des anomalies détectées")
                    for anomaly in anomalies:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"""
                            **🧾 Trace ID**: `{anomaly['trace_id']}`  
                            **⏱️ Timestamp**: `{anomaly['timestamp']}`  
                            **📛 Service**: `{anomaly['service_name']}`  
                            **📊 Score fusionné**: `{anomaly['fused_score']:.3f}`  
                            """)
                        with col2:
                            img_data = anomaly['plot_base64']
                            st.image(f"data:image/png;base64,{img_data}", use_column_width=True)

            else:
                st.error(f"Erreur API: {response.status_code} - {response.text}")
