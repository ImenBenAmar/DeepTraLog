import streamlit as st
import pandas as pd
import json
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="Multisource Anomaly Detection Dashboard", layout="wide")
st.title("🔍 Multisource Anomaly Detection Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.header("Graph (GGNN + Deep SVDD)")
    uploaded_file = st.file_uploader("Uploader un fichier .jsons de graphe", type=["json", "jsons"], key="file_graph")

    if uploaded_file:
        try:
            # lecture ligne par ligne
            lines = uploaded_file.read().decode("utf-8").splitlines()
            data_json_list = [json.loads(line) for line in lines if line.strip()]
        except Exception as e:
            st.error(f"Erreur chargement JSON: {e}")
            data_json_list = []

        if st.button("🧪 Lancer la détection DeepTraLog"):
            try:
                response = requests.post("http://localhost:5000/detect_anomaly", json=data_json_list)
                if response.status_code == 200:
                    st.session_state['deeptralog_results'] = response.json()
                else:
                    st.error(f"Erreur API DeepTraLog : {response.status_code} {response.text}")
            except Exception as e:
                st.error(f"Exception lors de l'appel API DeepTraLog : {e}")

    if 'deeptralog_results' in st.session_state:
        results = st.session_state['deeptralog_results']
        for i, res in enumerate(results):
            st.markdown(f"### Graphe {i+1} — Service : {res['service_name']} — Trace ID : {res['trace_id']}")
            st.markdown(f"Anomaly Score: {res['anomaly_score']:.6f} — Label: {'Anomalie' if res['label'] else 'Normal'}")
            if res['label'] == 1:
                st.error(f"⚠️ Le service **{res['service_name']}** a causé un problème avec la trace **{res['trace_id']}**")
            else:
                st.success("✅ Pas d'anomalies détectées dans ce graphe.")

with col2:
    st.header("Metric System")
    csv_file = st.file_uploader("Uploader un fichier CSV séries temporelles (timestamp + métriques)", type="csv", key="file_metric")

    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            st.write("Aperçu des données :", df.head())

            feature_names = ['cpu_r', 'load_1', 'load_5', 'mem_u', 'disk_q',
                             'disk_r', 'disk_w', 'disk_u', 'eth1_fi', 'eth1_fo', 'tcp_timeouts']

            if not all(f in df.columns for f in feature_names):
                st.error(f"Le fichier CSV doit contenir les colonnes suivantes: {feature_names}")
            else:
                data_dict = df[feature_names].to_dict(orient="records")

                if st.button("🧪 Lancer détection XGBoost"):
                    try:
                        response = requests.post("http://localhost:5000/batch_detect_xgb_anomaly", json={"data": data_dict})
                        if response.status_code == 200:
                            res = response.json()
                            df["anomaly"] = res.get("anomaly_labels", [0]*len(df))
                            st.session_state['xgb_df'] = df  # stocker df avec anomalies
                        else:
                            st.error(f"Erreur API XGBoost : {response.status_code} {response.text}")
                    except Exception as e:
                        st.error(f"Exception lors de la détection XGBoost : {e}")

        except Exception as e:
            st.error(f"Erreur lecture CSV : {e}")

    if 'xgb_df' in st.session_state:
        df = st.session_state['xgb_df']
        n_anomalies = df['anomaly'].sum()
        st.markdown(f"**Nombre d'anomalies détectées : {n_anomalies} / {len(df)}**")
        st.markdown("### Visualisation des métriques avec anomalies (en rouge)")

        feature_names = ['cpu_r', 'load_1', 'load_5', 'mem_u', 'disk_q',
                         'disk_r', 'disk_w', 'disk_u', 'eth1_fi', 'eth1_fo', 'tcp_timeouts']

        for feature in feature_names:
            plt.figure(figsize=(15, 4))
            plt.plot(df["timestamp"], df[feature], label='Normal', color='blue')
            anomalies_points = df[df["anomaly"] == 1]
            plt.scatter(anomalies_points["timestamp"], anomalies_points[feature],
                        color='red', label='Anomalie', s=20)
            plt.title(f'Série temporelle - {feature}')
            plt.xlabel('Timestamp')
            plt.ylabel(feature)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()

        st.markdown("### Score d'anomalie dans le temps")
        color_map = df["anomaly"].map({0: "green", 1: "red"})
        plt.figure(figsize=(12, 3))
        plt.bar(df["timestamp"], df["anomaly"], color=color_map, width=0.8)
        plt.ylim(-0.1, 1.1)
        plt.yticks([0,1], ["Normal", "Anomalie"])
        plt.xlabel("Timestamp")
        plt.grid(axis="x")
        st.pyplot(plt.gcf())
        plt.clf()
