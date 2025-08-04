from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
import joblib
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from torch_geometric.data import Data, Batch
from model_definitions import GGNN
from sklearn.preprocessing import MinMaxScaler
import subprocess
import gc
import os
from dotenv import load_dotenv
from mistralai import Mistral  # Updated import to use Mistral client

# Charger les variables d’environnement (.env)
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")
print("MISTRAL_API_KEY =", repr(mistral_api_key))

mistral_client = Mistral(api_key=mistral_api_key) 
gc.collect()
torch.cuda.empty_cache()

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('../models/ggnn_Deep_svdd .pth', map_location='cpu')
model = GGNN(num_node_features=306, hidden_channels=300, num_edge_types=3, num_iterations=2).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

c = checkpoint['center'].to(device)
R = checkpoint['radius'].to(device)

event_embeddings = np.load('../models/event_embeddings (1).npy', allow_pickle=True).item()
service_mapping = dict(pd.read_csv('../DeepTralog/id_service.csv', header=None, names=['ServiceId', 'Service']).values)

scaler = joblib.load('../models/scaler.pkl')
scaler_xG = MinMaxScaler()
xgb_model = joblib.load("../models/XG_metric_detector.pkl")

feature_names = ['cpu_r', 'load_1', 'load_5', 'mem_u', 'disk_q', 'disk_r',
                 'disk_w', 'disk_u', 'eth1_fi', 'eth1_fo', 'tcp_timeouts']
df_train = pd.read_csv("../Dataset_SMD/machine-1-1_train_filtered.csv")
scaler_xG.fit(df_train[feature_names])

def process_json_graph(data_json):
    edge_index = torch.tensor(data_json['edge_index'], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(data_json['edge_attr'], dtype=torch.long)
    node_info = np.array(data_json['node_info'], dtype=np.float32)
    template_ids = node_info[:, 0].astype(int)
    service_ids = node_info[:, 1].astype(int)
    numerical_features = node_info[:, 1:]
    numerical_features = scaler.transform(numerical_features)

    event_embs = np.array([
        event_embeddings.get(template_id, np.zeros(300, dtype=np.float32))
        for template_id in template_ids
    ])

    node_features = np.hstack([numerical_features, event_embs])
    node_features = torch.tensor(node_features, dtype=torch.float)

    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr).to(device)
    return graph, int(service_ids[0])

def generate_llm_explanation(timestamp, service_name, score_trace, score_metric, fused_score, metrics_row):
    prompt = f"""
Voici une anomalie détectée dans un système microservices sachant que Score trace > 0 est considére comme une anomalie et score métrique proche de 1 est considéré comme une anomalie  :

- Timestamp : {timestamp}
- Service concerné : {service_name}
- Score trace : {score_trace:.4f}
- Score métrique : {score_metric:.4f}
- Score fusionné : {fused_score:.4f}
- Valeurs des métriques système : {metrics_row.to_dict()}

Donne une explication probable de la cause de l’anomalie et propose une recommandation technique concrète pour la corriger directement .
    """

    response = mistral_client.chat.complete(
        model="mistral-medium",  
        messages=[
            {"role": "system", "content": "Tu es un assistant DevOps expert en surveillance de systèmes distribués."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=600
    )

    return response.choices[0].message.content.strip()

@app.route('/detect_fused_anomaly', methods=['POST'])
def detect_fused_anomaly():
    try:
        req_data = request.get_json()

        if "graph_data_list" not in req_data or "metrics_data" not in req_data:
            return jsonify({"error": "Missing 'graph_data_list' or 'metrics_data' in request"}), 400

        graph_data_list = req_data["graph_data_list"]
        metrics_data = pd.DataFrame(req_data["metrics_data"])

        if not all(f in metrics_data.columns for f in ["timestamp"] + feature_names):
            return jsonify({"error": "Missing required columns in metrics_data"}), 400

        metrics_data["timestamp"] = pd.to_datetime(metrics_data["timestamp"], unit='s')
        metrics_data = metrics_data.sort_values("timestamp").set_index("timestamp")

        graphs, service_ids, trace_ids, timestamps = [], [], [], []

        for gjson in graph_data_list:
            graph, service_id = process_json_graph(gjson)
            graphs.append(graph)
            service_ids.append(service_id)
            trace_ids.append(gjson.get("trace_id", "unknown_trace"))
            timestamps.append(pd.to_datetime(gjson["timestamp"], unit='s'))

        batch = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            embeddings = model(batch)
            distances = torch.sum((embeddings - c) ** 2, dim=1)
            scores_trace = distances - R ** 2
            labels_trace = (scores_trace > 0).int().tolist()

        metric_labels, metric_scores = [], []

        for ts in timestamps:
            nearest_pos = metrics_data.index.get_indexer([ts], method='nearest')[0]
            if nearest_pos == -1:
                metric_labels.append(0)
                metric_scores.append(0.0)
                continue

            row = metrics_data.iloc[[nearest_pos]]
            X_scaled = scaler_xG.transform(row[feature_names])
            label = int(xgb_model.predict(X_scaled)[0])
            score = float(xgb_model.predict_proba(X_scaled)[0][1])
            metric_labels.append(label)
            metric_scores.append(score)

        results = []

        for i, trace_id in enumerate(trace_ids):
            score_trace = scores_trace[i].item()
            score_metric = metric_scores[i]
            fused_score = 0.5 * score_metric + 0.5 * min(1.0, max(0.0, score_trace / 100))
            label_fused = int(fused_score >= 0.49)

            if label_fused == 1:
                ts = timestamps[i]
                row = metrics_data.loc[ts] if ts in metrics_data.index else metrics_data.iloc[0]

                fig, ax = plt.subplots(figsize=(10, 4))
                for col in feature_names:
                    ax.plot(metrics_data.index, metrics_data[col], alpha=0.3, label=col)
                ax.axvline(ts, color='red', linestyle='--', label='Anomaly')

                ax.set_title(f"Anomaly at {ts} - Trace {trace_id}")
                ax.legend(loc='upper right')
                ax.set_xlabel("Timestamp")
                ax.set_ylabel("Metric value")

                buf = BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()

                explanation = generate_llm_explanation(
                    timestamp=ts,
                    service_name=service_mapping.get(service_ids[i], f"unknown_{service_ids[i]}"),
                    score_trace=score_trace,
                    score_metric=score_metric,
                    fused_score=fused_score,
                    metrics_row=row[feature_names]
                )

                results.append({
                    "timestamp": int(ts.timestamp()),
                    "trace_id": trace_id,
                    "service_name": service_mapping.get(service_ids[i], f"unknown_{service_ids[i]}"),
                    "score_trace": score_trace,
                    "score_metric": score_metric,
                    "fused_score": fused_score,
                    "plot_base64": img_base64,
                    "explanation": explanation
                })

        return jsonify({"anomalies": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)