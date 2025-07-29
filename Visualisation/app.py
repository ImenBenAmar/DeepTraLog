from flask import Flask, request, jsonify
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import json
from model_definitions import GGNN  # Ton modèle
import os
import joblib  # Pour charger le scaler sauvegardé
from torch_geometric.data import Batch
from sklearn.preprocessing import MinMaxScaler



app = Flask(__name__)

# Charger le modèle Deep SVDD + GGNN
checkpoint = torch.load('../models/ggnn_Deep_svdd .pth', map_location='cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GGNN(num_node_features=306, hidden_channels=300, num_edge_types=3, num_iterations=2).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Charger centre et rayon
c = checkpoint['center'].to(device)
R = checkpoint['radius'].to(device)

# Charger embeddings
event_embeddings = np.load('../models/event_embeddings (1).npy', allow_pickle=True).item()

# Mapping ServiceId → Nom
id_service_df = pd.read_csv('../DeepTralog/id_service.csv', header=None, names=['ServiceId', 'Service'])
service_mapping = dict(zip(id_service_df['ServiceId'], id_service_df['Service']))

scaler = joblib.load('../models/scaler.pkl')  # Modifie ce chemin si nécessaire

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


@app.route('/detect_anomaly', methods=['POST'])
def detect_anomaly():
    try:
        data_list = request.json  
        if not isinstance(data_list, list):
            data_list = [data_list]  

        results = []
        graphs = []
        service_ids = []
        trace_ids = []

        for data_json in data_list:
            graph, service_id = process_json_graph(data_json)
            graphs.append(graph)
            service_ids.append(service_id)
            trace_ids.append(data_json.get("trace_id", "unknown_trace"))

        batch = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            embeddings = model(batch)
            distances = torch.sum((embeddings - c) ** 2, dim=1)
            scores = distances - R ** 2
            labels = (scores > 0).int().tolist()

        for i in range(len(graphs)):
            service_name = service_mapping.get(service_ids[i], f"unknown_service_{service_ids[i]}")
            results.append({
                "trace_id": trace_ids[i],
                "service_id": service_ids[i],
                "service_name": service_name,
                "anomaly_score": float(scores[i].item()),
                "label": labels[i]
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


xgb_model = joblib.load("../models/XG_metric_detector.pkl")

df_train = pd.read_csv("../Dataset_SMD/machine-1-1_train_filtered.csv")  
feature_names = ['cpu_r', 'load_1', 'load_5', 'mem_u', 'disk_q', 'disk_r',
                 'disk_w', 'disk_u', 'eth1_fi', 'eth1_fo', 'tcp_timeouts']
X_train = df_train[feature_names]


scaler_xG = MinMaxScaler()
scaler_xG.fit(X_train)

@app.route('/detect_xgb_anomaly', methods=['POST'])
def detect_xgb_anomaly():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400

        features = data["features"]
        if len(features) != len(feature_names):
            return jsonify({"error": f"Expected {len(feature_names)} features, got {len(features)}"}), 400

        X_input = pd.DataFrame([features], columns=feature_names)
        X_scaled = scaler_xG.transform(X_input)

        pred_label = int(xgb_model.predict(X_scaled)[0])
        pred_proba = float(xgb_model.predict_proba(X_scaled)[0][1])

        return jsonify({
            "label": pred_label,
            "anomaly_score": pred_proba
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/batch_detect_xgb_anomaly', methods=['POST'])
def batch_detect_xgb_anomaly():
    try:
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({"error": "Missing 'data' in request"}), 400

        records = data["data"]
        df_input = pd.DataFrame(records)

        if not all(f in df_input.columns for f in feature_names):
            return jsonify({"error": f"Some features are missing. Required: {feature_names}"}), 400

        X_scaled = scaler_xG.transform(df_input[feature_names])

        pred_labels = xgb_model.predict(X_scaled).astype(int).tolist()
        pred_probas = xgb_model.predict_proba(X_scaled)[:, 1].tolist()

        return jsonify({
            "anomaly_labels": pred_labels,
            "anomaly_scores": pred_probas
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
