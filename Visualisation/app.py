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
import gc
import os
from dotenv import load_dotenv
from mistralai import Mistral
from chronos import ChronosPipeline

# Charger Chronos-T5 small
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.float32,
)

def generate_forecast_with_chronos(metric_series, horizon=10):
    context = torch.tensor(metric_series.values, dtype=torch.float32)
    forecast = pipeline.predict(context, prediction_length=horizon)
    return forecast[0, 0, :].tolist()

load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=mistral_api_key)

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

Donne une explication probable de la cause de l’anomalie et propose une recommandation technique concrète pour la corriger directement tous ca dans 2 paragraphe ne depasse pas 600 mots  .
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

        metric_to_forecast = req_data.get("metric_to_forecast", "cpu_r")
        if metric_to_forecast not in feature_names:
            return jsonify({"error": f"metric_to_forecast '{metric_to_forecast}' not valid"}), 400

        include_forecast = req_data.get("include_forecast", True)

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

                # Plot principal (toutes métriques + anomalie)
                fig_main, ax_main = plt.subplots(figsize=(10, 4))
                for col in feature_names:
                    ax_main.plot(metrics_data.index, metrics_data[col], alpha=0.3, label=col)
                ax_main.axvline(ts, color='red', linestyle='--', label='Anomaly')
                ax_main.set_title(f"Anomaly at {ts} - Trace {trace_id}")
                ax_main.legend(loc='upper right')
                ax_main.set_xlabel("Timestamp")
                ax_main.set_ylabel("Metric value")
                buf_main = BytesIO()
                plt.savefig(buf_main, format="png", bbox_inches="tight")
                buf_main.seek(0)
                img_base64_main = base64.b64encode(buf_main.read()).decode('utf-8')
                plt.close(fig_main)

                explanation = generate_llm_explanation(
                    timestamp=ts,
                    service_name=service_mapping.get(service_ids[i], f"unknown_{service_ids[i]}"),
                    score_trace=score_trace,
                    score_metric=score_metric,
                    fused_score=fused_score,
                    metrics_row=row[feature_names]
                )

                anomaly_result = {
                    "timestamp": int(ts.timestamp()),
                    "trace_id": trace_id,
                    "service_name": service_mapping.get(service_ids[i], f"unknown_{service_ids[i]}"),
                    "score_trace": score_trace,
                    "score_metric": score_metric,
                    "fused_score": fused_score,
                    "plot_base64": img_base64_main,
                    "explanation": explanation
                }

                if include_forecast:
                    # Prévision
                    historical_data = metrics_data[metric_to_forecast].loc[:ts].tail(50)
                    forecast_values = generate_forecast_with_chronos(historical_data, horizon=10)

                    # Plot forecast (seulement la métrique choisie)
                    last_date = historical_data.index[-1]
                    freq = historical_data.index.inferred_freq or pd.infer_freq(historical_data.index)
                    if freq is None:
                        freq = 'T'  # Par défaut à la minute si non détectée
                    forecast_dates = pd.date_range(start=last_date, periods=len(forecast_values) + 1, freq=freq)[1:]

                    fig_forecast, ax_forecast = plt.subplots(figsize=(8, 3))
                    ax_forecast.plot(historical_data.index, historical_data.values, label="Historique")
                    ax_forecast.plot(forecast_dates, forecast_values, marker='o', color='blue', label=f"Prévision {metric_to_forecast}")
                    ax_forecast.set_title(f"Prévision de la métrique {metric_to_forecast} - Trace {trace_id}")
                    ax_forecast.set_xlabel("Timestamp")
                    ax_forecast.set_ylabel("Valeur")
                    ax_forecast.legend()
                    buf_forecast = BytesIO()
                    plt.savefig(buf_forecast, format="png", bbox_inches="tight")
                    buf_forecast.seek(0)
                    img_base64_forecast = base64.b64encode(buf_forecast.read()).decode('utf-8')
                    plt.close(fig_forecast)

                    anomaly_result["forecast"] = {
                        "metric": metric_to_forecast,
                        "predicted_values": forecast_values
                    }
                    anomaly_result["forecast_plot_base64"] = img_base64_forecast

                results.append(anomaly_result)

        return jsonify({"anomalies": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_forecast_plot', methods=['POST'])
def generate_forecast_plot():
    try:
        req_data = request.get_json()

        timestamp = req_data["timestamp"]  # int unix timestamp
        metric_to_forecast = req_data["metric_to_forecast"]
        metrics_data = pd.DataFrame(req_data["metrics_data"])
        trace_id = req_data.get("trace_id", "unknown_trace")

        if not all(f in metrics_data.columns for f in ["timestamp"] + feature_names):
            return jsonify({"error": "Missing required columns in metrics_data"}), 400

        if metric_to_forecast not in feature_names:
            return jsonify({"error": f"Invalid metric '{metric_to_forecast}'"}), 400

        metrics_data["timestamp"] = pd.to_datetime(metrics_data["timestamp"], unit='s')
        metrics_data = metrics_data.sort_values("timestamp").set_index("timestamp")

        ts = pd.to_datetime(timestamp, unit='s')

        historical_data = metrics_data[metric_to_forecast].loc[:ts].tail(50)
        forecast_values = generate_forecast_with_chronos(historical_data, horizon=10)

        last_date = historical_data.index[-1]
        freq = historical_data.index.inferred_freq or pd.infer_freq(historical_data.index)
        if freq is None:
            freq = 'T'  # Par défaut à la minute si non détectée
        forecast_dates = pd.date_range(start=last_date, periods=len(forecast_values) + 1, freq=freq)[1:]

        fig_forecast, ax_forecast = plt.subplots(figsize=(8, 3))
        ax_forecast.plot(historical_data.index, historical_data.values, label="Historique")
        ax_forecast.plot(forecast_dates, forecast_values, marker='o', color='blue', label=f"Prévision {metric_to_forecast}")
        ax_forecast.set_title(f"Prévision de la métrique {metric_to_forecast} - Trace {trace_id}")
        ax_forecast.set_xlabel("Timestamp")
        ax_forecast.set_ylabel("Valeur")
        ax_forecast.legend()
        buf_forecast = BytesIO()
        plt.savefig(buf_forecast, format="png", bbox_inches="tight")
        buf_forecast.seek(0)
        img_base64_forecast = base64.b64encode(buf_forecast.read()).decode('utf-8')
        plt.close(fig_forecast)

        return jsonify({
            "forecast_plot_base64": img_base64_forecast,
            "forecast": {
                "metric": metric_to_forecast,
                "predicted_values": forecast_values
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
