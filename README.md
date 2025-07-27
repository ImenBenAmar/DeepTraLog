# 🔍 DeepTraLog - Détection d'Anomalies et Analyse de Causes par Deep Learning

Ce dépôt propose un pipeline complet pour la détection d'anomalies et l'analyse de causes racines (RCA) dans des systèmes distribués, en s'appuyant sur des techniques de deep learning et de graphes, ainsi que sur des méthodes classiques d'anomaly detection sur métriques.

---

## 🗂️ Organisation du dépôt

```bash
|
├── cpu_anomaly_detection_univariate/
│   ├── CPU-anomaly-detection.ipynb      # Détection d'anomalies univariées sur la CPU (méthodes statistiques et ML)
│   └── anomaly-detection-autoencoder-cpu-util (1).ipynb # Détection d'anomalies CPU par autoencoder (Kaggle)
│
├── Metrics_detection_multivariate/
│   └── metrics-anomaly.ipynb            # Détection d'anomalies multivariées sur métriques système (Isolation Forest, Z-score, Autoencoder, LSTM, VAE, etc.)
│
├── DeepTralog/
│   ├── deeptralog_preprocesing.ipynb    # Prétraitement logs/traces, parsing, fusion, embeddings, construction des graphes, export format DeepTraLog
│   ├── eda-graphdata.ipynb              # Analyse exploratoire des graphes générés (EDA, statistiques, visualisations)
│   └── model-training.ipynb             # Entraînement et évaluation du modèle DeepTraLog (GGNN, DeepSVDD, métriques)
│
├── models/                              # Modèles entraînés, checkpoints, configurations
│
├── DeepTraLog architecture.pdf          # Schéma d'architecture du modèle DeepTraLog
├── metric_anomaly.pdf                   # Présentation sur la détection d'anomalies sur métriques
├── output.png                           # Exemple de sortie ou visualisation
├── README.md                            # Documentation du projet
```

---

## 📒 Description des notebooks principaux

### 1. **Détection d'anomalies sur métriques système**

- **cpu_anomaly_detection_univariate/CPU-anomaly-detection.ipynb**  
  Analyse univariée de séries temporelles CPU (AWS Cloudwatch) : Z-score, robust Z-score (MAD), IQR, Isolation Forest, LOF, One-Class SVM. Visualisations, confusion matrix, courbes ROC.

- **cpu_anomaly_detection_univariate/anomaly-detection-autoencoder-cpu-util (1).ipynb**  
  Approche autoencoder (Dense, Conv1D) pour la détection d'anomalies sur la CPU. Préparation des séquences, entraînement, visualisation des erreurs de reconstruction, comparaison avec les labels d'anomalie officiels.

- **Metrics_detection_multivariate/metrics-anomaly.ipynb**  
  Détection d'anomalies multivariées sur des métriques serveur (CPU, mémoire, disque, réseau, TCP).  
  - Prétraitement, sélection de features, analyse des corrélations.
  - Méthodes non supervisées : Isolation Forest (grid search), Z-score, robust Z-score (MAD), IQR.
  - Méthodes supervisées : Random Forest, XGBoost.
  - Deep learning : Autoencoder, LSTM, LSTM-VAE, OmniAI.
  - Visualisations, matrices de confusion, courbes ROC, analyse des performances.

---

### 2. **Pipeline DeepTraLog (logs + traces + graphes)**

- **DeepTralog/deeptralog_preprocesing.ipynb**  
  - Parsing avancé des logs (Drain), extraction TraceId/SpanId/Service.
  - Parsing et transformation des traces (spans) en événements structurés.
  - Fusion logs/traces, harmonisation, enrichissement des événements.
  - Génération d'embeddings sémantiques (TF-IDF + GloVe).
  - Construction des graphes orientés (TEG) pour chaque trace, typage des arêtes (sequence, sync/async).
  - Export des graphes et métadonnées au format officiel DeepTraLog (jsons/csv).

- **DeepTralog/eda-graphdata.ipynb**  
  - Analyse exploratoire des graphes générés : statistiques sur les nœuds, arêtes, types d'événements, distribution des labels.
  - Visualisation de sous-graphes, analyse des relations inter-services, typage des arêtes.

- **DeepTralog/model-training.ipynb**  
  - Chargement des graphes DeepTraLog, préparation des datasets (normal/anomalous).
  - Entraînement du modèle DeepTraLog (GGNN + DeepSVDD).
  - Évaluation des performances : MSE, F1-score, AUC, matrice de confusion, courbes ROC, analyse des scores d'anomalie.

---

### 3. **Autres dossiers et fichiers**

- **models/** : Modèles entraînés, checkpoints, fichiers de configuration pour la reproduction ou l'inférence.
- **DeepTraLog architecture.pdf** : Schéma d'architecture du modèle DeepTraLog.
- **metric_anomaly.pdf** : Présentation ou rapport sur la détection d'anomalies sur métriques.
- **output.png** : Exemple de sortie ou visualisation.
- **README.md** : Ce fichier de documentation.

---

## 📊 Résultats obtenus

Quelques scores typiques obtenus sur les jeux de données testés :

- **Détection univariée CPU (AE meilleur ) :**
  - F1-score : 0.82 
  - Précision : 0.77
  - Rappel : 0.78 
  - AUC : ~0.77

- **Détection multivariée (metrics-anomaly.ipynb) :**
  - Isolation Forest : Precision: 0.3466 Recall: 0.7108 F1-score: 0.4660
  - Z_score: Precision: 0.6472 Recall: 0.7866  F1-score: 0.7101
  - Autoencoder dense : Precision: 0.5671 Recall: 0.5995 F1-score: 0.5828
  - LSTM : Precision: 0.7617 Recall: 0.7249 F1-score: 0.7429
  - OmniAI : F1-score: 0.7430 Recall: 0.7335 Precision: 0.7528 AUC: 0.9519
  - XGBoost (supervisé) : Precision: 0.90 Recall: 0.97 F1-score: 0.93
  - Random Forest: Précision : 0.98 Rappel : 0.95 F1-score : 0.97

- **DeepTraLog (GGNN + DeepSVDD sur graphes logs+traces) :**
   -precsion 0.968
   -Recall: 0.673, 
   -F1-Score: 0.794 
   -AUC: 0.822
  - Matrice de confusion, courbes ROC et distributions des scores d'anomalie disponibles dans le notebook.

> Les résultats peuvent varier selon le dataset, le split et les hyperparamètres. Voir chaque notebook pour les détails et visualisations.

---

## 🚀 Utilisation des notebooks et modèles

1. **Prétraitement et parsing :**
   - Exécuter `DeepTralog/deeptralog_preprocesing.ipynb` pour parser les logs/traces, fusionner, générer les embeddings et exporter les graphes au format DeepTraLog.

2. **Analyse exploratoire :**
   - Utiliser `DeepTralog/eda-graphdata.ipynb` pour explorer les graphes générés, vérifier la distribution des labels, des types d'arêtes, etc.

3. **Entraînement et évaluation :**
   - Lancer `DeepTralog/model-training.ipynb` pour charger les graphes, entraîner le modèle GGNN + DeepSVDD, et obtenir les métriques (F1, AUC, courbes ROC).
   - Les checkpoints sont sauvegardés dans le dossier `models/` (ex : `ggnn_Deep_svdd.pth`)

4. **Détection sur métriques :**
   - Pour la détection sur métriques système, utiliser les notebooks dans `cpu_anomaly_detection_univariate/` et `Metrics_detection_multivariate/`.

5. **Reprise d'un modèle entraîné :**
   - Charger le modèle et les paramètres sauvegardés (voir la dernière cellule de `model-training.ipynb` pour un exemple de chargement avec PyTorch).

---

## 📄 Références

- Article original : [Trace-Log Combined Microservice Anomaly Detection through Graph-based Deep Learning (ICSE 2022)](https://cspengxin.github.io/publications/icse22-DeepTraLog.pdf)
- Données : [DeepTraLog Dataset (GitHub)](https://github.com/FudanSELab/DeepTraLog)
- Dataset Multivariée pour les metriques (CPU,..) : [SMD dataset (GitHub)] : (https://github.com/snareli/Server-Machine-Dataset)

---
