🚀 Complaint Priority Classification – End-to-End MLOps Project

An end-to-end production-ready MLOps system for classifying customer complaints by urgency, built using MLflow, DVC, DagsHub, and BentoML.

This project demonstrates the complete ML lifecycle: data versioning, experiment tracking, model registry, promotion to production, and API deployment.

📌 Problem Statement

Customer support teams receive thousands of complaints daily.
Not all complaints are equally urgent — delayed handling of high-priority issues leads to customer churn.

Goal:
Automatically classify complaints into urgent vs non-urgent categories and deploy the model as a scalable API.

🧠 Solution Overview

NLP-based text classification using sentence embeddings

Binary urgency classification (urgent / non_urgent)

Full MLOps workflow with reproducibility and traceability

Lightweight, production-optimized inference service

🏗️ System Architecture
Data → DVC → Training → MLflow Experiments
                     ↓
              Model Registry
                     ↓
            Production Promotion
                     ↓
              BentoML API

Tools Used

Python

DVC – Dataset versioning

MLflow – Experiment tracking & model registry

DagsHub – Remote backend for MLflow + DVC

Sentence Transformers – Text embeddings

Scikit-learn – Classification

BentoML – Model serving

📁 Project Structure
complaint-priority/
│
├── data/
│   └── raw/
│       └── complaints.csv        # Versioned with DVC
│
├── src/
│   ├── train.py                  # Training + MLflow logging
│   └── data_loader.py
│
├── scripts/
│   ├── cleanup_mlflow.py          # Experiment cleanup
│   └── promote_to_production.py  # Model promotion
│
├── service.py                    # BentoML service
├── requirements.txt
├── dvc.yaml
├── README.md
└── .gitignore

📊 Dataset

Synthetic but realistic complaint text

Multiple complaint categories

Imbalanced urgency distribution (real-world scenario)

Fully versioned using DVC

🔬 Model Training
Approach

Convert complaint text → sentence embeddings

Binary classification using Logistic Regression

Class balancing for realistic performance

Embeddings

Training: all-mpnet-base-v2 (high-quality embeddings)

Inference: all-MiniLM-L6-v2 (lightweight & fast)

Metrics Logged

Accuracy

Weighted F1 Score

Hyperparameters

Model artifacts

All experiments are tracked in MLflow (DagsHub backend).

🧪 Experiment Tracking (MLflow + DagsHub)

Each training run logged automatically

Multiple experiments supported

Best model selected manually

Model registered in MLflow Registry

Example:

complaint_priority_model_binary_embeddings
└── Version 4 → Production

🚦 Model Promotion

Only explicitly approved models are deployed.

python scripts/promote_to_production.py


This promotes the latest model version to Production in MLflow.

🚀 Deployment (BentoML)

The Production model is served via a REST API using BentoML.

Start the API
bentoml serve service:svc --reload


Server runs at:

http://localhost:3000

🔗 API Usage
Endpoint

POST /predict

Request
{
  "text": "My internet has been down for 3 days and nobody is responding"
}

Response
{
  "prediction": "urgent"
}

📈 Monitoring

Built-in Prometheus metrics

Available at:

http://localhost:3000/metrics

♻️ Reproducibility & Best Practices

Dataset versioned with DVC

Experiments reproducible via MLflow

Clear separation of:

Training

Registry

Deployment

Production model explicitly promoted

Lightweight inference for stability on limited hardware

🧠 Key MLOps Concepts Demonstrated

Experiment tracking

Data versioning

Model registry

Model version promotion

Production deployment

Inference optimization

API-based ML serving