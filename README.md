# 🚀 Complaint Priority Classification – End-to-End MLOps Project

An **end-to-end production-ready MLOps system** for classifying customer complaints by urgency, built using **MLflow, DVC, DagsHub, and BentoML**.

This project demonstrates the **complete machine learning lifecycle**, including:

- Data versioning
- Experiment tracking
- Model registry
- Model promotion to production
- API deployment for inference

---

# 📌 Problem Statement

Customer support teams receive **thousands of complaints daily**.

Not all complaints are equally urgent, and delayed responses to **high-priority issues** can lead to **customer dissatisfaction and churn**.

### Goal

Automatically classify complaints into:

- **urgent**
- **non_urgent**

and deploy the trained model as a **scalable production API**.

---

# 🧠 Solution Overview

- NLP-based **text classification**
- Sentence embedding representation of complaints
- Binary classification (**urgent / non_urgent**)
- Fully reproducible **MLOps workflow**
- Lightweight **production inference service**

---

# 🏗️ System Architecture

```
Data → DVC → Training → MLflow Experiments
                     ↓
              Model Registry
                     ↓
            Production Promotion
                     ↓
               BentoML API
```

---

# 🛠 Tools & Technologies

- **Python**
- **DVC** – Dataset versioning
- **MLflow** – Experiment tracking & model registry
- **DagsHub** – Remote backend for MLflow + DVC
- **Sentence Transformers** – Text embeddings
- **Scikit-learn** – Classification model
- **BentoML** – Model serving and deployment

---

# 📁 Project Structure

```
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
│   ├── cleanup_mlflow.py         # Experiment cleanup
│   └── promote_to_production.py  # Model promotion
│
├── service.py                    # BentoML service
├── requirements.txt
├── dvc.yaml
├── README.md
└── .gitignore
```

---

# 📊 Dataset

- Synthetic but **realistic complaint text**
- Multiple complaint categories
- **Imbalanced urgency distribution** (reflecting real-world data)
- Fully **versioned using DVC**

---

# 🔬 Model Training

### Approach

1. Convert complaint text → **sentence embeddings**
2. Train a **binary classifier**
3. Handle **class imbalance** for realistic performance

---

### Embedding Models

**Training**

```
all-mpnet-base-v2
```

High-quality embeddings for improved representation.

**Inference**

```
all-MiniLM-L6-v2
```

Lightweight model for **fast and efficient inference**.

---

### Classification Model

```
Logistic Regression
```

---

### Metrics Logged

- Accuracy
- Weighted F1 Score
- Hyperparameters
- Model artifacts

All experiments are automatically tracked using **MLflow (DagsHub backend)**.

---

# 🧪 Experiment Tracking (MLflow + DagsHub)

Each training run logs:

- Parameters
- Metrics
- Artifacts
- Model versions

Multiple experiments can be compared.

Example model registry entry:

```
complaint_priority_model_binary_embeddings
└── Version 4 → Production
```

---

# 🚦 Model Promotion

Only **approved models** are deployed to production.

Run:

```
python scripts/promote_to_production.py
```

This promotes the **latest model version** to **Production** in the MLflow registry.

---

# 🚀 Deployment (BentoML)

The **Production model** is deployed as a **REST API** using BentoML.

### Start the API

```
bentoml serve service:svc --reload
```

Server runs at:

```
http://localhost:3000
```

---

# 🔗 API Usage

### Endpoint

```
POST /predict
```

### Request

```
{
  "text": "My internet has been down for 3 days and nobody is responding"
}
```

### Response

```
{
  "prediction": "urgent"
}
```

---

# 📈 Monitoring

Built-in **Prometheus metrics** are available at:

```
http://localhost:3000/metrics
```

These metrics enable monitoring of:

- request counts
- latency
- model inference usage

---

# ♻️ Reproducibility & Best Practices

- Dataset versioned with **DVC**
- Experiments tracked with **MLflow**
- Models stored in **MLflow Registry**
- Explicit **production promotion**
- Lightweight inference for **stable deployment**

Clear separation of:

- Training
- Model registry
- Deployment
- Production inference

---

# 🧠 Key MLOps Concepts Demonstrated

- Experiment tracking
- Data versioning
- Model registry
- Model version promotion
- Production deployment
- API-based ML serving
- Inference optimization

---

# 👨‍💻 Author

**Aditya Raj Kaushik**
