# Complaint Priority Classification – End-to-End MLOps Project

An end-to-end, production-oriented **Machine Learning + MLOps** project that demonstrates how to build, track, version, and deploy a real-world ML system using **DVC, MLflow, DagsHub, and BentoML**.

This project focuses on **complaint priority classification** (low / medium / high) from noisy, realistic customer complaint text, with full experiment tracking and model lifecycle management.

---

## 🚀 Project Highlights

- Realistic, **highly complex synthetic dataset** (8k samples, noisy & non-deterministic)
- **DVC** for dataset versioning and reproducibility
- **MLflow** for experiment tracking, metrics, and model registry
- **DagsHub** as a remote backend for Git + DVC + MLflow
- Systematic **multi-experiment training**
- Clean experiment lifecycle (archived old experiments)
- Production-ready project structure
- Designed as a **resume-grade MLOps portfolio project**

---

## 🧠 Problem Statement

Customer support teams receive thousands of complaints daily.  
Automatically predicting the **priority** of a complaint helps route urgent issues faster and improve customer satisfaction.

**Goal:**  
Classify complaints into:
- `low`
- `medium`
- `high`

based on noisy, ambiguous, real-world text.

---

## 📂 Project Structure

complaint-priority/
│
├── data/
│ └── raw/
│ └── complaints.csv.dvc # DVC-tracked dataset
│
├── src/
│ ├── init.py
│ ├── data_loader.py # Data loading utilities
│ └── train.py # Multi-experiment MLflow training
│
├── scripts/
│ ├── cleanup_mlflow.py # MLflow experiment cleanup
│ └── promote_model.py # Model promotion script
│
├── generate_dataset.py # Complex dataset generator
├── requirements.txt
├── requirements-lock.txt
└── README.md


---

## 📊 Dataset Design

- ~8,000 complaint samples
- Overlapping vocabulary across priorities
- Implicit severity (no keyword leakage)
- Multi-issue complaints
- Contextual ambiguity
- Controlled label noise

This ensures the task is **non-trivial** and models must truly generalize.

---

## 🔬 Modeling Approach

- **Text Features:** TF-IDF (1–2 ngrams, limited feature space)
- **Model:** Logistic Regression
- **Evaluation Metrics:**
  - Primary: `f1_weighted`
  - Secondary: `accuracy`

Feature sizes intentionally constrained to avoid overfitting on small-to-medium datasets.

---

## 🧪 Experiment Tracking (MLflow)

- All experiments tracked using **MLflow**
- Remote backend hosted on **DagsHub**
- Multiple controlled experiments:
  - TF-IDF feature sizes
  - Regularization strengths
- Clean experiment namespace:
complaint_priority_v3_complex_data


Old exploratory experiments were explicitly removed to keep the UI clean and meaningful.

---

## 🗂️ Model Registry

- Best-performing model selected based on `f1_weighted`
- Model versioned and stored in **MLflow Model Registry**
- Promotion to **Production** stage handled programmatically

This enables reproducible deployment and rollback.

---

## 🔁 Reproducibility

- Dataset tracked via **DVC**
- Exact dataset version reproducible via:
```bash
dvc pull
Experiments reproducible by rerunning:

python -m src.train
🛠️ Tech Stack
Python

scikit-learn

MLflow

DVC

DagsHub

BentoML (deployment phase)

📌 How to Run Locally
1. Clone repository
git clone https://dagshub.com/Aditya-Raj-Kaushik/complaint-priority.git
cd complaint-priority
2. Install dependencies
pip install -r requirements.txt
3. Pull dataset
dvc pull
4. Run training experiments
python -m src.train