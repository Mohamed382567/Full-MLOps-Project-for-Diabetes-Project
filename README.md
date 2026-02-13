# 🏥 Clinical Diabetes Prediction - Full End-to-End MLOps System

This repository contains a professional, production-ready machine learning system designed to predict diabetes risk. The project demonstrates a complete **MLOps Lifecycle**, successfully transitioning a research notebook into a containerized, cloud-deployed microservice with automated monitoring and quality assurance.

---

## 🚀 Key Features
* **Cloud-Native Architecture**: Directly linked via **Docker Hub** for seamless, stable deployment on **Render**.
* **Distributed UI & Engine**: Powered by a **FastAPI** backend (Inference Engine) and a **Gradio** interactive frontend.
* **Automated CI/CD**: Integrated with **GitHub Actions** for automated training, testing, and Docker image publishing.
* **Experiment Monitoring**: Fully integrated with **MLflow and DagsHub** to track model accuracy, training parameters, and performance history.
* **Quality Assurance**: Includes an automated **Testing Suite (Pytest)** to ensure code reliability before any deployment.

---

## 📊 Live Monitoring & Demo
* **Live Web App**: [https://diabetes-mlops-l6uy.onrender.com/]
* **MLflow Dashboard**: [https://dagshub.com/Mohamed382567/Full-MLOps-Project-for-Diabetes-Project.mlflow/]

---

## 🛠️ Tech Stack
* **ML Core**: Scikit-Learn, Pandas, NumPy, Joblib, SMOTE
* **Monitoring**: MLflow & DagsHub
* **API & UI**: FastAPI, Uvicorn, Gradio
* **DevOps**: Docker, GitHub Actions, Docker Hub, Render

---

## ⚙️ How it Works
* **Training**: run_pipeline.py executes, trains the model using SMOTE for class balance, and logs all metrics/parameters to MLflow.

* **Testing**: Automated tests in the tests/ directory verify code integrity and API functionality.

* **Deployment**: Upon a successful push, GitHub Actions builds a new Docker Image and pushes it to Docker Hub.

* **Production**: Render automatically pulls the latest image from Docker Hub and serves the updated API and UI.

## 📂 Project Structure
```text
├── .github/workflows/    # CI/CD Pipelines (Docker Publishing & Automation)
├── src/
│   ├── app/              # Deployment code (FastAPI & Gradio logic)
│   ├── data/             # Data cleaning and preprocessing logic
│   ├── features/         # Feature engineering & scaling
│   └── models/           # Model architecture and training logic
├── tests/                # Automated test suite (Quality Gate)
├── diabetes-model-artifacts/ # Production-ready binaries (.pkl files)
├── run_pipeline.py       # The Orchestrator for the entire MLOps flow
├── Dockerfile            # Container environment configuration
├── requirements.txt      # Project dependencies
└── .gitignore            # Version control safety filter
