# COVID-19 Patient Mortality Predictor

An end-to-end AI project that starts with a messy real-world COVID-19 dataset and turns it into a deployable mortality-risk prediction system using an Artificial Neural Network, MLflow, FastAPI, and Docker.

## Overview

This repository documents the full lifecycle of a healthcare-focused machine learning project:

- raw data analysis and cleaning from a public COVID-19 dataset
- binary target creation for mortality prediction
- ANN model training and evaluation
- experiment tracking with MLflow
- deployment through a FastAPI backend and interactive web UI
- containerized execution with Docker

The goal of the project is to predict whether a patient is at elevated mortality risk based on demographic and clinical features.

## Why This Project Matters

Predicting mortality risk is a high-impact healthcare use case. In a real screening workflow, a model like this can support:

- early triage prioritization
- risk-aware patient monitoring
- hospital resource planning
- telemedicine decision support

This project is intended as a decision-support prototype, not a replacement for clinical judgment.

## Problem Statement

The original dataset was not model-ready. It contained coded categorical values, placeholder values such as `97`, `98`, and `99`, and a raw `DATE_DIED` field instead of a clean target column.

The project workflow involved:

1. understanding the raw schema
2. cleaning and transforming the data
3. creating the binary target `DIED`
4. preparing ANN-compatible features
5. training and evaluating the neural network
6. deploying the final model behind a web application

## Dataset Summary

| Dataset | Description |
| --- | --- |
| `Data/CovidData.csv` | Raw dataset with `1,048,575` rows and `21` columns |
| `Data/Cleaned_CovidData.csv` | Cleaned dataset used for ANN modeling |

Important cleaning logic included:

- converting `DATE_DIED` into the binary target `DIED`
- handling unknown and non-applicable coded values
- standardizing categorical medical indicators
- excluding leakage-prone fields such as `CLASIFFICATION_FINAL`

## Final Modeling Setup

### Target Variable

- `DIED`

### Final Input Features

- `USMER`
- `MEDICAL_UNIT`
- `SEX`
- `PATIENT_TYPE`
- `PNEUMONIA`
- `AGE`
- `DIABETES`
- `COPD`
- `ASTHMA`
- `INMSUPR`
- `HIPERTENSION`
- `OTHER_DISEASE`
- `CARDIOVASCULAR`
- `OBESITY`
- `RENAL_CHRONIC`
- `TOBACCO`

### Preprocessing Strategy

- stratified train/validation/test split: `70% / 15% / 15%`
- standardization for `AGE`
- one-hot encoding for `MEDICAL_UNIT`
- binary features passed in numeric form
- class weights for the imbalanced mortality target

## Model Architecture

The final model is a feedforward Artificial Neural Network implemented in TensorFlow/Keras.

- hidden layers: `64 -> 32 -> 16`
- activation: `ReLU`
- dropout: `0.30`, `0.20`
- output layer: `Sigmoid`
- optimizer: `Adam`
- loss: `Binary Cross-Entropy`
- early stopping on validation loss

## Model Performance

### Validation Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | `0.8842` |
| Precision | `0.3814` |
| Recall | `0.9488` |
| F1-score | `0.5441` |
| ROC-AUC | `0.9562` |

### Test Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | `0.8829` |
| Precision | `0.3787` |
| Recall | `0.9489` |
| F1-score | `0.5414` |
| ROC-AUC | `0.9563` |

The strongest result in this project is the very high recall, which is especially valuable in a healthcare risk-screening context where missing a genuinely high-risk patient is costlier than producing additional alerts.

## Project Workflow

1. Inspect the raw dataset and identify invalid coded values.
2. Convert `DATE_DIED` into the target column `DIED`.
3. Clean and transform the features into model-ready numeric inputs.
4. Perform stratified splitting to preserve class balance.
5. Build a preprocessing pipeline for scaling and encoding.
6. Train the ANN with class weights and early stopping.
7. Track experiments, metrics, and artifacts with MLflow.
8. Evaluate using confusion matrix, F1-score, recall, and ROC-AUC.
9. Save the trained model, preprocessor, and metadata.
10. Serve the model using FastAPI and a browser-based UI.
11. Package the full app inside a Docker container.

## Deployment Architecture

```mermaid
flowchart LR
    A["User Input"] --> B["FastAPI Backend"]
    B --> C["Saved Preprocessor"]
    C --> D["ANN Model (.keras)"]
    D --> E["Mortality Risk Probability"]
    E --> F["Interactive UI Response"]
```

## Tech Stack

| Area | Tools |
| --- | --- |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | TensorFlow, Keras, Scikit-learn |
| Experiment Tracking | MLflow |
| Deployment | FastAPI, Jinja2, JavaScript, CSS |
| Packaging | Docker |

## Repository Structure

```text
.
|-- App/
|   |-- main.py
|   |-- static/
|   |   |-- app.js
|   |   `-- styles.css
|   `-- templates/
|       `-- index.html
|-- Data/
|   |-- CovidData.csv
|   `-- Cleaned_CovidData.csv
|-- Model/
|   |-- ann_died_model.keras
|   |-- preprocessor.joblib
|   |-- model_metadata.json
|   |-- artifacts/
|   `-- mlruns/
|-- Notebook/
|   |-- Cleaning_and_analysis.ipynb
|   `-- Model_Building.ipynb
|-- Dockerfile
|-- README.md
`-- requirements.txt
```

## Running the Project

### Local Run

```bash
pip install -r requirements.txt
uvicorn App.main:app --reload
```

Open:

- `http://127.0.0.1:8000`
- `http://localhost:8000`

### Docker Run

```bash
docker build -t covid-ann-fastapi .
docker run -p 8000:8000 covid-ann-fastapi
```

Then open:

- `http://127.0.0.1:8000`

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serve the web interface |
| `GET` | `/api/health` | Health check and model status |
| `POST` | `/api/predict` | Generate mortality-risk prediction |

## Saved Artifacts

The project stores:

- trained ANN model: `Model/ann_died_model.keras`
- fitted preprocessing pipeline: `Model/preprocessor.joblib`
- experiment metadata: `Model/model_metadata.json`
- training and evaluation artifacts: `Model/artifacts/`
- MLflow runs: `Model/mlruns/`

## Notes

- The UI shows human-readable medical labels, while the backend still submits the encoded values expected by the trained model.
- The current deployment uses `tensorflow-cpu` for cleaner CPU-based container execution.
- This project should be treated as a machine learning prototype for educational and screening use, not as a production clinical system without external validation.


## Author

**Rohit Girhe**

Built as part of an ANN capstone workflow focused on real-world machine learning, experiment tracking, and deployment readiness.
