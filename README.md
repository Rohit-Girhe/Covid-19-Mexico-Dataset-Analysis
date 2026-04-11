# Covid-19-Mexico-Dataset-Analysis
The main goal of this project is to build a machine learning model that, given a Covid-19 patient's current symptom, status, and medical history, will predict whether the patient is in high risk or not.

## ANN Deployment

This project includes a FastAPI deployment for the trained ANN model and preprocessing pipeline.

### Run Locally

```bash
pip install -r requirements.txt
uvicorn App.main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

### Run With Docker

```bash
docker build -t covid-ann-fastapi .
docker run -p 8000:8000 covid-ann-fastapi
```

Then open `http://127.0.0.1:8000` in your browser.
