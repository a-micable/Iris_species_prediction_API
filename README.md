# Iris Species Prediction API

A FastAPI-based REST API that predicts Iris flower species (Setosa, Versicolor, or Virginica) based on sepal and petal measurements using a machine learning model trained on the Iris dataset.

## Features
- Trains a RandomForestClassifier model on the Iris dataset.
- Exposes a REST API endpoint to predict species from measurements.
- Returns predicted species and probability scores.
- Includes automatic Swagger UI documentation.

## Requirements
- Python 3.7+
- Required packages: `fastapi`, `uvicorn`, `scikit-learn`, `numpy`, `pydantic`

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/iris-species-api.git
   cd iris-species-api
