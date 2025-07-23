from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Initialize FastAPI app
app = FastAPI(title="Iris Species Prediction API")

# Define input data model
class IrisMeasurement(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load and train model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model to file
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load model for prediction
with open("iris_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Define species mapping
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.get("/")
async def root():
    return {"message": "Welcome to Iris Species Prediction API"}

@app.post("/predict")
async def predict_species(measurement: IrisMeasurement):
    try:
        # Prepare input data
        input_data = np.array([[
            measurement.sepal_length,
            measurement.sepal_width,
            measurement.petal_length,
            measurement.petal_width
        ]])
        
        # Make prediction
        prediction = loaded_model.predict(input_data)
        probability = loaded_model.predict_proba(input_data)[0]
        
        # Format response
        return {
            "species": species_map[int(prediction[0])],
            "probabilities": {
                species_map[i]: float(prob) for i, prob in enumerate(probability)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)