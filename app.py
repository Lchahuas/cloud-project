from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Error handling for loading the model and LabelEncoders
try:
    model = joblib.load("models/random_forest_model.pkl")
    le_sex = joblib.load('pickles/le_sex.pkl')
    le_chestpaintype = joblib.load('pickles/le_chestpaintype.pkl')
    le_restingecg = joblib.load('pickles/le_restingecg.pkl')
    le_exerciseangina = joblib.load('pickles/le_exerciseangina.pkl')
    le_stslope = joblib.load('pickles/le_stslope.pkl')
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

class InputData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

def preprocess_data(data: InputData) -> pd.DataFrame:
    # Convert input data to a DataFrame and apply LabelEncoders
    input_data = pd.DataFrame([data.dict()])
    input_data['Sex'] = le_sex.transform(input_data['Sex'])
    input_data['ChestPainType'] = le_chestpaintype.transform(input_data['ChestPainType'])
    input_data['RestingECG'] = le_restingecg.transform(input_data['RestingECG'])
    input_data['ExerciseAngina'] = le_exerciseangina.transform(input_data['ExerciseAngina'])
    input_data['ST_Slope'] = le_stslope.transform(input_data['ST_Slope'])
    return input_data

def validation_input(data: InputData):
    errors = []
    # Validate input data based on specified criteria

    if not (25 <= data.Age <= 80):
        errors["Age"] = "Age out of Range."
    
    if data.Sex not in {"M", "F"}:
        errors["Sex"] = "The 'Sex' field must be 'M' or 'F'."
    
    if data.ChestPainType not in {"ATA", "NAP", "ASY", "TA"}:
        errors["ChestPainType"] = "Invalid chest pain type."

    if not (0 <= data.RestingBP <= 210):
        errors["RestingBP"] = "Invalid RestingBP number."

    if not (0 <= data.Cholesterol <= 607):
        errors["Cholesterol"] = "Invalid Cholesterol number."

    if data.FastingBS not in {0, 1}:
        errors["FastingBS"] =  "The 'FastingBS' field must be 0 or 1"

    if data.RestingECG not in {"Normal", "LVH", "ST"}:
        errors["ChestPainType"] = "Invalid RestingECG type."

    if not (55 <= data.MaxHR <= 210):
        errors["MaxHR"] = "Invalid MaxHR number."
    
    if data.ExerciseAngina not in {"Y", "N"}:
        errors["ExerciseAngina"] = "Invalid ExerciseAngina type."
    
    if not (-4.0 <= data.Oldpeak <= 7.0):
        errors["Oldpeak"] = "Invalid Oldpeak number."

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "OK"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        errors = validation_input(data)
        if errors:
            raise HTTPException(status_code=400, detail={"error": "Validation errors", "details": errors})
        input_data = preprocess_data(data)
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")