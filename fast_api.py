from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import model as mod

app = FastAPI()

# Load your pretrained model once
model = joblib.load(mod.MODEL_DUMP_PATH)


class InferenceInput(BaseModel):
    # Using aliases to handle hyphenated names from CSV data
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num", example=5)
    marital_status: str = Field(..., alias="marital-status",
                                example="Never-married")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country",
                                example="United-States")

    class Config:
        allow_population_by_field_name = True


@app.get("/")
async def root() -> dict:
    return {"message": "Welcome to the RandomForestClassifier API"}


@app.post("/predict")
async def predict(input_data: InferenceInput) -> dict:
    # Extract data in the proper order expected by the model
    data = [[
        input_data.age,
        input_data.workclass,
        input_data.fnlgt,
        input_data.education,
        input_data.education_num,
        input_data.marital_status,
        input_data.occupation,
        input_data.relationship,
        input_data.race,
        input_data.sex,
        input_data.capital_gain,
        input_data.capital_loss,
        input_data.hours_per_week,
        input_data.native_country
    ]]
    prediction = model.predict(data)
    return {"prediction": prediction[0]}
