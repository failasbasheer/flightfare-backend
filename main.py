from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import datetime

app = FastAPI()

model = joblib.load("flight_rf.pkl")

class FlightInput(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Total_Stops: str
    Date_of_Journey: str
    Dep_Time: str
    Arrival_Time: str
    Duration: str


airline_map = {
    'Jet Airways': 0,
    'IndiGo': 1,
    'Air India': 2,
    'Multiple carriers': 3,
    'SpiceJet': 4,
    'Vistara': 5,
    'Air Asia': 6,
    'GoAir': 7,
    'Multiple carriers Premium economy': 8,
    'Jet Airways Business': 9,
    'Vistara Premium economy': 10,
    'Trujet': 11
}

source_map = {
    'Delhi': 0,
    'Kolkata': 1,
    'Banglore': 2,
    'Mumbai': 3,
    'Chennai': 4
}

destination_map = {
    'Cochin': 0,
    'Banglore': 1,
    'Delhi': 2,
    'New Delhi': 3,
    'Hyderabad': 4,
    'Kolkata': 5
}

stops_map = {
    'non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4
}


def preprocess(data: FlightInput):
    journey_date = datetime.datetime.strptime(data.Date_of_Journey, "%d/%m/%Y")
    Journey_day = journey_date.day
    Journey_month = journey_date.month

    Dep_hour, Dep_min = map(int, data.Dep_Time.split(":"))
    Arrival_hour, Arrival_min = map(int, data.Arrival_Time.split(":"))

    Duration_hour = int(data.Duration.split("h")[0])
    Duration_min = int(data.Duration.split("h")[1].replace("m", ""))

    Airline = airline_map[data.Airline]
    Source = source_map[data.Source]
    Destination = destination_map[data.Destination]
    Total_Stops = stops_map[data.Total_Stops]

    features = np.array([[
        Airline,
        Source,
        Destination,
        Total_Stops,
        Journey_day,
        Journey_month,
        Dep_hour,
        Dep_min,
        Arrival_hour,
        Arrival_min,
        Duration_hour,
        Duration_min
    ]])

    return features


@app.post("/predict")
def predict(data: FlightInput):
    X = preprocess(data)
    prediction = model.predict(X)[0]
    return {"predicted_price": float(prediction)}
