from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import datetime

app = FastAPI()

# Load the model
model = joblib.load("flight_rf.pkl")

# Input schema
class FlightInput(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Total_Stops: str
    Date_of_Journey: str          # "24/03/2019"
    Dep_Time: str                 # "22:20"
    Arrival_Time: str             # "01:10"
    Duration: str                 # "2h 50m"


# -------------------------
# MAPPINGS (from notebook)
# -------------------------
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


# -------------------------
# PREPROCESSOR
# -------------------------
def preprocess(data: FlightInput):

    # Journey day & month
    journey_date = datetime.datetime.strptime(data.Date_of_Journey, "%d/%m/%Y")
    Journey_day = journey_date.day
    Journey_month = journey_date.month

    # Departure
    Dep_hour = int(data.Dep_Time.split(":")[0])
    Dep_min = int(data.Dep_Time.split(":")[1])

    # Arrival
    Arrival_hour = int(data.Arrival_Time.split(":")[0])
    Arrival_min = int(data.Arrival_Time.split(":")[1])

    # Duration
    dur = data.Duration
    Duration_hour = int(dur.split("h")[0])
    Duration_min = int(dur.split("h")[1].replace("m", ""))

    # Convert categorical to mapped numbers
    Airline = airline_map[data.Airline]
    Source = source_map[data.Source]
    Destination = destination_map[data.Destination]
    Total_Stops = stops_map[data.Total_Stops]

    # Final feature vector
    final = pd.DataFrame([[
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
    ]], columns=[
        'Airline', 'Source', 'Destination', 'Total_Stops',
        'Journey_day', 'Journey_month',
        'Dep_hour', 'Dep_min',
        'Arrival_hour', 'Arrival_min',
        'Duration_hour', 'Duration_min'
    ])

    return final


@app.post("/predict")
def predict(data: FlightInput):
    X = preprocess(data)
    fare = model.predict(X)[0]
    return {"predicted_price": float(fare)}
