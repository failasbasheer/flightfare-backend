from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import datetime

# ------------------------------------------------
# FASTAPI APP + CORS (REQUIRED FOR FRONTEND ACCESS)
# ------------------------------------------------
app = FastAPI()

# 🔥 CORS FIX - THIS IS WHAT SOLVES YOUR ERROR
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for stricter use
    allow_credentials=True,
    allow_methods=["*"],   # allow POST, GET, OPTIONS
    allow_headers=["*"],
)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
model = joblib.load("flight_rf.pkl")


# ------------------------------------------------
# INPUT SCHEMAS
# ------------------------------------------------

# What frontend sends
class UserInput(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Total_Stops: str
    Date_of_Journey: str
    Time_Slot: str   # user selects morning / evening etc.


# What model internally needs
class ModelInput(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Total_Stops: str
    Date_of_Journey: str
    Dep_Time: str
    Arrival_Time: str
    Duration: str


# ------------------------------------------------
# TIME SLOT MAPPING
# ------------------------------------------------
def map_time_slot(slot: str):
    slot = slot.lower()

    if slot == "early_morning":
        return ("05:00", "07:30", "2h 30m")
    elif slot == "morning":
        return ("09:00", "11:30", "2h 30m")
    elif slot == "afternoon":
        return ("14:00", "16:30", "2h 30m")
    elif slot == "evening":
        return ("19:00", "21:30", "2h 30m")

    # fallback default
    return ("09:00", "11:30", "2h 30m")


# ------------------------------------------------
# LABEL ENCODERS
# ------------------------------------------------

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


# ------------------------------------------------
# PREPROCESS FUNCTION
# ------------------------------------------------
def preprocess(data: ModelInput):
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


# ------------------------------------------------
# PREDICT ENDPOINT
# ------------------------------------------------
@app.post("/predict")
def predict(data: UserInput):

    # Map simple time-slot → actual model fields
    dep, arr, dur = map_time_slot(data.Time_Slot)

    # Convert to the old internal model schema
    new_data = ModelInput(
        Airline=data.Airline,
        Source=data.Source,
        Destination=data.Destination,
        Total_Stops=data.Total_Stops,
        Date_of_Journey=data.Date_of_Journey,
        Dep_Time=dep,
        Arrival_Time=arr,
        Duration=dur
    )

    X = preprocess(new_data)
    prediction = model.predict(X)[0]

    return {"predicted_price": float(prediction)}
