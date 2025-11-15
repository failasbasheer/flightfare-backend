from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import datetime

# ------------------------------------------------
# FASTAPI INIT
# ------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
model = joblib.load("flight_rf.pkl")

# ------------------------------------------------
# USER INPUT (6 FIELDS)
# ------------------------------------------------
class UserInput(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Total_Stops: str
    Time_Slot: str
    Date_of_Journey: str  # YYYY-MM-DD

# ------------------------------------------------
# TIME SLOT → DEPARTURE TIME
# ------------------------------------------------
TIME_SLOT = {
    "early_morning": (5, 0),
    "morning": (9, 0),
    "afternoon": (14, 0),
    "evening": (19, 0),
    "night": (23, 0)
}

DEFAULT_SLOT = (9, 0)

# ------------------------------------------------
# ROUTE → DURATION MAP (ALL VALID ROUTES FROM DATASET)
# Realistic durations
# ------------------------------------------------
ROUTE_DURATION = {

    # ---------------------------
    # DELHI ROUTES
    # ---------------------------
    ("Delhi", "Banglore"): (2, 50),
    ("Delhi", "Kolkata"): (2, 15),
    ("Delhi", "Chennai"): (2, 55),
    ("Delhi", "Mumbai"): (2, 5),
    ("Delhi", "Cochin"): (3, 20),
    ("Delhi", "Hyderabad"): (2, 0),
    ("Delhi", "New Delhi"): (0, 50),

    ("Banglore", "Delhi"): (2, 50),
    ("Kolkata", "Delhi"): (2, 15),
    ("Chennai", "Delhi"): (2, 55),
    ("Mumbai", "Delhi"): (2, 5),
    ("Cochin", "Delhi"): (3, 20),
    ("Hyderabad", "Delhi"): (2, 0),
    ("New Delhi", "Delhi"): (0, 50),

    # ---------------------------
    # BANGLORE ROUTES
    # ---------------------------
    ("Banglore", "Mumbai"): (1, 35),
    ("Banglore", "Kolkata"): (2, 30),
    ("Banglore", "Chennai"): (1, 0),
    ("Banglore", "Cochin"): (1, 10),
    ("Banglore", "Hyderabad"): (1, 0),
    ("Banglore", "New Delhi"): (2, 55),

    ("Mumbai", "Banglore"): (1, 35),
    ("Kolkata", "Banglore"): (2, 30),
    ("Chennai", "Banglore"): (1, 0),
    ("Cochin", "Banglore"): (1, 10),
    ("Hyderabad", "Banglore"): (1, 0),
    ("New Delhi", "Banglore"): (2, 55),

    # ---------------------------
    # KOLKATA ROUTES
    # ---------------------------
    ("Kolkata", "Mumbai"): (2, 40),
    ("Kolkata", "Chennai"): (2, 20),
    ("Kolkata", "Cochin"): (3, 0),
    ("Kolkata", "Hyderabad"): (2, 0),
    ("Kolkata", "New Delhi"): (2, 20),

    ("Mumbai", "Kolkata"): (2, 40),
    ("Chennai", "Kolkata"): (2, 20),
    ("Cochin", "Kolkata"): (3, 0),
    ("Hyderabad", "Kolkata"): (2, 0),
    ("New Delhi", "Kolkata"): (2, 20),

    # ---------------------------
    # MUMBAI ROUTES
    # ---------------------------
    ("Mumbai", "Chennai"): (1, 40),
    ("Mumbai", "Cochin"): (1, 40),
    ("Mumbai", "Hyderabad"): (1, 30),
    ("Mumbai", "New Delhi"): (2, 0),

    ("Chennai", "Mumbai"): (1, 40),
    ("Cochin", "Mumbai"): (1, 40),
    ("Hyderabad", "Mumbai"): (1, 30),
    ("New Delhi", "Mumbai"): (2, 0),

    # ---------------------------
    # CHENNAI ROUTES
    # ---------------------------
    ("Chennai", "Cochin"): (1, 15),
    ("Chennai", "Hyderabad"): (1, 15),
    ("Chennai", "New Delhi"): (2, 50),

    ("Cochin", "Chennai"): (1, 15),
    ("Hyderabad", "Chennai"): (1, 15),
    ("New Delhi", "Chennai"): (2, 50),

    # ---------------------------
    # HYDERABAD ROUTES
    # ---------------------------
    ("Hyderabad", "Cochin"): (1, 30),
    ("Hyderabad", "New Delhi"): (2, 0),

    ("Cochin", "Hyderabad"): (1, 30),
    ("New Delhi", "Hyderabad"): (2, 0),
}

# ------------------------------------------------
# LABEL ENCODING (Matches model training EXACTLY)
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
    'Banglore': 2,
    'Kolkata': 1,
    'Delhi': 0,
    'Chennai': 4,
    'Mumbai': 3
}

destination_map = {
    'New Delhi': 3,
    'Banglore': 1,
    'Cochin': 0,
    'Kolkata': 5,
    'Delhi': 2,
    'Hyderabad': 4
}

stops_map = {
    'non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4
}

# ------------------------------------------------
# BUILD MODEL FEATURES (Final 12 fields)
# ------------------------------------------------
def build_features(data: UserInput):

    # Parse date
    d = datetime.datetime.strptime(data.Date_of_Journey, "%Y-%m-%d")
    Journey_day = d.day
    Journey_month = d.month

    # Time bucket → departure time
    Dep_hour, Dep_min = TIME_SLOT.get(data.Time_Slot, DEFAULT_SLOT)

    # Duration based on exact dataset routes
    Duration_hour, Duration_min = ROUTE_DURATION.get(
        (data.Source, data.Destination),
        (2, 30)  # fallback
    )

    # Compute arrival time
    dep_dt = datetime.datetime(2024, 1, 1, Dep_hour, Dep_min)
    arr_dt = dep_dt + datetime.timedelta(
        hours=Duration_hour,
        minutes=Duration_min
    )

    Arrival_hour = arr_dt.hour
    Arrival_min = arr_dt.minute

    # Encode categorical features
    Airline = airline_map[data.Airline]
    Source = source_map[data.Source]
    Destination = destination_map[data.Destination]
    Total_Stops = stops_map[data.Total_Stops]

    return np.array([[
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

# ------------------------------------------------
# PREDICT ENDPOINT
# ------------------------------------------------
@app.post("/predict")
def predict(data: UserInput):
    X = build_features(data)
    prediction = model.predict(X)[0]
    return {"predicted_price": float(prediction)}
