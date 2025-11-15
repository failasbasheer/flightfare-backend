from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import datetime
from typing import List, Dict

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
# ROUTE → DURATION MAP (ALL VALID ROUTES)
# ------------------------------------------------
ROUTE_DURATION = {
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

    ("Banglore", "Mumbai"): (1, 35),
    ("Banglore", "Kolkata"): (2, 30),
    ("Banglore", "Chennai"): (1, 0),
    ("Banglore", "Cochin"): (1, 10),
    ("Banglore", "Hyderabad"): (1, 0),
    ("Banglore", "New Delhi"): (2, 55),

    ("Kolkata", "Mumbai"): (2, 40),
    ("Kolkata", "Chennai"): (2, 20),
    ("Kolkata", "Cochin"): (3, 0),
    ("Kolkata", "Hyderabad"): (2, 0),
    ("Kolkata", "New Delhi"): (2, 20),

    ("Mumbai", "Chennai"): (1, 40),
    ("Mumbai", "Cochin"): (1, 40),
    ("Mumbai", "Hyderabad"): (1, 30),
    ("Mumbai", "New Delhi"): (2, 0),

    ("Chennai", "Cochin"): (1, 15),
    ("Chennai", "Hyderabad"): (1, 15),
    ("Chennai", "New Delhi"): (2, 50),

    ("Hyderabad", "Cochin"): (1, 30),
    ("Hyderabad", "New Delhi"): (2, 0),
}

# ------------------------------------------------
# ENCODING MAPS
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
# BUILD FEATURES
# ------------------------------------------------
def build_features(data: UserInput, *, override_date=None, override_airline=None, override_stops=None):
    # Date
    date_str = override_date or data.Date_of_Journey
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    Journey_day = d.day
    Journey_month = d.month

    # Time slot
    Dep_hour, Dep_min = TIME_SLOT.get(data.Time_Slot, DEFAULT_SLOT)

    # Duration (unchanged)
    route = (data.Source, data.Destination)
    Duration_hour, Duration_min = ROUTE_DURATION.get(route, (2, 30))

    # Arrival
    dep = datetime.datetime(2024, 1, 1, Dep_hour, Dep_min)
    arr = dep + datetime.timedelta(hours=Duration_hour, minutes=Duration_min)
    Arrival_hour = arr.hour
    Arrival_min = arr.minute

    # Encodings
    Airline = airline_map[override_airline or data.Airline]
    Source = source_map[data.Source]
    Destination = destination_map[data.Destination]
    Stops = stops_map[override_stops or data.Total_Stops]

    return np.array([[
        Airline,
        Source,
        Destination,
        Stops,
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
# NEARBY DATES
# ------------------------------------------------
def get_nearby_dates(date_str):
    base = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return [(base + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in [-2, -1, 0, 1, 2]]

# ------------------------------------------------
# ALL AIRLINES
# ------------------------------------------------
ALL_AIRLINES = list(airline_map.keys())

# ------------------------------------------------
# PREDICT ENDPOINT (EXTENDED)
# ------------------------------------------------
@app.post("/predict")
def predict(data: UserInput):

    # MAIN PREDICTION
    main_features = build_features(data)
    main_price = float(model.predict(main_features)[0])

    # NEARBY DAY PRICES
    nearby = []
    for d in get_nearby_dates(data.Date_of_Journey):
        f = build_features(data, override_date=d)
        price = float(model.predict(f)[0])
        nearby.append({"date": d, "price": price})

    # MULTI STOP PRICES
    multi = []
    for stops in ["non-stop", "1 stop", "2 stops"]:
        f = build_features(data, override_stops=stops)
        price = float(model.predict(f)[0])
        multi.append({"stops": stops, "price": price})

    # AIRLINE COMPARISON
    airlines = []
    for airline in ALL_AIRLINES:
        f = build_features(data, override_airline=airline)
        price = float(model.predict(f)[0])
        airlines.append({"airline": airline, "price": price})

    airlines = sorted(airlines, key=lambda x: x["price"])

    # FULL RESPONSE
    return {
        "predicted_price": main_price,
        "nearby_days": nearby,
        "multi_stop": multi,
        "airline_comparison": airlines
    }
