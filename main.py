from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Tuple
import joblib
import numpy as np
import datetime
import warnings
from functools import lru_cache

# ------------------------------------------------
# GLOBAL WARNING CONTROL (silence sklearn spam)
# ------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but",
    category=UserWarning,
)

# ------------------------------------------------
# FASTAPI INIT
# ------------------------------------------------
app = FastAPI(
    title="Flight Fare Prediction API",
    version="1.0.0",
    description="AI-powered domestic flight fare prediction",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# MODEL LOADING (CACHED)
# ------------------------------------------------
@lru_cache(maxsize=1)
def get_model():
    """Load the model once and cache it."""
    try:
        model = joblib.load("flight_rf.pkl")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    return model


# ------------------------------------------------
# CONSTANTS / ENCODING MAPS
# ------------------------------------------------
TIME_SLOT: Dict[str, Tuple[int, int]] = {
    "early_morning": (5, 0),
    "morning": (9, 0),
    "afternoon": (14, 0),
    "evening": (19, 0),
    "night": (23, 0),
}
DEFAULT_SLOT: Tuple[int, int] = (9, 0)

ROUTE_DURATION: Dict[Tuple[str, str], Tuple[int, int]] = {
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

airline_map: Dict[str, int] = {
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
    'Trujet': 11,
}

source_map: Dict[str, int] = {
    'Banglore': 2,
    'Kolkata': 1,
    'Delhi': 0,
    'Chennai': 4,
    'Mumbai': 3,
}

destination_map: Dict[str, int] = {
    'New Delhi': 3,
    'Banglore': 1,
    'Cochin': 0,
    'Kolkata': 5,
    'Delhi': 2,
    'Hyderabad': 4,
}

stops_map: Dict[str, int] = {
    'non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4,
}

ALL_AIRLINES: List[str] = list(airline_map.keys())

# ------------------------------------------------
# SCHEMAS
# ------------------------------------------------
class UserInput(BaseModel):
    Airline: str = Field(..., example="IndiGo")
    Source: str = Field(..., example="Delhi")
    Destination: str = Field(..., example="Banglore")
    Total_Stops: str = Field(..., example="1 stop")
    Time_Slot: str = Field(..., example="morning")
    Date_of_Journey: str = Field(..., example="2025-01-15")  # YYYY-MM-DD


class NearbyDayPrice(BaseModel):
    date: str
    price: float


class MultiStopPrice(BaseModel):
    stops: str
    price: float


class AirlinePrice(BaseModel):
    airline: str
    price: float


class PredictionResponse(BaseModel):
    predicted_price: float
    nearby_days: List[NearbyDayPrice]
    multi_stop: List[MultiStopPrice]
    airline_comparison: List[AirlinePrice]


# ------------------------------------------------
# VALIDATION HELPERS
# ------------------------------------------------
def parse_date(date_str: str) -> datetime.datetime:
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format '{date_str}'. Use YYYY-MM-DD.",
        )


def validate_input(data: UserInput) -> None:
    if data.Airline not in airline_map:
        raise HTTPException(400, detail=f"Unsupported airline: {data.Airline}")

    if data.Source not in source_map:
        raise HTTPException(400, detail=f"Unsupported source: {data.Source}")

    if data.Destination not in destination_map:
        raise HTTPException(400, detail=f"Unsupported destination: {data.Destination}")

    if data.Total_Stops not in stops_map:
        raise HTTPException(400, detail=f"Unsupported stops value: {data.Total_Stops}")

    if data.Time_Slot not in TIME_SLOT and data.Time_Slot not in TIME_SLOT.keys():
        # You only allow specific slots, no random strings
        raise HTTPException(400, detail=f"Unsupported time slot: {data.Time_Slot}")

    # Just to validate date format early
    _ = parse_date(data.Date_of_Journey)


# ------------------------------------------------
# CORE FEATURE BUILDER  (LOGIC PRESERVED)
# ------------------------------------------------
def build_features(
    data: UserInput,
    *,
    override_date: str | None = None,
    override_airline: str | None = None,
    override_stops: str | None = None,
) -> np.ndarray:
    """
    Build the 1x12 feature array for the model.
    Logic matches your original version.
    """

    # Date
    date_str = override_date or data.Date_of_Journey
    d = parse_date(date_str)
    Journey_day = d.day
    Journey_month = d.month

    # Time slot
    Dep_hour, Dep_min = TIME_SLOT.get(data.Time_Slot, DEFAULT_SLOT)

    # Duration
    route = (data.Source, data.Destination)
    Duration_hour, Duration_min = ROUTE_DURATION.get(route, (2, 30))

    # Arrival
    dep = datetime.datetime(2024, 1, 1, Dep_hour, Dep_min)
    arr = dep + datetime.timedelta(hours=Duration_hour, minutes=Duration_min)
    Arrival_hour = arr.hour
    Arrival_min = arr.minute

    # Encodings
    airline_key = override_airline or data.Airline
    stops_key = override_stops or data.Total_Stops

    Airline = airline_map[airline_key]
    Source = source_map[data.Source]
    Destination = destination_map[data.Destination]
    Stops = stops_map[stops_key]

    return np.array(
        [
            [
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
                Duration_min,
            ]
        ],
        dtype=float,
    )


# ------------------------------------------------
# NEARBY DATES
# ------------------------------------------------
def get_nearby_dates(date_str: str) -> List[str]:
    base = parse_date(date_str)
    return [
        (base + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        for i in [-2, -1, 0, 1, 2]
    ]


# ------------------------------------------------
# VECTORIZED PREDICTION HELPERS  (SPEED BOOST)
# ------------------------------------------------
def predict_batch(features: np.ndarray) -> np.ndarray:
    """
    One call to the model for many rows.
    This is where we win a lot of time vs spamming model.predict in loops.
    """
    model = get_model()
    return model.predict(features)


# ------------------------------------------------
# REQUEST-LEVEL CACHING (IDEMPOTENT)
# ------------------------------------------------
@lru_cache(maxsize=2048)
def cached_prediction(input_signature: str) -> PredictionResponse:
    """
    Cache full responses per unique input.
    input_signature is just the JSON string of the request.
    """
    # Parse back into UserInput
    try:
        data = UserInput.model_validate_json(input_signature)
    except ValidationError as e:
        # This shouldn't fire for real requests, only if we mess up
        raise HTTPException(400, detail=str(e))

    # We call the internal compute function
    return _compute_prediction(data)


# ------------------------------------------------
# CORE PREDICTION LOGIC (kept separate)
# ------------------------------------------------
def _compute_prediction(data: UserInput) -> PredictionResponse:
    """
    Core logic that builds features and calls the model.
    Same semantics as your original /predict, but vectorized.
    """

    # MAIN PREDICTION
    main_features = build_features(data)
    main_price = float(predict_batch(main_features)[0])

    # NEARBY DAYS (vectorized)
    nearby_dates = get_nearby_dates(data.Date_of_Journey)
    nearby_feature_rows = [build_features(data, override_date=d)[0] for d in nearby_dates]
    nearby_features = np.vstack(nearby_feature_rows)
    nearby_prices = predict_batch(nearby_features)

    nearby_days = [
        NearbyDayPrice(date=d, price=float(p))
        for d, p in zip(nearby_dates, nearby_prices)
    ]

    # MULTI STOP PRICES (vectorized)
    stop_options = ["non-stop", "1 stop", "2 stops"]
    multi_rows = [
        build_features(data, override_stops=st)[0]
        for st in stop_options
    ]
    multi_features = np.vstack(multi_rows)
    multi_prices = predict_batch(multi_features)

    multi_stop = [
        MultiStopPrice(stops=st, price=float(p))
        for st, p in zip(stop_options, multi_prices)
    ]

    # AIRLINE COMPARISON (vectorized)
    airline_rows = [
        build_features(data, override_airline=airline)[0]
        for airline in ALL_AIRLINES
    ]
    airline_features = np.vstack(airline_rows)
    airline_prices = predict_batch(airline_features)

    airline_comparison = [
        AirlinePrice(airline=airline, price=float(p))
        for airline, p in zip(ALL_AIRLINES, airline_prices)
    ]
    airline_comparison.sort(key=lambda x: x.price)

    return PredictionResponse(
        predicted_price=main_price,
        nearby_days=nearby_days,
        multi_stop=multi_stop,
        airline_comparison=airline_comparison,
    )


# ------------------------------------------------
# ROUTES
# ------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: UserInput):
    """
    Main prediction endpoint.
    - Validates input
    - Uses cached responses when possible
    - Handles failures cleanly
    """
    # Validate before anything
    validate_input(data)

    # Build a stable cache key (JSON string)
    input_signature = data.model_dump_json()

    try:
        response = cached_prediction(input_signature)
    except HTTPException:
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal prediction error: {e}",
        )

    return response
