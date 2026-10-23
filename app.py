
from pathlib import Path
from datetime import date, timedelta
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_linear_regression_model.pkl"
FEATURES_PATH = BASE_DIR / "final_model_features.pkl"

app = FastAPI(
    title="Daily Demand Forecasting API",
    description="Predict the next 7 working days from one set of inputs.",
    version="2.0.0",
)

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, "rb") as f:
        feature_names = list(pickle.load(f))

except Exception as exc:
    raise RuntimeError(f"Could not load model files: {exc}") from exc


EXPECTED_FEATURES = [
    "Week of the month (first week, second, third, fourth or fifth week",
    "Day of the week (Monday to Friday)",
    "Non-urgent order",
    "Urgent order",
    "Fiscal sector orders",
    "Orders from the traffic controller sector",
    "Banking orders (1)",
    "Banking orders (2)",
    "Banking orders (3)",
]

if feature_names != EXPECTED_FEATURES:
    raise RuntimeError(
        "Saved feature names/order do not match the expected 9 features. "
        f"Loaded features: {feature_names}"
    )


class OrderInput(BaseModel):
    non_urgent_order: float = Field(ge=0)
    urgent_order: float = Field(ge=0)
    fiscal_sector_orders: float = Field(ge=0)
    traffic_controller_orders: float = Field(ge=0)
    banking_orders_1: float = Field(ge=0)
    banking_orders_2: float = Field(ge=0)
    banking_orders_3: float = Field(ge=0)


class ForecastRequest(BaseModel):
    start_date: date
    inputs: OrderInput


def dataset_week_value(day_date: date) -> int:
    """Approximate week-of-month category, from 1 to 5."""
    return min(((day_date.day - 1) // 7) + 1, 5)


def dataset_day_value(day_date: date) -> int:
    """
    Dataset contains weekday codes 2–6.
    Assumption: Monday=2, Tuesday=3, Wednesday=4,
    Thursday=5, Friday=6.
    """
    return day_date.weekday() + 2


def make_feature_frame(
    day_date: date,
    inputs: OrderInput,
) -> pd.DataFrame:
    values = {
        "Week of the month (first week, second, third, fourth or fifth week":
            dataset_week_value(day_date),
        "Day of the week (Monday to Friday)":
            dataset_day_value(day_date),
        "Non-urgent order": inputs.non_urgent_order,
        "Urgent order": inputs.urgent_order,
        "Fiscal sector orders": inputs.fiscal_sector_orders,
        "Orders from the traffic controller sector":
            inputs.traffic_controller_orders,
        "Banking orders (1)": inputs.banking_orders_1,
        "Banking orders (2)": inputs.banking_orders_2,
        "Banking orders (3)": inputs.banking_orders_3,
    }

    return pd.DataFrame(
        [[values[name] for name in feature_names]],
        columns=feature_names,
    )


def predict_for_date(day_date: date, inputs: OrderInput) -> float:
    frame = make_feature_frame(day_date, inputs)
    prediction = float(model.predict(frame)[0])

    if not np.isfinite(prediction):
        raise HTTPException(
            status_code=500,
            detail=f"Invalid prediction for {day_date.isoformat()}",
        )

    return prediction


@app.get("/")
def home():
    return {"message": "Daily Demand Forecasting API is running"}


@app.get("/features")
def get_features():
    return {"feature_count": len(feature_names), "features": feature_names}


@app.post("/forecast/7-days")
def forecast_seven_days(request: ForecastRequest):
    forecasts = []
    current_date = request.start_date
    working_days_added = 0

    # Return 7 weekdays, skipping Saturday and Sunday.
    while working_days_added < 7:
        if current_date.weekday() < 5:
            prediction = predict_for_date(current_date, request.inputs)

            row = {
                "date": current_date.isoformat(),
                "day_name": current_date.strftime("%A"),
                "predicted_total_orders": round(prediction, 2),
            }

            if prediction < 0:
                row["warning"] = (
                    "Negative model estimate. Review the inputs and model."
                )

            forecasts.append(row)
            working_days_added += 1

        current_date += timedelta(days=1)

    return {
        "message": (
            "Seven working-day estimates. Order-related inputs are held "
            "constant across dates; only date-derived features change. "
            "This is not a fully automated future-demand model."
        ),
        "forecast": forecasts,
    }


@app.post("/predict")
def predict_single(request: ForecastRequest):
    prediction = predict_for_date(request.start_date, request.inputs)

    result = {
        "date": request.start_date.isoformat(),
        "predicted_total_orders": round(prediction, 2),
    }

    if prediction < 0:
        result["warning"] = (
            "Negative estimate. Review the inputs and model suitability."
        )

    return result


