import logging
import pandas as pd
import numpy as np
from typing import Dict
from mlforecast import MLForecast


logger = logging.getLogger(__name__)

def make_prediction(model: MLForecast, future_data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Make prediction with trained model.

    Args:
        model: Trained model.
        future_data: Data to make prediction with .
        parameters: Parameters defined in parameters/data_science.yml.

    Returns:
        Predictions of {horizon} steps.
    """
    horizon = parameters["horizon"]
    exogenous_features = parameters["exogenous_features"]
    expected_columns = ["unique_id", "ds"] + exogenous_features
    
    if not set(expected_columns).issubset(set(future_data.columns)):
        raise KeyError(f"Data to make prediction with must contain all these cloumns: {expected_columns}")
    
    future_data = future_data[expected_columns]
    pred = model.predict(h=horizon, X_df=future_data)
    pred = pred.rename(columns={'Pipeline':'RandomForestRegressor'})
    pred = pred.sort_values(by=["ds", "unique_id"])
    return pred
