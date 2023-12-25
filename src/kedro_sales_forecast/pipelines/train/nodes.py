import pandas as pd
import numpy as np
import logging
import json

from typing import Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from window_ops.rolling import rolling_mean, rolling_max, rolling_min


logger = logging.getLogger(__name__)

def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into train and test sets sorted.

    Args:
        data: Data containing static features and exogenous/dynamic features.
        parameters: Parameters defined in conf/base/parameters_data_science.yml.
    Returns:
        Split data.
    """
    split_date = parameters["split_date"]
    train_data = data.loc[data["ds"] < split_date]
    test_data = data.loc[data["ds"] >= split_date]
    
    train_data = train_data.sort_values(by=["ds", "unique_id"])
    test_data = test_data.sort_values(by=["ds", "unique_id"])
    
    return train_data, test_data

def train_model(train_data: pd.DataFrame, parameters: Dict) -> MLForecast:
    """Trains the model.

    Args:
        train_data: Training data.
        parameters: Parameters defined in conf/base/parameters_data_science.yml.

    Returns:
        Trained model.
    """
    horizon = parameters["horizon"]
    num_threads = parameters["num_threads"]
    random_state = parameters["random_state"]
    n_estimators = parameters["n_estimators"]
    freqency = parameters["freqency"]
    lags = parameters["lags"]
    date_features = parameters["date_features"]
    static_features = parameters["static_features"]
    inner_models = [make_pipeline(SimpleImputer(), RandomForestRegressor(random_state=random_state, n_estimators=n_estimators)), 
                    XGBRegressor(random_state=random_state, n_estimators=n_estimators),
                    LGBMRegressor(random_state=random_state, n_estimators=n_estimators),
                   ]

    model = MLForecast( models=inner_models,
                        freq=freqency,
                        lags=lags,
                        lag_transforms={
                            1: [(rolling_mean, horizon), (rolling_min, horizon), (rolling_max, horizon)],
                            2: [(rolling_mean, horizon), (rolling_min, horizon), (rolling_max, horizon)],
                            4: [(rolling_mean, horizon), (rolling_min, horizon), (rolling_max, horizon)],
                        },
                        date_features=date_features,
                        num_threads=num_threads
                       )
    model.fit(train_data, static_features=static_features)
    return model

def evaluate_model(model: MLForecast, test_data: pd.DataFrame, parameters: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate model and export Metrics.

    Args:
        model: Trained model.
        test_data: Testing data.
        parameters: Parameters defined in conf/base/parameters_data_science.yml.
    """
    horizon = parameters["horizon"]
    exogenous_features = parameters["exogenous_features"]
    future_data = test_data[["unique_id", "ds"] + exogenous_features]
    
    pred = model.predict(h=horizon, X_df=future_data)
    pred = pred.rename(columns={'Pipeline':'RandomForestRegressor'})
    pred = pred.merge(test_data[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")
    
    metrics = {'RandomForestRegressor': _smape(pred['y'], pred['RandomForestRegressor']), 
                'XGBRegressor': _smape(pred['y'], pred['XGBRegressor']), 
                'LGBMRegressor': _smape(pred['y'], pred['LGBMRegressor'])}
    metrics_str = json.dumps(metrics)
    
    for k, v in metrics.items():
        logger.info(f"SMAPE of model -  {k} is: {v}")
    
    return pred, metrics_str

def _smape(y_true, y_pred):
    return 100 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
