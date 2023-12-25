import pandas as pd
import pytest

from kedro_sales_forecast.pipelines.train.nodes import split_data, train_model, evaluate_model
from kedro_sales_forecast.pipelines.predict.nodes import make_prediction


@pytest.fixture
def dummy_data():
    data =  pd.read_csv('tests/data/test_data.csv')
    data["ds"] = pd.to_datetime(data["ds"])
    return data

@pytest.fixture
def dummy_parameters():
    parameters = {"model_options": {
                        'split_date': '2009-05-03',
                        'random_state': 42,
                        'n_estimators': 100,
                        'num_threads': 6,
                        'freqency': 'W',
                        'horizon': 4,
                        'lags': [1, 2, 4],
                        'date_features': ['week', 'month'],
                        'static_features': ['Size', 'Type_A', 'Type_B'],
                        'exogenous_features': ['IsHoliday', 'Temperature', 'Fuel_Price', 'Unemployment', 'CPI']
                        }
                 }
    return parameters


class TestTrainPipelineNodes:
    def test_split_data(self, dummy_data, dummy_parameters):
        train_data, test_data = split_data(dummy_data, dummy_parameters["model_options"])
        assert len(train_data) == 130
        assert len(test_data) == 10

    def test_train_model(self, dummy_data, dummy_parameters):
        train_data, _ = split_data(dummy_data, dummy_parameters["model_options"])
        model = train_model(train_data, dummy_parameters["model_options"])

        assert model.freq == 'W'
        assert model.models["XGBRegressor"].n_estimators == 100
        assert model.models["LGBMRegressor"].n_estimators == 100

    def test_evaluate_model(self, dummy_data, dummy_parameters):
        train_data, test_data = split_data(dummy_data, dummy_parameters["model_options"])
        model = train_model(train_data, dummy_parameters["model_options"])
        pred, _ = evaluate_model(model, test_data, dummy_parameters["model_options"])
        metric_columns = {'RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor'}
        
        assert metric_columns.issubset(set(list(pred.columns)))


class TestPredictPipelineNodes:
    def test_make_prediction(self, dummy_data, dummy_parameters):
        train_data, test_data = split_data(dummy_data, dummy_parameters["model_options"])
        model = train_model(train_data, dummy_parameters["model_options"])
        pred = make_prediction(model, test_data, dummy_parameters["model_options"])

        metric_columns = {'RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor'}
        
        assert metric_columns.issubset(set(list(pred.columns)))
