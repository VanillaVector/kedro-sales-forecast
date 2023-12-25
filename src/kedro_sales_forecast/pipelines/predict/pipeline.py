from kedro.pipeline import Pipeline, node, pipeline
from .nodes import make_prediction


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=make_prediction,
                inputs=["forecast_model", "future_data", "params:model_options"],
                outputs="predictions",
                name="make_prediction_node",
            ),
        ]
    )
