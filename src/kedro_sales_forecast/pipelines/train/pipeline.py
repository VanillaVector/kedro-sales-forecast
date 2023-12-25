from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["store_sales_weekly", "params:model_options"],
                outputs=["train_data", "test_data"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["train_data", "params:model_options"],
                outputs="forecast_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["forecast_model", "test_data", "params:model_options"],
                outputs=["pred", "metrics"],
                name="evaluate_model_node",
            ),
        ]
    )
