from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_store_sales_weekly, preprocess_stores, preprocess_sales


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_stores,
                inputs="stores",
                outputs="preprocessed_stores",
                name="preprocess_stores_node",
            ),
            node(
                func=preprocess_sales,
                inputs="sales",
                outputs="preprocessed_sales",
                name="preprocess_sales_node",
            ),
            node(
                func=create_store_sales_weekly,
                inputs=["preprocessed_sales", "preprocessed_stores", "calendar"],
                outputs="store_sales_weekly",
                name="create_store_sales_weekly_node",
            ),
        ]
    )
