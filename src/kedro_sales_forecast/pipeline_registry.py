"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    
    pipelines["all"] = sum(
        pipelines.values(),  # join all pipelines in the default
        start=Pipeline([]),  # use an empty pipeline if pipelines dict is empty
    )
    pipelines["__default__"] = Pipeline(
        [node for node in pipelines["all"].nodes if "predict" not in node.name]
    )
    
    return pipelines
