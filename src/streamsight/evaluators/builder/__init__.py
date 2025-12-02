"""Builder module for constructing evaluator objects.

This module provides builder classes for constructing evaluator objects in the
Streamsight library. Builders follow the builder pattern to facilitate the
construction of evaluators with proper validation and error checking.

## Builder Overview

The builder pattern is used to construct complex evaluator objects step by step.
Builders ensure that all necessary components (settings, metrics, algorithms)
are properly configured before building the evaluator, preventing runtime errors.

## Available Builders

- `Builder`: Abstract base class for all builder implementations
- `EvaluatorPipelineBuilder`: Builder for pipeline evaluators that evaluate
  multiple algorithms on static data
- `EvaluatorStreamerBuilder`: Builder for streaming evaluators that evaluate
  algorithms on streaming data

## Using Builders

### Basic Pipeline Evaluation

To evaluate multiple algorithms on a static dataset using a pipeline evaluator:

```python
from streamsight.evaluators.builder import EvaluatorPipelineBuilder
from streamsight.settings import Setting
from streamsight.datasets import AmazonMusicDataset

# Load dataset
dataset = AmazonMusicDataset()
data = dataset.load()

# Create setting
setting = Setting(data=data, top_K=10)
setting.split()

# Build evaluator
builder = EvaluatorPipelineBuilder(seed=42)
builder.add_setting(setting)
builder.set_metric_K(10)
builder.add_metric("PrecisionK")
builder.add_metric("RecallK")
builder.add_algorithm("MostPop")
builder.add_algorithm("RecentPop", params={"K": 10})

evaluator = builder.build()
results = evaluator.evaluate()
```

### Streaming Evaluation

To evaluate algorithms on streaming data:

```python
from streamsight.evaluators.builder import EvaluatorStreamerBuilder
from streamsight.settings import StreamingSetting
from streamsight.datasets import AmazonMusicDataset

# Load dataset
dataset = AmazonMusicDataset()
data = dataset.load()

# Create streaming setting
setting = StreamingSetting(data=data, top_K=10, window_size=1000)
setting.split()

# Build streaming evaluator
builder = EvaluatorStreamerBuilder(seed=42)
builder.add_setting(setting)
builder.set_metric_K(10)
builder.add_metric("HitK")
builder.add_metric("NDCGK")

evaluator = builder.build()
# The evaluator can now process streaming data
```

### Advanced Configuration

Builders support advanced configuration options:

```python
from streamsight.evaluators.builder import EvaluatorPipelineBuilder

builder = EvaluatorPipelineBuilder(
    ignore_unknown_user=False,  # Don't ignore unknown users
    ignore_unknown_item=True,   # Ignore unknown items
    seed=123
)

builder.add_setting(setting)
builder.set_metric_K(20)

# Add multiple metrics
metrics = ["PrecisionK", "RecallK", "DCGK", "NDCGK", "HitK"]
for metric in metrics:
    builder.add_metric(metric)

# Add algorithms with custom parameters
builder.add_algorithm("ItemKNN", params={"K": 50, "similarity": "cosine"})
builder.add_algorithm("DecayPop", params={"decay_factor": 0.9})

evaluator = builder.build()
```

## Extending the Framework

To create custom builders, inherit from the `Builder` base class and implement
the `build()` method. Ensure to call `super().__init__()` and implement proper
validation in `_check_ready()`.
"""

from .base import Builder
from .pipeline import EvaluatorPipelineBuilder
from .stream import EvaluatorStreamerBuilder


__all__ = [
    "Builder",
    "EvaluatorPipelineBuilder",
    "EvaluatorStreamerBuilder",
]
