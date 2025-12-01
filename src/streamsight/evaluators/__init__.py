"""Evaluator module for algorithm evaluation in streaming settings.

This module provides classes and utilities for evaluating recommendation algorithms
in streaming environments. It supports both batch pipeline evaluation and interactive
streaming evaluation with comprehensive metric computation and result analysis.

Evaluator Builder
=================

The evaluator module contains builder classes for constructing evaluator objects.
The builders provide a fluent API for configuring evaluators with proper validation
and error checking.

For detailed information about the builder classes and usage examples,
see the `streamsight.evaluators.builder` module.

Available Builders:
- `Builder`: Abstract base class for all builder implementations
- `EvaluatorPipelineBuilder`: Builder for pipeline evaluators
- `EvaluatorStreamerBuilder`: Builder for streaming evaluators

Evaluator Classes
=================

The core evaluator classes handle the evaluation of recommendation algorithms
on streaming data. The evaluators manage data splitting, algorithm training,
prediction generation, and metric computation.

EvaluatorPipeline
-----------------

For batch evaluation of multiple algorithms on static or sliding window settings.
This evaluator runs algorithms through a complete pipeline including training,
prediction, and evaluation phases.

EvaluatorStreamer
-----------------

For interactive streaming evaluation where algorithms can be registered and
evaluated in real-time as data streams in. This allows for more flexible
evaluation scenarios where algorithms can request data and submit predictions
asynchronously.

Both evaluators inherit from `EvaluatorBase` which provides common functionality
for metric computation, data masking, and result aggregation.

Key Features
------------

- **Multi-algorithm evaluation**: Evaluate multiple algorithms simultaneously
- **Streaming support**: Handle temporal data streams with sliding windows
- **Metric aggregation**: Compute metrics at different levels (user, window, macro, micro)
- **Data masking**: Properly handle unknown users and items during evaluation
- **Result analysis**: Rich DataFrame outputs for metric analysis and comparison

Basic Usage
-----------

Pipeline Evaluation:

```python
from streamsight.evaluators import EvaluatorPipeline
from streamsight.evaluators.builder import EvaluatorPipelineBuilder
from streamsight.settings import Setting
from streamsight.datasets import AmazonMusicDataset

# Load data and create setting
dataset = AmazonMusicDataset()
data = dataset.load()
setting = Setting(data=data, top_K=10)
setting.split()

# Build evaluator
builder = EvaluatorPipelineBuilder()
builder.add_setting(setting)
builder.set_metric_K(10)
builder.add_metric("PrecisionK")
builder.add_algorithm("MostPop")
evaluator = builder.build()

# Run evaluation
evaluator.run()

# Get results
results = evaluator.metric_results(level="macro")
```

Streaming Evaluation:

```python
from streamsight.evaluators import EvaluatorStreamer
from streamsight.evaluators.builder import EvaluatorStreamerBuilder
from streamsight.algorithms import MostPop

# Build streaming evaluator
builder = EvaluatorStreamerBuilder()
builder.add_setting(setting)
builder.set_metric_K(10)
builder.add_metric("HitK")
evaluator = builder.build()

# Start streaming
evaluator.start_stream()

# Register algorithm
algo_id = evaluator.register_algorithm(MostPop())

# Stream evaluation loop
while True:
    try:
        # Get training data
        training_data = evaluator.get_training_data(algo_id)

        # Get unlabeled data
        unlabeled_data = evaluator.get_unlabeled_data(algo_id)

        # Algorithm makes predictions
        predictions = algorithm.predict(unlabeled_data)

        # Submit predictions
        evaluator.submit_prediction(algo_id, predictions)

    except EOWSettingError:
        break
```

Metric Levels
-------------

Evaluators support computing metrics at different aggregation levels:

- **User level**: Metrics computed per user across all timestamps
- **Window level**: Metrics computed per time window, averaging user scores within each window
- **Macro level**: Overall metrics averaging across all windows equally
- **Micro level**: Overall metrics averaging across all user-timestamp combinations equally

Available Evaluator Classes:
- `EvaluatorBase`: Base class providing common evaluation functionality
- `EvaluatorPipeline`: Pipeline-based batch evaluator
- `EvaluatorStreamer`: Interactive streaming evaluator

Accumulator
==========

The `MetricAccumulator` class accumulates and stores metric results during
evaluation. The accumulator maintains a collection of metric objects organized
by algorithm name and provides methods for retrieving results in various formats.

Features:
- Storing metrics for multiple algorithms
- Computing aggregated results at different levels
- Exporting results to pandas DataFrames
- Filtering results by algorithm, timestamp, or metric type

Utility Classes
===============

Utility classes that support the evaluation process:

- `MetricLevelEnum`: Enumeration for different metric aggregation levels
- `UserItemBaseStatus`: Tracks known and unknown users/items during evaluation
- `AlgorithmStatusWarning`: Custom warning for algorithm state issues

These utilities handle the complex state management required for streaming
evaluation scenarios.
"""

from .accumulator import MetricAccumulator
from .base import EvaluatorBase
from .builder import (
    Builder,
    EvaluatorPipelineBuilder,
    EvaluatorStreamerBuilder,
)
from .evaluator_pipeline import EvaluatorPipeline
from .evaluator_stream import EvaluatorStreamer
from .util import MetricLevelEnum, UserItemBaseStatus
from .warning import AlgorithmStatusWarning


__all__ = [
    "Builder",
    "EvaluatorPipelineBuilder",
    "EvaluatorStreamerBuilder",
    "EvaluatorBase",
    "EvaluatorPipeline",
    "EvaluatorStreamer",
    "MetricAccumulator",
    "MetricLevelEnum",
    "UserItemBaseStatus",
    "AlgorithmStatusWarning",
]


__all__ = [
    "Builder",
    "EvaluatorPipelineBuilder",
    "EvaluatorStreamerBuilder",
    "EvaluatorBase",
    "EvaluatorPipeline",
    "EvaluatorStreamer",
    "MetricAccumulator",
    "MetricLevelEnum",
    "UserItemBaseStatus",
    "AlgorithmStatusWarning",
]
