"""Settings module for data splitting strategies.

The setting module contains classes that define how the data is split. To
generalize the splitting of the data, the interactions are first sorted in
temporal order, and then the split is performed based on the setting. As this
library only considers dataset with timestamp, we will consider the case of the
single time point setting and the sliding window setting. The single time point
setting is analogous to Setting 3 of [@sun2023]. The sliding window setting
is analogous to Setting 1 of [@sun2023].

![data_split_definition](../../../assets/_static/data_split_definition.png)

Observe the diagram below where the data split for Setting 1 is shown below. The
unlabeled data will contain interactions that are masked which occurs after the
current timestamp. The ground truth data will contain the actual interactions
which will be used for evaluation and then released to the algorithm.

![setting1_no_seq](../../../assets/_static/setting1_no_seq.png)

While the this setting allows us to test the algorithm in a real-world scenario,
there are times when the algorithm might require some sequential data before
a prediction can be made. While it is not the role of the evaluating platform
to provide this data, we have included the option to provide the last n interactions.

## Data Components

Each split produces three data components:

- **background_data**: Data that is used to train the algorithm before the first
  split.

- **unlabeled_data**: Data that is released to the algorithm for prediction.
  Contains the ID to be predicted and is labeled with "-1". Timestamps of the
  interactions to be predicted are preserved. Can contain the last n interactions
  split if specified in the parameter. The purpose is to provide sequential
  data to the algorithm.

- **ground_truth_data**: Data that is used to evaluate the algorithm. This data
  will contain the actual interactions. The unlabeled data with the masked data
  is a subset of the ground truth to ensure that there is an actual corresponding
  value to evaluate the prediction against.


## Available Settings

- `Setting`: Base class for data splitting settings
- `SingleTimePointSetting`: Single time point splitting strategy
- `SlidingWindowSetting`: Sliding window splitting strategy
- `LeaveNOutSetting`: Leave-N-out cross-validation strategy

## Usage

Settings are stateful. Thus, the initialization of the setting object only stores
the parameters that are passed. Calling of `Setting.split` is necessary
such that the attributes `Setting.background_data`, `Setting.unlabeled_data`
and `Setting.ground_truth_data` are populated.

```python
from streamsight.datasets import AmazonMovieDataset
from streamsight.settings import SlidingWindowSetting

dataset = AmazonMovieDataset(use_default_filters=False)
data = dataset.load()

setting = SlidingWindowSetting(
    background_t=1530000000,
    window_size=60 * 60 * 24 * 30,  # 30 days
    n_seq_data=1,
    top_K=10
)

setting.split(data)
```

## Splitters

Splitters are stateless utilities that split data into past and future interactions.
They abstract splitting logic from the setting architecture, enabling flexible
implementations.

### Available Splitters

- `TimestampSplitter`: Split data by timestamp
- `NLastInteractionTimestampSplitter`: Split using N past interactions and timestamp
- `NLastInteractionSplitter`: Split using N last interactions

## Processors

Processors handle data transformation. The current implementation masks prediction
data and injects it into unlabeled data. Custom processors can implement additional
transformations.

### Available Processors

- `Processor`: Base class for data processors
- `PredictionDataProcessor`: Masks and injects prediction data

## Exceptions

- `EOWSettingError`: Raised when end of window is reached
"""

from streamsight.settings.base import EOWSettingError, Setting
from streamsight.settings.leave_n_out_setting import LeaveNOutSetting
from streamsight.settings.processor import PredictionDataProcessor, Processor
from streamsight.settings.single_time_point_setting import SingleTimePointSetting
from streamsight.settings.sliding_window_setting import SlidingWindowSetting
from streamsight.settings.splitters import (
    NLastInteractionSplitter,
    NLastInteractionTimestampSplitter,
    TimestampSplitter,
)


__all__ = [
    "Setting",
    "SingleTimePointSetting",
    "SlidingWindowSetting",
    "LeaveNOutSetting",
    "Processor",
    "PredictionDataProcessor",
    "TimestampSplitter",
    "NLastInteractionTimestampSplitter",
    "NLastInteractionSplitter",
    "EOWSettingError",
]
