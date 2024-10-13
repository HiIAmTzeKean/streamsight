import pytest
from streamsight.datasets import TestDataset
from streamsight.settings import SlidingWindowSetting, SingleTimePointSetting
from streamsight.evaluators import EvaluatorPipelineBuilder, EvaluatorStreamerBuilder

@pytest.fixture()
def sliding_window():
    dataset = TestDataset()
    data = dataset.load()
    setting = SlidingWindowSetting(
        4,
        3
    )
    setting.split(data)
    return setting

@pytest.fixture()
def single_time_point():
    dataset = TestDataset()
    data = dataset.load()
    setting = SingleTimePointSetting(
        4,
    )
    setting.split(data)
    return setting

class TestFullRun:
    def test_sliding_window_without_unknown_user_item(self, sliding_window):
        b = EvaluatorPipelineBuilder(True,True)
        b.add_setting(sliding_window)
        b.add_algorithm("ItemKNNIncremental", {"K": 1})
        b.add_metric("PrecisionK")
        b.add_metric("RecallK")
        evaluator = b.build()
        evaluator.run()
        
    def test_sliding_window_without_unknown_user(self, sliding_window):
        b = EvaluatorPipelineBuilder(True,False)
        b.add_setting(sliding_window)
        b.add_algorithm("ItemKNNIncremental", {"K": 1})
        b.add_metric("PrecisionK")
        b.add_metric("RecallK")
        evaluator = b.build()
        evaluator.run()
    
    def test_sliding_window_with_unknowns(self, sliding_window):
        b = EvaluatorPipelineBuilder(False,False)
        b.add_setting(sliding_window)
        b.add_algorithm("ItemKNNIncremental", {"K": 1})
        b.add_metric("PrecisionK")
        b.add_metric("RecallK")
        evaluator = b.build()
        evaluator.run()
        
    def test_single_time_point(self, single_time_point):
        b = EvaluatorPipelineBuilder()
        b.add_setting(single_time_point)
        b.add_algorithm("ItemKNNIncremental", {"K": 1})
        b.add_metric("PrecisionK")
        b.add_metric("RecallK")
        evaluator = b.build()
        evaluator.run()
    
    def test_stream(self, sliding_window):
        b = EvaluatorStreamerBuilder()
        b.add_setting(sliding_window)
        b.add_metric("PrecisionK")
        evaluator = b.build()
        
        from streamsight.algorithms import ItemKNNIncremental

        algo = ItemKNNIncremental(K=10)
        algo_id = evaluator.register_algorithm(algo)
        
        evaluator.start_stream()
        
        data = evaluator.get_data(algo_id)
        algo.fit(data)
        
        unlabeled_data = evaluator.get_unlabeled_data(algo_id)
        prediction = algo.predict(unlabeled_data)
        
        evaluator.submit_prediction(algo_id, prediction)
        