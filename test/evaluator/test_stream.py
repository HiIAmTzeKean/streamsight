import pytest
from streamsight.settings.sliding_window_setting import SlidingWindowSetting
from test.conftest import BACKGROUND_T, WINDOW_SIZE, SEED, N_SEQ_DATA, SEED
from streamsight.evaluators import EvaluatorStreamerBuilder
from streamsight.algorithms import ItemKNNIncremental


@pytest.fixture()
def setting(test_dataset):
    data = test_dataset.load()
    setting_obj = SlidingWindowSetting(background_t=BACKGROUND_T,
                                window_size=WINDOW_SIZE,
                                n_seq_data=N_SEQ_DATA,
                                seed=SEED)
    setting_obj.split(data)
    return setting_obj

@pytest.fixture()
def k():
    return 10

class TestStreamer():
    def test_algorithm_in_different_data_segment_handling(self, setting, k):

        builder = EvaluatorStreamerBuilder()
        builder.add_setting(setting)
        builder.set_metric_K(k)
        builder.add_metric("PrecisionK")
        evaluator = builder.build()

        algo = ItemKNNIncremental(K=10)
        algo_id = evaluator.register_algorithm(algo)
        print(algo_id)

        from streamsight.algorithms import ItemKNNStatic
        external_model = ItemKNNIncremental(K=10)
        external_model_id = evaluator.register_algorithm(external_model)
        print(external_model_id)

        evaluator.start_stream()

        # first iteration
        data = evaluator.get_data(algo_id)
        algo.fit(data)
        unlabeled_data = evaluator.get_unlabeled_data(algo_id)
        prediction = algo.predict(unlabeled_data)
        evaluator.submit_prediction(algo_id, prediction)
        data = evaluator.get_data(external_model_id)
        external_model.fit(data)
        unlabeled_data = evaluator.get_unlabeled_data(external_model_id)
        prediction = external_model.predict(unlabeled_data)
        evaluator.submit_prediction(external_model_id, prediction)

        # second iteration
        print("Second iteration")
        data = evaluator.get_data(algo_id)
        algo.fit(data)
        unlabeled_data = evaluator.get_unlabeled_data(algo_id)
        prediction = algo.predict(unlabeled_data)
        evaluator.submit_prediction(algo_id, prediction)


        to_validate_data = evaluator.get_data(external_model_id)

        assert(to_validate_data == data)