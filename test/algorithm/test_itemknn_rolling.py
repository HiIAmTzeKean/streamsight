from streamsight.algorithms import ItemKNNRolling

def test_ItemKNNRolling(setting):
    algo = ItemKNNRolling(K=10)
    setting.background_data.mask_shape()
    algo.fit(setting.background_data)
    unlabeled_data = setting.unlabeled_data[0]
    unlabeled_data.mask_shape(setting.background_data.shape, True, True)
    X_pred = algo.predict(unlabeled_data)