import logging
import time
from warnings import warn
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from streamsight.matrix import (InteractionMatrix, ItemUserBasedEnum,
                                to_csr_matrix, Matrix)
from streamsight.utils.util import add_rows_to_csr_matrix

logger = logging.getLogger(__name__)


class Algorithm(BaseEstimator,ABC):
    """Base class for all streamsight algorithm implementations."""

    ITEM_USER_BASED: ItemUserBasedEnum

    def __init__(self):
        super().__init__()
        if not hasattr(self, "seed"):
            self.seed = 42
        self.rand_gen = np.random.default_rng(seed=self.seed)

    @property
    def name(self) -> str:
        """Name of the object's class.
        
        :return: Name of the object's class
        :rtype: str
        """
        return self.__class__.__name__

    @property
    def identifier(self) -> str:
        """Identifier of the object.

        Identifier is made by combining the class name with the parameters
        passed at construction time.

        Constructed by recreating the initialisation call.
        Example: `Algorithm(param_1=value)`
        
        :return: Identifier of the object
        :rtype: str
        """
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items()))
        return self.name + "(" + paramstring + ")"

    def __str__(self):
        return self.name

    def set_params(self, **params):
        """Set the parameters of the estimator.

        :param params: Estimator parameters
        :type params: dict
        """
        super().set_params(**params)

    @abstractmethod
    def _fit(self, X: csr_matrix):
        """Stub implementation for fitting an algorithm.

        Will be called by the `fit` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix to fit the model to
        :type X: csr_matrix
        :raises NotImplementedError: Implement this method in the child class
        """
        raise NotImplementedError("Please implement _fit")

    @abstractmethod
    def _predict(self, X: csr_matrix, predict_frame:Optional[pd.DataFrame]=None) -> csr_matrix:
        """Stub for predicting scores to users

        Will be called by the `predict` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix used as input to predict
        :type X: csr_matrix
        :param predict_frame: DataFrame containing the user IDs to predict for
        :type predict_frame: pd.DataFrame
        :raises NotImplementedError: Implement this method in the child class
        :return: Predictions made for all nonzero users in X
        :rtype: csr_matrix
        """
        raise NotImplementedError("Please implement _predict")

    def _check_fit_complete(self):
        """Helper function to check if model was correctly fitted

        Uses the sklearn check_is_fitted function,
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        """
        check_is_fitted(self)

    def _transform_fit_input(self, X: InteractionMatrix) -> csr_matrix:
        """Transform the training data to expected type

        Data will be turned into a binary csr matrix.

        :param X: User-item interaction matrix to fit the model to
        :type X: InteractionMatrix
        :return: Transformed user-item interaction matrix to fit the model
        :rtype: csr_matrix
        """
        return to_csr_matrix(X, binary=True)

    def _transform_predict_input(self, X: InteractionMatrix) -> csr_matrix:
        """Transform the input of predict to expected type

        Data will be turned into a binary csr matrix.

        :param X: User-item interaction matrix used as input to predict
        :type X: InteractionMatrix
        :return: Transformed user-item interaction matrix used as input to predict
        :rtype: csr_matrix
        """
        return to_csr_matrix(X, binary=True)

    def _assert_is_interaction_matrix(self, *matrices: Matrix) -> None:
        """Make sure that the passed matrices are all an InteractionMatrix."""
        for X in matrices:
            if type(X) != InteractionMatrix:
                raise TypeError(f"{self.name} requires Interaction Matrix as input. Got {type(X)}.")

    def _assert_has_timestamps(self, *matrices: InteractionMatrix):
        """Make sure that the matrices all have timestamp information."""
        for X in matrices:
            if not X.has_timestamps:
                raise ValueError(f"{self.name} requires timestamp information in the InteractionMatrix.")

    def fit(self, X: InteractionMatrix) -> "Algorithm":
        """Fit the model to the input interaction matrix.

        The input data is transformed to the expected type using
        :meth:`_transform_fit_input`. The fitting is done using the
        :meth:`_fit` method. Finally the method checks that the fitting
        was successful using :meth:`_check_fit_complete`.

        :param X: The interactions to fit the model on.
        :type X: InteractionMatrix
        :return: Fitted algorithm
        :rtype: Algorithm
        """
        start = time.time()
        X_transformed = self._transform_fit_input(X)
        self._fit(X_transformed)

        self._check_fit_complete()
        end = time.time()
        logger.debug(f"Fitting {self.name} complete - Took {end - start :.3}s")
        return self

    def _pad_predict(
        self,
        X_pred: csr_matrix,
        intended_shape: tuple,
        to_predict_frame: pd.DataFrame,
    ) -> csr_matrix:
        """Pad the predictions with random items for users that are not in the training data.

        :param X_pred: Predictions made by the algorithm
        :type X_pred: csr_matrix
        :param intended_shape: The intended shape of the prediction matrix
        :type intended_shape: tuple
        :param to_predict_frame: DataFrame containing the user IDs to predict for
        :type to_predict_frame: pd.DataFrame
        :return: The padded prediction matrix
        :rtype: csr_matrix
        """
        if X_pred.shape == intended_shape:
            return X_pred

        known_user_id, known_item_id = X_pred.shape
        X_pred = add_rows_to_csr_matrix(
            X_pred, intended_shape[0] - known_user_id
        )
        # pad users with random items
        logger.debug(
            f"Padding user ID in range({known_user_id}, {intended_shape[0]}) with random items"
        )
        to_predict = to_predict_frame.value_counts("uid")
        row = []
        col = []
        for user_id in to_predict.index:
            if user_id >= known_user_id:
                row += [user_id] * to_predict[user_id]
                col += self.rand_gen.integers(0, known_item_id, to_predict[user_id]).tolist()
        pad = csr_matrix((np.ones(len(row)), (row, col)), shape=intended_shape)
        X_pred += pad
        logger.debug(f"Padding by {self.name} completed")
        return X_pred

    def predict(self, X: InteractionMatrix, unlabeled_X: InteractionMatrix) -> csr_matrix:
        """Predicts scores, given the interactions in X

        The input data is transformed to the expected type using
        :meth:`_transform_predict_input`. The predictions are made
        using the :meth:`_predict` method. Finally the predictions
        are then padded with random items for users that are not in the
        training data.

        :param X: interactions to predict from.
        :type X: InteractionMatrix
        :return: The recommendation scores in a sparse matrix format.
        :rtype: csr_matrix
        """
        self._check_fit_complete()
        # X are all the labeled data, to_predict_frame are the IDs to predict and labels are -1
        X = self._transform_predict_input(X)

        to_predict_im = unlabeled_X.get_prediction_data()

        # X_pred = self._predict(X, to_predict_frame._df)
        X_pred = self._predict(X, to_predict_im)

        return X_pred

        # OLD PREDICT
        # self._check_fit_complete()

        # # X will contain past sequential interaction and IDs to predict
        # to_predict_frame = X.get_prediction_data()
        # prev_interaction = X.get_interaction_data()
        # prev_interaction = self._transform_predict_input(prev_interaction)
        # if to_predict_frame._df.empty:
        #     return csr_matrix(X.shape, dtype=int)

        # X_pred = self._predict(prev_interaction, to_predict_frame._df)
        # # known_user_id, known_item_id = X_pred.shape
        # logger.debug(f"Predictions by {self.name} completed")

        # # ID indexing starts at 0, so max_id + 1 is the number of unique IDs
        # max_user_id = to_predict_frame.max_user_id + 1
        # max_item_id = to_predict_frame.max_item_id + 1
        # intended_shape = (
        #     max(max_user_id, X.shape[0]),
        #     max(max_item_id, X.shape[1]),
        # )

        # return self._pad_predict(X_pred, intended_shape, to_predict_frame._df)

class ItemSimilarityMatrixAlgorithm(Algorithm):
    """Base algorithm for algorithms that fit an item to item similarity model

    Model that encodes the similarity between items is expected
    under the ``similarity_matrix_`` attribute.

    This matrix should have shape ``(|items| x |items|)``.
    This can be dense or sparse matrix depending on the algorithm used.

    Predictions are made by computing the dot product of the history vector of a user
    and the similarity matrix.

    Usually a new algorithm will have to
    implement just the :meth:`_fit` method,
    to construct the `self.similarity_matrix_` attribute.

    This code is adapted from RecPack :cite:`recpack`
    """

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Predict scores for nonzero users in X

        Scores are computed by matrix multiplication of X
        with the stored similarity matrix.

        :param X: csr_matrix with interactions
        :type X: csr_matrix
        :return: csr_matrix with scores
        :rtype: csr_matrix
        """
        scores = X @ self.similarity_matrix_

        # If self.similarity_matrix_ is not a csr matrix,
        # scores will also not be a csr matrix
        if not isinstance(scores, csr_matrix):
            scores = csr_matrix(scores)

        return scores

    def _check_fit_complete(self):
        """Helper function to check if model was correctly fitted

        Checks implemented:

        - Checks if the algorithm has been fitted, using sklearn's `check_is_fitted`
        - Checks if the fitted similarity matrix contains similar items for each item

        For failing checks a warning is printed.
        """
        # Use super to check is fitted
        super()._check_fit_complete()

        # Additional checks on the fitted matrix.
        # Check if actually exists!
        assert hasattr(self, "similarity_matrix_")

        # Check row wise, since that will determine the recommendation options.
        items_with_score = set(self.similarity_matrix_.nonzero()[0])

        missing = self.similarity_matrix_.shape[0] - len(items_with_score)
        if missing > 0:
            warn(f"{self.name} missing similar items for {missing} items.")

class TopKItemSimilarityMatrixAlgorithm(ItemSimilarityMatrixAlgorithm):
    """Base algorithm for algorithms that fit an item to item similarity model with K similar items for every item

    Model that encodes the similarity between items is expected
    under the ``similarity_matrix_`` attribute.

    This matrix should have shape ``(|items| x |items|)``.
    This can be dense or sparse matrix depending on the algorithm used.

    Predictions are made by computing the dot product of the history vector of a user
    and the similarity matrix.

    Usually a new algorithm will have to
    implement just the :meth:`_fit` method,
    to construct the `self.similarity_matrix_` attribute.

    :param K: How many similar items will be kept per item.
    :type K: int

    This code is adapted from RecPack :cite:`recpack`
    """

    def __init__(self, K):
        super().__init__()
        self.K = K