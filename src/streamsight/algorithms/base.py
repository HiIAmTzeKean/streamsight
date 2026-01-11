import logging
import time
from abc import abstractmethod
from inspect import Parameter, signature
from typing import Self

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from streamsight.matrix import InteractionMatrix, ItemUserBasedEnum, PredictionMatrix, to_csr_matrix
from streamsight.utils.util import add_columns_to_csr_matrix, add_rows_to_csr_matrix
from ..models import BaseModel, ParamMixin


logger = logging.getLogger(__name__)


class Algorithm(BaseEstimator, BaseModel, ParamMixin):
    """Base class for all streamsight algorithm implementations."""

    ITEM_USER_BASED: ItemUserBasedEnum

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self, "seed"):
            self.seed = 42
        self.rand_gen = np.random.default_rng(seed=self.seed)

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

    @classmethod
    def get_default_params(cls) -> dict:
        """Get default parameters without instantiation.

        Uses inspect.signature to extract __init__ parameters and their
        default values without instantiating the class.

        Returns:
            Dictionary of parameter names to default values.
            Parameters without defaults map to None.
        """
        try:
            sig = signature(cls.__init__)
        except (ValueError, TypeError):
            # Fallback for built-in types or special cases
            return {}

        params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                # Skip *args, **kwargs
                continue

            # Extract the default value
            if param.default is not Parameter.empty:
                params[param_name] = param.default
            else:
                params[param_name] = None

        return params

    def __str__(self) -> str:
        return self.name

    def set_params(self, **params) -> Self:
        """Set the parameters of the estimator.

        :param params: Estimator parameters
        :type params: dict
        """
        return super().set_params(**params)

    @abstractmethod
    def _fit(self, X: csr_matrix) -> Self:
        """Stub implementation for fitting an algorithm.

        Will be called by the `fit` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix to fit the model to
        :type X: csr_matrix
        :raises NotImplementedError: Implement this method in the child class
        """
        raise NotImplementedError("Please implement _fit")

    @abstractmethod
    def _predict(self, X: PredictionMatrix) -> csr_matrix:
        """Stub for predicting scores to users

        Will be called by the `predict` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix used as input to predict
        :type X: PredictionMatrix
        :raises NotImplementedError: Implement this method in the child class
        :return: Predictions made for all nonzero users in X
        :rtype: csr_matrix
        """
        raise NotImplementedError("Please implement _predict")

    def _check_fit_complete(self) -> None:
        """Helper function to check if model was correctly fitted

        Uses the sklearn check_is_fitted function,
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        """
        check_is_fitted(self)

    def _transform_fit_input(self, X: InteractionMatrix | csr_matrix) -> csr_matrix:
        """Transform the training data to expected type

        Data will be turned into a binary csr matrix.

        :param X: User-item interaction matrix to fit the model to
        :type X: InteractionMatrix | csr_matrix
        :return: Transformed user-item interaction matrix to fit the model
        :rtype: csr_matrix
        """
        return to_csr_matrix(X, binary=True)

    def fit(self, X: InteractionMatrix) -> Self:
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
        logger.debug(f"Fitting {self.name} complete - Took {end - start:.3}s")
        return self

    def _pad_unknown_iid_with_none_strategy(
        self,
        y_pred: csr_matrix,
        current_shape: tuple[int, int],
        intended_shape: tuple[int, int],
    ) -> csr_matrix:
        """Pad the predictions with empty fields for unknown items.

        This is to ensure that when we compute the performance of the prediction, we are
        comparing the prediction against the ground truth for the same set of items.
        """
        if y_pred.shape == intended_shape:
            return y_pred

        known_user_id, known_item_id = current_shape
        logger.debug(f"Padding item ID in range({known_item_id}, {intended_shape[1]}) with empty fields")
        y_pred = add_columns_to_csr_matrix(y_pred, intended_shape[1] - known_item_id)
        logger.debug(f"Padding by {self.name} completed")
        return y_pred

    # TODO change X_pred to y_pred for consistency
    def _pad_unknown_uid_with_random_strategy(
        self,
        X_pred: csr_matrix,
        current_shape: tuple[int, int],
        intended_shape: tuple[int, int],
        predict_ui_df: pd.DataFrame,
        k: int = 10,
    ) -> csr_matrix:
        """Pad the predictions with random items for users that are not in the training data.

        :param X_pred: Predictions made by the algorithm
        :type X_pred: csr_matrix
        :param intended_shape: The intended shape of the prediction matrix
        :type intended_shape: tuple[int, int]
        :param predict_ui_df: DataFrame containing the user IDs to predict for
        :type predict_ui_df: pd.DataFrame
        :return: The padded prediction matrix
        :rtype: csr_matrix
        """
        if X_pred.shape == intended_shape:
            return X_pred

        known_user_id, known_item_id = current_shape
        # +1 to include the last user id
        X_pred = add_rows_to_csr_matrix(X_pred, intended_shape[0] - known_user_id)
        # pad users with random items
        logger.debug(f"Padding user ID in range({known_user_id}, {intended_shape[0]}) with random items")
        to_predict = pd.Series(predict_ui_df.uid.unique())
        # Filter for users not in training data
        filtered = to_predict[to_predict >= known_user_id]
        filtered = filtered.sort_values(ignore_index=True)
        if not filtered.empty:
            row = filtered.values.repeat(k)
            total_pad = len(row)
            col = self.rand_gen.integers(0, known_item_id, total_pad)
            pad = csr_matrix((np.ones(total_pad), (row, col)), shape=intended_shape)
        else:
            pad = csr_matrix(intended_shape)
        X_pred += pad
        logger.debug(f"Padding by {self.name} completed")
        return X_pred

    def predict(self, X: PredictionMatrix) -> csr_matrix:
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
        X_pred = self._predict(X)
        return X_pred


class PopularityPaddingMixin:
    """Mixin class to add popularity-based padding to prediction methods."""

    def __init__(self, pad_with_popularity: bool = False) -> None:
        super().__init__()
        self.pad_with_popularity = pad_with_popularity

    def get_popularity_scores(self, X: csr_matrix) -> np.ndarray:
        """Compute a popularity-based scoring vector for items.

        This method calculates normalized interaction counts for each item,
        selects the top-K most popular items, and returns a vector where
        only those top-K items have their normalized scores (others are 0).
        This is used to pad predictions for unseen users with popular items.

        :param X: The interaction matrix (user-item) to compute popularity from.
        :type X: csr_matrix
        :return: A 1D array of shape (num_items,) with popularity scores for top-K items.
        :rtype: np.ndarray
        """
        interaction_counts = X.sum(axis=0).A[0]
        normalized_scores = interaction_counts / interaction_counts.max()

        num_items = X.shape[1]
        if hasattr(self, "K"):
            k_value = self.K
        else:
            k_value = 100
        if num_items < k_value:
            logger.warning("K is larger than the number of items.")

        effective_k = min(k_value, num_items)
        # Get indices of top-K items by popularity
        top_k_indices = np.argpartition(normalized_scores, -effective_k)[-effective_k:]
        popularity_vector = np.zeros(num_items)
        popularity_vector[top_k_indices] = normalized_scores[top_k_indices]

        return popularity_vector

    def _pad_uknown_uid_with_popularity_strategy(
        self,
        X_pred: csr_matrix,
        intended_shape: tuple,
        predict_ui_df: pd.DataFrame,
    ) -> csr_matrix:
        """Pad the predictions with popular items for users that are not in the training data.

        :param X_pred: Predictions made by the algorithm
        :type X_pred: csr_matrix
        :param intended_shape: The intended shape of the prediction matrix
        :type intended_shape: tuple
        :param predict_ui_df: DataFrame containing the user IDs to predict for
        :type predict_ui_df: pd.DataFrame
        :return: The padded prediction matrix
        :rtype: csr_matrix
        """
        if X_pred.shape == intended_shape:
            return X_pred

        known_user_id, known_item_id = X_pred.shape
        X_pred = add_rows_to_csr_matrix(X_pred, intended_shape[0] - known_user_id)
        # pad users with popular items
        logger.debug(f"Padding user ID in range({known_user_id}, {intended_shape[0]}) with popular items")
        popular_items = self.get_popularity_scores(X_pred)

        to_predict = predict_ui_df.value_counts("uid")
        # Filter for users not in training data
        filtered = to_predict[to_predict.index >= known_user_id]
        for user_id in filtered.index:
            if user_id >= known_user_id:
                X_pred[user_id, :] = popular_items
        return X_pred


class TopKAlgorithm(Algorithm):
    """Base algorithm for algorithms that recommend top-K items for every user."""

    def __init__(self, K: int = 10) -> None:
        super().__init__()
        self.K = K
        self.similarity_matrix_: csr_matrix


class TopKItemSimilarityMatrixAlgorithm(TopKAlgorithm):
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
    """

    similarity_matrix_: csr_matrix

    def _check_fit_complete(self) -> None:
        """Helper function to check if model was correctly fitted

        Checks implemented:

        - Checks if the algorithm has been fitted, using sklearn's `check_is_fitted`
        - Checks if the fitted similarity matrix contains similar items for each item

        For failing checks a warning is printed.
        """
        # Use super to check is fitted
        super()._check_fit_complete()

        # Ensures that similarity_matrix_ is computed
        if not hasattr(self, "similarity_matrix_"):
            raise AttributeError(f"{self.name} has no attribute similarity_matrix_ after fitting.")

        # Check row wise, since that will determine the recommendation options.
        items_with_score = set(self.similarity_matrix_.nonzero()[0])

        missing = self.similarity_matrix_.shape[0] - len(items_with_score)
        if missing > 0:
            logger.warning(f"{self.name} missing similar items for {missing} items.")
