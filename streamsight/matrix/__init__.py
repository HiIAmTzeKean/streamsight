"""
.. currentmodule:: streamsight.matrix

Interaction Matrix
-------------------
The InteractionMatrix class is used to create an interaction matrix from the
dataset loaded. The interaction matrix stores a dataframe under the hood and
provides functionality to convert the dataframe to a CSR matrix and other 
operations which are useful for building recommendation systems.

.. autosummary::
    :toctree: generated/

    InteractionMatrix

Utils
-----
Below are some of the utility classes and functions that are used in the matrix module.
These provides functionality for the matrix module and for use cases in other modules.

.. autosummary::
    :toctree: generated/

    to_csr_matrix
    ItemUserBasedEnum
    TimestampAttributeMissingError
"""
from streamsight.matrix.interaction_matrix import InteractionMatrix, ItemUserBasedEnum
from streamsight.matrix.util import to_csr_matrix
from streamsight.matrix.exception import TimestampAttributeMissingError