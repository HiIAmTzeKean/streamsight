# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert


from scipy.sparse import csr_matrix, diags
from sklearn.metrics.pairwise import cosine_similarity

from streamsight.utils import invert, to_binary


def compute_conditional_probability(X: csr_matrix, pop_discount: float = 0) -> csr_matrix:
    """Compute conditional probability like similarity.

    Computation using equation (3) from the original ItemKNN paper.
    'Item-based top-n recommendation algorithms.'
    Deshpande, Mukund, and George Karypis

    .. math ::
        sim(i,j) = \\frac{\\sum\\limits_{u \\in U} \\mathbb{I}_{u,i} X_{u,j}}{Freq(i) \\times Freq(j)^{\\alpha}}

    Where :math:`\\mathbb{I}_{ui}` is 1 if the user u has visited item i, and 0 otherwise.
    And alpha is the pop_discount parameter.
    Note that this is a non-symmetric similarity measure.
    Given that X is a binary matrix, and alpha is set to 0,
    this simplifies to pure conditional probability.

    .. math::
        sim(i,j) = \\frac{Freq(i \\land j)}{Freq(i)}

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_matrix
    :param pop_discount: Parameter defining popularity discount. Defaults to 0
    :type pop_discount: float, Optional.
    """
    # matrix with co_mat_i,j =  SUM(1_u,i * X_u,j for each user u)
    # If the input matrix is binary, this is the cooccurence count matrix.
    co_mat = to_binary(X).T @ X

    # Compute the inverse of the item frequencies
    A = invert(diags(to_binary(X).sum(axis=0).A[0]).tocsr())

    if pop_discount:
        # This has all item similarities
        # Co_mat is weighted by both the frequencies of item i
        # and the frequency of item j to the pop_discount power.
        # If pop_discount = 1, this similarity is symmetric again.
        item_cond_prob_similarities = A @ co_mat @ A.power(pop_discount)
    else:
        # Weight the co_mat with the amount of occurences of item i.
        item_cond_prob_similarities = A @ co_mat

    # Set diagonal to 0, because we don't support self similarity
    item_cond_prob_similarities.setdiag(0)

    return item_cond_prob_similarities


def compute_cosine_similarity(X: csr_matrix) -> csr_matrix:
    """Compute the cosine similarity between the items in the matrix.

    Self similarity is removed.

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_matrix
    :return: similarity matrix
    :rtype: csr_matrix
    """
    # X.T otherwise we are doing a user KNN
    item_cosine_similarities = cosine_similarity(X.T, dense_output=False)
    item_cosine_similarities.setdiag(0)
    # Set diagonal to 0, because we don't want to support self similarity

    return item_cosine_similarities


def compute_pearson_similarity(X: csr_matrix) -> csr_matrix:
    """Compute the pearson correlation as a similarity between each item in the matrix.

    Self similarity is removed.
    When computing similarity, the avg of nonzero entries per user is used.

    :param X: Rating or psuedo rating matrix.
    :type X: csr_matrix
    :return: similarity matrix.
    :rtype: csr_matrix
    """

    if (X == 1).sum() == X.nnz:
        raise ValueError("Pearson similarity can not be computed on a binary matrix.")

    count_per_item = (X > 0).sum(axis=0).A

    avg_per_item = X.sum(axis=0).A.astype(float)

    avg_per_item[count_per_item > 0] = (
        avg_per_item[count_per_item > 0] / count_per_item[count_per_item > 0]
    )

    X = X - (X > 0).multiply(avg_per_item)

    # Given the rescaled matrix, the pearson correlation is just cosine similarity on this matrix.
    return compute_cosine_similarity(X)