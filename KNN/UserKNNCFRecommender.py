#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author:
"""

from Base.Recommender_utils import check_matrix
from Base.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender

from Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np

from Base.Similarity.Compute_Similarity import Compute_Similarity
from SerialContentAnalysisFunctions import get_feature_vector_URM, concatenate_bw_features_with_URM

class UserKNNCFRecommender(BaseUserSimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]


    def __init__(self, URM_train, verbose = True):
        super(UserKNNCFRecommender, self).__init__(URM_train, verbose = verbose)




    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))


        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        similarity = Compute_Similarity(self.URM_train.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')


class UserKNNCFBingewatcherBingeworthyRecommender(UserKNNCFRecommender):
    """ UserKNN wit bingewatching and bingeworthy features recommender"""

    RECOMMENDER_NAME = "UserKNNCF_Bingewatcher_Bingeworthy"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]
    def __init__(self, URM_train, verbose = True):
        self.bingewatching_features, self.bingeworthy_features = get_feature_vector_URM(URM_train, explicit_URM=True)
        super(UserKNNCFBingewatcherBingeworthyRecommender, self).__init__(URM_train, verbose = verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", weight_bingewatchers=1, weight_bingeworthy=1, **similarity_args):
        URM_train_tmp = self.URM_train
        self.bingewatching_features = weight_bingewatchers * self.bingewatching_features
        self.bingeworthy_features = weight_bingeworthy * self.bingeworthy_features
        self.URM_extended = concatenate_bw_features_with_URM(self.URM_train, self.bingewatching_features,
                                                             self.bingeworthy_features)
        self.URM_train = self.URM_extended
        super(UserKNNCFBingewatcherBingeworthyRecommender, self).fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize, feature_weighting ="none")
        self.URM_train = URM_train_tmp
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_weights_array = self.W_sparse[user_id_array]
        print(user_weights_array.shape, self.URM_extended.shape, self.W_sparse.shape)
        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_extended.shape[1]), dtype=np.float32) * np.inf
            item_scores_all = user_weights_array.dot(self.URM_extended).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_extended).toarray()
            # item_scores[:, -1] = - np.inf

        return item_scores[:, :-1]

class UserKNNCFBingewatcherRecommender(UserKNNCFRecommender):
    """ UserKNN with bingewatching  features recommender"""

    RECOMMENDER_NAME = "UserKNNCF_Bingewatcher"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]
    def __init__(self, URM_train, verbose = True):
        self.bingewatching_features, _ = get_feature_vector_URM(URM_train, explicit_URM=True)
        super(UserKNNCFBingewatcherRecommender, self).__init__(URM_train, verbose = verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", weight_bingewatchers=1, weight_bingeworthy=1, **similarity_args):
        URM_train_tmp = self.URM_train
        self.bingewatching_features = weight_bingewatchers * self.bingewatching_features
        self.URM_extended = concatenate_bw_features_with_URM(self.URM_train, URM_bingewatchers_column_array=self.bingewatching_features)
        self.URM_train = self.URM_extended
        super(UserKNNCFBingewatcherRecommender, self).fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize, feature_weighting ="none")
        self.URM_train = URM_train_tmp

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_weights_array = self.W_sparse[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_extended.shape[1]), dtype=np.float32) * np.inf
            item_scores_all = user_weights_array.dot(self.URM_extended).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_extended).toarray()
            item_scores[:, -1] = - np.inf

        return item_scores[:, :-1]
class UserKNNCFBingeworthyRecommender(UserKNNCFRecommender):
    """ UserKNN with bingeworthy features recommender"""

    RECOMMENDER_NAME = "UserKNNCF_Bingeworthy"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]
    def __init__(self, URM_train, verbose = True):
        _, self.bingeworthy_features = get_feature_vector_URM(URM_train, explicit_URM=True)
        super(UserKNNCFBingeworthyRecommender, self).__init__(URM_train, verbose = verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", weight_bingewatchers=1, weight_bingeworthy=1, **similarity_args):
        URM_train_tmp = self.URM_train
        self.bingeworthy_features = weight_bingeworthy * self.bingeworthy_features
        print(self.bingeworthy_features)
        self.URM_extended = concatenate_bw_features_with_URM(self.URM_train, URM_bingeworthy_row_array=self.bingeworthy_features)
        self.URM_train = self.URM_extended
        super(UserKNNCFBingeworthyRecommender, self).fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize, feature_weighting ="none")
        self.URM_train = URM_train_tmp

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """
#
        self._check_format()

        user_weights_array = self.W_sparse[user_id_array]
        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_extended.shape[1]), dtype=np.float32) * np.inf
            item_scores_all = user_weights_array.dot(self.URM_extended).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_extended).toarray()
            item_scores[:, -1] = - np.inf

        return item_scores
