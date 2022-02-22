#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author:   
"""

from Base.Recommender_utils import check_matrix
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

from Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np
from SerialContentAnalysisFunctions import concatenate_bw_features_with_URM, get_feature_vector_URM
from Base.Similarity.Compute_Similarity import Compute_Similarity


class ItemKNNCFRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, verbose=True, evaluation_block_size: int = 500):
        super(ItemKNNCFRecommender, self).__init__(URM_train,
                                                   verbose=verbose,
                                                   evaluation_block_size=evaluation_block_size)

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

        similarity = Compute_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)


        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')


class ItemKNNCFBingewatcherBingeworthyRecommender(ItemKNNCFRecommender):
    """ ItemKNNExtended with bingewatchers and bingeworthy features recommender"""

    RECOMMENDER_NAME = "ItemKNNCF_Bingewatcher_Bingeworthy"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]
    def __init__(self, URM_train, verbose = True):
        self.bingewatching_features, self.bingeworthy_features = get_feature_vector_URM(URM_train, explicit_URM=True)
        super(ItemKNNCFBingewatcherBingeworthyRecommender, self).__init__(URM_train, verbose = verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", weight_bingewatchers=1, weight_bingeworthy=1, **similarity_args):
        URM_train_tmp = self.URM_train
        self.bingewatching_features = weight_bingewatchers * self.bingewatching_features
        self.bingeworthy_features = weight_bingeworthy * self.bingeworthy_features
        self.URM_extended = concatenate_bw_features_with_URM(self.URM_train, self.bingewatching_features,
                                                             self.bingeworthy_features)
        self.URM_train = self.URM_extended
        super(ItemKNNCFBingewatcherBingeworthyRecommender, self).fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize, feature_weighting ="none")
        self.URM_train = URM_train_tmp

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_extended[user_id_array]
        print(user_profile_array.shape, self.URM_extended.shape, self.W_sparse.shape)

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_extended.shape[1]), dtype=np.float32) * np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse).toarray()
            # item_scores[:, -1] = -np.inf
        # We don't wanna recommend the extra item so we exclude it from the recommendation
        return item_scores[:, :-1]


class ItemKNNCFBingewatcherRecommender(ItemKNNCFRecommender):
    """ ItemKNNExtended with bingewatchers features recommender"""

    RECOMMENDER_NAME = "ItemKNNCF_Bingewatcher"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]
    def __init__(self, URM_train, verbose = True):
        self.bingewatching_features, _ = get_feature_vector_URM(URM_train, explicit_URM=True)
        super(ItemKNNCFBingewatcherRecommender, self).__init__(URM_train, verbose = verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", weight_bingewatchers=1, **similarity_args):
        URM_train_tmp = self.URM_train
        self.bingewatching_features = weight_bingewatchers * self.bingewatching_features
        self.URM_extended = concatenate_bw_features_with_URM(self.URM_train, URM_bingewatchers_column_array=self.bingewatching_features)
        self.URM_train = self.URM_extended
        super(ItemKNNCFBingewatcherRecommender, self).fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize, feature_weighting ="none")
        self.URM_train = URM_train_tmp

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_extended[user_id_array]
        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_extended.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse).toarray()
            # item_scores[:, -1] = -np.inf
        # We don't wanna recommend the extra item so we exclude it from the recommendation
        return item_scores[:, :-1]


class ItemKNNCFBingeworthyRecommender(ItemKNNCFRecommender):
    """ ItemKNNExtended with bingeworthy features recommender"""

    RECOMMENDER_NAME = "ItemKNNCF_Bingeworthy"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]
    def __init__(self, URM_train, verbose = True):
        _, self.bingeworthy_features = get_feature_vector_URM(URM_train, explicit_URM=True)
        super(ItemKNNCFBingeworthyRecommender, self).__init__(URM_train, verbose = verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", weight_bingeworthy=1, **similarity_args):
        URM_train_tmp = self.URM_train
        self.bingeworthy_features = weight_bingeworthy * self.bingeworthy_features
        self.URM_extended = concatenate_bw_features_with_URM(self.URM_train, URM_bingeworthy_row_array=self.bingeworthy_features)
        self.URM_train = self.URM_extended
        super(ItemKNNCFBingeworthyRecommender, self).fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize, feature_weighting ="none")
        self.URM_train = URM_train_tmp
