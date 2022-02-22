import Data_manager.Dataset
from Base.BaseRecommender import BaseRecommender
from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np
import psutil
from SerialContentAnalysisFunctions import user_series_bw_ICMs, user_series_bw_ICMs_explicit

from Base.DataIO import DataIO
from tqdm import tqdm
import logging

logger = logging.getLogger("contentwise-impressions")



class LightFMWrapper(BaseRecommender):
    """LightFMWrapper"""

    RECOMMENDER_NAME = "LightFMWrapper"

    def __init__(self, URM_train):
        super(LightFMWrapper, self).__init__(URM_train)
        self.num_threads = int(psutil.cpu_count())

    def fit(self, item_alpha, num_components, epochs):
        # Let's fit a WARP model
        self.epochs = epochs
        self.lightFM_model = LightFM(loss='warp',
                                     item_alpha=item_alpha,
                                     no_components=num_components)

        self.lightFM_model = self.lightFM_model.fit(self.URM_train,
                                                    epochs=self.epochs,
                                                    num_threads=self.num_threads)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Create a single (n_items, ) array with the item score, then copy it for every user
        items_to_compute = np.arange(self.n_items) if items_to_compute is None else np.array(items_to_compute)

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        for user_index, user_id in enumerate(user_id_array):
            item_scores[user_index] = self.lightFM_model.predict(int(user_id),
                                                                 items_to_compute)

        return item_scores

    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        logger.info(f"Saving model in file '{folder_path + file_name}'")

        data_dict_to_save = {"item_alpha": self.lightFM_model.item_alpha,
                             "num_components": self.lightFM_model.no_components,
                             "epochs": self.epochs}




        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)


        self._print("Saving complete")


class LightFMWrapper_bingewatching_bingeworthy(LightFMWrapper):
    """ LightFMWrapper with bingewatching user and item vectors"""
    RECOMMENDER_NAME = "LightFMWrapper_bingewatching_bingeworthy"

    def __init__(self, URM_train):
        super(LightFMWrapper_bingewatching_bingeworthy, self).__init__(URM_train)
        self.dataset = Dataset()

        self.users_features, self.items_features = user_series_bw_ICMs_explicit(URM_train)

    def fit(self, item_alpha=0.0001, epochs=100, num_components=292, weight_bingewatchers=1, weight_bingeworthy=1):
        # Let's fit a WARP model
        self.epochs = epochs
        users_features = self.users_features * weight_bingewatchers
        items_features = self.items_features * weight_bingeworthy
        users_features_mapping, items_features_mapping = [], []
        for val in np.unique(users_features):
            users_features_mapping.append("bingewatching:" + str(val))
        for val in np.unique(items_features):
            items_features_mapping.append("bingeworthy:" + str(val))
        self.dataset.fit(np.arange(self.URM_train.shape[0]),
                         np.arange(self.URM_train.shape[1]),
                         user_features=users_features_mapping,
                         item_features=items_features_mapping)
        user_features_tuple, item_features_tuple = [], []
        for i in range(self.URM_train.shape[0]):
            user_features_tuple.append((i, ["bingewatching:" + str(users_features[i])]))
        for i in range(self.URM_train.shape[1]):
            item_features_tuple.append((i, ["bingeworthy:" + str(items_features[i])]))
        self.user_features = self.dataset.build_user_features(user_features_tuple, normalize=False)
        self.item_features = self.dataset.build_item_features(item_features_tuple, normalize=False)
        self.lightFM_model = LightFM(loss='warp',
                                     item_alpha=item_alpha,
                                     no_components=num_components)

        self.lightFM_model = self.lightFM_model.fit(self.URM_train,
                                                    epochs=self.epochs,
                                                    num_threads=self.num_threads,
                                                    user_features=self.user_features,
                                                    item_features=self.item_features)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Create a single (n_items, ) array with the item score, then copy it for every user
        items_to_compute = np.arange(self.n_items) if items_to_compute is None else np.array(items_to_compute)

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        for user_index, user_id in enumerate(user_id_array):
            item_scores[user_index] = self.lightFM_model.predict(int(user_id),
                                                                 items_to_compute,
                                                                 item_features=self.item_features,
                                                                 user_features=self.user_features,
                                                                 num_threads=self.num_threads)
        return item_scores


class LightFMWrapper_bingewatching(LightFMWrapper):
    """ LightFMWrapper with bingewatching user and item vectors"""
    RECOMMENDER_NAME = "LightFMWrapper_bingewatching"

    def __init__(self, URM_train):
        super(LightFMWrapper_bingewatching, self).__init__(URM_train)
        self.dataset = Dataset()

        self.users_features, _ = user_series_bw_ICMs_explicit(URM_train)

    def fit(self, item_alpha=0.0001, epochs=100, num_components=292, weight_bingewatchers=1):
        # Let's fit a WARP model
        self.epochs = epochs
        users_features = self.users_features * weight_bingewatchers
        users_features_mapping = []
        for val in np.unique(users_features):
            users_features_mapping.append("bingewatching:" + str(val))

        self.dataset.fit(np.arange(self.URM_train.shape[0]),
                         np.arange(self.URM_train.shape[1]),
                         user_features=users_features_mapping)
        user_features_tuple = []
        for i in range(self.URM_train.shape[0]):
            user_features_tuple.append((i, ["bingewatching:" + str(users_features[i])]))
        self.user_features = self.dataset.build_user_features(user_features_tuple, normalize=False)
        self.lightFM_model = LightFM(loss='warp',
                                     item_alpha=item_alpha,
                                     no_components=num_components)

        self.lightFM_model = self.lightFM_model.fit(self.URM_train,
                                                    epochs=self.epochs,
                                                    num_threads=self.num_threads,
                                                    user_features=self.user_features)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Create a single (n_items, ) array with the item score, then copy it for every user
        items_to_compute = np.arange(self.n_items) if items_to_compute is None else np.array(items_to_compute)

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        for user_index, user_id in enumerate(user_id_array):
            item_scores[user_index] = self.lightFM_model.predict(int(user_id),
                                                                 items_to_compute,
                                                                 user_features=self.user_features,
                                                                 num_threads=self.num_threads)
        return item_scores

class LightFMWrapper_bingeworthy(LightFMWrapper):
    """ LightFMWrapper with bingewatching user and item vectors"""
    RECOMMENDER_NAME = "LightFMWrapper_bingeworthy"

    def __init__(self, URM_train):
        super(LightFMWrapper_bingeworthy, self).__init__(URM_train)
        self.dataset = Dataset()

        _, self.items_features = user_series_bw_ICMs_explicit(URM_train)

    def fit(self, item_alpha=0.0001, epochs=100, num_components=292, weight_bingeworthy=1):
        # Let's fit a WARP model
        self.epochs = epochs
        items_features = self.items_features * weight_bingeworthy
        items_features_mapping = []
        for val in np.unique(items_features):
            items_features_mapping.append("bingeworthy:" + str(val))
        self.dataset.fit(np.arange(self.URM_train.shape[0]),
                         np.arange(self.URM_train.shape[1]),
                         item_features=items_features_mapping)
        user_features_tuple, item_features_tuple = [], []
        for i in range(self.URM_train.shape[1]):
            item_features_tuple.append((i, ["bingeworthy:" + str(items_features[i])]))
        self.item_features = self.dataset.build_item_features(item_features_tuple, normalize=False)
        self.lightFM_model = LightFM(loss='warp',
                                     item_alpha=item_alpha,
                                     no_components=num_components)

        self.lightFM_model = self.lightFM_model.fit(self.URM_train,
                                                    epochs=self.epochs,
                                                    num_threads=self.num_threads,
                                                    item_features=self.item_features)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Create a single (n_items, ) array with the item score, then copy it for every user
        items_to_compute = np.arange(self.n_items) if items_to_compute is None else np.array(items_to_compute)

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        for user_index, user_id in enumerate(user_id_array):
            item_scores[user_index] = self.lightFM_model.predict(int(user_id),
                                                                 items_to_compute,
                                                                 item_features=self.item_features,
                                                                 num_threads=self.num_threads)
        return item_scores
