# -*- coding: utf-8 -*-
"""
@author:

Set of functions to clean data, extract stats and plot graphs for [] Impression dataset
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tqdm import tqdm
from functools import partial
from scipy import stats
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import holoviews as hv
from holoviews import opts
from bokeh.models.mappers import LinearColorMapper, LogColorMapper
from bokeh.palettes import Viridis6
import os

hv.extension('bokeh')
from collections import defaultdict
import seaborn as sns
pd.io.parquet.get_engine("auto")
pd.set_option('mode.chained_assignment', None)
ROOT_PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
"""
-----------------------------------------------
|                DATA MANAGEMENT              |                                                              
-----------------------------------------------
"""

def clean_dataset(df, transpose_episode_1=True, sort_values=True, set_vision_factor=True, threshold=0.9):
    """
    Remove irrelevant interactions and converts timestamps into date-time instead of ms, more readable
    :param df: DataFrame of the ContentWise Impression dataset
    :return: DataFrame of the cleaned dataset
    """
    if transpose_episode_1:
        df[["episode_number", "series_length"]] = df[["episode_number", "series_length"]] - 1

    df = df[(df["item_type"] == 3) & (df["vision_factor"] != -1) & (df["interaction_type"] == 0) & (
            df["series_length"] > 2)]
    df["utc_ts_milliseconds"] = pd.to_datetime(df["utc_ts_milliseconds"], unit='ms')
    # df = df.rename(columns={"utc_ts_milliseconds": "utc_ts"})
    if sort_values:
        df = df.sort_values("utc_ts_milliseconds")
    if set_vision_factor:
        df = set_vision_factor_threshold(df, threshold)
    return df


def clean_dataset_CW_class(df,  set_vision_factor=True, threshold=0.9):
    """
    Remove irrelevant interactions and converts timestamps into date-time instead of ms, more readable
    :param df: DataFrame of the ContentWise Impression dataset
    :return: DataFrame of the cleaned dataset
    """

    df = df[(df["vision_factor"] != -1) & (df["interaction_type"] == 0)]
    if set_vision_factor:
        df = set_vision_factor_threshold(df, threshold)
    return df


def set_vision_factor_threshold(df, threshold):
    """
    Remove all interactions with vision_factor <= threshold
    :param df: DataFrame of the ContentWise Impression dataset
    :param threshold: float value between 0 and 1
    :return: DataFrame without the interactions with vision_factor <= threshold
    """
    if threshold < 0 or threshold > 1:
        ValueError(f'Threshold must be betweend 0 and 1.')

    df_filtered = df[df["vision_factor"] >= threshold]
    difference_after_filtering(df, df_filtered)
    return df_filtered


def dataframe_to_parquet(df, name="interactions"):
    df.to_parquet(os.path.join(ROOT_PROJECT_PATH, f"parquets/{name}.parquet"),
                                      compression=None, engine="fastparquet")


def get_episode_watched(df, include_percentage=True, already_cleaned=False):
    """
    Get a DataFrame containing the count of different episodes a user has watched over a series
    :param df: DataFrame of the ContentWise Impression dataset
    :param include_percentage: boolean, compute a percentage of episode watched column
    :param already_cleaned: boolean, if the dataset is not already cleaned clean it first
    :return: DataFrame with percentage and count of episodes watched
    """
    if not already_cleaned:
        print("Cleaning Dataset")
        df = clean_dataset(df)

    df_grouped = (df.groupby(["user_id", "series_id", "series_length"]).episode_number.nunique()).to_frame(
        name="episodes_watched").reset_index()
    df_grouped = df_grouped.sort_values(by=["episodes_watched", "user_id"], ascending=False)
    if include_percentage:
        df_grouped["percentage"] = (df_grouped["episodes_watched"] / df_grouped["series_length"]) * 100
        df_grouped = df_grouped.sort_values(by=["percentage", "user_id"], ascending=False)

    return df_grouped


def order_by_user_series_ts(df):
    """
    Orders the DataFrame by user_id, series_id, timestamps
    :param df: DataFrame of the ContentWise Impression dataset
    :return: ordered DataFrame
    """
    df = df.sort_values(by=["user_id", "series_id", "utc_ts_milliseconds", "vision_factor"])
    return df


def get_relevant_interactions(df, lbound_percentage=90, ubound_percentage=100):
    """
    Returns the relevant interactions to be used in meaningful statistics
    :param df: DataFrame of the ContentWise Impression dataset
    :return:
    """

    # Select the user_ids of (user, series) pair that have watched at least 90 % of a series
    df_grouped = get_episode_watched(df, already_cleaned=True)
    user_id_sequence_reconstruction = df_grouped[
        (df_grouped["percentage"] >= lbound_percentage) & (df_grouped["percentage"] <= ubound_percentage)].user_id
    user_id_sequence_reconstruction = pd.Series(user_id_sequence_reconstruction.values)
    print(user_id_sequence_reconstruction)
    df = order_by_user_series_ts(df)
    relevant_interactions = df[df["user_id"].isin(user_id_sequence_reconstruction.values)]
    relevant_interactions = relevant_interactions.reset_index(drop=True)
    relevant_interactions = relevant_interactions.drop_duplicates(
        subset=["user_id", "utc_ts_milliseconds", "series_id", "episode_number"], keep="first")

    return relevant_interactions


def get_full_view_interactions(df):
    """
    Returns the dataframe with full view to be used in meaningful statistics
    :param df: DataFrame of the ContentWise Impression dataset
    :return:
    """

    # Select the user_ids of (user, series) pair that have watched at least 90 % of a series
    df_grouped = get_episode_watched(df, already_cleaned=True)
    user_id_full_view = df_grouped[df_grouped["percentage"] == 100].user_id
    user_id_full_view = pd.Series(user_id_full_view.values)
    df = order_by_user_series_ts(df)
    full_view = df[df["user_id"].isin(user_id_full_view.values)]
    full_view = full_view.reset_index(drop=True)
    full_view = full_view.drop_duplicates(
        subset=["user_id", "utc_ts_milliseconds", "series_id", "episode_number"], keep="first")

    return full_view


def get_most_interactions(df, already_relevant=False):
    """
    Prepare a DataFrame to plot the corresponding heatmap
    :param df: DataFrame of the ContentWise Impression dataset
    :param already_relevant: True if df is have already filtered the relevant interactions, default false
    :return: DataFrame with series_id and interactions ordered by series with most interactions
    """
    if not already_relevant:
        df = get_relevant_interactions(df)
    most_interactions = (df.groupby(["series_id"]).count()).user_id.to_frame(name="interactions").reset_index()
    most_interactions = most_interactions.sort_values(by="interactions", ascending=False)
    return most_interactions


def get_vision_factor_count(df):
    """
    Returns a DataFrame containing the vision factor count for each value of the vision factor between 0 and 1
    :param df: DataFrame of the ContentWise Impression dataset
    :return: DataFrame with vision factor and number of times it is present in the dataset
    """
    vision_factor_count = (df["vision_factor"].value_counts())
    vision_factor_count = pd.DataFrame(data=vision_factor_count).reset_index()
    vision_factor_count = vision_factor_count.rename(columns={"index": "vision_factor", "vision_factor": "y_axis"})
    vision_factor_count = vision_factor_count.sort_values("vision_factor")
    return vision_factor_count


def _is_sorted(lst, key=lambda x: x):
    """
    Check if a session is sorted or not
    :param lst:
    :param key:
    :return:
    """
    for i, el in enumerate(lst[1:]):
        if key(el) != key(lst[i]) + 1:  # i is the index of the previous element
            return False
    return True


def extract_watching_session(df, session_threshold_hours=4):
    """
    Extract watching session of a (user, series) dataset chunk interaction.
    A watching session is a list of episodes a user has watched for a certain series
    For example, sequence [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16]
    becomes [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13],
            [4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16],
            [13, 14, 15, 16]
    that will be later appended to a dictionary series_id: sequences.
    When a user goes back while watching a certain series (for example, when it watches episode 4 after episode 13 or
    episode 13 after episode 16) it means a new sequence has begun, so we terminate a sequence and begin a new one in an
    empty array
    :param df: Relevant interactions of ContentWise Impression dataset
    :return: sequences for a dataset
    """
    # Variables initialization
    sessions = []
    previous = df.iloc[0]
    start_ts = previous.utc_ts_milliseconds
    sessions_time_intervals = []
    session_lengths = []
    session_in_order = []
    session = []
    count_final = 0
    for index, row in df.iterrows():
        # If episodes are out of order with respect to episode number, save up this sequence and empty the array
        if row["utc_ts_milliseconds"] > previous["utc_ts_milliseconds"] + pd.DateOffset(hours=session_threshold_hours):
            # Append to array of sequences and empty the sequence array so appended
            sessions.append(session)
            sessions_time_intervals.append(previous.utc_ts_milliseconds - start_ts)
            start_ts = row.utc_ts_milliseconds
            session_in_order.append(_is_sorted(session))
            session_lengths.append(len(session))
            session = []

        # Append current episode number to sequence, even if it was just emptied
        session.append(row["episode_number"])
        count_final += 1

        # When the last element of df is reached, return sequences
        if count_final == len(df):
            sessions_time_intervals.append(row.utc_ts_milliseconds - start_ts)
            session_in_order.append(_is_sorted(session))
            session_lengths.append(len(session))
            sessions.append(session)
            return sessions, len(sessions), session_lengths, np.array(session_lengths).mean(), session_in_order, (np.array(session_in_order) == True).mean() * 100, \
                   sessions_time_intervals, pd.Series(data=sessions_time_intervals).mean()
        previous = row


def extract_sequences(df):
    """
    Extract sequences of in-order of a (user, series) dataset chunk interaction.
    For example, sequence [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16]
    becomes [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13],
            [4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16],
            [13, 14, 15, 16]
    that will be later appended to a dictionary series_id: sequences.
    When a user goes back while watching a certain series (for example, when it watches episode 4 after episode 13 or
    episode 13 after episode 16) it means a new sequence has begun, so we terminate a sequence and begin a new one in an
    empty array
    :param df: Relevant interactions of ContentWise Impression dataset
    :return: sequences for a dataset
    """
    # Variables initialization
    sequences = []
    previous = df.iloc[0]
    sequence = [] * df["series_length"].iloc[0]
    count_final = 0
    for index, row in df.iterrows():
        # If episodes are out of order with respect to episode number, save up this sequence and empty the array
        if (row["user_id"] == previous["user_id"] and
                row["series_id"] == previous["series_id"] and
                row["episode_number"] < previous["episode_number"]):
            # Remove duplicates from episodes sequence list
            sequence = list(dict.fromkeys(sequence))
            # Append to array of sequences and empty the sequence array so appended
            sequences.append(sequence)
            sequence = [] * row["series_length"]

        # Append current episode number to sequence, even if it was just emptied
        sequence.append(row["episode_number"])
        count_final += 1

        # When the last element of df is reached, return sequences
        if count_final == len(df):
            sequences.append(sequence)
            return sequences
        previous = row


def separate_dataset_into_chunks(df):
    """
    Separate DataFrame into an array containing as element all the interactions of a (user_id, series_id)
    :param df: DataFrame of the relevant interactions of ContentWise Impression dataset
    :return: Array with dataset chunks
    """
    print("Separating DataFrame into chunks")
    dataset_chunk_array = [x for _, x in df.groupby(["user_id", "series_id"])]

    return dataset_chunk_array


def create_series_interactions_dictionary(dataset_chunk_array):
    """
    Create dictionary series_id: sequences
    :param dataset_chunk_array: Array with dataset chunks of ContentWise Impression dataset
    :return: dictionary series_id: sequences
    """
    print("Creating series: sequences dictionary")
    series_interactions = defaultdict(list)
    for df in tqdm(dataset_chunk_array):
        tmp_arr = extract_sequences(df)
        for tmp in tmp_arr:
            # Remove duplicates
            series_interactions[df["series_id"].iloc[0]].append(list(dict.fromkeys(tmp)))
    return series_interactions


def create_series_id_length_dict(df):
    """
    Creates a dictionary series_id: series_length
    :param df: DataFrame of ContentWise impression dataset
    :return: dictionary series_id: series_length
    """
    series_id_length_dict = pd.Series(df.series_length.values, index=df.series_id).to_dict()
    return series_id_length_dict


def create_probability_or_count_matrix(series_interactions, series_id_length_dict, normalize_probability=True):
    """
        Obtain from DataFrame the count matrix or the probability matrix (normalized count matrix)
        A count matrix is a matrix with shape [series_length x series_length] with row indices as the episode watched and column
        index as the position the episode was watched, and as element the number of times the i-th episode has been watched
        as j-th episode. This function is internal and should not be accessed
        :param series_interactions: relevant interactions of ContentWise Impression dataset
        :param series_id_length_dict: dictionary series id: series_length
        :param normalize_probability: if True normalize count matrix between 0 and 1
        :return: dictionary with series_id and corresponding count/probability matrix
    """
    # Dictionary {series_id: probability_matrix (np.array with shape [series_length, series_length])}
    id_probability_matrix_dict = defaultdict(partial(np.ndarray, 0))
    print("Computing probability/count matrix")
    # Loop through all the interactions in the id_sequence_matrix_dictionary
    for series_id, seq_vectors in tqdm(series_interactions.items()):
        # Generate a matrix of zeros of shape [series_length, series_length]
        count_matrix = np.zeros(shape=(series_id_length_dict[series_id], series_id_length_dict[series_id]))
        #     print(count_matrix)
        # Loop through indexes to compute elements of prob_matrix
        for seq_vector in seq_vectors:
            for ind, val in enumerate(seq_vector):
                # print(ind, val)
                count_matrix[val - 1, ind] += 1
        if normalize_probability:
            for ep_number in range(count_matrix.shape[0]):
                count_matrix[ep_number] = np.divide(count_matrix[ep_number],
                                                    max(np.sum(count_matrix[ep_number], axis=0), 1))

            id_probability_matrix_dict[series_id] = csr_matrix(count_matrix)
        else:
            id_probability_matrix_dict[series_id] = count_matrix

    return id_probability_matrix_dict


def get_series_with_n_interactions(df, n_interactions):
    """
    Get series with at least n_interactions
    :param df: DataFrame of ContentWise Impression dataset
    :param n_interactions: number of interactions we want
    :return: Dataframe with cols [series_id, interactions]
    """
    relevant_interactions = get_relevant_interactions(df)
    series_with_n_interactions = (relevant_interactions.groupby(["series_id"]).count()).user_id.to_frame(
        name="interactions").reset_index()
    series_with_n_interactions = series_with_n_interactions.sort_values(by="interactions", ascending=False)
    series_with_n_interactions = series_with_n_interactions[
        series_with_n_interactions["interactions"] >= n_interactions]
    return series_with_n_interactions


def compute_signal_function(parallel_chunk_array):
    """
    Get parallel function signal dictionary, a dictionary that tells how many series a user is watching in parlallel at each time
    :param parallel_chunk_array:
    :return:
    """
    user_parallel_signal_dict = defaultdict(list)
    for user_df in parallel_chunk_array:
        parallel_signal = []
        series_id_currently_watched = []
        for _, row in user_df.iterrows():
            if (row["series_id"] in series_id_currently_watched) and (row["episode_number"] == row["series_length"]):
                series_id_currently_watched.remove(row["series_id"])
            elif row["episode_number"] == 1:
                series_id_currently_watched.append(row["series_id"])
            parallel_signal.append(len(series_id_currently_watched))
        user_parallel_signal_dict[row["user_id"]] = parallel_signal
    return user_parallel_signal_dict


def _get_timestamp_differential_dataframe(dataset, percentage=100, session_threshold_hours=4, train=True, store=True):
    """
    Get timestamp differential dataframe. It will return a dataframe with:
    ->  delta_ts_episode_min, delta_ts_episode_max (respectively computing the timestamp differential between the minimum
        [maximum] timestamp of the last episode and the minimum timestamp of the first episode)
    ->  delta_ts_interaction: timestamp differential of the first and last interaction of a pair (user, series)
    ->  1_ep_every: average of the 3 delta_ts
    ->  n_interactions: number of interactions of a pair (user_series)
    ->  n_rewatches: difference between the number of interactions and the series length
    ->  sequences: the array of arrays of sequences of each (user, series) pair
    ->  sequences_len: lenght of each (user, series) sequences
    ->  first_episode: first episode of the first sequence
    ->  first_episode_categorical: if the first episode of the first sequence is "first_episode", "middle_episode", "last_episode"
    :param df: DataFrame of ContentWise Impression dataset
    :param percentage: percentage of at least watched episodes
    :param session_threshold_hours: hour threshold for computing a session, i.e. the allowed time to pass between two
                                    interactions for a (user, series) to be considered part of the same watching session
    :return: timestamp differential dataframe
    """
    if train:
        dataset, _, _ = user_temporal_split(dataset)

    df = clean_dataset(dataset)

    print(f"Getting timestamp differential DataFrame for series watched at least {percentage}%")
    ts_differential_df = watched_at_least_percentage_of_episodes(df, percentage)
    delta_ts_episode_min, delta_ts_episode_max, delta_ts_interactions, n_interactions, sequences, sequences_len, \
        first_ep, first_ep_categorical, watching_sessions, n_sessions, length_of_each_session, average_length_of_each_session,\
        n_sequential_sessions, percentages_sequential_sessions, sessions_time_intervals, sessions_time_average, \
        n_sessions_bingewatching_gt2, n_sessions_bingewatching_ge4, n_sessions_bingewatching_between_2_6, n_sessions_bingewatching_between_3_7 = ([] for _ in range(20))
    for _, row in tqdm(ts_differential_df.iterrows()):
        # delta_ts_episode_min row computation
        df_row = df[(df["user_id"] == row["user_id"]) & (df["series_id"] == row["series_id"])]
        max_episode = max(df_row.episode_number)
        delta_ts_episode_min.append(
            df_row[(df_row["episode_number"] == max_episode)].utc_ts_milliseconds.min() - df_row[
                df_row["episode_number"] == 1].utc_ts_milliseconds.min()
        )
        # delta_ts_max row computation
        delta_ts_episode_max.append(
            df_row[(df_row["episode_number"] == max_episode)].utc_ts_milliseconds.max() - df_row[
                df_row["episode_number"] == 1].utc_ts_milliseconds.min()
        )
        # delta_ts_interactions row computation
        delta_ts_interactions.append(
            df_row.iloc[-1].utc_ts_milliseconds - df_row.iloc[0].utc_ts_milliseconds
        )
        # n_interactions row append
        n_interactions.append(len(df_row))
        # sequence row append
        sequence = extract_sequences(df_row)
        sequences.append(sequence)
        # sequence length row append
        sequences_len.append(len(sequence))
        # first_ep row append
        first_ep.append(sequence[0][0])
        # first_ep_categorical row append
        if sequence[0][0] == 1:
            first_ep_categorical.append("first")
        elif sequence[0][0] == row["series_length"]:
            first_ep_categorical.append("last")
        else:
            first_ep_categorical.append("middle")
        # watching session stats row append
        watching_sessions_el, n_sessions_el, length_of_each_session_el, average_length_of_each_session_el, n_sequential_sessions_el, percentages_sequential_sessions_el, \
            sessions_time_intervals_el, sessions_time_average_el = extract_watching_session(df_row, session_threshold_hours=session_threshold_hours)
        watching_sessions.append(watching_sessions_el)
        n_sessions.append(n_sessions_el)
        length_of_each_session.append(length_of_each_session_el)
        average_length_of_each_session.append(average_length_of_each_session_el)
        n_sequential_sessions.append(n_sequential_sessions_el)
        percentages_sequential_sessions.append(percentages_sequential_sessions_el)
        sessions_time_intervals_el = [str(el) for el in sessions_time_intervals_el]
        sessions_time_intervals.append(sessions_time_intervals_el)
        sessions_time_average.append(str(sessions_time_average_el))

        lengths_of_each_session_np = np.array(length_of_each_session_el, dtype=int)
        n_sessions_bingewatching_gt2.append((lengths_of_each_session_np > 2).sum())
        n_sessions_bingewatching_ge4.append((lengths_of_each_session_np >= 4).sum())
        n_sessions_bingewatching_between_2_6.append(((lengths_of_each_session_np >= 2) & (lengths_of_each_session_np <= 6)).sum())
        n_sessions_bingewatching_between_3_7.append(((lengths_of_each_session_np >= 3) & (lengths_of_each_session_np <= 7)).sum())

    # delta_ts_episode_min append
    ts_differential_df["delta_ts_episode_min"] = delta_ts_episode_min
    # delta_ts_episode_max append
    ts_differential_df["delta_ts_episode_max"] = delta_ts_episode_max
    # delta_ts_interaction append
    ts_differential_df["delta_ts_interaction"] = delta_ts_interactions
    # 1_episode_every_min append
    ts_differential_df["1_episode_every_min"] = ts_differential_df["delta_ts_episode_min"] / ts_differential_df[
        "episodes_watched"]
    # 1_episode_every_max append
    ts_differential_df["1_episode_every_max"] = ts_differential_df["delta_ts_episode_max"] / ts_differential_df[
        "episodes_watched"]
    # 1_episode_every_interaction append
    ts_differential_df["1_episode_every_interaction"] = ts_differential_df["delta_ts_interaction"] / ts_differential_df[
        "episodes_watched"]
    # n_interactions append
    ts_differential_df["n_interactions"] = n_interactions
    # n_interactions - episodes watched absolute append
    ts_differential_df["interactions_episodes_watched_difference"] = ts_differential_df["n_interactions"] - \
                                                                     ts_differential_df["episodes_watched"]
    # n_interactions - episodes watched relative append
    ts_differential_df["interactions_episodes_watched_difference_rel"] = (ts_differential_df["n_interactions"] -
                                                                          ts_differential_df["episodes_watched"]) / \
                                                                         ts_differential_df["episodes_watched"]
    # sequences length append
    ts_differential_df["sequences_length"] = sequences_len
    # first episode append
    ts_differential_df["first_episode"] = first_ep
    # first episode categorical append
    ts_differential_df["first_episode_categorical"] = first_ep_categorical
    # sequences append
    ts_differential_df["sequences"] = sequences

    """
    Watching session elements append
    """
    # Watching session for a user, series
    ts_differential_df["watching_sessions"] = watching_sessions
    # Number of sessions for each user, series
    ts_differential_df["n_sessions"] = n_sessions
    # Bingewatching session with more than 2 episodes:
    ts_differential_df["n_bingewatching_sessions_3_or_more"] = n_sessions_bingewatching_gt2
    # Bingewatching session with 4 or more  episodes:
    ts_differential_df["n_bingewatching_sessions_4_or_more"] = n_sessions_bingewatching_ge4
    # Number of Bingewatching session between 2 and 6 episodes:
    ts_differential_df["n_bingewatching_sessions_2_to_6"] = n_sessions_bingewatching_between_2_6
    ts_differential_df["n_bingewatching_sessions_3_to_7"] = n_sessions_bingewatching_between_3_7

    # Length of each session for each user, series
    ts_differential_df["length_of_each_session"] = length_of_each_session
    # Average length of each session for each user, series
    ts_differential_df["average_length_of_each_session"] = average_length_of_each_session
    # Is in order? each session for each user, series
    ts_differential_df["is_session_sequential"] = n_sequential_sessions
    # Percentage of in order sequential sessions
    ts_differential_df["percentage_sequential_sessions"] = percentages_sequential_sessions
    # Sessions time intervals
    ts_differential_df["session_time_intervals"] = sessions_time_intervals
    # Session time averages
    ts_differential_df["average_time_watching_session"] = sessions_time_average

    if store:
        if train:
            ts_differential_df.to_parquet(os.path.join(ROOT_PROJECT_PATH,
                                                       f"parquets/ts_differential_df_{percentage}_{session_threshold_hours}hours_train.parquet"),
                                          compression=None, engine="fastparquet")
        else:
            ts_differential_df.to_parquet(os.path.join(ROOT_PROJECT_PATH,
                                                       f"parquets/ts_differential_df_{percentage}_{session_threshold_hours}hours.parquet"),
                                          compression=None, engine="fastparquet")

    return ts_differential_df


def get_timestamp_differential_dataframe(df=None, percentage=100, session_threshold_hours=4, train=True, load=True, store=True):
    """
    Get timestamp_differential_dataframe, if it exist a parquet save of it load it instad of recomputing
    :param df: DataFrame of ContentWise impression dataset
    :param percentage: percentage threshold, float
    :param session_threshold_hours: hour threshold for computing a session, i.e. the allowed time to pass between two
                                    interactions for a (user, series) to be considered part of the same watching session
    :param load: Bool, if True try loading from parquets folder
    :param store: Bool, if true store the DataFrame in the parquets folder
    :return:
    """
    if load:
        print(os.path.join(ROOT_PROJECT_PATH,
                                                f"parquets/ts_differential_df_{percentage}_{session_threshold_hours}hours.parquet" ))
        try:

            if train:
                print(
                    f"Loading existing parquet file in parquets/ts_differential_df_{percentage}_{session_threshold_hours}hours_train.parquet")
                return pd.read_parquet(os.path.join(ROOT_PROJECT_PATH,
                                                    f"parquets/ts_differential_df_{percentage}_{session_threshold_hours}hours_train.parquet"),
                                       engine="fastparquet")
            else:
                print(
                    f"Loading existing parquet file in parquets/ts_differential_df_{percentage}_{session_threshold_hours}hours.parquet")
                return pd.read_parquet(os.path.join(ROOT_PROJECT_PATH,
                                                    f"parquets/ts_differential_df_{percentage}_{session_threshold_hours}hours.parquet"),
                                       engine="fastparquet")
        except:
            if df is None:
                raise FileNotFoundError("Specify the df of CW impression in order to retrieve the BW information")
            print("Parquet non existent, computing and storing it...")
            return _get_timestamp_differential_dataframe(df, percentage=percentage,
                                                         session_threshold_hours=session_threshold_hours, store=store)
    else:
        return _get_timestamp_differential_dataframe(df, percentage=percentage, store=store, train=train)


def divide_df_into_quantiles(df, quantiles=[3, 8, 19, 31, 56]):
    """

    :param df:
    :param quantiles: list that contains the start interval of each quantiles. For example, [3, 8, 19, 31, 56] has
    (3,7), (8,18), (19, 30), (31, 55), (56, _) quantiles
    :return:
    """
    if quantiles is None:
        return df
    # Create DataFrame chunks based on array of series length, containing all series in one interval
    df_list = []
    for i in range(1, len(quantiles)):
        df_list.append(df[(df["series_length"] >= quantiles[i - 1]) & (df["series_length"] < quantiles[i])])
    # Last DF chunk will be the one with series with a series length greater than the last element passed as parameter
    df_list.append(df[df["series_length"] >= quantiles[len(quantiles) - 1]])
    return df_list


def get_series_length_groups(df, n_groups=3, mode="interactions", verbose=True):
    """
    Divide series into intervals of series lengths of the same number of elements (interactions, series_id or user_id)
    :param df: DataFrame of ContentWise Impression dataset
    :param n_groups: int, set the number of groups you want
    :param mode: interactions | series_id | user_id, decide what element to consider when partitioning
    :param verbose: bool, True to show prints
    :return:
    """
    assert mode == "interactions" or mode == "series_id" or mode == "user_id"
    count = 0
    df = df.sort_values("series_length")
    groups = [min(df.series_length)]
    if mode == "interactions":
        threshold = len(df) / n_groups
    else:
        threshold = df[mode].nunique() / n_groups
    if verbose:
        print("Threshold: ", threshold)
    for s_len in df.series_length.unique():
        if mode == "interactions":
            count += len(df[df["series_length"] == s_len])
        else:
            tmp_df = df[df["series_length"] == s_len]
            count += tmp_df[mode].nunique()
        if count > threshold:
            if verbose:
                print(f"{mode} for group {len(groups)} = {count}")

            groups.append(s_len)
            if len(groups) == n_groups:
                if verbose:
                    if mode == "interactions":
                        count = len(df[df["series_length"] >= s_len])
                    else:
                        tmp_df = df[df["series_length"] >= s_len]
                        count = tmp_df[mode].nunique()
                    print(f"{mode} for group {len(groups)} = {count}")
                    print("Groups: ", groups)
                return groups
            count = 0


"""
-----------------------------------------------
|                RECSYS UTILS                 |                                                              
-----------------------------------------------
"""

#
#
# def elasticnet_URM_train_with_bingewatching(URM_train, dataset=None, percentage_watched=50, hour_threshold=4, explicit_URM=True):
#     """
#     Add to the URM train of SLIM ElasticNet one row vector and one column vector:
#     - row vector contains the bingeworthiness of series j
#     - column vector contains if the user i is a bingewatcher or not
#     :param dataset: data
#     :param URM_train:
#     :param percentage_watched:
#     :param hour_threshold:
#     :return:
#     """
#     # Retrieve corresponding df bw
#     # if dataset is not None:
#     df_bw = get_timestamp_differential_dataframe(df=dataset, percentage=percentage_watched, session_threshold_hours=hour_threshold, train=True)
#     # else:
#     #     df_bw = get_timestamp_differential_dataframe(percentage=percentage_watched, session_threshold_hours=hour_threshold, use_items=use_items)
#     print("Df_bw loaded")
#     # Generate bingewatchers column to be addedat the right of the URM
#
#
#     if explicit_URM:
#
#         def _read_dictionary(local_file_path: str) -> dict:
#             with open(local_file_path, "r") as f:
#                 return json.load(f)
#
#         translation_user_id_index_urm = _read_dictionary(os.path.join(os.getcwd(),
#                                                "data",
#                                                "ContentWiseImpressions",
#                                                "CW10M",
#                                                "translation_user_id_index_urm.json"))
#         translation_series_id_index_urm = _read_dictionary(os.path.join(os.getcwd(),
#                                                "data",
#                                                "ContentWiseImpressions",
#                                                "CW10M",
#                                                "translation_series_id_index_urm.json"))
#
#         user_indices = np.unique(df_bw.user_id.values, return_index=True)[1]
#
#         bingewatchers = {translation_user_id_index_urm[str(user)]: bw for user, bw in zip([df_bw.user_id.values[index] for index in sorted(user_indices)],
#                                                                                           np.array((df_bw.groupby(
#                                                                                               "user_id").sum().n_bingewatching_sessions_3_to_7).values,
#                                                                                                    dtype=int))
#                          if bw != 0}
#
#         URM_bingewatchers_column_array = np.zeros((URM_train.shape[0]+1, 1), dtype=int)
#
#         for k, i in bingewatchers.items():
#             URM_bingewatchers_column_array[k, 0] = i
#
#         # Generate bingeworthy row to be added at the bottom of the URM
#         series_indices = np.unique(df_bw.series_id.values, return_index=True)[1]
#
#         bingeworthy_series = {translation_series_id_index_urm[str(series)]: bw for series, bw in zip([df_bw.series_id.values[index] for index in sorted(series_indices)],
#                                                                                                      np.array(
#                                                                                                          (df_bw.groupby(
#                                                                                                              "series_id").sum().n_bingewatching_sessions_3_to_7).values,
#                                                                                                          dtype=int))
#                             if bw != 0}
#
#         URM_bingeworthy_row_array = np.zeros((1, URM_train.shape[1]), dtype=int)
#
#         for k, i in bingeworthy_series.items():
#             URM_bingeworthy_row_array[0, k] = i
#     else:
#         user_indices = np.unique(df_bw.user_id.values, return_index=True)[1]
#
#         bingewatchers = {user: bw for user, bw in zip([df_bw.user_id.values[index] for index in sorted(user_indices)],
#                                                       np.array((df_bw.groupby(
#                                                           "user_id").sum().n_bingewatching_sessions_3_to_7 /
#                                                                 df_bw.groupby(
#                                                                     "user_id").sum().n_sessions).values >= 0.5,
#                                                                dtype=int))
#                          if bw != 0}
# #yoo
#         URM_bingewatchers_column_array = np.zeros((URM_train.shape[0]+1, 1), dtype=int)
#
#         URM_bingewatchers_column_array[[k for k in bingewatchers.keys()]] = 1
#
#         # Generate bingeworthy row to be added at the bottom of the URM
#         series_indices = np.unique(df_bw.series_id.values, return_index=True)[1]
#
#         bingeworthy_series = {series: bw for series, bw in
#                               zip([df_bw.series_id.values[index] for index in sorted(series_indices)],
#                                   np.array(
#                                       (df_bw.groupby("series_id").sum().n_bingewatching_sessions_3_to_7 /
#                                        df_bw.groupby("series_id").sum().n_sessions).values >= 0.5,
#                                       dtype=int))
#                               if bw != 0}
#
#         URM_bingeworthy_row_array = np.zeros((1, URM_train.shape[1]), dtype=int)
#
#         URM_bingeworthy_row_array[0, [k for k in bingeworthy_series.keys()]] = 1
#
#     return URM_bingewatchers_column_array, URM_bingeworthy_row_array
#     # URM_train = URM_train.todense()
#     # URM_train = np.concatenate((URM_train, URM_bingeworthy_row_array), axis=0)
#     # URM_train = np.concatenate((URM_train, URM_bingewatchers_column_array), axis=1)
#     # return csc_matrix(URM_train, dtype=int)
#


def get_feature_vector_URM(URM_train, dataset=None, percentage_watched=50, hour_threshold=4, explicit_URM=True):
    """
    Add to the URM train of SLIM ElasticNet one row vector and one column vector:
    - row vector contains the bingeworthiness of series j
    - column vector contains if the user i is a bingewatcher or not
    :param dataset: data
    :param URM_train:
    :param percentage_watched:
    :param hour_threshold:
    :return:
    """
    # Retrieve corresponding df bw
    # if dataset is not None:
    df_bw = get_timestamp_differential_dataframe(df=dataset, percentage=percentage_watched, session_threshold_hours=hour_threshold, train=True)
    # else:
    #     df_bw = get_timestamp_differential_dataframe(percentage=percentage_watched, session_threshold_hours=hour_threshold, use_items=use_items)
    print("Df_bw loaded")
    # Generate bingewatchers column to be addedat the right of the URM


    if explicit_URM:

        def _read_dictionary(local_file_path: str) -> dict:
            with open(local_file_path, "r") as f:
                return json.load(f)

        translation_user_id_index_urm = _read_dictionary(os.path.join(os.getcwd(),
                                               "data",
                                               "ContentWiseImpressions",
                                               "CW10M",
                                               "translation_user_id_index_urm.json"))
        translation_series_id_index_urm = _read_dictionary(os.path.join(os.getcwd(),
                                               "data",
                                               "ContentWiseImpressions",
                                               "CW10M",
                                               "translation_series_id_index_urm.json"))

        user_indices = np.unique(df_bw.user_id.values, return_index=True)[1]

        bingewatchers = {translation_user_id_index_urm[str(user)]: bw for user, bw in zip([df_bw.user_id.values[index] for index in sorted(user_indices)],
                                                                                          np.array((df_bw.groupby(
                                                                                              "user_id").sum().n_bingewatching_sessions_3_to_7).values,
                                                                                                   dtype=int))
                         if bw != 0}

        URM_bingewatchers_column_array = np.zeros((URM_train.shape[0]+1, 1), dtype=int)

        for k, i in bingewatchers.items():
            URM_bingewatchers_column_array[k, 0] = i

        # Generate bingeworthy row to be added at the bottom of the URM
        series_indices = np.unique(df_bw.series_id.values, return_index=True)[1]

        bingeworthy_series = {translation_series_id_index_urm[str(series)]: bw for series, bw in zip([df_bw.series_id.values[index] for index in sorted(series_indices)],
                                                                                                     np.array(
                                                                                                         (df_bw.groupby(
                                                                                                             "series_id").sum().n_bingewatching_sessions_3_to_7).values,
                                                                                                         dtype=int))
                            if bw != 0}

        URM_bingeworthy_row_array = np.zeros((1, URM_train.shape[1]), dtype=int)

        for k, i in bingeworthy_series.items():
            URM_bingeworthy_row_array[0, k] = i
    else:
        user_indices = np.unique(df_bw.user_id.values, return_index=True)[1]

        bingewatchers = {user: bw for user, bw in zip([df_bw.user_id.values[index] for index in sorted(user_indices)],
                                                      np.array((df_bw.groupby(
                                                          "user_id").sum().n_bingewatching_sessions_3_to_7 /
                                                                df_bw.groupby(
                                                                    "user_id").sum().n_sessions).values >= 0.5,
                                                               dtype=int))
                         if bw != 0}

        URM_bingewatchers_column_array = np.zeros((URM_train.shape[0]+1, 1), dtype=int)

        URM_bingewatchers_column_array[[k for k in bingewatchers.keys()]] = 1

        # Generate bingeworthy row to be added at the bottom of the URM
        series_indices = np.unique(df_bw.series_id.values, return_index=True)[1]

        bingeworthy_series = {series: bw for series, bw in
                              zip([df_bw.series_id.values[index] for index in sorted(series_indices)],
                                  np.array(
                                      (df_bw.groupby("series_id").sum().n_bingewatching_sessions_3_to_7 /
                                       df_bw.groupby("series_id").sum().n_sessions).values >= 0.5,
                                      dtype=int))
                              if bw != 0}

        URM_bingeworthy_row_array = np.zeros((1, URM_train.shape[1]), dtype=int)

        URM_bingeworthy_row_array[0, [k for k in bingeworthy_series.keys()]] = 1

    return URM_bingewatchers_column_array, URM_bingeworthy_row_array

def concatenate_bw_features_with_URM(URM_train, URM_bingewatchers_column_array=None, URM_bingeworthy_row_array=None):

    dtype=URM_train.dtype
    URM_train = URM_train.todense()
    if URM_bingeworthy_row_array is not None:
        URM_train = np.concatenate((URM_train, URM_bingeworthy_row_array), axis=0)
    if URM_bingewatchers_column_array is not None:
        if URM_bingeworthy_row_array is None:
            URM_train = np.concatenate((URM_train, URM_bingewatchers_column_array[:-1, :]), axis=1)
        else:
            URM_train = np.concatenate((URM_train, URM_bingewatchers_column_array), axis=1)
    # TODO:
    matrix = csc_matrix(URM_train, dtype=dtype)
    matrix.eliminate_zeros()
    return csc_matrix(matrix, dtype=dtype)


def lightFM_user_item_features(URM_train, dataset=None, percentage_watched=50, hour_threshold=4):
    """
    Create user and item features for LightFM:
    - row vector contains the bingeworthiness of series j
    - column vector contains if the user i is a bingewatcher or not
    :param dataset: data
    :param URM_train:
    :param percentage_watched:
    :param hour_threshold:
    :return:
    """
    # Retrieve corresponding df bw
    # if dataset is not None:
    df_bw = get_timestamp_differential_dataframe(df=dataset, percentage=percentage_watched, session_threshold_hours=hour_threshold, train=True)
    # else:
    #     df_bw = get_timestamp_differential_dataframe(percentage=percentage_watched, session_threshold_hours=hour_threshold, use_items=use_items)
    print("Df_bw loaded")
    # Generate bingewatchers column to be addedat the right of the URM
    user_indices = np.unique(df_bw.user_id.values, return_index=True)[1]

    bingewatchers = {user: bw for user, bw in zip([df_bw.user_id.values[index] for index in sorted(user_indices)],
                                                  np.array((df_bw.groupby("user_id").sum().n_bingewatching_sessions_3_to_7 /
                                                            df_bw.groupby("user_id").sum().n_sessions).values >= 0.5,dtype=int))
                     if bw != 0}


    user_features = [(user, "bingewatcher:1") for user in bingewatchers.keys()]
    not_bingewatchers = list(set(range(URM_train.shape[0])) - set(bingewatchers.keys()))
    for user in not_bingewatchers:
        user_features.append((user, "bingewatcher:0"))
    # Generate bingeworthy row to be added at the bottom of the URM
    series_indices = np.unique(df_bw.series_id.values, return_index=True)[1]

    bingeworthy_series = {series: bw for series, bw in zip([df_bw.series_id.values[index] for index in sorted(series_indices)],
                                                  np.array(
                                                      (df_bw.groupby("series_id").sum().n_bingewatching_sessions_3_to_7 /
                                                       df_bw.groupby("series_id").sum().n_sessions).values >= 0.5,
                                                      dtype=int))
                            if bw != 0}

    series_features = [(series, "bingeworthy:1") for series in bingeworthy_series.keys()]
    not_bingeworthy = list(set(range(URM_train.shape[1])) - set(bingeworthy_series.keys()))
    for series in not_bingeworthy:
        series_features.append((series, "bingeworthy:0"))

    # print(user_features[:4000])
    # print(series_features)
    return user_features, series_features


def lightFM_user_item_features_list(URM_train, dataset=None, percentage_watched=50, hour_threshold=4):
    """
    Create user and item features for LightFM:
    - row vector contains the bingeworthiness of series j
    - column vector contains if the user i is a bingewatcher or not
    :param dataset: data
    :param URM_train:
    :param percentage_watched:
    :param hour_threshold:
    :return:
    """
    # Retrieve corresponding df bw
    # if dataset is not None:
    df_bw = get_timestamp_differential_dataframe(df=dataset, percentage=percentage_watched, session_threshold_hours=hour_threshold, train=True)
    # else:
    #     df_bw = get_timestamp_differential_dataframe(percentage=percentage_watched, session_threshold_hours=hour_threshold, use_items=use_items)
    print("Df_bw loaded")
    # Generate bingewatchers column to be addedat the right of the URM
    user_indices = np.unique(df_bw.user_id.values, return_index=True)[1]

    bingewatchers = {user: bw for user, bw in zip([df_bw.user_id.values[index] for index in sorted(user_indices)],
                                                  np.array((df_bw.groupby("user_id").sum().n_bingewatching_sessions_3_to_7 /
                                                            df_bw.groupby("user_id").sum().n_sessions).values >= 0.5,dtype=int))
                     if bw != 0}


    user_features = [("bingewatcher:1") for user in bingewatchers.keys()]
    not_bingewatchers = list(set(range(URM_train.shape[0])) - set(bingewatchers.keys()))
    for user in not_bingewatchers:
        user_features.append(("bingewatcher:0"))
    # Generate bingeworthy row to be added at the bottom of the URM
    series_indices = np.unique(df_bw.series_id.values, return_index=True)[1]

    bingeworthy_series = {series: bw for series, bw in zip([df_bw.series_id.values[index] for index in sorted(series_indices)],
                                                  np.array(
                                                      (df_bw.groupby("series_id").sum().n_bingewatching_sessions_3_to_7 /
                                                       df_bw.groupby("series_id").sum().n_sessions).values >= 0.5,
                                                      dtype=int))
                            if bw != 0}

    series_features = [("bingeworthy:1") for series in bingeworthy_series.keys()]
    not_bingeworthy = list(set(range(URM_train.shape[1])) - set(bingeworthy_series.keys()))
    for series in not_bingeworthy:
        series_features.append(("bingeworthy:0"))

    # print(user_features[:4000])
    # print(series_features)
    return user_features, series_features


def user_series_bw_ICMs(URM_train, dataset=None, percentage_watched=50, hour_threshold=4):
    """
        Create user and item features for LightFM:
        - row vector contains the bingeworthiness of series j
        - column vector contains if the user i is a bingewatcher or not
        :param dataset: data
        :param URM_train:
        :param percentage_watched:
        :param hour_threshold:
        :return:
        """
    # Retrieve corresponding df bw
    # if dataset is not None:
    df_bw = get_timestamp_differential_dataframe(df=dataset, percentage=percentage_watched,
                                                 session_threshold_hours=hour_threshold, train=True)
    # else:
    #     df_bw = get_timestamp_differential_dataframe(percentage=percentage_watched, session_threshold_hours=hour_threshold, use_items=use_items)
    print("Df_bw loaded")
    # Generate bingewatchers column to be addedat the right of the URM
    user_indices = np.unique(df_bw.user_id.values, return_index=True)[1]

    bingewatchers = {user: bw for user, bw in zip([df_bw.user_id.values[index] for index in sorted(user_indices)],
                                                  np.array(
                                                      (df_bw.groupby("user_id").sum().n_bingewatching_sessions_3_to_7 /
                                                       df_bw.groupby("user_id").sum().n_sessions).values >= 0.5,
                                                      dtype=int))
                     if bw != 0}

    user_features = []
    print("Building user features...")
    for user in tqdm(range(URM_train.shape[0])):
        if user in bingewatchers:
            user_features.append([1, 1])
        else:
            user_features.append([1, 0])

    user_features_diagonal_offset = np.identity(len(user_features))
    user_features = np.array(user_features)
    # Generate bingeworthy row to be added at the bottom of the URM
    series_indices = np.unique(df_bw.series_id.values, return_index=True)[1]

    bingeworthy_series = {series: bw for series, bw in
                          zip([df_bw.series_id.values[index] for index in sorted(series_indices)],
                              np.array(
                                  (df_bw.groupby("series_id").sum().n_bingewatching_sessions_3_to_7 /
                                   df_bw.groupby("series_id").sum().n_sessions).values >= 0.5,
                                  dtype=int))
                          if bw != 0}

    series_features = []
    for series in tqdm(range(URM_train.shape[1])):
        if series in bingeworthy_series:
            series_features.append([1, 1])
        else:
            series_features.append([1, 0])
    series_features_diagonal_offset = np.identity(len(series_features))

    series_features = np.array(series_features)
    user_features, series_features = np.concatenate((user_features, user_features_diagonal_offset), axis=1), np.concatenate((series_features, series_features_diagonal_offset), axis=1)
    return csr_matrix(user_features), csr_matrix(series_features)


def user_series_bw_ICMs_explicit(URM_train, dataset=None, percentage_watched=50, hour_threshold=4):
    """
            Create user and item features for LightFM:
            - row vector contains the bingeworthiness of series j
            - column vector contains if the user i is a bingewatcher or not
            :param dataset: data
            :param URM_train:
            :param percentage_watched:
            :param hour_threshold:
            :return:
    """
    df_bw = get_timestamp_differential_dataframe(df=dataset, percentage=percentage_watched,
                                                 session_threshold_hours=hour_threshold, train=True)
    # else:
    #     df_bw = get_timestamp_differential_dataframe(percentage=percentage_watched, session_threshold_hours=hour_threshold, use_items=use_items)
    print("Df_bw loaded")
    def _read_dictionary(local_file_path: str) -> dict:
        with open(local_file_path, "r") as f:
            return json.load(f)

    translation_user_id_index_urm = _read_dictionary(os.path.join(os.getcwd(),
                                                                  "data",
                                                                  "ContentWiseImpressions",
                                                                  "CW10M",
                                                                  "translation_user_id_index_urm.json"))
    translation_series_id_index_urm = _read_dictionary(os.path.join(os.getcwd(),
                                                                    "data",
                                                                    "ContentWiseImpressions",
                                                                    "CW10M",
                                                                    "translation_series_id_index_urm.json"))

    user_indices = np.unique(df_bw.user_id.values, return_index=True)[1]

    bingewatchers = {translation_user_id_index_urm[str(user)]: bw for user, bw in
                     zip([df_bw.user_id.values[index] for index in sorted(user_indices)],
                         np.array(df_bw.groupby("user_id").sum().n_bingewatching_sessions_3_to_7.values,
                                  dtype=int))
                     if bw != 0}

    user_features = []
    for user in tqdm(range(URM_train.shape[0])):
        if user in bingewatchers:
            user_features.append(bingewatchers[user])
        else:
            user_features.append(0)



    # Generate bingeworthy row to be added at the bottom of the URM
    series_indices = np.unique(df_bw.series_id.values, return_index=True)[1]

    bingeworthy_series = {translation_series_id_index_urm[str(series)]: bw for series, bw in
                          zip([df_bw.series_id.values[index] for index in sorted(series_indices)],
                              np.array(
                                  df_bw.groupby("series_id").sum().n_bingewatching_sessions_3_to_7.values,
                                  dtype=int))
                          if bw != 0}
    #

    series_features = []
    for series in tqdm(range(URM_train.shape[1])):
        if series in bingeworthy_series:
            series_features.append(bingeworthy_series[series])
        else:
            series_features.append(0)

    return np.array(user_features, dtype=int), np.array(series_features, dtype=int)


"""
-----------------------------------------------
|                SPLIT FUNCTIONS              |                                                              
-----------------------------------------------
"""


def global_temporal_split(df):
    """
    Temporal global split of the dataset into train, test and validation
    :param df: dataset self.interactions
    :return: global temporal split, train (70%) validation (10%) test (20%)
    """
    return df[:int(len(df)*0.7)], df[int(len(df)*0.7):int(len(df)*0.8)], df[int(len(df)*0.8):]


def user_temporal_split(df):
    """
    User temporal split of the dataset into train, test and validation.
    For each user, take first 70% for training, next 10% for validation and next 20% for testing
    :param df: dataset self.interactions, already pruned from duplicates
    :return:
    """
    dataset_chunk_array = [x for _, x in df.groupby("user_id")]
    train, validation, test = [], [], []
    for d in dataset_chunk_array:
        train.append(d[:int(len(d) * 0.7)])
        validation.append(d[int(len(d) * 0.7):int(len(d) * 0.8)])
        test.append(d[int(len(d) * 0.8):])

    # Return train, test, validation
    return pd.concat(train), pd.concat(validation), pd.concat(test)

"""
-----------------------------------------------
|                STATISTICS                   |                                                              
-----------------------------------------------
"""


def get_dataset_statistics(df, df_type="dataset"):
    """
    Get number of unique users, series and number of interactions of a given DataFrame
    :param df_type: Type of DataFrame, "describe" | "dataset" (default)
    :param df: DataFrame of ContentWise Impression dataset
    :return: number of unique users, series and number of interactions of a given DataFrame
    """
    assert df_type == "dataset" or df_type == "describe"

    if df_type == "describe":
        print("Number of rows of the describe DataFrame: ", len(df))
        print("Number of corresponding interactions in the dataset: ", sum(df["n_interactions"]))
    else:
        print("Number of interactions in the dataset: ", len(df))
    print("Number of unique users: ", df["user_id"].nunique())
    print("Number of unique series: ", df["series_id"].nunique())


def difference_after_filtering(old_df, new_df, old_type="dataset", new_type="dataset", print_all=True):
    """
    Get difference in number of unique users, series and number of interactions of a given DataFrame with respect to the old DataFrame
    :param old_df: Old DataFrame
    :param new_df: Filtered DataFrame
    :return: difference in number of unique users, series and number of interactions of a given DataFrame with respect to the old DataFrame
    """
    assert old_type == "dataset" or old_type == "describe"
    assert new_type == "dataset" or new_type == "describe"

    if print_all:
        print(f"Unfiltered {old_type} statistics:")
        get_dataset_statistics(old_df, df_type=old_type)
        print("")
        print(f"Filtered {new_type} statistics:")
        get_dataset_statistics(new_df, df_type=new_type)
        print("")
    # If a DataFrame is of type "describe", the length of the DataFrame doesn't correspond to the number of interactions
    # instead, the number of interactions in the dataset is given by the sum of values over column "n_interactions"
    print("Differences between the two datasets:")
    if new_type == "describe":
        interactions_considered_new = sum(new_df["n_interactions"])
    else:
        interactions_considered_new = len(new_df)

    if old_type == "describe":
        interactions_considered_old = sum(old_df["n_interactions"])
    else:
        interactions_considered_old = len(old_df)
    print(f"Number of removed interactions: {interactions_considered_old - interactions_considered_new}, "
          f"{(interactions_considered_old - interactions_considered_new) / interactions_considered_old * 100}% of the unfiltered DataFrame")
    print(f"Number of removed users: {old_df['user_id'].nunique() - new_df['user_id'].nunique()}, "
          f"{(old_df['user_id'].nunique() - new_df['user_id'].nunique()) / old_df['user_id'].nunique() * 100}% of the unfiltered DataFrame")
    print(f"Number of removed series: {old_df['series_id'].nunique() - new_df['series_id'].nunique()}, "
          f"{(old_df['series_id'].nunique() - new_df['series_id'].nunique()) / old_df['series_id'].nunique() * 100}% of the unfiltered DataFrame")


def get_average_episodes_per_series(df):
    """
    Returns average number of episodes per series
    :param df: DataFrame of the ContentWise Impression dataset
    :return: average number of episodes per series
    """
    episodes_per_series = df[["series_id", "series_length"]].drop_duplicates()
    episodes_per_series = episodes_per_series.reset_index(drop=True)
    mean = episodes_per_series["series_length"].mean()
    return mean


def users_missing_n_episodes(df, n_episodes_missing):
    """
    Returns the number of users that are missing exactly n_episodes
    :param df: DataFrame of the ContentWise Impression dataset
    :param n_episodes_missing: number of missing episode for each user
    :return: number of (users, series_id) that have watched a series - n_episodes_missing
    """
    df_grouped = (df.groupby(["user_id", "series_id", "series_length"]).episode_number.nunique()).to_frame(
        name="episodes_watched").reset_index()
    df_grouped = df_grouped.sort_values(by=["episodes_watched", "user_id"], ascending=False)
    return df_grouped[(df_grouped["series_length"] - n_episodes_missing) == df_grouped["episodes_watched"]]


def users_missing_max_n_episodes(df, n_episodes_missing):
    """
    Returns the number of users that are missing at max n_episodes
    :param df: DataFrame of the ContentWise Impression dataset
    :param n_episodes_missing: number of missing episode for each user
    :return: number of (users, series_id) that have watched at least a series - n_episodes_missing
    """
    df_grouped = (df.groupby(["user_id", "series_id", "series_length"]).episode_number.nunique()).to_frame(
        name="episodes_watched").reset_index()
    df_grouped = df_grouped.sort_values(by=["episodes_watched", "user_id"], ascending=False)
    return df_grouped[(df_grouped["series_length"] - n_episodes_missing) <= df_grouped["episodes_watched"]]


def watched_at_least_percentage_of_episodes(df, percentage):
    """
    Returns the number of users that are missing at max n_episodes
    :param df: DataFrame of the ContentWise Impression dataset
    :param percentage: percentage watched of (user, series) paris
    :return: DataFrame [["user_id", "series_id", "episodes_watched", "series_length", "percentage"]] with percentage >= percentage
    """
    df_grouped = get_episode_watched(df, include_percentage=True, already_cleaned=True)
    return df_grouped[(df_grouped["percentage"] >= percentage) & (df_grouped["percentage"] <= 100)]


def probability_or_count_matrix(df, probability=True, already_relevant_interactions=False):
    """
    Obtain from DataFrame the count matrix or the probability matrix (normalized count matrix)
    A count matrix is a matrix with shape [series_length x series_length] with row indices as the episode watched and column
    index as the position the episode was watched, and as element the number of times the i-th episode has been watched
    as j-th episode
    :param df: DataFrame of the ContentWise impression dataset
    :param probability: if we want a normalized probability matrix or a count matrix
    :param already_relevant_interactions: if the DataFrame has been already filtered to obtain relevant interactions
    :return: count matrix or the probability matrix (normalized count matrix)
    """
    # If interactions are not already the relevant ones, filter them
    if not already_relevant_interactions:
        df = get_relevant_interactions(df)

    # 1) Create data chunk array
    data_chunk_array = separate_dataset_into_chunks(df)
    # 2) Create series_interactions dictionary with array of sequences for each series
    print("Extracting sequences")
    series_interactions = create_series_interactions_dictionary(data_chunk_array)
    # 3) Create series_id: series_length dictionary
    series_id_length_dict = create_series_id_length_dict(df)
    # 4) Create probability/count matrix
    prob_or_count_matrix = create_probability_or_count_matrix(series_interactions, series_id_length_dict,
                                                              normalize_probability=probability)

    return prob_or_count_matrix


def absolute_pearsonr_average_count_matrix(id_count_matrix_dict, top_n_series):
    """
    Absolute pearson r average between the count matrices of the top n series
    :param id_count_matrix_dict: Count matrix dictionary
    :param top_n_series: DataFrame containing the series with at least n interactions
    :return: Pearson r average
    """
    r_tot = 0
    for s_id in tqdm(top_n_series.series_id.values):
        """ 
        Create a diagonal matrix filled with 1 on diagonal of shape (series_length, series_length)
        This matrix represent the probability matrix of that series if users watched all episodes in sequence without
        any rewatches 
        """
        diag_matrix = np.zeros(shape=id_count_matrix_dict[s_id].shape)
        np.fill_diagonal(diag_matrix, 1)

        # Compute the pearson r value between the probability matrix and the 100% serial fake matrix created
        r_tot += (stats.pearsonr(id_count_matrix_dict[s_id].flatten(), diag_matrix.flatten())[0])
    return r_tot / len(top_n_series.series_id.values)


def absolute_pearsonr_average_probability_matrix(id_probability_matrix_dict, top_n_series):
    """
    Absolute pearson r average between the count matrices of the top n series
    :param id_probability_matrix_dict: Probability matrix dictionary
    :param top_n_series: DataFrame containing the series with at least n interactions
    :return: Pearson r average
    """
    r_tot = 0
    for s_id in tqdm(top_n_series.series_id.values):
        """ 
        Create a diagonal matrix filled with 1 on diagonal of shape (series_length, series_length)
        This matrix represent the probability matrix of that series if users watched all episodes in sequence without
        any rewatches 
        """
        diag_matrix = np.zeros(shape=id_probability_matrix_dict[s_id].toarray().shape)
        np.fill_diagonal(diag_matrix, 1)

        # Compute the pearson r value between the probability matrix and the 100% serial fake matrix created
        r_tot += (stats.pearsonr(id_probability_matrix_dict[s_id].toarray().flatten(), diag_matrix.flatten())[0])
    return r_tot / len(top_n_series.series_id.values)


def weighted_pearsonr_average_probability_matrix(id_probability_matrix_dict, top_n_series):
    """
    Weighted pearson r average between the probability matrices of the top n series
    :param id_probability_matrix_dict: probability matrix dict
    :param top_n_series: DataFrame containing the series with at least n interactions
    :return: weighted Pearson r average
    """
    r_tot = 0
    for _, row in tqdm(top_n_series.iterrows()):
        """ 
        Create a diagonal matrix filled with 1 on diagonal of shape (series_length, series_length)
        This matrix represent the probability matrix of that series if users watched all episodes in sequence without
        any rewatches 
        """
        diag_matrix = np.zeros(shape=id_probability_matrix_dict[row["series_id"]].toarray().shape)
        np.fill_diagonal(diag_matrix, 1)

        # Compute the pearson r value between the probability matrix and the 100% serial fake matrix created
        r_tot += (stats.pearsonr(id_probability_matrix_dict[row["series_id"]].toarray().flatten(),
                                 diag_matrix.flatten())[0]) * row["interactions"]
    return r_tot / np.sum(top_n_series.interactions.values)


def weighted_pearsonr_average_count_matrix(id_count_matrix_dict, top_n_series):
    """
    Weighted pearson r average between the count matrices of the top n series
    :param id_count_matrix_dict: count matrix dict
    :param top_n_series: DataFrame containing the series with at least n interactions
    :return: weighted Pearson r average
    """
    r_tot = 0
    for _, row in tqdm(top_n_series.iterrows()):
        """ 
        Create a diagonal matrix filled with 1 on diagonal of shape (series_length, series_length)
        This matrix represent the probability matrix of that series if users watched all episodes in sequence without
        any rewatches 
        """
        diag_matrix = np.zeros(shape=id_count_matrix_dict[row["series_id"]].shape)
        np.fill_diagonal(diag_matrix, 1)

        # Compute the pearson r value between the probability matrix and the 100% serial fake matrix created
        r_tot += (stats.pearsonr(id_count_matrix_dict[row["series_id"]].flatten(), diag_matrix.flatten())[0]) * row[
            "interactions"]
    return r_tot / np.sum(top_n_series.interactions.values)


def get_parallel_series_for_user(df, user_id=None):
    """
    Get parallel signal function, if user_id is specified return it for that user
    :param df:
    :param user_id:
    :return:
    """
    df_ts_differential = get_timestamp_differential_dataframe(df)
    parallel_chunk_array = [x.sort_values("utc_ts_milliseconds") for _, x in df_ts_differential.groupby(["user_id"])]
    signal_function = compute_signal_function(parallel_chunk_array)
    if user_id:
        return signal_function[user_id]
    return signal_function


def describe_users_or_series(df, type="user_id", file_name=None, load=True, store=True,
                             percentiles=[.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85,
                                          .9, .95]):
    """
    User/series groupby describe
    :param df: DataFrame of the Bingewatching stats
    :param type: "series_id" | "user_id"
    :param file_name: path to load/store parquet
    :param load: True to load parquet
    :param store: True to store parquet
    :return: user/series groupby describe DataFrame
    """
    assert type == "series_id" or type == "user_id"
    if file_name is None:
        print("Please specify a path to load/store parquet")
    if load and file_name is not None:
        try:
            print(f"Loading existing parquet file in parquets/{file_name}.parquet")
            return pd.read_parquet(os.path.join(ROOT_PROJECT_PATH, f"parquets/{file_name}.parquet"))
        except:
            print("Parquet non existent, computing it")

    df_describe = df.groupby(type).describe(percentiles=percentiles, include="all")
    if store:
        print(f"Storing {type} describe parquet into parquets/{file_name}.parquet")
        df_store = df_describe
        df_store.columns = df_store.columns.map(str)
        df_store.to_parquet(os.path.join(ROOT_PROJECT_PATH, f"parquets/{file_name}.parquet"), compression=None)
    return df_describe


def watching_sessions_stats(df, just_bingewatching=False, quantiles=[3, 8, 19, 31, 56]):
    """
    Return a DF with the describe for each quantile
    :param df: Bingewatching DF
    :param quantiles: list that contains the start interval of each quantiles. For example, [3, 8, 19, 31, 56] has
    (3,7), (8,18), (19, 30), (31, 55), (56, _) quantiles
    :return: list of DFs
    """
    if just_bingewatching:
        df = df[(df["n_bingewatching_sessions_3_or_more"] > 0) | (df["n_bingewatching_sessions_2_to_6"] > 0)]
    df_list = divide_df_into_quantiles(df, quantiles=quantiles)
    df_describes = []
    for chunk in df_list:
        df_tmp = chunk[["n_sessions", "n_bingewatching_sessions_3_or_more", "n_bingewatching_sessions_4_or_more",
                "n_bingewatching_sessions_2_to_6", "average_length_of_each_session", "percentage_sequential_sessions",
                "average_time_watching_session"]].describe()
        # df_tmp = df_tmp.loc["count"].apply(int)
        df_tmp = df_tmp.rename(index={"count": "number of (user, series)"})
        row_df = pd.DataFrame([[chunk.n_sessions.sum(), chunk.n_bingewatching_sessions_3_or_more.sum(),
                                chunk.n_bingewatching_sessions_4_or_more.sum(), chunk.n_bingewatching_sessions_2_to_6.sum(), "NaN", "NaN", "NaN"]],
                                index=["count"], columns=["n_sessions", "n_bingewatching_sessions_3_or_more", "n_bingewatching_sessions_4_or_more",
                                "n_bingewatching_sessions_2_to_6", "average_length_of_each_session", "percentage_sequential_sessions",
                                "average_time_watching_session"])
        df_tmp = pd.concat([row_df, df_tmp])

        df_describes.append(df_tmp)

    return df_describes


def _series_bingewatching_statistics_computation(df, series_bw_threshold=0.3):

    n_bingeworthy_series_ge3 = np.sum(np.array((df.groupby(
        "series_id").sum().n_bingewatching_sessions_3_or_more / df.groupby(
        "series_id").sum().n_sessions).values >= series_bw_threshold))
    n_bingeworthy_series_2_to_6 = np.sum(np.array((df.groupby(
                                   "series_id").sum().n_bingewatching_sessions_2_to_6 / df.groupby(
                                   "series_id").sum().n_sessions).values >= series_bw_threshold))
    n_bingeworthy_series_3_to_7 = np.sum(np.array((df.groupby(
        "series_id").sum().n_bingewatching_sessions_3_to_7 / df.groupby(
        "series_id").sum().n_sessions).values >= series_bw_threshold))
    df_stats = pd.DataFrame(data=[[df.series_id.nunique(), n_bingeworthy_series_ge3,
                                   n_bingeworthy_series_ge3 / df.series_id.nunique() * 100, n_bingeworthy_series_2_to_6,
                                   n_bingeworthy_series_2_to_6 / df.series_id.nunique() * 100, n_bingeworthy_series_3_to_7,
                                   n_bingeworthy_series_3_to_7 / df.series_id.nunique() * 100]],
                            columns=["n_series", "n_bingeworthy_series >= 3", "bingeworthy_series_percentage >= 3",
                                     "n_bingeworthy_series 2 to 6", "bingeworthy_series_percentage 2 to 6",
                                     "n_bingeworthy_series 3 to 7", "bingeworthy_series_percentage 3 to 7"])
    return df_stats


def series_bingewatching_statistics(df, percentage=50, session_threshold_hours=4, mode="interactions", quantile_division_on_whole_dataset=True, n_groups=3, series_bw_threshold=0.3, verbose=False):
    """
    Get statistics according to TiVO, Netflix and Trouleau et al. definitions from a series perspective. Returns the number
    of binge-worthy according to those definitions
    :param df: DataFrame of ContentWise impression dataset
    :param percentage: percentage of the binge-watching dataset
    :param session_threshold_hours: hour span for two episodes to be considered in the same session
    :param mode: "interactions | series_id", the groups will be selected with the same number of interactions or with the same
                number of series ids
    :param quantile_division_on_whole_dataset: True performs quantile division on the whole dataset, False performs it on
                                                the ones of the selected binge-watching DataFrame
    :param n_groups: number of the series length groups in which series will be divided
    :param series_bw_threshold: percentage of binge-watching sessions a series has to have in order to be considered a binge-worthy
    :param verbose:
    :return:
    """
    df_bw = get_timestamp_differential_dataframe(df, percentage=percentage, session_threshold_hours=session_threshold_hours)
    if quantile_division_on_whole_dataset:
        quantiles = get_series_length_groups(df, mode=mode, n_groups=n_groups, verbose=verbose)
    else:
        quantiles = get_series_length_groups(df_bw, mode=mode, n_groups=n_groups, verbose=verbose)
    df_list = divide_df_into_quantiles(df_bw, quantiles=quantiles)
    stats_dfs = []
    for chunk in df_list:
        stats_dfs.append(_series_bingewatching_statistics_computation(chunk, series_bw_threshold=series_bw_threshold))
    return stats_dfs


def user_bingewatching_statistics(df, percentage=50, session_threshold_hours=4, user_bw_threshold=0.3,):
    """
    Get statistics according to TiVO, Netflix and Trouleau et al. definitions from a user perspective. Returns the number
    of binge-watchers according to those definitions
    :param df: DataFrame of ContentWise impression dataset
    :param percentage: percentage of the binge-watching dataset
    :param session_threshold_hours: hour span for two episodes to be considered in the same session
    :param user_bw_threshold: percentage of binge-watching sessions a user has to have in order to be considered a binge-watcher
    :return: number of binge-watchers according to state of the art definitions
    """
    df_bw = get_timestamp_differential_dataframe(df, percentage=percentage, session_threshold_hours=session_threshold_hours)
    n_bingewatchers_ge3 = np.sum(np.array((df_bw.groupby("user_id").sum().n_bingewatching_sessions_3_or_more / df_bw.groupby(
        "user_id").sum().n_sessions).values >= user_bw_threshold))
    n_bingewatchers_2_to_6 = np.sum(np.array((df_bw.groupby("user_id").sum().n_bingewatching_sessions_2_to_6 / df_bw.groupby(
                              "user_id").sum().n_sessions).values >= user_bw_threshold))
    n_bingewatchers_3_to_7 = np.sum(np.array((df_bw.groupby("user_id").sum().n_bingewatching_sessions_3_to_7 / df_bw.groupby(
        "user_id").sum().n_sessions).values >= user_bw_threshold))

    df_stats = pd.DataFrame(data=[[df_bw.user_id.nunique(), n_bingewatchers_ge3, n_bingewatchers_ge3 / df_bw.user_id.nunique() * 100, n_bingewatchers_2_to_6,
                                   n_bingewatchers_2_to_6 / df_bw.user_id.nunique() * 100, n_bingewatchers_3_to_7,
                                   n_bingewatchers_3_to_7 / df_bw.user_id.nunique() * 100]],
                            columns=["n_users", "n_bingewatchers >= 3", "bingewatchers_percentage >= 3",
                                     "n_bingewatchers 2 to 6", "bingewatchers_percentage 2 to 6",
                                     "n_bingewatchers 3 to 7", "bingewatchers_percentage 3 to 7"])
    return df_stats


def bingewatching_urm_computation(dataset, shape, percentage=50, session_threshold_hours=4):
    df_bw = get_timestamp_differential_dataframe(dataset, percentage=percentage, session_threshold_hours=session_threshold_hours)
    urm_bw = np.zeros(shape=shape)
    for _, row in df_bw.iterrows():
        if row["n_bingewatching_sessions_3_to_7"] / row["n_sessions"] >= percentage:
            urm_bw[row["user_id"]][row["series_id"]] = 1
    return csr_matrix(urm_bw)

"""
-----------------------------------------------
|                PLOTTING                     |                                                              
-----------------------------------------------
"""


def plot_vision_factor_count(df, kind="line"):
    """
    Plots the vision factor with a line

    :param df: DataFrame of the ContentWise Impression dataset
    :param kind: type of plot, default line, could also be bar
    """
    vision_factor_df = get_vision_factor_count(df)
    vision_factor_df.sort_values("vision_factor").plot(kind=kind, x='vision_factor', y='y_axis', color='red',
                                                       logy=True)
    plt.show()


def plot_n_interactions_vision_factor_greater_than(df, threshold):
    """
    Pie graph with a count of all the interactions with vision factor greater than
    :param df: DataFrame of the ContentWise Impression dataset
    :param threshold: vision factor threshold
    :return: pie plot
    """
    # Variables to plot
    view_count = len(df[(df["vision_factor"] >= threshold)])
    total_count = len(df)

    # Actual plotting
    labels = f'Vision factor < {threshold}', 'Vision factor >= {threshold}'
    sizes = [total_count - view_count, view_count]
    explode = (0, 0.1)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


def plot_probability_matrix_heatmap(p_mat, size=(500, 500), norm=False):
    """
    Plot a heatmap of the Probability matrix
    :param p_mat: probability matrix
    :param size: size of the figure
    :param norm: True if you want normalized colorbar
    :return:
    """
    log_mapper = LogColorMapper(palette=Viridis6, low=0, high=1)

    hm = hv.HeatMap((np.arange(p_mat.toarray().shape[0]), np.arange(p_mat.toarray().shape[0]), p_mat.toarray()))
    hm.opts(opts.HeatMap(tools=['hover'], colorbar=True, width=size[0], height=size[1], toolbar='above', clim=(0, 1),
                         xlabel="Watched as", ylabel="Episode number", invert_yaxis=True, cmap="YlOrRd"))


def plot_count_matrix_heatmap(c_mat, size=(500, 500), norm=False):
    """
    Plot a heatmap of the Probability matrix
    :param c_mat: probability matrix
    :param size: size of the figure
    :param norm: True if you want normalized colorbar
    :return:
    """
    hm = hv.HeatMap((np.arange(c_mat.shape[0]), np.arange(c_mat.shape[0]), c_mat))
    hm.opts(opts.HeatMap(tools=['hover'], colorbar=True, width=size[0], height=size[1], toolbar='above', clim=(0, 1),
                         xlabel="Watched as", ylabel="Episode number", invert_yaxis=True, cmap="YlOrRd"))


def plot_probability_matrix_heatmap_sns(p_mat, size=(30, 26), norm=False, font_scale=3):
    """
    Plot a heatmap of the Probability matrix
    :param p_mat: probability matrix
    :param size: size of the figure
    :param norm: True if you want normalized colorbar
    :return:
    """
    sns.set(font_scale=font_scale)
    fig, ax = plt.subplots(figsize=size)
    if norm:
        ax = sns.heatmap(p_mat.toarray(), cbar=True, norm=LogNorm(0.01, p_mat.toarray().max()))
    else:
        ax = sns.heatmap(p_mat.toarray(), cbar=True)


def plot_parallel_signal(df, user_id):
    """
    Plot user parallel signal
    :param df: DataFrame of ContentWise impression dataset
    :param user_id: id of the user you want to print the signal function
    :return:
    """
    df_ts_differential = get_timestamp_differential_dataframe(df)
    parallel_signal = get_parallel_series_for_user(df, user_id=user_id)
    data = pd.DataFrame(parallel_signal, columns=["data"],
                        index=df_ts_differential[df_ts_differential["user_id"] == 2744].sort_values(
                            "utc_ts_milliseconds").utc_ts_milliseconds.values)
    data.index.name = "ts"
    data = data.reset_index()
    (hv.Curve(parallel_signal) * hv.Points(parallel_signal)).opts(opts.Points(color='red'))


def plot_box_hv(df, label="label"):
    """
    Plot holovies BoxWhisker of a DataFram or a series
    :param df: DataFrame or Series
    :param label: label to be put
    :return:
    """
    if df.dtype == "<m8[ns]":
        data = df.dt.days + (df.dt.seconds / (60 * 60 * 24))
    else:
        data = df

    boxwhisker = hv.BoxWhisker(data, label=label)
    boxwhisker.opts(show_legend=False, width=600, cmap='Set1')
    return boxwhisker


def plot_barchar_series_bingewatching(df, mode="interactions", width=1800, height=1200):
    """
    Plot barchar of bingewatching DataFrame
    :param df: bingewatching DataFrame for a certain percentage
    :param mode: "interactions" | "n_series" | "users"
    :return: Barchar plot
    """
    assert mode == "interactions" or mode == "n_series" or mode == "users"
    if mode == "interactions":
        bar = df.groupby("series_length").n_interactions.sum().to_frame()
    elif mode == "users":
        bar = df.groupby("series_length").user_id.count().to_frame()
    else:
        bar = df.groupby("series_length").series_id.nunique().to_frame()
    bars = hv.Bars(bar)

    bars.opts(show_legend=False, width=width, height=height, cmap='Set1', ylabel=mode)
    return bars


def plot_box_quantiles(df, column, quantiles=[3, 8, 19, 31, 56], ylabel="y", width=1800, height=600, fontscale=1.3):
    """
    Plot a Box Plot for each series length quantiles of df[column]
    :param df: Bingewatching describe DataFrame
    :param column: column on which filtering will be done
    :param quantiles: list that contains the start interval of each quantiles. For example, [3, 8, 19, 31, 56] has
    (3,7), (8,18), (19, 30), (31, 55), (56, _) quantiles
    :return: Layout of boxplots
    """
    # Series in the cleaned dataset have a series length of at least 3
    if quantiles[0] < 3:
        quantiles[0] = 3
    # Convert column data in days, hh:mm:ss if column data is datetime
    if df[column].dtype == "<m8[ns]":
        df[column] = df[column].dt.days + (df[column].dt.seconds / (60 * 60 * 24))
    # Create DataFrame chunks based on array of series length, containing all series in one interval
    df_list = divide_df_into_quantiles(df, quantiles=quantiles)

    # For each DataFrame chunk
    first_frame = df_list[0]
    box = hv.BoxWhisker(first_frame[column], label=f"{column}, series length {quantiles[0]}-{quantiles[1] - 1}")
    for quartile_i in range(1, len(df_list[1:])):
        box *= hv.BoxWhisker(df_list[quartile_i][column],
                             label=f"{column}, series length {quantiles[quartile_i]}-{quantiles[quartile_i + 1] - 1}")
    box *= hv.BoxWhisker(df_list[len(quantiles) - 1][column],
                         label=f"{column}, series length >= {quantiles[len(quantiles) - 1]}")
    box.opts(width=width, height=height, toolbar="left", fontscale=fontscale, ylabel=ylabel)
    return box
