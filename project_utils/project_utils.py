import numpy as np
import pandas as pd
import math
import scipy
from scipy import stats
from scipy.stats.mstats import gmean, hmean
import random
import os
from os import listdir
from os.path import isfile, join, isdir
import pickle
import sys
import time
from contextlib import contextmanager
from importlib import reload
from shutil import copyfile, move
import re
from pathlib import Path
from shutil import copyfile, move
import gc
import glob
from multiprocessing import Pool
from functools import partial
import traceback
import json
import datetime
# from datetime import datetime, timedelta
import requests
import subprocess
from sklearn.preprocessing import MinMaxScaler
import itertools
import subprocess
from typing import List, Optional, Dict, Union
import ml_config
import matplotlib.pyplot as plt
import openai
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import fuzz
from collections import Counter
import seaborn as sns
from cf_matrix import make_confusion_matrix


def get_time_string(format='%Y%m%d%H%M'):
    """
    Generate a time string representation of the time of call of this function.
    :param format
    :return: a string that represent the time of the functional call.
    """
    now = datetime.datetime.now()
    now = str(now.strftime(format))
    return now


def get_datetxt_delta(datetxt, delta, format='%Y-%m-%d'):
    src_dobj = datetime.datetime.strptime(datetxt, format).date()
    dst_dobj = src_dobj + datetime.timedelta(delta)
    dst_datetxt = datetime.datetime.strftime(dst_dobj, format)
    return dst_datetxt




def load_data(pickle_file):
    """
    load pickle data from file
    :param pickle_file: path of pickle data
    :return: data stored in pickle file
    """
    load_file = open(pickle_file, 'rb')
    data = pickle.load(load_file)
    return data



def read_file_into_list(file):
    """
    Read the contents of a file into a list, with each line as a separate element.

    This function opens the specified file, reads it line by line, and stores each line
    (with leading and trailing whitespace removed) as an element in a list.

    Args:
        file (str): The path to the file to be read.

    Returns:
        list: A list containing the lines of the file, with whitespace stripped.

    Note:
        This function assumes the file is text-based and can be read in 'r' mode.
        It automatically closes the file after reading, using a context manager.
    """
    lines = []
    # Open the file and read line by line
    with open(file, 'r') as file:
        for line in file:
            # Append each line to the list, stripping the newline characters
            lines.append(line.strip())
    return lines



def pickle_data(path, data, protocol=-1, timestamp=False, create_folder=True, dateonly=True,verbose=True):
    """
    Pickle data to specified file
    :param path: full path of file where data will be pickled to
    :param data: data to be pickled
    :param protocol: pickle protocol, -1 indicate to use the latest protocol
    :return: None
    """
    file = path

    if create_folder:
        folder = os.path.dirname(file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            if verbose:
                print(f'Created folder: {folder}')

    if timestamp:
        base_file = os.path.splitext(file)[0]
        time_str = '_' + get_time_string()
        if dateonly:
            time_str = time_str[0:9]
        ext = os.path.splitext(os.path.basename(file))[1]
        file = base_file + time_str + ext

    if verbose:
        print('creating file %s' % file)

    save_file = open(file, 'wb')
    pickle.dump(data, save_file, protocol=protocol)
    save_file.close()
    return file





def glob_folder_filelist(path, file_type='', recursive=True):
    """
    utility function that walk through a given directory, and return list of files in the directory
    :param path: the path of the directory
    :param file_type: if not '', this function would only consider the file type specified by this parameter
    :param recursive: if True, perform directory walk-fhrough recursively
    :return absfile: a list containing absolute path of each file in the directory
    :return base_files: a list containing base name of each file in the directory
    """
    if path[-1] != '/':
        path = path +'/'
    abs_files = []
    base_files = []
    patrn = '**' if recursive else '*'
    glob_path = path + patrn
    matches = glob.glob(glob_path, recursive=recursive)
    for f in matches:
        if os.path.isfile(f):
            include = True
            if len(file_type)>0:
                ext = os.path.splitext(f)[1]
                if ext[1:] != file_type:
                    include = False
            if include:
                abs_files.append(f)
                base_files.append(os.path.basename(f))
    return abs_files, base_files


def get_filefolder_info_df(path, sizeunit='mb', file_type='', recursive=True):
    content = []
    alist, blist = glob_folder_filelist(path, file_type, recursive)
    for afile, bfile in zip(alist, blist):
        item_dict = {}
        stat = os.stat(afile)
        if sizeunit == 'kb':
            size = stat.st_size / 1024
        elif sizeunit == 'mb':
            size = stat.st_size / (1024 * 1024)
        elif sizeunit == 'gb':
            size = stat.st_size / (1024 * 1024 * 1024)
        else:
            sizeunit = 'byte'
            size = stat.st_size
        ctime = datetime.datetime.fromtimestamp(stat.st_ctime).strftime(ml_config.DATETIME_FORMAT1)
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime(ml_config.DATETIME_FORMAT1)

        item_dict['filename'] = bfile
        item_dict['ext'] = bfile.split('.')[1]
        item_dict[f'size_{sizeunit}'] = round(size, 2)
        item_dict['last_modified'] = mtime
        item_dict['created'] = ctime
        item_dict['fullpath'] = afile
        content.append(item_dict)
    res_df = pd.DataFrame.from_dict(content)
    return res_df





def list_intersection(left, right):
    """
    take two list as input, conver them into sets, calculate the intersection of the two sets, and return this as a list
    :param left: the first input list
    :param right: the second input list
    :return: the intersection set of elements for both input list, as a list
    """
    left_set = set(left)
    right_set = set(right)
    return list(left_set.intersection(right_set))


def list_union(left, right):
    """
    take two list as input, conver them into sets, calculate the union of the two sets, and return this as a list
    :param left: the first input list
    :param right: the second input list
    :return: the union set of elements for both input list, as a list
    """
    left_set = set(left)
    right_set = set(right)
    return list(left_set.union(right_set))


def list_difference(left, right):
    """
    take two list as input, conver them into sets, calculate the difference of the first set to the second set, and return this as a list
    :param left: the first input list
    :param right: the second input list
    :return: the result of difference set operation on elements for both input list, as a list
    """
    left_set = set(left)
    right_set = set(right)
    return list(left_set.difference(right_set))


def is_listelements_identical(left, right):
    equal_length = (len(left)==len(right))
    zero_diff = (len(list_difference(left,right))==0)
    return equal_length & zero_diff





def np_corr(a, b):
    """
    take two numpy arrays, and compute their correlation
    :param a: the first numpy array input
    :param b: the second numpy array input
    :return: the correlation between the two input arrays
    """
    return pd.Series(a).corr(pd.Series(b))




def get_rank(data):
    """
    convert the values of a list or array into ranked percentage values
    :param data: the input data in the form of a list or an array
    :return: the return ranked percentage values in numpy array
    """
    ranks = pd.Series(data).rank(pct=True).values
    return ranks



def plot_feature_corr(df, features, figsize=(10,10), vmin=-1.0):
    """
    plot the pair-wise correlation matrix for specified features in a dataframe
    :param df: the input dataframe
    :param features: the list of features for which correlation matrix will be plotted
    :param figsize: the size of the displayed figure
    :param vmin: the minimum value of the correlation to be included in the plotting
    :return: the pair-wise correlation values in the form of pandas dataframe, the figure will be plotted during the operation of this function.
    """
    val_corr = df[features].corr().fillna(0)
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(val_corr, vmin=vmin, square=True)
    return val_corr




def decision_to_prob(data):
    """
    convert output value of a sklearn classifier (i.e. ridge classifier) decision function into probability
    :param data: output value of decision function in the form of a numpy array
    :return: value of probability in the form of a numpy array
    """
    prob = np.exp(data) / np.sum(np.exp(data))
    return prob



def ks_2samp_selection(train_df, test_df, pval=0.1):
    """
    use scipy ks_2samp function to select features that are statistically similar between the input train and test dataframe.
    :param train_df: the input train dataframe
    :param test_df: the input test dataframe
    :param pval: the p value threshold use to decide which features to be selected. Only features with value higher than the specified p value will be selected
    :return train_df: the return train dataframe with selected features
    :return test_df: the return test dataframe with selected features
    """
    list_p_value = []
    for i in train_df.columns.tolist():
        list_p_value.append(ks_2samp(train_df[i], test_df[i])[1])
    Se = pd.Series(list_p_value, index=train_df.columns.tolist()).sort_values()
    list_discarded = list(Se[Se < pval].index)
    train_df = train_df.drop(columns=list_discarded)
    test_df = test_df.drop(columns=list_discarded)
    return train_df, test_df


def df_balance_sampling(df, class_feature, minor_class=1, sample_ratio=1):
    """
    :param df:
    :param class_feature:
    :param minor_class:
    :param sample_ratio:
    :return:
    """
    minor_df = df[df[class_feature] == minor_class]
    major_df = df[df[class_feature] == (1 - minor_class)].sample(sample_ratio * len(minor_df))

    res_df = minor_df.append(major_df)
    res_df = res_df.sample(len(res_df)).reset_index(drop=True)
    return res_df


def prob2acc(label, probs, p=0.5):
    """
    calculate accuracy score  for probability predictions  with given threshold, as part of the process, the input probability predictions will be converted into discrete binary predictions
    :param label: labels used to evaluate accuracy score
    :param probs: probability predictions for which accuracy score will be calculated
    :param p: the threshold to be used for convert probabilites into discrete binary values 0 and 1
    :return acc: the computed accuracy score
    :return preds: predictions in discrete binary value
    """

    preds = (probs >= p).astype(np.uint8)
    acc = accuracy_score(label, preds)
    return acc, preds



def np_pearson(t,p):
    vt = t - t.mean()
    vp = p - p.mean()
    top = np.sum(vt*vp)
    bottom = np.sqrt(np.sum(vt**2)) * np.sqrt(np.sum(vp**2))
    res = top/bottom
    return res



def df_get_features_with_str(df, ptrn):
    """
    extract list of feature names from a data frame that contain the specified regular expression pattern
    :param df: the input dataframe of which features name to be analysed
    :param ptrn: the specified regular expression pattern
    :return: list of feature names that contained the specified regular expression
    """
    return [col for col in df.columns.tolist() if len(re.findall(ptrn, col)) > 0]


def df_fillna_with_other(df, src_feature, dst_feature):
    """
    fill the NA values of a specified feature in a dataframe with values of another feature from the same row.
    :param df: the input dataframe
    :param src_feature: the specified feature of which NA value will be filled
    :param dst_feature: the feature of which values will be used
    :return: a dataframe with the specified feature's NA value being filled by values from the "dst_feature"
    """
    src_vals = df[src_feature].values
    dst_vals = df[dst_feature].values
    argwhere_nan = np.argwhere(np.isnan(dst_vals)).flatten()
    dst_vals[argwhere_nan] = src_vals[argwhere_nan]
    df[dst_feature] = dst_vals
    return df



def plot_prediction_prob(y_pred_prob):
    """
    plot probability prediction values using histrogram
    :param y_pred_prob: the probability prediction values to be plotted
    :return: None, the plot will be plotted during the operation of the function.
    """
    prob_series = pd.Series(data=y_pred_prob)
    prob_series.name = 'prediction probability'
    prob_series.plot(kind='hist', figsize=(15, 5), bins=50)
    plt.show()
    print(prob_series.describe())





def df_traintest_split(df, split_var, seed=None, train_ratio=0.75):
    """
    perform train test split on a specified feature on a given dataframe wwith specified train ratio. Unique value of the specified feature will only present on either the resulted train or the test dataframe
    :param df: the input dataframe to be split
    :param split_var: the feature to be used as unique value to perform the split
    :param seed: the random used to facilitate the train test split
    :param train_ratio: the ratio of data to be split into the resulted train dataframe.
    :return train_df: the resulted train dataframe after the split
    :return test_df: the resulted test dataframe after the split
    """
    sv_list = df[split_var].unique().tolist()
    train_length = int(len(sv_list) * train_ratio)
    train_siv_list = pd.Series(df[split_var].unique()).sample(train_length, random_state=seed)
    train_idx = df.loc[df[split_var].isin(train_siv_list)].index.values
    test_idx = df.iloc[df.index.difference(train_idx)].index.values
    train_df = df.loc[train_idx].copy().reset_index(drop=True)
    test_df = df.loc[test_idx].copy().reset_index(drop=True)
    return train_df, test_df



# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df, verbose=True, exceiptions=[]):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    np_input = False
    if isinstance(df, np.ndarray):
        np_input = True
        df = pd.DataFrame(data=df)

    start_mem = df.memory_usage().sum() / 1024 ** 2
    col_id = 0
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        if verbose: print('doing %d: %s' % (col_id, col))
        col_type = df[col].dtype
        try:
            if (col_type != object) & (col not in exceiptions):
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        #                         df[col] = df[col].astype(np.float16)
                        #                     elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        #             else:
        #                 df[col] = df[col].astype('category')
        #                 pass
        except:
            pass
        col_id += 1
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    if np_input:
        return df.values
    else:
        return df





def get_xgb_featimp(model):
    imp_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    imp_dict = {}
    try:
        bst = model.get_booster()
    except:
        bst = model
    feature_names = bst.feature_names
    for impt in imp_type:
        imp_dict[impt] = []
        scores = bst.get_score(importance_type=impt)
        for feature in feature_names:
            if feature in scores.keys():
                imp_dict[impt].append(scores[feature])
            else:
                imp_dict[impt].append(np.nan)
    imp_df = pd.DataFrame(index=bst.feature_names, data=imp_dict)
    return imp_df




def get_df_rankavg(df):
    idx = df.index
    cols = df.columns.tolist()
    rankavg_dict = {}
    for col in cols:
        rankavg_dict[col]=df[col].rank(pct=True).tolist()
    rankavg_df = pd.DataFrame(index=idx, columns=cols, data=rankavg_dict)
    res_values = rankavg_df.mean(axis=1).values
    return res_values
    # return rankavg_df.sort_values(by='rankavg', ascending=False)


def get_list_gmean(lists):
    out = np.zeros((len(lists[0]), len(lists)))
    for i in range(0, len(lists)):
        out[:,i] = lists[i]
    gmean_out = gmean(out, axis=1)
    return gmean_out







def generate_nwise_combination(items, n=2):
    return list(itertools.combinations(items, n))


def pairwise_feature_generation(df, feature_list, operator='addition', verbose=True):
    feats_pair = generate_nwise_combination(feature_list, 2)
    result_df = pd.DataFrame()
    for pair in feats_pair:
        if verbose:
            print('generating %s of %s and %s' % (operator, pair[0], pair[1]))
        if operator == 'addition':
            feat_name = pair[0] + '_add_' + pair[1]
            result_df[feat_name] = df[pair[0]] + df[pair[1]]
        elif operator == 'multiplication':
            feat_name = pair[0] + '_mulp_' + pair[1]
            result_df[feat_name] = df[pair[0]] * df[pair[1]]
        elif operator == 'division':
            feat_name = pair[0] + '_div_' + pair[1]
            result_df[feat_name] = df[pair[0]] / df[pair[1]]
    return result_df





def try_divide(x, y, val=0.0):
    """
    try to perform division between two number, and return a default value if division by zero is detected
    :param x: the number to be used as dividend
    :param y: the number to be used as divisor
    :param val: the default output value
    :return: the output value, the default value of val will be returned if division by zero is detected
    """
    if y != 0.0:
        val = float(x) / y
    return val



def groupby_agg_execution(agg_recipies, df, verbose=True):
    result_dfs = dict()
    for groupby_cols, features, aggs in agg_recipies:
        group_object = df.groupby(groupby_cols)
        groupby_key = '_'.join(groupby_cols)
        if groupby_key not in list(result_dfs.keys()):
            result_dfs[groupby_key] = pd.DataFrame()
        for feature in features:
            rename_col = feature
            for agg in aggs:
                if isinstance(agg, dict):
                    agg_name = list(agg.keys())[0]
                    agg_func = agg[agg_name]
                else:
                    agg_name = agg
                    agg_func = agg
                if agg_name == 'count':
                    groupby_aggregate_name = '{}_{}'.format(groupby_key, agg_name)
                else:
                    groupby_aggregate_name = '{}_{}_{}'.format(groupby_key, feature, agg_name)
                verbose and print(f'generating statistic {groupby_aggregate_name}')
                groupby_res_df = group_object[feature].agg(agg_func).reset_index(drop=False)
                groupby_res_df = groupby_res_df.rename(columns={rename_col: groupby_aggregate_name})
                if len(result_dfs[groupby_key]) == 0:
                    result_dfs[groupby_key] = groupby_res_df
                else:
                    result_dfs[groupby_key][groupby_aggregate_name] = groupby_res_df[groupby_aggregate_name]
    return result_dfs





