import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
from project_tools import project_utils, project_class, torch_utils
import shap
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from typing import Optional
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from scipy.stats.mstats import gmean
from sklearn.linear_model import RidgeClassifier, Ridge, LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import gc
import time

# add groupby aggregator

class NumeraiLGB(BaseEstimator, TransformerMixin):
    """
    A wrapper class for the LightGBM package, to provide better customised support on training for different
    objectives, different tasks and model explanation
    """

    def __init__(self, params):
        """
        Constructor for the class, takes in the parameters for LightGBM, and initialise key attributes.
        :param params: parameters to be used by LightGBM model
        """
        classification_objectives = ['binary', 'multiclass']
        regression_objectives = ['regression', ' hubber', 'fair', 'xentropy']
        ranking_objectives = ['lambdarank', 'rank_xendcg']
        self.params = params
        if params['objective'] in classification_objectives:
            self.application = 'classification'
            self.clf = LGBMClassifier(**params)
        elif params['objective'] in ranking_objectives:
            self.application = 'ranking'
            self.clf = LGBMRanker(**params)
        else:
            self.application = 'regression'
            self.clf = LGBMRegressor(**params)
        if 'verbose_eval' in params.keys():
            self.verbose_eval = params['verbose_eval']
        else:
            self.verbose_eval = False
        if 'early_stopping' in params.keys():
            self.early_stopping = params['early_stopping']
        else:
            self.early_stopping = None
        self.impt_df = None
        self.impt_plot = None

    def fit(self, x, y, group=None, eval_set=None, num_boost_rounds=1000):
        lgb_train = lgb.Dataset(x, y, group=group, feature_name="auto")
        lgb_eval = []
        if eval_set is not None:
            for i in eval_set:
                if group is None:
                    lgb_eval_i = lgb.Dataset(i[0], i[1], reference=lgb_train)
                else:
                    lgb_eval_i = lgb.Dataset(i[0], i[1], group=i[2], reference=lgb_train)
                lgb_eval.append(lgb_eval_i)
        else:
            lgb_eval = None
        self.clf = lgb.train(self.params, lgb_train, valid_sets=lgb_eval, num_boost_round=num_boost_rounds,
                             verbose_eval=self.verbose_eval)
        pass

    def predict(self, x):
        # lgb_test = lgb.Dataset(x)
        y_preds = self.clf.predict(x)
        if self.application == 'ranking':
            y_preds = pd.Series(y_preds).rank(pct=True).values
        return y_preds

    def skl_fit(self, x, y, eval_set=None, early_stopping_rounds=None,
                feature_name='auto', categorical_feature='auto'):
        """
        model training method, to perform model fitting with the provided training data and label
        :param x:  the training data
        :param y:  the label
        :param eval_set:  if not None, evaluation set used for early stopping in the form for a tuple of (data, label)
        :param early_stopping_rounds:  round of training before early stopping
        :param feature_name:  name of the features to be used in the training, when "auto" the name will be automatically determined
        :param categorical_feature: a list of categorical features
        :return: None - the LightGBM model created as an attribute of the class instance will be fitted
        """
        self.clf.fit(x, y, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds,
                     verbose=self.verbose_eval, feature_name=feature_name, categorical_feature=categorical_feature)
        pass

    def skl_predict(self, x):
        """
        model prediction method, to perform prediction with the provided test data
        :param x: the test data for prediction making
        :return: the prediction values in numpy array format
        """
        if self.application == 'classification':
            y = self.clf.predict_proba(x)
        elif self.application == 'regression':
            y = self.clf.predict(x)
        else:
            print('the application parameter is neither classification nor regression')
            y = None
        return y

    def explain(self):
        """
        method to generate data for model explanation such as feature importance. The only implementation so far is the generate the feature importance in dataframe format. Future implementation could include generating Shap value data
        :param option:  indicate what type of model explanation data to generate, only feature importance has been implemented so far.
        :return: the feature importance for each feature in pandas dataframe format
        """
        self.impt_df = pd.DataFrame()
        self.impt_df['feature'] = self.clf.feature_name()
        self.impt_df['split_importance'] = self.clf.feature_importance(importance_type='split')
        self.impt_df['gain_importance'] = self.clf.feature_importance(importance_type='gain')
        self.impt_df['rankavg_importance'] = (self.impt_df['split_importance'].rank(pct=True, ascending=True) +
                                              self.impt_df['gain_importance'].rank(pct=True, ascending=True)) / 2
        self.impt_df = self.impt_df.sort_values(by='rankavg_importance', ascending=False).reset_index(drop=True)
        return self.impt_df


class NumeraiValidator():
    """
    A class for performing different validation scheme with provided training data, once validation methods are performed, the model fitted during the validation process are captured and can be used to perform inference.
    """

    def __init__(self, data, label, split_var, features=[], era_feat='era'):
        """
        Constructor for the class, allow specification of key attributes such as the dataset to be used, the feature to be used as label, and amount of data to be used for train test split.
        :param data: the dataset to be used to perform validation with
        :param label: the name of feature in the dataset to be used as label
        :param split_var: the name of the feature to be used to perform unique data split, such that data point with the same feature value would be allocated to the same split
        :param features: features to be used during model fitting in the validation process
        :param train_ratio: the ratio of the dataset to be used in the validation process, only used when the validation method is "traintest split", as specified by the perform_validation method
        """
        self.data = data
        self.label = label
        self.split_var = split_var
        self.era_feat = era_feat
        # self.train_ratio = train_ratio
        self.eval_results = {}
        self.overall_eval_results = {}
        self.predictions = []
        self.fold_index = []
        self.models = {}
        self.imp_df = None
        self.features = features
        self.method = None
        self.task = None
        self.prob_output = False  # not used for now
        self.algo = ''
        self.params = None
        self.ts_era = []
        self.ts_ytest = []
        pass

    def create_kfold_split(self, iteration=1, n_folds=5, shuffle=True, seed=None):
        """
        class method to create fold index for repeated k-fold cross validation
        :param iteration:  number of iterations for the k-fold cross validation operation to be performed
        :param n_folds: number of folds for the data to be split into in each iteration of k-fold
        :param shuffle: indicate if the order of data points will be shuffled within each fold split
        :param seed:  the random seed to be used to generated the fold splits
        """
        split_var_list = self.data[self.split_var].unique()  # .tolist()
        split_seed = seed
        for i in range(iteration):
            split_var_kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=split_seed)
            kf_idx = []
            for train_splitvar_idx, test_splitvar_idx in split_var_kf.split(split_var_list):
                train_idx = self.data[self.data[self.split_var].isin(split_var_list[train_splitvar_idx])].index.values
                test_idx = self.data[self.data[self.split_var].isin(split_var_list[test_splitvar_idx])].index.values
                kf_idx.append([train_idx, test_idx])
            self.fold_index.append(kf_idx)
            split_seed += 1

    def create_timesplit(self, timesplit_params, split_verbose=False):
        split_var_list = self.data[self.split_var].unique()
        timesplitter = TimeSeriesSplit(**timesplit_params)
        ts_idx = []
        fold_id = 0
        for train_splitvar_idx, test_splitvar_idx in timesplitter.split(split_var_list):
            fold_id += 1
            splitvar_train, splitvar_test = split_var_list[train_splitvar_idx], split_var_list[test_splitvar_idx]
            verbose_msg = f'Fold ID {fold_id}, train idx len {len(splitvar_train)}: {splitvar_train[0:10]} - test idx len {len(splitvar_test)}: {splitvar_test[0:10]}'
            split_verbose and print(verbose_msg)
            train_idx = self.data[self.data[self.split_var].isin(split_var_list[train_splitvar_idx])].index.values
            test_idx = self.data[self.data[self.split_var].isin(split_var_list[test_splitvar_idx])].index.values
            ts_idx.append([train_idx, test_idx])
            self.fold_index.append(ts_idx)
            pass

    def train_test_prediction(self, train_idx, test_idx, algo, params, iter_idx, kf_idx, additional_data=None,
                              additional_label=None):
        """
        class method to perform fitting on train data, and then prediction on test data. train and test data are specified by in the input indexes train_idx and test_idx that make reference to assoiated data points in the dataset associated to the class instance by the self.data attribuite.

        This method is strictly used by the validation operations of the class, therefore both train and test data used in this method come with label information that can be used to generate evaluation results.

        :param train_idx:  train data index - the indexes for data used for model fitting
        :param test_idx:  test data index - the indexes for data used for test prediction
        :param algo:  the type of algorithm to be used for model fitting
        :param params:  the parameters used by the selected algorithm
        :param additional_data:  if not None, addtitional data used to be appended to the train data for model fitting
        :param additional_label: the name of of feature in the additional dataset for model fitting
        :return model: the model fitted with the train data
        :return y_pred: prediction of the test data identified by test_idx
        :return result_dict: a dictionary containing evaluation results of the test predictions
        """
        x_train = self.data.loc[train_idx][self.features]
        x_test = self.data.loc[test_idx][self.features]
        y_train = self.data.loc[train_idx][self.label].values
        y_test = self.data.loc[test_idx][self.label].values
        torch_train = self.data.loc[train_idx]  # .reset_index(drop=True)
        torch_test = self.data.loc[test_idx]  # .reset_index(drop=True)
        test_era = self.data.loc[test_idx][self.era_feat].tolist()
        x_train_group = None

        if self.task == 'ranking':
            x_train_cdf = self.data.loc[train_idx].groupby(self.split_var).agg(['count'])
            x_train_group = x_train_cdf[x_train_cdf.columns[0]].values
            x_test_cdf = self.data.loc[test_idx].groupby(self.split_var).agg(['count'])
            x_test_group = x_test_cdf[x_test_cdf.columns[0]].values

        if additional_data is not None:
            x_train = x_train.append(additional_data)[self.features].reset_index(drop=True)
            y_train = np.hstack([y_train, additional_data[additional_label].values])

        if algo == 'nlgb':
            model = NumeraiLGB(params)
            if model.params['early_stopping'] is not None:
                # eval_set = [(x_train, y_train), (x_test, y_test)]
                if self.task == 'ranking':
                    eval_set = [(x_test, y_test, x_test_group)]
                else:
                    eval_set = [(x_test, y_test)]
            else:
                eval_set = None
            model.fit(x_train, y_train, group=x_train_group, eval_set=eval_set, num_boost_rounds=self.num_rounds)
            if self.prob_output:
                y_pred = model.predict(x_test)  # [:, 1]
            else:
                y_pred = model.predict(x_test)
        elif algo == 'ridge_reg':
            model = Ridge(**params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
        elif algo == 'torch':
            # for torch model, only return model file name
            params['prefix'] = f'iteration{iter_idx}kf{kf_idx}'
            params['postfix'] = project_utils.get_time_string()
            torchmodel = torch_utils.TorchModel(**params)
            torchmodel.fit(torch_train, torch_test, features=self.features, y_label=self.label)
            model = torchmodel.fitter.best_weight_file
            y_pred = torchmodel.predict(torch_test, features=self.features)
            del (torchmodel)
            gc.collect()
        else:
            # model.fit(x_train, self.train_label)
            print('not yet implemented')
            return None, None, None
        result_dict = project_utils.numerai_erascore(y_test, y_pred, test_era)
        return model, y_pred, result_dict

    def oof_prediction(self, iter_idx, algo, params, additional_data=None, additional_label=None, oof_verbose=False):
        """
        class method to perform out-of-fold prediction for one specific iteration of k-fold cross validation
        :param iter_idx: index id identifying specific iteration of the overall validation scheme
        :param algo:  the type of algorithm to be run in the cross validation process
        :param params:  the parameters to be used by the selected algorithm
        :param additional_data:  if not None, addtitional data used to be appended to the train data for model fitting
        :param additional_label: the name of of feature in the additional dataset for model fitting
        :return kf_models: the list of models fitted with the corresponding fold splits of train data
        :return y_pred: out of fold prediction of the dataset specified by the self.data attribute
        :return result_dict: a dictionary containing evaluation results of the out-of-fold test predictions
        """
        kf_idx = self.fold_index[iter_idx]
        kf_models = []
        y_pred = np.zeros(len(self.data))
        y_pred[:] = np.nan
        data_era = self.data[self.era_feat].tolist()
        y_data = self.data[self.label]
        for k in range(len(kf_idx)):
            train_idx = kf_idx[k][0]
            test_idx = kf_idx[k][1]
            model, fold_y_pred, fold_result_dict = self.train_test_prediction(train_idx, test_idx, algo, params,
                                                                              iter_idx, k,
                                                                              additional_data=additional_data,
                                                                              additional_label=additional_label)
            kf_models.append(model)
            y_pred[test_idx] = fold_y_pred
            oof_verbose and print('evaluation metrics for fold %d: %s' % (k, str(fold_result_dict)))
        result_dict = project_utils.numerai_erascore(y_data, y_pred, data_era)
        return kf_models, y_pred, result_dict

    def tsfold_prediction(self, algo, params, additional_data=None, additional_label=None, oof_verbose=False):
        ts_idx = self.fold_index
        ts_models = []
        y_tstest = []
        y_tspred = []
        y_tsera = []
        for k in range(len(ts_idx)):
            train_idx = ts_idx[0][k][0]
            test_idx = ts_idx[0][k][1]
            fold_era = self.data.loc[test_idx][self.era_feat].tolist()
            fold_ytest = self.data.loc[test_idx][self.label].tolist()
            model, fold_ypred, fold_result_dict = self.train_test_prediction(train_idx, test_idx, algo, params, 0, k,
                                                                             additional_data=additional_data,
                                                                             additional_label=additional_label)
            ts_models.append(model)
            y_tstest += fold_ypred.tolist()
            y_tspred += fold_ytest
            y_tsera += fold_era
            oof_verbose and print('evaluation metrics for fold %d: %s' % (k, str(fold_result_dict)))
        self.ts_era = y_tsera
        self.ts_ytest = y_tstest
        result_dict = project_utils.numerai_erascore(y_tstest, y_tspred, y_tsera)
        return ts_models, np.array(y_tspred), result_dict

    def calculate_feature_importance(self):
        """
        class method to calculate feature importance by aggretating feature importance from individual models in the overall validation scheme - only applicable to GBM based algos
        """
        self.imp_df = pd.DataFrame()
        self.imp_df["feature"] = list(self.features)
        self.imp_df['split_importance'] = 0
        self.imp_df['gain_importance'] = 0
        self.imp_df = self.imp_df.sort_values(by='feature', ascending=True).reset_index(drop=True)
        model_list = []
        if self.method == 'traintest_split':
            model_list = [model for key, model in self.models.items()]
        elif self.method == 'kfold':
            for iter_key in self.models.keys():
                model_list += self.models[iter_key]
            # for key, kf_models in self.models.items():
            #     model_list += kf_models

        # iterating through individual models of the model list, and aggregate feature importance score.
        for model in model_list:
            model_imp_df = model.explain().sort_values(by='feature', ascending=True).reset_index(drop=True)
            self.imp_df['split_importance'] += model_imp_df['split_importance']
            self.imp_df['gain_importance'] += model_imp_df['gain_importance']
        self.imp_df['rankavg_importance'] = (self.imp_df['split_importance'].rank(pct=True, ascending=True) +
                                             self.imp_df['gain_importance'].rank(pct=True, ascending=True)) / 2
        # create final aggretation score by creating range averaging bteween gain and split importance

        self.imp_df['split_importance'] /= len(model_list)
        self.imp_df['gain_importance'] /= len(model_list)
        self.imp_df = self.imp_df.sort_values(by='rankavg_importance', ascending=False).reset_index(drop=True)

    def perform_validation(self, params, algo='nlgb', method='kfold', additional_data=None, additional_label=None,
                           n_folds=5, iterations=1, shuffle=True, seed=None, iteration_verbose=2, oof_verbose=True,
                           num_rounds=1000, timesplit_params=None, split_verbose=False):
        """
        class method to define and perform the overall validation scheme.
        :param params: parameters to be used by the selected algorithm
        :param algo: the algorithm to be used during the validation operation
        :param method: the validation method - the options of traintest_split and kfold are supported
        :param additional_data:  if not None, addtitional data used to be appended to the train data for model fitting
        :param additional_label: the name of of feature in the additional dataset for model fitting
        :param n_folds: for kfold validation, number of fold splits to be create for each iteration
        :param iterations: number of iterations for the specified validation method to be performed
        :param shuffle: only used by kfold method, to indicate if the order of data points will be shuffled within each fold split
        :param seed: the random seed to be used to generated train test split or fold splits, depends on validation method
        :param kf_verbose: indicate if verbose message of validation operation will be printed on screen
        """
        self.algo = algo
        self.params = params
        if algo == 'nlgb':
            self.num_rounds = num_rounds
            if params['objective'] == 'binary':
                self.task = 'classification'
                self.prob_output = True
            elif params['objective'] in ['rmse', 'huber', 'fair']:
                self.task = 'regression'
                self.prob_output = False
            elif params['objective'] in ['lambdarank', 'rank_xendcg']:
                self.task = 'ranking'
                self.prob_output = False

        elif algo == 'ridge_reg':
            self.task = 'regression'
            self.prob_output = False
        elif algo == 'logit':
            self.task = 'classification'
            self.prob_output = True
        elif algo == 'torch':
            if isinstance(params['config'].criterion, torch_utils.RMSELoss):
                self.task = 'regression'
                self.prob_output = False
        else:
            print('algorithm %s has not been implemented yet' % algo)
            return None

        if method == 'kfold':
            self.create_kfold_split(iteration=iterations, n_folds=n_folds, shuffle=shuffle, seed=seed)
        elif method == 'timesplit':
            self.create_timesplit(timesplit_params=timesplit_params, split_verbose=split_verbose)
            iterations = 1
        else:
            print('only kfold method is implemented at the moment')
            return None

        self.method = method
        iter_eval_result = {}
        for i in range(iterations):
            if method == 'kfold':
                fold_models, y_pred, iter_eval_result = self.oof_prediction(i, algo, params,
                                                                            additional_data=additional_data,
                                                                            additional_label=additional_label,
                                                                            oof_verbose=oof_verbose)
            elif method == 'timesplit':
                fold_models, y_pred, iter_eval_result = self.tsfold_prediction(algo, params,
                                                                               additional_data=additional_data,
                                                                               additional_label=additional_label,
                                                                               oof_verbose=oof_verbose)
            self.predictions.append(y_pred)
            self.models['iteration %d' % i] = fold_models
            iteration_verbose > 1 and print('iteration %i  -' % i, end=' ')
            for key in iter_eval_result.keys():
                iteration_verbose > 1 and print('%s for prediction is %0.6f' % (key, iter_eval_result[key]), end='  ')
                if key not in self.eval_results.keys():
                    self.eval_results[key] = []
                self.eval_results[key].append(iter_eval_result[key])
            iteration_verbose > 1 and print('\r')  # print carriage return
        iteration_verbose > 0 and print('----- mean validation loss -----', end=' ')
        if algo == 'nlgb':
            self.calculate_feature_importance()
        for key in self.eval_results.keys():
            self.overall_eval_results[key] = np.mean(self.eval_results[key])
            iteration_verbose > 0 and print('%s is %0.6f' % (key, self.overall_eval_results[key]), end=' ')
        iteration_verbose > 0 and print('\r')

    def perform_inference(self, data, iteration=None, k=None, avg_method='gmean', torch_params=None):
        """
        class method to perform inference on new data that is not part of the dataset linked to self.data attribute
        :param data: the dataset for which inference is to be performed
        :param iteration: specify a specific iteration of models to be used, if None, all iterations are used
        :param k: specify a specific model within a iteration to be used, if None, all models within an iteration will be used
        :param avg_method: the method used to aggregate model prediction results, either 'gmean' for geometric mean, or 'amean' for arithmatic mean
        :return mean_prediction: the aggregated prediction results of the input dataset
        """
        model_list = []
        input_data = data[self.features]
        non_torch_model = True

        if self.method in ['kfold', 'timesplit']:
            if iteration is not None:
                if k is not None:
                    model_list.append(self.models['iteration %d' % iteration][k])
                else:
                    model_list += self.models['iteration %d' % iteration]
            else:
                for key, kf_models in self.models.items():
                    model_list += kf_models
        else:
            print('only kfold based validation and inference is supported.')
            return None
        print(f'perform inference using {len(model_list)} sub-models')
        predictions = np.zeros((len(data), len(model_list)))
        for i in range(len(model_list)):
            if 'algo' in self.__dict__.keys():
                if self.algo == 'torch':
                    non_torch_model = False

            if non_torch_model:  # (self.algo != 'torch'):
                if (self.task == 'classification') & (self.prob_output):
                    try:  # try lgbm interface first
                        model_pred = model_list[i].predict(input_data)[:, 1]
                    except:  # classic sklearn predict proba interface
                        model_pred = model_list[i].predict_proba(input_data)[:, 1]
                else:
                    model_pred = model_list[i].predict(input_data)
            else:
                # for torch based model, work out the filename of the best weight file
                print(f'loading model weigh from {model_list[i]}')
                prefix = model_list[i].split('/')[-1].split('_')[0]
                postfix = model_list[i].split('/')[-1].split('_')[-1].split('.')[0]
                params = self.params.copy()
                params['prefix'] = prefix
                params['postfix'] = postfix
                torchmodel = torch_utils.TorchModel(**params)
                model_pred = torchmodel.predict(input_data, self.features)
                del (torchmodel)
                gc.collect()
                # print('\n', model_pred.shape, model_pred[0:5])
            predictions[:, i] = model_pred
        if avg_method == 'gmean':
            mean_prediction = gmean(predictions, axis=1)
        elif avg_method == 'amean':
            mean_prediction = np.mean(predictions, axis=1)
        else:
            mean_prediction = None
        return mean_prediction  # , predictions

    def save(self, path, prefix='', slimsize=True):
        if len(prefix) > 0:
            self.prefix = prefix + '_'
        self.data = None
        if slimsize == True:
            self.predictions = None
            self.fold_index = None
        save_name = prefix + self.algo + '_' + project_utils.get_time_string() + '.pkl'
        model_path = os.path.join(path, save_name)
        project_utils.pickle_data(model_path, self)
        return model_path


class NumeraiNullHypoFeatureSelctor(NumeraiValidator):
    def __init__(self, data, label, split_var, features=[], era_feat='era', valid_data=None, holdout_data=None):
        super().__init__(data, label, split_var, features, era_feat)
        self.null_imp_df = None
        self.actual_imp_df = None
        self.corr_scores_df = None
        self.valid_data = valid_data
        self.holdout_data = holdout_data
        self.featsel_df = None
        self.featsel_results = {}
        self.validator = None

    def generate_null_importance(self, params, rounds=80, seed=42, num_boost_rounds=200, verbose=True):
        params['early_stopping'] = None
        x_train = self.data[self.features]
        x_train_group = None
        if params['objective'] in ['lambdarank', 'rank_xendcg']:
            x_train_cdf = self.data.groupby(self.split_var).agg(['count'])
            x_train_group = x_train_cdf[x_train_cdf.columns[0]].values

        start = time.time()
        dsp = ''
        self.null_imp_df = pd.DataFrame()
        for i in range(rounds):
            use_seed = seed + i
            y_train = self.data[self.label].sample(frac=1.0, random_state=use_seed).values
            model = NumeraiLGB(params)
            eval_set = None
            model.fit(x_train, y_train, group=x_train_group, eval_set=eval_set, num_boost_rounds=num_boost_rounds)
            spent = (time.time() - start) / 60
            for l in range(len(dsp)):
                verbose and print('\b', end='', flush=True)
            dsp = f'Done with {i + 1:3} of {rounds:3} (Spent {spent:4.1f} min)'
            model_imp_df = model.explain().sort_values(by='feature').reset_index(drop=True)
            model_imp_df['run'] = i + 1
            self.null_imp_df = pd.concat([self.null_imp_df, model_imp_df], axis=0).reset_index(drop=True)
            verbose and print(dsp, end='', flush=True)
        verbose and print()

    def generate_actual_importance(self, params, iterations=3, n_folds=3, seed=None, iteration_verbose=2,
                                   oof_verbose=None, num_boost_rounds=200):
        self.perform_validation(params=params, algo='nlgb', iterations=iterations, seed=seed, n_folds=n_folds,
                                oof_verbose=oof_verbose, iteration_verbose=iteration_verbose,
                                num_rounds=num_boost_rounds)
        self.actual_imp_df = self.imp_df.copy()

    def display_featimpt_distributions(self, feature, figsize=(20, 6)):
        if (self.null_imp_df is not None) and (self.actual_imp_df is not None):
            plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 3)
            # Plot Split importances
            ax = plt.subplot(gs[0, 0])
            a = ax.hist(self.null_imp_df.loc[self.null_imp_df['feature'] == feature, 'split_importance'].values,
                        label='Null importances')
            ax.vlines(x=self.actual_imp_df.loc[self.actual_imp_df['feature'] == feature, 'split_importance'].mean(),
                      ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
            ax.legend()
            ax.set_title('Split Importance of %s' % feature.upper(), fontweight='bold')
            plt.xlabel('Null Importance (split) Distribution for %s ' % feature.upper())

            # Plot Gain importances
            ax = plt.subplot(gs[0, 1])
            a = ax.hist(self.null_imp_df.loc[self.null_imp_df['feature'] == feature, 'gain_importance'].values,
                        label='Null importances')
            ax.vlines(x=self.actual_imp_df.loc[self.actual_imp_df['feature'] == feature, 'gain_importance'].mean(),
                      ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
            ax.legend()
            ax.set_title('Gain Importance of %s' % feature.upper(), fontweight='bold')
            plt.xlabel('Null Importance (gain) Distribution for %s ' % feature.upper())

            # Plot Rankavg importances
            ax = plt.subplot(gs[0, 2])
            a = ax.hist(self.null_imp_df.loc[self.null_imp_df['feature'] == feature, 'rankavg_importance'].values,
                        label='Null importances')
            ax.vlines(x=self.actual_imp_df.loc[self.actual_imp_df['feature'] == feature, 'rankavg_importance'].mean(),
                      ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
            ax.legend()
            ax.set_title('Rank Importance of %s' % feature.upper(), fontweight='bold')
            plt.xlabel('Null Importance (rank) Distribution for %s ' % feature.upper())
            return plt
        else:
            print('please generate null importance and actual importance first')
            return None

    def calculate_correlation_feature_score(self, plot=False, figsize=(20, 16), plot_top_n=70):
        correlation_scores = []
        if (self.null_imp_df is not None) and (self.actual_imp_df is not None):
            for _f in self.actual_imp_df['feature'].unique():
                f_null_imps = self.null_imp_df.loc[self.null_imp_df['feature'] == _f, 'gain_importance'].values
                f_act_imps = self.actual_imp_df.loc[self.actual_imp_df['feature'] == _f, 'gain_importance'].values
                gain_score = 100 * (f_null_imps < f_act_imps).sum() / f_null_imps.size
                f_null_imps = self.null_imp_df.loc[self.null_imp_df['feature'] == _f, 'split_importance'].values
                f_act_imps = self.actual_imp_df.loc[self.actual_imp_df['feature'] == _f, 'split_importance'].values
                split_score = 100 * (f_null_imps < f_act_imps).sum() / f_null_imps.size
                f_null_imps = self.null_imp_df.loc[self.null_imp_df['feature'] == _f, 'rankavg_importance'].values
                f_act_imps = self.actual_imp_df.loc[self.actual_imp_df['feature'] == _f, 'rankavg_importance'].values
                rank_score = 100 * (f_null_imps < f_act_imps).sum() / f_null_imps.size
                correlation_scores.append((_f, split_score, gain_score, rank_score))
            corr_scores_df = pd.DataFrame(correlation_scores,
                                          columns=['feature', 'split_score', 'gain_score', 'rank_score'])
            corr_scores_df = corr_scores_df.sort_values(by='rank_score', ascending=False).reset_index(drop=True)
            self.corr_scores_df = corr_scores_df
            if plot:
                fig = plt.figure(figsize=figsize)
                gs = gridspec.GridSpec(1, 3)
                # Plot Split importances
                ax = plt.subplot(gs[0, 0])
                sns.barplot(x='split_score', y='feature',
                            data=corr_scores_df.sort_values('split_score', ascending=False).iloc[0:plot_top_n], ax=ax)
                ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
                # Plot Gain importances
                ax = plt.subplot(gs[0, 1])
                sns.barplot(x='gain_score', y='feature',
                            data=corr_scores_df.sort_values('gain_score', ascending=False).iloc[0:plot_top_n], ax=ax)
                ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
                # Plot Rank importances
                ax = plt.subplot(gs[0, 2])
                sns.barplot(x='rank_score', y='feature',
                            data=corr_scores_df.sort_values('rank_score', ascending=False).iloc[0:plot_top_n], ax=ax)
                ax.set_title('Feature scores wrt rankavg importances', fontweight='bold', fontsize=14)
                plt.tight_layout()
                plt.suptitle("Features' split, gain & rank scores", fontweight='bold', fontsize=16)
                fig.subplots_adjust(top=0.93)
        else:
            print('please generate null importance and actual importance first')
            return None

    def score_feature_selection(self, thresholds, fns, expreds, val_info, params, iterations=3, n_folds=3, seed=None,
                                iteration_verbose=2, oof_verbose=None, num_boost_rounds=200, criteria='rank_score'):
        if criteria not in ['split_score', 'gain_score', 'rank_score']:
            print('invalid feature selection criteria')
            return None
        rank_corr_df = self.corr_scores_df.sort_values(by=criteria, ascending=False).reset_index(drop=True)
        last_feat_nums = 0
        featsel_evals = []
        for value in thresholds:
            featsel = rank_corr_df[rank_corr_df[criteria] >= value]['feature'].tolist()
            if len(featsel) != last_feat_nums:
                print(f'{len(featsel)} features are selected with {criteria} >= {value}')
                temp_validator = project_class.NumeraiValidator(data=self.data, label=self.label,
                                                                split_var=self.split_var, features=featsel)
                temp_validator.perform_validation(params=params, algo='nlgb', iterations=iterations, seed=seed,
                                                  n_folds=n_folds, oof_verbose=oof_verbose,
                                                  iteration_verbose=iteration_verbose, num_rounds=num_boost_rounds)
                val_pred = temp_validator.perform_inference(self.valid_data)
                for fn in fns:
                    model_name = f'{criteria[:-6]}{value}_fn{int(fn * 100)}'
                    print(f'calculating evaluation result for model {model_name}')
                    res = pd.DataFrame()
                    res['id'] = self.valid_data['id']
                    res['prediction'] = val_pred
                    if fn == 0:
                        res_eval = project_utils.single_v1v2validstat(model_name, res, val_info, expreds)
                    else:
                        res_fn = project_utils.fn_valid(res, self.valid_data, featsel, proportion=fn)
                        res_eval = project_utils.single_v1v2validstat(model_name, res_fn, val_info, expreds)
                    res_eval['features_num'] = len(featsel)
                    featsel_evals.append(res_eval)
                    print(res_eval)
                self.featsel_results[f'{criteria[:-6]}{value}'] = featsel
            else:
                print(f'same selection as the last threshold')
        if len(featsel_evals) > 0:
            featsel_df = pd.DataFrame.from_dict(featsel_evals)
            if self.featsel_df is None:
                self.featsel_df = featsel_df
            else:
                self.featsel_df = self.featsel_df.append(featsel_evals).reset_index(drop=True)
        return None

    def score_feature_selection_nd(self, thresholds, fns, expreds, val_info, params, iterations=3, n_folds=3, seed=None,
                                   iteration_verbose=2, oof_verbose=None, num_boost_rounds=200, criteria='rank_score',
                                   ts_params=None):
        if criteria not in ['split_score', 'gain_score', 'rank_score']:
            print('invalid feature selection criteria')
            return None
        rank_corr_df = self.corr_scores_df.sort_values(by=criteria, ascending=False).reset_index(drop=True)
        last_feat_nums = 0
        featsel_evals = []
        if ts_params is None:
            method = 'kfold'
            split_verbose = False
        else:
            method = 'timesplit'
            split_verbose = True
            iterations = 1
        for value in thresholds:
            featsel = rank_corr_df[rank_corr_df[criteria] >= value]['feature'].tolist()
            if len(featsel) != last_feat_nums:
                print(f'{len(featsel)} features are selected with {criteria} >= {value}')
                temp_validator = project_class.NumeraiValidator(data=self.data, label=self.label,
                                                                split_var=self.split_var, features=featsel)
                temp_validator.perform_validation(params=params, algo='nlgb', method=method, iterations=iterations,
                                                  seed=seed, n_folds=n_folds, oof_verbose=oof_verbose,
                                                  iteration_verbose=iteration_verbose, num_rounds=num_boost_rounds,
                                                  timesplit_params=ts_params, split_verbose=split_verbose)
                self.validator = temp_validator
                if ts_params is None:
                    local_eval = None
                else:
                    local_eval = temp_validator.eval_results
                val_pred = temp_validator.perform_inference(self.valid_data)
                for fn in fns:
                    model_name = f'{criteria[:-6]}{value}_fn{int(fn * 100)}'
                    print(f'calculating evaluation result for model {model_name}')
                    res = pd.DataFrame()
                    res['id'] = self.valid_data['id']
                    res['prediction'] = val_pred
                    if fn == 0:
                        res_eval = project_utils.newdata_validstat(model_name, res, val_info, expreds,
                                                                   local_eval=local_eval)
                    else:
                        res_fn = project_utils.fn_valid(res, self.valid_data, featsel, proportion=fn)
                        res_eval = project_utils.newdata_validstat(model_name, res_fn, val_info, expreds,
                                                                   local_eval=local_eval)
                    res_eval['features_num'] = len(featsel)
                    print(res_eval)
                    featsel_evals.append(res_eval)
                self.featsel_results[f'{criteria[:-6]}{value}'] = featsel
            else:
                print(f'same selection as the last threshold')
        if len(featsel_evals) > 0:
            featsel_df = pd.DataFrame.from_dict(featsel_evals)
            if self.featsel_df is None:
                self.featsel_df = featsel_df
            else:
                self.featsel_df = self.featsel_df.append(featsel_evals).reset_index(drop=True)
        return None

    def score_feature_selection_holdout(self, thresholds, fns, val_info, holdout_info, params, iterations=3,
                                        n_folds=3, seed=None, iteration_verbose=2, oof_verbose=None,
                                        num_boost_rounds=200, criteria='rank_score'):
        if criteria not in ['split_score', 'gain_score', 'rank_score']:
            print('invalid feature selection criteria')
            return None
        if self.holdout_data is None:
            print('need holdout data for this method')
            return None
        rank_corr_df = self.corr_scores_df.sort_values(by=criteria, ascending=False).reset_index(drop=True)
        last_feat_nums = 0
        featsel_evals = []
        method = 'kfold'
        split_verbose = False
        for value in thresholds:
            featsel = rank_corr_df[rank_corr_df[criteria] >= value]['feature'].tolist()
            if len(featsel) != last_feat_nums:
                print(f'{len(featsel)} features are selected with {criteria} >= {value}')
                temp_validator = project_class.NumeraiValidator(data=self.data, label=self.label,
                                                                split_var=self.split_var, features=featsel)
                temp_validator.perform_validation(params=params, algo='nlgb', method=method, iterations=iterations,
                                                  seed=seed, n_folds=n_folds, oof_verbose=oof_verbose,
                                                  iteration_verbose=iteration_verbose, num_rounds=num_boost_rounds,
                                                  split_verbose=split_verbose)
                self.validator = temp_validator
                holdout_pred = temp_validator.perform_inference(self.holdout_data)
                holdout_eval = project_utils.numerai_erascore(holdout_info['target'].values, holdout_pred,
                                                              holdout_info['era'].tolist())
                print(holdout_eval)
                val_pred = temp_validator.perform_inference(self.valid_data)
                for fn in fns:
                    model_name = f'{criteria[:-6]}{value}_fn{int(fn * 100)}'
                    print(f'calculating evaluation result for model {model_name}')
                    res = pd.DataFrame()
                    res['id'] = self.valid_data['id']
                    res['prediction'] = val_pred
                    if fn == 0:
                        res_eval = project_utils.newdata_validstat(model_name, res, val_info,
                                                                   local_eval=holdout_eval)
                    else:
                        res_fn = project_utils.fn_valid(res, self.valid_data, featsel, proportion=fn)
                        res_eval = project_utils.newdata_validstat(model_name, res_fn, val_info,
                                                                   local_eval=holdout_eval)
                    res_eval['features_num'] = len(featsel)
                    print(res_eval)
                    featsel_evals.append(res_eval)
                self.featsel_results[f'{criteria[:-6]}{value}'] = featsel
            else:
                print(f'same selection as the last threshold')
        if len(featsel_evals) > 0:
            featsel_df = pd.DataFrame.from_dict(featsel_evals)
            if self.featsel_df is None:
                self.featsel_df = featsel_df
            else:
                self.featsel_df = self.featsel_df.append(featsel_evals).reset_index(drop=True)
        return None


class NumeraiAdversarialValidator(NumeraiValidator):
    def __init__(self, data, label, split_var, features=[], era_feat='era'):
        super().__init__(data, label, split_var, features, era_feat)

    def train_test_prediction(self, train_idx, test_idx, algo, params, iter_idx, kf_idx, additional_data=None,
                              additional_label=None):
        """
        class method to perform fitting on train data, and then prediction on test data. train and test data are specified by in the input indexes train_idx and test_idx that make reference to assoiated data points in the dataset associated to the class instance by the self.data attribuite.

        This method is strictly used by the validation operations of the class, therefore both train and test data used in this method come with label information that can be used to generate evaluation results.

        :param train_idx:  train data index - the indexes for data used for model fitting
        :param test_idx:  test data index - the indexes for data used for test prediction
        :param algo:  the type of algorithm to be used for model fitting
        :param params:  the parameters used by the selected algorithm
        :param additional_data:  if not None, addtitional data used to be appended to the train data for model fitting
        :param additional_label: the name of of feature in the additional dataset for model fitting
        :return model: the model fitted with the train data
        :return y_pred: prediction of the test data identified by test_idx
        :return result_dict: a dictionary containing evaluation results of the test predictions
        """
        x_train = self.data.loc[train_idx][self.features]
        x_test = self.data.loc[test_idx][self.features]
        y_train = self.data.loc[train_idx][self.label].values
        y_test = self.data.loc[test_idx][self.label].values
        x_train_group = None
        result_dict = {}
        if algo == 'nlgb':
            model = NumeraiLGB(params)
            if model.params['early_stopping'] is not None:
                # eval_set = [(x_train, y_train), (x_test, y_test)]
                if self.task == 'ranking':
                    eval_set = [(x_test, y_test, x_test_group)]
                else:
                    eval_set = [(x_test, y_test)]
            else:
                eval_set = None
            model.fit(x_train, y_train, group=x_train_group, eval_set=eval_set, num_boost_round=self.num_rounds)
            y_pred = model.predict(x_test)  # [:, 1]
        else:
            # model.fit(x_train, self.train_label)
            print('not yet implemented')
            return None, None, None
        result_dict['auc'] = roc_auc_score(y_test, y_pred)
        return model, y_pred, result_dict

    def oof_prediction(self, iter_idx, algo, params, additional_data=None, additional_label=None, oof_verbose=False):
        """
        class method to perform out-of-fold prediction for one specific iteration of k-fold cross validation
        :param iter_idx: index id identifying specific iteration of the overall validation scheme
        :param algo:  the type of algorithm to be run in the cross validation process
        :param params:  the parameters to be used by the selected algorithm
        :param additional_data:  if not None, addtitional data used to be appended to the train data for model fitting
        :param additional_label: the name of of feature in the additional dataset for model fitting
        :return kf_models: the list of models fitted with the corresponding fold splits of train data
        :return y_pred: out of fold prediction of the dataset specified by the self.data attribute
        :return result_dict: a dictionary containing evaluation results of the out-of-fold test predictions
        """
        kf_idx = self.fold_index[iter_idx]
        kf_models = []
        y_pred = np.zeros(len(self.data))
        y_pred[:] = np.nan
        y_data = self.data[self.label].values
        result_dict = {}
        for k in range(len(kf_idx)):
            train_idx = kf_idx[k][0]
            test_idx = kf_idx[k][1]
            model, fold_y_pred, result_dict = self.train_test_prediction(train_idx, test_idx, algo, params, iter_idx, k,
                                                                         additional_data=additional_data,
                                                                         additional_label=additional_label)
            kf_models.append(model)
            y_pred[test_idx] = fold_y_pred
            oof_verbose and print('evaluation metrics for fold %d: %s' % (k, str(result_dict)))
        result_dict['auc'] = roc_auc_score(y_data, y_pred)
        return kf_models, y_pred, result_dict


class Data_Aggregrator(BaseEstimator, TransformerMixin):
    def __init__(self, handle_missing=False, droptimeindex=True):
        self.handle_missing = handle_missing
        self.droptimeindex = droptimeindex

    def fit(self, x):
        return self

    def transform(self, x):
        """
        x will be a list of dataframes to be aggregated
        """
        if len(x) > 1:
            data = pd.concat(x, axis=0).reset_index(drop=self.droptimeindex)
        else:
            data = x[0].reset_index(drop=self.droptimeindex)
        return data


class DataFrame_Preproc(BaseEstimator, TransformerMixin):
    'class to perform dataframe imputation while retain df format'

    def __init__(self, preproc_pipeline):
        self.preproc_pipeline = preproc_pipeline

    def fit(self, x):
        self.preproc_pipeline.fit(x.values)
        return self

    def transform(self, x):
        cols = x.columns.tolist()
        data = self.preproc_pipeline.transform(x.values)
        df = pd.DataFrame(columns=cols, data=data)
        return df


class TimeSeriesSplit(_BaseKFold):
    def __init__(self,
                 n_splits: Optional[int] = 5,
                 train_size: Optional[int] = None,
                 test_size: Optional[int] = None,
                 delay: int = 0,
                 force_step_size: Optional[int] = None):

        if n_splits and n_splits < 2:
            raise ValueError(f'Cannot have n_splits less than 5 (n_splits={n_splits})')
        super().__init__(n_splits, shuffle=False, random_state=None)

        self.train_size = train_size

        if test_size and test_size < 0:
            raise ValueError(f'Cannot have negative values of test_size (test_size={test_size})')
        self.test_size = test_size

        if delay < 0:
            raise ValueError(f'Cannot have negative values of delay (delay={delay})')
        self.delay = delay

        if force_step_size and force_step_size < 1:
            raise ValueError(f'Cannot have zero or negative values of force_step_size '
                             f'(force_step_size={force_step_size}).')

        self.force_step_size = force_step_size

    def split(self, X, y=None, groups=None, verbose=True):
        """Generate indices to split data into training and test set.

        Parameters:
            X : array-like, shape (n_samples, n_features)
                Training data, where n_samples is the number of samples  and n_features is the number of features.

            y : array-like, shape (n_samples,)
                Always ignored, exists for compatibility.

            groups : array-like, with shape (n_samples,), optional
                Always ignored, exists for compatibility.

        Yields:
            train : ndarray
                The training set indices for that split.

            test : ndarray
                The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)  # pylint: disable=unbalanced-tuple-unpacking
        n_samples = _num_samples(X)

        n_splits = self.n_splits
        n_folds = n_splits + 1
        delay = self.delay

        if n_folds > n_samples:
            raise ValueError(f'Cannot have number of folds={n_folds} greater than the number of samples: {n_samples}.')

        indices = np.arange(n_samples)
        split_size = n_samples // n_folds

        train_size = self.train_size or split_size * self.n_splits
        test_size = self.test_size or n_samples // n_folds
        full_test = test_size + delay

        if full_test + n_splits > n_samples:
            raise ValueError(f'test_size\\({test_size}\\) + delay\\({delay}\\) = {test_size + delay} + '
                             f'n_splits={n_splits} \n'
                             f' greater than the number of samples: {n_samples}. Cannot create fold logic.')

        # Generate logic for splits.
        # Overwrite fold test_starts ranges if force_step_size is specified.

        if self.force_step_size:
            step_size = self.force_step_size
            final_fold_start = n_samples - (train_size + full_test)
            range_start = (final_fold_start % step_size) + train_size
            test_starts = range(range_start, n_samples, step_size)

        else:
            if not self.train_size:
                step_size = split_size
                range_start = (split_size - full_test) + split_size + (n_samples % n_folds)
            else:
                step_size = (n_samples - (train_size + full_test)) // n_folds
                final_fold_start = n_samples - (train_size + full_test)
                range_start = (final_fold_start - (step_size * (n_splits - 1))) + train_size
            test_starts = range(range_start, n_samples, step_size)
        # print(test_starts)
        # Generate data splits.
        for test_start in test_starts:
            idx_start = test_start - train_size if self.train_size is not None else 0
            # print(idx_start, test_start, full_test)
            # Ensure we always return a test set of the same size
            if indices[test_start:test_start + full_test].size < full_test:
                continue
            yield (indices[idx_start:test_start],
                   indices[test_start + delay:test_start + full_test])
