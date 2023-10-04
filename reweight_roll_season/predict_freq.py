import argparse
import pandas as pd
from tqdm import tqdm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt

import numpy as np
import json
import os
from collections import Counter

from utils import predict_data_prepare_y_s, get_data_paths, get_years
import warnings
import logging


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def main(args):
    """ Predict Cluster Frequency
    """
    # Silence the Prophet
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.simplefilter("ignore", FutureWarning)
    logging.getLogger('prophet').setLevel(logging.ERROR)

    print('========== predict freq ==========')
    paths = get_data_paths(args)
    type_years = get_years(args)
    dataset_name = paths['dataset_name']
    train_data_paths = paths['train_data_paths']
    user_save_dirs = paths['user_save_dirs']
    cluster_name = paths['cluster_name']

    remark_threshold = args.predict_threshold

    cluster_res_dirs = [os.path.join(user_save_dir, 'cluster_res', args.embedding_type) for user_save_dir in user_save_dirs]
    cluster_res_paths = [os.path.join(cluster_res_dir, cluster_name+'_res.json') for cluster_res_dir in cluster_res_dirs]
    # print('user save dir:', user_save_dir)
    print('cluster_res_dir:', cluster_res_dirs)
    print('cluster_res_path:', cluster_res_paths)

    target_season_list = [i for i in range(1, 5)]
    total_mae = 0
    total_mse = 0
    total_mae_ratio = 0
    total_cluster_count = 0

    # ========== roll seasonal single pass ==========
    for index in tqdm(range(4)):
        # get cur path
        train_data_path = train_data_paths[index]
        user_save_dir = user_save_dirs[index]
        cluster_res_path = cluster_res_paths[index]
        target_season = target_season_list[index]

        predict_data_prepare_y_s(
            online_data_path=train_data_path, 
            user_save_dir=user_save_dir,
            cluster_res_path=cluster_res_path,
            remark_threshold=remark_threshold,
            cluster_name=cluster_name,
            start_year=type_years['start_year'],
            target_year=type_years['target_year'],
            target_season=target_season)

        # load in data
        distribution_path = os.path.join(user_save_dir, cluster_name + '_distribution.json')
        distribution_df = pd.read_json(distribution_path)

        save_pool = []
        for index in tqdm(range(distribution_df.shape[0])):
            cur_res = distribution_df.iloc[index]
            cur_cluster_id = cur_res['cluster_id']
            cur_count = cur_res['count']
            cur_distribution = cur_res['y_s_distribution']
            cur_distribution = pd.DataFrame(cur_res['y_s_distribution'])    
            cur_distribution = cur_distribution.drop(['count'], axis=1)
            
            cur_distribution.columns = ['ds', 'y']
            cur_distribution['ds'] = pd.to_datetime(cur_distribution['ds'])

            # divide training and testing sets
            cur_distribution['floor'] = 0
            train = cur_distribution

            test = [str(type_years['target_year'])+ '-Q' + str(target_season)]
            test = pd.DataFrame(test)
            test.columns = ['ds']
            test['ds'] = pd.to_datetime(test['ds'])
            test['floor'] = 0

            # model setting
            if args.predict_method == 'bsr':
                model = Prophet(yearly_seasonality=False)
                seasons = ['Q1', 'Q2', 'Q3', 'Q4']
                for i, season in enumerate(seasons):
                    cur_distribution[season] = (cur_distribution['ds'].dt.quarter == i + 1).values.astype('float')
                    train[season] = (train['ds'].dt.quarter == i + 1).values.astype('float')
                    test[season] = (test['ds'].dt.quarter == i + 1).values.astype('float')
                    model.add_regressor(season)
            elif args.predict_method == 'ys':
                model = Prophet()
            else:
                print('Error predict method!')
                exit(1)
            
            # fit training data
            with suppress_stdout_stderr():
                model.fit(train)

            # predict target time frequency
            pred = model.predict(test)
            pred['yhat'] = np.clip(pred['yhat'], a_min=0, a_max=None)

            y_pred = pred['yhat'].values

            # global prediction
            global_pred = model.predict(cur_distribution)
            global_pred['yhat'] = np.clip(global_pred['yhat'], a_min=0, a_max=None)
            global_y_true = cur_distribution['y'].values
            global_y_pred = global_pred['yhat'].values

            train_mae = mean_absolute_error(global_y_true, global_y_pred)
            train_mse = mean_squared_error(global_y_true, global_y_pred)
            train_mae_ratio = mean_absolute_error(global_y_true, global_y_pred)/global_y_true.mean()

            cur_res_pred = dict()
            cur_res = distribution_df.iloc[index]
            cur_res_pred['cluster_id'] = cur_res['cluster_id']
            cur_res_pred['count'] = cur_res['count']
            cur_res_pred['train_mae'] = train_mae
            cur_res_pred['train_mse'] = train_mse
            cur_res_pred['train_mae_ratio'] = train_mae_ratio

            total_mae += train_mae
            total_mse += train_mse
            total_mae_ratio += train_mae_ratio
            total_cluster_count += 1

            cur_distribution = pd.DataFrame(cur_res['y_s_distribution'])
            train_pred = []
            for index in range(cur_distribution.shape[0]):
                cur_train_pred = dict()
                cur_res = cur_distribution.iloc[index]
                cur_train_pred['y_s'] = cur_res['y_s']
                cur_train_pred['freq'] = cur_res['freq']
                cur_train_pred['count'] = cur_res['count']
                cur_train_pred['pred_freq'] = global_y_pred[index]
                train_pred.append(cur_train_pred)
            cur_res_pred['train_pred'] = train_pred

            test_pred = []
            for index, year_season in enumerate([str(type_years['target_year']) + '-Q' + str(target_season)]):
                cur_test_pred = dict()
                cur_test_pred['y_s'] = year_season
                cur_test_pred['pred_freq'] = y_pred[index]
                test_pred.append(cur_test_pred)
            cur_res_pred['test_pred'] = test_pred

            save_pool.append(cur_res_pred)

        pd.DataFrame(save_pool).to_json(os.path.join(user_save_dir , cluster_name+'_pred.json'), indent=2, force_ascii=False, orient='records')


if __name__ == '__main__':
    """ Main Entrance
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--embedding_type", type=str, default='SBERT')
    parser.add_argument("--cluster_threshold", type=float, default=0.6)
    parser.add_argument("--predict_method", type=str, default='ys')
    parser.add_argument("--predict_threshold", type=int, default=30)
    parser.add_argument("--year_type", type=str, default='online_bf22')
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--reweight_method", type=str, default='sw')
    parser.add_argument("--reweight_threshold", type=float, default=1)
    parser.add_argument("--thres_low", type=float, default=0)
    parser.add_argument("--thres_high", type=float, default=float('inf'))

    args = parser.parse_args()
    print(vars(args))
    main(args)