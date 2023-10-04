import argparse
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

import numpy as np
import json
import os
from collections import Counter
import math

from utils import weight_to_json_y_s, get_data_paths


def get_reweight_func(reweight_method):
    """ Return The Specified Weighting Func
    """
    if reweight_method == 'sw':
        return simple_weight
    else:
        print('No Match Reweight Method!')
        exit(1)


def simple_weight(pred_freq, train_freq, thres_low, thres_high):
    """ A Simple Weighted Mapping Func
    """
    weight = pred_freq / train_freq
    if weight < thres_low:
        weight = thres_low
    elif weight > thres_high:
        weight = thres_high

    return weight
    

def main(args):
    """ Generate Weighted Results Based On Predicted Frequency
    """
    print('========== weight score cal ==========')
    paths = get_data_paths(args)
    dataset_name = paths['dataset_name']
    train_data_paths = paths['train_data_paths']
    data_dirs = paths['data_dirs']
    user_save_dirs = paths['user_save_dirs']
    cluster_name = paths['cluster_name']
    mae_ratio_threshold = args.reweight_threshold
    thres_low, thres_high = args.thres_low, args.thres_high

    pred_paths = [os.path.join(user_save_dir, cluster_name + '_pred.json') for user_save_dir in user_save_dirs]
    cluster_res_dirs = [os.path.join(user_save_dir, 'cluster_res', args.embedding_type) for user_save_dir in user_save_dirs]
    cluster_res_paths = [os.path.join(cluster_res_dir, cluster_name+'_res.json') for cluster_res_dir in cluster_res_dirs]
    target_season_list = [i for i in range(1, 5)]

    # ========== roll seasonal single pass ==========
    for index in tqdm(range(4)):
        print('---------- season {} ----------'.format(index+1))
        # get path
        pred_path = pred_paths[index]
        cluster_res_path = cluster_res_paths[index]
        user_save_dir = user_save_dirs[index]
        target_season = target_season_list[index]
        data_dir = data_dirs[index]

        pred_df = pd.read_json(pred_path)
        cluster_res_df = pd.read_json(cluster_res_path)

        # ----- calculate the global frequency of each cluster on the training set -----
        total_count = cluster_res_df['count'].sum()

        cluster_freqs = []
        for index in range(cluster_res_df.shape[0]):
            cur_cluster_freq = cluster_res_df.iloc[index]['count'] / total_count
            cluster_freqs.append(cur_cluster_freq)

        cluster_res_df['train_freq'] = cluster_freqs
        global_weights = []

        # ----- calculate weighted value -----
        test_pred_pool = []
        for index in range(pred_df.shape[0]):
            cur_cluster = pred_df.iloc[index]
            cluster_id = cur_cluster['cluster_id']
            cur_train_freq = cluster_res_df[cluster_res_df['cluster_id'] == cluster_id].iloc[0]['train_freq']

            test_pred = pd.DataFrame(cur_cluster['test_pred'])
            weights = []
            for index in range(test_pred.shape[0]):
                cur_pred_freq = test_pred.iloc[index]['pred_freq']
                reweight_func = get_reweight_func(args.reweight_method)
                cur_weight = reweight_func(cur_pred_freq, cur_train_freq, thres_low, thres_high)
                weights.append(cur_weight)

            test_pred['weight'] = weights
            cur_y_s_pool  = []
            for index in range(test_pred.shape[0]):
                cur_test_pred = test_pred.iloc[index]
                cur_res = {
                    'y_s':  cur_test_pred['y_s'],
                    'pred_freq': cur_test_pred['pred_freq'],
                    'weight': cur_test_pred['weight']
                }
                cur_y_s_pool.append(cur_res)

            global_weights.extend(weights)
            test_pred_pool.append(cur_y_s_pool)

        # show weighted result distribution
        print('↓ global reweight show ↓')
        weights_view = pd.DataFrame(global_weights)
        percent_list = [i*5 / 100 for i in range(0, 21)]
        print(weights_view.describe(percentiles=percent_list))

        pred_df['test_pred'] = test_pred_pool
        pred_df = pred_df.drop('train_pred', axis=1)

        pred_save_path = os.path.join(user_save_dir, cluster_name + '_weight.json')
        pred_df.to_json(pred_save_path, indent=2, force_ascii=False, orient='records')

        # ----- generate the weighted training set -----
        weight_to_json_y_s(
            online_data_dir=data_dir, 
            res_path=cluster_res_path, 
            weight_path=pred_save_path, 
            save_dir=user_save_dir, 
            mae_ratio_threshold=mae_ratio_threshold)


if __name__ == '__main__':
    """ Main Entrance
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--embedding_type", type=str, default='SBERT')
    parser.add_argument("--cluster_threshold", type=float, default=0.6)
    parser.add_argument("--predict_method", type=str, default='ys')
    parser.add_argument("--predict_threshold", type=int, default=30)
    parser.add_argument("--reweight_method", type=str, default='sw')
    parser.add_argument("--reweight_threshold", type=float, default=1)
    parser.add_argument("--thres_low", type=float, default=0)
    parser.add_argument("--thres_high", type=float, default=float('inf'))

    args = parser.parse_args()
    print(vars(args))
    main(args)