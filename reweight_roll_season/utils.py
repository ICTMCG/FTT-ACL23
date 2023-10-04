import pandas as pd
from tqdm import tqdm
import os
import shutil
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY


def get_gap_year_month(st_y_m, ed_y_m):
    """ Get Time Gap Between Two Year_Month
    """
    strt_dt = datetime.strptime(st_y_m, '%Y-%m')
    end_dt = datetime.strptime(ed_y_m, '%Y-%m')
    gap_dates = [dt for dt in rrule(MONTHLY, dtstart=strt_dt, until=end_dt)]
    gap_year_month = ['-'.join(str(date).split('-')[:2]) for date in gap_dates]

    return gap_year_month


def get_gap_year_season(st_y_s, ed_y_s):
    """ Get Time Gap Between Two Year_Season
    """    
    st_y_s = pd.to_datetime(st_y_s)
    ed_y_s = pd.to_datetime(ed_y_s)
    gap_quarters = []
    for year in range(st_y_s.year, ed_y_s.year+1):
        for quarter in range(1, 5):
            if year >= ed_y_s.year and quarter > ed_y_s.quarter:
                break
            gap_quarters.append(str(year) + '-Q' + str(quarter))
    return gap_quarters


def weight_to_json_y_s(online_data_dir, res_path, weight_path, save_dir='./', mae_ratio_threshold=1):
    """ Generate Weighted Training Set
    """
    print('now generate new train files...')
    # ----- load in data -----
    train_data_path = os.path.join(online_data_dir, 'train.json')
    val_data_path = os.path.join(online_data_dir, 'val.json')
    test_data_path = os.path.join(online_data_dir, 'test.json')

    weight_df = pd.read_json(weight_path)
    res_df = pd.read_json(res_path)
    train_df = pd.read_json(train_data_path)
    val_df = pd.read_json(val_data_path)
    test_df = pd.read_json(test_data_path)

    # ----- filtering based on train_mae_ratio -----
    filter_weight_df = weight_df[weight_df['train_mae_ratio'] < mae_ratio_threshold]
    filter_count = filter_weight_df.sum()['count']
    cluster_count = weight_df.sum()['count']
    train_count = train_df.shape[0]
    print('Total number after filtering:', filter_count)
    print('Filtered data ratio in training set: {:.2f}%'.format(filter_count / train_count * 100))
    
    # ----- generate id-weight dict for target season -----
    seasonal_weights_dict = {}
    for index in range(filter_weight_df.shape[0]):
        cur_cluster = filter_weight_df.iloc[index]
        cur_cluster_id = cur_cluster['cluster_id']
        cur_test_pred  = cur_cluster['test_pred']

        target_res = res_df[res_df['cluster_id'] == cur_cluster_id].iloc[0]
        cur_instance_ids = pd.DataFrame(target_res['contents'])['id'].values

        for res in cur_test_pred:
            cur_season = str(res['y_s'].split('-')[-1])
            if cur_season not in seasonal_weights_dict:
                seasonal_weights_dict[cur_season] = dict()

            cur_weight = res['weight']
            for cur_id in cur_instance_ids:
                seasonal_weights_dict[cur_season][cur_id] = cur_weight

    # ----- generate weighted training set based on id-weight dict -----
    for season, res in tqdm(seasonal_weights_dict.items()):
        cur_save_dir = save_dir
        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
        cur_train_save_path = os.path.join(cur_save_dir, 'train.json')
        cur_val_save_path = os.path.join(cur_save_dir, 'val.json')
        cur_test_save_path = os.path.join(cur_save_dir, 'test.json')

        cur_train_df = train_df.copy()
        for id, weight in res.items():
            target_index = cur_train_df[cur_train_df['id'] == id].index[0]
            cur_train_df.loc[target_index, 'weight'] = weight
        cur_train_df.to_json(cur_train_save_path, indent=2, force_ascii=False, orient='records')

        cur_val_df = val_df
        cur_test_df = test_df
        cur_val_df.to_json(cur_val_save_path, indent=2, force_ascii=False, orient='records')
        cur_test_df.to_json(cur_test_save_path, indent=2, force_ascii=False, orient='records')


def predict_data_prepare_y_s(online_data_path, user_save_dir, cluster_res_path, remark_threshold, cluster_name, start_year, target_year, target_season):
    """ Process Data For Frequency Prediction
    """
    train_st_y_s = str(start_year) + '-Q1'
    if target_season <= 2:
        train_ed_y_s = str(int(target_year)-1) + '-Q' + str(target_season+2)
    else:
        train_ed_y_s = str(target_year) + '-Q' + str(target_season-2)

    year_season_keys = get_gap_year_season(train_st_y_s, train_ed_y_s)
    online_df = pd.read_json(online_data_path)

    y_s_pool = []
    for index in tqdm(range(online_df.shape[0])):
        cur_data = online_df.iloc[index]
        cur_y_s = str(cur_data['year']) + '-Q' + str(cur_data['season'])
        y_s_pool.append(cur_y_s)
    online_df['year_season'] = y_s_pool

    global_year_season_counter = dict()
    for time in year_season_keys:
        global_year_season_counter[time] = 0
        
    for index in tqdm(range(online_df.shape[0])):
        cur_data = online_df.iloc[index]
        cur_y_s = cur_data['year_season']
        try:
            global_year_season_counter[cur_y_s] += 1
        except:
            print('Warning! Year_season {} not in the train years setting!'.format(cur_y_s))
            continue

    # ----- read in data and perform cluster filtering -----
    cur_res = pd.read_json(cluster_res_path)
    # only process samples from clusters exceeding remark_threshold 
    remark_res = cur_res[cur_res['count'] > remark_threshold]
    remark_res = remark_res.sort_values(by='count', ascending=False)

    distribution_pool = []
    for index in tqdm(range(remark_res.shape[0])):
        cur_cluster = remark_res.iloc[index]
        year_season_counter = dict()
        year_season_freq = dict()
        for key in year_season_keys:
            year_season_counter[key] = 0

        for instance in cur_cluster['contents']:
            cur_year_season = str(instance['year']) + '-Q' + str(instance['season'])
            try:
                year_season_counter[cur_year_season] += 1
            except:
                print('Warning! year-season {} not in the year_season_keys!'.format(cur_year_season))
                continue

        for year_season in year_season_keys:
            year_season_freq[year_season] = year_season_counter[year_season] / global_year_season_counter[year_season]

        cur_dict = {}
        cur_dict['cluster_id'] = cur_cluster['cluster_id']
        cur_dict['count'] = cur_cluster['count']
        cur_dict['y_s_distribution'] = []

        for key in year_season_freq.keys():
            cur_dict['y_s_distribution'].append({
                'y_s': key,
                'freq': year_season_freq[key],
                'count': year_season_counter[key]
            })
        distribution_pool.append(cur_dict)

    distribution_save_path = os.path.join(user_save_dir, cluster_name + '_distribution.json')
    pd.DataFrame(distribution_pool).to_json(distribution_save_path, indent=2, force_ascii=False, orient='records')


def get_data_paths(args):
    """ Get paths for all
    """
    season_list = [i for i in range(1, 5)]
    data_dir = os.path.join('./roll_seasonal_data', args.data_name)
    data_dirs = [os.path.join(data_dir, 'roll_season_'+str(season)) for season in season_list]
    train_data_paths = [os.path.join(data_dir, 'roll_season_'+str(season), 'train.json') for season in season_list]
    dataset_name = args.data_name + \
        '_' + args.embedding_type + '-sg-' + str(args.cluster_threshold)  + \
        '_prophet-' + str(args.predict_method) + '-' + str(args.predict_threshold) + \
        '_' + str(args.reweight_method) + '-' + str(args.reweight_threshold) + '-' + str(args.thres_low) + '-' + str(args.thres_high)
    print('dataset_name:', dataset_name)
    user_save_dirs = [os.path.join('./roll_seasonal_user_data/', dataset_name, 'roll_season_'+str(season)) for season in season_list]
    cluster_name = args.embedding_type + '_' + str(args.cluster_threshold)
    for dir in user_save_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)     
    res = {
        'data_dirs': data_dirs,
        'train_data_paths': train_data_paths,
        'dataset_name': dataset_name,
        'user_save_dirs': user_save_dirs,
        'cluster_name': cluster_name
    }
    print(res)
    return res


def get_years(args):
    """ Get years for target dataset
    """
    if args.year_type == 'online_bf20':
        train_years = [2016, 2017, 2018, 2019]
        target_year = 2020
    else:
        print('No Match Year Type!')
        exit(1)
    res = {
        'train_years': train_years,
        'start_year': train_years[0],
        'target_year': target_year
    }
    print(res)
    return res