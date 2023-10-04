import os
import argparse
import json
import time
from datetime import datetime
from utils.utils import get_tensorboard_writer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='bigru_endef')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--aug_prob', type=float, default=0.1)
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--root_path', default='./data/')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=2378)
parser.add_argument('--gpu', default='0')
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--save_log_dir', default= './logs')
parser.add_argument('--save_param_dir', default= './param_model')
parser.add_argument('--param_log_dir', default = './logs/param')

parser.add_argument('--tensorboard_dir', default='./logs/tensorlog')
parser.add_argument('--bert_path', default = 'chinese-bert-wwm-ext')
parser.add_argument('--data_type', default = 'online')
parser.add_argument('--data_name', default = '')
parser.add_argument('--eval_mode', type=bool, default = False)

parser.add_argument('--split_level', default = 'monthly')
parser.add_argument('--specific_season', type=int, default = -1)
parser.add_argument('--train_shuffle', type=bool, default = False)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from grid_search import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {};'.format \
    (args.lr, args.model_name, args.batchsize, args.epoch, args.gpu))
print('data_type: {}; data_path: {}; data_name: {};'.format \
    (args.data_type, args.root_path, args.data_name))

config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'root_path': args.root_path,
        'aug_prob': args.aug_prob,
        'weight_decay': 5e-5,
        'model':
            {
            'mlp': {'dims': [384], 'dropout': 0.2}
            },
        'emb_dim': args.emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_log_dir': args.save_log_dir,
        'save_param_dir': args.save_param_dir,
        'param_log_dir': args.param_log_dir,

        'tensorboard_dir': args.tensorboard_dir,
        'bert_path': args.bert_path,
        'data_type': args.data_type,
        'data_name': args.data_name,
        'eval_mode': args.eval_mode,

        'split_level': args.split_level,
        'specific_season': args.specific_season,
        'train_shuffle': args.train_shuffle
        }

if __name__ == '__main__':
    st_time = time.time()
    metrics_dict_test, metrics_dict_test_in_train = dict(), dict()
    if config['split_level'] == 'seasonal':
        config['time_str'] = 'season'
        config['save_log_dir'] = './seasonal_logs'
        config['param_log_dir'] = './seasonal_logs/param'
        config['tensorboard_dir'] = './seasonal_logs/tensorlog'
        config['data_name'] = config['data_name']+'_seed-'+str(config['seed'])
        writer = get_tensorboard_writer(config)
        season_list = [i for i in range(1, 5)]
        if config['specific_season'] == -1:
            for season in season_list:
                config['time_index'] = season
                season_domain_mapper = {  # year mapper
                    1: 4,
                    2: 4,
                    3: 5,
                    4: 5
                }
                year_season_domain_mapper = {
                    1: 15,
                    2: 16,
                    3: 17,
                    4: 18
                }
                if 'ys' in config['model_name']:
                    config['domain_num'] = year_season_domain_mapper[season]
                else:
                    config['domain_num'] = season_domain_mapper[season]
                
                best_metric_test, best_metric_test_in_train = Run(config = config, writer = writer).main()
                metrics_dict_test[season] = best_metric_test
                metrics_dict_test_in_train[season] = best_metric_test_in_train
        else:
            config['time_index'] = config['specific_season']
            best_metric_test, best_metric_test_in_train = Run(config = config, writer = writer).main()
            metrics_dict_test[config['specific_season']] = best_metric_test
            metrics_dict_test_in_train[config['specific_season']] = best_metric_test_in_train  
        save_dir = './seasonal_logs/seasonal_log'
    else:
        print('Error split level!')
        exit(1)

    # Show training time
    ed_time = time.time()
    gap_time_sec = int(ed_time-st_time)
    m, s = divmod(gap_time_sec, 60)
    h, m = divmod(m, 60)
    print('training time: {:02d}:{:02d}:{:02d}'.format(h, m, s))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if config['model_name'] == 'bert':
        test_save_path = os.path.join(save_dir, config['data_name']+'_test.json')
    else:
        test_save_path = os.path.join(save_dir, config['model_name']+'_'+config['data_name']+'_test.json')
    with open(test_save_path, 'w') as file:
        json.dump(metrics_dict_test, file, indent=4, ensure_ascii=False)