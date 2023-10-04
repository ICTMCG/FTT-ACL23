# -*- coding: UTF-8 -*-
import argparse
import os
import sys
import jieba
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_data_paths


def res_process(cluster_res, online_path):
    """ Process The Results Into Analyzable Format
    """
    online_df = pd.read_json(online_path)
    cluster_view_pool, cluster_res_pool = [], []
    for cluster_id, cluster_list in tqdm(cluster_res.items()):
        count = len(cluster_list)
        view_contents, res_contents = [], []
        for id in cluster_list:
            cur_data = online_df.iloc[id]
            id = cur_data['id']
            m = cur_data['month']
            s = cur_data['season']
            y = cur_data['year']
            y_m = str(cur_data['year']) + '_' + str(cur_data['month'])
            y_s = str(cur_data['year']) + '_' + str(cur_data['month'])
            label = cur_data['label']
            content = cur_data['content']
            cluster_label = cluster_id

            res_contents.append({
                'id': id,
                'year': y,
                'season': s,
                'month': m,
                'year-season': y_s,
                'year-month': y_m,
                'label': label,
                'cluster_label': cluster_label,
                'content': content
            })
            view_contents.append(content)
        cur_view_cluster = {
            'cluster_id': cluster_id,
            'count': count,
            'contents': view_contents,
        }
        cur_res_cluster = {
            'cluster_id': cluster_id,
            'count': count,
            'contents': res_contents
        }
        cluster_view_pool.append(cur_view_cluster)
        cluster_res_pool.append(cur_res_cluster)

    return cluster_view_pool, cluster_res_pool


class SinglePassCluster():
    """ Single-Pass Clustering
    """
    def __init__(self, stopWords_path="", my_stopwords=None,
                 max_df=0.5, max_features=1000,
                 simi_threshold=0.5, cluster_name='', res_save_dir='', res_save_path="./cluster_res_ori.json"):
        self.simi_thr = simi_threshold
        self.cluster_center_vec = []
        self.cluster_vec_memory = []
        self.idx_2_text = {}
        self.cluster_2_idx = {}
        self.res_path = res_save_path
        self.cluster_name = cluster_name
        self.res_save_dir = res_save_dir
    
    def load_SBERT_embeddings(self, embedding_path):
        print('load in SBERT embeddings...')
        with open(embedding_path, 'rb') as handle:
            pkl_data = pickle.load(handle)

        for i, val in pkl_data.items():
            print('id:{} embedding shape:{} embedding type:{}'.format(i, val.shape, type(val)))
            break

        np_data = []
        for i, val in pkl_data.items():
            np_data.append(val)
        np_data = np.array(np_data)
        print('np_data shape:{} np_data type:{}'.format(np_data.shape, type(np_data)))

        return np_data

    def cut_sentences(self, data_path):
        if isinstance(data_path, str):
            if not os.path.exists(data_path):
                 print("path: {} is not exist !!!".format(data_path))
                 sys.exit()
            else:
                _texts = []
                df = pd.read_json(data_path)
                for index in range(df.shape[0]):
                    cur_data = df.iloc[index]
                    _texts.append(cur_data['content'].strip())
                texts = _texts
        texts_cut = [" ".join(jieba.lcut(t)) for t in texts]
        self.idx_2_text = {idx: text for idx, text in enumerate(texts)}
        return texts_cut

    def cosion_simi(self, vec):
        simi = cosine_similarity(np.array([vec]), np.array(self.cluster_center_vec))
        max_idx = np.argmax(simi, axis=1)[0]
        max_val = simi[0][max_idx]
        return max_val, max_idx

    def single_pass(self, text_path, embedding_type, embedding_path=''):        
        if embedding_type == 'SBERT':
            SBERT_embeddings = self.load_SBERT_embeddings(embedding_path)
            text_embeddings = SBERT_embeddings
        else:
            print('No match embedding type!')
            exit()
    
        # Start loop
        for idx, vec in tqdm(enumerate(text_embeddings)):
            # Init the first cluster
            if not self.cluster_center_vec:
                self.cluster_center_vec.append(vec)
                self.cluster_vec_memory.append([vec])
                self.cluster_2_idx[0] = [idx]
            # Clustering
            else:
                max_simi, max_idx = self.cosion_simi(vec)
                if max_simi >= self.simi_thr:
                    self.cluster_2_idx[max_idx].append(idx)
                    
                    # Update 
                    self.cluster_vec_memory[max_idx].append(vec)
                    self.cluster_center_vec[max_idx] = np.mean(self.cluster_vec_memory[max_idx], axis=0)
                else:
                    self.cluster_center_vec.append(vec)
                    self.cluster_2_idx[len(self.cluster_2_idx)] = [idx]

                    self.cluster_vec_memory.append([vec])

        with open(os.path.join(self.res_save_dir, self.cluster_name+'_ori.json'), "w", encoding="utf-8") as f:
            json.dump(self.cluster_2_idx, f, ensure_ascii=False)

        cluster_view_pool, cluster_res_pool = res_process(self.cluster_2_idx, text_path)
        pd.DataFrame(cluster_view_pool).to_json(os.path.join(self.res_save_dir, self.cluster_name+'_view.json'), indent=2, force_ascii=False, orient='records')
        pd.DataFrame(cluster_res_pool).to_json(os.path.join(self.res_save_dir, self.cluster_name+'_res.json'), indent=2, force_ascii=False, orient='records')


def main(args):
    """ Topic Discovery Using Single-Pass Clustering
    """
    print('========== single pass cluster ==========')
    paths = get_data_paths(args)
    dataset_name = paths['dataset_name']
    online_train_paths = paths['train_data_paths']
    user_save_dirs = paths['user_save_dirs']
    SBERT_embeddings_paths = [os.path.join(user_save_dir, 'SBERT_embedding.pkl') for user_save_dir in user_save_dirs]
    embedding_type = args.embedding_type

    threshold_list = [args.cluster_threshold]
    print('='* 60)
    print('='* 20, 'embedding_type:', embedding_type, '='*20)
    cur_dir_paths = [os.path.join(user_save_dir, 'cluster_res', args.embedding_type) for user_save_dir in user_save_dirs]
    
    # ========== roll seasonal single pass ==========
    for index in tqdm(range(4)):
        print('---------- season {} ----------'.format(index+1))
        # get cur path
        cur_dir_path = cur_dir_paths[index]
        online_train_path = online_train_paths[index]
        SBERT_embeddings_path = SBERT_embeddings_paths[index]

        if not os.path.exists(cur_dir_path):
            os.makedirs(cur_dir_path)
        for index, threshold in tqdm(enumerate(threshold_list)):
            print('-'*10, 'threshold:', threshold, '-'*10)
            cluster_name = embedding_type + '_' + str(threshold)

            cluster = SinglePassCluster(max_features=100, simi_threshold=threshold, cluster_name=cluster_name, res_save_dir=cur_dir_path)
            cluster.single_pass(online_train_path, embedding_type, SBERT_embeddings_path)


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



