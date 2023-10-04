import argparse
from ast import Str
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
from sentence_transformers import models, SentenceTransformer
from utils import get_data_paths


def main(args):
    """ Obtain News Representation With Sentence-BERT
    """
    print('========== get embeddings ==========')
    model_path = args.sentence_transformer_path
    paths = get_data_paths(args)
    dataset_name = paths['dataset_name']
    online_train_paths = paths['train_data_paths']
    user_save_dirs = paths['user_save_dirs']

    # ========== roll seasonal single pass ==========
    for index in tqdm(range(4)):
        print('---------- season {} ----------'.format(index+1))
        online_train_path = online_train_paths[index]
        user_save_dir = user_save_dirs[index]
        online_train_df = pd.read_json(online_train_path)

        # init model
        word_embedding_model = models.Transformer(model_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=True,
                                    pooling_mode_cls_token=False,
                                    pooling_mode_max_tokens=False)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cuda:0')

        # get news representation and save as dict
        embedding_dict = dict()
        for index in tqdm(range(len(online_train_df))):
            news = online_train_df.iloc[index]
            news_label = news['label']
            embedding_dict[news['id']] = model.encode(news['content'])

        # get embedding save path
        embedding_save_path = os.path.join(user_save_dir, 'SBERT_embedding.pkl')

        # save news representation into pickle format
        with open(embedding_save_path, 'wb') as handle:
            pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # read in news representation
        with open(embedding_save_path, 'rb') as handle:
            pkl_data = pickle.load(handle)

        print('embedding len:', len(pkl_data))
        for i, val in pkl_data.items():
            print('↓ embedding dict sample ↓')
            print('id:', i, ' single embedding shape:', val.shape)
            break


if __name__ == '__main__':
    """ Main Entrance
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--embedding_type", type=str, default='SBERT')
    parser.add_argument("--sentence_transformer_path", type=str, default='imxly/sentence_roberta_wwm_ext')
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