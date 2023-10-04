set -e

gpu='0'

# settings
# roll_online_bf20
data_name='roll_online_bf20_demo'
year_type='online_bf20'

sentence_transformer_path='path/to/sentence_roberta_wwm_ext'

embedding_type='SBERT'
cluster_threshold=0.6
predict_threshold=3
reweight_method='sw'
reweight_threshold=5.0
thres_low=0.5
thres_high=2.0
predict_method='bsr'

# S1: News Representation
CUDA_VISIBLE_DEVICES=${gpu} python get_embeddings.py \
                            --data_name ${data_name} \
                            --sentence_transformer_path ${sentence_transformer_path} \
                            --embedding_type ${embedding_type} \
                            --cluster_threshold ${cluster_threshold} \
                            --predict_method ${predict_method} \
                            --predict_threshold ${predict_threshold} \
                            --reweight_method ${reweight_method} \
                            --reweight_threshold ${reweight_threshold} \
                            --thres_low ${thres_low} \
                            --thres_high ${thres_high}

# S2: Topic Discovery
python single_pass_cluster.py \
    --data_name ${data_name} \
    --embedding_type ${embedding_type} \
    --cluster_threshold ${cluster_threshold} \
    --predict_method ${predict_method} \
    --predict_threshold ${predict_threshold} \
    --reweight_method ${reweight_method} \
    --reweight_threshold ${reweight_threshold} \
    --thres_low ${thres_low} \
    --thres_high ${thres_high}

# S3: Temporal Distribution Modeling and Forecasting
python predict_freq.py \
    --data_name ${data_name} \
    --embedding_type ${embedding_type} \
    --cluster_threshold ${cluster_threshold} \
    --predict_method ${predict_method} \
    --predict_threshold ${predict_threshold} \
    --reweight_method ${reweight_method} \
    --reweight_threshold ${reweight_threshold} \
    --year_type ${year_type} \
    --thres_low ${thres_low} \
    --thres_high ${thres_high}

# S4: Forecast-Based Adaptation
python weight_score_cal.py \
    --data_name ${data_name} \
    --embedding_type ${embedding_type} \
    --cluster_threshold ${cluster_threshold} \
    --predict_method ${predict_method} \
    --predict_threshold ${predict_threshold} \
    --reweight_method ${reweight_method} \
    --reweight_threshold ${reweight_threshold} \
    --thres_low ${thres_low} \
    --thres_high ${thres_high}