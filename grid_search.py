import logging
import os
import json

from models.bert import Trainer as BertTrainer
from models.eann import Trainer as EANNTrainer

def frange(x, y, jump):
  while x < y:
      x = round(x, 8)
      yield x
      x += jump
      
class Run():
    def __init__(self,
                 config,
                 writer
                 ):
        self.config = config
        self.writer = writer

    def getFileLogger(self, log_file):
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(level = logging.INFO)
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        param_log_dir = self.config['param_log_dir']
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        param_log_file = os.path.join(param_log_dir, self.config['model_name'] + '_' + self.config['data_name'] +'_'+ 'param.txt')
        logger = self.getFileLogger(param_log_file)
        
        train_param = {
            'lr': [self.config['lr']]
        }
        print(train_param)
        param = train_param
        best_param = []
        
        # json_path
        json_dir = os.path.join(
            './logs/json/',
            self.config['model_name'] + '_' + self.config['data_name']
        )
        if self.config['split_level'] == 'monthly':
            json_path = os.path.join(
                json_dir,
                'month_' + str(self.config['time_index']) + '.json'
            )
        elif self.config['split_level'] == 'seasonal':
            json_path = os.path.join(
                json_dir,
                'season_' + str(self.config['time_index']) + '.json'
            )
        else:
            print('Error split level!')
            exit(1)

        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        json_result = []
        for p, vs in param.items():
            best_metric = {}
            best_metric['metric'] = 0
            best_v = vs[0]
            best_model_path = None
            for i, v in enumerate(vs):
                self.config['lr'] = v
                if 'eann' in self.config['model_name'] and 'endef' not in self.config['model_name']:
                    trainer = EANNTrainer(self.config, self.writer)
                elif self.config['model_name'] == 'bert':
                    trainer = BertTrainer(self.config, self.writer)
                test_metrics, test_in_train_metrics, model_path = trainer.train(logger)
                json_result.append({
                    'lr': self.config['lr'],
                    'metric': test_metrics
                })
                if test_metrics['metric'] > best_metric['metric']:
                    best_metric = test_metrics
                    best_in_train_metric = test_in_train_metrics
                    best_v = v
                    best_model_path = model_path
            best_param.append({p: best_v})
            print("best model path:", best_model_path)
            print("best macro f1:", best_metric['metric'])
            print("best metric:", best_metric)
            logger.info("best model path:" + best_model_path)
            logger.info("best param " + p + ": " + str(best_v))
            logger.info("best metric:" + str(best_metric))
            logger.info('==================================================\n\n')
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)

        return best_metric, best_in_train_metric
