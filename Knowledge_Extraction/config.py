import json
import os
import pandas as pd
from transformers import BertTokenizer
class Config(object):
    def __init__(self):
        self.lr = 1e-5
        self.dataset = 'dataset'
        self.batch_size = 4
        self.max_epoch = 10
        self.max_len = 300
        self.bert_name = 'hfl/chinese-bert-wwm-ext'
        self.bert_dim = 768
        self.train_proportion = 0.8 #训练集占比

        self.train_path = self.dataset + '/train_data.json'
        self.dev_path = self.dataset + '/dev_data.json'
        self.rel_path = self.dataset + '/all_50_schemas'
        self.rel2ids = self.create_vocabulary(os.path.join(self.dataset, 'all_50_schemas'))
        self.num_relations = len(self.rel2ids)  # 注意， 该数据集存在重名关系“成立日期”

        self.alias_dict = self.get_loc_dict()   #地名别名到标准名字典
        self.chara_dict = self.get_chara_dict() #人物别名到标准名字典

        self.save_weights_dir = 'saved_weights/'
        self.save_logs_dir = 'saved_logs/'
        self.result_dir = 'results/' 

        self.period = 200
        self.test_epoch = 3
        self.weights_save_name = 'model.pt'
        self.log_save_name = 'model.out'
        self.result_save_name = 'result.json'

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name)

    def create_vocabulary(self, path):
        id2rel = dict()
        id2rel['unknown'] = 0
        count = 1
        with open(path, encoding='utf8') as f:
            for line in f:
                rel = json.loads(line)
                id2rel[rel['predicate']] = count
                count += 1
        return id2rel

    def get_loc_dict(self):
        path = "../spider_tools/locDict"
        alias_dict = dict()
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "r", encoding='utf-8') as f:
                for line in f:
                    tmp = json.loads(line)
                    if tmp.get('alias') is not None:
                        for a in tmp['alias']:
                            alias_dict[a] = tmp['name']

        return alias_dict

    def get_chara_dict(self):
        path = "../spider_tools/baike"
        alias_dict = dict()
        for filename in os.listdir(path):
            df = pd.read_csv(os.path.join(path, filename)).loc[:, ['name', 'alias']].dropna()
            for row_index, row in df.iterrows():
                alias = row['alias'].split('、')
                for a in alias:
                    alias_dict[a] = row['name']
        return alias_dict
