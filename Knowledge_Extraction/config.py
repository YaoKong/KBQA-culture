import json
import os
import pandas as pd
from transformers import BertTokenizer
import hanlp
import re
class Config(object):
    def __init__(self, ner_flag=False):
        self.lr = 1e-5
        self.dataset = 'dataset'
        self.batch_size = 4
        self.max_epoch = 5
        self.max_len = 512
        self.bert_name = 'hfl/chinese-bert-wwm-ext'
        self.bert_dim = 768
        self.train_proportion = 0.8 #训练集占比

        self.train_path = self.dataset + '/train_data.json'
        self.dev_path = self.dataset + '/dev_data.json'
        self.rel_path = self.dataset + '/all_50_schemas'
        self.rel2ids, self.ids2rel = self.create_vocabulary(os.path.join(self.dataset, 'all_50_schemas'))
        self.num_relations = len(self.rel2ids)  # 注意， 该数据集存在重名关系“成立日期”



        self.save_weights_dir = 'saved_weights/'
        self.save_logs_dir = 'saved_logs/'
        self.result_dir = 'results/' 

        self.period = 200
        self.test_epoch = 3
        self.weights_save_name = 'model.pt'
        self.log_save_name = 'model.out'
        self.result_save_name = 'result.json'

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name, model_max_length=512)
        if ner_flag:
            self.loc_dict = self.get_loc_dict()   #地名别名到标准名字典
            self.chara_dict = self.get_chara_dict() #人物别名到标准名字典
            hanlp.pretrained.mtl.ALL
            self.HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
            ner = self.HanLP['ner/msra']
            ner.dict_whitelist = self.create_dict_whitelist()


    def create_dict_whitelist(self):
        whitelist = dict()
        for key, value in self.loc_dict.items():
            whitelist[key] = 'LOCATION'
            whitelist[value] = 'LOCATION'

        for key, value in self.chara_dict.items():
            whitelist[key] = 'PERSON'
            whitelist[value] = 'PERSON'

        return whitelist
    def create_vocabulary(self, path):
        id2rel = dict()
        id2rel[0] = 'unknown'

        rel2id = dict()
        rel2id['unknown'] = 0
        count = 1

        with open(path, encoding='utf8') as f:
            for line in f:
                rel = json.loads(line)
                rel2id[rel['predicate']] = count
                id2rel[count] = rel['predicate']
                count += 1
        return rel2id, id2rel

    def getIdx2Rel(self, idx):
        for key in self.rel2ids:
            if self.rel2ids[key] == idx:
                return key
        return 'UNKNOWN'

    def split_to_list(self, str):
        '''
        把字符串拆分为列表，若strList已经拆分好则直接返回
        :param str:
        :return:
        '''
        if str.find('《') != -1:
            str = re.findall("[《](.*?)[》]", str)
        elif str.find('、') != -1:
            str = str.split('、')
        else:
            str = str.split()
        return str
    def get_loc_dict(self):
        path = "../spider_tools/locDict/locDict.csv"
        alias_dict = dict()
        df = pd.read_csv(path)
        for index, row in df.iterrows():
            if pd.notna(row['alias']):
                alias = row['alias']
                if isinstance(alias, list) and len(alias) == 1:
                    alias = self.split_to_list(alias[0])
                elif type(alias) == type('str'):
                    alias = [alias]
                if alias != '':
                    for a in alias:
                        alias_dict[a] = row['name']
        return alias_dict

    def get_chara_dict(self):
        path = "../spider_tools/baike"
        alias_dict = dict()
        for filename in os.listdir(path):
            df = pd.read_csv(os.path.join(path, filename)).loc[:, ['name', 'alias']].dropna()
            for row_index, row in df.iterrows():
                alias = row['alias']
                if isinstance(alias, list) and len(alias) == 1:
                    alias = self.split_to_list(alias[0])
                elif type(alias) == type('str'):
                    alias = [alias]
                for a in alias:
                    alias_dict[a] = row['name']
        return alias_dict

if __name__ == "__main__":
    config = Config(ner_flag=True)