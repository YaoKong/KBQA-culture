
import json
import os
import pandas as pd
def get_loc_dict():
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

def get_chara_dict():
    path = "../spider_tools/baike"
    alias_dict = dict()
    for filename in os.listdir(path):
        df = pd.read_csv(os.path.join(path, filename)).loc[:, ['name', 'alias']].dropna()
        for row_index, row in df.iterrows():
            alias = row['alias'].split('、')
            for a in alias:
                alias_dict[a] = row['name']
    return alias_dict
if __name__ == '__main__':
    # print(get_loc_dict())
    print(get_chara_dict())