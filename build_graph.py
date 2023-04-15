import os
from collections import defaultdict

from py2neo import Graph, Node
import pandas as pd
import hanlp
import pickle
import torch
import gc

from tqdm import tqdm


def generate_id_column(df, col_name, id_prefix):
    tmp = []
    for i in df.index:
        tmp.append("{}{:0>8d}".format(id_prefix, i))
    df[col_name] = tmp
class CultureGraph:
    def __init__(self):
        cur_sep = os.path.sep
        dir = cur_sep.join(os.path.abspath(__file__).split(cur_sep)[:-1])
        self.data_path = os.path.join(dir, 'data')
        self.graph = None


    def read_poetry_data(self):
        poetry_path = os.path.join(self.data_path, 'Poetry-master')

        poetry_df = None
        for filename in os.listdir(poetry_path):
            if "csv" in filename:
                if poetry_df is None:
                    poetry_df = pd.read_csv(os.path.join(poetry_path, filename))
                else:
                    poetry_df = pd.concat([poetry_df, pd.read_csv(os.path.join(poetry_path, filename))],
                                                 ignore_index=True)

        generate_id_column(poetry_df, 'pid:ID', "pid")
        # 生成”创作出“关系
        chara_df = poetry_df.loc[:, ["作者", "朝代"]].drop_duplicates()
        generate_id_column(chara_df, 'cid:ID', "cid")
        indite_rels = pd.merge(poetry_df.loc[:, ["pid:ID", "作者", "朝代"]], chara_df, how="left")
        # indite_rels = indite_rels.loc[:, ["cid:ID", "pid:ID"]]
        # indite_rels.columns = [":START_ID", ":END_ID"]
        indite_rels[":TYPE"] = "创作出"


        # 新增label列
        poetry_df[':LABEL'] = "诗词"
        chara_df[':LABEL'] = "人物"

        # 生成“提到”关系
        count = 0
        rels_refer_pos = [] #诗词提到的地区 start为诗词 end为地区
        rels_refer_char = [] #诗词提到人物
        location = []

        hanlp.pretrained.mtl.ALL  # MTL多任务，具体任务见模型名称，语种见名称最后一个字段或相应语料库
        HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
        count = 0

        for idx, data in tqdm(poetry_df.iterrows()):
            count += 1
            if count >= 10:
                break
            texts = [data["内容"][:3000] if len(data["内容"]) > 3000 else data["内容"]]
            texts.append(data["题目"])
            results = HanLP(texts, tasks='ner')["ner/msra"]

            for result in results:
                for r in result:
                    if r[1] == "LOCATION":
                        if len(r[0]) > 1:  # 防止加入单字地名
                            ID = "{}{:0>8d}".format("pos", len(location))
                            location.append([ID, r[0]])
                            rels_refer_pos.append([data["pid:ID"], data["题目"], ID, r[0]])
                    elif r[1] == "PERSON":  # 只在poetry数据集里出现的诗人找
                        if len(r[1]) > 1:
                            tmp = chara_df.loc[chara_df["作者"] == r[0]]
                            # 把同名同姓的一起连上
                            for i, j in tmp.iterrows():
                                rels_refer_char.append([data["pid:ID"], data['题目'], j["cid:ID"], r[0]])

        rels_refer_char = pd.DataFrame(rels_refer_char, columns=[":START_ID", "诗词", ":END_ID", "人物"])
        rels_refer_char[":TYPE"] = "提到"
        rels_refer_pos = pd.DataFrame(rels_refer_pos, columns=[":START_ID", "诗词", ":END_ID", "地点"])
        rels_refer_pos[":TYPE"] = "提到"

        location = pd.DataFrame(location, columns=["pos:ID", "名字"])
        location[":LABEL"] = "地点"

        chara_df.rename(index={"作者": "姓名"}) # 给索引改名

        # 导出为csv
        poetry_df.to_csv(os.path.join(self.data_path, 'poetry.csv'), index=False)
        chara_df.to_csv(os.path.join(self.data_path, 'character.csv'), index=False)
        location.to_csv(os.path.join(self.data_path, 'location.csv'), index=False)
        indite_rels.to_csv(os.path.join(self.data_path, 'indite_rels.csv'), index=False)
        rels_refer_char.to_csv(os.path.join(self.data_path, 'rels_refer_char.csv'), index=False)
        rels_refer_pos.to_csv(os.path.join(self.data_path, 'rels_refer_pos.csv'), index=False)





if __name__ == '__main__':
    graph = CultureGraph()
    graph.read_poetry_data()