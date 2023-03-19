import os
from collections import defaultdict

from py2neo import Graph, Node
import pandas as pd
import hanlp
import pickle
import torch
import gc

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

        self.generate_id_column(poetry_df, 'pid:ID', "pid")
        # 生成”创作出“关系
        chara_df = poetry_df.loc[:, ["作者", "朝代"]].drop_duplicates()
        self.generate_id_column(chara_df, 'cid:ID', "cid")
        indite_rels = pd.merge(poetry_df.loc[:, ["pid:ID", "作者", "朝代"]], chara_df, how="left")
        indite_rels = indite_rels.loc[:, ["cid:ID", "pid:ID"]]
        indite_rels.columns = [":START_ID", ":END_ID"]
        indite_rels[":TYPE"] = "创作出"


        # 新增label列
        poetry_df[':LABEL'] = "诗词"
        chara_df[':LABEL'] = "人物"

        # 生成“提到”关系
        count = 0
        # rels_refer_pos = pd.DataFrame([],  columns=[":START_ID", ":END_ID"]) #诗词提到的地区 start为诗词 end为地区
        # rels_refer_char = pd.DataFrame([],  columns=[":START_ID", ":END_ID"]) #诗词提到人物
        refer_rels = []
        location = []

        hanlp.pretrained.mtl.ALL  # MTL多任务，具体任务见模型名称，语种见名称最后一个字段或相应语料库
        HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
        count = 0
        batch_size = 128

        for idx, data in poetry_df.iterrows():
            count += 1
            content = data["内容"][:3000] if len(data["内容"]) > 3000 else data["内容"]
            result = HanLP(content, tasks='ner')["ner/msra"]

            for r in result:
                if r[1] == "LOCATION":
                    if len(r[0]) > 1:  # 防止加入单字地名
                        ID = "{}{:0>8d}".format("pos", len(location))
                        location.append([ID, r[0]])
                        refer_rels.append([data["pid:ID"], ID])
                elif r[1] == "PERSON":  # 只在poetry数据集里出现的诗人找
                    if len(r[1]) > 1:
                        tmp = chara_df.loc[chara_df["作者"] == r[1]]
                        # 把同名同姓的一起连上
                        for i, j in tmp.iterrows():
                            refer_rels.append([data["pid:ID"], j["cid:ID"]])

            if count % batch_size == 0:
                print("已处理(%d / %d)首诗词" % (count, poetry_df.shape[0]))
                gc.collect()
                torch.cuda.empty_cache()

        refer_rels = pd.DataFrame(refer_rels, columns=[":START_ID", ":END_ID"])
        refer_rels[":TYPE"] = "提到"

        location = pd.DataFrame(location, columns=["pos:ID", "名字"])
        location[":LABEL"] = "地点"

        chara_df.rename(index={"作者": "姓名"}) #给索引改名

        # 导出为csv
        poetry_df.to_csv(os.path.join(self.data_path, 'poetry.csv'), index=False)
        chara_df.to_csv(os.path.join(self.data_path, 'character.csv'), index=False)
        location.to_csv(os.path.join(self.data_path, 'location.csv'), index=False)
        indite_rels.to_csv(os.path.join(self.data_path, 'indite_rels.csv'), index=False)
        refer_rels.to_csv(os.path.join(self.data_path, 'refer_rels.csv'), index=False)


    def generate_id_column(self, df, col_name, id_prefix):
        tmp = []
        for i in df.index:
            tmp.append("{}{:0>8d}".format(id_prefix, i))
        df[col_name] = tmp


if __name__ == '__main__':
    graph = CultureGraph()
    graph.read_poetry_data()