import os
from collections import defaultdict

from py2neo import Graph, Node
import pandas as pd
import hanlp
import pickle
import torch
import gc
from GPUtil import showUtilization as gpu_usage

class CultureGraph:
    def __init__(self):
        cur_sep = os.path.sep
        dir = cur_sep.join(os.path.abspath(__file__).split(cur_sep)[:-1])
        self.data_path = os.path.join(dir, 'data')
        self.graph = Graph("http://localhost:7474", auth=("neo4j", "admin"), name="neo4j")

    def create_nodes(self, nodes):
        count = 0
        node_num = len(nodes)
        for node in nodes:
            self.graph.create(node)
            count +=1
            if count % 100 == 0:
                print("本次已生成 %s / %s个结点" % (count, node_num))

    def create_poetry_nodes(self, character, poetry, location):
        chara_nodes = []
        poet_nodes = []
        loc_nodes = []

        for chara in character:
            chara_nodes.append(Node("人物", description="", masterpiece=[],   # 介绍 代表作
                                    alias=[], name=chara, dynasty="",   # 别名 姓名 朝代
                               homeplace="", identity=[]))   # 出生地 身份

        for poet in poetry:
            poet_nodes.append(Node("诗词", author=poet["作者"], content=poet["内容"],
                                   dynasty=poet["朝代"], title=poet["题目"]))

        for loc in location:
            loc_nodes.append(Node("地区", description="", alias=[], name=loc))


        self.create_nodes(chara_nodes)
        print("人物结点生成完毕，本次生成", len(chara_nodes), "个结点")
        self.create_nodes(poet_nodes)
        print("诗词结点生成完毕，本次生成", len(poet_nodes), "个结点")
        self.create_nodes(loc_nodes)
        print("地区结点生成完毕，本次生成", len(loc_nodes), "个结点")


    '''创建实体关联边'''
    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        count = 0
        for edge in edges:
            edge = edge.split('=>')
            p = edge[0]
            q = edge[1]

            p = p[1:-1].split(',')
            q = q[1:-1].split(',')

            start_condition = ""
            for _p in p:
                start_condition += "p.%s and " % _p
            start_condition = start_condition[:-4]

            end_condition = ""
            for _q in q:
                end_condition += "q.%s and " % _q
            end_condition = end_condition[:-4]

            query = "match(p:%s),(q:%s) where %s and %s create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, start_condition, end_condition, rel_type, rel_name)
            try:
                self.graph.run(query)
                if count % 100:
                    print("本次生成(%s / %s)条边" % (count, len(edges)))
            except Exception as e:
                print(e)

    def create_poetry_rels(self, rels_refer_pos, rels_refer_char, rels_indite):
        self.create_relationship("诗词", "地区", rels_refer_pos, "refer_pos", "提到")
        self.create_relationship("诗词", "人物", rels_refer_char, "refer_char", "提到")
        self.create_relationship("人物", "诗词", rels_indite, "refer_pos", "创作")

    def read_poetry_data(self):
        poetry_path = os.path.join(self.data_path, 'Poetry-master')
        # calligraph = []  # 书法
        character = []  # 人物
        poetry = []  # 诗词
        location = set()  # 地区
        # scenic = []  # 景点

        rels_refer_pos = set()  #提到地区
        rels_refer_char = set() #提到人物
        rels_indite = set()  #创作

        export_path = os.path.join(self.data_path, 'pickle')
        if not os.path.exists(export_path):
            for filename in os.listdir(poetry_path):
                if "csv" in filename:
                    df = pd.read_csv(os.path.join(poetry_path, filename))
                    poet = defaultdict(list)
                    indite = defaultdict(list)  #创作出

                    poet = df.to_dict("records", into=poet)
                    poetry.extend(poet)

                    # 注意，不考虑同名同姓
                    chara = df["作者"].drop_duplicates().to_list()
                    character.extend(chara)

                    indite = df.loc[:, ["作者", "题目"]].to_dict("records", into=indite)
                    for i in indite:
                        rels_indite.add("(name='%s')=>(author='%s',title='%s')" % (i["作者"], i["作者"], i["题目"]))

            character = set(character)

            # 生成“提到”关系
            ''' 由于生成很慢，注释掉
            hanlp.pretrained.mtl.ALL  # MTL多任务，具体任务见模型名称，语种见名称最后一个字段或相应语料库
            HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
            count = 0
            batch_size = 128
            for poet in poetry:
                count += 1
                content = poet["内容"][:3000] if len(poet["内容"]) > 3000 else poet["内容"]
                result = HanLP(content, tasks='ner')["ner/msra"]

                for i in result:
                    if i[1] == "LOCATION":
                        location.add(i[0])
                        rels_refer_pos.add("(author='%s',title='%s')=>(name='%s')" % (poet["作者"], poet["题目"], i[0]))
                    elif i[1] == "PERSON":
                        if len(i[0]) > 1: # 防止加入代词
                            character.add(i[0])
                            rels_refer_char.add("(author='%s',title='%s')=>(name='%s')" % (poet["作者"], poet["题目"], i[0]))

                if count % batch_size == 0:
                    print("已处理(%d / %d)首诗词" % (count, len(poetry)))
                    gc.collect()
                    torch.cuda.empty_cache()
            '''
            os.makedirs(export_path)
            self.save_all_data(character, poetry, location, rels_refer_pos, rels_refer_char, rels_indite)
        else:
            character, poetry, location, rels_refer_pos, rels_refer_char, rels_indite = self.load_all_data()


        self.create_poetry_nodes(character, poetry, location)
        self.create_poetry_rels(rels_refer_pos, rels_refer_char, rels_indite)

    def save_all_data(self, character, poetry, location, rels_refer_pos, rels_refer_char, rels_indite):
        self.save_data(character, "character")
        self.save_data(poetry, "poetry")
        self.save_data(location, "location")
        self.save_data(rels_refer_pos, "rels_refer_pos")
        self.save_data(rels_refer_char, "rels_refer_char")
        self.save_data(rels_indite, "rels_indite")

    def save_data(self, data, file_name):
        file_name = os.path.join(self.data_path, "pickle", file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

    def load_all_data(self):
        character = self.load_data("character")
        poetry = self.load_data("poetry")
        location = self.load_data("location")
        rels_refer_pos = self.load_data("rels_refer_pos")
        rels_refer_char = self.load_data("rels_refer_char")
        rels_indite = self.load_data("rels_indite")
        return character, poetry, location, rels_refer_pos, rels_refer_char, rels_indite

    def load_data(self, file_name):
        file_name = os.path.join(self.data_path, "pickle", file_name)
        with open(file_name, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    graph = CultureGraph()
    graph.read_poetry_data()