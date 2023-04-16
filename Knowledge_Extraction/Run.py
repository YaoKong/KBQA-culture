import json
import os

import numpy as np
import pandas as pd
import torch

from Knowledge_Extraction.CasRel import CasRel
from config import Config
from data_loader import find_head_idx

def predict(texts,  config, model, device, h_bar=0.5, t_bar=0.5):
    '''
    :param texts: 必须为列表等可迭代类型
    :param config:
    :param model:
    :param device:
    :param h_bar:
    :param t_bar:
    :return:
    '''
    # texts = ['湖南展翼商贸有限公司成立于2011年，公司现阶段主要经营3C电子产品：计算机（Computer）、通讯（Communication）和消费电子产品（Consumer Electronic）',
    #          '就是这一次，导演尚敬从中看到了闫妮的喜剧才华，请她客串了《炊事班的故事》，也才有了后来《武林外传》中的佟湘玉']


    tokenizer = config.tokenizer
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=512).data

    HanLP = config.HanLP
    ner_result = HanLP(texts, tasks='ner/msra')['ner/msra']

    tokens_batch, mask_batch, head2tails, sub_lens = [], [], [], []
    for idx in range(len(texts)):
        token = tokens['input_ids'][idx]
        mask = tokens['attention_mask'][idx]
        for sub in ner_result[idx]:
            sub = tokenizer(sub[0], add_special_tokens=False)['input_ids']
            sub_head = find_head_idx(token, sub)
            if sub_head == -1:
                continue
            sub_tail= sub_head + len(sub) - 1
            head2tail = torch.zeros(len(token))
            head2tail[sub_head: sub_tail + 1] = 1
            sub_len = torch.tensor([sub_tail - sub_head + 1], dtype=torch.float)

            tokens_batch.append(token)
            mask_batch.append(mask)
            head2tails.append(head2tail)
            sub_lens.append(sub_len)

    tokens_batch = torch.tensor(tokens_batch).to(device)
    mask_batch = torch.tensor(mask_batch).to(device)
    head2tails = torch.stack(head2tails).to(device)
    sub_lens = torch.stack(sub_lens).to(device)

    batch_x = {
        'token_ids': tokens_batch,
        'mask': mask_batch,
        'head2tails': head2tails,
        'sub_lens': sub_lens
    }

    triples = []
    with torch.no_grad():
        logist = model(**batch_x)

        pred_sub_heads = (logist['pred_sub_heads'])
        pred_sub_tails = (logist['pred_sub_tails'])


        batch_size = batch_x['token_ids'].shape[0]
        pred_obj_heads = (logist['pred_obj_heads'])
        pred_obj_tails = (logist['pred_obj_tails'])


        for batch_index in range(batch_size):
            sub_heads = torch.where(pred_sub_heads[batch_index] > h_bar)[0]
            sub_tails = torch.where(pred_sub_tails[batch_index] > t_bar)[0]
            subjects = []
            for sub_head in sub_heads:
                sub_tail = sub_tails[sub_tails >= sub_head]
                if len(sub_tail) > 0:
                    sub_tail = sub_tail[0]
                    subject = ''.join(config.tokenizer.decode(batch_x['token_ids'][batch_index][sub_head: sub_tail + 1]).split())
                    subjects.append((subject, sub_head, sub_tail))
            if subjects:
                triple_list = []
                for subject_idx, subject in enumerate(subjects):
                    sub = subject[0]
                    obj_heads = torch.where(pred_obj_heads[batch_index] > h_bar)
                    obj_tails = torch.where(pred_obj_tails[batch_index] > t_bar)
                    for obj_head, rel_head in zip(*obj_heads):
                        for obj_tail, rel_tail in zip(*obj_tails):
                            if obj_head <= obj_tail and rel_head == rel_tail:
                                rel = config.getIdx2Rel(int(rel_head + 1))
                                obj = ''.join(config.tokenizer.decode(batch_x['token_ids'][batch_index][obj_head: obj_tail + 1]).split())
                                triple_list.append((sub, rel, obj))
                                break

                triple_set = set()
                for s, r, o in triple_list:
                    triple_set.add((s, r, o))
                pred_list = list(triple_set)

            else:
                pred_list = []

            if len(pred_list) > 0:
                triples.append(pred_list)

    triple_set = set()
    for triple in triples:
        for t in triple:
            triple_set.add(t)
    return triple_set

def extract_knowledge(config, device, model):

    locate_rels, born_rels, work_indite_rels, call_indite_rels = [], [], [], []
    data_path = "../spider_tools/"

    baike_path = data_path + 'baike'
    baike_path = os.path.join(baike_path, os.listdir(baike_path)[0])
    baike = pd.read_csv(baike_path).dropna(subset=['name'])
    for row in baike.itertuples():
        if pd.isnull(row.region) != True:
            born_rels.append([row.name, '出生于', row.region])
        if pd.isnull(row.masterpiece) != True:
            masterpiece = config.split_to_list(row.masterpiece)
            for m in masterpiece:
                work_indite_rels.append([row.name, '创作出', m])

    born_rels = pd.DataFrame(born_rels, columns=["名字", ":TYPE", "地区"])
    work_indite_rels = pd.DataFrame(work_indite_rels, columns=["名字", ":TYPE", "作品"])

    calligraphy_path = data_path + 'calligraphy/'
    calligraphy = pd.read_csv(calligraphy_path + 'calligraphy.csv').dropna(subset=['name']) # 提取创作出, ''书法家个人信息', '书法朝代(人物绑定)'
    calligrapher = pd.read_csv(calligraphy_path + 'calligrapher.csv')   # 提取创作出, ''书法家个人信息', '书法朝代(人物绑定)'
    for index, row in calligrapher.iterrows():
        triples = predict([row['summary']], config, model, device)
        alias, dynasty, region = [], [], []
        for triple in triples:
            if triple[0] == row['name'] and triple[1] == '朝代':
                if len(triple[2]) <= 2:
                    dynasty.append(triple[2])
            elif triple[0] == row['name'] and triple[1] == '出生地':
                region.append(triple[2])
            elif triple[0] == row['name'] and (triple[1] == '字' or triple[1] == '号'):
                alias.append(triple[2])

        calligrapher.loc[index, 'identity'] = '书法家'
        calligrapher.loc[index, 'alias'] = '，'.join(alias)
        calligrapher.loc[index, 'dynasty'] = '，'.join(dynasty)
        calligrapher.loc[index, 'region'] = '，'.join(region)

    for index, row in calligraphy.iterrows():
        call_indite_rels.append([row['calligrapher'], '创作出：书法', row['name']])
    call_indite_rels = pd.DataFrame(call_indite_rels, columns=["书法家", ":TYPE", "书法"])
    tmp = pd.merge(calligraphy, calligrapher, left_on='calligrapher', right_on='name')
    calligraphy = tmp.loc[:, ['name_x', 'style', 'calligrapher', 'url', 'carrier', 'dynasty_y', 'content']]
    calligraphy.columns = ['name', 'style', 'calligrapher', 'url', 'carrier', 'dynasty', 'content']

    location_path = data_path + 'location'
    location_path = os.path.join(location_path, os.listdir(location_path)[0])
    location = pd.read_csv(location_path) # 位于（？）
    locDict = pd.read_csv(data_path + 'locDict/locDict.csv')
    locate_rels = []
    celebrity_list = ['苏轼', '颜真卿', '韩愈', '柳宗元', '欧阳修', '王安石', '李白', '杜甫'] # 重点关照的名人
    visit_location_rel, visit_area_rel = [], []
    cityList = ['香港', '澳门', '北京市', '上海市']
    for index, row in location.iterrows():
        if row['province'] in cityList:
            location_name = row['province']
        else:
            location_name = row['city']
        locate_rels.append([row['name'], '位于', location_name])

        for celebrity in celebrity_list:
            if celebrity in row['description']:
                visit_location_rel.append([celebrity, '到过', location_name])
                visit_area_rel.append([celebrity, '到过', row['name']])

    locate_rels = pd.DataFrame(locate_rels, columns=["景区", ":TYPE", "地点"])
    visit_location_rel = pd.DataFrame(visit_location_rel, columns=["人物", ":TYPE", "地点"]).drop_duplicates()
    visit_area_rel = pd.DataFrame(visit_area_rel, columns=["人物", ":TYPE", "景区"]).drop_duplicates()

    # baike.to_csv(os.path.join(config.result_dir, 'character.csv'), index=False) # 人物
    # work_indite_rels.to_csv(os.path.join(config.result_dir, 'work_indite_rels.csv'), index=False)   # 人物创作
    # born_rels.to_csv(os.path.join(config.result_dir, 'born_rels.csv'), index=False)  # 出生于
    # calligraphy.to_csv(os.path.join(config.result_dir, 'calligraphy.csv'), index=False) # 书法
    # calligrapher.to_csv(os.path.join(config.result_dir, 'calligrapher.csv'), index=False)   # 书法家
    call_indite_rels.to_csv(os.path.join(config.result_dir, 'call_indite_rels.csv'), index=False) # 书法创作
    # location.to_csv(os.path.join(config.result_dir, 'area.csv'), index=False)   # 景区
    # locate_rels.to_csv(os.path.join(config.result_dir, 'locate_rels.csv'), index=False) # 位于
    # locDict.to_csv(os.path.join(config.result_dir, 'location.csv'), index=False)    # 地点
    # visit_location_rel.to_csv(os.path.join(config.result_dir, 'visit_location_rel.csv'), index=False)  # 人物到过地点
    # visit_area_rel.to_csv(os.path.join(config.result_dir, 'visit_area_rel.csv'), index=False)  # 人物到过景区
    pass

if __name__ == "__main__":
    config = Config(ner_flag=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = CasRel(config).to(device)
    model.load_state_dict(torch.load('model.pt'))
    extract_knowledge(config, device, model)
