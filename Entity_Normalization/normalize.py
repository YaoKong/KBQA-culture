import re

import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from build_graph import generate_id_column

def edit_distance(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]

def Jaccrad(terms_model,reference):
    grams_reference = set(reference)
    grams_model = set(terms_model)
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp
    jaccard_coefficient = float(temp / fenmu)
    return jaccard_coefficient

def get_avg_sim(a, b):
    td = Jaccrad(a, b)
    std = edit_distance(a, b) / max(len(a), len(b))
    fy = 1 - std
    return (td + fy) / 2

def link_characters():
    '''
    poetry_author, calligrapher, character, indite_rels, rels_refer_char
    :return:
    '''
    poetry_author = pd.read_csv("../data/character.csv")
    calligrapher = pd.read_csv("../Knowledge_Extraction/results/calligrapher.csv")
    character = pd.read_csv("../Knowledge_Extraction/results/character.csv")

    character = pd.merge(character, poetry_author, how='left', left_on='name', right_on='作者')

    for index, row in character.iterrows():
        if pd.notna(row['朝代']):
            character.loc[index, 'dynastry'] = row['朝代']
    character = pd.merge(character, calligrapher, how='outer').drop_duplicates(subset=['name'])

    generate_id_column(character, 'newcid', 'newcid')
    indite_rels = pd.read_csv("../data/indite_rels.csv")
    rels_refer_char = pd.read_csv("../data/rels_refer_char.csv")
    new_indite = pd.merge(indite_rels, character, how='left', on='cid:ID').dropna(subset=['newcid'])
    new_refer = pd.merge(rels_refer_char, character, how='left', left_on=':END_ID', right_on='cid:ID').dropna(subset=['newcid'])


    character = pd.concat([character.iloc[:,:7], character.iloc[:,-1]], axis=1)
    new_indite = new_indite.loc[:, ['newcid', ':TYPE', 'pid:ID']]
    new_refer = new_refer.loc[:, [':START_ID', ':TYPE', 'newcid']]
    return character, new_indite, new_refer

def link_location():
    '''
    area, loc_rel, location/KE, location/data, rels_refer_pos
    :return:
    '''

    # 由于poetry_location有重复，导致rels_refer_pos也有，先给这俩换个新ID
    poetry_location = pd.read_csv("../data/location.csv").drop_duplicates(subset='名字')
    generate_id_column(poetry_location, 'plid', 'plid')
    rels_refer_pos = pd.read_csv("../data/rels_refer_pos.csv")
    rels_refer_pos = pd.merge(rels_refer_pos, poetry_location, how='left', left_on='地点', right_on='名字')

    location = pd.read_csv("../Knowledge_Extraction/results/location.csv").fillna('')


    generate_id_column(location, 'locid', 'locid')
    poetry_location['locid'] = None

    # 保留location里有的poetry_location
    for index, row in poetry_location.iterrows():
        if row['名字'].find('?') != -1:   # 由于数据有点问题，需要去掉正则符,包括但不限于括号和问号
            continue
        if location[location['name'].str.contains(row['名字'])].empty == False:
            poetry_location.loc[index, 'locid'] = row['locid']
        else:
            df = location[location['alias'].str.contains(row['名字'])]
            if df.empty == False:
                sim = 0
                locid = ''
                for i, r in df.iterrows():
                    tmp_sim = get_avg_sim(location.loc[i, 'alias'], row['名字'])
                    if tmp_sim >= 0.1 and tmp_sim > sim:
                        sim = tmp_sim
                        locid = r['locid']
                if sim > 0:
                    poetry_location.loc[index, 'locid'] = locid

    poetry_location = pd.merge(location, poetry_location, how='inner', on='locid')

    # 丢弃rels_refer_pos中和poetry_location无交集的
    rels_refer_pos = pd.merge(rels_refer_pos, poetry_location, how='inner', on='plid')
    rels_refer_pos = rels_refer_pos.loc[:, [':START_ID', ':TYPE', 'locid']]

    # 开始处理景点
    area = pd.read_csv("../Knowledge_Extraction/results/area.csv")
    generate_id_column(area, 'areaid', 'areaid')
    area2location = dict()
    cityList = ['香港', '澳门', '北京市', '上海市']

    area['locid'] = None
    for index, row in area.iterrows():
        area.loc[index, 'description'] = re.sub(r'\s+', '', row['description'])   #去除换行，防止导出错误，若有()也要去掉或者替换成（）
        if row['province'] in cityList:
            city_name = row['province']
        else:
            city_name = row['city']

        if pd.isna(city_name):
            continue
        if area2location.get(city_name) is None:
            sim = 0
            locid = ''
            for i, r in location.iterrows():
                if pd.isna(r['name']):
                    continue
                tmp_sim = get_avg_sim(r['name'], city_name)
                if tmp_sim >= 0.2 and tmp_sim > sim:
                    sim = tmp_sim
                    locid = r['locid']
            if sim > 0:
                area.loc[index, 'locid'] = locid
            area2location[city_name] = locid
        else:
            area.loc[index, 'locid'] = area2location[city_name]

    area.dropna(subset=['locid'], inplace=True)
    area_locate_rel = area.loc[:, ['areaid', 'locid']]
    area_locate_rel['：TYPE'] = '位于'
    new_col = area_locate_rel.pop('locid')
    area_locate_rel.insert(2, 'locid', new_col)


    return location, area, rels_refer_pos, area_locate_rel

def link_dataset():
    character, poetry_indite, refer_char = link_characters()
    location, area, rels_refer_pos, area_locate_rel = link_location()

    '''
    born, call_indite, calligraphy, visit_area, visit_location, work_indite
    '''
    born_rels = pd.read_csv("../Knowledge_Extraction/results/born_rels.csv")
    born_rels = pd.merge(born_rels, character, how='left', left_on='名字', right_on='name').dropna(subset=['newcid'])
    born_rels['locid'] = None
    for index, row in born_rels.iterrows():
        df = location[location['name'].str.contains(row['地区'])]
        if df.empty:
            sim = 0
            for i, r in df.iterrows():
                tmp_sim = get_avg_sim(location.loc[i, 'name'], row['名字'])
                if tmp_sim >= 0.1 and tmp_sim > sim:
                    sim = tmp_sim
                    locid = r['locid']
            if sim > 0:
                born_rels.loc[index, 'locid'] = locid
        else:
            born_rels.loc[index, 'locid'] = df.iloc[0, 6]
    born_rels = born_rels.dropna(subset=['locid']).dropna(subset=['locid'])
    born_rels[':TYPE'] = '出生于'
    born_rels = born_rels.loc[:, ['newcid', ':TYPE', 'locid']]


    calligraphy = pd.read_csv("../Knowledge_Extraction/results/calligraphy.csv")
    generate_id_column(calligraphy, 'callid', 'callid')
    calligraphy = pd.merge(calligraphy, character, how='left', left_on='calligrapher', right_on='name')
    calligraphy[':TYPE'] = '创作出：书法'
    call_indite = calligraphy.loc[:, ['newcid', ':TYPE', 'callid']]


    visit_area = pd.read_csv("../Knowledge_Extraction/results/visit_area_rel.csv")
    visit_area = pd.merge(visit_area, area, how='left', left_on='景区', right_on='name')
    visit_area = pd.merge(visit_area, character, how='left', left_on='人物', right_on='name').dropna(subset=['areaid'])

    visit_area = visit_area.loc[:, ['newcid', ':TYPE', 'areaid']]

    visit_location = pd.read_csv("../Knowledge_Extraction/results/visit_location_rel.csv")
    visit_location = pd.merge(visit_location, character, how='left', left_on='人物', right_on='name')
    visit_location['locid'] = None
    for index, row in visit_location.iterrows():
        for i, r in location.iterrows():
            tmp_sim = get_avg_sim(location.loc[i, 'name'], row['地点'])
            if tmp_sim >= 0.2 and tmp_sim > sim:
                sim = tmp_sim
                locid = r['locid']
        if sim > 0:
            visit_location.loc[index, 'locid'] = locid
    visit_location.dropna(subset=['locid'], inplace=True)
    visit_location = visit_location.loc[:, ['newcid', ':TYPE', 'locid']]


    rels_list = [poetry_indite, refer_char, rels_refer_pos,
                 area_locate_rel, born_rels, call_indite, visit_area, visit_location]
    for rel in rels_list:
        rel.columns = [':START_ID', ':TYPE', ':END_ID']

    total_rels = pd.concat(rels_list, axis=0, ignore_index=True)
    total_rels.to_csv("total_rels.csv", index=False)

    character[':LABEL'] = '人物'
    character.rename(columns={'newcid': 'newcid:ID'}, inplace=True)
    character.to_csv("character.csv", index=False)

    location[':LABEL'] = '地区'
    location.rename(columns={'locid': 'locid:ID'}, inplace=True)
    location.drop(labels=['city', 'urls'], axis=1).to_csv("location.csv", index=False)

    area[':LABEL'] = '景点'
    area.rename(columns={'areaid': 'areaid:ID'}, inplace=True)
    area.drop(labels=['alias', 'locid'], axis=1).to_csv("area.csv", index=False)

    calligraphy = calligraphy.iloc[:, :8]
    calligraphy.columns = ['name', 'style', 'calligrapher', 'url', 'carrier', 'dynasty', 'content', 'callid:ID']
    calligraphy[':LABEL'] = '书法'
    calligraphy.to_csv("calligraphy.csv", index=False)


if __name__ == "__main__":
    link_dataset()