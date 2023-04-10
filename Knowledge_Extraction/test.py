import torch
import pandas as pd
from tqdm import tqdm

import data_loader
def test(model, test_iter, config):
    model.eval()
    test_data = data_loader.data_generator(test_iter, config)
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple']).fillna(0)

    idx = 0
    for inputs, labels in tqdm(test_data):
        # idx += 1
        # if idx > 1000:
        #     break
        logist = model(**inputs)

        pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
        pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])

        sub_heads = convert_score_to_zero_one(labels['sub_heads'])
        sub_tails = convert_score_to_zero_one(labels['sub_tails'])
        batch_size = inputs['token_ids'].shape[0]

        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])
        pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])

        for batch_index in range(batch_size):
            pred_subs = extract_sub(pred_sub_heads[batch_index].squeeze(), pred_sub_tails[batch_index].squeeze())
            true_subs = extract_sub(sub_heads[batch_index].squeeze(), sub_tails[batch_index].squeeze())

            pred_objs = extract_obj_and_rel(pred_obj_heads[batch_index], pred_obj_tails[batch_index])
            true_objs = extract_obj_and_rel(obj_heads[batch_index], obj_tails[batch_index])

            df['PRED']['sub'] += len(pred_subs)
            df['REAL']['sub'] += len(true_subs)
            for true_sub in true_subs:
                if true_sub in pred_subs:
                    df['TP']['sub'] += 1

            df['PRED']['triple'] += len(pred_objs)
            df['REAL']['triple'] += len(true_objs)
            for true_obj in true_objs:
                if true_obj in pred_objs:
                    df['TP']['triple'] += 1

    df.loc['sub', 'p'] = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    df.loc['sub', 'r'] = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
    df.loc['sub', 'f1'] = 2 * df['p']['sub'] * df['r']['sub'] / (df['p']['sub'] + df['r']['sub'] + 1e-9)

    sub_precision = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    sub_recall = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
    sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall + 1e-9)

    df.loc['triple', 'p'] = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    df.loc['triple', 'r'] = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    df.loc['triple', 'f1'] = 2 * df['p']['triple'] * df['r']['triple'] / (
            df['p']['triple'] + df['r']['triple'] + 1e-9)

    triple_precision = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    triple_recall = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    triple_f1 = 2 * triple_precision * triple_recall / (
            triple_precision + triple_recall + 1e-9)

    return sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df


def extract_sub(pred_sub_heads, pred_sub_tails):
    subs = []
    heads = torch.arange(0, len(pred_sub_heads))[pred_sub_heads == 1]
    tails = torch.arange(0, len(pred_sub_tails))[pred_sub_tails == 1]

    for head, tail in zip(heads, tails):
        if tail >= head:
            subs.append((head.item(), tail.item()))
    return subs


def extract_obj_and_rel(obj_heads, obj_tails):
    obj_heads = obj_heads.T
    obj_tails = obj_tails.T
    rel_count = obj_heads.shape[0]
    obj_and_rels = []  # [(rel_index,strart_index,end_index),(rel_index,strart_index,end_index)]

    for rel_index in range(rel_count):
        obj_head = obj_heads[rel_index]
        obj_tail = obj_tails[rel_index]

        objs = extract_sub(obj_head, obj_tail)
        if objs:
            for obj in objs:
                start_index, end_index = obj
                obj_and_rels.append((rel_index, start_index, end_index))
    return obj_and_rels


def convert_score_to_zero_one(tensor):
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor

