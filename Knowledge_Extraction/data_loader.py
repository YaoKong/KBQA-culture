import json
from random import choice
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

class JSONDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.dataset = []
        with open(path, encoding='utf8') as f:
            for line in f:
                self.dataset.append(json.loads(line))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        spo_list = self.dataset[idx]['spo_list']
        return text, spo_list

def collate(batch):
    tmp = list(zip(*batch))
    text = tmp[-2]
    spo_list = tmp[-1]

    return text, spo_list

def prepare_dataset(config):
    data = JSONDataset(config.train_path)
    dev_data = JSONDataset(config.dev_path)

    train_size = int(config.train_proportion * len(data))
    test_size = len(data) - train_size

    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

    train_iter = DataLoader(train_data, batch_size=config.batch_size, collate_fn=collate, shuffle=True, drop_last=False)
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size, collate_fn=collate, shuffle=True, drop_last=False)
    test_iter = DataLoader(test_data, batch_size=config.batch_size, collate_fn=collate, shuffle=True, drop_last=False)
    return train_iter, dev_iter, test_iter

def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

class data_generator:
    def __init__(self, data, config):
        self.data = data
        self.batch_size = config.batch_size
        self.tokenizer = config.tokenizer
        self.max_len = config.max_len
        self.rel2ids = config.rel2ids
        self.num_relations = config.num_relations
        self.steps = len(self.data)
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def __len__(self):
        return self.steps
    def __iter__(self):
        tokens_batch, mask_batch, sub_heads_batch, sub_tails_batch, head2tails, sub_lens, obj_heads_batch, obj_tails_batch = [], [], [], [], [], [], [], []
        for texts, spo_list in self.data:
            tokens = self.tokenizer(texts, padding=True).data
            for idx in range(len(texts)):
                token = tokens['input_ids'][idx]
                mask = tokens['attention_mask'][idx]

                triples = spo_list[idx]
                token_len = len(token)

                s2ro_map = {}
                for triple_i in triples:
                    triple = (self.tokenizer(triple_i['subject'], add_special_tokens=False)['input_ids'],
                              self.rel2ids.get(triple_i['predicate']),
                              self.tokenizer(triple_i['object'], add_special_tokens=False)['input_ids'])
                    sub_head_idx = find_head_idx(token, triple[0])
                    obj_head_idx = find_head_idx(token, triple[2])
                    if sub_head_idx != -1 and obj_head_idx != -1:
                        sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                        if sub not in s2ro_map:
                            s2ro_map[sub] = []
                        s2ro_map[sub].append((obj_head_idx,
                                           obj_head_idx + len(triple[2]) - 1,
                                           triple[1]))

                if s2ro_map:
                    sub_heads, sub_tails = torch.zeros(token_len), torch.zeros(token_len)
                    for s in s2ro_map:
                        sub_heads[s[0]] = 1
                        sub_tails[s[1]] = 1

                    sub_head, sub_tail = choice(list(s2ro_map.keys()))
                    head2tail = torch.zeros(token_len)
                    head2tail[sub_head : sub_tail + 1] = 1
                    sub_len = torch.tensor([sub_tail - sub_head + 1], dtype=torch.float)

                    obj_heads, obj_tails = torch.zeros((token_len, self.num_relations)), torch.zeros((token_len, self.num_relations))
                    for ro in s2ro_map.get((sub_head, sub_tail), []):
                        obj_heads[ro[0]][ro[2] - 1] = 1
                        obj_tails[ro[1]][ro[2] - 1] = 1

                    tokens_batch.append(token)
                    mask_batch.append(mask)
                    sub_heads_batch.append(sub_heads)
                    sub_tails_batch.append(sub_tails)
                    head2tails.append(head2tail)
                    sub_lens.append(sub_len)
                    obj_heads_batch.append(obj_heads)
                    obj_tails_batch.append(obj_tails)
                else:
                    print("test!")

            tokens_batch = torch.tensor(tokens_batch).to(self.device)
            mask_batch = torch.tensor(mask_batch).to(self.device)
            sub_heads_batch = torch.stack(sub_heads_batch).to(self.device)
            sub_tails_batch = torch.stack(sub_tails_batch).to(self.device)
            head2tails = torch.stack(head2tails).to(self.device)
            sub_lens = torch.stack(sub_lens).to(self.device)
            obj_heads_batch = torch.stack(obj_heads_batch).to(self.device)
            obj_tails_batch = torch.stack(obj_tails_batch).to(self.device)
            yield {
               'token_ids': tokens_batch,
               'mask': mask_batch,
               'head2tails': head2tails,
               'sub_lens': sub_lens
           }, {
               'sub_heads': sub_heads_batch,
               'sub_tails': sub_tails_batch,
               'obj_heads': obj_heads_batch,
               'obj_tails': obj_tails_batch
           }

            tokens_batch, mask_batch, sub_heads_batch, sub_tails_batch, head2tails, sub_lens, obj_heads_batch, obj_tails_batch = [], [], [], [], [], [], [], []

if __name__ == "__main__":
    config = Config()
    train_iter, dev_iter, test_iter = prepare_dataset(config)
    dg = data_generator(dev_iter, config)
    idx = 0
    for inputs, labels in dg:
        print(idx)
        idx += 1
    # for text, triple in (dev_iter):
    #     print(text, triple)