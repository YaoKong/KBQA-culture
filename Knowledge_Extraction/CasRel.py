import torch
import torch.nn as nn

from transformers import BertModel
class CasRel(nn.Module):
    def __init__(self, config):
        super(CasRel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_name)
        self.sub_heads_linear = nn.Linear(self.config.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.config.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.config.bert_dim, self.config.num_relations)
        self.obj_tails_linear = nn.Linear(self.config.bert_dim, self.config.num_relations)

        self.alpha = 0.25
        self.gamma = 2

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, head2tails, sub_lens, encoded_text):
        head2tail_mapping = head2tails.unsqueeze(1)
        # print("h2t shape{}, text shape{}".format(head2tail_mapping.shape, encoded_text.shape))
        sub = torch.matmul(head2tail_mapping, encoded_text)
        sub_lens = sub_lens.unsqueeze(1)
        sub = sub / sub_lens
        encoded_text = sub + encoded_text
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pred_obj_tails
    def forward(self, token_ids, mask, head2tails, sub_lens):
        encoded_text = self.get_encoded_text(token_ids, mask)
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)

        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(head2tails, sub_lens,
                                                                       encoded_text)

        return {
            "pred_sub_heads": pred_sub_heads,
            "pred_sub_tails": pred_sub_tails,
            "pred_obj_heads": pred_obj_heads,
            "pred_obj_tails": pre_obj_tails,
            'mask': mask
        }

    def loss_fn(self, pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails, mask, sub_heads,
                     sub_tails, obj_heads, obj_tails):
        def calc_loss(pred, label, mask):
            pred = pred.squeeze(-1)
            loss = nn.functional.binary_cross_entropy(pred, label, reduction='none')
            if loss.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            return torch.sum(loss * mask) / torch.sum(mask)

            # count = torch.sum(mask)
            # logist = pred.view(-1)
            # label = label.view(-1)
            # mask = mask.view(-1)
            #
            # alpha_factor = torch.where(torch.eq(label, 1), 1 - self.alpha, self.alpha)
            # focal_weight = torch.where(torch.eq(label, 1), 1 - logist, logist)
            #
            # loss = -(torch.log(logist) * label + torch.log(1 - logist) * (1 - label)) * mask
            # return torch.sum(focal_weight * loss) / count

        rel_count = obj_heads.shape[-1]
        rel_mask = mask.unsqueeze(-1).repeat(1, 1, rel_count)

        return calc_loss(pred_sub_heads, sub_heads, mask) + \
            calc_loss(pred_sub_tails, sub_tails, mask) + \
            calc_loss(pred_obj_heads, obj_heads, rel_mask) + \
            calc_loss(pred_obj_tails, obj_tails, rel_mask)