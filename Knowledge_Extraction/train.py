import torch
from torch.optim import AdamW
from CasRel import CasRel
from test import test
from config import Config
import data_loader as dl
from tqdm import tqdm
def load_model(config):
    model = CasRel(config)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    # prepare optimzier
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=10e-8)

    return model, optimizer

def train(model, optimizer, config, train_iter, dev_iter):
    epochs = config.max_epoch
    train_data = dl.data_generator(train_iter, config)
    best_triple_f1 = 0
    step = 0
    for epoch in range(epochs):
        for inputs, labels in tqdm(train_data):
            model.train()
            logist = model(**inputs)
            loss = model.loss_fn(**logist, **labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # 每500步做一次验证
            step += 1
            if step % 200 == 0:
                sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df = test(model,
                                                                                                         dev_iter,
                                                                                                         config)
                if triple_f1 > best_triple_f1:
                    best_triple_f1 = triple_f1
                    # 直接保存模型
                    torch.save(model, 'best_f1.pth')
                    print(
                        'epoch:{},step:{}/{},sub_precision:{:.4f}, sub_recall:{:.4f}, sub_f1:{:.4f}, triple_precision:{:.4f}, triple_recall:{:.4f}, triple_f1:{:.4f},train loss:{:.4f}'.format(
                            epoch, step, train_data.steps, sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1,
                            loss.item()))
                    print(df)

if __name__ == "__main__":
    config = Config()
    model, optimizer = load_model(config)
    train_iter, dev_iter, test_iter = dl.prepare_dataset(config)
    train(model, optimizer, config, train_iter, dev_iter)