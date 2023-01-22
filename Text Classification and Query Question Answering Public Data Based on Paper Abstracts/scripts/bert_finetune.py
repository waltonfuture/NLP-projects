
import codecs
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
import re
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup


# preprocess the data
train_data = pd.read_csv('../data/train.csv', sep=',')
test_data = pd.read_csv('../data/test.csv', sep=',')
train_data = train_data[~train_data['Topic(Label)'].isnull()]
train_data['Topic(Label)'], lbl = pd.factorize(train_data['Topic(Label)'])

train_data['Title'] = train_data['Title'].apply(lambda x: x.strip())
train_data['Abstract'] = train_data['Abstract'].fillna(
    '').apply(lambda x: x.strip())
train_data['text'] = train_data['Title'] + ' ' + train_data['Abstract']
train_data['text'] = train_data['text'].str.lower()

test_data['Title'] = test_data['Title'].apply(lambda x: x.strip())
test_data['Abstract'] = test_data['Abstract'].fillna('').apply(lambda x: x.strip())
test_data['text'] = test_data['Title'] + ' ' + test_data['Abstract']
test_data['text'] = test_data['text'].str.lower()

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
train_encoding = tokenizer(train_data['text'].to_list()[:], truncation=True, padding=True, max_length=512)
test_encoding = tokenizer(test_data['text'].to_list()[:], truncation=True, padding=True, max_length=512)

# build datasets

class QDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = QDataset(train_encoding, train_data['Topic(Label)'].to_list())
test_dataset = QDataset(test_encoding, [0] * len(test_data))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# build the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optim = AdamW(model.parameters(), lr=1e-5)
total_steps = len(train_loader) * 1

def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        iter_num += 1
        if(iter_num % 100 == 0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" %
                  (epoch, iter_num, loss.item(), iter_num/total_iter*100))

    print("Epoch: %d, Average training loss: %.4f" %
          (epoch, total_train_loss/len(train_loader)))


def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" %
          (total_eval_loss/len(test_dataloader)))
    print("-------------------------------")

for epoch in range(2):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    validation()

def prediction():
    model.eval()
    test_label = []
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            pred = model(input_ids, attention_mask).logits
            test_label += list(pred.argmax(1).data.cpu().numpy())
    return test_label

test_predict = prediction()

test_data['Topic(Label)'] = [lbl[x] for x in test_predict]
test_data[['Topic(Label)']].to_csv('bert.csv', index=None)