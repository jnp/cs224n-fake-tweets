import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn, cuda, optim
import pandas as pd
import argparse
from tqdm import tqdm
device = 'cuda' if cuda.is_available() else 'cpu'

argp = argparse.ArgumentParser()
argp.add_argument('--reading_params_path',default=None)
argp.add_argument('--writing_params_path',default=None)
argp.add_argument('--train_data_path', default=None)
argp.add_argument('--test_data_path', default=None)
argp.add_argument('--outputs_path', default=None)
argp.add_argument('--checkpoint_path', default=None)
argp.add_argument('--learning_rate', default=6e-5, type=float)
argp.add_argument('--weight_decay', default=0.1, type=float)
argp.add_argument('--learning_decay', default=0.01, type=float)
argp.add_argument('--batch_size', default=256, type=int)
argp.add_argument('--epochs', default=10, type=int)
args = argp.parse_args()

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


class TweetDataset(Dataset):
    def __init__(self, data_df):
        self.input_df = data_df
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.encoded_input = self.tokenizer(self.input_df['text'].tolist(), padding=True, truncation=True)
        print("Number of examples = ", len(self.encoded_input['input_ids']))
        if 'target' in self.input_df.columns:
            self.labels = self.input_df['target'].tolist()
        else:
            self.labels = None

    def __len__(self):
        return len(self.encoded_input['input_ids'])

    def __getitem__(self, idx):
        input_ids = self.encoded_input['input_ids'][idx]
        attention_mask = self.encoded_input['input_ids'][idx]
        if self.labels:
            label = int(self.labels[idx])
            if label != 0 and label != 1:
                raise("Invalid label")
        else:
            label = -1
        return input_ids, attention_mask, label


def custom_collate_fn(inp):
    ids_list = []
    masks_list = []
    label_list = []
    for ids, masks, label in inp:
        ids_list.append(ids)
        masks_list.append(masks)
        label_list.append(label)
    return ids_list, masks_list, label_list


def train_epoch(optimizer, model, loader):
    # Keep track of the total loss for the batch
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    for a in tqdm(loader):
        input_ids = torch.from_numpy(np.array(a[0])).to(device)
        masks = torch.from_numpy(np.array(a[1])).to(device)
        labels = torch.from_numpy(np.array(a[2])).to(device)
        # Clear the gradients
        optimizer.zero_grad()
        # Run a forward pass
        output = model(input_ids, masks)
        # probs = torch.softmax(output.logits, dim=1)
        # max_probs, indices = torch.max(torch.softmax(output.logits, dim=1), dim = 1)
        # print(max_probs)
        # print(labels)
        loss = loss_fn(output.logits, labels)

        # Calculate the gradients
        loss.backward()
        # Update the parameteres
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def eval_model(model, loader):
    # Keep track of the total loss for the batch
    total_correct = 0
    total_evaluated = 0
    with torch.no_grad():
        for a in tqdm(loader):
            input_ids = torch.from_numpy(np.array(a[0])).to(device)
            masks = torch.from_numpy(np.array(a[1])).to(device)
            labels = torch.from_numpy(np.array(a[2])).to(device)
            output = model(input_ids, masks)
            predictions = torch.argmax(output.logits, dim=1)
            total_correct += torch.sum((labels == predictions).int())
            total_evaluated = total_evaluated + input_ids.shape[0]

    return total_correct / total_evaluated * 100

def save_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return
    ckpt_model = model.module if hasattr(model, "module") else model
    torch.save(ckpt_model.state_dict(), ckpt_path)
    print('Saved checkpoint')

input_data_df = pd.read_csv(args.train_data_path)
split_mask = np.random.rand(len(input_data_df)) < 0.8
train_data_df = input_data_df[split_mask]
eval_data_df = input_data_df[~split_mask]

train_tweet_ds = TweetDataset(train_data_df)
eval_tweet_ds = TweetDataset(eval_data_df)

# test_data_df = pd.read_csv(args.test_data_path)
# test_tweet_ds = TweetDataset(test_data_df)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

##Train
train_loader = DataLoader(train_tweet_ds, shuffle=True, batch_size=args.batch_size, collate_fn=custom_collate_fn)
eval_loader = DataLoader(eval_tweet_ds, shuffle=True, batch_size=args.batch_size, collate_fn=custom_collate_fn)
for epoch in range(args.epochs+1):
    print("Epoch: {}/{}".format(epoch, args.epochs))
    model.train()
    epoch_loss = train_epoch(optimizer, model, train_loader)
    print(epoch_loss)
    if epoch > 0 and epoch % 10 == 0:
        model.eval()
        accuracy = eval_model(model, eval_loader)
        print("Evaluation Accuracy = {}$".format(accuracy))
        save_checkpoint(model, args.checkpoint_path)