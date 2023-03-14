import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForMaskedLM
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn, cuda, optim
import pandas as pd
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import mlflow
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
argp.add_argument('--crisis_data', action='store_true')
argp.add_argument('--lm', default=None)
argp.add_argument('--contrastive_loss_lambda', default=0.5, type=float)
args = argp.parse_args()

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


def get_language_model():
    model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
    if args.lm:
        print("Load base model from checkpoint")
        model.load_state_dict(torch.load(args.lm))
    return model


## Use the classifier head from RoBERTa
class WrapperModelForCl(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.bert = get_language_model()
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(768, 1600)
        self.dense2 = nn.Linear(1600, 768)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, input_id, mask):
        pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        features = pooled_output[0][:,0,:]
        x = self.dropout(features)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def get_wrapper_model_for_classification():
    return WrapperModelForCl().to(device)

def get_classification_model_pretrained():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model_version = "af0f99b"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=id2label, label2id=label2id
    ).to(device)
    return model

def get_classification_tokenizer():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model_version = "af0f99b"
    return AutoTokenizer.from_pretrained(model_name)


class TweetDataset(Dataset):
    def __init__(self, data_df):
        self.input_df = data_df
        self.tokenizer = get_classification_tokenizer()
        self.encoded_input = self.tokenizer(self.input_df['text'].tolist(), padding=True, truncation=True)
        print("Number of examples = ", len(self.encoded_input['input_ids']))
        if 'target' in self.input_df.columns:
            self.labels = self.input_df['target'].tolist()
        else:
            self.labels = None

    def __len__(self):
        return len(self.encoded_input['input_ids'])

    def __getitem__(self, idx):
        try:
            input_ids = self.encoded_input['input_ids'][idx]
            attention_mask = self.encoded_input['input_ids'][idx]
            if self.labels:
                label = int(self.labels[idx])
                if label != 0 and label != 1:
                    raise("Invalid label")
            else:
                label = -1
            return input_ids, attention_mask, label, self.input_df['id'][idx]
        except Exception as ex:
            print("Exception in processing idx: ", idx)
            raise ex


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def custom_collate_fn(inp):
    ids_list = []
    masks_list = []
    label_list = []
    id_list = []
    for ids, masks, label, id in inp:
        ids_list.append(ids)
        masks_list.append(masks)
        label_list.append(label)
        id_list.append(id)
    return ids_list, masks_list, label_list, id_list



# def get_contrastive_loss(phi, labels, tau=0.5):
#     label_counts = torch.bincount(labels)
#     N_0 = label_counts[0].item()
#     N_1 = label_counts[1].item()
#     outer_product = torch.matmul(phi, phi.T) / tau
#     #make diagonal zero
#     outer_product.fill_diagonal_(0)
#     #Indicator matrix for labels
#     labels_square = labels.view(-1, 1).repeat(1, labels.shape[0])
#     labels_indicators = (labels_square == labels_square.T).long()
#     exponential_matrix = torch.exp(outer_product)
#     exponential_matrix.fill_diagonal_(0)
#     log_sum_rows = torch.log(torch.sum(exponential_matrix, dim=1))
#     log_softmax = outer_product - log_sum_rows.view(-1,1)
#
#     matching_matrix = log_softmax * labels_indicators
#     c_loss = torch.sum(matching_matrix, dim=1)
#     label_counts = torch.empty(labels.shape).to(device)
#     label_counts[labels == 1] = N_1 - 1
#     label_counts[labels == 0] = N_0 - 1
#     c_loss = c_loss / label_counts
#     total_c_loss = - torch.sum(c_loss)
#     return total_c_loss


# def total_loss_function(outputs, labels, cl_lambda=0.5, tau=0.5):
#     ce_loss = nn.CrossEntropyLoss()(outputs, labels)
#     if cl_lambda > 0:
#         sc_loss = get_contrastive_loss(outputs, labels, tau)
#     else:
#         sc_loss = 0
#     return (1-cl_lambda) * ce_loss + cl_lambda * sc_loss

def train_epoch(optimizer, model, loader):
    # Keep track of the total loss for the batch
    total_loss = 0
    ce_loss = nn.CrossEntropyLoss()
    sc_loss = SupConLoss()
    for a in tqdm(loader):
        input_ids = torch.from_numpy(np.array(a[0])).to(device)
        masks = torch.from_numpy(np.array(a[1])).to(device)
        labels = torch.from_numpy(np.array(a[2])).to(device)
        # Clear the gradients
        optimizer.zero_grad()
        # Run a forward pass
        output = model(input_ids, masks)
        cross_entropy_loss = ce_loss(output, labels)
        contrastive_loss = sc_loss(output.view(output.shape[0], 1, output.shape[1]), labels=labels)
        cl_lambda = args.contrastive_loss_lambda
        loss = (1 - cl_lambda) * cross_entropy_loss + cl_lambda * contrastive_loss

        # Calculate the gradients
        loss.backward()
        # Update the parameteres
        optimizer.step()
        total_loss += loss.item()

    return total_loss

def test_model(model, loader):
    model.eval()
    output_df = pd.DataFrame()
    output_df['id'] = []
    output_df['target'] = []
    with torch.no_grad():
        for a in tqdm(loader):
            input_ids = torch.from_numpy(np.array(a[0])).to(device)
            masks = torch.from_numpy(np.array(a[1])).to(device)
            #labels = torch.from_numpy(np.array(a[2])).to(device)
            output = model(input_ids, masks)
            output_labels = torch.argmax(output.logits, dim=1).int()
            tmp_df = pd.DataFrame()
            tmp_df['id'] = a[3]
            tmp_df['target'] = output_labels.tolist()
            output_df = output_df.append(tmp_df)
    output_df = output_df.astype({'id': 'int', 'target': 'int'})
    return output_df


def compare_scalar(t_tensor, s):
    return t_tensor.eq(torch.from_numpy(np.array([s])).to(device))

def get_accuracy_stats(predictions, labels):
    true_positives = torch.sum(torch.logical_and(compare_scalar(labels, 1), compare_scalar(predictions, 1)).int()).item()
    false_positives = torch.sum(torch.logical_and(compare_scalar(labels, 0), compare_scalar(predictions, 1)).int()).item()
    false_negatives = torch.sum(torch.logical_and(compare_scalar(labels, 1), compare_scalar(predictions, 0)).int()).item()
    true_negatives = torch.sum(torch.logical_and(compare_scalar(labels, 0), compare_scalar(predictions, 0)).int()).item()
    return true_positives, false_positives, false_negatives, true_negatives


def eval_model(model, loader):
    # Keep track of the total loss for the batch
    print("Evaluating Model")
    true_positives, false_positives, false_negatives, true_negatives = 0, 0, 0, 0
    with torch.no_grad():
        for a in tqdm(loader):
            input_ids = torch.from_numpy(np.array(a[0])).to(device)
            masks = torch.from_numpy(np.array(a[1])).to(device)
            labels = torch.from_numpy(np.array(a[2])).to(device)
            output = model(input_ids, masks)
            predictions = torch.argmax(output, dim=1).int()
            tp, fp, fn, tn = get_accuracy_stats(predictions, labels)
            true_positives += tp
            false_positives += fp
            false_negatives += fn
            true_negatives += tn

    f1_score = true_positives / (true_positives + 0.5 *(false_positives + false_negatives))
    accuracy = (true_positives+true_negatives) / (true_positives + false_positives + false_negatives + true_negatives) * 100
    return accuracy, f1_score

def save_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return
    ckpt_model = model.module if hasattr(model, "module") else model
    torch.save(ckpt_model.state_dict(), ckpt_path)
    print('Saved checkpoint')

#model = get_classification_model_pretrained()
model = get_wrapper_model_for_classification()

if args.test_data_path:
    test_data_df = pd.read_csv(args.test_data_path)
    test_tweet_ds = TweetDataset(test_data_df)
    loader = DataLoader(test_tweet_ds, shuffle=False, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    model.load_state_dict(torch.load((args.checkpoint_path)))
    output_df = test_model(model, loader)
    test_output_file = "./test_output.csv"
    output_df.to_csv(test_output_file, index=False)
    print('test output writtin to', test_output_file)
    exit(0)


training_input_df = pd.read_csv(args.train_data_path)
training_input_df = training_input_df.drop(columns=['keyword', 'location']).dropna()
if args.crisis_data:
    crisis_data_df = pd.read_csv('../data/crisis6.csv').dropna()
    training_input_df = training_input_df.append(crisis_data_df)
split_mask = np.random.rand(len(training_input_df)) < 0.8
train_data_df = training_input_df[split_mask].reset_index(drop=True)
eval_data_df = training_input_df[~split_mask].reset_index(drop=True)

print("Create dataset for training")
train_tweet_ds = TweetDataset(train_data_df)
print("Create dataset for evaluation")
eval_tweet_ds = TweetDataset(eval_data_df)

# test_data_df = pd.read_csv(args.test_data_path)
# test_tweet_ds = TweetDataset(test_data_df)

optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

# if args.checkpoint_path:
#     if os.path.exists(args.checkpoint_path):
#         print('Loading model from checkpoint')
#         model.load_state_dict(torch.load((args.checkpoint_path)))

##Train
train_loader = DataLoader(train_tweet_ds, shuffle=False, batch_size=args.batch_size, collate_fn=custom_collate_fn)
eval_loader = DataLoader(eval_tweet_ds, shuffle=True, batch_size=args.batch_size, collate_fn=custom_collate_fn)

with mlflow.start_run() as run:
    mlflow.log_param("args", str(args))
    for epoch in range(args.epochs+1):
        print("Epoch: {}/{}".format(epoch, args.epochs))
        model.train()
        scheduler.step(epoch)
        epoch_loss = train_epoch(optimizer, model, train_loader)
        mlflow.log_metric("loss", epoch_loss)
        print(epoch_loss)
        if epoch % 5 == 0:
            model.eval()
            accuracy, f1_score = eval_model(model, train_loader)
            print("Train Evaluation Accuracy = {}".format(accuracy))
            print("Train Evaluation F1 Score = {}".format(f1_score))
            dev_accuracy, dev_f1_score = eval_model(model, eval_loader)
            print("Dev Evaluation Accuracy = {}".format(dev_accuracy))
            print("Dev Evaluation F1 Score = {}".format(dev_f1_score))
            mlflow.log_metrics({"Train_Accuracy": accuracy, "Train_F1_score": f1_score,
                                "Dev_Accuracy": dev_accuracy, "Dev_F1_Score": dev_f1_score})
            save_checkpoint(model, args.checkpoint_path)