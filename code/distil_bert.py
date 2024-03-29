import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForMaskedLM
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn, cuda, optim
import pandas as pd
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
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
argp.add_argument('--batch_size', default=32, type=int)
argp.add_argument('--epochs', default=10, type=int)
argp.add_argument('--crisis_data', action='store_true')
argp.add_argument('--lm', default=None)
argp.add_argument('--contrastive_loss_lambda', default=0, type=float)
argp.add_argument('--few_shots', default=-1, type=int)
argp.add_argument('--prototypical_network', action='store_true')
argp.add_argument('--model_summary', action='store_true')
args = argp.parse_args()

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

classification_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
#classification_model_name = "roberta-base"
language_model_name = 'distilbert-base-uncased'
#language_model_name = 'roberta-base'
def get_language_model():
    model = AutoModel.from_pretrained(language_model_name).to(device)
    if args.lm:
        print("Load base model from checkpoint")
        model.load_state_dict(torch.load(args.lm))
        for param in model.parameters():
            param.requires_grad = False
    return model


## Use the classifier head from RoBERTa
class WrapperModelForCl(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.bert = get_language_model()
            # for param in self.bert.parameters():
            #     param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(768, 768)
        #self.dense2 = nn.Linear(1600, 768)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, input_id, mask, labels):
        pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        features = pooled_output[0][:,0,:]
        x = self.dropout(features)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        #x = self.dense2(x)
        #x = torch.relu(x)
        #x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ProtoTypeNetwork(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.bert = get_language_model()
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(768, 768)
        self.c_0_proto = None
        self.c_1_proto = None

    def forward(self, input_id, mask, labels=None):
        pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        features = pooled_output[0][:,0,:]
        x = self.dropout(features)
        x = self.dense(x)
        embeddings = torch.relu(x)

        if labels != None:
            one_embeddings = embeddings[(labels == 1)]
            zero_embeddings = embeddings[(labels == 0)]
            self.c_0_proto = torch.sum(zero_embeddings, dim=0) / zero_embeddings.shape[0]
            self.c_1_proto = torch.sum(one_embeddings, dim=0) / one_embeddings.shape[0]

        distance_c_0 = torch.cdist(embeddings.view(embeddings.shape[0], embeddings.shape[1]),
                                   self.c_0_proto.view(1, -1))
        distance_c_1 = torch.cdist(embeddings.view(embeddings.shape[0], embeddings.shape[1]),
                                   self.c_1_proto.view(1, -1))
        all_distances = -1 * torch.cat([distance_c_0, distance_c_1], dim=1)
        #softmax_distances = torch.softmax(all_distances, dim=1)
        return all_distances

def get_wrapper_for_prototypical_network():
    print("Loading prototypical network")
    return ProtoTypeNetwork().to(device)

def get_wrapper_model_for_classification():
    return WrapperModelForCl().to(device)

def get_classification_model_pretrained():
    model = AutoModelForSequenceClassification.from_pretrained(
        classification_model_name, num_labels=2, id2label=id2label, label2id=label2id
    ).to(device)
    return model

def get_classification_tokenizer():
    return AutoTokenizer.from_pretrained(classification_model_name)

# to clean data
def normalize_text(text):
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#","") # replaces hashtags
    text = text.str.replace(r"http\S+","URL")  # remove URL addresses
    text = text.str.replace(r"@","")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text

class TweetDataset(Dataset):
    def __init__(self, data_df):
        self.input_df = data_df
        self.tokenizer = get_classification_tokenizer()
        self.encoded_input = self.tokenizer(self.input_df['text'].tolist(), padding='max_length', truncation=True, max_length=76)
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
            attention_mask = self.encoded_input['attention_mask'][idx]
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
    def __init__(self, temperature=0.3, contrast_mode='all',
                 base_temperature=0.3):
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
    true_positives, false_positives, false_negatives, true_negatives = 0, 0, 0, 0
    for a in tqdm(loader):
        input_ids = torch.from_numpy(np.array(a[0])).to(device)
        masks = torch.from_numpy(np.array(a[1])).to(device)
        labels = torch.from_numpy(np.array(a[2])).to(device)
        # Clear the gradients
        optimizer.zero_grad()
        # Run a forward pass
        output = model(input_ids, masks, labels)
        cross_entropy_loss = ce_loss(output, labels)
        cl_lambda = args.contrastive_loss_lambda
        if cl_lambda > 0:
            contrastive_loss = sc_loss(output.view(output.shape[0], 1, output.shape[1]), labels=labels)
            loss = (1 - cl_lambda) * cross_entropy_loss + cl_lambda * contrastive_loss
        else:
            loss = cross_entropy_loss

        # Calculate the gradients
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update the parameteres
        optimizer.step()
        total_loss += loss.item()
        tp, fp, fn, tn = get_accuracy_stats(output, labels)
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        true_negatives += tn

    accuracy, f1_score = compute_accuracy(true_positives, false_positives, false_negatives, true_negatives)
    return accuracy, f1_score, total_loss

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
            output_labels = torch.argmax(output, dim=1).int()
            tmp_df = pd.DataFrame()
            tmp_df['id'] = a[3]
            tmp_df['target'] = output_labels.tolist()
            output_df = output_df.append(tmp_df)
    output_df = output_df.astype({'id': 'int', 'target': 'int'})
    return output_df


def compare_scalar(t_tensor, s):
    return t_tensor.eq(torch.from_numpy(np.array([s])).to(device))

def get_accuracy_stats(logits, labels):
    predictions = torch.argmax(logits, dim=1).int()
    true_positives = torch.sum(torch.logical_and(compare_scalar(labels, 1), compare_scalar(predictions, 1)).int()).item()
    false_positives = torch.sum(torch.logical_and(compare_scalar(labels, 0), compare_scalar(predictions, 1)).int()).item()
    false_negatives = torch.sum(torch.logical_and(compare_scalar(labels, 1), compare_scalar(predictions, 0)).int()).item()
    true_negatives = torch.sum(torch.logical_and(compare_scalar(labels, 0), compare_scalar(predictions, 0)).int()).item()
    return true_positives, false_positives, false_negatives, true_negatives

def compute_accuracy(true_positives, false_positives, false_negatives, true_negatives):
    f1_score = true_positives / (true_positives + 0.5 *(false_positives + false_negatives))
    accuracy = (true_positives+true_negatives) / (true_positives + false_positives + false_negatives + true_negatives) * 100
    return accuracy, f1_score

def eval_model(model, loader):
    # Keep track of the total loss for the batch
    true_positives, false_positives, false_negatives, true_negatives = 0, 0, 0, 0
    ce_loss = nn.CrossEntropyLoss()
    sc_loss = SupConLoss()
    total_loss = 0
    with torch.no_grad():
        for a in tqdm(loader):
            input_ids = torch.from_numpy(np.array(a[0])).to(device)
            masks = torch.from_numpy(np.array(a[1])).to(device)
            labels = torch.from_numpy(np.array(a[2])).to(device)
            output = model(input_ids, masks)
            cross_entropy_loss = ce_loss(output, labels)
            cl_lambda = args.contrastive_loss_lambda
            if cl_lambda > 0:
                contrastive_loss = sc_loss(output.view(output.shape[0], 1, output.shape[1]), labels=labels)
                loss = (1 - cl_lambda) * cross_entropy_loss + cl_lambda * contrastive_loss
            else:
                loss = cross_entropy_loss
            tp, fp, fn, tn = get_accuracy_stats(output, labels)
            true_positives += tp
            false_positives += fp
            false_negatives += fn
            true_negatives += tn
            total_loss += loss.item()

    accuracy, f1_score = compute_accuracy(true_positives, false_positives, false_negatives, true_negatives)
    return accuracy, f1_score, total_loss

def save_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return
    ckpt_model = model.module if hasattr(model, "module") else model
    torch.save(ckpt_model.state_dict(), ckpt_path)
    print('Saved checkpoint')

#model = get_classification_model_pretrained()
if args.prototypical_network:
    model = get_wrapper_for_prototypical_network()
else:
    model = get_wrapper_model_for_classification()


if args.model_summary:
   print(model)
   exit(0)

if args.test_data_path:
    test_data_df = pd.read_csv(args.test_data_path)
    test_data_df['text'] = normalize_text(test_data_df['text'])
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

training_input_df['text'] = normalize_text(training_input_df['text'])
train_data_df, eval_data_df = train_test_split(training_input_df)

if args.few_shots > 0:
    print('Few Shots = ', args.few_shots)
    train_data_df = train_data_df.sample(n=args.few_shots)

train_data_df = train_data_df.reset_index(drop=True)
eval_data_df = eval_data_df.reset_index(drop=True)
# split_mask = np.random.rand(len(training_input_df)) < 0.8
# train_data_df = training_input_df[split_mask].reset_index(drop=True)
# train_data_df['text'] = normalize_text(train_data_df['text'])
# eval_data_df = training_input_df[~split_mask].reset_index(drop=True)
# eval_data_df['text'] = normalize_text(eval_data_df['text'])

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
    patience = 5
    num_no_improvement = 0
    best_val_loss = float('inf')
    store_val_loss = []
    best_val_accuracy = -float('inf')
    for epoch in range(args.epochs+1):
        print("Epoch: {}/{}".format(epoch, args.epochs))
        model.train()
        scheduler.step(epoch)
        train_accuracy, train_f1_score, epoch_loss = train_epoch(optimizer, model, train_loader)
        mlflow.log_metric("train_loss", epoch_loss)
        print('Training Loss', epoch_loss)
        if epoch > 3 and epoch % 2 == 0:
            print("Evaluating Model")
            model.eval()
            dev_accuracy, dev_f1_score, eval_loss = eval_model(model, eval_loader)
            print('Dev loss', eval_loss)
            print("Train Evaluation Accuracy = {}".format(train_accuracy))
            print("Train Evaluation F1 Score = {}".format(train_f1_score))
            print("Dev Evaluation Accuracy = {}".format(dev_accuracy))
            print("Dev Evaluation F1 Score = {}".format(dev_f1_score))
            print("Best Dev Accuracy = {}".format(best_val_accuracy))
            mlflow.log_metrics({"Train_Accuracy": train_accuracy, "Train_F1_score": train_f1_score,
                                "Dev_Accuracy": dev_accuracy, "Dev_F1_Score": dev_f1_score, "dev_loss": eval_loss})
            store_val_loss.append(eval_loss)

            # Check if validation loss has improved
            mean_val_loss = np.mean(store_val_loss)
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                #save_checkpoint(model, args.checkpoint_path)
                num_no_improvement = 0
            else:
                num_no_improvement += 1
            if dev_accuracy > best_val_accuracy:
                best_val_accuracy = dev_accuracy
                save_checkpoint(model, args.checkpoint_path)
            if num_no_improvement == patience:
                print('No more improvement expected, exiting')
                break
