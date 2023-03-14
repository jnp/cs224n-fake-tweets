import argparse
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModel
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torch import optim, nn, softmax
from torch import cuda
from tqdm import tqdm
import torch
import mlflow
device = 'cuda' if cuda.is_available() else 'cpu'


argp = argparse.ArgumentParser()
argp.add_argument('--reading_params_path',default=None)
argp.add_argument('--writing_params_path',default=None)
argp.add_argument('--train_data_path', default=None)
argp.add_argument('--eval_data_path', default=None)
argp.add_argument('--outputs_path', default=None)
argp.add_argument('--checkpoint_path', default=None)
argp.add_argument('--learning_rate', default=6e-4, type=float)
argp.add_argument('--weight_decay', default=0.1, type=float)
argp.add_argument('--learning_decay', default=0.01, type=float)
argp.add_argument('--batch_size', default=16, type=int)
argp.add_argument('--epochs', default=10, type=int)
args = argp.parse_args()

##num_workers is 0 for CPU
num_workers = 2


class WrapperModelForMLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.linear1 = nn.Linear(768, 768, bias=True)
        self.layerNorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear2 = nn.Linear(768, 30522, bias=True)

    def forward(self, input_id, mask):
        pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        layer1_output = self.linear1(pooled_output[0])
        norm_output = self.layerNorm(layer1_output)
        final_output = self.linear2(norm_output)
        return final_output

def get_wrapper_model_for_mlm():
    return WrapperModelForMLM().to(device)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

log_dataset = load_dataset("text", data_files=[args.train_data_path])

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    chunk_size = 128
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def ce_loss_function(batch_outputs, batch_labels):
    # Calculate the loss for the whole batch
    celoss = nn.CrossEntropyLoss()
    loss = celoss(batch_outputs, batch_labels)

    # Rescale the loss. Remember that we have used lengths to store the
    # number of words in each training example
    loss = loss / batch_labels.shape[0]
    return loss

def qa_loss_function(batch_start_output, batch_end_output, batch_start_labels, batch_end_labels):
    loss_start = ce_loss_function(batch_start_output, batch_start_labels)
    loss_end = ce_loss_function(batch_end_output, batch_end_labels)
    return loss_start + loss_end

def train_epoch(loss_function, optimizer, model, loader):
    # Keep track of the total loss for the batch
    total_loss = 0
    for a in tqdm(loader):
        input_ids = a['input_ids'].to(device)
        masks = a['attention_mask'].to(device)
        labels = a['labels'].to(device)
        # Clear the gradients
        optimizer.zero_grad()
        # Run a forward pass
        outputs = model(input_ids, masks)
        # Compute the batch loss
        #loss = loss_function(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        loss = loss_function(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        # Calculate the gradients
        loss.backward()
        # Update the parameteres
        optimizer.step()
        total_loss += loss.item()

    return total_loss

def save_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return
    torch.save(model.bert.state_dict(), ckpt_path)
    print('Saved base model checkpoint')

tokenized_datasets = log_dataset.map(tokenize_function, batched=True, remove_columns=["text"])


lm_datasets = tokenized_datasets.map(group_texts, batched=True)

lm_datasets.set_format(type='torch')


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

loader = DataLoader(lm_datasets['train'], batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)

# model_checkpoint = "distilbert-base-uncased"
# model = AutoModelForMaskedLM.from_pretrained(model_checkpoint).to(device)

model = get_wrapper_model_for_mlm()

optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

with mlflow.start_run():
    mlflow.log_param("args", str(args))
    for epoch in range(args.epochs):
        print("Epoch: {}/{}".format(epoch, args.epochs))
        epoch_loss = train_epoch(ce_loss_function, optimizer, model, loader)
        print(epoch_loss)
        mlflow.log_metric('loss', epoch_loss)
        save_checkpoint(model, args.checkpoint_path)