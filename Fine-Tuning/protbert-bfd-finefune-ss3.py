# **1. Check the GPU device**

# **2. Load necessry libraries including huggingface transformers**

# !pip -q install transformers seqeval

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForTokenClassification, BertTokenizerFast, EvalPrediction
from torch.utils.data import Dataset
import os
import pandas as pd
import requests
from tqdm.auto import tqdm
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import re

# **3. Select the model you want to fine-tune**

model_name = 'Rostlab/prot_bert_bfd'

# **4. Download Netsurfp dataset**

def downloadNetsurfpDataset():
    netsurfpDatasetTrainUrl = 'https://www.dropbox.com/s/98hovta9qjmmiby/Train_HHblits.csv?dl=1'
    casp12DatasetValidUrl = 'https://www.dropbox.com/s/te0vn0t7ocdkra7/CASP12_HHblits.csv?dl=1'
    cb513DatasetValidUrl = 'https://www.dropbox.com/s/9mat2fqqkcvdr67/CB513_HHblits.csv?dl=1'
    ts115DatasetValidUrl = 'https://www.dropbox.com/s/68pknljl9la8ax3/TS115_HHblits.csv?dl=1'

    datasetFolderPath = "dataset/"
    trainFilePath = os.path.join(datasetFolderPath, 'Train_HHblits.csv')
    casp12testFilePath = os.path.join(datasetFolderPath, 'CASP12_HHblits.csv')
    cb513testFilePath = os.path.join(datasetFolderPath, 'CB513_HHblits.csv')
    ts115testFilePath = os.path.join(datasetFolderPath, 'TS115_HHblits.csv')
    combinedtestFilePath = os.path.join(datasetFolderPath, 'Validation_HHblits.csv')

    if not os.path.exists(datasetFolderPath):
        os.makedirs(datasetFolderPath)

    def download_file(url, filename):
        response = requests.get(url, stream=True)
        with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                          total=int(response.headers.get('content-length', 0)),
                          desc=filename) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)

    if not os.path.exists(trainFilePath):
        download_file(netsurfpDatasetTrainUrl, trainFilePath)

    if not os.path.exists(casp12testFilePath):
        download_file(casp12DatasetValidUrl, casp12testFilePath)

    if not os.path.exists(cb513testFilePath):
        download_file(cb513DatasetValidUrl, cb513testFilePath)

    if not os.path.exists(ts115testFilePath):
        download_file(ts115DatasetValidUrl, ts115testFilePath)

    if not os.path.exists(combinedtestFilePath):
        #combine all test dataset files
        combined_csv = pd.concat([pd.read_csv(f) for f in [casp12testFilePath,cb513testFilePath,ts115testFilePath] ])
        #export to csv
        combined_csv.to_csv( os.path.join(datasetFolderPath, "Validation_HHblits.csv"),
                          index=False,
                          encoding='utf-8-sig')

downloadNetsurfpDataset()

# **5. Load dataset into memory**

def load_dataset(path, max_length):
    df = pd.read_csv(path,names=['input','dssp3','dssp8','disorder','cb513_mask'],skiprows=1)
    
    df['input_fixed'] = ["".join(seq.split()) for seq in df['input']]
    df['input_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input_fixed']]
    seqs = [ list(seq)[:max_length-2] for seq in df['input_fixed']]

    df['label_fixed'] = ["".join(label.split()) for label in df['dssp3']]
    labels = [ list(label)[:max_length-2] for label in df['label_fixed']]

    df['disorder_fixed'] = [" ".join(disorder.split()) for disorder in df['disorder']]
    disorder = [ disorder.split()[:max_length-2] for disorder in df['disorder_fixed']]

    assert len(seqs) == len(labels) == len(disorder)
    return seqs, labels, disorder

max_length = 1024

train_seqs, train_labels, train_disorder = load_dataset('dataset/Train_HHblits.csv', max_length)
val_seqs, val_labels, val_disorder = load_dataset('dataset/Validation_HHblits.csv', max_length)
casp12_test_seqs, casp12_test_labels, casp12_test_disorder = load_dataset('dataset/CASP12_HHblits.csv', max_length)
cb513_test_seqs, cb513_test_labels, cb513_test_disorder = load_dataset('dataset/CB513_HHblits.csv', max_length)
ts115_test_seqs, ts115_test_labels, ts115_test_disorder = load_dataset('dataset/TS115_HHblits.csv', max_length)

print(train_seqs[0][10:30], train_labels[0][10:30], train_disorder[0][10:30], sep='\n')

# **6. Tokenize sequences**

seq_tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)

train_seqs_encodings = seq_tokenizer(train_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
val_seqs_encodings = seq_tokenizer(val_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
casp12_test_seqs_encodings = seq_tokenizer(casp12_test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
cb513_test_seqs_encodings = seq_tokenizer(cb513_test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
ts115_test_seqs_encodings = seq_tokenizer(ts115_test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)

# **7. Tokenize labels**

# Consider each label as a tag for each token
unique_tags = set(tag for doc in train_labels for tag in doc)
unique_tags  = sorted(list(unique_tags))  # make the order of the labels unchanged
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels_encodings = encode_tags(train_labels, train_seqs_encodings)
val_labels_encodings = encode_tags(val_labels, val_seqs_encodings)
casp12_test_labels_encodings = encode_tags(casp12_test_labels, casp12_test_seqs_encodings)
cb513_test_labels_encodings = encode_tags(cb513_test_labels, cb513_test_seqs_encodings)
ts115_test_labels_encodings = encode_tags(ts115_test_labels, ts115_test_seqs_encodings)

# **8. Mask disorder tokens**

def mask_disorder(labels, masks):
    for label, mask in zip(labels,masks):
        for i, disorder in enumerate(mask):
            if disorder == "0.0":
                #shift by one because of the CLS token at index 0
                label[i+1] = -100

mask_disorder(train_labels_encodings, train_disorder)
mask_disorder(val_labels_encodings, val_disorder)
mask_disorder(casp12_test_labels_encodings, casp12_test_disorder)
mask_disorder(cb513_test_labels_encodings, cb513_test_disorder)
mask_disorder(ts115_test_labels_encodings, ts115_test_disorder)

# **9. Create SS3 Dataset**

class SS3Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# we don't want to pass this to the model
_ = train_seqs_encodings.pop("offset_mapping")
_ = val_seqs_encodings.pop("offset_mapping")
_ = casp12_test_seqs_encodings.pop("offset_mapping")
_ = cb513_test_seqs_encodings.pop("offset_mapping")
_ = ts115_test_seqs_encodings.pop("offset_mapping")

train_dataset = SS3Dataset(train_seqs_encodings, train_labels_encodings)
val_dataset = SS3Dataset(val_seqs_encodings, val_labels_encodings)
casp12_test_dataset = SS3Dataset(casp12_test_seqs_encodings, casp12_test_labels_encodings)
cb513_test_dataset = SS3Dataset(cb513_test_seqs_encodings, cb513_test_labels_encodings)
ts115_test_dataset = SS3Dataset(ts115_test_seqs_encodings, ts115_test_labels_encodings)

# **10. Define the evaluation metrics**

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray):
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(id2tag[label_ids[i][j]])
                preds_list[i].append(id2tag[preds[i][j]])

    return preds_list, out_label_list

def compute_metrics(p: EvalPrediction):
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
    return {
        "accuracy": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

# **11. Create the model**

def model_init():
    return AutoModelForTokenClassification.from_pretrained(model_name,
                                                         num_labels=len(unique_tags),
                                                         id2label=id2tag,
                                                         label2id=tag2id,
                                                         gradient_checkpointing=False)


# **12. Define the training args and start the trainer**

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=200,                # number of warmup steps for learning rate scheduler
    learning_rate=3e-05,             # learning rate
    weight_decay=0.0,                # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=200,               # How often to print logs
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="epoch",     # evalute after each epoch
    gradient_accumulation_steps=32,  # total number of steps before back propagation
    fp16=False,                       # Use mixed precision
    fp16_opt_level="02",             # mixed precision mode
    run_name="ProBert-BFD-SS3",      # experiment name
    seed=3,                         # Seed for experiment reproducibility
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
)

trainer = Trainer(
    model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=train_dataset,          # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics = compute_metrics,    # evaluation metrics
)

trainer.train()

# **13. Make predictions and evaluate**

predictions, label_ids, metrics = trainer.predict(casp12_test_dataset)

print(metrics)

predictions, label_ids, metrics = trainer.predict(ts115_test_dataset)

print(metrics)

predictions, label_ids, metrics = trainer.predict(cb513_test_dataset)

print(metrics)

idx = 0
sample_ground_truth = " ".join([id2tag[int(tag)] for tag in cb513_test_dataset[idx]['labels'][cb513_test_dataset[idx]['labels'] != torch.nn.CrossEntropyLoss().ignore_index]])
sample_predictions =  " ".join([id2tag[int(tag)] for tag in np.argmax(predictions[idx], axis=1)[np.argmax(predictions[idx], axis=1) != torch.nn.CrossEntropyLoss().ignore_index]])

sample_sequence = seq_tokenizer.decode(list(cb513_test_dataset[idx]['input_ids']), skip_special_tokens=True)

print("Sequence       : {} \nGround Truth is: {}\nprediction is  : {}".format(sample_sequence,
                                                                      sample_ground_truth,
                                                                      # Remove the first token on prediction becuase its CLS token
                                                                      # and only show up to the input length
                                                                      sample_predictions[2:len(sample_sequence)+2]))

# **14. Save the model**

trainer.save_model('prot_bert_bfd_ss3/')

seq_tokenizer.save_pretrained('prot_bert_bfd_ss3/')
