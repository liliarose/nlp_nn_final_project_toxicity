# imports
import argparse 
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import accuracy_score, recall_score, precision_score

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  binary_recall = recall_score(labels, preds, average='binary')
  binary_precision = precision_score(labels, preds, average='binary')
  return {
      'accuracy': acc,
      'binary_recall': binary_recall,
      'binary_precision': binary_precision
  }


class Dataset(torch.utils.data.Dataset):
    '''
      Defining a dataset for our dataset 
      uses the gobal variables label_col & text_col
      And has the 3 required functions 
    '''
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
                     
    def __len__(self):
        # returns the size of dataset 
        return len(self.labels)
    
    def __getitem__(self, idx):
        # used for the iterator
        # returns a batch text & its label
        item = {k: torch.tensor(v[idx]) for k, v in self.texts.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

 
# also assumes that they have comment_text & toxic columns
def main(args):
    # deal with this....
    train_set = pd.read_csv(f'{args.data}/train.csv')
    test_set = pd.read_csv(f'{args.data}/test.csv')

    if args.test:
        L = 2*args.batch_size + args.batch_size//2
        train_set = train_set.iloc[:L]
        test_set = test_set.iloc[:L]

    X_train = train_set['comment_text']
    y_train = train_set['toxic']
    X_test = test_set['comment_text']
    y_test = test_set['toxic']
 
    # cleans by default
    tokenizer = BertTokenizerFast.from_pretrained(args.bertmodel, do_lower_case=True)
    max_length = 512
    train_encodings = tokenizer(X_train.to_list(), truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(X_test.to_list(), truncation=True, padding=True, max_length=max_length)

    train_dataset = Dataset(train_encodings, y_train)
    test_dataset = Dataset(test_encodings, y_test)
    print('done reading data')

    # model & all freeze layers except classification 
    model = BertForSequenceClassification.from_pretrained(args.bertmodel, num_labels=2).to("cuda")

    for param in model.bert.parameters():
        param.requires_grad = False
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,              # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        warmup_steps=args.warm_up,                # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        evaluation_strategy="steps",     # evaluate each `logging_steps`
    )

    print('entering train')
    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,          # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='folder with data')
    parser.add_argument('--batch_size', '-bs', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', '-ep', default=4, type=int, help='epochs')
    parser.add_argument('--warm_up', '-wp', default=500, type=int, help='warm up steps for the lr scheduler')
    parser.add_argument('--weight_decay', '-wd', default=0.01, type=float, help='strength of weight decay')
    parser.add_argument('--test', '-t', action='store_true', help='tests on a smaller subset')
    parser.add_argument('--bertmodel', '-b', default='bert-base-uncased', help='type of bert to use')
    parser.add_argument('--output_dir', '-o', default='models/', help='where to save models or results; make sure to include / at the end or a prefix')
    args = parser.parse_args()
    main(args)



# from train_bert2 import *
# every_train = pd.read_csv('wiki/train.csv')
# train_set = every_train.iloc[:10][['comment_text', 'toxic']]
# test_set = every_train.iloc[10:20][['comment_text', 'toxic']]
# train('bert-base-uncased', train_set, test_set, 1e-5, 2, 'models/', 32)
