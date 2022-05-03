import argparse 
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

label_col = 'toxic'
text_col = 'comment_text'

class Dataset(torch.utils.data.Dataset):

    # the default functions that need to be defined 
    def __init__(self, df, tokenizer):
        self.labels = [label for label in df[label_col]]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df[text_col]]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        batch_texts = self.texts[idx]
        batch_y = np.array(self.labels[idx])

        return batch_texts, batch_y

    def classes(self):
        return self.labels

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5, bertmodel='bert-base-cased'):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(bertmodel)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

def train(bertmodel, train_data, val_data, learning_rate,
     epochs, sav_loc, bs=16, calc_class=1, dropout=0.5):

    # create a bert model 
    model = BertClassifier(bertmodel=bertmodel, dropout=dropout)
    # create a tokenizer & tokenize classes 
    tokenizer = BertTokenizer.from_pretrained(bertmodel)
    train, val = Dataset(train_data, tokenizer), Dataset(val_data, tokenizer)
    print('created datasets')

    # create dataloader 
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=bs)
    print('finished loading data')

    # check if we can use gpu?
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('use_cuda', use_cuda)

    # initialize cross entropy loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    print('finished deciding cuda & other initalization')

    # initialized to check when to save for stuff 
    min_val_loss = float('inf')

    for epoch_num in range(epochs):
        # for calculating accuracy & loss & recall
        total_acc_train = 0
        total_loss_train = 0
        target_true_train = 0
        correct_true_train = 0 
        for train_input, train_label in tqdm(train_dataloader):

            # sets up the data 
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            # run through model 
            output = model(input_id, mask)
            
            # calculate the loss & update total loss 
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            
            # for accuracy 
            acc = (output.argmax(dim=1) == train_label).sum().item() 
            total_acc_train += acc

            # for recall 
            predicted_classes = (output.argmax(dim=1) == calc_class)
            correct_classes = (predicted_classes == train_label).int() 
            true_classes = (predicted_classes == calc_class).int()
            target_true_train = torch.sum(true_classes).int()
            correct_true_train += torch.sum( correct_classes* true_classes).int()

            # updates 
            model.zero_grad() # clears old gradients from the last step 
            batch_loss.backward() # back propagation 
            optimizer.step() # optimizer takes a step 
        
        print('finished training')
        total_acc_val = 0
        total_loss_val = 0
        correct_true_val = 0
        target_true_val = 0
        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                # everything here is the same as before except we use  
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

                # for recall 
                predicted_classes = (output.argmax(dim=1) == calc_class)
                correct_classes = (predicted_classes == val_label).int() 
                true_classes = (predicted_classes == calc_class).int()
                target_true_val = torch.sum(true_classes).int()
                correct_true_train += torch.sum(correct_classes* true_classes).int()
            
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Train Recall: {correct_true_train / target_true_train : .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}\
            | Val Recall: {correct_true_val / target_true_val : .3f}')
        
        if total_loss_val < min_val_loss:
            print('SAVING checkpoint')
            fn = f'{sav_loc}epoch_{epoch_num}_{total_loss_val / len(val_data)}.pt'
            save_checkpoint(fn, model, total_loss_val)
            min_val_loss = total_loss_val
                
# also assumes that they have comment_text & toxic columns
def main(args):
    # deal with this....
    train_set = pd.read_csv(f'{args.data}/train.csv')
    test_set = pd.read_csv(f'{args.data}/test.csv')

    if args.test:
        train_set = train_set.iloc[:(2*args.batch_size)]
        test_set = test_set.iloc[:(2*args.batch_size)]
    print('done reading data')

    print('entering train')
    train(args.bertmodel, train_set, test_set, args.learning_rate, args.epochs, 
        args.save_checkpoint, args.batch_size, args.calc_class, args.dropout)
    # train('bert-base-uncased', train_set, test_set, 1e-5, 2, 'models/', 32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='folder with data')
    parser.add_argument('--batch_size', '-bs', default=32, type=int, help='batch size')
    parser.add_argument('--test', '-t', action='store_true', help='tests on a smaller subset')
    parser.add_argument('--learning_rate', '-lr', default=0.0001, type=float, help='default adam learning rate')
    parser.add_argument('--bertmodel', '-b', default='bert-base-uncased', help='type of bert to use')
    parser.add_argument('--epochs', '-ep', default=4, type=int, help='number of epochs')
    parser.add_argument('--dropout', '-dr', default=0.5, type=float, help='dropout rate for the linear layer')
    parser.add_argument('--calc_class', '-cc', default=1, type=int, help='class to calculate recall & auc')
    parser.add_argument('--save_checkpoint', '-sc', default='models/', help='where to save models or results; make sure to include / at the end or a prefix')
    args = parser.parse_args()
    main(args)



# from train_bert2 import *
# every_train = pd.read_csv('wiki/train.csv')
# train_set = every_train.iloc[:10][['comment_text', 'toxic']]
# test_set = every_train.iloc[10:20][['comment_text', 'toxic']]
# train('bert-base-uncased', train_set, test_set, 1e-5, 2, 'models/', 32)
