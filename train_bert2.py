import argparse 
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm

from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

label_col = 'toxic'
text_col = 'comment_text'

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer):
        self.labels = [label for label in df[label_col]]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df[text_col]]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# def load_checkpoint(load_path, model):
    
#     if load_path==None:
#         return
    
#     state_dict = torch.load(load_path, map_location=device)
#     print(f'Model loaded from <== {load_path}')
    
#     model.load_state_dict(state_dict['model_state_dict'])
#     return state_dict['valid_loss']

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

def train(bertmodel, train_data, val_data, learning_rate, epochs, bs=16):
    model = BertClassifier(bertmodel=bertmodel)
    tokenizer = BertTokenizer.from_pretrained(bertmodel)
    train, val = Dataset(train_data, tokenizer), Dataset(val_data, tokenizer)

    print('created datasets')
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=bs)

    print('finished loading data')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)
    print('use_cuda', use_cuda)
    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    print('finished deciding cuda & other initalization')
    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
# also assumes that they have comment_text & toxic columns
def main(args):
    # deal with this....
    train_set = pd.read_csv(f'{args.data}/train.csv')
    test_set = pd.read_csv(f'{args.data}/test.csv')

    print('done reading data')

    print('entering train')
    train(args.bertmodel, train_set, test_set, args.learning_rate, args.epochs, args.batch_size)

    # # tokenize & loading data 

    # # convert to Tensor

    # print('done converting labels to tensor')

    # # create dataloader for training & testing

    # print('done creating dataloader for train')

    # print('done creating dataloader for validation')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='folder with data')
    parser.add_argument('--batch_size', '-bs', default=32, help='batch size', type=int)
    parser.add_argument('--learning_rate', '-lr', default=0.0001, help='default adam learning rate', type=float)
    parser.add_argument('--bertmodel', '-b', default='bert-base-uncased', help='type of bert to use')
    # parser.add_argument('--max_len', '-ml', default=512, help='max length of tokens')
    # parser.add_argument('--warm_up_epochs', '-wep', default=2, help='number of epochs for warming up')
    parser.add_argument('--epochs', '-ep', default=2, help='number of epochs')
    parser.add_argument('--save_checkpoint', '-sc', default='models/', help='where to save models; make sure to include / at the end or a prefix')
    # parser.add_argument('--seed', '-sd', default=42, help='seed to set it at')
    args = parser.parse_args()
    main(args)


