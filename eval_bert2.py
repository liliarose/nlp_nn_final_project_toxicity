import argparse 
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from train_bert2 import Dataset, BertClassifier


def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def evaluate(model, val_dataloader, sav_loc, val_data_len, calc_class=1):
    '''
      evaluates the model & saves the probabilities in a numpy file 
      Also calcuates recall and accuracy along the way as well
      Code very similar to one in  
    '''

    # check if we can use gpu 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('use_cuda', use_cuda)

    # if we can use gpu, make sure the model & gradient computation 
    # are doing so 
    if use_cuda:
        model = model.cuda()
    print('finished deciding cuda & other initalization')

    # for calculation of metrics 
    total_acc_val = 0
    correct_true_val = 0
    target_true_val = 0
    # will be array of all probabilites 
    all_probs = None
    with torch.no_grad():
        for val_input, val_label in val_dataloader:
            # forward pass 
            val_label = val_label.to(device)
            mask = val_input['attention_mask'].to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            
            # for accuracy 
            acc = (output.argmax(dim=1) == val_label).sum().item()
            total_acc_val += acc

            # for recall 
            predicted_classes = (output.argmax(dim=1) == calc_class)        # calculates the # predicted for the class we're interested       
            correct_classes = (predicted_classes == val_label).int()      # calculates the number acutally correctly calculated class
            target_true_val += torch.sum(val_label == calc_class)       # the number actually in the classes that we're interested
            # the # correctly calculated in the class we're interested in 
            correct_true_val += torch.sum(correct_classes* (predicted_classes == calc_class).int()).int()

            # for saving the probabilities 
            if use_cuda:
                curr_probs = output.cpu().numpy() 
            else:
                curr_probs = output.detach().numpy()
            if all_probs is None:
                all_probs = curr_probs
            else:
                all_probs = np.concatenate([all_probs, curr_probs])
                

    # prints metrics   
    print(
        f'Val Accuracy: {total_acc_val / val_data_len: .3f}\
        | Val Recall: {correct_true_val / target_true_val : .3f}')

    # saves the metrics 
    with open(sav_loc, 'wb') as f:
        np.save(f, all_probs)
                
# also assumes that they have comment_text & toxic columns
def main(args):
    # deal with this....
    test_set = pd.read_csv(f'{args.data}/test.csv')

    if args.test:
        test_set = test_set.iloc[:(args.batch_size + args.batch_size//2)]
    print('done reading data')

    tokenizer = BertTokenizer.from_pretrained(args.bertmodel)
    model = BertClassifier(bertmodel=args.bertmodel)
    val = Dataset(test_set, tokenizer)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=args.batch_size)
    load_checkpoint(args.model_fn, model)

    
    save_loc = args.save_loc
    if args.save_loc == '':
        save_loc = args.model_fn.split('.pt')[0] + f'_{args.test}.npy'

    print('entering evaluation')
    evaluate(model, val_dataloader, save_loc, len(test_set), args.calc_class)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='folder with data')
    parser.add_argument('model_fn', help='model to evaluate')
    parser.add_argument('--bertmodel', '-bm', default='bert-base-uncased', help='type of bert to use')
    parser.add_argument('--calc_class', '-cc', default=1, type=int, help='class to calculate recall & auc')
    parser.add_argument('--save_loc', '-sl', default='', help='where to save models or results; make sure to include / at the end or a prefix')
    parser.add_argument('--test', '-t', action='store_true', help='tests the code using a smaller subset (just a 1 & 1/2 batch)')
    parser.add_argument('--batch_size', '-bs', default=32, type=int, help='batch size')
    args = parser.parse_args()
    main(args)



# from train_bert2 import *
# every_train = pd.read_csv('wiki/train.csv')
# train_set = every_train.iloc[:10][['comment_text', 'toxic']]
# test_set = every_train.iloc[10:20][['comment_text', 'toxic']]
# train('bert-base-uncased', train_set, test_set, 1e-5, 2, 'models/', 32)
