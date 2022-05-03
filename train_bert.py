import os
import pandas as pd
import torch
import numpy as np
import re 
import argparse
import random
import time

from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# def text_preprocessing(text):
#     """
#     - Remove entity mentions (eg. '@united')
#     - Correct errors (eg. '&amp;' to '&')
#     """
#     # Remove '@name'
#     text = re.sub(r'(@.*?)[\s]', ' ', text)

#     # Replace '&amp;' with '&'
#     text = re.sub(r'&amp;', '&', text)

#     # Remove trailing whitespace
#     text = re.sub(r'\s+', ' ', text).strip()

#     return text


# Create a function to tokenize a set of texts
def preprocessing_for_bert(data, tokenizer, max_len=512):
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=max_len,                  # Max length to truncate/pad
            truncation=True,                 # truncating 
            pad_to_max_length=True,         # Pad sentence to max length
            return_attention_mask=True      # Return attention mask
            )

        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, bert_type='bert-base-uncased', freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(bert_type)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

def initialize_model(epochs=4, lr=5e-5, bert_type='bert-base-uncased'):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(bert_type, freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def train(model, train_dataloader, val_dataloader, sav_loc, optimizer, scheduler, epochs=4):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")

    min_val_loss = float('inf')

    for epoch_i in range(epochs):
        # #################
        # Training
        # #################
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 25 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # #################
        # Evaluation
        # #################
        # After the completion of each training epoch, measure the model's performance
        # on our validation set.
        val_loss, val_accuracy = evaluate(model, val_dataloader)

        # Print performance over the entire training data
        time_elapsed = time.time() - t0_epoch
        
        # save if smaller
        if min_val_loss > val_loss:
            fn = f'{sav_loc}epoch_{epoch_i}/{epoch_i}_{val_loss}.pt'
            save_checkpoint(fn, model, val_loss)
        print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
        print("-"*70)
        print("\n")
    
    print("Training complete!")


# also assumes that they have comment_text & toxic columns
def main(args):
    # deal with this....
    train = pd.read_csv(f'{args.data}/train.csv')
    test = pd.read_csv(f'{args.data}/test.csv')
    X_train, X_val = train['comment_text'].tolist(), train['toxic'].tolist()
    y_train, y_val = test['comment_text'].tolist(), test['toxic'].tolist()

    print('done reading data')

    # tokenize
    tokenizer = BertTokenizer.from_pretrained(args.bert_type, do_lower_case=True)
    train_inputs, train_masks = preprocessing_for_bert(X_train, tokenizer, args.max_len)
    val_inputs, val_masks = preprocessing_for_bert(X_val, tokenizer, args.max_len)
    print('done tokkenizing')

    # convert to Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    print('done converting labels to tensor')

    # create dataloader for training & testing
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    print('done creating dataloader for train')
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)
    print('done creating dataloader for validation')

    set_seed(args.seed)
    bert_classifier, optimizer, scheduler = initialize_model(epochs=args.warm_up_epochs, lr=args.learning_rate, bert_type=args.bert_type)
    print('done initializing the model')
    train(bert_classifier, train_dataloader, val_dataloader, args.save_checkpoint, optimizer, scheduler, epochs=args.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='folder with data')
    parser.add_argument('--batch_size', '-bs', default=32, help='batch size', type=int)
    parser.add_argument('--learning_rate', '-lr', default=0.0001, help='default adam learning rate', type=float)
    parser.add_argument('--bert_type', '-b', default='bert-base-uncased', help='type of bert to use')
    parser.add_argument('--max_len', '-ml', default=512, help='max length of tokens')
    parser.add_argument('--warm_up_epochs', '-wep', default=2, help='number of epochs for warming up')
    parser.add_argument('--epochs', '-ep', default=2, help='number of epochs')
    parser.add_argument('--save_checkpoint', '-sc', default='models/', help='where to save models; make sure to include / at the end or a prefix')
    parser.add_argument('--seed', '-sd', default=42, help='seed to set it at')
    args = parser.parse_args()
    main(args)


