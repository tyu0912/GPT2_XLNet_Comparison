import torch
import pandas as pd
import io
import numpy as np
import argparse

from tqdm import tqdm, trange
from transformers import AdamW
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def do_tokenize(tokenizer, args, sentences):
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=args.max_length, dtype="long", truncating="post", padding="post")

    # Create attention mask of 1s for each token followed by 0s for padding
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    
    return input_ids, attention_masks


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")

    parser.add_argument("--input_path", default=None, type=str, required=True, 
    help="Full path of the input text file")

    parser.add_argument("--output_dir", default=None, type=str, required=True, 
    help="Output of training")

    parser.add_argument("--max_length", default=128, type=int, help="Max length of sentences")

    parser.add_argument("--epochs", default=4, type=int, help="Number of training epochs. Recommend between 2 and 4")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size of training. Recommend a batch size of 32, 48, or 128")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    train_sentences = open(f"{args.input_path}/all_obama.txt").read().splitlines()
    train_sentences = [sentence + " [SEP] [CLS]" for sentence in train_sentences]

    validation_sentences = open(f"{args.input_path}/all_reagan.txt").read().splitlines()
    validation_sentences = [sentence + " [SEP] [CLS]" for sentence in validation_sentences]

    # Tokenize the sentences and convert to index
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

    train_inputs, train_masks = do_tokenize(tokenizer, args, train_sentences)
    validation_inputs, validation_masks = do_tokenize(tokenizer, args, validation_sentences)

    train_labels = [1]*len(train_sentences)
    validation_labels = [1]*len(validation_sentences)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.batch_size)

    #### Training from here
    # Load XLNEtForSequenceClassification, the pretrained XLNet model with a single linear classification layer on top. 

    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
    #model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]


    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    # Store our loss and accuracy for plotting
    train_loss_set = []

    
    for _ in trange(args.epochs, desc="Epoch"):
        # Training
        # trange is a tqdm wrapper around the normal python range
    
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
           
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
           
            # Forward pass
            outputs = model(b_input_ids.to(torch.int64), token_type_ids=None, attention_mask=b_input_mask.to(torch.int64), labels=b_labels.to(torch.int64))
            loss = outputs[0]
            logits = outputs[1]
            train_loss_set.append(loss.item())    
            
            # Backward pass
            loss.backward()
            
            # Update parameters and take a step using the computed gradient
            optimizer.step()
              
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        
        # Validation
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and speeding up validation

            with torch.no_grad():
                # Forward pass, calculate logit predictions
                output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = output[0]
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))


    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


if __name__ == "__main__":
    main()






