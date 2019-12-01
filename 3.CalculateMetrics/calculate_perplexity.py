import math
import statistics
import argparse
import torch
import os
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer)
}

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def score(sentence, tokenizer, model, args):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

    tensor_input = tensor_input.to(args.device)

    loss=model(tensor_input, labels=tensor_input)[0].item()

    return math.exp(loss)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default=None, type=str, required=True,
    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
    help="Path to pre-trained model stuff")

    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")

    parser.add_argument("--input_path", default=None, type=str, required=True, 
    help="Full path of the input text file")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--out", required=True, help="The path for output")

    parser.add_argument("--generated", required=False, help="Run actual sentences", default=True)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)

    model.to(args.device)
    model.eval()

    input_f = open(args.input_path, 'r').read().replace("<end_line>","").split("<new_line>")[1:]
    
    filename, file_extension = os.path.splitext(args.input_path)
    output_f = open(args.out, 'w+')

    #results = []

    for l in input_f:
       
        lines = l.split('<entry>')

        input_sentence = lines[0] if len(lines) >= 1 else "<NO_VAL>"
        actual_sentence = lines[1] if len(lines) >=2 else "<NO_VAL>"
        pred_sentence = lines[2] if len(lines) >=3 else "<NO_VAL>"

        if not args.generated:
            sentence = input_sentence + pred_sentence
        else:
            sentence = input_sentence + actual_sentence

        try:

            the_score = score(sentence, tokenizer, model, args)

        except:
            print(sentence)
            continue

        output_f.write("<new_line>" + input_sentence + "<entry>" + actual_sentence + "<entry>" + pred_sentence + "<entry>" + str(the_score))
        #results.append(the_score)


    #output_f.write('OVERALL:'+str(statistics.mean(results)))


if __name__ == '__main__':
    main()
