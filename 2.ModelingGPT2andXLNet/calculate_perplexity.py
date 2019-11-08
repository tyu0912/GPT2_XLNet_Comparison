import math
import statistics
import argparse
import torch
import os
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'xlnet': (XLMWithLMHeadModel, XLMTokenizer)
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

    input_f = open(args.input_path, 'r').read().splitlines()
    
    filename, file_extension = os.path.splitext(args.input_path)
    output_p = filename + '_results.txt'
    output_f = open(output_p, 'w+')

    results = []

    for l in input_f:
        the_score = score(l, tokenizer, model, args)
        output_f.write(str(the_score) + '\n')
        results.append(the_score)


    output_f.write('OVERALL:'+str(statistics.mean(results)))


if __name__ == '__main__':
    main()
