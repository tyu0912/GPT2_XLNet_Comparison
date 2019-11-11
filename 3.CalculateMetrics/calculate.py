import sys
import argparse
import os

sys.path.insert(1, '../Metrics')

import bert_score.bert_score.bert_score.score as score
import nmt.nmt.nmt.scripts.rouge as rouge
import nmt.nmt.nmt.scripts.bleu as bleu

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, type=str, required=True, help="file path")
    parser.add_argument("--out", default="result.txt", type=str, required=False, help="out name")
    
    args = parser.parse_args()

    fh = open(args.file).read().splitlines()
    result = open(args.out, 'w')

    inp = []
    actual = []
    pred = []
    bleu_res = []
    rouge_n = []
    rouge_l = []
    
    counter = 0

    for lines in fh:
        
        lines = lines.split('|')

        input_sentence = lines[0] if len(lines) >= 1 else "<NO_VAL>"
        actual_sentence = lines[1] if len(lines) >= 2 else "<NO_VAL>"
        pred_sentence = lines[2] if len(lines) >= 3 else "<NO_VAL>"

        inp.append(input_sentence)
        actual.append(actual_sentence)
        pred.append(pred_sentence)

        bleu_res.append(bleu.compute_bleu(actual_sentence, pred_sentence)[0])
        rouge_n.append(rouge.rouge_n(actual_sentence, pred_sentence, n=2)[0])
        rouge_l.append(rouge.rouge_l_sentence_level(actual_sentence, pred_sentence)[0])

        counter += 1
        if counter % 1000 == 0:
            print(f"calculated {counter} values")
            


    BERT_P, BERT_R, BERT_F1 = score(actual, pred, lang='en', verbose=True)

    all_data = zip(inp, actual, pred, bleu_res, rouge_n, rouge_l, BERT_P.tolist(), BERT_R.tolist(), BERT_F1.tolist())
    result.write("Input,Actual,Prediction,Bleu Score,Rouge N,Rouge L,BERT Precision,BERT Recall,BERT F1\n")


    print("Writing data")
    for data in all_data:
        data = list(map(str, data))
        result.write('|'.join(data) + '\n')


if __name__ == '__main__':
    main()
