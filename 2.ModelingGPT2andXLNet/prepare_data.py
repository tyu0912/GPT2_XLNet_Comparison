import argparse



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", default=None, type=str, required=True, help="which speech?")

    args = parser.parse_args()
    in_path = '../1.DataPreparationResults/' + args.file 
    out_path = 'processed_data/speech_level_train/' + args.file

    fh = open(in_path).read().split('<speech_sep>')
    out = open(out_path, 'w')

    for sentence in fh:
        sentence = sentence.strip()
        sentence = '<sod> ' + sentence + '. <eod>\n'
        out.write(sentence)
        

if __name__ == "__main__":
    main()
