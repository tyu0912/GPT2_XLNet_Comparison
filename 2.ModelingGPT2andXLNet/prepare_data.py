import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", default=None, type=str, required=True, help="which speech?")
    parser.add_argument("--level", default=None, type=str, required=True, help="which level?")

    args = parser.parse_args()
    in_path = '../1.DataPreparationResults/' + args.file 
    
    fh = open(in_path).read().split('<speech_sep>')

    if args.level == 'speech':
        out_path = 'processed_data/speech_level_train/' + args.file
        out = open(out_path, 'w')

        for sentence in fh:
            sentence = sentence.strip()
            sentence = '<sod> ' + sentence + '. <eod>\n'
            out.write(sentence)
                                                                                                            
    elif args.level == 'sentence':
        out_path = 'processed_data/sentence_level_train/' + args.file
        out = open(out_path, 'w')

        fh = list(map(lambda x: x.split('.'), fh))

        fh = [item for sublist in fh for item in sublist] 

        fh = list(map(lambda x: x.strip(), fh))

        while "''" in fh:
            fh = fh.remove('')
        
        for i, sentence in enumerate(fh):
            try:
                sentence = '<sod> ' + sentence + '.' + fh[i+1] + '. <eod>\n'
                out.write(sentence)
            except:
                pass

    elif args.level == 'test':
        out_path = 'testing_data_run/' + args.file
        out = open(out_path, 'w')

        fh = list(map(lambda x: x.split('.'), fh))
        fh = [item for sublist in fh for item in sublist]
        fh = list(map(lambda x: x.strip(), fh))

        while "''" in fh:
            fh = fh.remove('')

        for sentence in fh:
            try:
                sentence = '<sod> ' + sentence + '.\n'
                out.write(sentence)
            except:
                pass

            
if __name__ == "__main__":
    main()
