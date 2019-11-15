import tensorflow as tf
opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
conf = tf.ConfigProto(gpu_options=opts)
tf.enable_eager_execution(config=conf)

import random
import sys
import os
import re
import numpy as np
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
import gensim

rootdir = f"../1.DataPreparationResults"
files_train_lst = []
files_val_lst = []
files_test_lst = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if 'train.txt' in file:
            files_train_lst.append(os.path.join(subdir, file))
        elif 'val.txt' in file:
            files_val_lst.append(os.path.join(subdir, file))
        else:
            files_test_lst.append(os.path.join(subdir, file))

num_file = len(files_train_lst)
            
file_train = open(f"{files_train_lst[0]}", 'r').read()
file_val = open(f"{files_val_lst[0]}", 'r').read()
file_test = open(f"{files_test_lst[0]}", 'r').read()

x_len = 30
x_step = 1

google_word_model = gensim.models.KeyedVectors.load_word2vec_format('../../test/GoogleNews-vectors-negative300.bin', binary=True)
pretrained_weights = google_word_model.wv.vectors
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)

def word2idx(word):
    return google_word_model.wv.vocab[word].index

def idx2word(idx):
    return google_word_model.wv.index2word[idx]

vocab = google_word_model.wv.vocab

def google_preprocess(file):
    # e.g., there is no harrison train.txt
    if len(file) != 0:
        file2 = re.sub('\d', '#', file)
        file2 = re.sub(' a ', ' A ', file2)
        file2 = re.sub(' and ', ' And ', file2)
        file2 = re.sub(' of ', ' Of ', file2)
        file2 = re.sub(' to ', ' To ', file2)
        file2 = re.sub(' , ', ' . ', file2)
        # Add spaces around <speech_sep>
        # Create a set of all words in file.txt but remove <speech_sep>
        unique_words = set(file2.replace("<speech_sep>", " <speech_sep> ").split())
        unique_words.remove("<speech_sep>")
        return file2, unique_words
    else:
        return file, set()

def file_to_sentences(file, sentences):
    sentences2 = []
    next_words = []
    list_words = []
    
    for speech in file.split("<speech_sep>"):
        list_words = speech.split()

        if len(list_words) == 0:
            break
        
        for i in range(0,len(list_words)-x_len-1, x_step):
            sentences2 = [word for word in list_words[i: i + x_len + 1]]
            sentences.append(sentences2)
            
    return sentences

def multiple_google_preprocess_to_file(files_lst):
    unique_words_lst = []
    sentences = []

    for i in range(num_file):
        file_google_i, vars()["unique_words_"+str(i)] = google_preprocess(open(f"{files_lst[i]}", 'r').read())
        sentences = file_to_sentences(file_google_i, sentences)
        unique_words_lst.append(vars()["unique_words_"+str(i)])

    unique_words_set = frozenset().union(*unique_words_lst)
    
    return sentences, unique_words_set

sentences_train, unique_words_train = multiple_google_preprocess_to_file(files_train_lst)
sentences_val, unique_words_val = multiple_google_preprocess_to_file(files_val_lst)
sentences_test, unique_words_test = multiple_google_preprocess_to_file(files_test_lst)

unique_words_all = unique_words_train.union(unique_words_val.union(unique_words_test))
print("total number of unique words: ",len(unique_words_all))

pretrained_weights_mini = []
vocab_mini_lst = []
vocab_mini_dict = dict()

# index 0: unknown words => </s>
pretrained_weights_mini.append(pretrained_weights[0])
vocab_mini_lst.append(google_word_model.wv.index2word[0])
vocab_mini_dict[google_word_model.wv.index2word[0]] = 0

# index 1: , or . => np.zeros
pretrained_weights_mini.append(np.zeros((300)))
vocab_mini_lst.append('.')
vocab_mini_dict['.'] = 1

# index 2+
i = 2
for word in unique_words_all:
    if word in google_word_model.wv.vocab:
        pretrained_weights_mini.append(pretrained_weights[google_word_model.wv.vocab[word].index])
        vocab_mini_lst.append(word)
        vocab_mini_dict[word] = i
        i += 1
        
pretrained_weights_mini = np.array(pretrained_weights_mini)
print(pretrained_weights_mini.shape)
vocab_size, emdedding_size = pretrained_weights_mini.shape
print(len(vocab_mini_lst))

def word2idx(word):
    return vocab_mini_dict[word]
 
def idx2word(idx):
    return vocab_mini_lst[idx]

def sentences_to_2darray(sentences):
    
    missing_words = set()
    
    x = np.zeros([len(sentences), x_len], dtype=np.int32)
    y = np.zeros([len(sentences)], dtype=np.int32)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence[:-1]):
            x[i, t] = vocab_mini_dict.get(word, 0)
            if x[i, t] == 0:
                missing_words.add(word)
        y[i] = vocab_mini_dict.get(sentence[-1], 0)
        if y[i] == 0:
            missing_words.add(sentence[-1])
    print(missing_words) 
        
    return x, y

train_X, train_Y = sentences_to_2darray(sentences_train)
print('train_X shape:', train_X.shape)
print('train_Y shape:', train_Y.shape)

val_X, val_Y = sentences_to_2darray(sentences_val)
print(val_X.shape)
print(val_Y.shape)

print(vocab_size)
print(emdedding_size)

model = keras.Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights_mini],trainable=True))
model.add(LSTM(emdedding_size))
model.add(Dropout(0.2))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')


checkpoint_path = "model_history_attempt4/checkpoint_model-{epoch:03d}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False)
desired_callbacks = [checkpoint]

history = model.fit(train_X, train_Y, epochs=2, batch_size=12800, validation_data=(val_X,val_Y), callbacks=desired_callbacks)
model.save('model_history_attempt4/checkpoint_model.hdf5') 

train_loss = history.history['loss']
val_loss = history.history['val_loss']

np.savetxt("model_history_attempt4/loss_history_train.txt", np.array(train_loss), delimiter=",")
np.savetxt("model_history_attempt4/loss_history_val.txt", np.array(val_loss), delimiter=",")