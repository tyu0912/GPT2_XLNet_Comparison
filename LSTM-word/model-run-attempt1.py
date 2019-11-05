#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Reference: https://github.com/vlraik/word-level-rnn-keras/blob/master/lstm_text_generation.py

import random
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# In[2]:


dir_split = f"../1.DataPreparationResults/obama"

file_train = open(f"{dir_split}/train.txt", 'r').read()
file_val = open(f"{dir_split}/val.txt", 'r').read()
file_test = open(f"{dir_split}/test.txt", 'r').read()

# Add spaces around <speech_sep>
# Create a set of all words in train.txt but remove <speech_sep>
word_train = set(file_train.replace("<speech_sep>", " <speech_sep> ").split())
word_train.remove("<speech_sep>")

print("total number of unique words: ",len(word_train))

word_indices = dict((c, i) for i, c in enumerate(word_train))
indices_word = dict((i, c) for i, c in enumerate(word_train))


# In[3]:


x_len = 30
x_step = 1


# In[4]:


def vectorization(file):
    sentences = []
    sentences2 = []
    next_words = []
    list_words = []

    for speech in file.split("<speech_sep>"):
        list_words = speech.split()
        # I noticed the last speech has zero word 
        # because <speech_sep> is the last character
        if len(list_words) == 0:
            break

        for i in range(0,len(list_words)-x_len, x_step):
            sentences2 = ' '.join(list_words[i: i + x_len])
            sentences.append(sentences2)
            next_words.append(list_words[i + x_len])

    x = np.zeros((len(sentences), x_len, len(word_train)), dtype=np.bool)
    y = np.zeros((len(sentences), len(word_train)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence.split()):
            # For words that don't exist in train.txt but exist in val.txt or test.txt,
            #     X[i, t] would be all zeros
            if word in word_train:
                x[i, t, word_indices[word]] = 1
        if next_words[i] in word_train:
            y[i, word_indices[next_words[i]]] = 1
            
    return x, y


# In[5]:


# Run into memory issue with huge arrays
# Reference: https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type

train_X, train_Y = vectorization(file_train)
print(train_X.shape)
print(train_Y.shape)

val_X, val_Y = vectorization(file_val)
print(val_X.shape)
print(val_Y.shape)


# In[6]:


model = keras.Sequential()
model.add(LSTM(512, input_shape=(x_len, len(word_train)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(word_train), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


# In[ ]:


checkpoint_path = "model_history_attempt1/checkpoint_model-{epoch:03d}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False)
desired_callbacks = [checkpoint]


# In[14]:


# Capture fit history
# Reference: https://chrisalbon.com/deep_learning/keras/visualize_loss_history/
history = model.fit(train_X, train_Y, epochs=200, batch_size=1280, validation_data=(val_X,val_Y), callbacks=desired_callbacks)
model.save('model_history_attempt1/checkpoint_model.hdf5') 

# In[ ]:

train_loss = history.history['loss']
val_loss = history.history['val_loss']

np.savetxt("model_history_attempt1/loss_history_train.txt", np.array(train_loss), delimiter=",")
np.savetxt("model_history_attempt1/loss_history_val.txt", np.array(val_loss), delimiter=",")


