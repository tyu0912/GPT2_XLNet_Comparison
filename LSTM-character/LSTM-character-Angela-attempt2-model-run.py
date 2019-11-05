#!/usr/bin/env python
# coding: utf-8

# Reference:
# https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/

# ### Section 0: Import packages

# In[1]:


import numpy as np
import sys
import re
import nltk
# from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
# from keras.utils import np_utils
# from keras.callbacks import ModelCheckpoint


# In[2]:


tf.__version__


# In[3]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[4]:


# if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
#     # Reference: https://towardsdatascience.com/optimize-your-cpu-for-deep-learning-424a199d7a87
#     NUM_PARALLEL_EXEC_UNITS = 4
#     # Reference: https://stackoverflow.com/questions/56127592/attributeerror-module-tensorflow-has-no-attribute-configproto
#     config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
#                            allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
#     # Reference: https://stackoverflow.com/questions/55142951/tensorflow-2-0-attributeerror-module-tensorflow-has-no-attribute-session
#     session = tf.compat.v1.Session(config=config)
#     # Reference: https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/backend/set_session
#     tf.compat.v1.keras.backend.set_session(session)


# ### Section 1: Select Training/Validaiton/Test Files

# _Modeler Input:_  
# * Select a president to build models on  
# * Select the percentages of files in training, validation and test sets

# In[4]:


from os import listdir
from os.path import isfile, join
# Select a president to build models on
dir_president = "../CorpusOfPresidentialSpeeches/obama"
# split_pct = [training_pct, validation_pct, test_pct]
split_pct = [.4, .4, .3]
# Use x number of characters/digits to predict the next character
seq_length = 100
# Set sed number
np.random.seed(266)


# Select training/validaiton/test files

# In[5]:


# onlyfiles contains a list of files (not directories) under path_president
# Reference: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
onlyfiles_lst = [f for f in listdir(dir_president) if isfile(join(dir_president, f))]
num_of_files = len(onlyfiles_lst)
# Reference: https://stackoverflow.com/questions/15511349/select-50-items-from-list-at-random-to-write-to-file/39585770
files_train_arr = np.random.choice(onlyfiles_lst, round(num_of_files*split_pct[0]), replace=False)
# Set substraction: https://stackoverflow.com/questions/3428536/python-list-subtraction-operation
files_val_test_lst = list(set(onlyfiles_lst) - set(files_train_arr))
files_val_arr = np.random.choice(files_val_test_lst, round(len(files_val_test_lst)*split_pct[1]/(split_pct[1]+split_pct[2])), replace=False)
files_test_arr = np.array(list((set(files_val_test_lst) - set(files_val_arr))))

print('Training set:')
print(files_train_arr)
print('Validation set:')
print(files_val_arr)
print('Test set:')
print(files_test_arr)


# ### Section 2: Pre-processing Data so that It Can Be Consumed by _tensorflow.keras.layers.LSTM_

# _**Questions**_:
# * Why remove special characters?  

# In[6]:


def tokenize_words(input_file):
    """
    This function accomplishes four purposes:
    1. Remove the title and date (the first two rows) from the input file
    2. Remove all special characters except for . and ,
    3. Convert all characters to lower case
    4. Tokenize words
    
    Args:
        input_file (str): input file
        
    Returns:
        output_file (str): tokenized strings separated by space
    """
    contractions_dict = {"ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have",\
                         "'cause": "because","could've": "could have","couldn't": "could not",\
                         "couldn't've": "could not have","didn't": "did not","doesn't": "does not",\
                         "don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not",\
                         "haven't": "have not","he'd": "he had","he'd've": "he would have","he'll": "he will",\
                         "he'll've": "he will have","he's": "he is","how'd": "how did","how'd'y": "how do you",\
                         "how'll": "how will","how's": "how is","I'd": "I had","I'd've": "I would have",\
                         "I'll": "I will","I'll've": "I will have","I'm": "I am","I've": "I have","isn't": "is not",\
                         "it'd": "it had","it'd've": "it would have","it'll": "it will","it'll've": "iit will have",\
                         "it's": "it is","let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have",\
                         "mightn't": "might not","mightn't've": "might not have","must've": "must have",\
                         "mustn't": "must not","mustn't've": "must not have","needn't": "need not",\
                         "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",\
                         "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",\
                         "shan't've": "shall not have","she'd": "she had","she'd've": "she would have",\
                         "she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have",\
                         "shouldn't": "should not","shouldn't've": "should not have","so've": "so have",\
                         "so's": "so is","that'd": "that had","that'd've": "that would have","that's": "that is",\
                         "there'd": "there had","there'd've": "there would have","there's": "there is",\
                         "they'd": "they had","they'd've": "they would have","they'll": "they will",\
                         "they'll've": "they will have","they're": "they are","they've": "they have",\
                         "to've": "to have","wasn't": "was not","we'd": "we had","we'd've": "we would have",\
                         "we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",\
                         "weren't": "were not","what'll": "what will","what'll've": "what will have",\
                         "what're": "what are","what's": "what is","what've": "what have","when's": "when is",\
                         "when've": "when have","where'd": "where did","where's": "where is","where've": "where have",\
                         "who'll": "who will","who'll've": "who will have","who's": "who is","who've": "who have",\
                         "why's": "why is","why've": "why have","will've": "will have","won't": "will not",\
                         "won't've": "will not have","would've": "would have","wouldn't": "would not",\
                         "wouldn't've": "would not have","y'all": "you all","y'all'd": "you all would",\
                         "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",\
                         "you'd": "you had","you'd've": "you would have","you'll": "you will","you'll've": "you will have",\
                         "you're": "you are","you've": "you have"}
    country_acronyms = {"U.S": "United States", "U.S.A": "United States of America", "U.A.E": "United Arab Emirates", \
                        "U.S.S.R": "Union of Soviet Socialist Republics"}
    
    
    # Remove the title and date (the first two rows)
    startChar = [word.end() for word in re.finditer("\n", input_file)][1]
    input2 = input_file[startChar:]
    
    # lowercase everything so that we have less tokens to predict. i.e., no need to distinguish a vs. A
    input2 = input2.lower()
    
    # Remove things in angle quotes which are added to account for crowd reactions
    input2 = re.sub(r"\<[^\>]*\>", '', input2)
    
    # lowercase everything so that we have less tokens to predict. i.e., no need to distinguish a vs. A
    input2 = input2.lower()
    
    # Standardize contractions
    for k, v in contractions_dict.items():
        input2 = input2.replace(k, v) 
        
        k_caps = k[:1].upper() + k[1:]
        v_caps = v[:1].upper() + v[1:]
        
        input2 = input2.replace(k_caps, v_caps)
        
    # Replace country acronyms
    for k, v in country_acronyms.items():
        input2 = input2.replace(k, v)
        
    # Remove middle initial
    input2 = re.sub(r"([A-Z])\W ", '', input2)

    # Keep all the words and digitis
    # Keep only two special characters: . and ,
    # If we want to keep carriage return, add |\n
    tokenizer = RegexpTokenizer(r'\w+|[\.\,]')
    tokens = tokenizer.tokenize(input2)
    output_file = " ".join(tokens)
    
    return output_file


# Define a list of all possible characters and digits in the data

# In[7]:


# It's possible that digits in the validation/test sets are not training set
# To make sure every character/digit can be converted to a number 
#     and subsequently scored appropriately for validation/test sets,
# We define chars_lst as all possible characters/digits we can observe from training/validaiton/test sets
# The code below only captures characters/digits in the training set and thus inappropriate
#     chars_lst = sorted(list(set(tokenized_file)))
# Reference: https://stackoverflow.com/questions/16060899/alphabet-range-on-python
chars_lst = [' ',',','.'] + [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'),ord('z')+1)]


# _**Question**_:
# * I don't understand the logic of converting `X` to float or divided by vocab_len so that all Xs are smaller than 1

# In[8]:


def create_x_y_num(input_file, chars_lst, seq_length):
    """
    This function creates raw input data and raw target character.
    
    Args:
        input_file (str): tokenized file
        chars_list (list): a list of all possible characters and digits in the data
        seq_length (int): the number of characters/digits as input
        
    Returns:
        x_data (list): a list of rolling ?-character sequences converted to floats
            number of elements (i.e., sequences) in the list = input_len - seq_length
            every element is an array with dimension (seq_length x 1)
        y_data (list): the next character for every rolling sequence
            number of elements = input_len - seq_length
    """
    
    # input_len - seq_length = the beginning character of the last row of input data
    # vocab_len is used to standardized the input data
    input_len = len(input_file)
    vocab_len = len(chars_lst)
    # print ("Total number of characters:", input_len)
    # print ("Total vocab:", vocab_len)
    
    # Define the dictionary that map characters/digits to numbers
    char_to_num = dict((c, i) for i, c in enumerate(chars_lst))

    # Initialize the data
    x_data_temp = []
    y_data = []
    
    # loop through inputs, start at the beginning and go until we hit
    # the final character we can create a sequence out of
    for i in range(0, input_len - seq_length, 1):
        # Define input and output sequences
        # Input is the current character plus desired sequence length
        in_seq = input_file[i:i + seq_length]

        # Out sequence is the initial character plus total sequence length
        out_seq = input_file[i + seq_length]

        # Convert list of characters to integers 
        x_data_temp.append([char_to_num[char] for char in in_seq])
        y_data.append(char_to_num[out_seq])
        
    # Convert the input sequences 
    #     (a list containning sublists, with each sublist represent a 100-character sequence)
    #     into a processed numpy array that our network can use
    n_patterns = len(x_data_temp)
    x_data_reshape = np.reshape(x_data_temp, (n_patterns, seq_length, 1))

    # Convert intergers into floats 
    # so that the sigmoid activation function our network uses can interpret them and output probabilities from 0 to 1
    x_data = list(x_data_reshape/float(vocab_len))
        
    return x_data, y_data


# In[9]:


def combine_x_y(dir_president, files_arr):
    """
    """
    X_temp = []
    Y_temp = []
    for i in range(files_arr.shape[0]):
        file = open(join(dir_president, files_arr[i])).read()
        
        # Tokenize the file
        tokenized_file = tokenize_words(file)
        
        # Create raw x and y for a given file in a format that can be merged with other files
        x_data, y_data = create_x_y_num(tokenized_file, chars_lst, seq_length)
        
        # Use extend not append
        #     append adds an element that's a list itself
        #     extend adds elements from the new list to the existing list
        # Reference: https://stackabuse.com/append-vs-extend-in-python-lists/
        X_temp.extend(x_data)
        Y_temp.extend(y_data)
    
    x = np.array(X_temp)
    # One-hot encode the label data
    y = keras.utils.to_categorical(Y_temp)
    
    return x, y


# In[10]:


train_X, train_Y = combine_x_y(dir_president, files_train_arr)
print(train_X.shape)
print(train_Y.shape)

val_X, val_Y = combine_x_y(dir_president, files_val_arr)
print(val_X.shape)
print(val_Y.shape)

test_X, test_Y = combine_x_y(dir_president, files_test_arr)
print(test_X.shape)
print(test_Y.shape)


# ### Section 3: LSTM

# In[11]:


model = keras.Sequential()
model.add(LSTM(256, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(train_Y.shape[1], activation='softmax'))


# The default learning rate for adam optimizer is 0.001.  
# (Reference: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)  
# To change the learning rate, see https://www.tensorflow.org/guide/keras/train_and_evaluate (tensor), https://keras.io/optimizers/ (keras)  
# 
# _**Note**_: maybe research on the optimizer to use??

# In[12]:


# model.compile(loss='categorical_crossentropy', optimizer='adam'(learning_rate=1e-3))
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.01))


# In[ ]:


checkpoint_path = "model_history_attempt2/checkpoint_model-{epoch:03d}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False)
desired_callbacks = [checkpoint]


# In[14]:


# Capture fit history
# Reference: https://chrisalbon.com/deep_learning/keras/visualize_loss_history/
history = model.fit(train_X, train_Y, epochs=200, batch_size=3200, validation_data=(val_X,val_Y), callbacks=desired_callbacks)
model.save('model_history_attempt2/checkpoint_model.hdf5') 

# In[ ]:

train_loss = history.history['loss']
val_loss = history.history['val_loss']

np.savetxt("model_history_attempt2/loss_history_train.txt", np.array(train_loss), delimiter=",")
np.savetxt("model_history_attempt2/loss_history_val.txt", np.array(val_loss), delimiter=",")

