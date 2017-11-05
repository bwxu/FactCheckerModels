from __future__ import print_function

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Conv1D, Dense, Dropout, Embedding, Flatten, Input, MaxPooling1D, Merge
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from parse_data import get_glove_vectors, get_input_data

# Location of data files
GLOVE_VECTOR_PATH = "data/glove.6B.300d.txt"
TRAINING_DATA_PATH = "data/train.tsv"
VALIDATION_DATA_PATH = "data/valid.tsv"
TEST_DATA_PATH = "data/test.tsv"

# Argument for preparing setences, labels, and embedding matrix
LABEL_MAPPING = {"pants-fire": 1, 
                 "false": 2,
                 "barely-true": 3,
                 "half-true": 4,
                 "mostly-true": 5,
                 "true": 6}
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300

# Parameters for Model
TRAIN_EMBEDDINGS = True
FILTER_SIZE_LIST = [2, 3, 4]
NUM_FILTERS = [128, 128, 128]
DROPOUT_PROB = 0.8

def main():
    train_model()

def train_model():
    print("Reading word vectors... ")
    embeddings = get_glove_vectors(GLOVE_VECTOR_PATH)
    print("-- DONE --")

    print("Getting input data... ")
    train_data = get_input_data(TRAINING_DATA_PATH)
    train_labels = [data[1] for data in train_data]
    train_sentences = [data[2] for data in train_data]

    val_data = get_input_data(VALIDATION_DATA_PATH)
    val_labels = [data[1] for data in val_data]
    val_sentences = [data[2] for data in val_data]
    print("-- DONE --")

    print("Preparing data for model... ")
    # Convert the input sentences into sequences of integers length 100 where
    # each integer maps to a word and the sequence only considers the first 100
    # words in the statement being evaluated
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(train_sentences + val_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    word_index = tokenizer.word_index

    x_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x_val = pad_sequences(val_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Convert labels to categorical variables
    train_labels = [LABEL_MAPPING[label] for label in train_labels]
    val_labels = [LABEL_MAPPING[label] for label in val_labels]

    y_train = to_categorical(np.asarray(train_labels))
    y_val = to_categorical(np.asarray(val_labels))

    # Create embedding matrix for embedding layer. Matrix will be 
    # (num_words + 1) x EMBEDDING_DIM since word_index starts at 1
    num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
    for word, rank in word_index.items():
        if rank <= MAX_NUM_WORDS:
            embedding = embeddings.get(word, None)
            if embedding is not None:
                embedding_matrix[rank] = embedding

    print(embedding_matrix.shape)
    # if the number of words was less than the max, set the max to the number of words
    print("-- DONE --")
    
    print("Creating model... ")
    create_model(embedding_matrix, num_words)
    print("-- DONE --")

def create_model(embedding_matrix, num_words):
    embedding_layer = Embedding(num_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=TRAIN_EMBEDDINGS)
    
    input_node = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
    
    conv_list = []
    for index, filter_size in enumerate(FILTER_SIZE_LIST):
        num_filters = NUM_FILTERS[index]
        conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(input_node)
        print("conv")
        print(conv.shape)
        print(conv.shape[1])
        pool = MaxPooling1D(pool_size=int(conv.shape[1]))(conv)
        print("pool")
        print(pool.shape)
        flatten = Flatten()(pool)
        print("flatten")
        print(flatten.shape)
        conv_list.append(flatten)
    
    conv_output = Merge(mode='concat')(conv_list)
    
    conv_layer = Model(input=input_node, output=conv_output)

    model = Sequential()
    model.add(embedding_layer)
    model.add(conv_layer)
    model.add(Dropout(rate=DROPOUT_PROB))
    model.add(Dense(100))
    model.add(Dropout(rate=DROPOUT_PROB))
    model.add(Activation('relu'))
    model.add(Dense(len(LABEL_MAPPING), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(),
                  metrics=['acc'])
    return model

if __name__ == '__main__':
    main()
