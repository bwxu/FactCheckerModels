from __future__ import print_function

import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, Input, MaxPooling1D
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
LABEL_MAPPING = {"pants-fire": 0, 
                 "false": 1,
                 "barely-true": 2,
                 "half-true": 3,
                 "mostly-true": 4,
                 "true": 5}
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300

# Parameters for Model
TRAIN_EMBEDDINGS = True
FILTER_SIZE_LIST = [2, 3, 4]
NUM_FILTERS = [128, 128, 128]
DROPOUT_PROB = 0.2

# Training Parameters
NUM_EPOCHS = 10
BATCH_SIZE = 64

# where to save the model
FOLDER_NAME = "trained_models"
FILE_NAME = "cnn.weights.hdf5"

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
    print("-- DONE --")
    
    print("Creating model... ")
    model = cnn_model(embedding_matrix, num_words)
    print("-- DONE --")
    
    # Save trained model after each epoch
    checkpoint_file = os.path.join(FOLDER_NAME, FILE_NAME)
    checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True)
    callbacks = [checkpoint]
    
    print("Training model... ")
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
    print("-- DONE --")

    print("Testing trained model... ")
    # Run trained model on test_sequences
    test_data = get_input_data(TEST_DATA_PATH)
    test_labels = [data[1] for data in test_data]
    test_sentences = [data[2] for data in test_data]
    tokenizer.fit_on_texts(test_sentences)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test_labels = [LABEL_MAPPING[label] for label in test_labels]
    y_test = to_categorical(np.asarray(test_labels))
    
    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print()
    print("test loss = %0.4f, test acc = %0.4f" % (score[0], score[1]))
    print("-- DONE --")

def cnn_model(embedding_matrix, num_words):
    embedding_layer = Embedding(num_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=TRAIN_EMBEDDINGS)
    
    input_node = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
    
    # Create the convolution layer which involves using different filter sizes
    conv_list = []
    for index, filter_size in enumerate(FILTER_SIZE_LIST):
        num_filters = NUM_FILTERS[index]
        conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(input_node)
        pool = MaxPooling1D(pool_size=int(conv.shape[1]))(conv)
        flatten = Flatten()(pool)
        conv_list.append(flatten)
    
    concat = Concatenate()
    conv_output = concat(conv_list)
    
    conv_layer = Model(inputs=input_node, outputs=conv_output)

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
    model.summary()
    return model

if __name__ == '__main__':
    main()
