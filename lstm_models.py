from keras.layers import Activation, Bidirectional, Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, Input, LSTM, Permute
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam

import tensorflow as tf
import numpy as np
import var

def lstm_model(embedding_matrix, num_words):
    # Model Definition
    model = Sequential()
    model.add(Embedding(num_words + 1,
                        var.EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=var.MAX_SEQUENCE_LENGTH,
                        trainable=var.TRAIN_EMBEDDINGS))
    model.add(LSTM(var.LSTM_OUT_DIM, return_sequences=True))
    model.add(Flatten())
    model.add(Dropout(rate=var.DROPOUT_PROB))
    model.add(Dense(100))
    model.add(Dropout(rate=var.DROPOUT_PROB))
    model.add(Activation('relu'))
    model.add(Dense(len(var.LABEL_MAPPING), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])
   
    return model

