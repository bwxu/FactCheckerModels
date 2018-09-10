'''
definitions_of_models.py

This file contains many tensorflow model architectures for extracting features
from sentences.
'''

from keras.layers import Activation, AveragePooling1D, Bidirectional, \
                        Concatenate, Conv1D, Dense, Dropout, Embedding, \
                        Flatten, Input, LSTM, MaxPooling1D, Permute
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam

import tensorflow as tf
import numpy as np
import var


def conv_layer(pooling, input_dimension):
    '''
    Generic convolutional layer. Applies relu activation along with user
    specified filters, pooling type, and embedding dimensions.

    Inputs:
      pooling: what kind of pooling to use (MAX, AVG, MAXOUT)
      input_dimension: dimension of input embeddings

    Outputs:
      keras Model object representing a convolutional layer
    '''
    # Create the convolution layer which involves using different filter sizes
    input_node = Input(shape=(var.MAX_SEQUENCE_LENGTH, input_dimension))
    conv_list = []

    for index, filter_size in enumerate(var.FILTER_SIZE_LIST):
        num_filters = var.NUM_FILTERS[index]
        conv = Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            activation='relu')(input_node)
        if pooling == "MAX":
            pool = MaxPooling1D(pool_size=int(conv.shape[1]))(conv)
        elif pooling == "AVG":
            pool = AveragePooling1D(pool_size=int(conv.shape[1]))(conv)
        elif pooling == "MAXOUT":
            conv = Permute((2, 1))(conv)
            pool = MaxPooling1D(pool_size=int(conv.shape[1]))(conv)
        else:
            raise Exception("Invalid pooling parameter")

        flatten = Flatten()(pool)
        conv_list.append(flatten)

    concat = Concatenate()
    conv_output = concat(conv_list)
    return Model(inputs=input_node, outputs=conv_output)


def add_aux_metadata(
        in_list,
        out_list,
        subject=False,
        party=False,
        credit=False,
        pos=False):
    '''
    Adds input and output tensors to in_list and out_list respectively based
    on whether the subject, party, credit, and pos metadata is used.

    Inputs:
      in_list: list representing input tensors to be modified
      out_list: list representing output tensors to be modified
      subject: boolean representing whether the subject metadata is used
      party: boolean representing whether the party metadata is used
      credit: boolean representing whether the credit metadata is used
      pos: boolean representing whether the part of speech metadata is used
    '''
    # Create auxiliary metadata models
    if subject:
        print("Using Subject Metadata")
        aux_in_subject = Input(
            shape=(
                var.NUM_SUBJECTS,
            ),
            dtype='float32',
            name='aux_in_subject')
        in_list.append(aux_in_subject)
        out_list.append(aux_in_subject)
    if party:
        print("Using Party Metadata")
        aux_in_party = Input(
            shape=(
                var.NUM_PARTIES,
            ),
            dtype='float32',
            name='aux_in_party')
        in_list.append(aux_in_party)
        out_list.append(aux_in_party)
    if credit:
        print("Using Credit Metadata")
        aux_in_credit = Input(
            shape=(
                var.NUM_CREDIT_TYPES,
            ),
            dtype='float32',
            name='aux_in_credit')
        in_list.append(aux_in_credit)
        out_list.append(aux_in_credit)
    if pos:
        print("Using Part of Speech Metadata")
        aux_in_pos = Input(
            shape=(
                var.POS_TAG_SET_LENGTH,
            ),
            dtype='float32',
            name='aux_in_pos')
        in_list.append(aux_in_pos)
        out_list.append(aux_in_pos)


def cnn_model(
        embedding_matrix,
        num_words,
        pooling="MAX",
        subject=False,
        party=False,
        credit=False,
        pos=False):
    '''
    Creates a CNN model with a Convolutional layer followed by a generic
    hidden layer and softmax combination.

    Inputs:
      embedding_matrix: matrix of embeddings derived from Word2Vec or GloVe
      num_words: number of words in the embedding matrix
      pooling: The type of pooling to use (MAX, AVG, MAXOUT)
      subject: boolean representing whether the subject metadata is used
      party: boolean representing whether the party metadata is used
      credit: boolean representing whether the credit metadata is used
      pos: boolean representing whether the part of speech metadata is used

    Output:
      Returns keras Model which represents a complete CNN neural network
    '''
    # Maintain lists of model inputs and outputs
    # Create main embedding model
    main_in = Input(
        shape=(
            var.MAX_SEQUENCE_LENGTH,
        ),
        dtype='int32',
        name='main_in')
    main_out = Embedding(input_dim=num_words + 1,
                         output_dim=var.EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=var.MAX_SEQUENCE_LENGTH,
                         trainable=var.TRAIN_EMBEDDINGS)(main_in)
    main_out = conv_layer(pooling, var.EMBEDDING_DIM)(main_out)
    main_out = Dropout(rate=var.DROPOUT_PROB)(main_out)

    in_list = [main_in]
    out_list = [main_out]

    add_aux_metadata(in_list, out_list, subject, party, credit, pos)

    # Combine main model and auxilary model
    if len(out_list) > 1:
        combined_out = Concatenate()(out_list)
    else:
        combined_out = out_list

    hidden = Dense(var.HIDDEN_LAYER_SIZE, activation='relu')(combined_out)
    hidden = Dropout(rate=var.DROPOUT_PROB)(hidden)
    predictions = Dense(len(var.LABEL_MAPPING), activation='softmax')(hidden)
    model = Model(inputs=in_list, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])

    return model


def bi_lstm_model(
        embedding_matrix,
        num_words,
        subject=False,
        party=False,
        credit=False,
        pos=False):
    '''
    Creates a Bi-LSTM model using a Bidirection LSTM layer followed by a
    generic hidden layer and softmax combination.

    Inputs:
      embedding_matrix: matrix of embeddings derived from Word2Vec or GloVe
      num_words: number of words in the embedding matrix
      pooling: The type of pooling to use (MAX, AVG, MAXOUT)
      subject: boolean representing whether the subject metadata is used
      party: boolean representing whether the party metadata is used
      credit: boolean representing whether the credit metadata is used
      pos: boolean representing whether the part of speech metadata is used

    Output:
      Returns keras Model which represents a complete Bi-LSTM neural network
    '''
    # Create main embedding model
    main_in = Input(
        shape=(
            var.MAX_SEQUENCE_LENGTH,
        ),
        dtype='int32',
        name='main_in')
    main_out = Embedding(input_dim=num_words + 1,
                         output_dim=var.EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=var.MAX_SEQUENCE_LENGTH,
                         trainable=var.TRAIN_EMBEDDINGS)(main_in)
    main_out = Bidirectional(LSTM(var.LSTM_OUT_DIM,
                                  dropout=var.LSTM_DROPOUT,
                                  recurrent_dropout=var.LSTM_DROPOUT,
                                  return_sequences=True))(main_out)
    main_out = Flatten()(main_out)
    main_out = Dropout(rate=var.DROPOUT_PROB)(main_out)

    in_list = [main_in]
    out_list = [main_out]

    add_aux_metadata(in_list, out_list, subject, party, credit, pos)

    # Combine main model and auxilary model
    if len(out_list) > 1:
        combined_out = Concatenate()(out_list)
    else:
        combined_out = out_list

    hidden = Dense(var.HIDDEN_LAYER_SIZE, activation='relu')(combined_out)
    hidden = Dropout(rate=var.DROPOUT_PROB)(hidden)
    predictions = Dense(len(var.LABEL_MAPPING), activation='softmax')(hidden)
    model = Model(inputs=in_list, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])

    return model


def bi_lstm_cnn_model(
        embedding_matrix,
        num_words,
        pooling="MAX",
        subject=False,
        party=False,
        credit=False,
        pos=False):
    '''
    Creates a stacked Bi-LSTM and CNN model involving a Bidirectional LSTM
    layer followed by a Convolutional layer followed by a generic hidden
    layer and softmax combination.

    Inputs:
      embedding_matrix: matrix of embeddings derived from Word2Vec or GloVe
      num_words: number of words in the embedding matrix
      pooling: The type of pooling to use (MAX, AVG, MAXOUT)
      subject: boolean representing whether the subject metadata is used
      party: boolean representing whether the party metadata is used
      credit: boolean representing whether the credit metadata is used
      pos: boolean representing whether the part of speech metadata is used

    Output:
      Returns keras Model which represents a complete stacked Bi-LSTM and CNN
      neural network
    '''
    # Create main embedding model
    main_in = Input(
        shape=(
            var.MAX_SEQUENCE_LENGTH,
        ),
        dtype='int32',
        name='main_in')
    main_out = Embedding(input_dim=num_words + 1,
                         output_dim=var.EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=var.MAX_SEQUENCE_LENGTH,
                         trainable=var.TRAIN_EMBEDDINGS)(main_in)
    main_out = Bidirectional(LSTM(var.LSTM_OUT_DIM,
                                  dropout=var.LSTM_DROPOUT,
                                  recurrent_dropout=var.LSTM_DROPOUT,
                                  return_sequences=True))(main_out)
    main_out = conv_layer(pooling, var.LSTM_OUT_DIM * 2)(main_out)
    main_out = Dropout(rate=var.DROPOUT_PROB)(main_out)

    in_list = [main_in]
    out_list = [main_out]

    add_aux_metadata(in_list, out_list, subject, party, credit, pos)

    # Combine main model and auxilary model
    if len(out_list) > 1:
        combined_out = Concatenate()(out_list)
    else:
        combined_out = out_list

    hidden = Dense(var.HIDDEN_LAYER_SIZE, activation='relu')(combined_out)
    hidden = Dropout(rate=var.DROPOUT_PROB)(hidden)
    predictions = Dense(len(var.LABEL_MAPPING), activation='softmax')(hidden)
    model = Model(inputs=in_list, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])

    return model


def cnn_bi_lstm_model(
        embedding_matrix,
        num_words,
        subject=False,
        party=False,
        credit=False,
        pos=False):
    '''
    Creates a stacked CNN and Bi-LSTM model involving a Convolutional layer
    followed by a Bidirectional LSTM layer followed by a generic hidden layer
    and softmax combination.

    Inputs:
      embedding_matrix: matrix of embeddings derived from Word2Vec or GloVe
      num_words: number of words in the embedding matrix
      pooling: The type of pooling to use (MAX, AVG, MAXOUT)
      subject: boolean representing whether the subject metadata is used
      party: boolean representing whether the party metadata is used
      credit: boolean representing whether the credit metadata is used
      pos: boolean representing whether the part of speech metadata is used

    Output:
      Returns keras Model which represents a complete stacked CNN and Bi-LSTM
      neural network
    '''
    # Create main embedding model
    main_in = Input(
        shape=(
            var.MAX_SEQUENCE_LENGTH,
        ),
        dtype='int32',
        name='main_in')
    main_out = Embedding(input_dim=num_words + 1,
                         output_dim=var.EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=var.MAX_SEQUENCE_LENGTH,
                         trainable=var.TRAIN_EMBEDDINGS)(main_in)
    main_out = Conv1D(var.SINGLE_NUM_FILTERS,
                      var.SINGLE_FILTER_SIZE,
                      activation='relu')(main_out)
    main_out = Bidirectional(LSTM(var.LSTM_OUT_DIM,
                                  dropout=var.LSTM_DROPOUT,
                                  recurrent_dropout=var.LSTM_DROPOUT,
                                  return_sequences=True))(main_out)
    main_out = Flatten()(main_out)
    main_out = Dropout(rate=var.DROPOUT_PROB)(main_out)

    in_list = [main_in]
    out_list = [main_out]

    add_aux_metadata(in_list, out_list, subject, party, credit, pos)

    # Combine main model and auxilary model
    if len(out_list) > 1:
        combined_out = Concatenate()(out_list)
    else:
        combined_out = out_list

    hidden = Dense(var.HIDDEN_LAYER_SIZE, activation='relu')(combined_out)
    hidden = Dropout(rate=var.DROPOUT_PROB)(hidden)
    predictions = Dense(len(var.LABEL_MAPPING), activation='softmax')(hidden)
    model = Model(inputs=in_list, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])

    return model


def parallel_cnn_bi_lstm_model(
        embedding_matrix,
        num_words,
        pooling="MAX",
        subject=False,
        party=False,
        credit=False,
        pos=False):
    '''
    Creates a parallel CNN and Bi-LSTM model involving a Convolutional layer
    and a Bidirectional LSTM layer which are combined and fed into a generic
    hidden layer and softmax combination.

    Inputs:
      embedding_matrix: matrix of embeddings derived from Word2Vec or GloVe
      num_words: number of words in the embedding matrix
      pooling: The type of pooling to use (MAX, AVG, MAXOUT)
      subject: boolean representing whether the subject metadata is used
      party: boolean representing whether the party metadata is used
      credit: boolean representing whether the credit metadata is used
      pos: boolean representing whether the part of speech metadata is used

    Output:
      Returns keras Model which represents a complete parallel CNN and Bi-LSTM
      neural network
    '''

    # input handling
    main_in = Input(
        shape=(
            var.MAX_SEQUENCE_LENGTH,
        ),
        dtype='int32',
        name='main_in')
    main_out = Embedding(input_dim=num_words + 1,
                         output_dim=var.EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=var.MAX_SEQUENCE_LENGTH,
                         trainable=var.TRAIN_EMBEDDINGS)(main_in)

    # lstm branch
    lstm_out = Bidirectional(LSTM(var.LSTM_OUT_DIM,
                                  dropout=var.LSTM_DROPOUT,
                                  recurrent_dropout=var.LSTM_DROPOUT,
                                  return_sequences=True))(main_out)
    lstm_out = Flatten()(lstm_out)
    lstm_out = Dropout(rate=var.DROPOUT_PROB)(lstm_out)

    # cnn branch
    cnn_out = conv_layer(pooling, var.EMBEDDING_DIM)(main_out)
    cnn_out = Dropout(rate=var.DROPOUT_PROB)(cnn_out)

    in_list = [main_in]
    out_list = [lstm_out, cnn_out]

    add_aux_metadata(in_list, out_list, subject, party, credit, pos)

    # Combine main model and auxilary model
    if len(out_list) > 1:
        combined_out = Concatenate()(out_list)
    else:
        combined_out = out_list

    hidden = Dense(var.HIDDEN_LAYER_SIZE, activation='relu')(combined_out)
    hidden = Dropout(rate=var.DROPOUT_PROB)(hidden)
    predictions = Dense(len(var.LABEL_MAPPING), activation='softmax')(hidden)
    model = Model(inputs=in_list, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])

    return model
