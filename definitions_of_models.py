from keras.layers import Activation, AveragePooling1D, Bidirectional, Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, Input, LSTM, MaxPooling1D, Permute
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam

import tensorflow as tf
import numpy as np
import var

def conv_layer(pooling="MAX"):
    # Create the convolution layer which involves using different filter sizes
    input_node = Input(shape=(var.MAX_SEQUENCE_LENGTH, var.EMBEDDING_DIM))
    conv_list = []
    
    for index, filter_size in enumerate(var.FILTER_SIZE_LIST):
        num_filters = var.NUM_FILTERS[index]
        conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(input_node)
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

def add_aux_metadata(in_list, out_list, subject=False, party=False, credit=False, pos=False):
     # Create auxiliary metadata models
    if subject:
        print("Using Subject Metadata")
        aux_in_subject = Input(shape=(var.NUM_SUBJECTS,), dtype='float32', name='aux_in_subject')
        in_list.append(aux_in_subject)
        out_list.append(aux_in_subject)
    if party:
        print("Using Party Metadata")
        aux_in_party = Input(shape=(var.NUM_PARTIES,), dtype='float32', name='aux_in_party')
        in_list.append(aux_in_party)
        out_list.append(aux_in_party)
    if credit:
        print("Using Credit Metadata")
        aux_in_credit = Input(shape=(var.NUM_CREDIT_TYPES,), dtype='float32', name='aux_in_credit')
        in_list.append(aux_in_credit)
        out_list.append(aux_in_credit)


def cnn_model(embedding_matrix, num_words, pooling="MAX", subject=False, party=False, credit=False, pos=False):
    # Maintain lists of model inputs and outputs
    # Create main embedding model
    main_in = Input(shape=(var.MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_in')
    main_out = Embedding(input_dim=num_words + 1,
                         output_dim=var.EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=var.MAX_SEQUENCE_LENGTH,
                         trainable=var.TRAIN_EMBEDDINGS)(main_in)
    main_out = conv_layer(pooling)(main_out)
    main_out = Dropout(rate=var.DROPOUT_PROB)(main_out)
    
    in_list = [main_in]
    out_list = [main_out]

    add_aux_metadata(in_list, out_list, subject, party, credit, pos)

    # Combine main model and auxilary model
    if len(out_list) > 1:
        combined_out = Concatenate()(out_list)
    else:
        combined_out = out_list
    combined_layer = Model(inputs=in_list, outputs=combined_out)
    
    # Model Definition
    model = Sequential()
    model.add(combined_layer)
    model.add(Dense(var.HIDDEN_LAYER_SIZE))
    model.add(Dropout(rate=var.DROPOUT_PROB))
    model.add(Activation('relu'))
    model.add(Dense(len(var.LABEL_MAPPING), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])
   
    return model

def bi_lstm_model(embedding_matrix, num_words, subject=False, party=False, credit=False, pos=False):
    # Create main embedding model
    main_in = Input(shape=(var.MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_in')
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
    combined_layer = Model(inputs=in_list, outputs=combined_out)
   
    # Model Definition
    model = Sequential()
    model.add(combined_layer)
    model.add(Dense(var.HIDDEN_LAYER_SIZE))
    model.add(Dropout(rate=var.DROPOUT_PROB))
    model.add(Activation('relu'))
    model.add(Dense(len(var.LABEL_MAPPING), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])
   
    return model

