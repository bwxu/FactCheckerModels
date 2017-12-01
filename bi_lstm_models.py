from keras.layers import Activation, Bidirectional, Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, Input, LSTM, Permute
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam

import tensorflow as tf
import numpy as np
import var

def bi_lstm_model(embedding_matrix, num_words):
    # Model Definition
    model = Sequential()
    model.add(Embedding(num_words + 1,
                        var.EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=var.MAX_SEQUENCE_LENGTH,
                        trainable=var.TRAIN_EMBEDDINGS))
    model.add(Bidirectional(LSTM(var.LSTM_OUT_DIM,
                                 dropout=var.LSTM_DROPOUT, 
                                 recurrent_dropout=var.LSTM_DROPOUT, 
                                 return_sequences=True)))
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

def bi_lstm_model_with_subject(embedding_matrix, num_words, pooling="MAX"):
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
    main_out = Dropout(rate=var.DROPOUT_PROB)(main_out)

    # Create auxiliary subject model
    aux_in = Input(shape=(var.NUM_SUBJECTS,), dtype='float32', name='aux_in')

    # Combine main model and auxilary model
    combined_out = Concatenate()([main_out, aux_in])
    combined_layer = Model(inputs=[main_in, aux_in], outputs=combined_out)
    
    # Model Definition
    model = Sequential()
    model.add(combined_layer)
    model.add(Dense(100))
    model.add(Dropout(rate=var.DROPOUT_PROB))
    model.add(Activation('relu'))
    model.add(Dense(len(var.LABEL_MAPPING), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])
   
    return model

def bi_lstm_model_with_party(embedding_matrix, num_words, pooling="MAX"):
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
    main_out = Dropout(rate=var.DROPOUT_PROB)(main_out)
    # Create auxiliary party model
    aux_in = Input(shape=(var.NUM_PARTIES,), dtype='float32', name='aux_in')

    # Combine main model and auxilary model
    combined_out = Concatenate()([main_out, aux_in])
    combined_layer = Model(inputs=[main_in, aux_in], outputs=combined_out)
    
    # Model Definition
    model = Sequential()
    model.add(combined_layer)
    model.add(Dense(100))
    model.add(Dropout(rate=var.DROPOUT_PROB))
    model.add(Activation('relu'))
    model.add(Dense(len(var.LABEL_MAPPING), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])
   
    return model

def bi_lstm_model_with_credit(embedding_matrix, num_words, pooling="MAX"):
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
    main_out = Dropout(rate=var.DROPOUT_PROB)(main_out)

    # Create auxiliary credit model
    aux_in = Input(shape=(var.NUM_CREDIT_TYPES,), dtype='float32', name='aux_in')

    # Combine main model and auxilary model
    combined_out = Concatenate()([main_out, aux_in])
    combined_layer = Model(inputs=[main_in, aux_in], outputs=combined_out)
    
    # Model Definition
    model = Sequential()
    model.add(combined_layer)
    model.add(Dense(100))
    model.add(Dropout(rate=var.DROPOUT_PROB))
    model.add(Activation('relu'))
    model.add(Dense(len(var.LABEL_MAPPING), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])
   
    return model

def bi_lstm_model_with_all(embedding_matrix, num_words, pooling="MAX"):
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
    main_out = Dropout(rate=var.DROPOUT_PROB)(main_out)

    # Create auxiliary metadata models
    aux_in_subject = Input(shape=(var.NUM_SUBJECTS,), dtype='float32', name='aux_in_subject')
    aux_in_party = Input(shape=(var.NUM_PARTIES,), dtype='float32', name='aux_in_party')
    aux_in_credit = Input(shape=(var.NUM_CREDIT_TYPES,), dtype='float32', name='aux_in_credit')

    # Combine main model and auxilary model
    combined_out = Concatenate()([main_out, aux_in_subject, aux_in_party, aux_in_credit])
    combined_layer = Model(inputs=[main_in, aux_in_subject, aux_in_party, aux_in_credit], outputs=combined_out)
    
    # Model Definition
    model = Sequential()
    model.add(combined_layer)
    model.add(Dense(100))
    model.add(Dropout(rate=var.DROPOUT_PROB))
    model.add(Activation('relu'))
    model.add(Dense(len(var.LABEL_MAPPING), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])
   
    return model

