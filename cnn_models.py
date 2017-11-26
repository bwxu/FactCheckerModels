from keras.layers import Activation, AveragePooling1D, Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, Input, MaxPooling1D, Permute
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


def cnn_model(embedding_matrix, num_words, pooling="MAX"):
    # Model Definition
    model = Sequential()
    model.add(Embedding(num_words + 1,
                        var.EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=var.MAX_SEQUENCE_LENGTH,
                        trainable=var.TRAIN_EMBEDDINGS))
    model.add(conv_layer(pooling))
    model.add(Dropout(rate=var.DROPOUT_PROB))
    model.add(Dense(100))
    model.add(Dropout(rate=var.DROPOUT_PROB))
    model.add(Activation('relu'))
    model.add(Dense(len(var.LABEL_MAPPING), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])
   
    return model


def cnn_model_with_subject(embedding_matrix, num_words, pooling="MAX"):
    # Create main embedding model
    main_in = Input(shape=(var.MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_in')
    main_out = Embedding(input_dim=num_words + 1,
                         output_dim=var.EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=var.MAX_SEQUENCE_LENGTH,
                         trainable=var.TRAIN_EMBEDDINGS)(main_in)
    main_out = conv_layer(pooling)(main_out)
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

def cnn_model_with_party(embedding_matrix, num_words, pooling="MAX"):
    # Create main embedding model
    main_in = Input(shape=(var.MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_in')
    main_out = Embedding(input_dim=num_words + 1,
                         output_dim=var.EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=var.MAX_SEQUENCE_LENGTH,
                         trainable=var.TRAIN_EMBEDDINGS)(main_in)
    main_out = conv_layer(pooling)(main_out)
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

def cnn_model_with_credit(embedding_matrix, num_words, pooling="MAX"):
    # Create main embedding model
    main_in = Input(shape=(var.MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_in')
    main_out = Embedding(input_dim=num_words + 1,
                         output_dim=var.EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=var.MAX_SEQUENCE_LENGTH,
                         trainable=var.TRAIN_EMBEDDINGS)(main_in)
    main_out = conv_layer(pooling)(main_out)
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

def cnn_model_with_all(embedding_matrix, num_words, pooling="MAX"):
    # Create main embedding model
    main_in = Input(shape=(var.MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_in')
    main_out = Embedding(input_dim=num_words + 1,
                         output_dim=var.EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=var.MAX_SEQUENCE_LENGTH,
                         trainable=var.TRAIN_EMBEDDINGS)(main_in)
    main_out = conv_layer(pooling)(main_out)
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

