from keras.layers import Activation, Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, Input, MaxPooling1D
from keras.models import Model, Sequential
from keras.optimizers import SGD

# Arguments for preparing sentences, labels, and embedding matrix
LABEL_MAPPING = {"pants-fire": 0, 
                 "false": 1,
                 "barely-true": 2,
                 "half-true": 3,
                 "mostly-true": 4,
                 "true": 5}
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 67
EMBEDDING_DIM = 300

# Parameters for model construction
TRAIN_EMBEDDINGS = True
FILTER_SIZE_LIST = [2, 3, 4]
NUM_FILTERS = [128, 128, 128]
DROPOUT_PROB = 0.2

# Training parameters
NUM_EPOCHS = 20
BATCH_SIZE = 64

# Parameters for subject augmentation
NUM_SUBJECTS = 143


def cnn_model(embedding_matrix, num_words):
    # Create the convolution layer which involves using different filter sizes
    input_node = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
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

    # Model Definition
    model = Sequential()
    model.add(Embedding(num_words + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=TRAIN_EMBEDDINGS))
    model.add(conv_layer)
    model.add(Dropout(rate=DROPOUT_PROB))
    model.add(Dense(50))
    model.add(Dropout(rate=DROPOUT_PROB))
    model.add(Activation('relu'))
    model.add(Dense(len(LABEL_MAPPING), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(),
                  metrics=['acc'])
    model.summary()
   
    return model


def cnn_model_with_subject(embedding_matrix, num_words):
    # Create the convolution layer which involves using different filter sizes
    input_node = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
    conv_list = []
    for index, filter_size in enumerate(FILTER_SIZE_LIST):
        num_filters = NUM_FILTERS[index]
        conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(input_node)
        pool = MaxPooling1D(pool_size=int(conv.shape[1]))(conv)
        flatten = Flatten()(pool)
        conv_list.append(flatten)
    conv_output = Concatenate()(conv_list)
    conv_layer = Model(inputs=input_node, outputs=conv_output)

    # Create main embedding model
    main_in = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_in')
    main_out = Embedding(input_dim=num_words + 1,
                         output_dim=EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=MAX_SEQUENCE_LENGTH,
                         trainable=TRAIN_EMBEDDINGS)(main_in)
    main_out = conv_layer(main_out)
    main_out = Dropout(rate=DROPOUT_PROB)(main_out)

    # Create auxiliary subject model
    aux_in = Input(shape=(NUM_SUBJECTS,), dtype='float32', name='aux_in')

    # Combine main model and auxilary model
    combined_out = Concatenate()([main_out, aux_in])
    combined_layer = Model(inputs=[main_in, aux_in], outputs=combined_out)
    
    # Model Definition
    model = Sequential()
    model.add(combined_layer)
    model.add(Dense(50))
    model.add(Dropout(rate=DROPOUT_PROB))
    model.add(Activation('relu'))
    model.add(Dense(len(LABEL_MAPPING), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(),
                  metrics=['acc'])
    model.summary()
   
    return model

