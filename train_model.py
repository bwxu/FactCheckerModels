import os
import numpy as np
from shutil import copyfile

from gensim.models.keyedvectors import KeyedVectors
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from parse_data import clean_credit, get_glove_vectors, get_data, get_mapping, get_one_hot_vectors, normalize_vectors, get_pos_freqs, remove_stop_words
from definitions_of_models import cnn_model, bi_lstm_model, bi_lstm_cnn_model, cnn_bi_lstm_model, parallel_cnn_bi_lstm_model
import var

def train_model():
    '''
    Trains a model using the parameters specified in var.py. The model training checkpoints
    are saved to var.FOLDER_NAME. A copy of the var.py used is also saved in that folder. Only the lowest
    val loss models for each trained model is saved.
    '''
    copyfile('var.py', os.path.join(var.FOLDER_NAME, 'var.py'))

    print("Reading word vectors... ")
    embeddings = None
    if var.USE_WORD2VEC:
        embeddings = KeyedVectors.load_word2vec_format(var.WORD2VEC_BIN_PATH, binary=True)
    else:
        embeddings = get_glove_vectors(var.GLOVE_VECTOR_PATH)
    print("--- DONE ---")

    print("Getting input data... ")
    train_labels, train_sentences, train_subjects, train_party, train_credit = get_data(var.TRAINING_DATA_PATH)
    val_labels, val_sentences, val_subjects, val_party, val_credit = get_data(var.VALIDATION_DATA_PATH)
    print("--- DONE ---")
    
    if var.NO_STOP_WORDS:    
        train_sentences = remove_stop_words(train_sentences)
        val_sentences = remove_stop_words(val_sentences)

    print("Preparing data for model... ")
    # Convert the input sentences into sequences of integers length MAX_SEQUENCE_LENGTH where
    # each integer maps to a word and the sequence only considers the first MAX_SEQUENCE_LENGTH
    # words in the statement being evaluated
    tokenizer = Tokenizer(num_words=var.MAX_NUM_WORDS)
    tokenizer.fit_on_texts(train_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    word_index = tokenizer.word_index

    x_train = pad_sequences(train_sequences, maxlen=var.MAX_SEQUENCE_LENGTH)
    x_val = pad_sequences(val_sequences, maxlen=var.MAX_SEQUENCE_LENGTH)
    y_train = to_categorical(np.asarray([var.LABEL_MAPPING[label] for label in train_labels]))
    y_val = to_categorical(np.asarray([var.LABEL_MAPPING[label] for label in val_labels]))

    # Get the Part of Speech frequencies
    x_train_pos = np.asarray(get_pos_freqs(train_sentences))
    x_val_pos = np.asarray(get_pos_freqs(val_sentences))

    # Populate SUBJECT_MAPPING with freq information from training data
    var.SUBJECT_MAPPING = get_mapping(train_subjects)

    # Create one hot vectors for the subject
    x_train_subject = np.asarray(get_one_hot_vectors(train_subjects, var.NUM_SUBJECTS, var.SUBJECT_MAPPING))
    x_val_subject = np.asarray(get_one_hot_vectors(val_subjects, var.NUM_SUBJECTS, var.SUBJECT_MAPPING))

    # Convert party to list of list format
    train_party = [[party] for party in train_party]
    val_party = [[party] for party in val_party]

    # Populate PARTY_MAPPING with training data
    var.PARTY_MAPPING = get_mapping(train_party)

    # Get One Hot Vectors representing Party
    x_train_party = np.asarray(get_one_hot_vectors(train_party, var.NUM_PARTIES, var.PARTY_MAPPING))
    x_val_party = np.asarray(get_one_hot_vectors(val_party, var.NUM_PARTIES, var.PARTY_MAPPING))

    # Remove current label from credit vector
    train_credit = clean_credit(train_labels, train_credit)
    val_credit = clean_credit(val_labels, val_credit)

    # Normalize Credit Vector
    x_train_credit = np.asarray(normalize_vectors(train_credit))
    x_val_credit = np.asarray(normalize_vectors(val_credit))

    # Create embedding matrix for embedding layer. Matrix will be 
    # (num_words + 1) x EMBEDDING_DIM since word_index starts at 1
    num_words = min(var.MAX_NUM_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, var.EMBEDDING_DIM))
    for word, rank in word_index.items():
        if rank <= var.MAX_NUM_WORDS:
            embedding = None
            if var.USE_WORD2VEC:
                if word in embeddings.vocab:
                    embedding = embeddings[word]
            else:
                embedding = embeddings.get(word, None)
            if embedding is not None:
                embedding_matrix[rank] = embedding
    print("--- DONE ---")
    
    for i in range(var.NUM_MODELS):
        print("Creating model " + str(i + 1) + " out of " + str(var.NUM_MODELS) + " ...")
        if var.MODEL_TYPE == "CNN":
            model = cnn_model(embedding_matrix, num_words, pooling=var.POOLING,
                              subject=var.USE_SUBJECTS, party=var.USE_PARTY,
                              credit=var.USE_CREDIT, pos=var.USE_POS)
        elif var.MODEL_TYPE == "BI_LSTM":
            model = bi_lstm_model(embedding_matrix, num_words,
                                  subject=var.USE_SUBJECTS, party=var.USE_PARTY,
                                  credit=var.USE_CREDIT, pos=var.USE_POS)
        elif var.MODEL_TYPE == "BI_LSTM_CNN":
            model = bi_lstm_cnn_model(embedding_matrix, num_words, pooling=var.POOLING,
                                      subject=var.USE_SUBJECTS, party=var.USE_PARTY,
                                      credit=var.USE_CREDIT, pos=var.USE_POS)
        elif var.MODEL_TYPE == "CNN_BI_LSTM":
            model = cnn_bi_lstm_model(embedding_matrix, num_words,
                                      subject=var.USE_SUBJECTS, party=var.USE_PARTY,
                                      credit=var.USE_CREDIT, pos=var.USE_POS)
        elif var.MODEL_TYPE == "PARALLEL":
            model = parallel_cnn_bi_lstm_model(embedding_matrix, num_words, pooling=var.POOLING,
                                               subject=var.USE_SUBJECTS, party=var.USE_PARTY,
                                               credit=var.USE_CREDIT, pos=var.USE_POS)
        else:
            raise Exception("Invalid MODEL_TYPE")
        print("--- DONE ---")
       
        print("Training model... ")
        # Save trained model after each epoch

        checkpoint_file = os.path.join(var.FOLDER_NAME, str(i).zfill(2) + '_' + var.FILE_NAME)
        checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True)
        callbacks = [checkpoint]

        train_input = [x_train]
        val_input = [x_val]
        if var.USE_SUBJECTS:
            train_input.append(x_train_subject)
            val_input.append(x_val_subject)
        if var.USE_PARTY:
            train_input.append(x_train_party)
            val_input.append(x_val_party)
        if var.USE_CREDIT:
            train_input.append(x_train_credit)
            val_input.append(x_val_credit)
        if var.USE_POS:
            train_input.append(x_train_pos)
            val_input.append(x_val_pos)

        print("train_input_size", len(train_input))
        
        model.fit(train_input, y_train,
                  validation_data=(val_input, y_val),
                  epochs=var.NUM_EPOCHS,
                  batch_size=var.BATCH_SIZE,
                  callbacks=callbacks)
        print("--- DONE ---")

        model.summary()
       
        del model
        
        #print("--- DONE ---")


if __name__ == '__main__':
    train_model()

