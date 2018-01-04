from __future__ import print_function
import os
import numpy as np
from shutil import copyfile

from gensim.models.keyedvectors import KeyedVectors
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from parse_data import clean_credit, get_glove_vectors, get_data, get_mapping, get_one_hot_vectors, normalize_vectors, get_pos_freqs 
from cnn_models import cnn_model, cnn_model_with_subject, cnn_model_with_party, cnn_model_with_credit, cnn_model_with_all
from bi_lstm_models import bi_lstm_model, bi_lstm_model_with_subject, bi_lstm_model_with_party, bi_lstm_model_with_credit, bi_lstm_model_with_all
from bi_lstm_cnn_models import bi_lstm_cnn_model, bi_lstm_cnn_model_with_subject, bi_lstm_cnn_model_with_party, bi_lstm_cnn_model_with_credit, bi_lstm_cnn_model_with_all
from cnn_bi_lstm_models import cnn_bi_lstm_model, cnn_bi_lstm_model_with_subject, cnn_bi_lstm_model_with_party, cnn_bi_lstm_model_with_credit, cnn_bi_lstm_model_with_all
from parallel_cnn_bi_lstm_models import parallel_cnn_bi_lstm_model, parallel_cnn_bi_lstm_model_with_subject, parallel_cnn_bi_lstm_model_with_credit, parallel_cnn_bi_lstm_model_with_party, parallel_cnn_bi_lstm_model_with_all
import var

def train_model():
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

    print("Preparing data for model... ")
    # Convert the input sentences into sequences of integers length 100 where
    # each integer maps to a word and the sequence only considers the first 100
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
    train_pos = get_pos_freqs(train_sentences)
    val_pos = get_pos_freqs(val_sentences)

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
            if var.USE_SUBJECTS and var.USE_PARTY and var.USE_CREDIT:
                print("  Using all subject, party, credit metadata")
                model = cnn_model_with_all(embedding_matrix, num_words, pooling=var.POOLING)
            elif var.USE_SUBJECTS:
                print("  Using Subject Metadata")
                model = cnn_model_with_subject(embedding_matrix, num_words, pooling=var.POOLING)
            elif var.USE_PARTY:
                print("  Using Party Metadata")
                model = cnn_model_with_party(embedding_matrix, num_words, pooling=var.POOLING)
            elif var.USE_CREDIT:
                print("  Using Credit Metadata")
                model = cnn_model_with_credit(embedding_matrix, num_words, pooling=var.POOLING)
            else:
                model = cnn_model(embedding_matrix, num_words, pooling=var.POOLING)
        elif var.MODEL_TYPE == "BI_LSTM":
            if var.USE_SUBJECTS and var.USE_PARTY and var.USE_CREDIT:
                print("  Using all subject, party, credit metadata")
                model = bi_lstm_model_with_all(embedding_matrix, num_words)
            elif var.USE_SUBJECTS:
                print("  Using Subject Metadata")
                model = bi_lstm_model_with_subject(embedding_matrix, num_words)
            elif var.USE_PARTY:
                print("  Using Party Metadata")
                model = bi_lstm_model_with_party(embedding_matrix, num_words)
            elif var.USE_CREDIT:
                print("  Using Credit Metadata")
                model = bi_lstm_model_with_credit(embedding_matrix, num_words)
            else:
                model = bi_lstm_model(embedding_matrix, num_words)
        elif var.MODEL_TYPE == "BI_LSTM_CNN":
            if var.USE_SUBJECTS and var.USE_PARTY and var.USE_CREDIT:
                print("  Using all subject, party, credit metadata")
                model = bi_lstm_cnn_model_with_all(embedding_matrix, num_words, var.POOLING)
            elif var.USE_SUBJECTS:
                print("  Using Subject Metadata")
                model = bi_lstm_cnn_model_with_subject(embedding_matrix, num_words, var.POOLING)
            elif var.USE_PARTY:
                print("  Using Party Metadata")
                model = bi_lstm_cnn_model_with_party(embedding_matrix, num_words, var.POOLING)
            elif var.USE_CREDIT:
                print("  Using Credit Metadata")
                model = bi_lstm_cnn_model_with_credit(embedding_matrix, num_words, var.POOLING)
            else:
                model = bi_lstm_cnn_model(embedding_matrix, num_words, var.POOLING)
        elif var.MODEL_TYPE == "CNN_BI_LSTM":
            if var.USE_SUBJECTS and var.USE_PARTY and var.USE_CREDIT:
                print("  Using all subject, party, credit metadata")
                model = cnn_bi_lstm_model_with_all(embedding_matrix, num_words)
            elif var.USE_SUBJECTS:
                print("  Using Subject Metadata")
                model = cnn_bi_lstm_model_with_subject(embedding_matrix, num_words)
            elif var.USE_PARTY:
                print("  Using Party Metadata")
                model = cnn_bi_lstm_model_with_party(embedding_matrix, num_words)
            elif var.USE_CREDIT:
                print("  Using Credit Metadata")
                model = cnn_bi_lstm_model_with_credit(embedding_matrix, num_words)
            else:
                model = cnn_bi_lstm_model(embedding_matrix, num_words)
        elif var.MODEL_TYPE == "PARALLEL":
            if var.USE_SUBJECTS and var.USE_PARTY and var.USE_CREDIT:
                print("  Using all subject, party, credit metadata")
                model = parallel_cnn_bi_lstm_model_with_all(embedding_matrix, num_words)
            elif var.USE_SUBJECTS:
                print("  Using Subject Metadata")
                model = parallel_cnn_bi_lstm_model_with_subject(embedding_matrix, num_words)
            elif var.USE_PARTY:
                print("  Using Party Metadata")
                model = parallel_cnn_bi_lstm_model_with_party(embedding_matrix, num_words)
            elif var.USE_CREDIT:
                print("  Using Credit Metadata")
                model = parallel_cnn_bi_lstm_model_with_credit(embedding_matrix, num_words)
            else:
                model = parallel_cnn_bi_lstm_model(embedding_matrix, num_words)
        else:
            raise Exception("Invalid MODEL_TYPE")
        print("--- DONE ---")
       
        print("Training model... ")
        # Save trained model after each epoch

        checkpoint_file = os.path.join(var.FOLDER_NAME, str(i).zfill(2) + var.FILE_NAME)
        checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True)
        callbacks = [checkpoint]
        
        if var.USE_SUBJECTS and var.USE_PARTY and var.USE_CREDIT:
            model.fit([x_train, x_train_subject, x_train_party, x_train_credit], y_train,
                       validation_data=([x_val, x_val_subject, x_val_party, x_val_credit], y_val),
                       epochs=var.NUM_EPOCHS, batch_size=var.BATCH_SIZE, callbacks=callbacks)
        elif var.USE_SUBJECTS:
            model.fit([x_train, x_train_subject], y_train, validation_data=([x_val, x_val_subject], y_val),
                      epochs=var.NUM_EPOCHS, batch_size=var.BATCH_SIZE, callbacks=callbacks)
        elif var.USE_PARTY:
            model.fit([x_train, x_train_party], y_train, validation_data=([x_val, x_val_party], y_val), 
                      epochs=var.NUM_EPOCHS, batch_size=var.BATCH_SIZE, callbacks=callbacks)
        elif var.USE_CREDIT:
            model.fit([x_train, x_train_credit], y_train, validation_data=([x_val, x_val_credit], y_val), 
                      epochs=var.NUM_EPOCHS, batch_size=var.BATCH_SIZE, callbacks=callbacks)
        else:
            model.fit(x_train, y_train, validation_data=(x_val, y_val),
                      epochs=var.NUM_EPOCHS, batch_size=var.BATCH_SIZE, callbacks=callbacks)
        print("--- DONE ---")

        model.summary()
       
        del model
        
        #print("--- DONE ---")


if __name__ == '__main__':
    train_model()

