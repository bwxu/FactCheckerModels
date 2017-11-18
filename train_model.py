from __future__ import print_function
import os
import numpy as np
from shutil import copyfile

from gensim.models.keyedvectors import KeyedVectors
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from parse_data import get_glove_vectors, get_labels_sentences_subjects, get_mapping, get_one_hot_vectors
from cnn_models import cnn_model, cnn_model_with_subject
import var

def train_model():
    copyfile('var.py', var.FOLDER_NAME)

    print("Reading word vectors... ")
    embeddings = None
    if var.USE_WORD2VEC:
        embeddings = KeyedVectors.load_word2vec_format(var.WORD2VEC_BIN_PATH, binary=True)
    else:
        embeddings = get_glove_vectors(var.GLOVE_VECTOR_PATH)
    print("--- DONE ---")

    print("Getting input data... ")
    train_labels, train_sentences, train_subjects = get_labels_sentences_subjects(var.TRAINING_DATA_PATH)
    val_labels, val_sentences, val_subjects = get_labels_sentences_subjects(var.VALIDATION_DATA_PATH)
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

    # Convert labels to categorical variables
    y_train = to_categorical(np.asarray([var.LABEL_MAPPING[label] for label in train_labels]))
    y_val = to_categorical(np.asarray([var.LABEL_MAPPING[label] for label in val_labels]))

    # Populate SUBJECT_MAPPING with training data
    var.SUBJECT_MAPPING = get_mapping(train_subjects)
    print(var.SUBJECT_MAPPING)

    # Create one hot vectors for the subject
    x_train_subject = np.asarray(get_one_hot_vectors(train_subjects, var.NUM_SUBJECTS, var.SUBJECT_MAPPING))
    x_val_subject = np.asarray(get_one_hot_vectors(val_subjects, var.NUM_SUBJECTS, var.SUBJECT_MAPPING))

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
        if var.USE_SUBJECTS:
            print("  Using Subject Metadata")
            model = cnn_model_with_subject(embedding_matrix, num_words)
        else:
            model = cnn_model(embedding_matrix, num_words)
        print("--- DONE ---")
       
        print("Training model... ")
        # Save trained model after each epoch

        checkpoint_file = os.path.join(var.FOLDER_NAME, str(i).zfill(2) + var.FILE_NAME)
        checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True)
        callbacks = [checkpoint]
        
        if var.USE_SUBJECTS:
            model.fit([x_train, x_train_subject], y_train, validation_data=([x_val, x_val_subject], y_val), epochs=var.NUM_EPOCHS, batch_size=var.BATCH_SIZE, callbacks=callbacks)
        else:
            model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=var.NUM_EPOCHS, batch_size=var.BATCH_SIZE, callbacks=callbacks)
        print("--- DONE ---")

        #print("Testing trained model... ")
        # Run trained model on test_sequences
        #test_labels, test_sentences, test_subjects = get_labels_sentences_subjects(var.TEST_DATA_PATH)
        #test_sequences = tokenizer.texts_to_sequences(test_sentences)
        #x_test = pad_sequences(test_sequences, maxlen=var.MAX_SEQUENCE_LENGTH)
        #y_test = to_categorical(np.asarray([var.LABEL_MAPPING[label] for label in test_labels]))
        #x_test_subject = np.asarray(get_one_hot_vectors(test_subjects, var.NUM_SUBJECTS, var.SUBJECT_MAPPING))
        
        
        #if var.USE_SUBJECTS:
        #    score = model.evaluate([x_test, x_test_subject], y_test, batch_size=var.BATCH_SIZE)
        #else:
        #    score = model.evaluate(x_test, y_test, batch_size=var.BATCH_SIZE)
        #print()
        #print("test loss = %0.4f, test acc = %0.4f" % (score[0], score[1]))
        
        del model
        
        #print("--- DONE ---")


if __name__ == '__main__':
    train_model()

