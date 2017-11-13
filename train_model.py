from __future__ import print_function
import os
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from parse_data import get_glove_vectors, get_labels_sentences_subjects, get_one_hot_vectors
from cnn_models import cnn_model, cnn_model_with_subject

# Location of data files
USE_WORD2VEC = False
WORD2VEC_BIN_PATH = "data/GoogleNews-vectors-negative300.bin"
GLOVE_VECTOR_PATH = "data/glove.6B.300d.txt"
TRAINING_DATA_PATH = "data/train.tsv"
VALIDATION_DATA_PATH = "data/valid.tsv"
TEST_DATA_PATH = "data/test.tsv"

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

# Parameters for saving the trained model
FOLDER_NAME = "models"
FILE_NAME = "subjects-epoch-{epoch:02d}-val_acc-{val_acc:.4f}.hdf5"

USE_SUBJECTS = True
NUM_SUBJECTS = 143
SUBJECT_MAPPING = {}

def train_model():
    print("Reading word vectors... ")
    embeddings = None
    if USE_WORD2VEC:
        embeddings = KeyedVectors.load_word2vec_format(WORD2VEC_BIN_PATH, binary=True)
    else:
        embeddings = get_glove_vectors(GLOVE_VECTOR_PATH)
    print("--- DONE ---")

    print("Getting input data... ")
    train_labels, train_sentences, train_subjects = get_labels_sentences_subjects(TRAINING_DATA_PATH)
    val_labels, val_sentences, val_subjects = get_labels_sentences_subjects(VALIDATION_DATA_PATH)
    print("--- DONE ---")

    print("Preparing data for model... ")
    # Convert the input sentences into sequences of integers length 100 where
    # each integer maps to a word and the sequence only considers the first 100
    # words in the statement being evaluated
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(train_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    word_index = tokenizer.word_index

    x_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x_val = pad_sequences(val_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Convert labels to categorical variables
    y_train = to_categorical(np.asarray([LABEL_MAPPING[label] for label in train_labels]))
    y_val = to_categorical(np.asarray([LABEL_MAPPING[label] for label in val_labels]))

    # Populate SUBJECT_MAPPING with training data
    subject_num = 0
    for subjects in train_subjects:
        for subject in subjects:
            if subject not in SUBJECT_MAPPING:
                SUBJECT_MAPPING[subject] = subject_num
                subject_num += 1

    # Create one hot vectors for the subject
    x_train_subject = np.asarray(get_one_hot_vectors(train_subjects, NUM_SUBJECTS, SUBJECT_MAPPING))
    x_val_subject = np.asarray(get_one_hot_vectors(val_subjects, NUM_SUBJECTS, SUBJECT_MAPPING))

    # Create embedding matrix for embedding layer. Matrix will be 
    # (num_words + 1) x EMBEDDING_DIM since word_index starts at 1
    num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
    for word, rank in word_index.items():
        if rank <= MAX_NUM_WORDS:
            embedding = None
            if USE_WORD2VEC:
                if word in embeddings.vocab:
                    embedding = embeddings[word]
            else:
                embedding = embeddings.get(word, None)
            if embedding is not None:
                embedding_matrix[rank] = embedding
    print("--- DONE ---")
    
    print("Creating model... ")
    if USE_SUBJECTS:
        print("  Using Subject Metadata")
        model = cnn_model_with_subject(embedding_matrix, num_words)
    else:
        model = cnn_model(embedding_matrix, num_words)
    print("--- DONE ---")
   
    print("Training model... ")
    # Save trained model after each epoch
    checkpoint_file = os.path.join(FOLDER_NAME, FILE_NAME)
    checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks = [checkpoint]
    
    if USE_SUBJECTS:
        model.fit([x_train, x_train_subject], y_train, validation_data=([x_val, x_val_subject], y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
    else:
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
    print("--- DONE ---")

    print("Testing trained model... ")
    # Run trained model on test_sequences
    test_labels, test_sentences, test_subjects = get_labels_sentences_subjects(TEST_DATA_PATH)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y_test = to_categorical(np.asarray([LABEL_MAPPING[label] for label in test_labels]))
    x_test_subject = np.asarray(get_one_hot_vectors(test_subjects, NUM_SUBJECTS, SUBJECT_MAPPING))

    if USE_SUBJECTS:
        score = model.evaluate([x_test, x_test_subject], y_test, batch_size=BATCH_SIZE)
    else:
        score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print()
    print("test loss = %0.4f, test acc = %0.4f" % (score[0], score[1]))
    print("--- DONE ---")


if __name__ == '__main__':
    train_model()

