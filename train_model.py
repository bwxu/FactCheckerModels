from __future__ import print_function

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from parse_data import get_glove_vectors, get_input_data

# Location of data files
GLOVE_VECTOR_PATH = "data/glove.6B.300d.txt"
TRAINING_DATA_PATH = "data/train.tsv"
VALIDATION_DATA_PATH = "data/valid.tsv"
TEST_DATA_PATH = "data/test.tsv"

# Argument for preparing setences, labels, and embedding matrix
LABEL_MAPPING = {"pants-fire": 1, 
                 "false": 2,
                 "barely-true": 3,
                 "half-true": 4,
                 "mostly-true": 5,
                 "true": 6}
MAX_NUM_WORDS = 20000
MAX_STATEMENT_LENGTH = 100
EMBEDDING_DIM = 300
ACTUAL_NUM_WORDS = None


def main():
    train_model()

def train_model():
    print("Reading word vectors... ")
    
    embeddings = get_glove_vectors(GLOVE_VECTOR_PATH)
    
    print("-- DONE --")

    print("Getting input data... ")
    
    train_data = get_input_data(TRAINING_DATA_PATH)
    train_labels = [data[1] for data in train_data]
    train_sentences = [data[2] for data in train_data]

    val_data = get_input_data(VALIDATION_DATA_PATH)
    val_labels = [data[1] for data in val_data]
    val_sentences = [data[2] for data in val_data]
    
    print("-- DONE --")

    print("Preparing data for model... ")
    
    # Convert the input sentences into sequences of integers length 100 where
    # each integer maps to a word and the sequence only considers the first 100
    # words in the statement being evaluated

    tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
    tokenizer.fit_on_texts(train_sentences + val_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    word_index = tokenizer.word_index

    x_train = pad_sequences(train_sequences, maxlen=MAX_STATEMENT_LENGTH)
    x_val = pad_sequences(val_sequences, maxlen=MAX_STATEMENT_LENGTH)

    # Convert labels to categorical variables
    
    train_labels = [LABEL_MAPPING[label] for label in train_labels]
    val_labels = [LABEL_MAPPING[label] for label in val_labels]

    y_train = to_categorical(np.asarray(train_labels))
    y_val = to_categorical(np.asarray(val_labels))

    # Create embedding matrix for embedding layer. Matrix will be 
    # (num_words + 1) x EMBEDDING_DIM since word_index starts at 1

    num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
    for word, rank in word_index.items():
        if rank <= MAX_NUM_WORDS:
            embedding = embeddings.get(word, None)
            if embedding is not None:
                embedding_matrix[rank] = embedding
    ACTUAL_NUM_WORDS = num_words
    
    print("-- DONE --")

if __name__ == '__main__':
    main()
