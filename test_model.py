from __future__ import print_function
import numpy as np
import os
import sys

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from parse_data import get_labels_sentences_subjects, get_one_hot_vectors 

# WARNING: All constants must be the same as in train_model.py

# Location of data files
TRAINING_DATA_PATH = "data/train.tsv"
TEST_DATA_PATH = "data/test.tsv"

# Arguments for preparing labels and sentences
LABEL_MAPPING = {"pants-fire": 0, 
                 "false": 1,
                 "barely-true": 2,
                 "half-true": 3,
                 "mostly-true": 4,
                 "true": 5}
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 67

# Parameters for model evaluation
BATCH_SIZE = 64

NUM_SUBJECTS = 143
USE_SUBJECTS = True
SUBJECT_MAPPING = {}

def test_model():
    # Load the trained model to test. Model path should be the first argument
    model = load_model(sys.argv[1])

    # Recreate the input tokenizer
    train_labels, train_sentences, train_subjects = get_labels_sentences_subjects(TRAINING_DATA_PATH)
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(train_sentences)

    # Get the test input via tokenizer and test labels
    test_labels, test_sentences, test_subjects = get_labels_sentences_subjects(TEST_DATA_PATH)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y_test = to_categorical(np.asarray([LABEL_MAPPING[label] for label in test_labels]))

    if USE_SUBJECTS:
        # Populate SUBJECT_MAPPING
        subject_num = 0
        for subjects in train_subjects:
            for subject in subjects:
                if subject not in SUBJECT_MAPPING:
                    SUBJECT_MAPPING[subject] = subject_num
                    subject_num += 1

    x_test_subjects = np.asarray(get_one_hot_vectors(test_subjects, NUM_SUBJECTS, SUBJECT_MAPPING))

    print(x_test_subjects)
    # Evaluate test accuracy
    if USE_SUBJECTS:
        score = model.evaluate([x_test, x_test_subjects], y_test, batch_size=BATCH_SIZE)
    else:
        score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print()
    print("test loss = %0.4f, test acc = %0.4f" % (score[0], score[1]))


if __name__ == '__main__':
    test_model()

