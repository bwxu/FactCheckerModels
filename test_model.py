from __future__ import print_function
import numpy as np
import os

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from parse_data import get_labels_and_sentences

# WARNING: All constants must be the sameas in train_model.py

# Location of data files
MODEL_FOLDER = "models"
MODEL_FILE_NAME = "epoch-19-val_acc-0.2632.hdf5"
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


def test_model():
    # Load the trained model to test
    model = load_model(os.path.join(MODEL_FOLDER, MODEL_FILE_NAME))

    # Recreate the input tokenizer
    train_labels, train_sentences = get_labels_and_sentences(TRAINING_DATA_PATH)
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(train_sentences)

    # Get the test input via tokenizer and test labels
    test_labels, test_sentences = get_labels_and_sentences(TEST_DATA_PATH)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y_test = to_categorical(np.asarray([LABEL_MAPPING[label] for label in test_labels]))

    # Evaluate test accuracy
    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print()
    print("test loss = %0.4f, test acc = %0.4f" % (score[0], score[1]))


if __name__ == '__main__':
    test_model()

