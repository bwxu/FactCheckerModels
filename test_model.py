from __future__ import print_function
import numpy as np
import os
import sys

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from parse_data import get_labels_sentences_subjects, get_mapping, get_one_hot_vectors 

import var

def test_model():
    # Load the trained model to test. Model path should be the first argument
    paths = None
    model_path = sys.argv[1]

    if os.path.isdir(model_path):
        paths = [os.path.join(model_path, f) for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
    elif os.path.isfile(model_path):
        paths = [model_path]
    
    for path in paths:
        model = load_model(path)

        # Recreate the input tokenizer
        train_labels, train_sentences, train_subjects = get_labels_sentences_subjects(var.TRAINING_DATA_PATH)
        tokenizer = Tokenizer(num_words=var.MAX_NUM_WORDS)
        tokenizer.fit_on_texts(train_sentences)

        # Get the val input via tokenizer and val labels
        val_labels, val_sentences, val_subjects = get_labels_sentences_subjects(var.VALIDATION_DATA_PATH)
        val_sequences = tokenizer.texts_to_sequences(val_sentences)
        x_val = pad_sequences(val_sequences, maxlen=var.MAX_SEQUENCE_LENGTH)
        y_val = to_categorical(np.asarray([var.LABEL_MAPPING[label] for label in val_labels]))
        
        # Get the test input via tokenizer and test labels
        test_labels, test_sentences, test_subjects = get_labels_sentences_subjects(var.TEST_DATA_PATH)
        test_sequences = tokenizer.texts_to_sequences(test_sentences)
        x_test = pad_sequences(test_sequences, maxlen=var.MAX_SEQUENCE_LENGTH)
        y_test = to_categorical(np.asarray([var.LABEL_MAPPING[label] for label in test_labels]))

        # Get the subject vectors if necessary
        if var.USE_SUBJECTS:
            var.SUBJECT_MAPPING = get_mapping(train_subjects)
            x_test_subjects = np.asarray(get_one_hot_vectors(test_subjects, var.NUM_SUBJECTS, var.SUBJECT_MAPPING))
            x_val_subjects = np.asarray(get_one_hot_vectors(val_subjects, var.NUM_SUBJECTS, var.SUBJECT_MAPPING))

        # Evaluate test accuracy
        if var.USE_SUBJECTS:
            test_score = model.evaluate([x_test, x_test_subjects], y_test, batch_size=var.BATCH_SIZE)
            val_score = model.evaluate([x_val, x_val_subjects], y_val, batch_size=var.BATCH_SIZE)
        else:
            test_score = model.evaluate(x_test, y_test, batch_size=var.BATCH_SIZE)
            val_score = model.evalulate(x_val, y_val, batch_size=var.BATCH_SIZE)

        print()
        print()
        print("model = " + str(path))
        print("val loss = %0.4f, val acc = %0.4f" % (val_score[0], val_score[1]))
        print("test loss = %0.4f, test acc = %0.4f" % (test_score[0], test_score[1]))


if __name__ == '__main__':
    test_model()

