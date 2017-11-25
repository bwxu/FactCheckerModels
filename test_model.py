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
    folder = ''

    if os.path.isdir(model_path):
        paths = [os.path.join(model_path, f) for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f)) and f.endswith(".hdf5")]
        folder = model_path
    elif os.path.isfile(model_path):
        paths = [model_path]

    print()
    print("List of files to test")
    print(paths)
    print()
    model_results = []

    for i, path in enumerate(paths):
        print()
        print("Testing file " + str(i + 1) + " out of " + str(len(paths)))

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
            val_score = model.evaluate(x_val, y_val, batch_size=var.BATCH_SIZE)

        print()
        print("model = " + str(path))
        print("val loss = %0.4f, val acc = %0.4f" % (val_score[0], val_score[1]))
        print("test loss = %0.4f, test acc = %0.4f" % (test_score[0], test_score[1]))
        model_results.append((os.path.basename(path), round(val_score[0], 4), round(val_score[1], 4), round(test_score[0], 4), round(test_score[1], 4)))

    average_val_loss = round(sum([info[1] for info in model_results])/len(model_results), 4)
    average_val_acc = round(sum([info[2] for info in model_results])/len(model_results), 4)
    average_test_loss = round(sum([info[3] for info in model_results])/len(model_results), 4)
    average_test_acc = round(sum([info[4] for info in model_results])/len(model_results), 4)
    
    best_val = sorted(model_results, key=lambda x: x[2], reverse=True)[:5]
    best_test = sorted(model_results, key=lambda x: x[4], reverse=True)[:5]

    with open(os.path.join(folder, "stats.txt"), "w") as f:
        f.write("Num models = " + str(len(model_results)) + "\n")
        f.write("Average val_loss = " + str(average_val_loss) + "\n")
        f.write("Average val_acc = " + str(average_val_acc) + "\n")
        f.write("Average test_loss = " + str(average_test_loss) + "\n")
        f.write("Average test_acc = " + str(average_test_acc) + "\n")
        f.write("\n")
        f.write("Top 5 validation accuracies:\n")
        for model in best_val:
            f.write("\t" + str(model[2]) + "\t" + str(model[0]) + "\n")
        f.write("\n")
        f.write("Top 5 test accuracies:\n")
        for model in best_test:
            f.write("\t" + str(model[4]) + "\t" + str(model[0]) + "\n")
        f.write("\n")
        f.write("Results for all models:\n")
        for model in sorted(model_results, key=lambda x: x[0]):
            f.write("\t" + str(model) + "\n")

    print()
    print("------------------------- OVERALL STATISTICS --------------------------")
    print()
    with open(os.path.join(folder, "stats.txt"), "r") as f:
        print(f.read())


if __name__ == '__main__':
    test_model()

