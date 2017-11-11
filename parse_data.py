from __future__ import print_function

import csv
import io
import math
import numpy as np


def get_glove_vectors(path):
    # Given a glove embeddings txt file, parse the file into a dictionary where every key 
    # in the dictionary is a word and the value is the word vector associated with the word
    
    embeddings = {}
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split()
            word = data[0]
            vector = np.asarray(data[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def get_input_data(path):
    # Given an input TSV from the liar dataset, return a list of lists where each index 
    # contains a list of all of the information for one data point. The values of the indices
    # of the sublist are consistent with the TSV format as follows.
    #
    # Description of the TSV format:
    #
    # Column 1: the ID of the statement ([ID].json).
    # Column 2: the label.
    # Column 3: the statement.
    # Column 4: the subject(s).
    # Column 5: the speaker.
    # Column 6: the speaker's job title.
    # Column 7: the state info.
    # Column 8: the party affiliation.
    # Column 9-13: the total credit history count, including the current statement.
    # 9: barely true counts.
    # 10: false counts.
    # 11: half true counts.
    # 12: mostly true counts.
    # 13: pants on fire counts.
    # Column 14: the context (venue / location of the speech or statement).
    #
    # Note that we do not provide the full-text verdict report in this current version of the dataset,
    # but you can use the following command to access the full verdict report and links to the source documents:
    # wget http://www.politifact.com//api/v/2/statement/[ID]/?format=json
    
    with io.open(path, 'r', newline='', encoding='utf-8') as f:
        f = (line.encode('utf-8') for line in f)
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        data = [[index for index in row] for row in reader]
    return data


def get_labels_sentences_subjects(path):
    # Given a data file path, get the labels and sentences as lists
    data = get_input_data(path)
    labels = [datum[1] for datum in data]
    sentences = [datum[2] for datum in data]
    subjects = [datum[3].split(',') for datum in data]
    return labels, sentences, subjects

