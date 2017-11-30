from __future__ import print_function

import csv
import io
import math
import numpy as np
import var

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


def get_data(path):
    # Given a data file path, get the labels and sentences as lists
    data = get_input_data(path)
    labels = [datum[1] for datum in data]
    sentences = [datum[2] for datum in data]
    subjects = [datum[3].split(',') for datum in data]
    party = [datum[7] for datum in data]
    history = [[int(datum[i]) for i in range(8, 13)] for datum in data]
    return labels, sentences, subjects, party, history


def get_mapping(list_of_list_of_items):
    # Return a frequency mapping for each item
    mapping = {}
    for list_of_items in list_of_list_of_items:
        for item in list_of_items:
            if item not in mapping:
                mapping[item] = 0
            mapping[item] += 1
    return mapping


def get_one_hot_vectors(list_of_values, length, mapping):
    # takes in list of list of values (one list per data point), length of one hot vector
    # and a frequency mapping
    top_values = sorted(mapping.items(), key=lambda x: x[1], reverse=True)[:length]
    one_hot_values = {top_values[i][0]: i for i in range(len(top_values))}

    vectors = []
    for values in list_of_values:
        vector = [0]*length
        for value in values:
            if value in one_hot_values:
                index = one_hot_values[value]
                vector[index] = 1
        vectors.append(vector)
    return vectors

def normalize_vectors(list_of_values):
    normalized = []
    for values in list_of_values:
        sum = 0.0
        for value in values:
            sum += value
        # zero sum case
        if sum == 0:
            sum = 1
        normalized.append([value/sum for value in values])
    return normalized

def clean_credit(labels, credit):
    # remove from credit vector the current label
    for i in range(len(labels)):
        print(labels[i], credit[i])
        if labels[i] in var.CREDIT_MAPPING:  
            remove_index = var.CREDIT_MAPPING[labels[i]]
            print(remove_index)
            credit[i][remove_index] -= 1
        print(credit[i])
    return credit

