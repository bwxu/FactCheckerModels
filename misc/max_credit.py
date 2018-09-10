from __future__ import print_function

from parse_data import clean_credit, get_data
import var


def max_credit_model():
    '''
    Computes the accuracy of a "max credit" model where each statement is
    labeled with the most frequent label of the previous statements by the
    speaker.
    '''
    print("NUM CORRECT", "\t", "ACCURACY")

    train_labels, _, _, _, train_credit = get_data(var.TRAINING_DATA_PATH)
    val_labels, _, _, _, val_credit = get_data(var.VALIDATION_DATA_PATH)
    test_labels, _, _, _, test_credit = get_data(var.TEST_DATA_PATH)

    # 9: barely true counts.
    # 10: false counts.
    # 11: half true counts.
    # 12: mostly true counts.
    # 13: pants on fire counts.
    credit_mapping = [
        "barely-true",
        "false",
        "half-true",
        "mostly-true",
        "pants-fire"]

    train_correct = 0.0

    train_credit = clean_credit(train_labels, train_credit)
    for i in range(len(train_labels)):
        max_credit_index = train_credit[i].index(max(train_credit[i]))
        if credit_mapping[max_credit_index] == train_labels[i]:
            train_correct += 1.0

    print(
        train_correct,
        "\t\t",
        train_correct /
        len(train_labels),
        "\tTraining Data")

    val_correct = 0.0

    val_credit = clean_credit(val_labels, val_credit)
    for i in range(len(val_labels)):
        max_credit_index = val_credit[i].index(max(val_credit[i]))
        if credit_mapping[max_credit_index] == val_labels[i]:
            val_correct += 1.0

    print(
        val_correct,
        "\t\t",
        val_correct /
        len(val_labels),
        "\tValidation Data")

    test_correct = 0.0

    test_credit = clean_credit(test_labels, test_credit)
    for i in range(len(test_labels)):
        max_credit_index = test_credit[i].index(max(test_credit[i]))
        if credit_mapping[max_credit_index] == test_labels[i]:
            test_correct += 1.0

    print(test_correct, "\t\t", test_correct / len(test_labels), "\tTest Data")


max_credit_model()
