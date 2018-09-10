# File containing all of the parameters to be used for train_model or test_model

# Metadata Selection
# NOTE: When testing, the parameters in this section must be the same as training
USE_SUBJECTS = True
USE_PARTY = True
USE_CREDIT = True
USE_POS = False

NO_STOP_WORDS = False

# Model Selection
# NOTE: When testing, the parameters in this section must be the same as training
# CNN, BI_LSTM, BI_LSTM_CNN, CNN_BI_LSTM, PARALLEL
MODEL_TYPE = "PARALLEL"

NUM_MODELS = 10

HIDDEN_LAYER_SIZE = 100

LSTM_OUT_DIM = 64
LSTM_DROPOUT = 0.4

# Location of data files
USE_WORD2VEC = True
WORD2VEC_BIN_PATH = "data/GoogleNews-vectors-negative300.bin"
GLOVE_VECTOR_PATH = "data/glove.840B.300d.txt"
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
MAX_SEQUENCE_LENGTH = 60
EMBEDDING_DIM = 300

# Parameters for model construction
TRAIN_EMBEDDINGS = False
FILTER_SIZE_LIST = [2, 3, 4]
NUM_FILTERS = [128, 128, 128]
SINGLE_FILTER_SIZE = 4
SINGLE_NUM_FILTERS = 128
DROPOUT_PROB = 0.2
# "MAX" or "AVG" or "MAXOUT"
POOLING = "AVG"

# Training parameters
NUM_EPOCHS = 10
BATCH_SIZE = 64

# Parameters for saving the trained model
FOLDER_NAME = "models/test"
FILE_NAME = 'lowest_val_loss.hdf5'

NUM_SUBJECTS = 30
SUBJECT_MAPPING = {}

NUM_PARTIES = 10
PARTY_MAPPING = {}

CREDIT_MAPPING = {"barely-true": 0, 
                  "false": 1, 
                  "half-true": 2, 
                  "mostly-true": 3,
                  "pants-fire": 4}
NUM_CREDIT_TYPES = len(CREDIT_MAPPING)

POS_TAG_SET = {
    'PRP$': 0, 'VBG': 1, 'VBD': 2, '``': 3, 'VBN': 4,
    ',': 5, "''": 6, 'VBP': 7, 'WDT': 8, 'JJ': 9,
    'WP': 10, 'VBZ': 11, 'DT': 12, 'RP':13, '$': 14,
    'NN': 15, ')': 16, '(': 17, 'FW': 18, 'POS': 19,
    '.': 20, 'TO': 21, 'LS': 22, 'RB': 23, ':': 24,
    'NNS': 25, 'NNP': 26, 'VB': 27, 'WRB': 28, 'CC': 29, 
    'PDT': 30, 'RBS': 31, 'RBR': 32, 'CD': 33, 'PRP': 34,
    'EX': 35, 'IN': 36, 'WP$': 37, 'MD': 38, 'NNPS': 39,
    '--': 40, 'JJS': 41, 'JJR': 42, 'SYM': 43, 'UH': 44,
    '#': 45
    }
POS_TAG_SET_LENGTH = len(POS_TAG_SET)

