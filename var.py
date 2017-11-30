# Location of data files
USE_WORD2VEC = False
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
MAX_SEQUENCE_LENGTH = 67
EMBEDDING_DIM = 300

# Parameters for model construction
TRAIN_EMBEDDINGS = True
FILTER_SIZE_LIST = [2, 3, 4]
NUM_FILTERS = [128, 128, 128]
DROPOUT_PROB = 0.2
# "MAX" or "AVG" or "MAXOUT"
POOLING = "MAXOUT"

# Training parameters
NUM_EPOCHS = 10
BATCH_SIZE = 64

# Parameters for saving the trained model
FOLDER_NAME = "models/10iter/maxout_credit"
# FILE_NAME = "new-epoch-{epoch:02d}-val_loss-{val_loss:.4f}.hdf5"
FILE_NAME = '_lowest_val_loss.hdf5'

USE_SUBJECTS = False
NUM_SUBJECTS = 30
SUBJECT_MAPPING = {}

USE_PARTY = False
NUM_PARTIES = 10
PARTY_MAPPING = {}

USE_CREDIT = True
NUM_CREDIT_TYPES = 5
CREDIT_MAPPING = {"barely-true": 0, 
                  "false": 1, 
                  "half-true": 2, 
                  "mostly-true": 3, 
                  "pants-fire": 4}

NUM_MODELS = 10
