# config.py
import os

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# File paths - For TSV files
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.tsv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.tsv')
GLOVE_PATH = os.path.join(DATA_DIR, 'glove.6B.100d.txt')  # Download from https://nlp.stanford.edu/data/glove.6B.zip and place 100d in data/

# Model parameters
MAX_FEATURES = 20000  # Increased for better features
MAX_LEN = 100  # Reduced for phrases
EMBEDDING_DIM = 100  # Matches GloVe
BATCH_SIZE = 128  # Increased for faster training
EPOCHS = 20  # Increased for better convergence

# Preprocessing
USE_3_CLASS = True
CSV_SEPARATOR = '\t'

# Training
USE_CLASS_WEIGHTS = True  # Keep as backup to balancing