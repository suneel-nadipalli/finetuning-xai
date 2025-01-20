import torch

IF_NB = False

# path_append = "../" if IF_NB else "../../"

path_append = ""

DATA_DIR = f"{path_append}data/fine_tuning"

EMBED_DIR = f"{path_append}data/processed/embeddings"

ACTS_DIR = f"{path_append}data/processed/activations"

MODELS_DIR = f"{path_append}models/fine_tuning"

SAES_DIR = f"{path_append}models/saes"

LOGS_DIR = f"{path_append}results/logs"

PLOTS_DIR = f"{path_append}results/plots"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEVICE = torch.device("cpu")

DATA_FRAC = 0.01  # Percentage of dataset to use

TRAIN_FRAC = 0.8  # Percentage of data to use for training

EPOCHS = 2

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

MAX_LENGTH = 512

INPUT_DIM = 768  # Hidden size of BERT

HIDDEN_DIM = 1024  # Hidden size for autoencoder

SPARSITY_LAMBDA = 1e-3

MAPPINGS = {
    
    "imdb": { 0: "positive", 1: "negative" },
    
    "spotify": { 0: "1", 1: "2", 2: "3", 3: "4", 4: "5" },
    
    "news": { 0: "wellness", 1: "sports", 2: "entertainment", 3: "politics", 4: "food" }
}