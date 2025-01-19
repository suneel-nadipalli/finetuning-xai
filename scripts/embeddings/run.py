import sys, torch, gc

sys.path.append("../../")

from scripts.embeddings.comparison import *
from scripts.embeddings.extraction import *

from utils.utils import *

datasets = ["imdb", "spotify", "news"]

labels = [2, 5, 5]

sel_idx = 0

dataset_name, num_labels = datasets[sel_idx], labels[sel_idx]

model_name = "bert-base-uncased"

model_pt, tokenizer_pt = load_model_tok(dataset_name=dataset_name, model_name=model_name, ft=False, num_labels=num_labels)

train_loader, val_loader = load_and_preprocess_data(dataset_name=dataset_name, tokenizer=tokenizer_pt)

# pt_embeddings = extract_embeddings(model_pt, val_loader)

# save_embeddings(pt_embeddings, dataset_name=dataset_name, model_name=model_name, ft=False)

pt_embeddings = load_embeddings(dataset_name=dataset_name, model_name=model_name, ft=False)

model_ft, tokenizer_ft = load_model_tok(dataset_name=dataset_name, model_name=model_name, ft=True, num_labels=num_labels)

train_loader, val_loader = load_and_preprocess_data(dataset_name=dataset_name, tokenizer=tokenizer_ft)

# ft_embeddings = extract_embeddings(model_ft, val_loader)

# save_embeddings(ft_embeddings, dataset_name=dataset_name, model_name=model_name, ft=True)

ft_embeddings = load_embeddings(dataset_name=dataset_name, model_name=model_name, ft=True)

layer_similarity = compare_embeddings(pt_embeddings, ft_embeddings)

visualize_similarity(layer_similarity, dataset_name=dataset_name, model_name=model_name)

plot_summary()