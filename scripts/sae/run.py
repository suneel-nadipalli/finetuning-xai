import sys, torch, gc

sys.path.append("../../")

from scripts.sae.train_sae import *
from scripts.sae.infer_sae import *

from utils.utils import *

datasets = ["imdb", "spotify", "news"]

labels = [2, 5, 5]

sel_idx = 0

layers = [3, 6, 12]

layer_idx = layers[0]

dataset_name, num_labels = datasets[sel_idx], labels[sel_idx]

model_name = "bert-base-uncased"

model, tokenizer = load_model_tok(dataset_name=dataset_name, model_name=model_name, num_labels=num_labels, ft=True,)

autoencoder, activations, labels = load_sae(dataset_name=dataset_name, model_name=model_name, layer_idx=layer_idx,ft=True,)

features = extract_features(autoencoder=autoencoder, activations=activations,)

top_features_indices, top_features_variances = get_top_features(features, 3)

print(top_features_indices)

feature_idx = 1015

top_examples = get_top_examples(
    feature_idx=feature_idx,
    features=features,
    dataset_name=dataset_name,
    layer_idx=layer_idx,
    top_k=-1,
)

sentence = """
Today was a good day.
"""

token_level_activations(sentence=sentence, autoencoder=autoencoder, model=model, feature_idx=1015, layer_idx=layer_idx, tokenizer=tokenizer)

# for ds_idx, dataset in enumerate(datasets):
#     for layer_idx, layer in enumerate(layers):
        
#         dataset_name = dataset

#         model, tokenizer = load_model_tok(
#             dataset_name=dataset_name,
#             model_name=model_name,
#             num_labels=2,
#             ft=True,
#         )

#         train_loader, val_loader = load_and_preprocess_data(dataset_name=dataset_name, tokenizer=tokenizer)

#         activations, labels = extract_activations(model, train_loader, dataset_name, model_name, layer_idx)

#         train_sae(activations=activations, dataset_name=dataset_name, model_name=model_name, layer_idx=layer_idx, ft=True)

#     del autoencoder, activations, labels

#     torch.cuda.empty_cache()

#     gc.collect()