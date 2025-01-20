import sys, torch
import streamlit as st

sys.path.append('..')

from scripts.sae.infer_sae import *
from scripts.sae.train_sae import *

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEAT_IDX_MAPPINGS = {
    "imdb-3-true": 1015,
    "imdb-3-false": 690,
    "imdb-6-true": 224,
    "imdb-6-false": 75,
    "imdb-12-true": 509,
    "imdb-12-false": 332,
    "spotify-3-true": 169,
    "spotify-3-false": 997,
    "spotify-6-true": 17,
    "spotify-6-false": 442,
    "spotify-12-true": 912,
    "spotify-12-false": 927,
    "news-3-true": 168,
    "news-3-false": 343,
    "news-6-true": 242,
    "news-6-false": 694,
    "news-12-true": 934,
    "news-12-false": 287,
}

def activation_to_color(value):
    """Map activation values to RGBA colors with transparency for better text visibility."""
    red = int(min(max(value, 0), 1) * 255)  # Full red range
    blue = int(255 * (1 - min(max(value, 0), 1)))  # Full blue range
    opacity = 0.5  # Set opacity to 50%
    return f"rgba({red}, 0, {blue}, {opacity})"

def normalize_activations(activations):
    """Normalize activation values to the range [0, 1]."""
    min_val = min(activations)
    max_val = max(activations)
    if max_val == min_val:
        # Avoid division by zero; return uniform values
        return [0.5] * len(activations)
    return [(val - min_val) / (max_val - min_val) for val in activations]

def activation_helper(dataset, layer, ft, sentence):

    dataset_name = dataset.lower()

    model_name = "bert-base-uncased"

    layer_idx = int(layer.split()[-1])

    model, tokenizer = load_model_tok(
        dataset_name=dataset_name,
        model_name=model_name,
        num_labels=2,
        ft=ft,
        device=device,
    )

    autoencoder, activations, labels = load_sae(
        dataset_name=dataset_name,
        model_name=model_name,
        layer_idx=layer_idx,
        ft=ft,
        device=device,
    )

    feature_idx = FEAT_IDX_MAPPINGS[f"{dataset_name}-{layer_idx}-{str(ft).lower()}"]

    act_dict = token_level_activations(sentence=sentence, autoencoder=autoencoder, 
                                           model=model, feature_idx=feature_idx, 
                                           layer_idx=layer_idx, tokenizer=tokenizer, 
                                           device=device, interactive=False)
    
    return act_dict

def plot_token_activations(token_activations):
    """
    Plot token activation values as a bar chart.
    Args:
        token_activations (list): List of dictionaries with 'Token' and 'Activation' keys.
    """
    # Extract tokens and activations
    tokens = [item["Token"] for item in token_activations]
    activations = [item["Activation"] for item in token_activations]

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(tokens, activations, color="#2b6cb0", alpha=0.8)

    # Formatting the chart
    ax.set_xlabel("Tokens", fontsize=12)
    ax.set_ylabel("Activation Values", fontsize=12)
    ax.set_title("Token Activations", fontsize=14)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Display the chart in Streamlit
    st.pyplot(fig)