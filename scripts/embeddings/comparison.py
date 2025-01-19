import sys

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('../')

from utils.config import *

def compute_layer_similarity(pretrained_embeddings, finetuned_embeddings):
    """
    Computes the average cosine similarity between embeddings of two models for a single layer.

    Args:
    - pretrained_embeddings (torch.Tensor): Embeddings from the pre-trained model (shape: [num_samples, hidden_size]).
    - finetuned_embeddings (torch.Tensor): Embeddings from the fine-tuned model (shape: [num_samples, hidden_size]).

    Returns:
    - float: Average cosine similarity for the layer.
    """
    # Convert to numpy for pairwise operations
    pretrained = pretrained_embeddings.cpu().numpy()
    finetuned = finetuned_embeddings.cpu().numpy()

    # Compute cosine similarity for all samples
    similarity_matrix = cosine_similarity(pretrained, finetuned)

    # Take the diagonal (self-to-self similarity) and compute the average
    average_similarity = np.mean(np.diag(similarity_matrix))
    return average_similarity

def compare_embeddings(pretrained_embeddings, finetuned_embeddings):
    """
    Compares embeddings layer by layer using cosine similarity.

    Args:
    - pretrained_embeddings (dict): Layer-wise embeddings from the pre-trained model.
    - finetuned_embeddings (dict): Layer-wise embeddings from the fine-tuned model.

    Returns:
    - dict: Layer-wise cosine similarity scores.
    """
    layer_similarity = {}

    for layer_idx in pretrained_embeddings.keys():
        similarity = compute_layer_similarity(
            pretrained_embeddings[layer_idx],
            finetuned_embeddings[layer_idx]
        )
        layer_similarity[layer_idx] = similarity
        print(f"Layer {layer_idx}: Cosine Similarity = {similarity:.4f}")
    
    return layer_similarity


def visualize_similarity(layer_similarity, dataset_name, model_name, title="Layer-Wise Cosine Similarity"):
    """
    Visualizes layer-wise cosine similarity as a line plot.

    Args:
    - layer_similarity (dict): Layer-wise cosine similarity scores.
    - title (str): Title for the plot.

    Returns:
    - None
    """
    layers = list(layer_similarity.keys())
    
    similarities = list(layer_similarity.values())

    model_name = model_name.replace("-", "_")

    save_path = f"{PLOTS_DIR}/{dataset_name}/{dataset_name}_{model_name}_similarity.png"

    plt.figure(figsize=(10, 6))
    
    plt.plot(layers, similarities, marker='o', label='Cosine Similarity')
    
    plt.title(title)
    
    plt.xlabel("Layer")
    
    plt.ylabel("Cosine Similarity")
    
    plt.xticks(layers)
    
    plt.grid(True)
    
    plt.legend()
    
    # plt.savefig(save_path)
    
    plt.show()

def plot_summary():
    
    data = pd.read_csv(f"{LOGS_DIR}/similarity_summary.csv")

    sns.set(style="whitegrid")

    # Plot 1: One plot for each dataset with all model sizes
    datasets = data["dataset"].unique()
    
    for dataset in datasets:
        
        plt.figure(figsize=(10, 6))
        
        subset = data[data["dataset"] == dataset]
        
        sns.lineplot(data=subset, x="layer", y="similarity", hue="model_size", marker="o")
        
        plt.title(f"Similarity Across Layers for {dataset.capitalize()} Dataset")
        
        plt.xlabel("Layer")
        
        plt.ylabel("Cosine Similarity")
        
        plt.legend(title="Model Size")
        
        plt.grid(True)
        
        plt.tight_layout()
        
        # plt.savefig(f"{PLOTS_DIR}/{dataset}_similarity_summary.png")
        
        plt.show()

    # Plot 2: One plot for each model size with all datasets
    model_sizes = data["model_size"].unique()
    
    for model_size in model_sizes:
        
        plt.figure(figsize=(10, 6))
        
        subset = data[data["model_size"] == model_size]
        
        sns.lineplot(data=subset, x="layer", y="similarity", hue="dataset", marker="o")
        
        plt.title(f"Similarity Across Layers for {model_size.capitalize()} Model")
        
        plt.xlabel("Layer")
        
        plt.ylabel("Cosine Similarity")
        
        plt.legend(title="Dataset")
        
        plt.grid(True)
        
        plt.tight_layout()
        
        # plt.savefig(f"{PLOTS_DIR}/{model_size}_similarity_summary.png")
        
        plt.show()