import sys, torch

sys.path.append('../')

from utils.config import *

def initialize_embedding_dict(num_layers):
    """
    Initializes an empty dictionary to store embeddings for each layer.

    Args:
    - num_layers (int): Number of layers in the model.

    Returns:
    - dict: A dictionary with keys as layer indices and empty lists as values.
    """
    return {layer: [] for layer in range(num_layers + 1)}  # Include embeddings for the input layer

def filter_padding(embeddings, attention_mask):
    """
    Filters out padding tokens from embeddings using the attention mask.

    Args:
    - embeddings (torch.Tensor): Layer embeddings of shape [batch_size, seq_len, hidden_size].
    - attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len].

    Returns:
    - torch.Tensor: Filtered embeddings of shape [num_valid_tokens, hidden_size].
    """
    valid_tokens = attention_mask.bool()  # Convert attention mask to boolean
    return embeddings[valid_tokens]

def extract_embeddings(model, dataloader, device=None):
    """
    Extracts layer-wise embeddings from the model for the given dataset.

    Args:
    - model (torch.nn.Module): Pre-trained or fine-tuned BERT model.
    - dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    - device (torch.device): Device to run the model on (CPU/GPU).

    Returns:
    - dict: A dictionary of embeddings for each layer.
    """
    if device is None:
        device = DEVICE

    model.eval()  # Set model to evaluation mode
    num_layers = model.config.num_hidden_layers
    all_layer_embeddings = initialize_embedding_dict(num_layers)

    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to the device
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)

            # Forward pass with hidden states output
            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states  # Tuple of tensors, one per layer

            # Collect embeddings for each layer
            for layer_idx, layer_embeddings in enumerate(hidden_states):
                valid_embeddings = filter_padding(layer_embeddings, b_attention_mask)
                all_layer_embeddings[layer_idx].append(valid_embeddings)

    # Concatenate embeddings across all batches
    for layer_idx in all_layer_embeddings:
        all_layer_embeddings[layer_idx] = torch.cat(all_layer_embeddings[layer_idx], dim=0)

    return all_layer_embeddings

def save_embeddings(embeddings, dataset_name, model_name, ft=False):
    """
    Saves embeddings to a file.

    Args:
    - embeddings (dict): Layer-wise embeddings.
    - filename (str): File path to save the embeddings.

    Returns:
    - None
    """
    model_name = model_name.replace("-", "_")

    if ft:
        save_path = f"{EMBED_DIR}/{dataset_name}/{dataset_name}_{model_name}_ft_embeddings.pt"
    else:
        save_path = f"{EMBED_DIR}/{dataset_name}/{dataset_name}_{model_name}_pt_embeddings.pt"

    torch.save(embeddings, save_path)
    
    print(f"Embeddings saved to {save_path}")


def load_embeddings(dataset_name, model_name, ft=False):
    """
    Loads embeddings from a file.

    Args:
    - filename (str): File path to load the embeddings.

    Returns:
    - dict: Layer-wise embeddings.
    """
    model_name = model_name.replace("-", "_")

    if ft:
        load_path = f"{EMBED_DIR}/{dataset_name}/{dataset_name}_{model_name}_ft_embeddings.pt"
    else:
        load_path = f"{EMBED_DIR}/{dataset_name}/{dataset_name}_{model_name}_pt_embeddings.pt"

    return torch.load(load_path)