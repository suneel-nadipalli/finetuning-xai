import sys, torch

from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import display

import nltk
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

sys.path.append('../')

from utils.config import *
from utils.utils import *

def extract_features(autoencoder, activations):

    autoencoder.to(DEVICE).eval()
    
    with torch.no_grad():
        _, features = autoencoder(activations.to(DEVICE))

    return features


def get_top_features(features, top_n=10):
    """
    Extract the top n features with the highest variance across activations.

    Args:
        features (torch.Tensor): The feature activations. Shape: [num_samples, num_features].
        top_n (int): The number of features to extract with the highest variance.

    Returns:
        top_features_indices (torch.Tensor): Indices of the top n features.
        top_features_variances (torch.Tensor): Variance values for the top n features.
    """
    # Step 1: Compute variance for each feature across samples
    feature_variances = features.var(dim=0)  # Variance along the sample dimension

    # Step 2: Get indices of the top n features with the highest variance
    top_features = torch.topk(feature_variances, k=top_n)
    top_features_indices = top_features.indices
    top_features_variances = top_features.values

    # Step 3: Return results
    return top_features_indices, top_features_variances

def get_top_examples(feature_idx, features, dataset_name, layer_idx, top_k=10):

    data_pth = f"{DATA_DIR}/{dataset_name}.csv"

    mapping = MAPPINGS[dataset_name]

    df = pd.read_csv(data_pth)

    texts = df['text'].tolist()

    labels = df['label'].tolist()

    if top_k == -1:
        top_k = len(texts)

    activations = features[:, feature_idx]
    top_indices = activations.argsort(descending=True)[:top_k]

    top_act = {
        "text": [],
        "label": [],
        "activations": []
    }

    for i in top_indices:
        top_act["text"].append(texts[i])
        top_act["label"].append(mapping[labels[i]])
        top_act["activations"].append(activations[i].item())
    
    top_act_df = pd.DataFrame(top_act)
    
    return top_act_df

def plot_activation(features, feature_idx):
    activations = features[:, feature_idx].cpu().numpy()
    plt.hist(activations, bins=50, log=True)
    plt.title(f"Feature {feature_idx} Activation Distribution")
    plt.xlabel("Activation")
    plt.ylabel("Frequency (Log Scale)")
    plt.show()

def token_level_activations(sentence, autoencoder, model, feature_idx, layer_idx, tokenizer, max_tokens=30, interactive=True):
    """
    Plots token-level activations for a given feature.

    Args:
        sentence (str): Input sentence to analyze.
        autoencoder (torch.nn.Module): Trained sparse autoencoder.
        model (torch.nn.Module): Pre-trained or fine-tuned transformer model.
        feature_idx (int): Index of the feature to analyze.
        tokenizer: Tokenizer corresponding to the transformer model.
        device: PyTorch device (e.g., "cuda" or "cpu").
        max_tokens (int): Maximum number of tokens to display in the plot. Defaults to 30.
        interactive (bool): If True, uses Plotly for interactive plotting. Defaults to False.
    """
    
    if len(sentence.split()) > max_tokens:
        print(f"Sentence contains more than {max_tokens} tokens. Truncating for visualization.")
        sentence = " ".join(sentence.split()[:max_tokens])
    
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}  # Move inputs to the specified device

    # Extract token-level activations from the model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        token_activations = outputs.hidden_states[layer_idx].to(DEVICE)  # Layer 8 activations
        _, token_features = autoencoder(token_activations)  # Sparse token-level features

    # Extract activations for the specific feature
    feature_activations = token_features[0, :, feature_idx].cpu().numpy()  # Move to CPU for plotting

    # Convert token IDs to text
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu())

    # Handle long sentences by truncating tokens

    if interactive:
        # Use Plotly for an interactive plot
        import plotly.express as px
        import pandas as pd
        df = pd.DataFrame({"Token": tokens, "Activation": feature_activations})
        fig = px.bar(df, x="Token", y="Activation", title=f"Token-Level Activations for Feature {feature_idx}", width=1200)
        fig.update_layout(xaxis_tickangle=-45)
        fig.show()
    else:
        # Static plot using Matplotlib
        plt.figure(figsize=(min(20, len(tokens)), 5))  # Adjust figure size based on token count
        plt.bar(range(len(tokens)), feature_activations, color='skyblue')
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
        plt.title(f"Token-Level Activations for Feature {feature_idx}")
        plt.xlabel("Tokens")
        plt.ylabel("Activation Value")
        plt.tight_layout()  # Ensure everything fits
        plt.show()


pd.set_option('display.max_colwidth', None)  # Prevent truncation of long text in Jupyter

def analyze_activations(data, top_n=5):
    """
    Analyzes activation patterns in a dataset and provides insights on distributions, labels, and top examples.
    
    Args:
        data (pd.DataFrame): DataFrame containing columns `text`, `label`, and `activations`.
        top_n (int): Number of top-activating sentences to analyze within each label.
    
    Returns:
        dict: Summary of findings including activation stats and top examples by label.
    """
    findings = {}

    # ---- Compare Positive vs. Negative Activations ----
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x="label", y="activations", palette="Set2")
    plt.title("Activation Values by Label")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Activation Value")
    plt.show()

    # Mean and standard deviation by label
    label_stats = data.groupby('label')['activations'].agg(['mean', 'std']).reset_index()
    findings['label_stats'] = label_stats
    print("Activation Statistics by Label:")
    print(label_stats)

    # ---- Top Activating Sentences by Label ----
    top_sentences_by_label = {}
    for label in data['label'].unique():
        label_data = data[data['label'] == label]
        top_examples = label_data.sort_values(by="activations", ascending=False).head(top_n)
        top_sentences_by_label[label] = top_examples

        print(f"\nTop {top_n} Activating Sentences for Label: {label}")
        display(top_examples[['activations', 'text']])  # Display full text in Jupyter Notebook

    print(f"Max activation value: {data['activations'].max()}")

    print(f"Min activation value: {data['activations'].min()}")