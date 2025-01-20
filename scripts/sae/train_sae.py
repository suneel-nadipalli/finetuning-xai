import sys, torch

from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

import nltk
# nltk.download("punkt")
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


sys.path.append('../')

from utils.config import *
from utils.utils import *

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_lambda):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded

    def sparsity_loss(self, encoded):
        return self.sparsity_lambda * torch.mean(torch.abs(encoded))

def extract_activations(model, dataloader, dataset_name, model_name, layer_idx, ft=True, device=None):

    if device is None:
        device = DEVICE
    
    activations = []
    
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting activations"):
            
            input_ids = batch[0].to(device)
            
            attention_mask = batch[1].to(device)
            
            label = batch[2]

            # Pass data through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states[layer_idx]

            # Aggregate token-level activations (e.g., mean pooling)
            sentence_activations = hidden_states.mean(dim=1)
            
            activations.append(sentence_activations.cpu())
            
            labels.append(label)

    # Concatenate all activations and labels
    activations = torch.cat(activations)
    
    labels = torch.cat(labels)

    # model_name = model_name.replace("-", "_")

    if ft:
        torch.save({"activations": activations, "labels": labels}, f"{ACTS_DIR}/{dataset_name}/{dataset_name}_{model_name}_{layer_idx}_act_ft.pt")
    else:
        torch.save({"activations": activations, "labels": labels}, f"{ACTS_DIR}/{dataset_name}/{dataset_name}_{model_name}_{layer_idx}_act_pt.pt")
    
    return activations, labels

def train_sae(activations, dataset_name, model_name, layer_idx, ft=True, device=None):

    if device is None:
        device = DEVICE
    
    autoencoder = SparseAutoencoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, sparsity_lambda=SPARSITY_LAMBDA)
    
    autoencoder.to(device)
    
    # Create a DataLoader
    dataloader = DataLoader(activations, batch_size=BATCH_SIZE, shuffle=True)
    
    # Define optimizer and loss criterion
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            # Move the batch to the device
            batch = batch.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            reconstructed, encoded = autoencoder(batch)
            loss = criterion(reconstructed, batch) + autoencoder.sparsity_loss(encoded)

            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader)}")
    
    # model_name = model_name.replace("-", "_")

    if ft:
        torch.save(autoencoder.state_dict(), f"{SAES_DIR}/{dataset_name}/{dataset_name}_{model_name}_{layer_idx}_sae_ft.pt")
    else:
        torch.save(autoencoder.state_dict(), f"{SAES_DIR}/{dataset_name}/{dataset_name}_{model_name}_{layer_idx}_sae_pt.pt")
    
    return autoencoder

def load_sae(dataset_name, model_name, layer_idx, ft=True, device=None):

    if device is None:
        device = DEVICE
    
    model_name = model_name.replace("-", "_")

    if ft:
        print("Loading fine-tuned autoencoder...")
        model_pth = f"{SAES_DIR}/{dataset_name}/{dataset_name}_{model_name}_{layer_idx}_sae_ft.pt"
    
    else:
        print("Loading pre-trained autoencoder...")
        model_pth = f"{SAES_DIR}/{dataset_name}/{dataset_name}_{model_name}_{layer_idx}_sae_pt.pt"

    autoencoder = SparseAutoencoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, sparsity_lambda=SPARSITY_LAMBDA)

    autoencoder.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))

    autoencoder.to(device)

    if ft:
        print("Loading fine-tuned activations...")
        activation_pth = f"{ACTS_DIR}/{dataset_name}/{dataset_name}_{model_name}_{layer_idx}_act_ft.pt"
    
    else:
        print("Loading pre-trained activations...")
        activation_pth = f"{ACTS_DIR}/{dataset_name}/{dataset_name}_{model_name}_{layer_idx}_act_pt.pt"


    activations = torch.load(activation_pth, map_location=torch.device('cpu'))["activations"]

    activations.to(device)

    labels.to(device)

    return autoencoder, activations, labels
