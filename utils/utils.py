import sys, torch, re

from transformers import BertForSequenceClassification, AutoTokenizer

import pandas as pd
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.append("..")

from utils.config import *

def load_model_tok(dataset_name, model_name, ft=True, num_labels=5, device=None):
    
    if device is None:
        device = DEVICE

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_name = model_name.replace("-", "_")

    if ft:
        print("Loading fine-tuned model...")

        model_pth = f"{MODELS_DIR}/{dataset_name}/{dataset_name}_{model_name}_model.pt"

        if device == torch.device("cpu"):
            model.load_state_dict(torch.load(model_pth, map_location=device))
        
        model.load_state_dict(torch.load(model_pth))
    
    else:
        print("Loading pre-trained model...")
    
    model.to(device)

    model.eval()  # Set to evaluation mode

    return model, tokenizer

# Clean text function
def clean_text(text):
    sw = stopwords.words('english')
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')
    text = " ".join([word for word in text.split() if word not in sw])
    return text

# Load and preprocess data
def load_and_preprocess_data(dataset_name, tokenizer, seed=42):
    
    data_path = f"{DATA_DIR}/{dataset_name}.csv"

    df = pd.read_csv(data_path).sample(frac=DATA_FRAC, random_state=seed)
    
    df['text'] = df['text'].apply(clean_text)
    try:
        df['label'] = df['label'].replace({'positive': 0, 'negative': 1})
    except:
        pass

    # Tokenize data
    encodings = tokenizer(
        list(df['text']),
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )

    labels = torch.tensor(df['label'].values)

    # Create dataset
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)

    # Split into train and validation
    train_size = int(TRAIN_FRAC * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = create_dataloader(train_dataset, BATCH_SIZE, shuffle=True)

    val_loader = create_dataloader(val_dataset, BATCH_SIZE)

    return train_loader, val_loader

# Create DataLoader
def create_dataloader(dataset, batch_size, shuffle=False):
    sampler = torch.utils.data.RandomSampler(dataset) if shuffle else torch.utils.data.SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)
