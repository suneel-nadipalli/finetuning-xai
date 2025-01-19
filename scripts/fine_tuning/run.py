import sys, torch, gc

sys.path.append("../../")

from utils.utils import load_and_preprocess_data
from scripts.fine_tuning.train_model import train_model, validate_model
from scripts.fine_tuning.infer_model import gen_clf_report

from utils.utils import *

datasets = ["imdb", "spotify", "news"]

labels = [2, 5, 5]

sel_idx = 0

dataset_name, num_labels = datasets[sel_idx], labels[sel_idx]

model_name = "bert-base-uncased"

for dataset, num_labels in zip(datasets, labels):
    
    print(f"Running evaluation pipeline for {dataset} dataset")

    # Initialize the pre-trained model

    model_pt, tokenizer_pt = load_model_tok(dataset_name=dataset, model_name="bert-base-uncased",
                                            ft=False, num_labels=num_labels)
    
    # Load and preprocess the data

    train_loader, val_loader = load_and_preprocess_data(dataset_name=dataset, tokenizer=tokenizer_pt)

    # Validate the model

    loss, accuracy = validate_model(model_pt, val_loader)

    # Generate classification report

    report = gen_clf_report(model_pt, val_loader, dataset, "bert-base-uncased")

    print(report)

    print(f"Finished evaluation pipeline for {dataset} dataset")

    print(f"\n{'='*50}\n")


for dataset, num_labels in zip(datasets, labels):
    
    print(f"Running evaluation pipeline for {dataset} dataset")

    # Initialize the fine-tuned model

    model_pt, tokenizer_pt = load_model_tok(dataset_name=dataset, model_name="bert-base-uncased",
                                            ft=True, num_labels=num_labels)
    
    # Load and preprocess the data

    train_loader, val_loader = load_and_preprocess_data(dataset_name=dataset, tokenizer=tokenizer_pt)

    # Validate the model

    loss, accuracy = validate_model(model_pt, val_loader)

    print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    # Generate classification report

    report = gen_clf_report(model_pt, val_loader, dataset, "bert-base-uncased")

    print(report)

    print(f"Finished evaluation pipeline for {dataset} dataset")

    print(f"\n{'='*50}\n")


model, tokenizer = load_model_tok(dataset_name=dataset_name, model_name=model_name, ft=False, num_labels=num_labels)

train_loader, val_loader = load_and_preprocess_data(dataset_name=dataset_name, tokenizer=tokenizer)

# train_model(model, train_loader, val_loader, dataset_name, model_name)

torch.cuda.empty_cache()

gc.collect()