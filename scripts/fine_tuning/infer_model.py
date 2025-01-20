import sys, torch

sys.path.append("../")

from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from utils.config import *

# Predict single sentence
def predict_single_sentence(model, tokenizer, sentence, device=None):

    if device is None:
        device = DEVICE

    inputs = tokenizer(
        sentence,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
        return torch.argmax(logits, dim=1).item()

# Classification report and heatmap
def gen_clf_report(model, dataloader, dataset_name, model_name, device=None):

    if device is None:
        device = DEVICE
    
    model.eval()
    
    predictions, true_labels = [], []

    class_names = list(MAPPINGS[dataset_name].values())

    model_name = model_name.replace("-", "_")

    save_path = f"{PLOTS_DIR}/{dataset_name}/{dataset_name}_{model_name}_classification_report.png"

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            logits = model(input_ids, attention_mask=attention_mask).logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)

    report_values = np.array([[v['precision'], v['recall'], v['f1-score']] for k, v in report.items() if k in class_names])

    report = classification_report(true_labels, predictions, target_names=class_names)
    
    sns.heatmap(report_values, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Precision', 'Recall', 'F1-Score'], yticklabels=class_names)
    
    plt.title("Classification Report Heatmap")
    
    # plt.savefig(save_path)
    
    plt.show()

    return report