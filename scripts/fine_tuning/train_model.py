import sys, torch

sys.path.append("../..")

from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertTokenizer

from utils.config import *

# Initialize model
def initialize_model(num_labels, device, learning_rate=2e-5, model_name="bert-base-uncased"):
    
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    ).to(device)

    tokenizer = BertTokenizer.from_pretrained(model_name)

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    model_save_name = model_name.split("/")[-1].replace("-", "_")
    
    return model, tokenizer, optimizer, model_save_name

# Train model
def train_model(model, train_loader, val_loader, dataset_name, model_name):
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

    model_name = model_name.replace("-", "_")

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS)

    save_path = f"{MODELS_DIR}/{dataset_name}/{dataset_name}_{model_name}_model.pt"
    
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch in train_loader:
            b_input_ids = batch[0].to(DEVICE)
            b_attention_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == b_labels).sum().item()
            total += b_labels.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        val_loss, val_accuracy = validate_model(model, val_loader, DEVICE)

        # Save model if accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with accuracy: {val_accuracy:.4f}")

# Validate model
def validate_model(model, val_loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            b_input_ids = batch[0].to(DEVICE)
            b_attention_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)

            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == b_labels).sum().item()
            total += b_labels.size(0)

    avg_loss = total_loss / len(val_loader)
    
    accuracy = correct / total
    
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy