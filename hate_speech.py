import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Define device (check for GPU availability)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("Using CPU for training. Training might be slower.")

# Define IndicBERT model and tokenizer
model_name = 'ai4bharat/indic-bert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Assuming binary sentiment classification

# Load the dataset (replace with your actual file path)
df = pd.read_csv("Copy of assamese_dataset_1 - Sheet1.csv")

# Tokenize the comments
tokenized_comments = df['Comment'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# Get maximum sequence length
max_len = max(len(seq) for seq in tokenized_comments)

# Pad sequences with zero-padding on the right side
padded_comments = [torch.nn.functional.pad(torch.tensor(seq), (0, max_len - len(seq)), 'constant', 0) for seq in tokenized_comments]

# Convert list of tensors to tensor
padded_comments = torch.stack(padded_comments)

# Prepare labels
labels = torch.tensor(df['Correct Label'].values)

# Prepare input tensors
input_ids = padded_comments
attention_masks = torch.where(input_ids != 0, torch.tensor(1), torch.tensor(0))

# Split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks = \
    train_test_split(input_ids, labels, attention_masks, random_state=42, test_size=0.1)

# Create DataLoader for training and validation sets
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=8, shuffle=False)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # Adjust learning rate as needed

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        batch = tuple(t.to(device) for t in batch)  # Move data to device
        input_ids, attention_mask, labels = batch

        model.zero_grad()
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)

    # Validation
    model.eval()
    val_accuracy = 0
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch

            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = torch.nn.functional.cross_entropy(logits, labels)
            val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            val_accuracy += torch.sum(preds == labels).item()

    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_accuracy = val_accuracy / len(val_data)

    print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}, Val Accuracy: {avg_val_accuracy}")

# Save the trained model and tokenizer
output_dir = "path/to/save/model"  # Replace with your desired output directory
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved successfully in: {output_dir}")