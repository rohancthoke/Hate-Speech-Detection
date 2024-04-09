import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your CSV data
data = pd.read_csv("Copy of assamese_dataset_1 - Sheet1.csv")

# Extract relevant columns
comments = data["Comment"]
correct_labels = data["Correct Label"]
chatgpt_predictions = data["chatgpt by rohan"]

# Create a confusion matrix
from sklearn.metrics import confusion_matrix
y_true = correct_labels
y_pred = chatgpt_predictions
confusion_matrix = confusion_matrix(y_true, y_pred)

# Print the confusion matrix
print("Confusion Matrix:\n", confusion_matrix)

# Calculate evaluation metrics
accuracy = accuracy_score(correct_labels, chatgpt_predictions)
precision = precision_score(correct_labels, chatgpt_predictions)
recall = recall_score(correct_labels, chatgpt_predictions)
f1 = f1_score(correct_labels, chatgpt_predictions)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
