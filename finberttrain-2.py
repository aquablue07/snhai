from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Config
MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = "finbert-loan-classifier"
LABELS = ["APPROVE", "REJECT", "FLAG_REVIEW"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

# Load and Prepare Dataset
dataset = load_dataset("json", data_files={
    "train": "train_data.jsonl",
    "validation": "val_data.jsonl"
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_format(examples):
    # Extract the label from the output_text
    labels_str = [text.split(" – ")[0] for text in examples["output_text"]]
    labels_id = [label2id[label] for label in labels_str]
    
    # Tokenize the input text
    tokenized = tokenizer(
        examples["input_text"],
        max_length=256,
        padding="max_length",
        truncation=True
    )
    tokenized["label"] = labels_id
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_format, batched=True)

#  Model initialize
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

# get metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Train
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f" FinBERT training complete. Model saved to '{OUTPUT_DIR}'.")