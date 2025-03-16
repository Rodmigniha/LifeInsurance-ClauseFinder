from transformers import FlaubertForSequenceClassification, FlaubertTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_excel("data/contrats_annotes.xlsx")

print("Distribution des labels :")
print(df["label"].value_counts())

dataset = Dataset.from_pandas(df)

model = FlaubertForSequenceClassification.from_pretrained("flaubert/flaubert_base_cased", num_labels=2)
tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

encoded_dataset = dataset.map(preprocess_function, batched=True)

split_dataset = encoded_dataset.train_test_split(test_size=0.2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  
    learning_rate=1e-5,  
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,
    num_train_epochs=10,  
    weight_decay=0.01,
    save_strategy="epoch",  
    logging_dir="./logs",  
    logging_steps=10,  
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("fine-tuned-flaubert")
tokenizer.save_pretrained("fine-tuned-flaubert")