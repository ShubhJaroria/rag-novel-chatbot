import pandas as pd
import re
import nltk
import torch
from bs4 import BeautifulSoup
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import evaluate
from transformers import AdamW
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Ensure you have the necessary NLTK data
nltk.download('punkt')

# Configuration
data_paths = ["sherlock.csv", "zorro.csv","anthropology.csv","England.csv","Peter.csv","Pride.csv","Proposal.csv","Romeo.csv","Salome.csv","Theodore.csv"]  # Add more file names here
text_column_name = "text"
label_column_name = "category"
model_name = "distilbert-base-uncased"
test_size = 0.2
num_labels = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define Cleaner class
class Cleaner():
    def __init__(self):
        pass
    def put_line_breaks(self, text):
        return text.replace('</p>', '</p>\n')
    def remove_html_tags(self, text):
        return BeautifulSoup(text, "lxml").text
    def clean(self, text):
        text = self.put_line_breaks(text)
        text = self.remove_html_tags(text)
        return text

# Process and concatenate all CSVs
'''
all_dfs = []
for data_path in data_paths:
    df = pd.read_csv(data_path)
    cleaner = Cleaner()
    df['text_cleaned'] = df[text_column_name].apply(cleaner.clean)
    le = preprocessing.LabelEncoder()
    le.fit(df[label_column_name].tolist())
    df['label'] = le.transform(df[label_column_name].tolist())
    all_dfs.append(df)
    '''
all_dfs = []
for data_path in data_paths:
    df = pd.read_csv(data_path, nrows=2)  # Read only the first two rows
    cleaner = Cleaner()
    df['text_cleaned'] = df[text_column_name].apply(cleaner.clean)
    le = preprocessing.LabelEncoder()
    le.fit(df[label_column_name].tolist())
    df['label'] = le.transform(df[label_column_name].tolist())
    all_dfs.append(df)
# Combine all data into a single DataFrame
combined_df = pd.concat(all_dfs, ignore_index=True)

# Split the combined data
df_train, df_test = train_test_split(combined_df, test_size=test_size)
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess_function(examples):
    return tokenizer(examples["text_cleaned"], truncation=True)
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.load("accuracy")

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./final_model_output3",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=200,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    gradient_accumulation_steps=16,
    load_best_model_at_end=True,
    save_strategy="epoch",
    greater_is_better=True
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Custom optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Update trainer with optimizer and scheduler
trainer.args.optimizer = optimizer
trainer.args.lr_scheduler = scheduler
trainer.args.max_grad_norm = 1.0

# Train the model
trainer.train()

# Save the model
#trainer.save_model('./final_trained_model')



trainer.save_model('./final_model3')





