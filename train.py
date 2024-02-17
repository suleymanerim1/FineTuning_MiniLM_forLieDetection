import datasets
from numba import cuda
import pandas as pd
from transformers import DataCollatorForSeq2Seq
from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer,AutoModel
from transformers import create_optimizer
from datasets import Dataset,DatasetDict
from tqdm import tqdm
from transformers import get_scheduler
from torch.optim import AdamW
import torch.cuda
from typing import Tuple, Dict
import numpy as np
import evaluate
import os


seed = 42
np.random.seed(seed)


def csv_to_df(path: str = None,question_type:int=1, sample: int = 1000) -> pd.DataFrame:
    """

    :param path: "csv file path
    :param question_type: for q1 ->1, for q2->2, for q1+q2->3
    :param sample: 0 is full dataset, anything else is number of rows
    :return: pandas dataframe
    """
    df = pd.read_csv(path, encoding="ISO-8859-1")

    if sample != 0:
        df = df.sample(n=sample, replace='False')


    df.loc[df['outcome_class'] == 't', 'outcome_class'] = 'T'
    df.loc[df['outcome_class'] == 'd', 'outcome_class'] = 'F'
    if question_type == 1:
        df['q'] = df['q1'].apply(lambda x: x.replace('\n', ''))
    elif question_type ==2:
        df['q'] = df['q2'].apply(lambda x: x.replace('\n', ''))
    elif question_type == 3:
        df['q1'] = df['q1'].apply(lambda x: x.replace('\n', ''))
        df['q2'] = df['q2'].apply(lambda x: x.replace('\n', ''))
        df['q'] = df['q1'] + df['q2']

    df = df.rename(columns={'q': 'sent', 'outcome_class': 'labels'})
    df = df[['sent', 'labels']]
    return df
def tokenize_function(examples: datasets.Dataset) -> datasets.Dataset:
    model_inputs = tokenizer(
        examples["sent"], text_target=examples["labels"], truncation=True)
    return model_inputs


data_path = 'data/sign_events_data_statements.csv'
question_type = 3
data_sample_size = 100
intent_df = csv_to_df(path=data_path,
                      question_type=question_type,
                      sample = data_sample_size)

dataset = Dataset.from_pandas(intent_df)
train_testvalid = dataset.train_test_split(test_size= 0.1, shuffle=True)
valid_test = train_testvalid["test"].train_test_split(test_size=0.5,shuffle=True)
dataset_splitted = DatasetDict({
    'train': train_testvalid["train"],
    'valid': valid_test["train"],
    'test': valid_test["test"]
})


checkpoint = "microsoft/Multilingual-MiniLM-L12-H384"
path = os.path.join("./models/pretrained",checkpoint)
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path)



tokenized_dataset = dataset_splitted.map(tokenize_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

num_epochs = 3
training_args = Seq2SeqTrainingArguments(
    output_dir="small_scenario_intent",
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()


print("end")

