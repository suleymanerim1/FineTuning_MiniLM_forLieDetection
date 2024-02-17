# https://docs.wandb.ai/guides/integrations/huggingface
import pandas as pd
import numpy as np
import os
from transformers import  (AutoModel,AutoModelForSequenceClassification,
                           DataCollatorWithPadding,AutoTokenizer,TrainingArguments,Trainer)
from datasets import Dataset,DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import torch
from sklearn.metrics import accuracy_score
import wandb

question_type = 1
data_sample_size = 800
num_epochs = 10
lr = 1e-3
batch_size = 4
run_name = "q3_Fullsample_50epochs"
"""
# Model is MiniLM
# Small dataset :  800 samples , #Full Dataset (input 0 to processing function)
# Choose 1 or 2 or 3 for question type (q1, q2 or q1+q2)
# Two methods : Lora, Transfer Learning


800 samples with q1
800 samples with q2
800 samples with q1+q2
full samples with q1
full with q2
full with q1+q2
"""
os.environ["WANDB_PROJECT"]="lie_detection"
checkpoint = "microsoft/Multilingual-MiniLM-L12-H384"
model_tokenizer_path = os.path.join("./models/pretrained", checkpoint)
data_path = 'data/sign_events_data_statements.csv'

seed = 42
np.random.seed(seed)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def csv_to_df(path: str = None,question_type:int=1, sample: int = 0) -> pd.DataFrame:
    """

    :param path: "csv file path
    :param question_type: for q1 ->1, for q2->2, for q1+q2->3
    :param sample: 0 is full dataset, anything else is number of rows
    :return: pandas dataframe
    """
    df = pd.read_csv(path, encoding="ISO-8859-1")

    if sample != 0:
        df = df.sample(n=sample, replace='False')


    df.loc[df['outcome_class'] == 't', 'outcome_class'] = 1
    df.loc[df['outcome_class'] == 'd', 'outcome_class'] = 0
    if question_type == 1:
        df['q'] = df['q1'].apply(lambda x: x.replace('\n', ''))
    elif question_type ==2:
        df['q'] = df['q2'].apply(lambda x: x.replace('\n', ''))
    elif question_type == 3:
        df['q1'] = df['q1'].apply(lambda x: x.replace('\n', ''))
        df['q2'] = df['q2'].apply(lambda x: x.replace('\n', ''))
        df['q'] = df['q1'] + df['q2']

    df = df.rename(columns={'q': 'sent', 'outcome_class': 'label'})
    df = df[['sent', 'label']]
    return df
def tokenize_function(examples: Dataset) -> Dataset:
    model_inputs = tokenizer(examples["sent"],truncation=True,padding="max_length",max_length=512)
    return model_inputs



#--------------------------------------------------------------------------------
#Dataset Creation
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


id2label = {0: "F", 1: "T"}
label2id = {"F": 0, "T": 1}

tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path)
tokenized_dataset = dataset_splitted.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#-----------------------------------------------------------------------------------------
# Model Creation
model = AutoModel.from_pretrained(model_tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(
model_tokenizer_path, num_labels=2, id2label=id2label, label2id=label2id)
#print(f"full fine tuning trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

#https://huggingface.co/blog/peft
# https://huggingface.co/docs/peft/task_guides/token-classification-lora#lora-for-token-classification
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
)
model = get_peft_model(model, peft_config)
#wandb.log({"PEFT trainable parameters" : model.print_trainable_parameters()})
print(f"PEFT trainable parameters: {model.print_trainable_parameters()}")


#-------------------------------------------------------------------------------------
#Training

training_args = TrainingArguments(
    output_dir="outputs",
    learning_rate= lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    use_cpu= False,
    logging_first_step=True,
    disable_tqdm=True,
    logging_steps=50,
    report_to = "wandb",  # enable logging to W&B
    run_name = run_name,  # name of the W&B run (optional)

)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

#-------------------------------------------------------------------------------------------#
# Prediction
predictions = []
labels = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for example in tokenized_dataset["test"]:
    # Tokenize input
    inputs = tokenizer(example['sent'], return_tensors='pt')
    inputs = inputs.to(device)

    # Run inference
    outputs = model(**inputs)

    # Get predictions and labels
    prediction = torch.argmax(outputs.logits).item()
    label = example['label']  # Replace 'label' with the actual key in your dataset

    predictions.append(prediction)
    labels.append(label)

# Calculate accuracy
accuracy = accuracy_score(labels, predictions)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
# 🐝 Log accuracy to wandb
#wandb.log({"Test Accuracy": {accuracy * 100}})


