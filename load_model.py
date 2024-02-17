from transformers import AutoTokenizer, AutoModel
import os

checkpoint = "microsoft/Multilingual-MiniLM-L12-H384"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

path = os.path.join("./models/pretrained",checkpoint)
tokenizer.save_pretrained(path)
model.save_pretrained(path)

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path)

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
print("outputs")