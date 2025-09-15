from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer,DataCollatorWithPadding
import numpy as np
import yaml

config_file = "/kaggle/working/Orpheus-TTS-NL/finetune/config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenizer.pad_token_id = 128263

ds = load_dataset(dsn, split="train") 


training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}", 
    save_steps=save_steps,
    remove_unused_columns=False, 
    learning_rate=learning_rate,
    report_to="none" 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

trainer.train()

