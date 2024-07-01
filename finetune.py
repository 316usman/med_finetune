# This code is for fine-tuning Phi-3 Models.
# Note thi requires 7.4 GB of GPU RAM for the process.
# Model available at https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3
# Model Names 
# microsoft/Phi-3-mini-4k-instruct
# microsoft/Phi-3-mini-128k-instruct
# microsoft/Phi-3-small-8k-instruct
# microsoft/Phi-3-small-128k-instruct
# microsoft/Phi-3-medium-4k-instruct
# microsoft/Phi-3-medium-128k-instruct
# microsoft/Phi-3-vision-128k-instruct
# microsoft/Phi-3-mini-4k-instruct-onnx
# microsoft/Phi-3-mini-4k-instruct-onnx-web
# microsoft/Phi-3-mini-128k-instruct-onnx
# microsoft/Phi-3-small-8k-instruct-onnx-cuda
# microsoft/Phi-3-small-128k-instruct-onnx-cuda
# microsoft/Phi-3-medium-4k-instruct-onnx-cpu
# microsoft/Phi-3-medium-4k-instruct-onnx-cuda
# microsoft/Phi-3-medium-4k-instruct-onnx-directml
# microsoft/Phi-3-medium-128k-instruct-onnx-cpu
# microsoft/Phi-3-medium-128k-instruct-onnx-cuda
# microsoft/Phi-3-medium-128k-instruct-onnx-directml
# microsoft/Phi-3-mini-4k-instruct-gguf
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')

# Load the dataset for fine-tuning
dataset = load_dataset("316usman/pubmedqa_train_cleaned", split="train")

# # Define the formatting function for the prompts
# def formatting_prompts_func(examples):
#     convos = examples["conversations"]
#     texts = []
#     mapper = {"system": "system\n", "human": "\nuser\n", "gpt": "\nassistant\n"}
#     end_mapper = {"system": "", "human": "", "gpt": ""}
#     for convo in convos:
#         text = "".join(f"{mapper[(turn := x['from'])]} {x['value']}\n{end_mapper[turn]}" for x in convo)
#         texts.append(f"{text}{tokenizer.eos_token}")
#     return {"text": texts}

# I see, in that case, we can modify the function to return a dictionary with a single field text that contains a string representation of the formatted example. Here's an updated version of the function:

def formatting_prompts_func(example):
    texts=[]
    user_input = example["data"]["QUESTION"] + " " + example["data"]["CONTEXT"]
    # Use the LONG_ANSWER field as the assistant's response
    assistant_response = example["data"]["LONG_ANSWER"]
    # Format the example as a string
    formatted_example = f"{user_input}\n\n{assistant_response}"
    # Append the formatted example to the list
    texts.append(formatted_example)
    # Return a dictionary with the formatted examples
    return {"text": texts}

# Apply the formatting function to the dataset
dataset1 = dataset.map(formatting_prompts_func, batched=False)
print (dataset1[20000]['text'])
# Define the training arguments
# args = TrainingArguments(
#     evaluation_strategy="steps",
#     per_device_train_batch_size=7,
#     gradient_accumulation_steps=4,
#     gradient_checkpointing=True,
#     learning_rate=1e-4,
#     fp16=True,
#     max_steps=-1,
#     num_train_epochs=1,
#     save_strategy="epoch",
#     logging_steps=10,
#     output_dir='finetuned_model',
#     optim="paged_adamw_32bit",
#     lr_scheduler_type="linear"
# )

# # Create the trainer
# trainer = SFTTrainer(
#     model=model,
#     args=args,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     max_seq_length=128,
#     formatting_func=formatting_prompts_func,
#     tokenizer = tokenizer
# )

# # Start the training process
# trainer.train()
