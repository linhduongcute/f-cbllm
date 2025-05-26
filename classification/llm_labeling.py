import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import config as CFG
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers.utils import logging
import json

logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load Data ──────────────────────────────────────────────────────────────
print("loading data...")
train_dataset = load_dataset(args.dataset, split='train')
if args.dataset == 'SetFit/sst2':
    val_dataset = load_dataset(args.dataset, split='validation')

print("training data len: ", len(train_dataset))
if args.dataset == 'SetFit/sst2':
    print("val data len: ", len(val_dataset))

# ─── Balance Dataset ────────────────────────────────────────────────────────
d_list = []
for i in range(CFG.class_num[args.dataset]):
    d_list.append(train_dataset.filter(lambda e: e['label'] == i).select(range(1000 // CFG.class_num[args.dataset])))
train_dataset = concatenate_datasets(d_list)

if args.dataset == 'SetFit/sst2':
    d_list = []
    for i in range(CFG.class_num[args.dataset]):
        d_list.append(val_dataset.filter(lambda e: e['label'] == i).select(range(80 // CFG.class_num[args.dataset])))
    val_dataset = concatenate_datasets(d_list)

print("training labeled data len: ", len(train_dataset))
if args.dataset == 'SetFit/sst2':
    print("val labeled data len: ", len(val_dataset))

# ─── Load Model and Tokenizer ───────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

concept_set = CFG.concept_set[args.dataset]

instr = "You will be given a yes/no question, please answer with only yes or no."
temp = ""
if args.dataset == 'SetFit/sst2':
    temp = "According to the movie review: '{}', the movie has '{}' sentiment. yes or no?"
elif args.dataset == 'yelp_polarity':
    temp = "According to the Yelp review: '{}', the Yelp review has '{}' sentiment. yes or no?"
elif args.dataset == 'agnews':
    temp = "According to the news article: '{}', the news article is about '{}' topic. yes or no?"
elif args.dataset == 'dbpedia_14':
    temp = "According to the Wikipedia article: '{}', the Wikipedia article is about '{}' topic. yes or no?"

# ─── Generate Labels for Train ──────────────────────────────────────────────
print("generating train labels")
train_labels = []
train_logs = []

for i in range(len(train_dataset)):
    print("sample", str(i), end="\r")
    sample = train_dataset[CFG.example_name[args.dataset]][i]

    labels = []
    q_list = []

    for j in range(len(concept_set)):
        concept = concept_set[j]
        question = temp.format(sample, concept)
        q_list.append(question)

        messages = [
            {"role": "system", "content": "You are a chatbot who always solve the given problem exactly!"},
            {"role": "user", "content": instr + " Questions:\n" + question},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        prompt_length = input_ids.shape[1]

        for k in range(10):
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
            answer = tokenizer.decode(outputs[0][prompt_length:]).replace('<|eot_id|>', '').strip()
            if answer.lower() in ["yes", "no"]:
                break

        labels.append(1 if answer.lower() == "yes" else 0)

    train_labels.append(labels)
    train_logs.append({
        "sample": sample,
        "concept_labels": labels,
        "questions": q_list
    })

# ─── Generate Labels for Validation ─────────────────────────────────────────
val_labels = []
val_logs = []

if args.dataset == 'SetFit/sst2':
    print("generating val labels")
    for i in range(len(val_dataset)):
        print("sample", str(i), end="\r")
        sample = val_dataset[CFG.example_name[args.dataset]][i]

        labels = []
        q_list = []

        for j in range(len(concept_set)):
            concept = concept_set[j]
            question = temp.format(sample, concept)
            q_list.append(question)

            messages = [
                {"role": "system", "content": "You are a chatbot who always solve the given problem exactly!"},
                {"role": "user", "content": instr + " Questions:\n" + question},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            prompt_length = input_ids.shape[1]

            for k in range(10):
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9
                )
                answer = tokenizer.decode(outputs[0][prompt_length:]).replace('<|eot_id|>', '').strip()
                if answer.lower() in ["yes", "no"]:
                    break

            labels.append(1 if answer.lower() == "yes" else 0)

        val_labels.append(labels)
        val_logs.append({
            "sample": sample,
            "concept_labels": labels,
            "questions": q_list
        })

# ─── Save Results ───────────────────────────────────────────────────────────
d_name = args.dataset.replace('/', '_')
prefix = f"./llm_labeling/{d_name}/"
os.makedirs(prefix, exist_ok=True)

np.save(prefix + "concept_labels_train.npy", np.asarray(train_labels))
with open(prefix + "concept_labels_train.json", "w") as f:
    json.dump(train_logs, f, indent=2)

if args.dataset == 'SetFit/sst2':
    np.save(prefix + "concept_labels_val.npy", np.asarray(val_labels))
    with open(prefix + "concept_labels_val.json", "w") as f:
        json.dump(val_logs, f, indent=2)
