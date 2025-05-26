import argparse
import os
import json
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from transformers.utils import logging
from vllm import LLM, SamplingParams
import config as CFG

logging.set_verbosity_error()

def batch_label_split(dataset, split_name, llm, tokenizer, sampling_params, instr, temp):
    n_samples = len(dataset)
    n_concepts = len(CFG.concept_set[args.dataset])
    # Prepare logs structure
    logs = []
    for i in range(n_samples):
        sample = dataset[CFG.example_name[args.dataset]][i]
        questions = [temp.format(sample, c) for c in CFG.concept_set[args.dataset]]
        logs.append({
            "sample": sample,
            "questions": questions,
            "concept_labels": [None] * n_concepts
        })

    # Build flat list of prompts + mapping back to (i,j)
    prompts = []
    mapping = []
    for i in range(n_samples):
        for j, concept in enumerate(CFG.concept_set[args.dataset]):
            question = logs[i]["questions"][j]
            messages = [
                {"role": "system", "content": "You are a chatbot who always solve the given problem exactly!"},
                {"role": "user",   "content": instr + " Questions:\n" + question},
            ]
            enc = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(enc)
            mapping.append((i, j))

    # All prompt indices still pending
    pending = list(range(len(prompts)))

    for attempt in range(1, 11):
        if not pending:
            break
        batch_prompts = [prompts[k] for k in pending]
        outs = llm.generate(prompts=batch_prompts, sampling_params=sampling_params)

        next_pending = []
        for idx, out in enumerate(outs):
            text = out.outputs[0].text.replace("<|eot_id|>", "").strip()
            i, j = mapping[pending[idx]]
            if text.lower() in ("yes", "no"):
                logs[i]["concept_labels"][j] = 1 if text.lower() == "yes" else 0
            else:
                next_pending.append(pending[idx])
        pending = next_pending

    # Default remaining to no
    for k in pending:
        i, j = mapping[k]
        logs[i]["concept_labels"][j] = 0

    # Build label array
    labels = np.array([entry["concept_labels"] for entry in logs], dtype=int)
    return labels, logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SetFit/sst2")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── load & balance ───────────────────────────────────────────────────────
    train_ds = load_dataset(args.dataset, split="train")
    if args.dataset == "SetFit/sst2":
        val_ds = load_dataset(args.dataset, split="validation")

    # small demo balance, adjust as you need
    def balance(ds, k):
        buckets = []
        for lbl in range(CFG.class_num[args.dataset]):
            buckets.append(ds.filter(lambda e, lbl=lbl: e["label"] == lbl).select(range(k)))
        return concatenate_datasets(buckets)

    train_ds = balance(train_ds, 1000 // CFG.class_num[args.dataset])
    if args.dataset == "SetFit/sst2":
        val_ds = balance(val_ds, 80 // CFG.class_num[args.dataset])

    # ─── vLLM init ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=torch.bfloat16,
    )
    sampling_params = SamplingParams(max_tokens=256, temperature=0.6, top_p=0.9)

    # ─── prompt templates ────────────────────────────────────────────────────
    instr = "You will be given a yes/no question, please answer with only yes or no."
    if args.dataset == "SetFit/sst2":
        temp = "According to the movie review: '{}', the movie has '{}' sentiment. yes or no?"
    elif args.dataset == "yelp_polarity":
        temp = "According to the Yelp review: '{}', the Yelp review has '{}' sentiment. yes or no?"
    elif args.dataset == "agnews":
        temp = "According to the news article: '{}', the news article is about '{}' topic. yes or no?"
    elif args.dataset == "dbpedia_14":
        temp = "According to the Wikipedia article: '{}', the Wikipedia article is about '{}' topic. yes or no?"
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    # ─── batch label train & val ──────────────────────────────────────────────
    train_labels, train_logs = batch_label_split(
        train_ds, "train", llm, tokenizer, sampling_params, instr, temp
    )
    if args.dataset == "SetFit/sst2":
        val_labels, val_logs = batch_label_split(
            val_ds, "val", llm, tokenizer, sampling_params, instr, temp
        )

    # ─── save ─────────────────────────────────────────────────────────────────
    outdir = f"./llm_labeling/{args.dataset.replace('/', '_')}/"
    os.makedirs(outdir, exist_ok=True)

    np.save(os.path.join(outdir, "concept_labels_train.npy"), train_labels)
    with open(os.path.join(outdir, "concept_labels_train.json"), "w") as f:
        json.dump(train_logs, f, indent=2)

    if args.dataset == "SetFit/sst2":
        np.save(os.path.join(outdir, "concept_labels_val.npy"), val_labels)
        with open(os.path.join(outdir, "concept_labels_val.json"), "w") as f:
            json.dump(val_logs, f, indent=2)

    print("Done.")
