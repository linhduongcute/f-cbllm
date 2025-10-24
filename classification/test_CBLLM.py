import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset
import evaluate
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from utils import normalize, eos_pooling

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--cbl_path", type=str, default="mpnet_acs/SetFit_sst2/roberta_cbm/cbl.pt")
parser.add_argument('--sparse', action=argparse.BooleanOptionalAction)
parser.add_argument("--batch_size", type=int, default=256)

parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.texts.items()}
        return t

    def __len__(self):
        return len(self.texts['input_ids'])


def build_loaders(texts, mode):
    dataset = ClassificationDataset(texts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    acs = args.cbl_path.split("/")[0]
    dataset = args.cbl_path.split("/")[1] if 'SOF' not in args.cbl_path.split("/")[1] else args.cbl_path.split("/")[1].replace('_', '/')
    backbone = args.cbl_path.split("/")[2]
    cbl_name = args.cbl_path.split("/")[-1]
    
    print("loading data...")
    test_dataset = load_dataset(dataset, split='test')
    print("test data len: ", len(test_dataset))
    print("tokenizing...")
    if 'roberta' in backbone:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif 'gpt2' in backbone:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise Exception("backbone should be roberta or gpt2")

    encoded_test_dataset = test_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True, batch_size=len(test_dataset))
    encoded_test_dataset = encoded_test_dataset.remove_columns([CFG.example_name[dataset]])
    if dataset == 'SetFit/sst2':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['label_text'])
    if dataset == 'dbpedia_14':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['title'])
    encoded_test_dataset = encoded_test_dataset[:len(encoded_test_dataset)]

    print("creating loader...")
    test_loader = build_loaders(encoded_test_dataset, mode="test")


    concept_set = CFG.concept_set[dataset]
    if 'roberta' in backbone:
        if 'no_backbone' in cbl_name:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            cbl.eval()
            preLM = RobertaModel.from_pretrained('roberta-base').to(device)
            preLM.eval()
        else:
            print("preparing backbone(roberta)+CBL...")
            backbone_cbl = RobertaCBL(len(concept_set), args.dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            backbone_cbl.eval()
    elif 'gpt2' in backbone:
        if 'no_backbone' in cbl_name:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            cbl.eval()
            preLM = GPT2Model.from_pretrained('gpt2').to(device)
            preLM.eval()
        else:
            print("preparing backbone(gpt2)+CBL...")
            backbone_cbl = GPT2CBL(len(concept_set), args.dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            backbone_cbl.eval()
    else:
        raise Exception("backbone should be roberta or gpt2")

    print("get concept features...")
    FL_test_features = []
    h_pre_list = []
    h_post_list = []
    
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if 'no_backbone' in cbl_name:
                # backbone riêng + CBL riêng
                h_pre = preLM(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"]).last_hidden_state
                if args.backbone == 'roberta':
                    h_pre = h_pre[:, 0, :]
                elif args.backbone == 'gpt2':
                    h_pre = eos_pooling(h_pre, batch["attention_mask"])
                else:
                    raise Exception("backbone should be roberta or gpt2")

                h_post = cbl(h_pre)

            else:
                # Trường hợp backbone_cbl là RobertaCBL (gộp sẵn backbone + projection)
                # Tách ra thủ công từng bước để lấy cả hai dạng embedding
                h_pre = backbone_cbl.preLM(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"]).last_hidden_state[:, 0, :]
                projected = backbone_cbl.projection(h_pre)
                x = backbone_cbl.gelu(projected)
                x = backbone_cbl.fc(x)
                x = backbone_cbl.dropout(x)
                h_post = x + projected

            # Lưu
            h_pre_list.append(h_pre.detach().cpu())
            h_post_list.append(h_post.detach().cpu())
            FL_test_features.append(h_post)

    # Concat embedding
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()
    h_pre = torch.cat(h_pre_list, dim=0)
    h_post = torch.cat(h_post_list, dim=0)

    # save h_pre and h_post
    prefix = "./" + acs + "/" + dataset.replace('/', '_') + "/" + backbone + "/"
    save_dir = prefix + "embeddings/"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(h_pre, os.path.join(save_dir, "h_pre.pt"))
    torch.save(h_post, os.path.join(save_dir, "h_post.pt"))
    print(f"Saved h_pre and h_post to {save_dir}")


    prefix = "./" + acs + "/" + dataset.replace('/', '_') + "/" + backbone + "/"
    model_name = cbl_name[3:]
    train_mean = torch.load(prefix + 'train_mean' + model_name)
    train_std = torch.load(prefix + 'train_std' + model_name)

    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)

    test_y = torch.LongTensor(encoded_test_dataset["label"])

    final = torch.nn.Linear(in_features=len(concept_set), out_features=CFG.class_num[dataset])
    W_g_path = prefix + "W_g"
    b_g_path = prefix + "b_g"
    if args.sparse:
        W_g_path += "_sparse"
        b_g_path += "_sparse"
    W_g_path += model_name
    b_g_path += model_name
    W_g = torch.load(W_g_path)
    b_g = torch.load(b_g_path)
    final.load_state_dict({"weight": W_g, "bias": b_g})
    metric = evaluate.load("accuracy")
    with torch.torch.no_grad():
        pred = np.argmax(final(test_c).detach().numpy(), axis=-1)
    metric.add_batch(predictions=pred, references=encoded_test_dataset["label"])
    print(metric.compute())