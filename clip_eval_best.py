# -*- coding: utf-8 -*-
"""
使用 best.pt 对测试集进行评估
支持分布式（DDP）环境，可直接用于 CLIP + LoRA 微调后的模型验证
"""

import os
import json
import logging
import torch
import clip
import argparse
from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import defaultdict
import pandas as pd
import torch.distributed as dist

# =============================
# Utils
# =============================
def setup_logger(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
    )
    logging.info(f"[Log] Evaluation log file: {log_file}")
    return log_file

def ddp_is_available():
    return dist.is_available() and dist.is_initialized()

def unwrap(model):
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

def get_rank():
    return dist.get_rank() if ddp_is_available() else 0

def get_world_size():
    return dist.get_world_size() if ddp_is_available() else 1

def is_main():
    return get_rank() == 0

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

def cleanup_ddp():
    if ddp_is_available():
        dist.barrier()
        dist.destroy_process_group()

# =============================
# Dataset
# =============================
class ImageTextDataset(Dataset):
    def __init__(self, json_path, img_dir, preprocess, prompt_tmpl=None):
        self.records = json.load(open(json_path, "r", encoding="utf-8"))
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.prompt_tmpl = prompt_tmpl

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        img_path = os.path.join(self.img_dir, rec["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)
        caption = rec["captions"][0]  # 取第一条描述
        if self.prompt_tmpl:
            caption = self.prompt_tmpl.format(c=caption)
        tokens = clip.tokenize([caption], truncate=True)[0]
        return image, tokens, rec["image"]

    def flatten_texts(self):
        texts, owners = [], []
        for rec in self.records:
            for c in rec["captions"]:
                if self.prompt_tmpl:
                    c = self.prompt_tmpl.format(c=c)
                texts.append(c)
                owners.append(rec["image"])
        return texts, owners

def collate_fn(batch):
    images, tokens, names = zip(*batch)
    images = torch.stack(images)
    tokens = torch.stack(tokens)
    return images, tokens.squeeze(1), list(names)

# =============================
# Eval helpers
# =============================
@torch.no_grad()
def l2_normalize(x, dim=1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

@torch.no_grad()
def encode_images(model, dataloader, device):
    net = unwrap(model)
    feats, names = [], []
    for images, _, nms in dataloader:
        images = images.to(device)
        emb = net.encode_image(images)
        feats.append(l2_normalize(emb))
        names += nms
    feats = torch.cat(feats, dim=0)
    return feats, names

@torch.no_grad()
def encode_texts(model, texts, device, batch_size=256):
    net = unwrap(model)
    feats = []
    for i in range(0, len(texts), batch_size):
        toks = clip.tokenize(texts[i:i+batch_size]).to(device)
        emb = net.encode_text(toks)
        feats.append(l2_normalize(emb))
    return torch.cat(feats, dim=0)

def build_gt_maps(image_names, text_owners):
    img_id2idx = {n: i for i, n in enumerate(image_names)}
    gt_t2i = [{img_id2idx[o]} for o in text_owners]
    img_to_text = defaultdict(list)
    for tidx, owner in enumerate(text_owners):
        img_to_text[owner].append(tidx)
    gt_i2t = [set(img_to_text[n]) for n in image_names]
    return gt_t2i, gt_i2t

def recall_at_k(sim, gt_sets, K):
    topk = sim.topk(K, dim=1).indices.cpu().numpy()
    hit = sum(1 for i, cand in enumerate(topk) if len(set(cand).intersection(gt_sets[i])) > 0)
    return hit / len(gt_sets)

@torch.no_grad()
def evaluate(model, dataloader, texts, owners, device):
    model.eval()
    img_feats, image_names = encode_images(model, dataloader, device)
    text_feats = encode_texts(model, texts, device)
    sim_t2i = text_feats @ img_feats.T
    sim_i2t = img_feats @ text_feats.T
    gt_t2i, gt_i2t = build_gt_maps(image_names, owners)
    res = {}
    for K in [1, 5, 10]:
        res[f"t2i@{K}"] = recall_at_k(sim_t2i, gt_t2i, K)
        res[f"i2t@{K}"] = recall_at_k(sim_i2t, gt_i2t, K)
    return res

# =============================
# Main eval
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--ann_test", type=str, required=True)
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="A photo of {c}")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out", type=str, default="runs/eval")
    args = parser.parse_args()

    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    log_file = setup_logger(args.out)
    logging.info("===== Evaluation Configuration =====")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # 加载 CLIP
    model, preprocess = clip.load(args.model, device=device, jit=False)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    logging.info(f"[Load] Loaded checkpoint: {args.ckpt}")

    # 测试集
    test_set = ImageTextDataset(args.ann_test, args.img_dir, preprocess, args.prompt)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_texts, test_owners = test_set.flatten_texts()

    # 评估
    logging.info("[Eval] Running evaluation...")
    res = evaluate(model, test_loader, test_texts, test_owners, device)

    # 打印结果
    msg = "  ".join([f"{k}={v:.3f}" for k, v in res.items()])
    logging.info(f"[test_final] {msg}")

    # 保存结果到 CSV
    csv_path = os.path.join(args.out, "eval_results.csv")
    df = pd.DataFrame([res])
    df.insert(0, "checkpoint", os.path.basename(args.ckpt))
    df.to_csv(csv_path, index=False)
    logging.info(f"[CSV] Saved evaluation results to {csv_path}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
