# -*- coding: utf-8 -*-
"""
CLIP 微调（DDP + LoRA + 文本日志 + CSV + 最佳模型）
用法示例：
  torchrun --nproc_per_node=4 clip_finetune_ddp_lora_log_csv.py ...
"""
import os, json, math, time, random, argparse, logging, csv
from datetime import datetime
from collections import defaultdict
from typing import List

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

import clip  # openai/CLIP

# =============================
# DDP helpers
# =============================
def unwrap(model):
    return model.module if isinstance(model, DDP) else model

def ddp_is_available():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if ddp_is_available() else 0

def get_world_size():
    return dist.get_world_size() if ddp_is_available() else 1

def is_main():
    return get_rank() == 0

def setup_ddp():
    """初始化分布式（torchrun 环境变量）"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

def cleanup_ddp():
    if ddp_is_available():
        dist.barrier()
        dist.destroy_process_group()

# =============================
# Logger
# =============================
def setup_logger(base_out: str, run_name: str) -> str:
    log_dir = os.path.join(base_out, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{run_name}.log")

    handlers = []
    if is_main():
        handlers = [
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    else:
        handlers = [logging.StreamHandler()]

    logging.basicConfig(
        level=logging.INFO if is_main() else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers
    )
    if is_main():
        logging.info(f"Logging to: {log_file}")
    return log_file

def init_csv_logger(base_out: str, run_name: str) -> str:
    csv_dir = os.path.join(base_out, "logs")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{run_name}.csv")
    # 先写基础列头
    if is_main():
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "lr", "steps_per_sec", "elapsed_sec"])
    return csv_path


def csv_log_epoch(csv_path: str, row_dict: dict):
    """
    追加一行。如果出现新指标列（如 t2i@1/i2t@1/...），自动扩展表头：
      1) 读回所有行
      2) 合并新列头
      3) 重写整个文件
    """
    if not is_main():
        return

    # 读取现有内容
    rows = []
    header = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, r in enumerate(reader):
                if i == 0:
                    header = r
                else:
                    rows.append(r)

    # 目标列集合 = 旧表头 ∪ 新键
    # 基础列优先保持在前面
    base_cols = ["epoch", "train_loss", "lr", "steps_per_sec", "elapsed_sec"]
    new_keys = [k for k in row_dict.keys() if k not in base_cols]
    # 已有 header 里可能也有一些指标列
    old_metric_cols = [c for c in header if c not in base_cols] if header else []
    # 合并：基础列 + 旧指标列 + 新指标列（去重，按名称排序保证稳定）
    metric_merged = sorted(list(dict.fromkeys(old_metric_cols + new_keys)))
    target_header = base_cols + metric_merged

    # 把新的一行映射成 target_header 顺序
    def map_row(d: dict, cols: list):
        return [d.get(c, "") for c in cols]

    # 旧行需要 pad 到新表头
    # 先把旧 header 映射到 target_header 的列索引
    old_idx = {c: i for i, c in enumerate(header)} if header else {}
    remapped_rows = []
    if rows:
        for r in rows:
            out = []
            for c in target_header:
                if c in old_idx and old_idx[c] < len(r):
                    out.append(r[old_idx[c]])
                else:
                    out.append("")
            remapped_rows.append(out)

    # 追加当前行
    remapped_rows.append(map_row(row_dict, target_header))

    # 重写文件
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(target_header)
        writer.writerows(remapped_rows)


# =============================
# LoRA
# =============================
class LoRALinear(nn.Module):
    """
    y = x W^T + scale * (x B) A^T
    - 冻结原始 W，仅训练 A(out_features, r) 与 B(in_features, r)
    - scale = alpha / r
    """
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0, train_bias: bool = True):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        device = base_linear.weight.device

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = (self.alpha / float(self.r)) if self.r > 0 else 0.0

        # 基座权重冻结（可选择 bias 是否训练）
        self.weight = nn.Parameter(base_linear.weight.data.clone().to(device), requires_grad=False)
        if base_linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(base_linear.bias.data.clone().to(device), requires_grad=train_bias)

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.out_features, self.r, device=device))
            self.lora_B = nn.Parameter(torch.zeros(self.in_features, self.r, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        y = x.matmul(self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        if self.r > 0:
            y = y + self.scale * self.dropout(x).matmul(self.lora_B).matmul(self.lora_A.t())
        return y

def inject_lora(model: nn.Module, r=8, alpha=16, dropout=0.0, train_bias=True):
    """
    在文本塔中为 Transformer 的 MLP (c_fc, c_proj) 与注意力 out_proj 注入 LoRA。
    不改 MHA 的 in_proj_weight。
    """
    replace = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if ("transformer.resblocks" in name) and (
                name.endswith("mlp.c_fc") or name.endswith("mlp.c_proj") or name.endswith("attn.out_proj")
            ):
                parent_name = ".".join(name.split(".")[:-1])
                leaf_name = name.split(".")[-1]
                parent = model
                for p in parent_name.split("."):
                    parent = getattr(parent, p)
                setattr(parent, leaf_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout, train_bias=train_bias))
                replace += 1
    if is_main():
        logging.info(f"[LoRA] Injected {replace} Linear layers (r={r}, alpha={alpha}, dropout={dropout}, train_bias={train_bias}).")

# =============================
# Dataset
# =============================
class ImageTextDataset(Dataset):
    """
    标注 JSON 结构：
    [
      {"image":"0001.jpg","captions":["a dog on grass","..."]},
      ...
    ]
    """
    def __init__(self, json_path, img_dir, preprocess, prompt_tmpl=None, max_captions_per_image=None):
        self.records = json.load(open(json_path, "r", encoding="utf-8"))
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.prompt_tmpl = prompt_tmpl
        self.max_captions_per_image = max_captions_per_image
        self._flat_texts = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        img_path = os.path.join(self.img_dir, rec["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)

        caption = random.choice(rec["captions"])
        if self.prompt_tmpl:
            caption = self.prompt_tmpl.format(c=caption)
        tokens = clip.tokenize([caption], truncate=True)[0]
        return image, tokens, rec["image"]

    def flatten_texts(self):
        if self._flat_texts is not None:
            return self._flat_texts
        texts, owners = [], []
        for rec in self.records:
            caps = rec["captions"]
            if self.max_captions_per_image:
                caps = caps[: self.max_captions_per_image]
            for c in caps:
                if self.prompt_tmpl:
                    c = self.prompt_tmpl.format(c=c)
                texts.append(c)
                owners.append(rec["image"])
        self._flat_texts = (texts, owners)
        return self._flat_texts

def collate_fn(batch):
    images, tokens, names = zip(*batch)
    images = torch.stack(images, dim=0)
    tokens = torch.stack(tokens, dim=0).squeeze(1)
    return images, tokens, list(names)

# =============================
# Train / Eval utils
# =============================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def cosine_lr(optimizer, base_lr, warmup, it, total_it):
    if it < warmup:
        lr = base_lr * it / max(1, warmup)
    else:
        q = (it - warmup) / max(1, total_it - warmup)
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * q))
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)
    return lr

def l2_normalize(x: torch.Tensor, dim=1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def all_gather_concat(t: torch.Tensor) -> torch.Tensor:
    """支持变长 batch 的安全 gather：用 padding + sizes 同步，然后裁剪。"""
    if get_world_size() == 1:
        return t
    with torch.no_grad():
        local_size = torch.tensor([t.shape[0]], device=t.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_size) for _ in range(get_world_size())]
        dist.all_gather(sizes, local_size)
        max_size = int(torch.stack(sizes).max())
        pad = max_size - t.shape[0]
        if pad > 0:
            t = torch.cat([t, torch.zeros(pad, *t.shape[1:], device=t.device, dtype=t.dtype)], dim=0)
        gathered = [torch.zeros_like(t) for _ in range(get_world_size())]
        dist.all_gather(gathered, t)
        outs = []
        for g, sz in zip(gathered, sizes):
            outs.append(g[: int(sz.item())])
        return torch.cat(outs, dim=0)

@torch.no_grad()
def encode_images(model, dl_img, device):
    net = unwrap(model)
    feats, names = [], []
    for images, _, nms in dl_img:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            emb = net.encode_image(images)
        emb = l2_normalize(emb)
        feats.append(emb)
        names += nms
    feats = torch.cat(feats, dim=0)
    feats = all_gather_concat(feats)
    if ddp_is_available():
        gathered_names = [None for _ in range(get_world_size())]
        dist.all_gather_object(gathered_names, names)
        names = sum(gathered_names, [])
    return feats, names

@torch.no_grad()
def encode_texts(model, texts: List[str], device, batch_size=256):
    net = unwrap(model)
    feats = []
    for i in range(0, len(texts), batch_size):
        toks = clip.tokenize(texts[i:i+batch_size]).to(device)
        with torch.no_grad():
            emb = net.encode_text(toks)
        emb = l2_normalize(emb)
        feats.append(emb)
    feats = torch.cat(feats, dim=0)
    # 不做 all_gather：每卡文本相同
    return feats

def build_gt_maps(image_names: List[str], text_owners: List[str]):
    img_id2idx = {name: i for i, name in enumerate(image_names)}
    gt_t2i = [ {img_id2idx[o]} for o in text_owners ]
    img_to_text = defaultdict(list)
    for tidx, owner in enumerate(text_owners):
        img_to_text[owner].append(tidx)
    gt_i2t = [ set(img_to_text[name]) for name in image_names ]
    return gt_t2i, gt_i2t

def recall_at_k(sim: torch.Tensor, gt_sets: List[set], K: int) -> float:
    topk = sim.topk(K, dim=1).indices.cpu().numpy()
    hit = 0
    for i, cand in enumerate(topk):
        if len(set(cand).intersection(gt_sets[i])) > 0:
            hit += 1
    return hit / len(gt_sets)

@torch.no_grad()
def evaluate(model, dl_img, texts, owners, device):
    """多卡评测：图像特征 all_gather；指标仅在 rank0 计算并返回。"""
    model.eval()
    img_feats, image_names = encode_images(model, dl_img, device)
    text_feats = encode_texts(model, texts, device)
    sim_t2i = text_feats @ img_feats.T
    sim_i2t = img_feats @ text_feats.T

    if ddp_is_available() and not is_main():
        return {}

    gt_t2i, gt_i2t = build_gt_maps(image_names, owners)
    res = {}
    for K in [1, 5, 10]:
        res[f"t2i@{K}"] = recall_at_k(sim_t2i, gt_t2i, K)
        res[f"i2t@{K}"] = recall_at_k(sim_i2t, gt_i2t, K)
    return res

# =============================
# Freeze & Param groups
# =============================
def make_param_groups(model, mode, lora_enabled=False):
    """
    mode in {"full","text-only","linear-probe","lora-only"}。
    - lora-only：仅训练 LoRA 与 logit_scale。
    """
    def freeze(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    if mode == "text-only":
        freeze(model.visual)
        params = []
        for n, p in model.named_parameters():
            if n.startswith("transformer.") or n.startswith("token_embedding") or \
               n.startswith("positional_embedding") or n.startswith("ln_final") or \
               n.startswith("text_projection") or n.startswith("logit_scale"):
                p.requires_grad = True
                params.append(p)
            else:
                p.requires_grad = False
        return [{"params": params}]
    elif mode == "linear-probe":
        freeze(model.visual)
        for n, p in model.named_parameters():
            if n.startswith("text_projection") or n.startswith("logit_scale"):
                p.requires_grad = True
            else:
                p.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params}]
    elif mode == "lora-only" and lora_enabled:
        for p in model.parameters():
            p.requires_grad = False
        trainables = []
        for m in model.modules():
            if isinstance(m, LoRALinear):
                for p in m.parameters():
                    p.requires_grad = True
                trainables += list(m.parameters())
        model.logit_scale.requires_grad = True
        trainables.append(model.logit_scale)
        return [{"params": trainables}]
    else:
        return [{"params": model.parameters()}]

# =============================
# Train loop
# =============================
def train_one_epoch(model, dl, optimizer, scaler, device, epoch, args, total_steps, base_lr):
    model.train()
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    it_global = epoch * len(dl)
    t0 = time.time()
    t_epoch0 = t0

    loss_sum = 0.0
    n_steps = 0
    last_lr = None

    for it, (images, tokens, _) in enumerate(dl):
        step = it_global + it
        lr = cosine_lr(optimizer, base_lr, args.warmup_steps, step, total_steps)
        last_lr = lr

        images = images.to(device, non_blocking=True)
        tokens  = tokens.to(device, non_blocking=True)
        targets = torch.arange(images.size(0), device=device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            logits_per_image, logits_per_text = model(images, tokens)
            loss = (loss_img(logits_per_image, targets) + loss_txt(logits_per_text, targets)) / 2

        if args.amp:
            scaler.scale(loss).backward()
            if args.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        loss_sum += loss.item()
        n_steps += 1

        if is_main() and ((it + 1) % args.log_every == 0):
            dt = time.time() - t0; t0 = time.time()
            logging.info(f"[epoch {epoch}] step {it+1}/{len(dl)}  loss={loss.item():.4f}  lr={lr:.2e}  dt={dt:.1f}s")

    elapsed = time.time() - t_epoch0
    steps_per_sec = n_steps / elapsed if elapsed > 0 else 0.0
    avg_loss = loss_sum / max(1, n_steps)

    return {
        "train_loss": avg_loss,
        "lr": last_lr if last_lr is not None else 0.0,
        "steps_per_sec": steps_per_sec,
        "elapsed_sec": elapsed
    }

def save_ckpt(model, optimizer, scaler, epoch, out_dir, filename=None):
    if not is_main():
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_epoch{epoch}.pt") if filename is None else os.path.join(out_dir, filename)
    state = {
        "model": (model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()),
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch
    }
    torch.save(state, path)
    logging.info(f"[ckpt] saved to {path}")

def load_ckpt(model, optimizer, scaler, path, device):
    if not path or not os.path.isfile(path):
        return 0
    map_location = {"cuda:%d" % 0: "cuda:%d" % torch.cuda.current_device()} if torch.cuda.is_available() else "cpu"
    obj = torch.load(path, map_location=map_location)
    try:
        (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).load_state_dict(obj["model"], strict=True)
    except Exception as e:
        if is_main():
            logging.warning(f"[ckpt] strict load failed ({e}), try strict=False")
        (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).load_state_dict(obj["model"], strict=False)
    if optimizer and "opt" in obj:
        optimizer.load_state_dict(obj["opt"])
    if scaler and obj.get("scaler"):
        scaler.load_state_dict(obj["scaler"])
    if is_main():
        logging.info(f"[ckpt] load {path}")
    return obj.get("epoch", 0) + 1

# =============================
# Args
# =============================
def build_argparser():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--ann_train", type=str, required=True)
    p.add_argument("--ann_val", type=str, required=True)
    p.add_argument("--ann_test", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--max_captions_per_image", type=int, default=None)
    # training
    p.add_argument("--model", type=str, default="ViT-B/16",
                   choices=["RN50","RN101","RN50x4","RN50x16","RN50x64","ViT-B/32","ViT-B/16","ViT-L/14","ViT-L/14@336px"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.2)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--freeze", type=str, default="full", choices=["full","text-only","linear-probe","lora-only"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=20)
    # lora
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_train_bias", action="store_true", help="训练 LoRA 层的 bias（默认不加此标志也会训练 bias）")
    # out/ckpt
    p.add_argument("--out", type=str, default="runs/clip_finetune")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval_every", type=int, default=1)
    # misc
    p.add_argument("--seed", type=int, default=42)
    return p

# =============================
# Main
# =============================
def main():
    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = True

    parser = build_argparser()
    args = parser.parse_args()

    # 运行名与日志
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.out, exist_ok=True)
    log_file = setup_logger(args.out, run_name)
    csv_path = init_csv_logger(args.out, run_name)

    # 保存超参数（rank0）
    if is_main():
        os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
        cfg_path = os.path.join(args.out, "logs", f"{run_name}_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        logging.info("===== Training Configuration =====")
        for k, v in vars(args).items():
            logging.info(f"{k}: {v}")

    set_seed(args.seed)

    if is_main():
        logging.info(f"World size={get_world_size()}, Rank={get_rank()}, Local rank={local_rank}")

    # 加载 CLIP（先 CPU -> 注入 LoRA -> 再整体搬到 GPU）
    model, preprocess = clip.load(args.model, device="cpu", jit=False)
    if args.lora:
        inject_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, train_bias=True)
    model = model.to(device)

    # 冻结策略
    param_groups = make_param_groups(model, args.freeze, lora_enabled=args.lora)
    for pg in param_groups:
        if "lr_scale" not in pg:
            pg["lr_scale"] = 1.0

    optimizer = AdamW(
        [{"params": pg["params"], "lr": args.lr * pg.get("lr_scale", 1.0), "weight_decay": args.weight_decay} for pg in param_groups]
    )
    scaler = GradScaler(enabled=args.amp)

    # DDP 包装
    if get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=True
        )

    # 数据
    train_set = ImageTextDataset(args.ann_train, args.img_dir, preprocess, args.prompt)
    val_set   = ImageTextDataset(args.ann_val,   args.img_dir, preprocess, args.prompt, args.max_captions_per_image)
    test_set  = ImageTextDataset(args.ann_test,  args.img_dir, preprocess, args.prompt, args.max_captions_per_image) if args.ann_test else None

    train_sampler = DistributedSampler(train_set, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=True) if get_world_size()>1 else None
    val_sampler   = DistributedSampler(val_set,   num_replicas=get_world_size(), rank=get_rank(), shuffle=False, drop_last=False) if get_world_size()>1 else None
    test_sampler  = DistributedSampler(test_set,  num_replicas=get_world_size(), rank=get_rank(), shuffle=False, drop_last=False) if (get_world_size()>1 and test_set) else None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              sampler=train_sampler, collate_fn=collate_fn)

    def make_img_loader(ds, sampler):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                          pin_memory=True, drop_last=False, sampler=sampler, collate_fn=collate_fn)
    val_img_loader  = make_img_loader(val_set, val_sampler)
    test_img_loader = make_img_loader(test_set, test_sampler) if test_set else None

    # 评测文本（各进程相同）
    val_texts, val_owners = val_set.flatten_texts()
    if test_set:
        test_texts, test_owners = test_set.flatten_texts()

    # 恢复
    start_epoch = load_ckpt(model, optimizer, scaler, args.resume, device)

    total_steps = args.epochs * len(train_loader)
    start_time = datetime.now()

    # 记录 best
    best_score = -1.0
    best_epoch = -1

    # 训练
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        stats = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args, total_steps, args.lr)
        save_ckpt(model, optimizer, scaler, epoch, args.out)

        epoch_row = {
            "epoch": epoch,
            "train_loss": stats["train_loss"],
            "lr": stats["lr"],
            "steps_per_sec": stats["steps_per_sec"],
            "elapsed_sec": stats["elapsed_sec"],
        }

        if ((epoch + 1) % args.eval_every == 0):
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            res = evaluate(model, val_img_loader, val_texts, val_owners, device)
            if is_main():
                logging.info(f"[val@epoch{epoch+1}] " + "  ".join([f"{k}={v:.3f}" for k,v in res.items()]))
                for k, v in res.items():
                    epoch_row[k] = v

                # 选择 best 指标：t2i@1 与 i2t@1 的均值
                score = None
                if "t2i@1" in res and "i2t@1" in res:
                    score = 0.5 * (res["t2i@1"] + res["i2t@1"])
                elif "t2i@1" in res:
                    score = res["t2i@1"]
                elif "i2t@1" in res:
                    score = res["i2t@1"]

                if score is not None and score > best_score:
                    best_score = score
                    best_epoch = epoch
                    save_ckpt(model, optimizer, scaler, epoch, args.out, filename="best.pt")
                    logging.info(f"[best] epoch={epoch}  score={best_score:.4f}  -> saved best.pt")

        # 写 CSV
        csv_log_epoch(csv_path, epoch_row)

    # 最终评测与总结
    res = evaluate(model, val_img_loader, val_texts, val_owners, device)
    if is_main():
        logging.info("[val_final] " + "  ".join([f"{k}={v:.3f}" for k,v in res.items()]))
        # 也记录一行 final（epoch 用总轮数）
        final_row = {"epoch": args.epochs, "train_loss": "", "lr": "", "steps_per_sec": "", "elapsed_sec": ""}
        for k, v in res.items():
            final_row[k] = v
        csv_log_epoch(csv_path, final_row)

    if test_set:
        res_t = evaluate(model, test_img_loader, test_texts, test_owners, device)
        if is_main():
            logging.info("[test_final] " + "  ".join([f"{k}={v:.3f}" for k,v in res_t.items()]))

    if is_main():
        end_time = datetime.now()
        elapsed = end_time - start_time
        logging.info("===== Training Complete =====")
        logging.info(f"Total time: {elapsed}")
        logging.info(f"Best epoch: {best_epoch}, best score: {best_score:.4f}")
        logging.info(f"Logs saved to: {log_file}")
        logging.info(f"CSV saved to:  {csv_path}")
        logging.info(f"Checkpoints dir: {args.out}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
