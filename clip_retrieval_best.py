# -*- coding: utf-8 -*-
"""
使用 best.pt 做自主图文检索（增强可视化版）：
- 文本查询：给一句话，检索最相似的图片，保存 Top-K 拼图到 out/
- 图片查询：给一张图片，检索最相似的文本，保存「查询图+Top-K 文本」大图到 out/
会自动：
  1) 加载 CLIP 及 best.pt
  2) 自动识别并注入 LoRA（如 ckpt 中出现 .lora_A/.lora_B）
  3) 预编码并缓存数据集的图像/文本向量，加速后续查询
  4) Top-K 结果打印到控制台，同时保存 CSV 与可视化图片
"""

import os
import re
import csv
import json
import math
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torch import nn
import clip

# -----------------------------
# Logging
# -----------------------------
def setup_logger(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"retrieval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
    )
    logging.info(f"[Log] {log_path}")
    return log_path

# -----------------------------
# LoRA（与训练时一致的简化版）
# -----------------------------
class LoRALinear(nn.Module):
    """y = x W^T + scale * (x B) A^T"""
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.0, train_bias=True):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = (self.alpha / float(self.r)) if self.r > 0 else 0.0

        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        if base_linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=train_bias)

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.out_features, self.r))
            self.lora_B = nn.Parameter(torch.zeros(self.in_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
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

def inject_lora(model: nn.Module, r=8, alpha=16, dropout=0.0, train_bias=True) -> int:
    """
    在文本塔的 Transformer 里为 mlp.c_fc / mlp.c_proj / attn.out_proj 注入 LoRA。
    返回替换层数。
    """
    replaced = 0
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
                replaced += 1
    return replaced

def ckpt_has_lora(state_dict: Dict[str, torch.Tensor]) -> bool:
    for k in state_dict.keys():
        if ".lora_A" in k or ".lora_B" in k:
            return True
    return False

# -----------------------------
# 数据集与编码
# -----------------------------
def load_dataset(json_path: str, img_dir: str, prompt_tmpl: str = None):
    records = json.load(open(json_path, "r", encoding="utf-8"))
    image_paths = [os.path.join(img_dir, r["image"]) for r in records]
    texts, owners = [], []
    for r in records:
        for c in r["captions"]:
            if prompt_tmpl:
                c = prompt_tmpl.format(c=c)
            texts.append(c)
            owners.append(r["image"])
    return image_paths, texts, owners

@torch.no_grad()
def l2_normalize(x, dim=1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

@torch.no_grad()
def encode_images(model, preprocess, image_paths: List[str], device, batch_size=128) -> np.ndarray:
    vecs = []
    for i in range(0, len(image_paths), batch_size):
        batch = []
        for p in image_paths[i:i+batch_size]:
            img = Image.open(p).convert("RGB")
            batch.append(preprocess(img))
        images = torch.stack(batch, dim=0).to(device)
        feats = model.encode_image(images)
        feats = l2_normalize(feats).cpu().numpy()
        vecs.append(feats)
    return np.concatenate(vecs, axis=0)

@torch.no_grad()
def encode_texts(model, texts: List[str], device, batch_size=256) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        toks = clip.tokenize(texts[i:i+batch_size], truncate=True).to(device)
        feats = model.encode_text(toks)
        feats = l2_normalize(feats).cpu().numpy()
        vecs.append(feats)
    return np.concatenate(vecs, axis=0)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def npy_cache_path(cache_dir: str, base: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    return os.path.join(cache_dir, safe + ".npy")

# -----------------------------
# 检索
# -----------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T  # 已 L2-normalize

# -----------------------------
# 可视化辅助
# -----------------------------
def _try_load_font(size=16):
    # 尝试加载常见字体；失败则用默认位图字体
    for name in ["Arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()

def _make_thumb(path: str, thumb_size: int = 256) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((thumb_size, thumb_size), Image.BICUBIC)
    # pad 到正方形
    w, h = img.size
    canvas = Image.new("RGB", (thumb_size, thumb_size), (30, 30, 30))
    canvas.paste(img, ((thumb_size - w) // 2, (thumb_size - h) // 2))
    return canvas

def viz_text2image_grid(
    query_text: str,
    top_paths: List[str],
    top_scores: List[float],
    save_path: str,
    grid_cols: int = 5,
    thumb: int = 224,
):
    font = _try_load_font(16)
    # 行数
    rows = math.ceil(len(top_paths) / grid_cols)
    # 每个格子下方留 36px 写 rank/score
    cell_w, cell_h = thumb, thumb + 36
    W, H = grid_cols * cell_w, rows * cell_h + 60  # 上面 60px 写 query
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    # 标题（查询文本）
    draw.text((10, 10), f"Query: {query_text}", fill=(255, 255, 255), font=font)

    for idx, p in enumerate(top_paths):
        r = idx // grid_cols
        c = idx % grid_cols
        x0, y0 = c * cell_w, 60 + r * cell_h
        thumb_img = _make_thumb(p, thumb)
        canvas.paste(thumb_img, (x0, y0))
        # 文字：rank + score + 文件名
        caption = f"#{idx+1}  score={top_scores[idx]:.4f}\n{os.path.basename(p)}"
        draw.text((x0 + 6, y0 + thumb + 4), caption, fill=(230, 230, 230), font=font)

    canvas.save(save_path, quality=95)
    return save_path

def viz_image2text_panel(
    query_image_path: str,
    top_texts: List[str],
    top_owners: List[str],
    top_scores: List[float],
    save_path: str,
    panel_w: int = 1000,
):
    # 左：查询图；右：文本列表
    font_title = _try_load_font(18)
    font_line = _try_load_font(16)

    # 准备查询图
    left_img = _make_thumb(query_image_path, 512)
    lw, lh = left_img.size

    # 右侧面板宽度
    rw = panel_w - lw - 40
    # 每条文本的高度估算
    line_h = 24
    lines_per_item = 3  # caption 可能换行，用 3*line_h 预留
    items_h = len(top_texts) * lines_per_item * line_h + 80
    H = max(lh + 40, items_h)
    W = lw + rw + 40

    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    # 粘贴查询图
    canvas.paste(left_img, (20, 20))
    draw.text((20, lh + 30), f"Query image: {os.path.basename(query_image_path)}", fill=(255, 255, 255), font=font_title)

    # 右侧文本
    x = 20 + lw + 20
    y = 20
    draw.text((x, y), "Top-K captions:", fill=(255, 255, 255), font=font_title)
    y += 36
    for i, (t, o, s) in enumerate(zip(top_texts, top_owners, top_scores), 1):
        header = f"#{i}  score={s:.4f}  owner={o}"
        draw.text((x, y), header, fill=(230, 230, 230), font=font_line)
        y += line_h
        # 分行画 caption（最长每行 50 字符左右）
        wrap = 50
        lines = [t[j:j+wrap] for j in range(0, len(t), wrap)]
        for ln in lines:
            draw.text((x+8, y), ln, fill=(200, 200, 200), font=font_line)
            y += line_h
        y += 8

    canvas.save(save_path, quality=95)
    return save_path

# -----------------------------
# 主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # 数据与模型
    parser.add_argument("--img_dir", type=str, required=True, help="图片根目录")
    parser.add_argument("--ann", type=str, required=True, help="包含 image/captions 的 JSON")
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--ckpt", type=str, required=True, help="best.pt 路径")
    parser.add_argument("--prompt", type=str, default="A photo of {c}")
    # 查询
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="文本查询（Text→Image）")
    group.add_argument("--image", type=str, help="图片查询（Image→Text）")
    parser.add_argument("--topk", type=int, default=5)
    # 预编码缓存
    parser.add_argument("--cache_dir", type=str, default="runs/retrieval_cache")
    # 推理
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None, help="如 cuda:0 / cpu，默认自动")
    # 输出
    parser.add_argument("--out", type=str, default="runs/retrieval_out")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out)
    ensure_dir(args.cache_dir)
    log_file = setup_logger(args.out)

    logging.info("===== Retrieval Configuration =====")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # 1) 加载 CLIP（先 CPU，便于注入 LoRA 后再统一搬到 GPU）
    model, preprocess = clip.load(args.model, device="cpu", jit=False)

    # 2) 加载 checkpoint（自动识别是否包含 LoRA）
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("model", ckpt)  # 兼容直接 state_dict 的情况
    if ckpt_has_lora(sd):
        replaced = inject_lora(model, r=8, alpha=16, dropout=0.05, train_bias=True)
        logging.info(f"[LoRA] Injected {replaced} layers (r=8, alpha=16, dropout=0.05).")
    else:
        logging.info("[LoRA] No LoRA keys found in checkpoint; load as vanilla CLIP.")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logging.info(f"[Load] Missing keys: {len(missing)} (first 5) {missing[:5]}")
    if unexpected:
        logging.info(f"[Load] Unexpected keys: {len(unexpected)} (first 5) {unexpected[:5]}")
    model = model.to(device).eval()
    logging.info(f"[Load] Loaded checkpoint: {args.ckpt}")

    # 3) 读数据与/或缓存
    image_paths, texts, owners = load_dataset(args.ann, args.img_dir, args.prompt)

    img_cache = npy_cache_path(args.cache_dir, f"{os.path.basename(args.ann)}_{args.model}_img_feats")
    txt_cache = npy_cache_path(args.cache_dir, f"{os.path.basename(args.ann)}_{args.model}_txt_feats")

    if os.path.exists(img_cache):
        img_feats = np.load(img_cache)
        logging.info(f"[Cache] Loaded image feats: {img_cache} shape={img_feats.shape}")
    else:
        logging.info("[Encode] Encoding images ...")
        img_feats = encode_images(model, preprocess, image_paths, device, batch_size=args.batch_size)
        np.save(img_cache, img_feats)
        logging.info(f"[Cache] Saved image feats to {img_cache}")

    if os.path.exists(txt_cache):
        txt_feats = np.load(txt_cache)
        logging.info(f"[Cache] Loaded text feats: {txt_cache} shape={txt_feats.shape}")
    else:
        logging.info("[Encode] Encoding texts ...")
        txt_feats = encode_texts(model, texts, device, batch_size=256)
        np.save(txt_cache, txt_feats)
        logging.info(f"[Cache] Saved text feats to {txt_cache}")

    # 4) 执行查询 + 可视化 + CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.text is not None:
        # Text → Image
        with torch.no_grad():
            q_tok = clip.tokenize([args.text], truncate=True).to(device)
            q_feat = model.encode_text(q_tok)
            q_feat = l2_normalize(q_feat).cpu().numpy()  # [1, D]
        sim = cosine_sim(q_feat, img_feats)            # [1, N_img]
        idx = np.argsort(-sim[0])[:args.topk]
        scores = [float(sim[0, i]) for i in idx]
        paths = [image_paths[i] for i in idx]

        logging.info(f"[Query: Text→Image] \"{args.text}\"")
        for r, (p, s) in enumerate(zip(paths, scores), 1):
            logging.info(f"Top{r}: {p}  (score={s:.4f})")

        # CSV
        csv_path = os.path.join(args.out, "text2image_results.csv")
        new_file = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["time", "query", "rank", "image_path", "score"])
            for r, (p, s) in enumerate(zip(paths, scores), 1):
                writer.writerow([timestamp, args.text, r, p, s])
        logging.info(f"[CSV] {csv_path}")

        # 可视化拼图
        vis_path = os.path.join(args.out, f"t2i_{timestamp}.jpg")
        viz_text2image_grid(args.text, paths, scores, vis_path, grid_cols=min(5, args.topk), thumb=224)
        logging.info(f"[VIS] Saved Top-{args.topk} grid to {vis_path}")

        # 尝试打开图片（在桌面环境/VSCode 中会弹出查看器；纯 SSH 环境会忽略）
        try:
            Image.open(vis_path).show()
        except Exception:
            pass

    if args.image is not None:
        # Image → Text
        with torch.no_grad():
            img = Image.open(args.image).convert("RGB")
            q_img = preprocess(img).unsqueeze(0).to(device)
            q_feat = model.encode_image(q_img)
            q_feat = l2_normalize(q_feat).cpu().numpy()  # [1, D]
        sim = cosine_sim(q_feat, txt_feats)             # [1, N_txt]
        idx = np.argsort(-sim[0])[:args.topk]
        scores = [float(sim[0, i]) for i in idx]
        txts = [texts[i] for i in idx]
        ownr = [owners[i] for i in idx]

        logging.info(f"[Query: Image→Text] {args.image}")
        for r, (t, o, s) in enumerate(zip(txts, ownr, scores), 1):
            logging.info(f"Top{r}: \"{t}\"  (owner={o})  (score={s:.4f})")

        # CSV
        csv_path = os.path.join(args.out, "image2text_results.csv")
        new_file = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["time", "query_image", "rank", "caption", "owner_image", "score"])
            for r, (t, o, s) in enumerate(zip(txts, ownr, scores), 1):
                writer.writerow([timestamp, args.image, r, t, o, s])
        logging.info(f"[CSV] {csv_path}")

        # 可视化「查询图 + 文本面板」
        vis_path = os.path.join(args.out, f"i2t_{timestamp}.jpg")
        viz_image2text_panel(args.image, txts, ownr, scores, vis_path, panel_w=1200)
        logging.info(f"[VIS] Saved panel to {vis_path}")

        # 尝试打开图片
        try:
            Image.open(vis_path).show()
        except Exception:
            pass

    logging.info("[Done] Retrieval finished.")

if __name__ == "__main__":
    main()
