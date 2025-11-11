import json
import os
from collections import defaultdict
import random


def process_captions(input_file, train_size=6000, val_size=1000, test_size=1000):
    # 读取并分组图像的所有描述
    image_captions = defaultdict(list)
    with open(input_file, 'r', encoding='utf-8') as f:
        # 跳过第一行（如果第一行是表头），根据实际情况决定是否保留
        # next(f)  # 若第一行是表头则解开注释

        for line in f:
            line = line.strip()
            if not line:
                continue
            # 分割图像名称和描述（处理可能包含逗号的描述）
            # 按第一个逗号分割（因为图像名不含逗号）
            parts = line.split(',', 1)
            if len(parts) != 2:
                print(f"跳过无效行: {line}")
                continue
            image_name, caption = parts[0], parts[1].strip()
            image_captions[image_name].append(caption)

    # 检查每个图像是否有5条描述
    valid_images = []
    for img, caps in image_captions.items():
        if len(caps) == 5:
            valid_images.append({"image": img, "captions": caps})
        else:
            print(f"图像 {img} 描述数量不符（实际{len(caps)}条，需5条），已跳过")

    # 检查总数量是否满足需求
    total_needed = train_size + val_size + test_size
    if len(valid_images) < total_needed:
        raise ValueError(f"有效图像数量不足（仅{len(valid_images)}个），至少需要{total_needed}个")

    # 随机打乱顺序
    random.seed(42)  # 固定种子确保划分可复现
    random.shuffle(valid_images)

    # 划分数据集
    train = valid_images[:train_size]
    val = valid_images[train_size: train_size + val_size]
    test = valid_images[train_size + val_size: train_size + val_size + test_size]

    # 保存为JSON文件
    def save_json(data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    save_json(train, 'data/train.json')
    save_json(val, 'data/val.json')
    save_json(test, 'data/test.json')

    print(f"处理完成：")
    print(f"训练集: {len(train)}个图像")
    print(f"验证集: {len(val)}个图像")
    print(f"测试集: {len(test)}个图像")


if __name__ == "__main__":
    # 替换为你的captions.txt路径
    input_file = "data/captions.txt"
    process_captions(input_file)