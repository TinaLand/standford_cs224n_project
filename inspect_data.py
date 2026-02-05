#!/usr/bin/env python3
"""
查看训练时会用到的数据：来源、任务、标签含义、以及若干条原始样本。
运行: python inspect_data.py --dataset mrpc
      python inspect_data.py --dataset imdb
      python inspect_data.py --dataset sst2
"""
import argparse
from datasets import load_dataset


# 各数据集的说明（来源、任务、标签是否正确）
DATASET_INFO = {
    "mrpc": {
        "name": "GLUE MRPC (Microsoft Research Paraphrase Corpus)",
        "source": "HuggingFace: glue/mrpc，来自微软研究发布的句子对语料",
        "task": "句子对语义是否等价（二分类）",
        "label_0": "不等价 (Not paraphrase)",
        "label_1": "等价 (Paraphrase)",
        "columns": ["sentence1", "sentence2", "label"],
        "train_size": 3668,
        "val_size": 408,
    },
    "imdb": {
        "name": "IMDB 电影评论情感",
        "source": "HuggingFace: imdb，来自 IMDB 英文影评",
        "task": "情感二分类（正面/负面）",
        "label_0": "负面 (Negative)",
        "label_1": "正面 (Positive)",
        "columns": ["text", "label"],
        "train_size": 25000,
        "val_size": 25000,  # 用 test 做 val
    },
    "sst2": {
        "name": "GLUE SST-2 (Stanford Sentiment Treebank)",
        "source": "HuggingFace: glue/sst2，来自斯坦福情感树库",
        "task": "单句情感二分类",
        "label_0": "负面 (Negative)",
        "label_1": "正面 (Positive)",
        "columns": ["sentence", "label"],
        "train_size": 67349,
        "val_size": 872,
    },
}


def main():
    parser = argparse.ArgumentParser(description="查看 train_mrbert.py 会拉取的数据")
    parser.add_argument("--dataset", type=str, default="mrpc", choices=["mrpc", "imdb", "sst2"])
    parser.add_argument("--num_samples", type=int, default=5, help="打印前几条样本")
    args = parser.parse_args()

    info = DATASET_INFO[args.dataset]
    print("=" * 60)
    print("数据集说明（训练时拉取的数据是否正确的参考）")
    print("=" * 60)
    print(f"名称: {info['name']}")
    print(f"来源: {info['source']}")
    print(f"任务: {info['task']}")
    print(f"标签 0: {info['label_0']}")
    print(f"标签 1: {info['label_1']}")
    print(f"训练集大小: {info['train_size']} 条")
    print(f"验证集大小: {info['val_size']} 条")
    print()

    print("正在从 HuggingFace 拉取数据（首次会下载）...")
    if args.dataset == "mrpc":
        ds = load_dataset("glue", "mrpc", trust_remote_code=True)
    elif args.dataset == "imdb":
        ds = load_dataset("imdb", trust_remote_code=True)
    else:
        ds = load_dataset("glue", "sst2", trust_remote_code=True)

    train = ds["train"]
    print(f"实际拉取到的 train 条数: {len(train)}")
    print()
    print(f"前 {args.num_samples} 条原始样本（你可据此判断数据是否正确）：")
    print("-" * 60)

    for i in range(min(args.num_samples, len(train))):
        row = train[i]
        print(f"样本 {i+1}:")
        if "sentence1" in row:
            print(f"  sentence1: {row['sentence1'][:80]}...")
            print(f"  sentence2: {row['sentence2'][:80]}...")
        elif "sentence" in row:
            print(f"  sentence: {row['sentence'][:120]}...")
        elif "text" in row:
            print(f"  text: {row['text'][:120]}...")
        print(f"  label: {row['label']} ({info['label_' + str(row['label'])]})")
        print()

    print("=" * 60)
    print("结论: 上面即 train_mrbert.py 使用的数据。若来源和样本符合预期，即表示拉取正确。")
    print("=" * 60)


if __name__ == "__main__":
    main()
