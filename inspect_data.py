#!/usr/bin/env python3
"""
Inspect data used by training: source, task, label meanings, and a few raw samples.
Run: python inspect_data.py --dataset mrpc
     python inspect_data.py --dataset imdb
     python inspect_data.py --dataset sst2
"""
import argparse
from datasets import load_dataset


# Dataset metadata (source, task, labels)
DATASET_INFO = {
    "mrpc": {
        "name": "GLUE MRPC (Microsoft Research Paraphrase Corpus)",
        "source": "HuggingFace: glue/mrpc (Microsoft Research paraphrase corpus)",
        "task": "Sentence-pair semantic equivalence (binary classification)",
        "label_0": "Not paraphrase",
        "label_1": "Paraphrase",
        "columns": ["sentence1", "sentence2", "label"],
        "train_size": 3668,
        "val_size": 408,
    },
    "imdb": {
        "name": "IMDB movie review sentiment",
        "source": "HuggingFace: imdb (IMDB English reviews)",
        "task": "Sentiment binary classification (positive/negative)",
        "label_0": "Negative",
        "label_1": "Positive",
        "columns": ["text", "label"],
        "train_size": 25000,
        "val_size": 25000,  # test used as val
    },
    "sst2": {
        "name": "GLUE SST-2 (Stanford Sentiment Treebank)",
        "source": "HuggingFace: glue/sst2 (Stanford Sentiment Treebank)",
        "task": "Single-sentence sentiment binary classification",
        "label_0": "Negative",
        "label_1": "Positive",
        "columns": ["sentence", "label"],
        "train_size": 67349,
        "val_size": 872,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Inspect data that train_mrbert.py will load")
    parser.add_argument("--dataset", type=str, default="mrpc", choices=["mrpc", "imdb", "sst2"])
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to print")
    args = parser.parse_args()

    info = DATASET_INFO[args.dataset]
    print("=" * 60)
    print("Dataset info (reference for verifying training data)")
    print("=" * 60)
    print(f"Name: {info['name']}")
    print(f"Source: {info['source']}")
    print(f"Task: {info['task']}")
    print(f"Label 0: {info['label_0']}")
    print(f"Label 1: {info['label_1']}")
    print(f"Train size: {info['train_size']} examples")
    print(f"Validation size: {info['val_size']} examples")
    print()

    print("Loading from HuggingFace (downloads on first run)...")
    if args.dataset == "mrpc":
        ds = load_dataset("glue", "mrpc", trust_remote_code=True)
    elif args.dataset == "imdb":
        ds = load_dataset("imdb", trust_remote_code=True)
    else:
        ds = load_dataset("glue", "sst2", trust_remote_code=True)

    train = ds["train"]
    print(f"Actual train size loaded: {len(train)}")
    print()
    print(f"First {args.num_samples} raw samples (use these to verify data):")
    print("-" * 60)

    for i in range(min(args.num_samples, len(train))):
        row = train[i]
        print(f"Sample {i+1}:")
        if "sentence1" in row:
            print(f"  sentence1: {row['sentence1'][:80]}...")
            print(f"  sentence2: {row['sentence2'][:80]}...")
        elif "sentence" in row:
            print(f"  sentence: {row['sentence'][:120]}...")
        elif "text" in row:
            print(f"  text: {row['text'][:120]}...")
        lbl = row["label"]
        print(f"  label: {lbl} ({info['label_' + str(lbl)]})")
        print()

    print("=" * 60)
    print("Conclusion: The above is the data used by train_mrbert.py. If source and samples look correct, the data is loaded correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
