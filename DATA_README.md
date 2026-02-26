# Training Data: What Is Loaded and How to Verify

`train_mrbert.py` automatically downloads one of the datasets below from **HuggingFace Datasets** (via `--dataset`). They are public English classification benchmarks used for **sentence/text binary classification**, matching MrBERT’s classification head.

---

## 1. MRPC (default) `--dataset mrpc`

| Item | Description |
|------|--------------|
| **Full name** | GLUE MRPC - Microsoft Research Paraphrase Corpus |
| **Source** | HuggingFace `glue` subset `mrpc` |
| **Task** | Whether two sentences are semantically equivalent (binary) |
| **Labels** | 0 = not equivalent, 1 = equivalent |
| **Train/val** | ~3668 train, 408 validation |
| **Fit for MrBERT** | Yes; binary classification, `num_labels=2` |

---

## 2. IMDB `--dataset imdb`

| Item | Description |
|------|--------------|
| **Full name** | IMDB movie review sentiment |
| **Source** | HuggingFace `imdb` |
| **Task** | Sentiment binary classification (positive/negative) |
| **Labels** | 0 = negative, 1 = positive |
| **Train/val** | ~25000 each (test used as validation) |
| **Fit for MrBERT** | Yes; binary sentiment, matches head |

---

## 3. SST-2 `--dataset sst2`

| Item | Description |
|------|--------------|
| **Full name** | GLUE SST-2 - Stanford Sentiment Treebank |
| **Source** | HuggingFace `glue` subset `sst2` |
| **Task** | Single-sentence sentiment binary classification |
| **Labels** | 0 = negative, 1 = positive |
| **Train/val** | ~67349 train, 872 validation |
| **Fit for MrBERT** | Yes; binary, matches head |

---

## How to verify that the loaded data is correct

From the project directory:

```bash
source .venv/bin/activate
python inspect_data.py --dataset mrpc
```

This prints:

- **Source, task, and label meanings** for that dataset;
- **Actual number of training examples** loaded;
- **First 5 raw samples** (text + label).

Check that labels match the content (e.g. MRPC label=1 pairs are paraphrases). Use `--dataset imdb` or `--dataset sst2` for the other datasets.

---

## Summary

- **What is loaded**: Public English classification data from HuggingFace (MRPC / IMDB / SST-2), not local files.
- **Correctness**: All are binary classification with clear 0/1 labels, matching MrBERT’s 2-class head; run `inspect_data.py` on a few samples to confirm.
