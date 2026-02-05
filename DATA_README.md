# 训练数据说明：拉取的是什么、是否正确

`train_mrbert.py` 会从 **HuggingFace Datasets** 自动下载下面三种数据集之一（由 `--dataset` 指定）。数据都是公开、常用的英文分类 benchmark，**用于做句子/文本二分类**，和 MrBERT 的分类头是匹配的。

---

## 1. MRPC（默认）`--dataset mrpc`

| 项目 | 说明 |
|------|------|
| **全称** | GLUE MRPC - Microsoft Research Paraphrase Corpus |
| **拉取来源** | HuggingFace 上的 `glue` 子集 `mrpc` |
| **任务** | 判断两个句子是否语义等价（二分类） |
| **标签** | 0 = 不等价，1 = 等价 |
| **训练/验证** | 约 3668 条训练，408 条验证 |
| **是否适合 MrBERT** | 适合，任务就是二分类，和 `num_labels=2` 一致 |

---

## 2. IMDB `--dataset imdb`

| 项目 | 说明 |
|------|------|
| **全称** | IMDB 电影评论情感 |
| **拉取来源** | HuggingFace 上的 `imdb` |
| **任务** | 影评情感二分类（正面/负面） |
| **标签** | 0 = 负面，1 = 正面 |
| **训练/验证** | 各约 25000 条（用 test 做验证） |
| **是否适合 MrBERT** | 适合，二分类情感，和分类头一致 |

---

## 3. SST-2 `--dataset sst2`

| 项目 | 说明 |
|------|------|
| **全称** | GLUE SST-2 - Stanford Sentiment Treebank |
| **拉取来源** | HuggingFace 上的 `glue` 子集 `sst2` |
| **任务** | 单句情感二分类 |
| **标签** | 0 = 负面，1 = 正面 |
| **训练/验证** | 约 67349 条训练，872 条验证 |
| **是否适合 MrBERT** | 适合，二分类，和分类头一致 |

---

## 如何确认「拉取的数据是否正确」？

在项目目录下运行：

```bash
source .venv/bin/activate
python inspect_data.py --dataset mrpc
```

会打印：
- 该数据集的**来源、任务、标签含义**；
- **实际拉取到的训练集条数**；
- **前 5 条原始样本**（句子 + 标签）。

你看一下句子和标签是否合理（例如 MRPC 里 label=1 的两句是否确实语义等价），就能判断数据是否正确。换数据集可加 `--dataset imdb` 或 `--dataset sst2`。

---

## 总结

- **拉取的是**：HuggingFace 上的公开英文分类数据（MRPC / IMDB / SST-2），不是本地文件。
- **是否正确**：任务都是二分类，标签 0/1 含义明确，和 MrBERT 的 2 类分类头一致；用 `inspect_data.py` 看几条样本即可最终确认。
