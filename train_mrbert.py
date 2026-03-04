#!/usr/bin/env python3
"""
Train MrBERT on a real dataset (e.g. GLUE MRPC or IMDB).
Loss: L = L_CE + alpha * L_G (paper Eq.3); alpha = gate_weight or from PI controller.
Optional two-phase: Phase A = gate adaptation (first N steps with gate regularizer),
Phase B = task finetuning (same loss, typically lower gate_weight).
"""
import argparse
import time
import sys
import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset

from mrbert import MrBertForSequenceClassification, MrBertForQuestionAnswering, MrXLMRobertaForSequenceClassification, MrXLMRobertaForQuestionAnswering
from mrbert.pi_controller import PIController


def _prepare_tydiqa_example(ex, tokenizer, max_length):
    """Tokenize one TyDi QA example and compute start/end token positions. Drops if answer not in span.
    Works for both BERT (token_type_ids) and XLM-R (no token_type_ids; use second </s> to find context)."""
    question = ex["question"]
    context = ex["context"]
    answers = ex["answers"]
    if not answers or not answers["text"] or not answers["answer_start"]:
        return None
    answer_start_char = answers["answer_start"][0]
    answer_text = answers["text"][0]
    answer_end_char = answer_start_char + len(answer_text)
    inputs = tokenizer(
        question,
        context,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )
    offset_mapping = inputs.pop("offset_mapping")
    token_type_ids = inputs.get("token_type_ids")
    input_ids = inputs.get("input_ids", [])

    # Find first token index of context (so we only consider context tokens for answer span).
    # BERT: token_type_ids == 1; XLM-R: no token_type_ids, context starts after second </s>.
    context_start_idx = 0
    if token_type_ids is not None:
        for i in range(len(token_type_ids)):
            if token_type_ids[i] == 1:
                context_start_idx = i
                break
    else:
        # XLM-R / RoBERTa: <s> question </s></s> context </s>. Find second sep token.
        sep_id = getattr(tokenizer, "sep_token_id", None) or getattr(
            tokenizer, "pad_token_id", tokenizer.convert_tokens_to_ids("</s>")
        )
        sep_count = 0
        for i, tid in enumerate(input_ids):
            if tid == sep_id:
                sep_count += 1
                if sep_count >= 2:
                    context_start_idx = i + 1
                    break

    # Character start of context in the concatenated string (for offset_mapping).
    context_start_char = 0
    if context_start_idx < len(offset_mapping) and offset_mapping[context_start_idx] != (0, 0):
        context_start_char = offset_mapping[context_start_idx][0]
    answer_start_global = context_start_char + answer_start_char
    answer_end_global = context_start_char + answer_end_char

    start_token = -1
    end_token = -1
    for i in range(context_start_idx, len(offset_mapping)):
        s, e = offset_mapping[i]
        if s == 0 and e == 0:
            continue
        if s <= answer_start_global < e:
            start_token = i
        if s < answer_end_global <= e:
            end_token = i
            break
    if start_token < 0 or end_token < 0 or end_token < start_token:
        return None
    inputs["start_positions"] = start_token
    inputs["end_positions"] = end_token
    # XLM-R does not use token_type_ids; remove so downstream can omit the key.
    if token_type_ids is None:
        inputs.pop("token_type_ids", None)
    return inputs


def get_dataset_and_tokenizer(dataset_name: str, max_length: int, backbone: str = "bert"):
    """Load dataset and tokenizer. Supports 'mrpc', 'imdb', 'sst2', 'snli', 'xnli', 'tydiqa'.
    backbone: 'bert' or 'xlmr'. Returns (train_ds, val_ds, tokenizer, num_labels, task_type)."""
    if backbone == "xlmr":
        from transformers import XLMRobertaTokenizerFast
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if dataset_name == "mrpc":
        ds = load_dataset("glue", "mrpc")
        def tokenize(ex):
            return tokenizer(
                ex["sentence1"],
                ex["sentence2"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        num_labels = 2
        task_type = "classification"
    elif dataset_name == "imdb":
        ds = load_dataset("imdb")
        def tokenize(ex):
            return tokenizer(
                ex["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        num_labels = 2
        task_type = "classification"
    elif dataset_name == "sst2":
        ds = load_dataset("glue", "sst2")
        def tokenize(ex):
            return tokenizer(
                ex["sentence"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        num_labels = 2
        task_type = "classification"
    elif dataset_name == "snli":
        ds = load_dataset("snli")
        # SNLI has label -1 for some invalid rows; keep only 0, 1, 2
        for split in list(ds.keys()):
            ds[split] = ds[split].filter(lambda x: x["label"] >= 0)
        def tokenize(ex):
            return tokenizer(
                ex["premise"],
                ex["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        num_labels = 3  # entailment (0), neutral (1), contradiction (2)
        task_type = "classification"
    elif dataset_name == "xnli":
        # XNLI: cross-lingual NLI (default English for training; can eval other langs later)
        ds = load_dataset("xnli", "en")
        def tokenize(ex):
            return tokenizer(
                ex["premise"],
                ex["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        num_labels = 3  # entailment (0), neutral (1), contradiction (2)
        task_type = "classification"
    elif dataset_name == "tydiqa":
        ds = load_dataset("tydiqa", "secondary_task")
        def prepare_tydiqa_batch(examples):
            out = {"input_ids": [], "attention_mask": [], "start_positions": [], "end_positions": []}
            for i in range(len(examples["id"])):
                ex = {k: v[i] for k, v in examples.items()}
                row = _prepare_tydiqa_example(ex, tokenizer, max_length)
                if row is None:
                    continue
                out["input_ids"].append(row["input_ids"])
                out["attention_mask"].append(row["attention_mask"])
                if "token_type_ids" in row:
                    out.setdefault("token_type_ids", []).append(row["token_type_ids"])
                out["start_positions"].append(row["start_positions"])
                out["end_positions"].append(row["end_positions"])
            return out
        def map_tydiqa(split_ds):
            mapped = split_ds.map(
                lambda ex: _prepare_tydiqa_example(
                    {k: ex[k][0] for k in ex},
                    tokenizer,
                    max_length,
                ),
                remove_columns=split_ds.column_names,
                num_proc=1,
            )
            filtered = [x for x in mapped if x is not None]
            if not filtered:
                return split_ds.flatten_indices()
            from datasets import Dataset
            return Dataset.from_list(filtered)
        train_raw = ds["train"]
        val_raw = ds["validation"]
        train_list = []
        for i in range(len(train_raw)):
            ex = train_raw[i]
            row = _prepare_tydiqa_example(ex, tokenizer, max_length)
            if row is not None:
                train_list.append(row)
        from datasets import Dataset
        train_ds = Dataset.from_list(train_list)
        val_list = []
        for i in range(len(val_raw)):
            ex = val_raw[i]
            row = _prepare_tydiqa_example(ex, tokenizer, max_length)
            if row is not None:
                val_list.append(row)
        val_ds = Dataset.from_list(val_list)
        qa_columns = ["input_ids", "attention_mask", "start_positions", "end_positions"]
        if train_list and "token_type_ids" in train_list[0]:
            qa_columns.insert(2, "token_type_ids")
        train_ds.set_format("torch", columns=qa_columns)
        val_ds.set_format("torch", columns=qa_columns)
        return train_ds, val_ds, tokenizer, None, "qa"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use mrpc, imdb, sst2, snli, xnli, or tydiqa.")

    # Keep 'label' (and 'idx' if present)
    train_remove_cols = [c for c in ds["train"].column_names if c not in ("label", "idx")]
    train_ds = ds["train"].map(tokenize, batched=True, remove_columns=train_remove_cols)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    if "validation" in ds:
        val_remove_cols = [c for c in ds["validation"].column_names if c not in ("label", "idx")]
        val_ds = ds["validation"].map(tokenize, batched=True, remove_columns=val_remove_cols)
    else:
        val_remove_cols = [c for c in ds["test"].column_names if c not in ("label", "idx")]
        val_ds = ds["test"].map(tokenize, batched=True, remove_columns=val_remove_cols)
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_ds, val_ds, tokenizer, num_labels, task_type


def main():
    parser = argparse.ArgumentParser(description="Train MrBERT on a classification dataset")
    parser.add_argument("--dataset", type=str, default="mrpc", choices=["mrpc", "imdb", "sst2", "snli", "xnli", "tydiqa"],
                        help="Dataset: mrpc, imdb, sst2, snli, xnli, or tydiqa")
    parser.add_argument("--backbone", type=str, default="bert", choices=["bert", "xlmr"],
                        help="Backbone: bert (MrBERT) or xlmr (MrXLM / XLM-R). QA only supported with bert.")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Train batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--gate_weight", type=float, default=1e-4, help="Gate regularizer weight (alpha)")
    parser.add_argument("--use_pi", action="store_true", help="Use PI controller for target deletion ratio")
    parser.add_argument("--target_deletion", type=float, default=0.5, help="Target deletion ratio (for PI)")
    parser.add_argument("--gate_warmup_steps", type=int, default=0,
                        help="Gate warm-up: first N steps use gate_regularizer_weight=0 (gate computed but no deletion pressure). 0 = disabled.")
    parser.add_argument("--phase1_steps", type=int, default=0,
                        help="Phase A: first N steps with --phase1_gate_weight (gate adaptation); 0 = disabled")
    parser.add_argument("--phase1_gate_weight", type=float, default=1e-3,
                        help="Gate regularizer weight during phase 1 (gate adaptation)")
    parser.add_argument("--output_result", type=str, default=None,
                        help="Append one JSON line with dataset, gate_weight, use_pi, target_deletion, val_acc to this file")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Use only this many training samples (for quick GPU smoke test)")
    parser.add_argument("--log_level", type=int, default=0, choices=[0, 1, 2, 3],
                        help="0=none, 1=params+loss_ce/loss_gate, 2=+PI state, 3=+dropped-token summary after val")
    parser.add_argument("--use_wandb", action="store_true", help="Log metrics to Weights & Biases (wandb)")
    parser.add_argument("--wandb_project", type=str, default="mrbert", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (default: dataset + model type)")
    args = parser.parse_args()

    # If verbose logging is requested, tee stdout/stderr to a log file.
    if args.log_level > 0:
        # Import locally to avoid any issues with shadowed names on older copies.
        from pathlib import Path as _Path
        logs_dir = _Path(__file__).resolve().parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"train_{args.dataset}_log{args.log_level}_{ts}.txt"

        class _Tee:
            def __init__(self, *streams):
                self._streams = streams

            def write(self, data):
                for s in self._streams:
                    try:
                        s.write(data)
                    except Exception:
                        pass
                for s in self._streams:
                    try:
                        s.flush()
                    except Exception:
                        pass

            def flush(self):
                for s in self._streams:
                    try:
                        s.flush()
                    except Exception:
                        pass

            def isatty(self):
                # Delegate TTY detection to the first underlying stream (typically the real stdout/stderr).
                if not self._streams:
                    return False
                s = self._streams[0]
                try:
                    return s.isatty()
                except Exception:
                    return False

        log_fh = log_path.open("a", buffering=1, encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, log_fh)
        sys.stderr = _Tee(sys.stderr, log_fh)
        print(f"[LOG] Writing detailed logs to {log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_wandb = False
    if args.use_wandb:
        try:
            import wandb
            use_wandb = True
        except ImportError:
            print("Warning: wandb not installed; run pip install wandb to enable logging.")

    print(f"Loading dataset: {args.dataset} (backbone={args.backbone})")

    train_ds, val_ds, tokenizer, num_labels, task_type = get_dataset_and_tokenizer(
        args.dataset, args.max_length, backbone=args.backbone
    )
    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        print(f"Using first {len(train_ds)} train samples (--max_train_samples)")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    is_qa = task_type == "qa"
    if args.backbone == "xlmr":
        if is_qa:
            model = MrXLMRobertaForQuestionAnswering.from_pretrained_xlm(
                "xlm-roberta-base",
                gate_layer_index=3,
                gate_k=-30.0,
            )
        else:
            model = MrXLMRobertaForSequenceClassification.from_pretrained_xlm(
                "xlm-roberta-base",
                num_labels=num_labels,
                gate_layer_index=3,
                gate_k=-30.0,
            )
    elif is_qa:
        model = MrBertForQuestionAnswering.from_bert_pretrained(
            "bert-base-uncased",
            gate_layer_index=3,
            gate_k=-30.0,
            attn_implementation="eager",
        )
    else:
        model = MrBertForSequenceClassification.from_bert_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
            gate_layer_index=3,
            gate_k=-30.0,
            attn_implementation="eager",
        )
    model = model.to(device)
    if args.log_level >= 1:
        from mrbert.diagnostics import log_parameter_summary
        log_parameter_summary(model, label="MrBERT" if (args.gate_weight != 0 or args.use_pi) else "Baseline BERT")
        encoder = getattr(model, "mrbert", None) or getattr(model, "mrxlm", None)
        if encoder is not None and hasattr(encoder, "config"):
            gl = getattr(encoder.config, "gate_layer_index", None)
            if gl is not None:
                print(f"[MrBERT] Gate placement: Layer {gl}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pi = PIController(target_ratio=args.target_deletion, kp=0.5, ki=1e-5) if args.use_pi else None
    gate_regularizer_weight = args.gate_weight

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config={
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "gate_weight": args.gate_weight,
                "use_pi": args.use_pi,
                "target_deletion": args.target_deletion,
                "max_length": args.max_length,
                "task_type": task_type,
                "max_train_samples": args.max_train_samples,
                "gate_warmup_steps": args.gate_warmup_steps,
            },
            name=args.wandb_run_name or f"{args.dataset}_{'mrbert' if (args.gate_weight != 0 or args.use_pi) else 'baseline'}",
        )
        # Define x-axis for cleaner charts: train/pruning/pi vs step, val vs step (after each epoch)
        wandb.define_metric("step")
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("pruning/*", step_metric="step")
        wandb.define_metric("pi/*", step_metric="step")
        wandb.define_metric("val/*", step_metric="step")
        wandb.define_metric("sys/*", step_metric="step")
        wandb.define_metric("epoch")

    # PI failure warnings (log_level >= 2)
    _pi_warned_alpha_zero = False
    _pi_warned_500 = False

    # Tracking: deletion rate, gate stats, time (baseline has no gate)
    gate_k, threshold_ratio = -30.0, 0.5
    threshold = gate_k * threshold_ratio
    n_gate_batches = 0
    sum_del_rate = 0.0
    sum_gate_mean = 0.0
    sum_gate_var = 0.0
    final_avg_train_loss = None

    start_time = time.time()
    total_steps = len(train_loader) * args.epochs
    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            # Gate warm-up: first N steps no gate loss (model learns on full sequence)
            if args.gate_warmup_steps > 0 and step < args.gate_warmup_steps:
                alpha = 0.0
            else:
                alpha = args.phase1_gate_weight if (args.phase1_steps > 0 and step < args.phase1_steps) else gate_regularizer_weight
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            if is_qa:
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    gate_regularizer_weight=alpha,
                )
            else:
                labels = batch["label"].to(device)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    gate_regularizer_weight=alpha,
                )
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            pi_state = None
            # Update PI only after gate warm-up (so alpha is not driven up during warm-up)
            if pi is not None and out.get("gate") is not None and (args.gate_warmup_steps <= 0 or step > args.gate_warmup_steps):
                res = pi.step(out["gate"], gate_k=gate_k, return_state=(args.log_level >= 2 or use_wandb))
                if isinstance(res, tuple):
                    gate_regularizer_weight, pi_state = res
                else:
                    gate_regularizer_weight = res
            gate = out.get("gate")
            if gate is not None:
                with torch.no_grad():
                    del_rate = (gate < threshold).float().mean().item()
                    sum_del_rate += del_rate
                    sum_gate_mean += gate.mean().item()
                    sum_gate_var += gate.var().item()
                    n_gate_batches += 1
            if step == args.gate_warmup_steps and args.gate_warmup_steps > 0:
                print(f"  Gate warm-up done at step {step}; enabling gate regularizer (alpha={gate_regularizer_weight:.2e})")
            if step == args.phase1_steps and args.phase1_steps > 0:
                print(f"  Phase A done at step {step}; switching to Phase B (gate_weight={gate_regularizer_weight:.2e})")
            if step % 100 == 0:
                if use_wandb:
                    log_dict = {"step": step, "train/loss": loss.item()}
                    if out.get("loss_ce") is not None:
                        log_dict["train/loss_ce"] = out["loss_ce"].item()
                    if out.get("loss_gate") is not None:
                        log_dict["train/loss_gate"] = out["loss_gate"].item()
                    if gate is not None:
                        del_rate_step = (gate < threshold).float().mean().item()
                        log_dict["pruning/actual_deletion_rate"] = del_rate_step
                        log_dict["pruning/gate_mean"] = gate.mean().item()
                        log_dict["pruning/gate_std"] = gate.var().sqrt().item()
                        if args.use_pi:
                            log_dict["pruning/target_deletion"] = args.target_deletion
                    if pi_state is not None:
                        log_dict["pi/error"] = pi_state.get("error", 0)
                        log_dict["pi/alpha"] = pi_state.get("alpha", gate_regularizer_weight)
                        log_dict["pi/integral_term"] = pi_state.get("i", 0)
                        log_dict["pi/proportional_term"] = pi_state.get("p", 0)
                    wandb.log(log_dict, step=step)
                msg = f"  step {step}/{total_steps} loss: {loss.item():.4f}"
                if args.log_level >= 1:
                    lce = out.get("loss_ce")
                    lg = out.get("loss_gate")
                    if lce is not None:
                        msg += f"  L_CE: {lce.item():.4f}"
                    if lg is not None:
                        msg += f"  L_G: {lg.item():.4f}"
                if args.log_level >= 2 and pi_state is not None:
                    msg += f"  | PI: err={pi_state['error']:.4f} P={pi_state['p']:.4f} I={pi_state['i']:.6f} alpha={pi_state['alpha']:.6f} del%={pi_state['current_ratio']:.2%}"
                    if pi_state["alpha"] == 0 and not _pi_warned_alpha_zero:
                        print("  [PI WARNING] alpha=0 — controller may not be driving deletion.")
                        _pi_warned_alpha_zero = True
                    if step >= 500 and not _pi_warned_500:
                        _pi_warned_500 = True
                        err_abs = abs(pi_state["current_ratio"] - args.target_deletion)
                        if err_abs > 0.15:
                            print(
                                f"  [PI WARNING] After 500 steps deletion rate still far from target: "
                                f"current={pi_state['current_ratio']:.1%} target={args.target_deletion:.1%}"
                            )
                print(msg)

        avg_train = epoch_loss / len(train_loader)
        final_avg_train_loss = avg_train
        if use_wandb:
            wandb.log({"step": step, "epoch": epoch + 1, "train/epoch_loss": avg_train, "train/avg_train_loss": avg_train}, step=step)
        print(f"Epoch {epoch + 1}/{args.epochs} avg train loss: {avg_train:.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    qa_exact_match = 0
    sum_kept_tokens, n_kept_batches = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            if is_qa:
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    gate_regularizer_weight=0.0,
                )
                pred_start_short = out["start_logits"].argmax(dim=1)
                pred_end_short = out["end_logits"].argmax(dim=1)
                keep_indices = out.get("keep_indices")
                kept_lengths = out.get("kept_lengths")
                if keep_indices is not None and kept_lengths is not None:
                    # Hard deletion: logits are over shortened sequence; map predictions to original indices
                    max_valid = (kept_lengths - 1).clamp(min=0)
                    pred_start_short = torch.minimum(pred_start_short, max_valid)
                    pred_end_short = torch.minimum(pred_end_short, max_valid)
                    pred_start = keep_indices.gather(1, pred_start_short.unsqueeze(1)).squeeze(1)
                    pred_end = keep_indices.gather(1, pred_end_short.unsqueeze(1)).squeeze(1)
                    sum_kept_tokens += kept_lengths.sum().item()
                    n_kept_batches += start_positions.size(0)
                else:
                    pred_start, pred_end = pred_start_short, pred_end_short
                qa_exact_match += ((pred_start == start_positions) & (pred_end == end_positions)).sum().item()
                total += start_positions.size(0)
            else:
                labels = batch["label"].to(device)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gate_regularizer_weight=0.0,
                )
                preds = out["logits"].argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                kept_lengths = out.get("kept_lengths")
                if kept_lengths is not None:
                    sum_kept_tokens += kept_lengths.sum().item()
                    n_kept_batches += labels.size(0)
    if is_qa:
        val_acc = qa_exact_match / total if total else 0.0
        print(f"Validation exact match: {val_acc:.4f} ({qa_exact_match}/{total})")
    else:
        val_acc = correct / total if total else 0.0
        print(f"Validation accuracy: {val_acc:.4f} ({correct}/{total})")

    encoder = getattr(model, "mrbert", None) or getattr(model, "mrxlm", None)
    if args.log_level >= 3 and encoder is not None and getattr(encoder, "delete_gate", None) is not None:
        from mrbert.diagnostics import print_dropped_token_summary
        first_batch = next(iter(val_loader))
        inp = {k: v.to(device) if hasattr(v, "to") else v for k, v in first_batch.items()}
        old_log_shapes = getattr(encoder.config, "log_shapes", False)
        encoder.config.log_shapes = True
        with torch.no_grad():
            out = model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                token_type_ids=inp.get("token_type_ids"),
                gate_regularizer_weight=0.0,
            )
        gate = out.get("gate")
        if gate is not None:
            thresh = gate_k * threshold_ratio
            print_dropped_token_summary(
                tokenizer,
                first_batch["input_ids"],
                gate.cpu(),
                thresh,
                batch_index=0,
                max_show=25,
            )
        encoder.config.log_shapes = old_log_shapes

    # Inference-time benchmark: forward-only, no backward (eval mode + no_grad)
    model.eval()
    num_inference_warmup, num_inference_batches = 5, 50
    inf_batches_done, inf_total_samples = 0, 0
    inf_timed_start = None
    with torch.no_grad():
        val_iter = iter(val_loader)
        for _ in range(num_inference_warmup + num_inference_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if is_qa:
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    gate_regularizer_weight=0.0,
                )
            else:
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gate_regularizer_weight=0.0,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            if inf_batches_done == num_inference_warmup:
                inf_timed_start = time.perf_counter()
            if inf_batches_done >= num_inference_warmup:
                inf_total_samples += input_ids.size(0)
            inf_batches_done += 1
            if inf_batches_done >= num_inference_warmup + num_inference_batches:
                break
    inf_timed_end = time.perf_counter()
    inf_elapsed_sec = (inf_timed_end - inf_timed_start) if inf_timed_start is not None else 0
    timed_batches = num_inference_batches
    inference_latency_ms = round(inf_elapsed_sec * 1000 / timed_batches, 2) if (timed_batches > 0 and inf_elapsed_sec > 0) else None
    inference_throughput = round(inf_total_samples / inf_elapsed_sec, 2) if inf_elapsed_sec > 0 else None
    if inference_latency_ms is not None:
        print(f"Inference: {inference_latency_ms} ms/batch  |  {inference_throughput} samples/sec")

    duration_sec = round(time.time() - start_time, 1)
    actual_del_rate = round(sum_del_rate / n_gate_batches, 4) if n_gate_batches else None
    gate_mean = round(sum_gate_mean / n_gate_batches, 4) if n_gate_batches else None
    gate_std = round((sum_gate_var / n_gate_batches) ** 0.5, 4) if n_gate_batches else None
    alpha_final = round(gate_regularizer_weight, 6) if (pi is not None and n_gate_batches) else None
    kept_tokens_avg = round(sum_kept_tokens / n_kept_batches, 1) if n_kept_batches else None
    total_train_samples = len(train_loader.dataset) * args.epochs
    throughput = round(total_train_samples / duration_sec, 2) if duration_sec > 0 else None
    total_batches = len(train_loader) * args.epochs
    avg_latency_ms = round(duration_sec * 1000 / total_batches, 2) if total_batches > 0 else None
    max_mem_gb = round(torch.cuda.max_memory_allocated(device) / 1e9, 4) if device.type == "cuda" else None
    if n_gate_batches:
        print(f"Actual deletion rate (train): {actual_del_rate:.2%}  |  gate mean: {gate_mean}  std: {gate_std}  |  alpha_final: {alpha_final}  |  time: {duration_sec}s")

    if use_wandb:
        wandb.log({
            "step": step,
            "val/accuracy": val_acc if not is_qa else None,
            "val/em": val_acc if is_qa else None,
            "val/exact_match": val_acc if is_qa else None,
            "pruning/actual_deletion_rate": actual_del_rate,
            "pruning/gate_mean": gate_mean,
            "pruning/gate_std": gate_std,
            "pruning/target_deletion": args.target_deletion if args.use_pi else None,
            "pruning/kept_tokens": kept_tokens_avg,
            "pi/alpha_final": alpha_final,
            "train/avg_train_loss": final_avg_train_loss,
            "duration_sec": duration_sec,
            "sys/throughput": throughput,
            "sys/latency_ms": avg_latency_ms,
            "sys/inference_latency_ms": inference_latency_ms,
            "sys/inference_throughput": inference_throughput,
            "sys/max_memory_allocated": max_mem_gb,
        }, step=step)
        wandb.finish()

    if args.output_result:
        import json
        from pathlib import Path
        Path(args.output_result).parent.mkdir(parents=True, exist_ok=True)
        row = {
            "dataset": args.dataset,
            "backbone": args.backbone,
            "gate_weight": args.gate_weight,
            "use_pi": args.use_pi,
            "target_deletion": args.target_deletion,
            "val_acc": round(val_acc, 4),
            "avg_train_loss": round(final_avg_train_loss, 4) if final_avg_train_loss is not None else None,
            "actual_deletion_rate": actual_del_rate,
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "alpha_final": alpha_final,
            "duration_sec": duration_sec,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        }
        with open(args.output_result, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"Result appended to {args.output_result}")

    print("Done.")


if __name__ == "__main__":
    main()
