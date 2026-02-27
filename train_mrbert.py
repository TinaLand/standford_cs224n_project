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

from mrbert import MrBertForSequenceClassification, MrBertForQuestionAnswering
from mrbert.pi_controller import PIController


def _prepare_tydiqa_example(ex, tokenizer, max_length):
    """Tokenize one TyDi QA example and compute start/end token positions. Drops if answer not in span."""
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
    start_token = -1
    end_token = -1
    for i, (s, e) in enumerate(offset_mapping):
        if s == 0 and e == 0:
            continue
        if token_type_ids and token_type_ids[i] == 0:
            continue
        if s <= answer_start_char < e:
            start_token = i
        if s < answer_end_char <= e:
            end_token = i
            break
    if start_token < 0 or end_token < 0 or end_token < start_token:
        return None
    inputs["start_positions"] = start_token
    inputs["end_positions"] = end_token
    return inputs


def get_dataset_and_tokenizer(dataset_name: str, max_length: int):
    """Load dataset and tokenizer. Supports 'mrpc', 'imdb', 'sst2', 'snli', 'tydiqa'.
    Returns (train_ds, val_ds, tokenizer, num_labels, task_type). task_type is 'classification' or 'qa'."""
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
    elif dataset_name == "tydiqa":
        ds = load_dataset("tydiqa", "secondary_task")
        def prepare_tydiqa_batch(examples):
            out = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "start_positions": [], "end_positions": []}
            for i in range(len(examples["id"])):
                ex = {k: v[i] for k, v in examples.items()}
                row = _prepare_tydiqa_example(ex, tokenizer, max_length)
                if row is None:
                    continue
                out["input_ids"].append(row["input_ids"])
                out["attention_mask"].append(row["attention_mask"])
                out["token_type_ids"].append(row["token_type_ids"])
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
        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"])
        val_ds.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"])
        return train_ds, val_ds, tokenizer, None, "qa"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use mrpc, imdb, sst2, snli, or tydiqa.")

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
    parser.add_argument("--dataset", type=str, default="mrpc", choices=["mrpc", "imdb", "sst2", "snli", "tydiqa"],
                        help="Dataset: mrpc, imdb, sst2, snli, or tydiqa")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Train batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--gate_weight", type=float, default=1e-4, help="Gate regularizer weight (alpha)")
    parser.add_argument("--use_pi", action="store_true", help="Use PI controller for target deletion ratio")
    parser.add_argument("--target_deletion", type=float, default=0.5, help="Target deletion ratio (for PI)")
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
    args = parser.parse_args()

    # If verbose logging is requested, tee stdout/stderr to a log file.
    if args.log_level > 0:
        logs_dir = Path(__file__).resolve().parent / "logs"
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

        log_fh = log_path.open("w", buffering=1, encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, log_fh)
        sys.stderr = _Tee(sys.stderr, log_fh)
        print(f"[LOG] Writing detailed logs to {log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading dataset: {args.dataset}")

    train_ds, val_ds, tokenizer, num_labels, task_type = get_dataset_and_tokenizer(args.dataset, args.max_length)
    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        print(f"Using first {len(train_ds)} train samples (--max_train_samples)")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    is_qa = task_type == "qa"
    if is_qa:
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
        if hasattr(model, "mrbert") and hasattr(model.mrbert, "config"):
            gl = getattr(model.mrbert.config, "gate_layer_index", None)
            if gl is not None:
                print(f"[MrBERT] Gate placement: Layer {gl}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pi = PIController(target_ratio=args.target_deletion, kp=0.5, ki=1e-5) if args.use_pi else None
    gate_regularizer_weight = args.gate_weight

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
            if pi is not None and out.get("gate") is not None:
                res = pi.step(out["gate"], gate_k=gate_k, return_state=(args.log_level >= 2))
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
            if step == args.phase1_steps and args.phase1_steps > 0:
                print(f"  Phase A done at step {step}; switching to Phase B (gate_weight={gate_regularizer_weight:.2e})")
            if step % 100 == 0:
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
        print(f"Epoch {epoch + 1}/{args.epochs} avg train loss: {avg_train:.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    qa_exact_match = 0
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
    if is_qa:
        val_acc = qa_exact_match / total if total else 0.0
        print(f"Validation exact match: {val_acc:.4f} ({qa_exact_match}/{total})")
    else:
        val_acc = correct / total if total else 0.0
        print(f"Validation accuracy: {val_acc:.4f} ({correct}/{total})")

    if args.log_level >= 3 and hasattr(model, "mrbert") and getattr(model.mrbert, "delete_gate", None) is not None:
        from mrbert.diagnostics import print_dropped_token_summary
        first_batch = next(iter(val_loader))
        inp = {k: v.to(device) if hasattr(v, "to") else v for k, v in first_batch.items()}
        old_log_shapes = getattr(model.mrbert.config, "log_shapes", False)
        model.mrbert.config.log_shapes = True
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
        model.mrbert.config.log_shapes = old_log_shapes


    duration_sec = round(time.time() - start_time, 1)
    actual_del_rate = round(sum_del_rate / n_gate_batches, 4) if n_gate_batches else None
    gate_mean = round(sum_gate_mean / n_gate_batches, 4) if n_gate_batches else None
    gate_std = round((sum_gate_var / n_gate_batches) ** 0.5, 4) if n_gate_batches else None
    alpha_final = round(gate_regularizer_weight, 6) if (pi is not None and n_gate_batches) else None
    if n_gate_batches:
        print(f"Actual deletion rate (train): {actual_del_rate:.2%}  |  gate mean: {gate_mean}  std: {gate_std}  |  alpha_final: {alpha_final}  |  time: {duration_sec}s")

    if args.output_result:
        import json
        from pathlib import Path
        Path(args.output_result).parent.mkdir(parents=True, exist_ok=True)
        row = {
            "dataset": args.dataset,
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
