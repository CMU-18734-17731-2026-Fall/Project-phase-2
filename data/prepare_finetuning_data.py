#!/usr/bin/env python3
"""Draw a deterministic GPT-2 fine-tuning set from WikiText-103."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--min-tokens", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="gpt2")
    return parser.parse_args()


def main():
    args = parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train"
    )

    dataset = dataset.filter(
        lambda example: isinstance(example.get("text"), str)
        and bool(example["text"].strip())
    )

    def normalize(example):
        return {"text": " ".join(example["text"].split())}

    dataset = dataset.map(normalize)

    def add_lengths(batch):
        tokenized = tokenizer(batch["text"], add_special_tokens=False)
        return {"_token_count": [len(ids) for ids in tokenized["input_ids"]]}

    dataset = dataset.map(add_lengths, batched=True)
    dataset = dataset.filter(
        lambda example: example["_token_count"] >= args.min_tokens
    )
    dataset = dataset.remove_columns(
        [name for name in dataset.column_names if name != "text"]
    )

    if len(dataset) < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} samples, but only {len(dataset)} remain"
        )

    selected = dataset.shuffle(seed=args.seed).select(range(args.num_samples))
    records = [{"text": text} for text in selected["text"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "train_finetune.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} examples to {output_path}")
    print(f"Sampling seed: {args.seed}")


if __name__ == "__main__":
    main()

