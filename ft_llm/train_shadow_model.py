#!/usr/bin/env python3
"""Train a GPT-2 LoRA shadow model with the target-model configuration."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-file", default="train_finetune.json")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules", default="c_attn,c_fc,c_proj",
        help="Comma-separated LoRA target module names",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--merge-lora", action="store_true")
    return parser.parse_args()


def read_training_texts(path):
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("Training data must be a non-empty JSON list")

    texts = []
    for index, record in enumerate(records):
        text = record if isinstance(record, str) else record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid text at training index {index}")
        texts.append(text)
    return texts


def choose_precision(requested, torch):
    if requested == "auto":
        if not torch.cuda.is_available():
            return "fp32"
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    if requested in {"fp16", "bf16"} and not torch.cuda.is_available():
        raise ValueError(f"{requested} training requires a CUDA GPU")
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support BF16 training")
    return requested


def main():
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    train_path = args.data_dir / args.train_file
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.outdir}")

    texts = read_training_texts(train_path)
    set_seed(args.seed)
    precision = choose_precision(args.precision, torch)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.block_size,
            return_attention_mask=True,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    targets = [name.strip() for name in args.target_modules.split(",") if name.strip()]
    if not targets:
        raise ValueError("At least one LoRA target module is required")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.config.use_cache = False

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training examples: {len(tokenized)}")
    print(f"Device: {device_name}")
    print(f"Precision: {precision}")

    training_args = TrainingArguments(
        output_dir=str(args.outdir),
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=100,
        save_strategy="no",
        seed=args.seed,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.merge_lora:
        if not isinstance(model, PeftModel):
            raise TypeError("Expected a PEFT model before merging")
        model = model.merge_and_unload()
    model.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print(f"Saved model to {args.outdir}")


if __name__ == "__main__":
    main()

