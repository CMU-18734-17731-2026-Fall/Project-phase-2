#!/usr/bin/env python3
"""Offline integrity and leakage checks for the public student release."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "gpt2_3_lora32_adamw_b8_lr2"

EXPECTED_HASHES = {
    "data/train/train_finetune.json": "cac5d355a93434130b31c158c9dd7f6010086301627beee19baada3041c46163",
    "data/train/test.json": "47803134aa86a76b7e37394a641e1be78915a2afd53052627c0af28d5dd9dcb9",
    "data/train/test_label.json": "fb1fbca08b43ed44f6a72db93167e7cb1fe8bf0c8978889be7935bc1cf74d43b",
    "data/val/test.json": "fcb8f73e9aad9f27becd111a20a0ede631f41b0b23b976dfbcc5da242d1fd693",
    "data/final/test.json": "d8e229446ae8533575ebeedd92af73c6357ef75778565e74c3cd0371468f8880",
    f"models/train/{MODEL_NAME}/adapter_model.safetensors": "4a6bdef73680fa37957fc9331f06dc9fbe545c48bb0201c48f567f7244ceb68f",
    f"models/val/{MODEL_NAME}/adapter_model.safetensors": "f43f07378edca179c3c283913be47f2c0e1c686d7ec66d5f8b935c91a93dbd94",
    f"models/final/{MODEL_NAME}/adapter_model.safetensors": "24b314d025c303ef60c0661e76b0dbeb7ab9e44cad7fbd193d86e2bf6488486f",
}

FORBIDDEN_PATHS = (
    "data/val/test_label.json",
    "data/final/test_label.json",
    "competition.yaml",
)

FORBIDDEN_TEXT = (
    "2020" + "pyfcrawl",
    "competitions/" + "11238",
    "2025" + " fall",
    "/" + "users" + "/",
)


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(relative_path):
    path = ROOT / relative_path
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {relative_path}: {error}")


def check_hashes():
    for relative_path, expected in EXPECTED_HASHES.items():
        path = ROOT / relative_path
        if not path.is_file():
            fail(f"missing required file: {relative_path}")
        if path.suffix == ".safetensors" and path.stat().st_size < 1_000_000:
            fail(f"{relative_path} appears to be a Git LFS pointer; run git lfs pull")
        actual = sha256(path)
        if actual != expected:
            fail(f"hash mismatch for {relative_path}: {actual}")
    print(f"Verified {len(EXPECTED_HASHES)} release hashes")


def check_data():
    finetune = read_json("data/train/train_finetune.json")
    if len(finetune) != 10_000:
        fail(f"train_finetune.json has {len(finetune)} examples, expected 10000")
    if not all(
        isinstance(record, dict)
        and isinstance(record.get("text"), str)
        and bool(record["text"].strip())
        for record in finetune
    ):
        fail("train_finetune.json must contain non-empty {'text': str} records")

    for phase in ("train", "val", "final"):
        examples = read_json(f"data/{phase}/test.json")
        if len(examples) != 2_000:
            fail(f"data/{phase}/test.json has {len(examples)} examples, expected 2000")
        if not all(isinstance(text, str) and text.strip() for text in examples):
            fail(f"data/{phase}/test.json contains an invalid text")

    labels = read_json("data/train/test_label.json")
    if len(labels) != 2_000 or Counter(labels) != Counter({0: 1000, 1: 1000}):
        fail("train labels must contain exactly 1000 zeros and 1000 ones")
    print("Verified data schemas, counts, and public train labels")


def check_models():
    required_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    for phase in ("train", "val", "final"):
        model_dir = ROOT / "models" / phase / MODEL_NAME
        for filename in required_files:
            if not (model_dir / filename).is_file():
                fail(f"missing model file: models/{phase}/{MODEL_NAME}/{filename}")
        config = read_json(f"models/{phase}/{MODEL_NAME}/adapter_config.json")
        if config.get("base_model_name_or_path") != "gpt2":
            fail(f"unexpected base model in {phase} adapter")
        if config.get("r") != 32 or config.get("lora_alpha") != 64.0:
            fail(f"unexpected LoRA configuration in {phase} adapter")
        if set(config.get("target_modules", [])) != {"c_attn", "c_fc", "c_proj"}:
            fail(f"unexpected LoRA target modules in {phase} adapter")
    print("Verified train, val, and final adapter configurations")


def check_no_hidden_material():
    for relative_path in FORBIDDEN_PATHS:
        if (ROOT / relative_path).exists():
            fail(f"organizer-only file is present: {relative_path}")
    for pattern in ("reference_data-*.zip", "scoring_program-*.zip"):
        matches = list(ROOT.rglob(pattern))
        if matches:
            fail(f"organizer-only archive is present: {matches[0].relative_to(ROOT)}")
    print("Verified that hidden labels and organizer archives are absent")


def check_notebook():
    notebook = read_json("MIA_phase2.ipynb")
    if notebook.get("nbformat") != 4:
        fail("MIA_phase2.ipynb is not notebook format 4")
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            if cell.get("execution_count") is not None:
                fail(f"notebook cell {index} has an execution count")
            if cell.get("outputs"):
                fail(f"notebook cell {index} contains saved output")
    print("Verified clean notebook metadata and outputs")


def check_text_traces():
    suffixes = {".md", ".py", ".txt", ".ipynb", ".sh"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if ".git" in path.parts or "models" in path.parts or "data" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for forbidden in FORBIDDEN_TEXT:
            if forbidden in text:
                fail(f"found forbidden release trace {forbidden!r} in {path.relative_to(ROOT)}")
    print("Verified that stale links and personal paths are absent")


def main():
    check_hashes()
    check_data()
    check_models()
    check_no_hidden_material()
    check_notebook()
    check_text_traces()
    print("Phase 2 release verification: PASS")


if __name__ == "__main__":
    main()
