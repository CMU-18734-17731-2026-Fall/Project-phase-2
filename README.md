# Membership Inference Attack Competition: Phases 2 and 3

Starter kit for the CMU 18-734/17-731 course project. In Phase 2, teams develop
a white-box membership inference attack against LoRA-fine-tuned GPT-2 target
models. The main challenge is to design a method that transfers from the
labeled development target to targets whose membership labels are hidden.

## Phase 3

Use this same starter kit for the final evaluation. See the
[Phase 3 instructions](PHASE3.md) for the final submission window, report and
slides, and the grading rubric.

## What Is Included

- `MIA_phase2.ipynb`: model loading, evaluation, prediction export, and
  submission packaging.
- `data/train/`: the development target's fine-tuning data, 2,000 test texts,
  and 2,000 public membership labels.
- `data/val/`: 2,000 validation texts. Membership labels are held by the
  competition server.
- `data/final/`: 2,000 final texts for the later project phase. Membership
  labels are held by the competition server.
- `models/{train,val,final}/`: separate GPT-2 LoRA adapters trained with the
  same configuration on different fine-tuning sets.
- `ft_llm/train_shadow_model.py`: a configurable LoRA shadow-model trainer.
- `data/prepare_finetuning_data.py`: a utility for drawing additional
  WikiText-103 fine-tuning data with a chosen seed.
- `scripts/verify_release.py`: an offline integrity and label-leakage check.
- `scripts/make_submission.py` and `scripts/validate_submission.py`: submission
  packaging and validation utilities.

The starter notebook intentionally leaves `your_attack` unfinished. Designing
and implementing that function is the central student task.

## Compute Requirements

- Python 3.11
- Git LFS
- An NVIDIA CUDA GPU or a Google Colab GPU runtime for shadow-model training
- At least 12 GB of system memory recommended for local work

Target-model inference and small debugging runs can use CPU, but training
multiple shadow models on CPU is not a practical primary workflow. A Colab T4
or a CUDA GPU is the supported path for full Phase 2 experiments.

## Local Setup

Clone the repository with Git LFS enabled:

```bash
git lfs install
git clone https://github.com/CMU-18734-17731-2026-Fall/Project-phase-2.git
cd Project-phase-2
git lfs pull
```

Create the supported environment and verify the release:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead.

The expected final verification line is:

```text
Phase 2 release verification: PASS
```

The first model load downloads the public GPT-2 base model from Hugging Face.
The course-specific LoRA adapters are included in this repository.

## Google Colab

Select a GPU runtime, then run these setup cells in a blank Colab notebook:

```python
!git lfs install
!git clone https://github.com/CMU-18734-17731-2026-Fall/Project-phase-2.git
%cd Project-phase-2
!git lfs pull
%pip install -r requirements.txt
!python scripts/verify_release.py
```

Restart the runtime after installation if Colab requests it. Upload or open
`MIA_phase2.ipynb` only after setup completes. The notebook does not clone or
delete repositories.

## Project Workflow

1. Develop and evaluate an attack with the `train` target and its public labels.
2. Use the provided fine-tuning distribution to design shadow-model datasets.
3. Train one or more shadow models and use them to calibrate membership scores.
4. Set `PHASE = "val"` in the notebook and produce one score per validation
   example.
5. Validate and upload the resulting ZIP to the
   [Phase 2 Codabench competition](https://www.codabench.org/competitions/18268/).
6. Submit your group's PDF report on Gradescope as described below.

All three target models use GPT-2 with LoRA rank 32, alpha 64, dropout 0.05,
three epochs, block size 512, batch size 8, and learning rate `2e-4`. Their
fine-tuning data differ.

### Shadow-Model Data

The data utility can draw a new fine-tuning set from WikiText-103:

```bash
python data/prepare_finetuning_data.py \
  --seed 1001 \
  --num-samples 10000 \
  --output-dir data/shadow_1001
```

It writes `train_finetune.json`. Choosing member/non-member evaluation examples,
preventing unintended overlap, and combining results across shadow models are
part of the attack design.

Train a LoRA shadow model with the target configuration:

```bash
python ft_llm/train_shadow_model.py \
  --data-dir data/shadow_1001 \
  --outdir models/shadow_1001 \
  --epochs 3 \
  --batch-size 8 \
  --block-size 512 \
  --lr 2e-4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05
```

The trainer automatically uses FP16 on GPUs such as the Colab T4, BF16 on
supported GPUs, and FP32 elsewhere. Reduce `--batch-size` and increase
`--gradient-accumulation-steps` if GPU memory is limited.

## Evaluation

The leaderboard metric is **TPR at FPR strictly below 0.01**:

```python
max(tpr[fpr < 0.01])
```

There are 1,000 non-members in each test set, so the strict boundary admits at
most nine false positives. Attack scores must follow this convention:

- larger score: more likely to be a member;
- smaller score: more likely to be a non-member.

The notebook reports ROC AUC and the competition metric for the public `train`
labels. Validation and final labels are never available in this repository.

## Phase 2 Submission

**Deadline for both submissions: Friday, October 23, 2026, at 11:59 PM
Eastern Time (EDT, UTC-4).**

Each group must submit validation predictions to Codabench and one PDF report
to Gradescope.

### 1. Validation Predictions on Codabench

Upload a ZIP to the `val` phase of the
[Phase 2 competition](https://www.codabench.org/competitions/18268/) containing
exactly this logical path:

```text
val/
└── prediction.csv
```

`prediction.csv` must contain exactly 2,000 finite numeric scores, one per line,
in the same order as `data/val/test.json`. Do not add a header or index column.

The notebook writes the prediction file and creates the ZIP. It can also be
packaged and checked from the command line:

```bash
python scripts/make_submission.py --phase val
python scripts/validate_submission.py project_phase2_val_submission.zip --phase val
```

Submit through the Codabench organization used by your team. No `final`
predictions are required for Phase 2.

### 2. PDF Report on Gradescope

Submit one PDF per group to
[Project Phase 2: Membership Inference Attack on Gradescope](https://www.gradescope.com/courses/1411687/assignments/8893015).
Include all group members on the Gradescope submission and identify your team
and members in the report. Assign the relevant report pages to the three
questions when submitting.

The PDF must contain:

1. **Methodology (maximum 3 pages):** explain your attack, include a pipeline
   diagram, and describe the experimental settings needed to reproduce it.
2. **Validation results:** include a screenshot of your Codabench validation
   score and the corresponding submission ID.
3. **Core attack code:** include the code implementing your attack in an
   appendix. The code must correspond to the reported submission.

The 3-page limit applies to the methodology section; the results and code
appendix are separate from that limit.

The report is worth **20 points**: methodology 12, validation results 6, and
core attack code 2.

Phase 2 grading uses the Codabench submission identified in your report, which
must be submitted before the deadline above. The leaderboard may remain open
for the later project phase; scores submitted after the Phase 2 deadline do
not change your Phase 2 grade.

## Reproducibility and Rules

- Do not modify, reorder, or remove the provided test examples.
- Do not use the public `train` labels as an input feature to the attack.
- Keep random seeds and experimental settings in the report.
- Use a small subset while debugging, then generate predictions for all 2,000
  examples.
- Notebook outputs, local environments, generated models, and prediction files
  should not be committed.

## Troubleshooting

**An adapter file is only about 130 bytes.** It is a Git LFS pointer. Run
`git lfs pull`, followed by `python scripts/verify_release.py`.

**The base model cannot be loaded.** Confirm that the runtime has internet
access for the first GPT-2 download and that `transformers` and `peft` were
installed from `requirements.txt`.

**CUDA runs out of memory.** Lower the training or inference batch size. For
training, raise `--gradient-accumulation-steps` to retain a similar effective
batch size.

**The submission validator reports the wrong row count.** Regenerate scores for
the complete validation set and make sure the CSV has no header or blank rows.

## Attribution

The base model is GPT-2, and the fine-tuning and test texts derive from
WikiText-103. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for citations
and upstream notices.
