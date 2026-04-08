"""
Scripta AI Text Detector - Local Training Script
=================================================
Requirements: pip install transformers datasets torch scikit-learn
Dataset: ai-text-detection-pile from HuggingFace (auto-downloaded)

Run: python train_detector.py
Output: scripta_model/ folder (your trained model)
"""

import os
import random
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ─────────────────────────────────────────────
# 1. CONFIG - change these if you want
# ─────────────────────────────────────────────
MODEL_NAME    = "distilbert-base-uncased"  # small, fast, accurate
OUTPUT_DIR    = "./scripta_model"          # where your trained model saves
NUM_EPOCHS    = 3                          # 3 is enough; raise to 5 for better accuracy
BATCH_SIZE    = 16                         # lower to 8 if you get memory errors
MAX_LENGTH    = 512                        # max tokens per text sample
LEARNING_RATE = 2e-5
MAX_SAMPLES   = 50_000                     # cap for reasonable CPU training time

print("=" * 55)
print("  Scripta - AI Detector Training Pipeline")
print("=" * 55)

# ─────────────────────────────────────────────
# 2. LOAD DATASET
#    'artem9k/ai-text-detection-pile'
#    Modern Parquet format - no script errors
#    Columns: text (str), label (0=human, 1=AI)
# ─────────────────────────────────────────────
print("\n[1/5] Loading dataset from HuggingFace...")
raw = load_dataset("artem9k/ai-text-detection-pile", split="train")
print(f"   Total rows in dataset: {len(raw):,}")

# Subsample to MAX_SAMPLES for manageable CPU training time
random.seed(42)
indices = random.sample(range(len(raw)), min(MAX_SAMPLES, len(raw)))
flat = raw.select(indices)

# Split into train / validation
split = flat.train_test_split(test_size=0.15, seed=42)
train_ds = split["train"]
val_ds   = split["test"]

print(f"   Train samples : {len(train_ds):,}")
print(f"   Val samples   : {len(val_ds):,}")
print(f"   Columns: {train_ds.column_names}")

# This dataset uses 'source' as the label column.
# Values like 'human', 'wiki', 'reddit' = 0 (human)
# Values like 'gpt', 'ai', 'chatgpt', 'davinci' etc = 1 (AI)
AI_SOURCES = {"gpt", "gpt2", "gpt3", "gpt4", "chatgpt", "ai", "davinci",
              "curie", "babbage", "ada", "claude", "llama", "generated",
              "grover", "xlm", "xlnet", "fair", "fast"}

def cast_source_label(example):
    src = str(example.get("source", "")).lower().strip()
    # If any AI keyword appears in the source string → label 1
    example["label"] = 1 if any(k in src for k in AI_SOURCES) else 0
    return example

print("   Mapping 'source' column to binary labels (0=human, 1=AI)...")
train_ds = train_ds.map(cast_source_label)
val_ds   = val_ds.map(cast_source_label)

# Show unique sources so user can verify mapping
unique_sources = list(set(train_ds["source"]))[:20]
print(f"   Unique sources (sample): {unique_sources}")

human_count = sum(1 for x in train_ds["label"] if x == 0)
ai_count    = sum(1 for x in train_ds["label"] if x == 1)
print(f"   Label split   : {human_count:,} human · {ai_count:,} AI")

# ─────────────────────────────────────────────
# 3. TOKENIZE
# ─────────────────────────────────────────────
print(f"\n[2/5] Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,      # DataCollator handles padding
    )

print("   Tokenizing train set...")
train_ds = train_ds.map(tokenize, batched=True, desc="Tokenizing train")
print("   Tokenizing val set...")
val_ds   = val_ds.map(tokenize, batched=True, desc="Tokenizing val")

# Drop all non-model columns
cols_to_remove = [c for c in train_ds.column_names if c not in ("input_ids", "attention_mask", "label")]
train_ds = train_ds.remove_columns(cols_to_remove)
val_ds   = val_ds.remove_columns([c for c in val_ds.column_names if c not in ("input_ids", "attention_mask", "label")])
train_ds.set_format("torch")
val_ds.set_format("torch")

# ─────────────────────────────────────────────
# 4. LOAD MODEL
# ─────────────────────────────────────────────
print(f"\n[3/5] Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "HUMAN", 1: "AI"},
    label2id={"HUMAN": 0, "AI": 1},
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device.upper()}")
if device == "cpu":
    print("   ⚠  No GPU found - training on CPU (~30–60 min)")
    print("      Tip: install CUDA or use Google Colab for GPU speed")
else:
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",         # renamed from evaluation_strategy in newer Transformers
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    fp16=(device == "cuda"),       # faster on GPU
    report_to="none",              # no wandb / external logging
    dataloader_num_workers=0,      # safe default for Windows
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,    # renamed from tokenizer in newer Transformers
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(f"\n[4/5] Training for {NUM_EPOCHS} epoch(s)...")
print("      This will take ~30–60 min on CPU, ~5 min on GPU\n")
trainer.train()

# ─────────────────────────────────────────────
# 6. SAVE + EVALUATE
# ─────────────────────────────────────────────
print(f"\n[5/5] Saving model to ./{OUTPUT_DIR}/")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nFinal evaluation on validation set:")
preds_out = trainer.predict(val_ds)
preds     = np.argmax(preds_out.predictions, axis=-1)
labels    = preds_out.label_ids
print(classification_report(labels, preds, target_names=["Human", "AI"]))

acc = accuracy_score(labels, preds)
print(f"\n✅ Done! Accuracy: {acc*100:.1f}%")
print(f"   Model saved to: ./{OUTPUT_DIR}/")
print("\nNext step → run: python app.py")
print("Then open: http://localhost:5000")