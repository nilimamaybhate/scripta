"""
Scripta AI Text Detector - v4 final
"""
import os, random, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay)
import torch.nn.functional as F
import transformers

# ─────────────────────────────────────────────
# DEBUG SWITCH  ← flip to False for full run
# ─────────────────────────────────────────────
DEBUG = False

MODEL_NAME      = "roberta-base"
OUTPUT_DIR      = "./scripta_model_v4"
LABEL_SMOOTHING = 0.05
SEED            = 42

if DEBUG:
    SAMPLES_PER_SRC = 300
    NUM_EPOCHS      = 1
    MAX_LENGTH      = 128
    BATCH_SIZE      = 8
else:
    SAMPLES_PER_SRC = 5_000
    NUM_EPOCHS      = 4
    MAX_LENGTH      = 512
    BATCH_SIZE      = 16

LEARNING_RATE = 2e-5
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("="*60)
print(f"  Scripta v4  ({'DEBUG' if DEBUG else 'FULL'} mode)")
print("="*60)

# ─────────────────────────────────────────────
# LOAD YOUR SCRIPTA DATASET FROM HUB
# ─────────────────────────────────────────────
print("\n[0/6] Loading your Scripta dataset from HuggingFace Hub...")
scripta_ds = load_dataset("nilima1704/scripta-dataset", split="train")
scripta_samples = [
    {"text": row["text"], "label": row["label"]}
    for row in scripta_ds
    if row.get("text") and len(row["text"].strip()) >= 80
]
print(f"   Loaded {len(scripta_samples):,} samples from nilima1704/scripta-dataset")

# ─────────────────────────────────────────────
# 1. BUILD DATASET
# ─────────────────────────────────────────────
def collect(iterable, n, text_fn, label):
    out = []
    for item in iterable:
        if len(out) >= n: break
        try:
            t = text_fn(item)
        except Exception:
            continue
        if t and len(t.strip()) >= 80:
            out.append({"text": t.strip()[:2000], "label": label})
    return out

print("\n[1/6] Building multi-source dataset...")
all_samples = list(scripta_samples)   # start with your own dataset

SOURCES = [
    # (display_name, dataset_id, config, split, text_fn, label, trust_remote_code)

    # ── HUMAN ──────────────────────────────────────────────────
    ("HC3 human",
     "Hello-SimpleAI/HC3", "all", "train",
     lambda x: x["human_answers"][0]
               if x.get("human_answers") and len(x["human_answers"][0]) > 100
               else None,
     0, True),

    ("OpenAssist human",
     "OpenAssistant/oasst1", None, "train",
     lambda x: x["text"]
               if x.get("role") == "prompter" and x.get("lang") == "en"
               else None,
     0, False),

    # ── AI ─────────────────────────────────────────────────────
    ("HC3 ChatGPT",
     "Hello-SimpleAI/HC3", "all", "train",
     lambda x: x["chatgpt_answers"][0]
               if x.get("chatgpt_answers") and len(x["chatgpt_answers"][0]) > 100
               else None,
     1, True),

    ("Alpaca GPT-4",
     "tatsu-lab/alpaca", None, "train",
     lambda x: x.get("output") if x.get("output") and len(x["output"]) > 150 else None,
     1, False),

    ("OpenAssist AI",
     "OpenAssistant/oasst1", None, "train",
     lambda x: x["text"]
               if x.get("role") == "assistant" and x.get("lang") == "en"
               else None,
     1, False),

    ("Dolly AI",
     "databricks/databricks-dolly-15k", None, "train",
     lambda x: x.get("response"),
     1, False),
]

for name, ds_id, cfg, split, fn, lbl, trust in SOURCES:
    try:
        kw = {"split": split, "streaming": True, "trust_remote_code": trust}
        if cfg:
            kw["name"] = cfg
        ds = load_dataset(ds_id, **kw)
        s  = collect(ds, SAMPLES_PER_SRC, fn, lbl)
        all_samples += s
        print(f"   {name:<22}: {len(s):,}")
    except Exception as e:
        print(f"   {name:<22}: SKIPPED ({e})")

# Check human coverage
human_n = sum(1 for s in all_samples if s["label"] == 0)
print(f"   Human samples loaded : {human_n:,}")
if human_n < 300:
    raise RuntimeError(
        f"Only {human_n} human samples — check trust_remote_code=True and HF token."
    )

# ── Deduplicate ────────────────────────────────────────────
before = len(all_samples)
seen, deduped = set(), []
for s in all_samples:
    key = s["text"][:200]
    if key not in seen:
        seen.add(key)
        deduped.append(s)
all_samples = deduped
print(f"\n   Dedup: {before:,} → {len(all_samples):,}")

# ── Balance ────────────────────────────────────────────────
human_s = [s for s in all_samples if s["label"] == 0]
ai_s    = [s for s in all_samples if s["label"] == 1]
random.shuffle(human_s); random.shuffle(ai_s)
n = min(len(human_s), len(ai_s))
all_samples = human_s[:n] + ai_s[:n]
random.shuffle(all_samples)
print(f"   Balanced: {len(all_samples):,}  (human:{n:,}  AI:{n:,})")

if len(all_samples) < 500:
    raise RuntimeError("Too few samples — check HF token / network.")

# ── Data inspection ────────────────────────────────────────
print("\n🔍 DATA INSPECTION")
for i in range(2):
    print(f"\n--- HUMAN #{i+1} ---"); print(human_s[i]["text"][:300])
    print(f"\n--- AI #{i+1} ---");    print(ai_s[i]["text"][:300])

# ─────────────────────────────────────────────
# SPLIT  70 / 15 / 15
# ─────────────────────────────────────────────
texts  = [s["text"]  for s in all_samples]
labels = [s["label"] for s in all_samples]
full_ds = Dataset.from_dict({"text": texts, "label": labels})
s1 = full_ds.train_test_split(test_size=0.30, seed=SEED)
s2 = s1["test"].train_test_split(test_size=0.50, seed=SEED)
train_ds, val_ds, calib_ds = s1["train"], s2["train"], s2["test"]
print(f"\n   Train:{len(train_ds):,}  Val:{len(val_ds):,}  Calib:{len(calib_ds):,}")

# ─────────────────────────────────────────────
# 2. TOKENIZE
# ─────────────────────────────────────────────
print(f"\n[2/6] Tokenizing (max_length={MAX_LENGTH})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    pad = "max_length" if DEBUG else False
    return tokenizer(batch["text"], truncation=True,
                     max_length=MAX_LENGTH, padding=pad)

KEEP = {"input_ids", "attention_mask", "label"}
for name, ds in [("train", train_ds), ("val", val_ds), ("calib", calib_ds)]:
    ds = ds.map(tokenize, batched=True, desc=f"Tokenizing {name}")
    ds = ds.remove_columns([c for c in ds.column_names if c not in KEEP])
    ds.set_format("torch")
    if name == "train":   train_ds = ds
    elif name == "val":   val_ds   = ds
    else:                 calib_ds = ds

# ─────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────
print(f"\n[3/6] Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
    id2label={0: "HUMAN", 1: "AI"}, label2id={"HUMAN": 0, "AI": 1})
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   {device.upper()}" +
      (f" - {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))

class SmoothedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        lbls   = inputs.pop("labels")
        logits = model(**inputs).logits
        log_p  = F.log_softmax(logits, dim=-1)
        nc     = logits.size(-1)
        sm     = LABEL_SMOOTHING / (nc - 1)
        tgt    = torch.full_like(log_p, sm)
        tgt.scatter_(1, lbls.unsqueeze(1), 1.0 - LABEL_SMOOTHING)
        loss   = -(tgt * log_p).sum(dim=-1).mean()
        return (loss, model(**inputs)) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, lbls = eval_pred
    preds  = np.argmax(logits, axis=-1)
    probs  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    confs  = probs.max(axis=-1)
    ece = sum(
        abs((preds[(confs>=lo)&(confs<lo+.1)] == lbls[(confs>=lo)&(confs<lo+.1)]).mean()
            - confs[(confs>=lo)&(confs<lo+.1)].mean())
        * ((confs>=lo)&(confs<lo+.1)).mean()
        for lo in np.arange(0, 1, .1)
        if ((confs>=lo)&(confs<lo+.1)).sum() > 0
    )
    return {
        "accuracy": accuracy_score(lbls, preds),
        "f1":       f1_score(lbls, preds, average="weighted"),
        "ece":      round(float(ece), 4),
    }

# ─────────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────────
t_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=max(1, len(train_ds) // BATCH_SIZE),
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50 if DEBUG else 100,
    fp16=(device == "cuda"),
    report_to="none",
    dataloader_num_workers=0,
    seed=SEED,
)

_ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
kw = dict(model=model, args=t_args, train_dataset=train_ds, eval_dataset=val_ds,
          data_collator=DataCollatorWithPadding(tokenizer),
          compute_metrics=compute_metrics)
kw["processing_class" if _ver >= (4, 38) else "tokenizer"] = tokenizer
trainer = SmoothedTrainer(**kw)

if DEBUG:
    print(f"\n⚡ DEBUG: sanity check on 50 samples...")
    trainer.train_dataset = train_ds.select(range(min(50, len(train_ds))))
    trainer.eval_dataset  = val_ds.select(range(min(50, len(val_ds))))
    trainer.train()
    print(f"\n   {trainer.evaluate()}")
    print("\n✅ DEBUG passed — set DEBUG=False and rerun.")
    exit()

print(f"\n[4/6] Training {NUM_EPOCHS} epochs on {len(train_ds):,} samples...")
trainer.train()

# ─────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────
print(f"\n[5/6] Saving to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ─────────────────────────────────────────────
# 6. CALIBRATE + EVALUATE
# ─────────────────────────────────────────────
print("\n[6/6] Calibrating threshold...")
calib_out   = trainer.predict(calib_ds)
calib_probs = torch.softmax(torch.tensor(calib_out.predictions), dim=-1).numpy()[:, 1]
calib_true  = calib_out.label_ids

best_t, best_f1 = 0.50, 0.0
for t in np.arange(0.30, 0.80, 0.01):
    f = f1_score(calib_true, (calib_probs >= t).astype(int), average="weighted")
    if f > best_f1:
        best_f1, best_t = f, t

print(f"   Best threshold: {best_t:.2f}  (F1={best_f1:.4f})")
with open(os.path.join(OUTPUT_DIR, "threshold.txt"), "w") as fh:
    fh.write(str(best_t))

val_out   = trainer.predict(val_ds)
val_probs = torch.softmax(torch.tensor(val_out.predictions), dim=-1).numpy()
val_true  = val_out.label_ids
val_confs = val_probs.max(axis=-1)
val_preds = (val_probs[:, 1] >= best_t).astype(int)

print("\nSample predictions (first 15):")
for i in range(min(15, len(val_preds))):
    flag = "✓" if val_preds[i] == val_true[i] else "✗"
    print(f"  {flag} Pred:{'AI' if val_preds[i] else 'HUMAN':<6} "
          f"True:{'AI' if val_true[i] else 'HUMAN':<6} "
          f"Conf:{val_confs[i]:.3f}")

print("\n" + classification_report(val_true, val_preds, target_names=["Human", "AI"]))
print(f"✅ Validation accuracy: {accuracy_score(val_true, val_preds)*100:.2f}%")

print("\nConfidence-bucket accuracy:")
for lo in np.arange(0.5, 1.0, 0.05):
    mask = (val_confs >= lo) & (val_confs < lo + .05)
    if mask.sum() == 0: continue
    print(f"  [{lo:.2f}-{lo+.05:.2f})  n={mask.sum():>5}  "
          f"acc={(val_preds[mask]==val_true[mask]).mean():.3f}  "
          f"conf={val_confs[mask].mean():.3f}")

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(val_true, val_preds),
                       display_labels=["Human", "AI"]).plot(ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix (thresh={best_t:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
print(f"\n   Saved confusion matrix → {OUTPUT_DIR}/confusion_matrix.png")

# ─────────────────────────────────────────────
# OOD INFERENCE TEST
# ─────────────────────────────────────────────
def predict(text):
    inp = tokenizer(text, return_tensors="pt",
                    truncation=True, max_length=MAX_LENGTH)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    with torch.no_grad():
        p = torch.softmax(model(**inp).logits, dim=-1)[0]
    ai_p = p[1].item(); conf = p.max().item()
    return ("UNCERTAIN" if conf < best_t + .10
            else ("AI" if ai_p >= best_t else "HUMAN")), conf

print("\n── OOD Inference Test ──")
tests = [
    ("AI",    "Artificial intelligence is rapidly transforming industries by automating "
              "complex tasks and enhancing decision-making."),
    ("HUMAN", "Honestly I just woke up and have no idea what I'm doing today lol"),
    ("HUMAN", "The results suggest a statistically significant correlation (p < 0.01) "
              "between sleep deprivation and cognitive decline."),
    ("HUMAN", "I think the whole AI hype is overblown tbh. my phone autocomplete does half this stuff"),
    ("AI",    "Hey there! I totally get where you're coming from. Here are some tips "
              "that might help you navigate this situation effectively."),
    ("HUMAN", "so i was reading about quantum computing and it broke my brain a little?? "
              "the idea that qubits can be 0 and 1 at the same time is wild"),
    ("AI",    "In conclusion, machine learning integration offers significant opportunities "
              "for improving system reliability, scalability, and performance metrics."),
]

correct = 0
print(f"  {'Exp':<7} {'Pred':<12} {'Conf':>6}  Text")
print("  " + "─"*65)
for exp, text in tests:
    lbl, conf = predict(text)
    flag = "✓" if lbl == exp else "✗"
    if flag == "✓": correct += 1
    print(f"  {flag} {exp:<7} {lbl:<12} {conf:.3f}  {text[:55]}...")

print(f"\n  OOD accuracy: {correct}/{len(tests)}")
print(f"\n{'='*60}\n  Done! → {OUTPUT_DIR}\n{'='*60}")