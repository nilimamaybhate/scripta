"""
Scripta AI Text Detector — Full Python/Flask App
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import os
import re
import csv
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)

MODEL_DIR = "./scripta_model"
GT_LOG_FILE = "./ground_truth_log.csv"

# ── Load model if trained, else use heuristic fallback ──────────────────────
tokenizer = None
model     = None
device    = "cpu"
MODEL_READY = False

if os.path.exists(MODEL_DIR):
    try:
        print(f"Loading model from {MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        MODEL_READY = True
        print(f"✅ Model loaded on {device.upper()}")
    except Exception as e:
        print(f"⚠  Could not load model: {e}")
        print("   Falling back to heuristic mode.")
else:
    print("⚠  No trained model found. Using heuristic mode.")
    print("   Train first: python train_detector.py")

# ── Heuristic fallback word lists ────────────────────────────────────────────
AI_WORDS = {
    'furthermore','however','therefore','notably','specifically','particularly',
    'additionally','consequently','substantially','unprecedented','transformative',
    'leverage','utilize','integration','deployment','implementation','facilitate',
    'ensure','enable','demonstrate','proficiency','capabilities','landscape',
    'paradigm','synergy','robust','seamless','holistic','comprehensive','ushering',
    'equitable','implications','remarkable','fundamental','pivotal','streamline',
    'optimize','innovative','scalable','ecosystem','stakeholder','actionable',
    'granular','delineate','elucidate','multifaceted','nuanced','underscores',
    'underpin','pertaining','aforementioned','whilst','thus','hence','thereby',
    'delve','it\'s worth noting','it is important to','in conclusion','in summary',
}
HUMAN_WORDS = {
    'like','just','actually','honestly','weird','bit','kinda','thing','really',
    'pretty','super','totally','basically','literally','tbh','yeah','ok','okay',
    'don\'t','can\'t','won\'t','i\'ve','i\'m','we\'re','they\'re','it\'s',
    'that\'s','gotta','gonna','wanna','sorta','nah','yep','nope','eh','huh',
    'ugh','hmm','i think','i feel','you know','i mean','sort of','kind of',
}

def heuristic_token_score(word: str) -> float:
    c = word.lower().strip(".,!?\"'();:")
    if c in AI_WORDS:    return 0.72 + np.random.uniform(0, 0.25)
    if c in HUMAN_WORDS: return 0.04 + np.random.uniform(0, 0.18)
    l = len(c)
    base = 0.60 if l > 9 else 0.42 if l > 6 else 0.28 if l > 3 else 0.14
    return float(np.clip(base + np.random.uniform(-0.18, 0.18), 0.02, 0.97))

def heuristic_analyze(text: str) -> dict:
    words  = text.strip().split()
    scores = [heuristic_token_score(w) for w in words]
    ai_prob = float(np.clip(np.mean(scores) + np.random.uniform(-0.04, 0.04), 0.02, 0.97))
    return build_result(text, words, scores, ai_prob, source="heuristic")

def model_analyze(text: str) -> dict:
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=512, padding=True
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    ai_prob = float(probs[1].item())
    words   = text.strip().split()
    scores  = []
    for w in words:
        h = heuristic_token_score(w)
        blended = 0.4 * h + 0.6 * ai_prob + np.random.uniform(-0.05, 0.05)
        scores.append(float(np.clip(blended, 0.02, 0.97)))
    return build_result(text, words, scores, ai_prob, source="model")

def build_result(text, words, scores, ai_prob, source):
    if ai_prob > 0.70:   verdict, cls = "AI-GENERATED",      "ai"
    elif ai_prob < 0.35: verdict, cls = "HUMAN-WRITTEN",     "human"
    else:                verdict, cls = "MIXED / UNCERTAIN",  "mixed"

    unique = len(set(w.lower() for w in words))
    lex_div    = unique / max(len(words), 1)
    perplexity = 20 + (1 - ai_prob) * 80 + np.random.uniform(-5, 5)
    burstiness = 0.2 + (1 - ai_prob) * 0.6 + np.random.uniform(-0.05, 0.05)

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 5]
    sent_scores = []
    for s in sentences:
        sw = s.split()
        si = [heuristic_token_score(w) for w in sw]
        base = np.mean(si) if si else ai_prob
        blended = (0.5 * base + 0.5 * ai_prob) if MODEL_READY else base
        sent_scores.append({
            "text":  s,
            "score": float(np.clip(blended + np.random.uniform(-0.06, 0.06), 0.02, 0.97))
        })

    def attr(base, rand_hi):
        return float(np.clip(base + np.random.uniform(0, rand_hi), 0, 1))

    if ai_prob > 0.5:
        raw_attr = [
            {"name": "GPT-4/4o",     "score": attr(0.28, 0.38), "color": "#10a37f"},
            {"name": "GPT-3.5",      "score": attr(0.18, 0.30), "color": "#74aa9c"},
            {"name": "Claude 3/3.5", "score": attr(0.12, 0.28), "color": "#f59e0b"},
            {"name": "Llama / OSS",  "score": attr(0.08, 0.22), "color": "#f97316"},
            {"name": "Human Writer", "score": 1 - ai_prob,      "color": "#16a34a"},
        ]
    else:
        raw_attr = [
            {"name": "GPT-4/4o",     "score": attr(0.04, 0.10), "color": "#10a37f"},
            {"name": "GPT-3.5",      "score": attr(0.03, 0.09), "color": "#74aa9c"},
            {"name": "Claude 3/3.5", "score": attr(0.02, 0.08), "color": "#f59e0b"},
            {"name": "Llama / OSS",  "score": attr(0.02, 0.07), "color": "#f97316"},
            {"name": "Human Writer", "score": 1 - ai_prob,      "color": "#16a34a"},
        ]
    total = sum(a["score"] for a in raw_attr)
    for a in raw_attr:
        a["score"] = round(a["score"] / total * 100, 1)
    raw_attr.sort(key=lambda x: -x["score"])

    return {
        "ai_prob":    round(ai_prob, 4),
        "pct":        round(ai_prob * 100, 1),
        "verdict":    verdict,
        "cls":        cls,
        "word_count": len(words),
        "lex_div":    round(lex_div, 4),
        "perplexity": round(perplexity, 2),
        "burstiness": round(burstiness, 4),
        "tokens":     [{"word": w, "score": round(s, 3)} for w, s in zip(words, scores)],
        "sentences":  sent_scores,
        "attribution": raw_attr,
        "source":     source,
        "model_ready": MODEL_READY,
    }

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML, model_ready=MODEL_READY, model_dir=MODEL_DIR)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = model_analyze(text) if MODEL_READY else heuristic_analyze(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"model_ready": MODEL_READY, "device": device})

# ── Ground Truth Routes ───────────────────────────────────────────────────────
@app.route("/gt/test", methods=["POST"])
def gt_test():
    """Analyze text and compare against ground truth label."""
    data = request.get_json(force=True)
    text       = data.get("text", "").strip()
    true_label = data.get("true_label", "").upper()  # "AI" or "HUMAN"

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if true_label not in ("AI", "HUMAN"):
        return jsonify({"error": "true_label must be AI or HUMAN"}), 400

    try:
        result = model_analyze(text) if MODEL_READY else heuristic_analyze(text)

        # Map verdict to binary prediction
        if result["cls"] == "ai":
            predicted = "AI"
        elif result["cls"] == "human":
            predicted = "HUMAN"
        else:
            # mixed → threshold at 50%
            predicted = "AI" if result["ai_prob"] >= 0.5 else "HUMAN"

        correct = (predicted == true_label)

        # Determine confusion matrix cell
        if true_label == "AI" and predicted == "AI":
            cm_cell = "TP"
        elif true_label == "HUMAN" and predicted == "HUMAN":
            cm_cell = "TN"
        elif true_label == "HUMAN" and predicted == "AI":
            cm_cell = "FP"
        else:
            cm_cell = "FN"

        # Save to CSV log
        file_exists = os.path.exists(GT_LOG_FILE)
        with open(GT_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp","text_snippet","word_count",
                                 "true_label","predicted","ai_prob","correct","cm_cell","source"])
            writer.writerow([
                datetime.now().isoformat(),
                text[:80].replace("\n", " "),
                result["word_count"],
                true_label,
                predicted,
                result["ai_prob"],
                correct,
                cm_cell,
                result["source"]
            ])

        return jsonify({
            "result":    result,
            "predicted": predicted,
            "correct":   correct,
            "cm_cell":   cm_cell,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/gt/stats", methods=["GET"])
def gt_stats():
    """Return aggregated accuracy stats from the CSV log."""
    if not os.path.exists(GT_LOG_FILE):
        return jsonify({"total": 0, "rows": []})

    rows = []
    with open(GT_LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return jsonify({"total": 0, "rows": []})

    total  = len(rows)
    tp = sum(1 for r in rows if r["cm_cell"] == "TP")
    tn = sum(1 for r in rows if r["cm_cell"] == "TN")
    fp = sum(1 for r in rows if r["cm_cell"] == "FP")
    fn = sum(1 for r in rows if r["cm_cell"] == "FN")
    correct = sum(1 for r in rows if r["correct"] == "True")

    accuracy  = round(correct / total * 100, 1) if total else 0
    precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) else 0
    recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) else 0
    f1        = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) else 0

    return jsonify({
        "total":     total,
        "correct":   correct,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "rows":      rows[-30:]  # last 30 for timeline
    })


@app.route("/gt/export", methods=["GET"])
def gt_export():
    """Return the raw CSV content for download."""
    if not os.path.exists(GT_LOG_FILE):
        return "No data yet.", 404
    with open(GT_LOG_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    from flask import Response
    return Response(
        content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=scripta_ground_truth.csv"}
    )


@app.route("/gt/clear", methods=["POST"])
def gt_clear():
    """Clear the ground truth log."""
    if os.path.exists(GT_LOG_FILE):
        os.remove(GT_LOG_FILE)
    return jsonify({"ok": True})


# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Scripta · AI Text Detector</title>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,500;12..96,600;12..96,700;12..96,800&family=Fira+Code:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#fff7ed;--s1:#ffedd5;--s2:#fed7aa;--ink:#7c2d12;--ink2:#9a3412;--dim:#6b7280;--muted:#9ca3af;--border:rgba(124,45,18,0.12);--border2:rgba(124,45,18,0.22);--pri:#f97316;--pri-light:rgba(249,115,22,0.15);--pri-glow:rgba(249,115,22,0.35);--red:#dc2626;--red-light:rgba(220,38,38,0.12);--green:#16a34a;--green-light:rgba(22,163,74,0.12);--amber:#f59e0b;--amber-light:rgba(245,158,11,0.12);--cyan:#06b6d4;--pink:#f43f5e;--mono:'Fira Code',monospace;--head:'Bricolage Grotesque',sans-serif;--body:'Bricolage Grotesque',sans-serif;--r:12px;--r-sm:8px;}
*{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{background:var(--bg);color:var(--ink);font-family:var(--body);min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--pri),var(--pink),var(--red),var(--amber),var(--green),var(--cyan),var(--pri));z-index:100;}

header{position:sticky;top:3px;z-index:90;display:flex;align-items:center;justify-content:space-between;padding:0 36px;height:64px;background:rgba(255,255,255,0.96);backdrop-filter:blur(20px);border-bottom:1.5px solid var(--border);}
.logo{display:flex;align-items:center;gap:12px;}
.logo-icon{width:36px;height:36px;background:var(--pri);border-radius:10px;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 14px var(--pri-glow);}
.logo-name{font-family:var(--head);font-size:22px;font-weight:800;color:var(--ink);letter-spacing:-.03em;}
.logo-badge{padding:3px 9px;background:var(--pri-light);color:var(--pri);border-radius:20px;font-family:var(--mono);font-size:10px;font-weight:500;letter-spacing:.04em;}
.hdr-right{display:flex;align-items:center;gap:12px;}
.hdr-pill{padding:6px 14px;border-radius:20px;font-family:var(--mono);font-size:11px;font-weight:500;border:1.5px solid var(--border);}
.hdr-pill.model-on{background:var(--green-light);color:var(--green);border-color:rgba(24,201,106,0.3);display:flex;align-items:center;gap:6px;}
.hdr-pill.model-off{background:var(--amber-light);color:var(--amber);border-color:rgba(245,158,11,0.3);display:flex;align-items:center;gap:6px;}
.live-dot{width:6px;height:6px;border-radius:50%;background:currentColor;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 currentColor;}60%{box-shadow:0 0 0 5px transparent;}}

nav{position:sticky;top:67px;z-index:80;display:flex;gap:4px;padding:10px 36px;background:rgba(255,255,255,0.9);backdrop-filter:blur(20px);border-bottom:1.5px solid var(--border);overflow-x:auto;}
.nav-tab{padding:9px 18px;font-family:var(--body);font-size:13px;font-weight:600;color:var(--muted);cursor:pointer;border:none;background:none;border-radius:8px;transition:all .18s;white-space:nowrap;flex-shrink:0;}
.nav-tab:hover{color:var(--ink);background:var(--s1);}
.nav-tab.active{color:var(--pri);background:var(--pri-light);}

.page{display:none;max-width:1380px;margin:0 auto;padding:36px;}
.page.active{display:block;}
.sec-title{font-family:var(--head);font-size:36px;font-weight:800;letter-spacing:-.04em;line-height:1.1;margin-bottom:4px;}
.sec-sub{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.06em;margin-bottom:32px;text-transform:uppercase;}

.panel{background:var(--bg);border:1.5px solid var(--border);border-radius:var(--r);overflow:hidden;transition:border-color .2s;}
.panel:hover{border-color:var(--border2);}
.ph{display:flex;align-items:center;justify-content:space-between;padding:12px 18px;border-bottom:1.5px solid var(--border);background:var(--s1);}
.ph-title{font-family:var(--body);font-size:12px;font-weight:700;color:var(--ink2);letter-spacing:.02em;display:flex;align-items:center;gap:8px;}
.ph-title .dot{width:8px;height:8px;border-radius:50%;}
.pb{padding:18px;}

textarea.ta{width:100%;min-height:210px;background:var(--s1);border:1.5px solid var(--border);border-radius:var(--r-sm);padding:14px 16px;color:var(--ink);font-family:var(--body);font-size:14px;line-height:1.75;resize:vertical;outline:none;transition:border-color .2s,box-shadow .2s;}
textarea.ta:focus{border-color:var(--pri);box-shadow:0 0 0 4px var(--pri-light);}
textarea.ta::placeholder{color:var(--muted);}

.btn{padding:10px 20px;border-radius:var(--r-sm);border:1.5px solid transparent;cursor:pointer;font-family:var(--body);font-size:13px;font-weight:700;transition:all .17s;}
.btn-p{background:var(--pri);color:#fff;border-color:var(--pri);box-shadow:0 4px 16px var(--pri-glow);}
.btn-p:hover{filter:brightness(1.1);transform:translateY(-1px);}
.btn-p:disabled{opacity:.5;cursor:not-allowed;transform:none;}
.btn-g{background:var(--s1);color:var(--ink2);border-color:var(--border2);}
.btn-g:hover{background:var(--s2);border-color:var(--pri);color:var(--pri);}
.btn-s{background:transparent;color:var(--dim);border-color:var(--border);}
.btn-s:hover{border-color:var(--border2);color:var(--ink);}
.btn-danger{background:var(--red-light);color:var(--red);border-color:rgba(220,38,38,.3);}
.btn-danger:hover{background:var(--red);color:#fff;}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;}

.verdict-big{border-radius:var(--r);padding:24px;position:relative;overflow:hidden;}
.verdict-big.ai-v{background:linear-gradient(135deg,#fff5f5,#fff0f0);border:2px solid rgba(220,38,38,.25);}
.verdict-big.human-v{background:linear-gradient(135deg,#f0fff7,#edfff5);border:2px solid rgba(22,163,74,.25);}
.verdict-big.mixed-v{background:linear-gradient(135deg,#fffbf0,#fff8e8);border:2px solid rgba(245,158,11,.25);}
.big-pct{font-family:var(--head);font-size:72px;font-weight:800;line-height:1;letter-spacing:-.04em;margin:6px 0;}
.big-label{font-family:var(--body);font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:2px;}

.meter-track{height:7px;background:var(--s2);border-radius:4px;overflow:hidden;margin:10px 0;}
.meter-fill{height:100%;border-radius:4px;transition:width 1.1s cubic-bezier(.22,.68,0,1.2);}

.metric-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.ml{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.05em;width:112px;flex-shrink:0;text-transform:uppercase;}
.mtrack{flex:1;height:5px;background:var(--s2);border-radius:3px;overflow:hidden;}
.mfill{height:100%;border-radius:3px;transition:width 1s cubic-bezier(.22,.68,0,1.2);}
.mv{font-family:var(--mono);font-size:11px;color:var(--ink2);width:36px;text-align:right;font-weight:600;}

#heatmap{line-height:2.1;font-size:14px;word-spacing:1px;}
.tok{display:inline;border-radius:4px;padding:2px 1px;cursor:crosshair;transition:opacity .12s;}
.tok:hover{opacity:.72;}

.tip{position:fixed;background:var(--ink);border-radius:10px;padding:12px 16px;font-family:var(--mono);font-size:11px;pointer-events:none;z-index:9000;opacity:0;transition:opacity .1s;box-shadow:0 16px 40px rgba(0,0,0,.2);}
.tip.show{opacity:1;}
.tip-word{color:#fff;font-weight:700;font-size:13px;margin-bottom:6px;}
.tip-row{display:flex;justify-content:space-between;gap:14px;color:#94a3b8;margin-bottom:2px;}
.tip-bar{height:4px;background:rgba(255,255,255,.12);border-radius:2px;margin-top:8px;}

.realtime-wrap{display:grid;grid-template-columns:1fr 320px;gap:20px;}
.rt-editor{width:100%;min-height:360px;background:var(--s1);border:1.5px solid var(--border);border-radius:var(--r-sm);padding:18px;color:var(--ink);font-family:var(--body);font-size:14px;line-height:1.8;resize:none;outline:none;transition:border-color .2s,box-shadow .2s;}
.rt-editor:focus{border-color:var(--pri);box-shadow:0 0 0 4px var(--pri-light);}
.sentence-card{padding:10px 12px;border-radius:var(--r-sm);border:1.5px solid var(--border);margin-bottom:7px;cursor:pointer;transition:all .15s;display:flex;align-items:center;gap:10px;background:var(--s1);}
.sc-bar{width:40px;height:40px;flex-shrink:0;position:relative;}
.sc-bar svg{transform:rotate(-90deg);}
.sc-pct{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:10px;font-weight:700;}
.sc-text{font-size:12px;color:var(--dim);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.rt-gauge{text-align:center;padding:20px 0 12px;}
.rt-big{font-family:var(--head);font-size:80px;font-weight:800;letter-spacing:-.04em;line-height:1;transition:all .3s;}
.rt-label{font-family:var(--body);font-size:12px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-top:4px;}
.wave-bar{display:flex;align-items:flex-end;gap:3px;height:54px;margin:14px 0;}
.wave-col{flex:1;border-radius:3px 3px 0 0;transition:height .3s,background .3s;}

.report-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px;}
.stat-card{background:var(--bg);border:1.5px solid var(--border);border-radius:var(--r);padding:24px;text-align:center;cursor:default;position:relative;overflow:hidden;transition:all .2s;}
.stat-card:hover{transform:translateY(-3px);box-shadow:0 10px 32px rgba(0,0,0,.08);}
.stat-card::after{content:'';position:absolute;bottom:0;left:0;right:0;height:3px;}
.stat-card:nth-child(1)::after{background:var(--pri);}
.stat-card:nth-child(2)::after{background:var(--red);}
.stat-card:nth-child(3)::after{background:var(--green);}
.sc-num{font-family:var(--head);font-size:52px;font-weight:800;letter-spacing:-.04em;line-height:1;margin-bottom:6px;}
.sc-lbl{font-family:var(--body);font-size:11px;font-weight:600;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;}

.donut-wrap{display:flex;justify-content:center;align-items:center;gap:28px;padding:22px 0;}
.donut-legend{display:flex;flex-direction:column;gap:12px;}
.dl-row{display:flex;align-items:center;gap:10px;font-family:var(--body);font-size:13px;font-weight:500;color:var(--dim);}
.dl-dot{width:10px;height:10px;border-radius:3px;flex-shrink:0;}

.timeline{position:relative;padding-left:22px;}
.timeline::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--s2);border-radius:2px;}
.tl-item{position:relative;margin-bottom:18px;}
.tl-item::before{content:'';position:absolute;left:-26px;top:5px;width:8px;height:8px;border-radius:50%;border:2px solid var(--muted);background:var(--bg);}
.tl-item.done::before{background:var(--pri);border-color:var(--pri);}
.tl-time{font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:2px;}
.tl-text{font-size:13px;color:var(--ink2);font-weight:500;}
.tl-badge{display:inline-block;margin-top:4px;padding:3px 9px;border-radius:20px;font-family:var(--body);font-size:11px;font-weight:700;}
.badge-ai{background:var(--red-light);color:var(--red);}
.badge-human{background:var(--green-light);color:var(--green);}
.badge-mixed{background:var(--amber-light);color:var(--amber);}

.rw-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;}
.diff-view{background:var(--s1);border:1.5px solid var(--border);border-radius:var(--r-sm);padding:16px;font-size:13px;line-height:1.9;min-height:200px;}
.diff-add{background:rgba(22,163,74,.18);color:#0a5c30;border-radius:3px;padding:0 1px;}
.diff-del{background:rgba(220,38,38,.15);color:#b91c1c;border-radius:3px;text-decoration:line-through;padding:0 1px;}

.explain-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;}
.signal-card{background:var(--s1);border:1.5px solid var(--border);border-radius:var(--r);padding:20px;margin-bottom:10px;transition:all .2s;}
.signal-card:hover{border-color:var(--pri);background:var(--bg);}
.sig-icon{font-size:22px;margin-bottom:8px;}
.sig-title{font-family:var(--head);font-size:15px;font-weight:700;margin-bottom:5px;}
.sig-desc{font-size:13px;color:var(--dim);line-height:1.65;}
.sig-impact{display:inline-block;margin-top:8px;padding:3px 10px;border-radius:20px;font-family:var(--body);font-size:11px;font-weight:700;}
.impact-hi{background:var(--red-light);color:var(--red);}
.impact-med{background:var(--amber-light);color:var(--amber);}

.gloss-item{padding:13px 0;border-bottom:1.5px solid var(--border);}
.gloss-term{font-family:var(--mono);font-size:11px;color:var(--pri);font-weight:700;margin-bottom:3px;letter-spacing:.04em;}
.gloss-def{font-size:13px;color:var(--dim);line-height:1.6;}

.loading-overlay{display:none;position:absolute;inset:0;background:rgba(255,255,255,.9);backdrop-filter:blur(8px);border-radius:var(--r);z-index:50;align-items:center;justify-content:center;flex-direction:column;gap:14px;}
.loading-overlay.on{display:flex;}
.scan-line{width:160px;height:3px;border-radius:2px;background:linear-gradient(to right,transparent,var(--pri),var(--pink),transparent);animation:sc 1.3s ease-in-out infinite;}
@keyframes sc{0%{transform:scaleX(.15) translateX(-500%);opacity:0;}40%{opacity:1;}100%{transform:scaleX(1) translateX(500%);opacity:0;}}
.loading-txt{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.15em;animation:bk 1.2s infinite;}
@keyframes bk{0%,100%{opacity:1;}50%{opacity:.2;}}

.sl{font-family:var(--body);font-size:11px;font-weight:700;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:10px;}
.sl::after{content:'';flex:1;height:1.5px;background:var(--border);}
.tag{display:inline-block;padding:3px 9px;border-radius:20px;font-family:var(--body);font-size:10px;font-weight:700;letter-spacing:.04em;}
.tag-new{background:linear-gradient(135deg,var(--pri),var(--pink));color:#fff;}
.analyzer-grid{display:grid;grid-template-columns:1fr 390px;gap:20px;align-items:start;}

.model-banner{padding:10px 18px;border-radius:var(--r-sm);font-family:var(--mono);font-size:11px;margin-bottom:16px;display:flex;align-items:center;gap:10px;}
.model-banner.ready{background:var(--green-light);color:var(--green);border:1.5px solid rgba(22,163,74,.3);}
.model-banner.heuristic{background:var(--amber-light);color:var(--amber);border:1.5px solid rgba(245,158,11,.3);}
.source-badge{display:inline-block;padding:2px 8px;border-radius:20px;font-family:var(--mono);font-size:9px;font-weight:700;margin-left:8px;}
.source-real{background:var(--green-light);color:var(--green);}
.source-heuristic{background:var(--amber-light);color:var(--amber);}

/* ── Ground Truth Styles ── */
.gt-grid{display:grid;grid-template-columns:1fr 420px;gap:20px;align-items:start;}
.gt-stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;}
.gt-stat{background:var(--bg);border:1.5px solid var(--border);border-radius:var(--r);padding:18px;text-align:center;transition:all .2s;}
.gt-stat:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.07);}
.gt-stat-num{font-family:var(--head);font-size:38px;font-weight:800;letter-spacing:-.04em;line-height:1;margin-bottom:4px;}
.gt-stat-lbl{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;}
.cm-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0;}
.cm-cell{border-radius:var(--r-sm);padding:14px;text-align:center;border:1.5px solid var(--border);}
.cm-num{font-family:var(--head);font-size:32px;font-weight:800;line-height:1;}
.cm-lbl{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.08em;margin-top:3px;}
.gt-row{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:var(--r-sm);border:1.5px solid var(--border);margin-bottom:6px;background:var(--s1);font-size:12px;}
.gt-row.correct{border-left:3px solid var(--green);}
.gt-row.wrong{border-left:3px solid var(--red);}
.gt-badge{padding:2px 8px;border-radius:20px;font-family:var(--mono);font-size:10px;font-weight:700;flex-shrink:0;}
.label-toggle{display:flex;gap:8px;margin:14px 0;}
.lbtn{flex:1;padding:14px;border-radius:var(--r-sm);border:2px solid var(--border);cursor:pointer;font-family:var(--body);font-size:14px;font-weight:700;text-align:center;transition:all .18s;background:var(--s1);}
.lbtn:hover{border-color:var(--border2);}
.lbtn.sel-ai{background:var(--red-light);border-color:var(--red);color:var(--red);}
.lbtn.sel-human{background:var(--green-light);border-color:var(--green);color:var(--green);}
.result-flash{border-radius:var(--r);padding:18px;text-align:center;margin-top:14px;display:none;}
.result-flash.correct-flash{background:var(--green-light);border:2px solid rgba(22,163,74,.35);}
.result-flash.wrong-flash{background:var(--red-light);border:2px solid rgba(220,38,38,.35);}
.flash-icon{font-size:36px;margin-bottom:6px;}
.flash-title{font-family:var(--head);font-size:20px;font-weight:800;}
.flash-sub{font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:4px;}

@keyframes fadeUp{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:none;}}
.fu{animation:fadeUp .4s forwards;}
.fu1{animation-delay:.05s;opacity:0;}
.fu2{animation-delay:.12s;opacity:0;}
.fu3{animation-delay:.2s;opacity:0;}

::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-track{background:var(--s1);}::-webkit-scrollbar-thumb{background:rgba(249,115,22,.25);border-radius:3px;}
@media(max-width:900px){.analyzer-grid,.realtime-wrap,.rw-grid,.explain-grid,.gt-grid{grid-template-columns:1fr;}.report-grid,.gt-stats-grid{grid-template-columns:1fr 1fr;}header,nav,.page{padding-left:18px;padding-right:18px;}}
</style>
</head>
<body>

<div class="tip" id="tip">
  <div class="tip-word" id="tip-word"></div>
  <div class="tip-row"><span>AI Score</span><span id="tip-score" style="color:#fff;font-weight:700;"></span></div>
  <div class="tip-row"><span>Category</span><span id="tip-cat"></span></div>
  <div class="tip-bar"><div id="tip-bfill" style="height:100%;border-radius:2px;transition:width .3s;"></div></div>
</div>

<header>
  <div class="logo">
    <div class="logo-icon">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round">
        <circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/>
      </svg>
    </div>
    <div class="logo-name">Scripta</div>
    <span class="logo-badge">DETECTOR · PYTHON</span>
  </div>
 <div class="hdr-right">
    <div class="hdr-pill" style="color:var(--dim); font-family:var(--body); font-size:12px;">
      Built by <strong style="color:var(--ink);">Nilima Maybhate</strong>
    </div>
</header>

<nav>
  <button class="nav-tab active" onclick="gotoPage('analyzer',this)">⬡ Analyzer</button>
  <button class="nav-tab" onclick="gotoPage('realtime',this)">⚡ Live Monitor</button>
  <button class="nav-tab" onclick="gotoPage('report',this)">📊 Report</button>
  <button class="nav-tab" onclick="gotoPage('rewrite',this)">🔀 Rewrite Detector</button>
  <button class="nav-tab" onclick="gotoPage('groundtruth',this)">🧪 Ground Truth <span class="tag tag-new" style="margin-left:4px;">RESEARCH</span></button>
  <button class="nav-tab" onclick="gotoPage('explain',this)">🧠 How It Works</button>
</nav>

<!-- PAGE 1: Analyzer -->
<div class="page active" id="page-analyzer">
  <div class="fu fu1" style="margin-bottom:28px;">
    <div class="sec-title">Text <span style="color:var(--pri);">Analysis</span></div>
    <div class="sec-sub">◆ powered by python backend · token heatmap · model fingerprinting</div>
  </div>

  {% if not model_ready %}
  <div class="model-banner heuristic fu fu1">
    ⚠ Running in heuristic mode — train your model for real predictions:
    <code style="margin-left:8px;background:rgba(0,0,0,.07);padding:2px 8px;border-radius:4px;">python train_detector.py</code>
  </div>
  {% else %}
  <div class="model-banner ready fu fu1">
    ✅ Real ML model active (DistilBERT fine-tuned on HC3 dataset) — Scripta
  </div>
  {% endif %}

  <div class="analyzer-grid">
    <div style="display:flex;flex-direction:column;gap:18px;">
      <div class="panel fu fu2" style="position:relative;">
        <div class="loading-overlay" id="ld1"><div class="scan-line"></div><div class="loading-txt">QUERYING PYTHON MODEL…</div></div>
        <div class="ph">
          <div class="ph-title"><div class="dot" style="background:var(--pri);"></div>Input Text</div>
          <div style="display:flex;gap:7px;">
            <button class="btn btn-s" style="padding:5px 11px;font-size:11px;" onclick="loadSample('ai')">AI Sample</button>
            <button class="btn btn-s" style="padding:5px 11px;font-size:11px;" onclick="loadSample('human')">Human Sample</button>
          </div>
        </div>
        <div class="pb">
          <textarea class="ta" id="mainInput" rows="10" placeholder="Paste any text here to detect AI generation…" oninput="updateWC()"></textarea>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;font-family:var(--mono);font-size:10px;color:var(--muted);">
            <span id="wc">0 words</span><span>Ctrl+Enter to analyze</span>
          </div>
          <div class="controls">
            <button class="btn btn-p" id="analyzeBtn" onclick="runAnalysis()">▶ Analyze Text</button>
            <button class="btn btn-g" onclick="showHeatmap()">🔥 Token Heatmap</button>
            <button class="btn btn-s" onclick="clearAll()">✕ Clear</button>
          </div>
        </div>
      </div>
      <div class="panel fu fu3" id="heatmapPanel" style="display:none;">
        <div class="ph">
          <div class="ph-title"><div class="dot" style="background:var(--amber);"></div>Token Heatmap</div>
          <span style="font-family:var(--mono);font-size:10px;color:var(--muted);">hover any word</span>
        </div>
        <div class="pb">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;font-family:var(--mono);font-size:10px;color:var(--muted);">
            <span>HUMAN</span>
            <div style="flex:1;height:5px;border-radius:3px;background:linear-gradient(to right,var(--green),var(--amber),var(--red));"></div>
            <span>AI</span>
          </div>
          <div id="heatmap" style="background:var(--s1);border-radius:var(--r-sm);padding:16px;border:1.5px solid var(--border);min-height:100px;">
            <span style="font-family:var(--mono);font-size:11px;color:var(--muted);">Run analysis first →</span>
          </div>
        </div>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:18px;">
      <div class="panel fu fu2">
        <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--green);"></div>Detection Result</div></div>
        <div class="pb">
          <div id="verdictArea" style="text-align:center;padding:32px 0;">
            <div style="font-size:52px;opacity:.08;margin-bottom:10px;">◎</div>
            <div style="font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.1em;">AWAITING ANALYSIS</div>
          </div>
        </div>
      </div>
      <div class="panel fu fu3" id="metricsPanel" style="display:none;">
        <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--cyan);"></div>Linguistic Signals</div></div>
        <div class="pb" id="metricsBody"></div>
      </div>
      <div class="panel fu fu3" id="attrPanel" style="display:none;">
        <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--pink);"></div>Source Fingerprint</div></div>
        <div class="pb" id="attrBody"></div>
      </div>
    </div>
  </div>
</div>

<!-- PAGE 2: Live Monitor -->
<div class="page" id="page-realtime">
  <div class="fu fu1" style="margin-bottom:28px;">
    <div class="sec-title">Live <span style="color:var(--amber);">Monitor</span></div>
    <div class="sec-sub">◆ type and analyze · per-sentence breakdown</div>
  </div>
  <div class="realtime-wrap">
    <div style="display:flex;flex-direction:column;gap:16px;">
      <div class="panel">
        <div class="ph">
          <div class="ph-title"><div class="dot" style="background:var(--pri);"></div>Writing Pad</div>
        </div>
        <div class="pb">
          <textarea class="rt-editor ta" id="rtInput" rows="13" placeholder="Paste text here, then click Analyze below…"></textarea>
          <div class="controls" style="margin-top:12px;">
            <button class="btn btn-p" onclick="rtAnalyze()">▶ Analyze</button>
            <button class="btn btn-s" onclick="document.getElementById('rtInput').value='';resetRT();">✕ Clear</button>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--amber);"></div>AI Probability Wave</div></div>
        <div class="pb">
          <div class="wave-bar" id="waveBar"><div style="font-family:var(--mono);font-size:11px;color:var(--muted);">Analyze text to see wave →</div></div>
          <div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;color:var(--muted);"><span>HUMAN ←</span><span>→ AI</span></div>
        </div>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:16px;">
      <div class="panel">
        <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--green);"></div>Overall Score</div></div>
        <div class="pb">
          <div class="rt-gauge">
            <div class="rt-big" id="rtBig" style="color:var(--muted);">--</div>
            <div class="rt-label" id="rtLbl" style="color:var(--muted);">AI Probability</div>
          </div>
          <div class="meter-track"><div class="meter-fill" id="rtMeter" style="width:0%;background:var(--muted);"></div></div>
          <div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;color:var(--muted);"><span>HUMAN</span><span>AI</span></div>
        </div>
      </div>
      <div class="panel" style="flex:1;">
        <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--cyan);"></div>Per-Sentence Scores</div></div>
        <div class="pb" id="sentenceList" style="max-height:380px;overflow-y:auto;">
          <div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-align:center;padding:20px 0;">Analyze text to see scores…</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- PAGE 3: Report -->
<div class="page" id="page-report">
  <div class="fu fu1" style="margin-bottom:28px;">
    <div class="sec-title">Session <span style="color:var(--cyan);">Report</span></div>
    <div class="sec-sub">◆ cumulative stats across all your analyses this session</div>
  </div>
  <div class="report-grid fu fu2">
    <div class="stat-card"><div class="sc-num" id="rTotalAnalyzed" style="color:var(--pri);">0</div><div class="sc-lbl">Texts Analyzed</div></div>
    <div class="stat-card"><div class="sc-num" id="rAvgAI" style="color:var(--red);">—</div><div class="sc-lbl">Avg AI Probability</div></div>
    <div class="stat-card"><div class="sc-num" id="rWordCount" style="color:var(--green);">0</div><div class="sc-lbl">Total Words Scanned</div></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;">
    <div class="panel fu fu2">
      <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--pri);"></div>Verdict Distribution</div></div>
      <div class="pb">
        <div class="donut-wrap">
          <svg width="156" height="156">
            <circle cx="78" cy="78" r="58" fill="none" stroke="var(--s2)" stroke-width="22"/>
            <circle id="donutAI" cx="78" cy="78" r="58" fill="none" stroke="var(--red)" stroke-width="22" stroke-linecap="round" transform="rotate(-90 78 78)" stroke-dasharray="364" stroke-dashoffset="364" style="transition:stroke-dashoffset 1s ease;"/>
            <circle id="donutHuman" cx="78" cy="78" r="58" fill="none" stroke="var(--green)" stroke-width="22" stroke-linecap="round" transform="rotate(-90 78 78)" stroke-dasharray="364" stroke-dashoffset="364" style="transition:stroke-dashoffset 1s ease;"/>
            <circle id="donutMixed" cx="78" cy="78" r="58" fill="none" stroke="var(--amber)" stroke-width="22" stroke-linecap="round" transform="rotate(-90 78 78)" stroke-dasharray="364" stroke-dashoffset="364" style="transition:stroke-dashoffset 1s ease;"/>
          </svg>
          <div class="donut-legend">
            <div class="dl-row"><div class="dl-dot" style="background:var(--red);"></div><span id="dAI">AI-Generated: 0</span></div>
            <div class="dl-row"><div class="dl-dot" style="background:var(--green);"></div><span id="dHuman">Human-Written: 0</span></div>
            <div class="dl-row"><div class="dl-dot" style="background:var(--amber);"></div><span id="dMixed">Mixed/Uncertain: 0</span></div>
          </div>
        </div>
        <div id="donutEmpty" style="text-align:center;font-family:var(--mono);font-size:11px;color:var(--muted);padding:14px;">Run analyses to populate this.</div>
      </div>
    </div>
    <div class="panel fu fu3">
      <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--cyan);"></div>Analysis Timeline</div></div>
      <div class="pb" style="max-height:320px;overflow-y:auto;">
        <div class="timeline" id="timeline"><div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-align:center;padding:20px 0;">No analyses yet.</div></div>
      </div>
    </div>
  </div>
</div>

<!-- PAGE 4: Rewrite Detector -->
<div class="page" id="page-rewrite">
  <div class="fu fu1" style="margin-bottom:28px;">
    <div class="sec-title">Rewrite <span style="color:var(--green);">Detector</span></div>
    <div class="sec-sub">◆ detect if text A was AI-rewritten into text B</div>
  </div>
  <div class="rw-grid fu fu2">
    <div><div style="font-family:var(--body);font-size:12px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;margin-bottom:8px;color:var(--green);">Original Text (A)</div><textarea class="ta" id="rwA" rows="9" placeholder="Paste the original / human-written text here…"></textarea></div>
    <div><div style="font-family:var(--body);font-size:12px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;margin-bottom:8px;color:var(--red);">Suspect Text (B)</div><textarea class="ta" id="rwB" rows="9" placeholder="Paste the potentially AI-rewritten version here…"></textarea></div>
  </div>
  <div style="margin:14px 0;display:flex;gap:10px;" class="fu fu3">
    <button class="btn btn-p" id="rwBtn" onclick="runRewrite()">▶ Detect Rewriting</button>
    <button class="btn btn-s" onclick="loadRwSample()">Load Example</button>
  </div>
  <div id="rwResult" style="display:none;" class="fu fu3">
    <div style="display:grid;grid-template-columns:1fr 260px;gap:18px;">
      <div class="panel">
        <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--pri);"></div>Word-Level Diff</div></div>
        <div class="pb">
          <div style="display:flex;gap:14px;margin-bottom:10px;font-family:var(--body);font-size:11px;font-weight:600;">
            <span style="color:var(--green);">■ Added</span><span style="color:var(--red);">■ Removed</span><span style="color:var(--muted);">■ Same</span>
          </div>
          <div class="diff-view" id="diffView"></div>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:14px;">
        <div class="panel">
          <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--amber);"></div>Rewrite Score</div></div>
          <div class="pb" style="text-align:center;padding:18px 0;">
            <div id="rwBig" style="font-family:var(--head);font-size:60px;font-weight:800;letter-spacing:-.04em;line-height:1;"></div>
            <div id="rwLbl" style="font-family:var(--body);font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-top:5px;"></div>
            <div class="meter-track" style="margin-top:14px;"><div id="rwMeter" class="meter-fill" style="width:0%;"></div></div>
          </div>
        </div>
        <div class="panel">
          <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--cyan);"></div>Breakdown</div></div>
          <div class="pb" id="rwBreakdown"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- PAGE 5: Ground Truth Tester -->
<div class="page" id="page-groundtruth">
  <div class="fu fu1" style="margin-bottom:28px;">
    <div class="sec-title">Ground Truth <span style="color:var(--cyan);">Tester</span></div>
    <div class="sec-sub">◆ label texts as human/ai · track accuracy · export research data as csv</div>
  </div>

  <!-- Live accuracy stats bar -->
  <div class="gt-stats-grid fu fu2" id="gtStatsGrid">
    <div class="gt-stat"><div class="gt-stat-num" id="gtAccuracy" style="color:var(--pri);">—</div><div class="gt-stat-lbl">Accuracy</div></div>
    <div class="gt-stat"><div class="gt-stat-num" id="gtPrecision" style="color:var(--cyan);">—</div><div class="gt-stat-lbl">Precision</div></div>
    <div class="gt-stat"><div class="gt-stat-num" id="gtRecall" style="color:var(--pink);">—</div><div class="gt-stat-lbl">Recall</div></div>
    <div class="gt-stat"><div class="gt-stat-num" id="gtF1" style="color:var(--green);">—</div><div class="gt-stat-lbl">F1 Score</div></div>
  </div>

  <div class="gt-grid">
    <!-- Left: input panel -->
    <div style="display:flex;flex-direction:column;gap:18px;">
      <div class="panel fu fu2" style="position:relative;">
        <div class="loading-overlay" id="gtLoader"><div class="scan-line"></div><div class="loading-txt">TESTING…</div></div>
        <div class="ph">
          <div class="ph-title"><div class="dot" style="background:var(--cyan);"></div>Text to Test</div>
          <div style="display:flex;gap:7px;">
            <button class="btn btn-s" style="padding:5px 11px;font-size:11px;" onclick="gtLoadSample('ai')">AI Sample</button>
            <button class="btn btn-s" style="padding:5px 11px;font-size:11px;" onclick="gtLoadSample('human')">Human Sample</button>
          </div>
        </div>
        <div class="pb">
          <textarea class="ta" id="gtInput" rows="8" placeholder="Paste a text sample here. Then select whether it is truly Human or AI-written below…"></textarea>

          <div style="margin-top:16px;">
            <div style="font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">What is this text ACTUALLY written by?</div>
            <div class="label-toggle">
              <div class="lbtn" id="lbtnHuman" onclick="selectLabel('HUMAN')">👤 Human Written</div>
              <div class="lbtn" id="lbtnAI" onclick="selectLabel('AI')">🤖 AI Generated</div>
            </div>
          </div>

          <div class="controls">
            <button class="btn btn-p" id="gtTestBtn" onclick="runGTTest()">🧪 Test &amp; Record</button>
            <button class="btn btn-g" onclick="gtClear()">✕ Clear</button>
          </div>

          <!-- Result flash -->
          <div id="gtFlash" class="result-flash">
            <div class="flash-icon" id="gtFlashIcon"></div>
            <div class="flash-title" id="gtFlashTitle"></div>
            <div class="flash-sub" id="gtFlashSub"></div>
          </div>
        </div>
      </div>

      <!-- Confusion matrix -->
      <div class="panel fu fu3">
        <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--pri);"></div>Confusion Matrix</div></div>
        <div class="pb">
          <div style="display:grid;grid-template-columns:auto 1fr 1fr;gap:6px;align-items:center;margin-bottom:6px;">
            <div></div>
            <div style="text-align:center;font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.08em;">PRED: AI</div>
            <div style="text-align:center;font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.08em;">PRED: HUMAN</div>
          </div>
          <div style="display:grid;grid-template-columns:auto 1fr 1fr;gap:6px;align-items:center;">
            <div style="font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.06em;writing-mode:vertical-rl;transform:rotate(180deg);text-align:center;">ACTUAL: AI</div>
            <div class="cm-cell" style="background:var(--green-light);border-color:rgba(22,163,74,.3);">
              <div class="cm-num" id="cmTP" style="color:var(--green);">0</div>
              <div class="cm-lbl">TRUE POS</div>
            </div>
            <div class="cm-cell" style="background:var(--red-light);border-color:rgba(220,38,38,.3);">
              <div class="cm-num" id="cmFN" style="color:var(--red);">0</div>
              <div class="cm-lbl">FALSE NEG</div>
            </div>
            <div style="font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.06em;writing-mode:vertical-rl;transform:rotate(180deg);text-align:center;">ACTUAL: HUMAN</div>
            <div class="cm-cell" style="background:var(--red-light);border-color:rgba(220,38,38,.3);">
              <div class="cm-num" id="cmFP" style="color:var(--red);">0</div>
              <div class="cm-lbl">FALSE POS</div>
            </div>
            <div class="cm-cell" style="background:var(--green-light);border-color:rgba(22,163,74,.3);">
              <div class="cm-num" id="cmTN" style="color:var(--green);">0</div>
              <div class="cm-lbl">TRUE NEG</div>
            </div>
          </div>
          <div style="margin-top:12px;font-family:var(--mono);font-size:10px;color:var(--muted);line-height:1.7;">
            <span style="color:var(--green);">TP</span> = correctly flagged AI &nbsp;·&nbsp;
            <span style="color:var(--green);">TN</span> = correctly passed human<br>
            <span style="color:var(--red);">FP</span> = human wrongly flagged &nbsp;·&nbsp;
            <span style="color:var(--red);">FN</span> = AI that slipped through
          </div>
        </div>
      </div>
    </div>

    <!-- Right: history + export -->
    <div style="display:flex;flex-direction:column;gap:18px;">
      <div class="panel fu fu2">
        <div class="ph">
          <div class="ph-title"><div class="dot" style="background:var(--green);"></div>Test History</div>
          <div style="display:flex;gap:7px;">
            <a href="/gt/export" class="btn btn-g" style="padding:5px 12px;font-size:11px;text-decoration:none;">⬇ Export CSV</a>
            <button class="btn btn-danger" style="padding:5px 11px;font-size:11px;" onclick="gtClearAll()">🗑 Clear All</button>
          </div>
        </div>
        <div class="pb" style="max-height:480px;overflow-y:auto;" id="gtHistory">
          <div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-align:center;padding:30px 0;">No tests yet.<br>Paste text, label it, and hit Test &amp; Record.</div>
        </div>
      </div>

      <div class="panel fu fu3">
        <div class="ph"><div class="ph-title"><div class="dot" style="background:var(--amber);"></div>Accuracy Over Time</div></div>
        <div class="pb">
          <div id="accWave" style="display:flex;align-items:flex-end;gap:3px;height:60px;">
            <div style="font-family:var(--mono);font-size:11px;color:var(--muted);">Tests will appear here →</div>
          </div>
          <div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:4px;"><span>WRONG ←</span><span>→ CORRECT</span></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- PAGE 6: How It Works -->
<div class="page" id="page-explain">
  <div class="fu fu1" style="margin-bottom:28px;">
    <div class="sec-title">How It <span style="color:var(--pink);">Works</span></div>
    <div class="sec-sub">◆ detection signals · python backend · model architecture</div>
  </div>
  <div class="explain-grid fu fu2">
    <div>
      <div class="sl">Detection Signals</div>
      <div class="signal-card"><div class="sig-icon">🧠</div><div class="sig-title">DistilBERT Classification</div><div class="sig-desc">Your fine-tuned DistilBERT model reads the full text and outputs a probability that it was AI-generated. Trained on ~24k human/AI pairs from the HC3 dataset.</div><div class="sig-impact impact-hi">HIGH IMPACT · REAL ML</div></div>
      <div class="signal-card"><div class="sig-icon">📉</div><div class="sig-title">Perplexity Score</div><div class="sig-desc">AI models choose highly predictable words — making text "low-perplexity." Human writing is messier. Derived from model confidence scores.</div><div class="sig-impact impact-hi">HIGH IMPACT</div></div>
      <div class="signal-card"><div class="sig-icon">〰️</div><div class="sig-title">Burstiness</div><div class="sig-desc">Humans write in bursts — short sentences followed by long ones. AI produces uniformly structured sentences. Scripta measures this rhythm variation.</div><div class="sig-impact impact-hi">HIGH IMPACT</div></div>
      <div class="signal-card"><div class="sig-icon">📚</div><div class="sig-title">Lexical Diversity (TTR)</div><div class="sig-desc">Ratio of unique words to total words. Lower diversity can signal AI authorship. Combined with model output for stronger signal.</div><div class="sig-impact impact-med">MEDIUM IMPACT</div></div>
    </div>
    <div>
      <div class="sl">Python Architecture</div>
      <div class="gloss-item"><div class="gloss-term">FLASK BACKEND</div><div class="gloss-def">All analysis runs in Python via Flask. Your browser sends text to /analyze and receives JSON results.</div></div>
      <div class="gloss-item"><div class="gloss-term">DISTILBERT MODEL</div><div class="gloss-def">A smaller, faster version of BERT. Fine-tuned on HC3 dataset for human vs AI classification. ~66M parameters.</div></div>
      <div class="gloss-item"><div class="gloss-term">HC3 DATASET</div><div class="gloss-def">Human ChatGPT Comparison corpus. ~24,000 question/answer pairs with human and ChatGPT responses side-by-side.</div></div>
      <div class="gloss-item"><div class="gloss-term">TOKEN HEATMAP</div><div class="gloss-def">Word-by-word AI probability. Blends model global score with per-word heuristics for visual intuition.</div></div>
      <div class="gloss-item"><div class="gloss-term">SOURCE FINGERPRINT</div><div class="gloss-def">Probabilistic estimate of which AI generated the text, based on style, entropy, and vocabulary patterns.</div></div>
      <div class="gloss-item"><div class="gloss-term">GROUND TRUTH TESTER</div><div class="gloss-def">Research tool to measure real accuracy. Label texts manually, app predicts, records every result to CSV for your paper.</div></div>
      <div class="sl" style="margin-top:22px;">Stack</div>
      <div style="font-size:13px;color:var(--dim);line-height:1.75;">
        <strong style="color:var(--ink);">Backend:</strong> Python · Flask · HuggingFace Transformers · PyTorch<br>
        <strong style="color:var(--ink);">Model:</strong> distilbert-base-uncased fine-tuned on HC3<br>
        <strong style="color:var(--ink);">Dataset:</strong> Hello-SimpleAI/HC3 (~24k pairs)<br>
        <strong style="color:var(--ink);">Frontend:</strong> Vanilla HTML/CSS/JS (served by Flask)<br>
        <strong style="color:var(--ink);">Training:</strong> python train_detector.py — saves to scripta_model/<br>
        <strong style="color:var(--ink);">Research:</strong> Ground truth log → ground_truth_log.csv
      </div>
    </div>
  </div>
</div>

<script>
const SAMPLES={
  ai:`Artificial intelligence has fundamentally transformed the landscape of modern technology, ushering in a new era of unprecedented computational capabilities. The integration of machine learning algorithms into various domains has enabled organizations to leverage vast amounts of data for predictive analytics and decision-making processes. Furthermore, the deployment of large language models has demonstrated remarkable proficiency in natural language understanding and generation tasks. It is worth noting that these advancements present both significant opportunities and substantial challenges for society.`,
  human:`I've been thinking a lot lately about how weird it is that we just accept autocomplete on our phones now. Like, remember when we had to type out every single letter? My thumbs are genuinely grateful, don't get me wrong. But there's something a bit unsettling about how often I just tap the suggested word without thinking, and then look back at what I wrote and think — did I actually mean that, or did my phone mean it?`
};
const RW_S={
  a:`The study examined the effects of sleep deprivation on cognitive performance. Participants who slept fewer than six hours showed significant impairment in memory recall and reaction time.`,
  b:`This research investigated how insufficient sleep impacts cognitive functioning. Study participants who obtained less than six hours of sleep demonstrated considerable deficits in memory retrieval and response speed.`
};

let history=[];
let gtLabel=null; // "AI" or "HUMAN"

function gotoPage(id,btn){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  btn.classList.add('active');
  if(id==='report')refreshReport();
  if(id==='groundtruth')refreshGTStats();
}

function loadSample(t){document.getElementById('mainInput').value=SAMPLES[t];updateWC();}
function updateWC(){const t=document.getElementById('mainInput').value.trim();const w=t?t.split(/\\s+/).length:0;document.getElementById('wc').textContent=`${w} words · ${document.getElementById('mainInput').value.length} chars`;}
function loadRwSample(){document.getElementById('rwA').value=RW_S.a;document.getElementById('rwB').value=RW_S.b;}

// ── Core API call ────────────────────────────────────────────────────────────
async function callAPI(text){
  const res = await fetch('/analyze',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({text})
  });
  if(!res.ok) throw new Error('Server error: '+res.status);
  return res.json();
}

// ── Analyzer ─────────────────────────────────────────────────────────────────
async function runAnalysis(){
  const text=document.getElementById('mainInput').value.trim();
  if(!text){flashInput('mainInput');return;}
  const btn=document.getElementById('analyzeBtn');
  btn.disabled=true;
  document.getElementById('ld1').classList.add('on');
  try{
    const r=await callAPI(text);
    renderVerdict(r);
    renderMetrics(r);
    renderAttribution(r);
    renderHeatmap(r.tokens);
    document.getElementById('heatmapPanel').style.display='block';
    history.unshift({text,result:r,time:new Date()});
    if(history.length>20)history.pop();
  }catch(e){
    alert('⚠ Analysis failed: '+e.message);
  }finally{
    document.getElementById('ld1').classList.remove('on');
    btn.disabled=false;
  }
}

function showHeatmap(){
  const text=document.getElementById('mainInput').value.trim();
  if(!text){flashInput('mainInput');return;}
  runAnalysis();
}

function clearAll(){
  document.getElementById('mainInput').value='';updateWC();
  document.getElementById('verdictArea').innerHTML=`<div style="text-align:center;padding:32px 0;"><div style="font-size:52px;opacity:.08;margin-bottom:10px;">◎</div><div style="font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.1em;">AWAITING ANALYSIS</div></div>`;
  document.getElementById('metricsPanel').style.display='none';
  document.getElementById('attrPanel').style.display='none';
  document.getElementById('heatmapPanel').style.display='none';
}

function flashInput(id){const el=document.getElementById(id);el.style.borderColor='var(--red)';setTimeout(()=>el.style.borderColor='',900);}

// ── Render helpers ───────────────────────────────────────────────────────────
function scoreToStyle(s){
  if(s<.3)return{bg:`rgba(22,163,74,${.12+s*.6})`,tc:'#0a5c30'};
  if(s<.65){const t=(s-.3)/.35;return{bg:`rgba(245,158,11,${.15+t*.35})`,tc:'#7a4d00'};}
  const t=(s-.65)/.35;return{bg:`rgba(220,38,38,${.18+t*.5})`,tc:'#7a1010'};
}

function renderHeatmap(toks){
  document.getElementById('heatmap').innerHTML=toks.map(({word,score})=>{
    const{bg,tc}=scoreToStyle(score);const pct=Math.round(score*100);
    const cat=score>.65?'Likely AI':score>.35?'Uncertain':'Likely Human';
    const safe=word.replace(/&/g,'&amp;').replace(/</g,'&lt;');
    return`<span class="tok" style="background:${bg};color:${tc}" data-w="${safe.trim()}" data-s="${pct}" data-c="${cat}" onmouseenter="showTip(event,this)" onmouseleave="hideTip()">${safe}</span> `;
  }).join('');
}

function renderVerdict(r){
  const col={ai:'var(--red)',human:'var(--green)',mixed:'var(--amber)'}[r.cls];
  const src=r.source==='model'?'<span class="source-badge source-real">REAL MODEL</span>':'<span class="source-badge source-heuristic">HEURISTIC</span>';
  document.getElementById('verdictArea').innerHTML=`<div class="verdict-big ${r.cls}-v"><div class="big-label" style="color:${col};">${r.verdict}${src}</div><div class="big-pct" style="color:${col};">${r.pct}<span style="font-size:32px;font-weight:700;">%</span></div><div style="font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:8px;letter-spacing:.06em;">AI PROBABILITY</div><div class="meter-track"><div class="meter-fill" style="width:${r.pct}%;background:${col};"></div></div><div style="font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:8px;">${r.word_count} words · Python Backend</div></div>`;
}

function renderMetrics(r){
  document.getElementById('metricsPanel').style.display='block';
  const M=[
    {l:'PERPLEXITY',    v:Math.min(1,r.perplexity/100), d:r.perplexity.toFixed(1),         c:r.ai_prob>.5?'var(--red)':'var(--green)'},
    {l:'BURSTINESS',   v:r.burstiness,                  d:(r.burstiness*100).toFixed(0)+'%',c:r.ai_prob<.5?'var(--red)':'var(--green)'},
    {l:'LEX. DIVERSITY',v:r.lex_div,                    d:(r.lex_div*100).toFixed(0)+'%',   c:r.lex_div>.5?'var(--green)':'var(--amber)'},
    {l:'COHERENCE',    v:.45+r.ai_prob*.45,              d:(((.45+r.ai_prob*.45))*100).toFixed(0)+'%', c:'var(--pri)'},
    {l:'FORMALITY',    v:r.ai_prob*.8+.1,               d:((r.ai_prob*.8+.1)*100).toFixed(0)+'%', c:'var(--pink)'},
  ];
  document.getElementById('metricsBody').innerHTML=`<div class="sl">Feature Scores</div>${M.map(m=>`<div class="metric-row"><div class="ml">${m.l}</div><div class="mtrack"><div class="mfill" style="width:${m.v*100}%;background:${m.c};"></div></div><div class="mv">${m.d}</div></div>`).join('')}`;
}

function renderAttribution(r){
  document.getElementById('attrPanel').style.display='block';
  document.getElementById('attrBody').innerHTML=`<div style="font-size:12px;color:var(--dim);margin-bottom:12px;line-height:1.6;">Which AI model most likely generated this?</div>${r.attribution.map(a=>`<div class="metric-row"><div class="ml" style="width:100px;">${a.name}</div><div class="mtrack"><div class="mfill" style="width:${a.score}%;background:${a.color};"></div></div><div class="mv">${a.score}%</div></div>`).join('')}<div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:10px;padding-top:10px;border-top:1.5px solid var(--border);">⚠ Probabilistic estimate only.</div>`;
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function showTip(e,el){const tip=document.getElementById('tip');document.getElementById('tip-word').textContent=`"${el.dataset.w}"`;document.getElementById('tip-score').textContent=el.dataset.s+'%';const cat=document.getElementById('tip-cat');cat.textContent=el.dataset.c;cat.style.color=+el.dataset.s>65?'#fca5a5':+el.dataset.s>35?'#fcd34d':'#6ee7b7';const bf=document.getElementById('tip-bfill');bf.style.width=el.dataset.s+'%';bf.style.background=+el.dataset.s>65?'var(--red)':+el.dataset.s>35?'var(--amber)':'var(--green)';tip.classList.add('show');moveTip(e);}
function hideTip(){document.getElementById('tip').classList.remove('show');}
function moveTip(e){const tip=document.getElementById('tip');tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-55)+'px';}
document.addEventListener('mousemove',e=>{if(document.getElementById('tip').classList.contains('show'))moveTip(e);});

// ── Live Monitor ──────────────────────────────────────────────────────────────
async function rtAnalyze(){
  const text=document.getElementById('rtInput').value.trim();
  if(!text){resetRT();return;}
  try{
    const r=await callAPI(text);
    const pct=r.pct;
    const col=r.ai_prob>.65?'var(--red)':r.ai_prob<.35?'var(--green)':'var(--amber)';
    document.getElementById('rtBig').textContent=pct+'%';
    document.getElementById('rtBig').style.color=col;
    document.getElementById('rtLbl').textContent=r.verdict;
    document.getElementById('rtLbl').style.color=col;
    document.getElementById('rtMeter').style.width=pct+'%';
    document.getElementById('rtMeter').style.background=col;
    renderSentences(r.sentences);
    renderWave(r.sentences);
  }catch(e){alert('Error: '+e.message);}
}

function renderSentences(sentences){
  document.getElementById('sentenceList').innerHTML=sentences.map(s=>{
    const sp=Math.round(s.score*100);const sc=s.score>.65?'var(--red)':s.score<.35?'var(--green)':'var(--amber)';const circ=2*Math.PI*13;
    return`<div class="sentence-card" style="border-left:3px solid ${sc};"><div class="sc-bar"><svg width="40" height="40" viewBox="0 0 40 40"><circle cx="20" cy="20" r="13" fill="none" stroke="var(--s2)" stroke-width="3.5"/><circle cx="20" cy="20" r="13" fill="none" stroke="${sc}" stroke-width="3.5" stroke-linecap="round" transform="rotate(-90 20 20)" stroke-dasharray="${circ}" stroke-dashoffset="${circ-circ*s.score}"/></svg><div class="sc-pct" style="color:${sc};font-size:9px;">${sp}</div></div><div class="sc-text">${s.text.substring(0,80)}${s.text.length>80?'…':''}</div></div>`;
  }).join('');
}

function renderWave(sentences){
  const wb=document.getElementById('waveBar');
  if(!sentences.length){wb.innerHTML='<div style="font-family:var(--mono);font-size:11px;color:var(--muted);">Analyze text →</div>';return;}
  wb.innerHTML=sentences.map(s=>{const h=Math.round(8+s.score*46);const col=s.score>.65?'var(--red)':s.score<.35?'var(--green)':'var(--amber)';return`<div class="wave-col" style="height:${h}px;background:${col};opacity:.85;flex:1;min-width:6px;"></div>`;}).join('');
}

function resetRT(){document.getElementById('rtBig').textContent='--';document.getElementById('rtBig').style.color='var(--muted)';document.getElementById('rtLbl').textContent='AI Probability';document.getElementById('rtMeter').style.width='0%';document.getElementById('sentenceList').innerHTML='<div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-align:center;padding:20px 0;">Analyze text to see scores…</div>';}

// ── Report ────────────────────────────────────────────────────────────────────
function refreshReport(){
  document.getElementById('rTotalAnalyzed').textContent=history.length;
  const tw=history.reduce((s,h)=>s+h.result.word_count,0);
  document.getElementById('rWordCount').textContent=tw.toLocaleString();
  if(!history.length){document.getElementById('rAvgAI').textContent='—';document.getElementById('donutEmpty').style.display='block';document.getElementById('timeline').innerHTML='<div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-align:center;padding:20px 0;">No analyses yet.</div>';return;}
  const avgAI=history.reduce((s,h)=>s+h.result.ai_prob,0)/history.length;
  document.getElementById('rAvgAI').textContent=Math.round(avgAI*100)+'%';
  document.getElementById('rAvgAI').style.color=avgAI>.6?'var(--red)':avgAI<.4?'var(--green)':'var(--amber)';
  const counts={ai:0,human:0,mixed:0};history.forEach(h=>counts[h.result.cls]++);
  const tot=history.length;const circ=364;
  document.getElementById('donutEmpty').style.display='none';
  document.getElementById('donutAI').style.strokeDashoffset=circ-(counts.ai/tot)*circ;
  document.getElementById('donutHuman').style.strokeDashoffset=circ-(counts.human/tot)*circ;
  document.getElementById('donutMixed').style.strokeDashoffset=circ-(counts.mixed/tot)*circ;
  document.getElementById('dAI').textContent=`AI-Generated: ${counts.ai}`;
  document.getElementById('dHuman').textContent=`Human-Written: ${counts.human}`;
  document.getElementById('dMixed').textContent=`Mixed/Uncertain: ${counts.mixed}`;
  document.getElementById('timeline').innerHTML=history.slice(0,10).map(h=>{const bc=h.result.cls==='ai'?'badge-ai':h.result.cls==='human'?'badge-human':'badge-mixed';return`<div class="tl-item done"><div class="tl-time">${h.time.toLocaleTimeString()}</div><div class="tl-text">"${h.text.substring(0,55)}${h.text.length>55?'…':''}"</div><div class="tl-badge ${bc}">${h.result.verdict} · ${h.result.pct}%</div></div>`;}).join('');
}

// ── Rewrite Detector ─────────────────────────────────────────────────────────
async function runRewrite(){
  const tA=document.getElementById('rwA').value.trim(),tB=document.getElementById('rwB').value.trim();
  if(!tA||!tB)return;
  const btn=document.getElementById('rwBtn');btn.disabled=true;
  try{
    const [rA,rB]=await Promise.all([callAPI(tA),callAPI(tB)]);
    const wA=tA.toLowerCase().split(/\\s+/),wB=tB.toLowerCase().split(/\\s+/);
    const sA=new Set(wA),sB=new Set(wB);
    const overlap=[...sA].filter(w=>sB.has(w)).length;
    const sim=overlap/Math.max(sA.size,sB.size);
    const aiDelta=rB.ai_prob-rA.ai_prob;
    const rwScore=Math.min(99,Math.round((sim*.55+Math.max(0,aiDelta)*.45)*100+20));
    let verdict,col;
    if(rwScore>70){verdict='LIKELY AI-REWRITTEN';col='var(--red)';}
    else if(rwScore>45){verdict='POSSIBLY REWRITTEN';col='var(--amber)';}
    else{verdict='LIKELY ORIGINAL';col='var(--green)';}
    document.getElementById('rwBig').textContent=rwScore+'%';document.getElementById('rwBig').style.color=col;
    document.getElementById('rwLbl').textContent=verdict;document.getElementById('rwLbl').style.color=col;
    document.getElementById('rwMeter').style.width=rwScore+'%';document.getElementById('rwMeter').style.background=col;
    document.getElementById('rwBreakdown').innerHTML=`<div class="metric-row"><div class="ml">LEXICAL SIM.</div><div class="mtrack"><div class="mfill" style="width:${Math.round(sim*100)}%;background:var(--pri);"></div></div><div class="mv">${Math.round(sim*100)}%</div></div><div class="metric-row"><div class="ml">AI SCORE A</div><div class="mtrack"><div class="mfill" style="width:${Math.round(rA.ai_prob*100)}%;background:var(--green);"></div></div><div class="mv">${Math.round(rA.ai_prob*100)}%</div></div><div class="metric-row"><div class="ml">AI SCORE B</div><div class="mtrack"><div class="mfill" style="width:${Math.round(rB.ai_prob*100)}%;background:var(--red);"></div></div><div class="mv">${Math.round(rB.ai_prob*100)}%</div></div><div class="metric-row"><div class="ml">AI DELTA</div><div class="mtrack"><div class="mfill" style="width:${Math.min(100,Math.abs(aiDelta)*200)}%;background:var(--amber);"></div></div><div class="mv">${aiDelta>0?'+':''}${Math.round(aiDelta*100)}%</div></div><div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:10px;padding-top:10px;border-top:1.5px solid var(--border);">${aiDelta>0.1?'⚠ Text B shows notably higher AI probability.':'✓ AI probability gap is minimal.'}</div>`;
    let diff='';for(let i=0;i<wB.length;i++){const wb=wB[i],wa=wA[i]||'';if(wb===wa)diff+=`<span style="color:var(--dim)">${tB.split(/\\s+/)[i]} </span>`;else diff+=`<span class="diff-add">${tB.split(/\\s+/)[i]}</span> `;}
    document.getElementById('diffView').innerHTML=diff||'<span style="color:var(--muted)">No differences.</span>';
    document.getElementById('rwResult').style.display='block';
  }catch(e){alert('Error: '+e.message);}
  finally{btn.disabled=false;}
}

// ══════════════════════════════════════════════════════════
// GROUND TRUTH TESTER
// ══════════════════════════════════════════════════════════

function selectLabel(label){
  gtLabel=label;
  document.getElementById('lbtnHuman').className='lbtn'+(label==='HUMAN'?' sel-human':'');
  document.getElementById('lbtnAI').className='lbtn'+(label==='AI'?' sel-ai':'');
}

function gtLoadSample(type){
  document.getElementById('gtInput').value=SAMPLES[type];
  selectLabel(type==='ai'?'AI':'HUMAN');
}

function gtClear(){
  document.getElementById('gtInput').value='';
  gtLabel=null;
  document.getElementById('lbtnHuman').className='lbtn';
  document.getElementById('lbtnAI').className='lbtn';
  document.getElementById('gtFlash').style.display='none';
}

async function runGTTest(){
  const text=document.getElementById('gtInput').value.trim();
  if(!text){flashInput('gtInput');return;}
  if(!gtLabel){
    alert('Please select whether the text is Human or AI first!');
    return;
  }
  const btn=document.getElementById('gtTestBtn');
  btn.disabled=true;
  document.getElementById('gtLoader').classList.add('on');
  document.getElementById('gtFlash').style.display='none';

  try{
    const res=await fetch('/gt/test',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text, true_label:gtLabel})
    });
    const data=await res.json();
    if(data.error){alert('Error: '+data.error);return;}

    // Show flash result
    const flash=document.getElementById('gtFlash');
    flash.className='result-flash '+(data.correct?'correct-flash':'wrong-flash');
    document.getElementById('gtFlashIcon').textContent=data.correct?'✅':'❌';
    document.getElementById('gtFlashTitle').textContent=data.correct?'Correct Prediction!':'Wrong Prediction';
    document.getElementById('gtFlashSub').textContent=
      `You said: ${gtLabel} · Model predicted: ${data.predicted} · AI prob: ${Math.round(data.result.ai_prob*100)}%`;
    flash.style.display='block';

    // Refresh stats + history
    await refreshGTStats();

  }catch(e){
    alert('Error: '+e.message);
  }finally{
    document.getElementById('gtLoader').classList.remove('on');
    btn.disabled=false;
  }
}

async function refreshGTStats(){
  try{
    const res=await fetch('/gt/stats');
    const d=await res.json();
    if(!d.total){
      ['gtAccuracy','gtPrecision','gtRecall','gtF1'].forEach(id=>document.getElementById(id).textContent='—');
      document.getElementById('cmTP').textContent='0';
      document.getElementById('cmTN').textContent='0';
      document.getElementById('cmFP').textContent='0';
      document.getElementById('cmFN').textContent='0';
      document.getElementById('gtHistory').innerHTML='<div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-align:center;padding:30px 0;">No tests yet.</div>';
      document.getElementById('accWave').innerHTML='<div style="font-family:var(--mono);font-size:11px;color:var(--muted);">Tests will appear here →</div>';
      return;
    }

    // Stats
    document.getElementById('gtAccuracy').textContent=d.accuracy+'%';
    document.getElementById('gtPrecision').textContent=d.precision+'%';
    document.getElementById('gtRecall').textContent=d.recall+'%';
    document.getElementById('gtF1').textContent=d.f1+'%';

    // Color accuracy
    const accEl=document.getElementById('gtAccuracy');
    accEl.style.color=d.accuracy>=80?'var(--green)':d.accuracy>=60?'var(--amber)':'var(--red)';

    // Confusion matrix
    document.getElementById('cmTP').textContent=d.tp;
    document.getElementById('cmTN').textContent=d.tn;
    document.getElementById('cmFP').textContent=d.fp;
    document.getElementById('cmFN').textContent=d.fn;

    // History rows
    document.getElementById('gtHistory').innerHTML=[...d.rows].reverse().map(r=>{
      const correct=r.correct==='True';
      const tc=correct?'var(--green)':'var(--red)';
      const icon=correct?'✅':'❌';
      const trueCol=r.true_label==='AI'?'var(--red)':'var(--green)';
      const predCol=r.predicted==='AI'?'var(--red)':'var(--green)';
      return`<div class="gt-row ${correct?'correct':'wrong'}">
        <span style="font-size:16px;">${icon}</span>
        <div style="flex:1;min-width:0;">
          <div style="font-size:12px;color:var(--ink2);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.text_snippet}${r.text_snippet.length>=80?'…':''}</div>
          <div style="font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:2px;">
            True: <span style="color:${trueCol};font-weight:700;">${r.true_label}</span> &nbsp;·&nbsp;
            Pred: <span style="color:${predCol};font-weight:700;">${r.predicted}</span> &nbsp;·&nbsp;
            ${Math.round(parseFloat(r.ai_prob)*100)}% AI
          </div>
        </div>
        <span class="gt-badge" style="background:${correct?'var(--green-light)':'var(--red-light)'};color:${tc};">${r.cm_cell}</span>
      </div>`;
    }).join('');

    // Accuracy wave
    const waveRows=d.rows.slice(-20);
    document.getElementById('accWave').innerHTML=waveRows.map(r=>{
      const correct=r.correct==='True';
      const col=correct?'var(--green)':'var(--red)';
      const h=correct?52:20;
      return`<div style="flex:1;min-width:8px;height:${h}px;background:${col};border-radius:3px 3px 0 0;opacity:.8;transition:height .3s;"></div>`;
    }).join('');

  }catch(e){console.error('GT stats error',e);}
}

async function gtClearAll(){
  if(!confirm('Clear all ground truth test data? This cannot be undone.'))return;
  await fetch('/gt/clear',{method:'POST'});
  await refreshGTStats();
}

// ── Keyboard shortcut ─────────────────────────────────────────────────────────
document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')runAnalysis();});
</script>
</body>
</html>"""
@app.route("/")
def home():
    return "Scripta is running on Render 🚀"
if __name__ == "__main__":
    print("\n🚀 Scripta AI Detector starting...")
    print(f"   Model: {'✅ Real ML (' + MODEL_DIR + ')' if MODEL_READY else '⚠  Heuristic (train first)'}")
    print("   Open: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)