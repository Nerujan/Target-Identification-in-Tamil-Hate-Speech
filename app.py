from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import re
import requests

app = Flask(__name__, template_folder="templates", static_folder="static")

HATE_MODEL_PATH   = r"C://Users//satha//Desktop//Model-V4//MURIL_HATE//best_model-20251030T082528Z-1-001//best_model"
NER_MODEL_PATH    = r"C://Users//satha//Desktop//Model-V4//XLMR_NEW"
TARGET_MODEL_PATH = r"C://Users//satha//Desktop//Model-V4//MURIL_Target//muril-entity-Model-26OCT-20251025T211633Z-1-001"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def safe_load_seq_model(path_or_name):
    tok = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModelForSequenceClassification.from_pretrained(path_or_name).to(device)
    model.eval()
    return tok, model

def safe_load_token_model(path_or_name, fallback="xlm-roberta-base"):
    try:
        tok = AutoTokenizer.from_pretrained(path_or_name)
        model = AutoModelForTokenClassification.from_pretrained(path_or_name).to(device)
        model.eval()
        return tok, model
    except Exception as e:
        app.logger.warning("Failed to load local NER model (%s). Falling back to %s. Error: %s", path_or_name, fallback, e)
        tok = AutoTokenizer.from_pretrained(fallback)
        model = AutoModelForTokenClassification.from_pretrained(fallback).to(device)
        model.eval()
        return tok, model

hate_tokenizer, hate_model = safe_load_seq_model(HATE_MODEL_PATH)
ner_tokenizer, ner_model   = safe_load_token_model(NER_MODEL_PATH, fallback="xlm-roberta-base")
target_tokenizer, target_model = safe_load_seq_model(TARGET_MODEL_PATH)

_raw_hate_map = getattr(hate_model.config, "id2label", None)
HATE_ID2LABEL = {int(k): v for k, v in _raw_hate_map.items()} if _raw_hate_map else None
_raw_target_map = getattr(target_model.config, "id2label", None)
TARGET_ID2LABEL = {int(k): v for k, v in _raw_target_map.items()} if _raw_target_map else None
_raw_ner_map = getattr(ner_model.config, "id2label", None)
NER_ID2LABEL = {int(k): v for k, v in _raw_ner_map.items()} if _raw_ner_map else None

HATE_THRESHOLD = 0.5

RE_TAMIL_CHAR = re.compile(r"[\u0B80-\u0BFF]")

def tanglish_to_tamil_google(text):
    url = "https://inputtools.google.com/request"
    payload = {"text": text, "itc": "ta-t-i0-und", "num": 1}
    response = requests.post(url, data=payload)
    try:
        data = response.json()
        return data[1][0][1][0]
    except Exception:
        return text

def normalize_tanglish_mixed(text):
    tokens = text.split()
    normalized_tokens = []
    for tok in tokens:
        if RE_TAMIL_CHAR.search(tok):
            normalized_tokens.append(tok)
        else:
            normalized_tokens.append(tanglish_to_tamil_google(tok))
    return " ".join(normalized_tokens)

def detect_hate(text):
    inputs = hate_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outs = hate_model(**inputs)
    logits = outs.logits[0]
    probs = F.softmax(logits, dim=-1).cpu().tolist()
    hate_index = 1
    pred_idx = int(torch.argmax(logits).item())
    pred_label = HATE_ID2LABEL.get(pred_idx, f"LABEL_{pred_idx}") if HATE_ID2LABEL else str(pred_idx)
    return {"hate_prob": float(probs[hate_index]), "pred_index": pred_idx, "pred_label": pred_label, "probs": [round(p,4) for p in probs]}

def run_ner(text):
    inputs = ner_tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True).to(device)
    offset_mapping = inputs.pop("offset_mapping")[0].cpu()
    with torch.no_grad():
        outs = ner_model(**inputs)
    probs = F.softmax(outs.logits, dim=-1).cpu()
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    id2label = ner_model.config.id2label
    results = []
    current_word = ""
    current_start = None
    current_end = None
    current_entity = None
    current_scores = None
    for i, (token, offset) in enumerate(zip(tokens, offset_mapping)):
        if token in ner_tokenizer.all_special_tokens:
            continue
        start, end = int(offset[0]), int(offset[1])
        if start == end:
            continue
        token_text = text[start:end]
        prob_vector = probs[0, i]
        grouped = {"COMM": 0.0, "PER": 0.0, "ORG": 0.0, "O": 0.0}
        for idx, score in enumerate(prob_vector):
            label = id2label[str(idx)] if isinstance(list(id2label.keys())[0], str) else id2label[idx]
            if label == "O":
                grouped["O"] = float(score)
            elif label.startswith("B-") or label.startswith("I-"):
                ent = label.split("-")[1]
                grouped[ent] = grouped.get(ent, 0.0) + float(score)
        sorted_scores = sorted(grouped.items(), key=lambda x: x[1], reverse=True)
        top_entity = sorted_scores[0][0]
        if token.startswith("▁") or not current_word:
            if current_word:
                results.append({"token": current_word, "start": current_start, "end": current_end, "entity": current_entity, "scores": {k: round(v,4) for k,v in current_scores.items()}})
            current_word = token_text.lstrip("▁")
            current_start = start
            current_end = end
            current_entity = top_entity
            current_scores = grouped
        else:
            current_word += token_text
            current_end = end
    if current_word:
        results.append({"token": current_word, "start": current_start, "end": current_end, "entity": current_entity, "scores": {k: round(v,4) for k,v in current_scores.items()}})
    return [r for r in results if r["entity"] != "O"]

def classify_entity_target(entity_text, full_text, threshold=0.5):
    """
    Lower threshold to 0.3 for short/colloquial sentences.
    """
    input_text = f"{entity_text} ||| {full_text}"
    inputs = target_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outs = target_model(**inputs)
    probs = F.softmax(outs.logits, dim=-1)[0].cpu().tolist()
    target_index = 1
    target_prob = probs[target_index]
    pred_label = 1 if target_prob >= threshold else 0
    return {"prob": float(target_prob), "label": int(pred_label)}

def human_readable_summary(obj):
    lines = []
    if obj.get("hate"):
        lines.append(f'🔥 <strong>HATE SPEECH</strong> — confidence <strong>{obj["hate_prob"]*100:.1f}%</strong>')
    else:
        lines.append(f'✅ <strong>NOT hate speech</strong> — confidence <strong>{(1-obj["hate_prob"])*100:.1f}%</strong>')
    entities = obj.get("entities", [])
    if entities:
        tlist = [f'<span class="badge">{e["token"]}</span>' for e in entities if e.get("target_label")==1]
        if tlist:
            lines.append("🎯 Targets: " + " ".join(tlist))
        else:
            lines.append("⚪ No target entities detected.")
    return "<br/>".join(lines)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error":"No text provided"}), 400

    hate_res = detect_hate(text)
    hate_prob = hate_res["hate_prob"]
    is_hate = hate_prob >= HATE_THRESHOLD
    response = {"hate": bool(is_hate), "hate_prob": round(hate_prob,4), "hate_debug": hate_res}

    if is_hate:
        normalized_text = normalize_tanglish_mixed(text)
        response["translated_text"] = normalized_text
        response["translation_performed"] = True
        response["normalize_method"] = "google-tanglish"

        entities_raw = run_ner(normalized_text)
        response["entities_raw"] = entities_raw

        classified = []
        for ent in entities_raw:
            ent_text = ent["token"]
            t = classify_entity_target(ent_text, normalized_text, threshold=0.5)  
            classified.append({
                "token": ent_text,
                "start": ent["start"],
                "end": ent["end"],
                "entity_type": ent["entity"],
                "ner_scores": ent["scores"],
                "target_prob": round(t["prob"],4),
                "target_label": t["label"]
            })
        response["entities"] = classified
    else:
        response["translated_text"] = text
        response["translation_performed"] = False
        response["normalize_method"] = None
        response["entities_raw"] = []
        response["entities"] = []

    response["summary_html"] = human_readable_summary(response)
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
