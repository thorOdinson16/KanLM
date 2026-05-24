"""Evaluate sentiment classification using frozen LM features."""
import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from config import CHECKPOINT_DIR
import os

jsonl_path = "data/cleaned/kannada_sentiment.jsonl"

if os.path.exists(jsonl_path):
    texts, labels = [], []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(obj["label"])
    print(f"Loaded {len(texts)} labeled examples from {jsonl_path}")
else:
    print(f"{jsonl_path} not found! Run build_sentiment_data.py first.")
    exit(1)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

model_path = f"{CHECKPOINT_DIR}/final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
model.config.output_hidden_states = True
model.eval()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.hidden_states[-1]
        mean_emb = last_hidden.mean(dim=1).cpu().numpy()
    return mean_emb[0]

print("Extracting features...")
X_train_emb = np.array([get_embedding(t) for t in X_train])
X_test_emb = np.array([get_embedding(t) for t in X_test])

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_emb, y_train)
y_pred = clf.predict(X_test_emb)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])

print("\nSentiment Classification Results:")
print(f"Accuracy: {acc:.4f}")
print(f"F1 (macro): {f1:.4f}")
print("\nClassification Report:")
print(report)

results = {
    "accuracy": acc,
    "f1_macro": f1,
    "num_train": len(X_train),
    "num_test": len(X_test),
    "note": "Labels derived from keyword matching on corpus sentences"
}
os.makedirs("eval", exist_ok=True)
with open("eval/sentiment_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to eval/sentiment_results.json")