"""Build a small sentiment dataset by extracting positive/negative sentences from the corpus."""
import random
import json
import os

CLEAN_FILE = "data/cleaned/kannada_clean.txt"
OUTPUT_FILE = "data/cleaned/kannada_sentiment.jsonl"

positive_words = [
    "ಚೆನ್ನಾಗಿದೆ", "ಅದ್ಭುತ", "ಸಂತೋಷ", "ಇಷ್ಟ", "ಸುಂದರ", "ಉತ್ತಮ", "ಪ್ರೀತಿ",
    "ಅಭಿನಂದನೆ", "ಶ್ರೇಷ್ಠ", "ಆನಂದ", "ಹರ್ಷ", "ಮೆಚ್ಚುಗೆ", "ಧನ್ಯವಾದ", "ಶುಭ",
    "ಗೆಲುವು", "ಯಶಸ್ಸು", "ಹೊಗಳಿಕೆ", "ಸಿಹಿ", "ಬಹುಮಾನ", "ಗೌರವ"
]

negative_words = [
    "ಕೆಟ್ಟ", "ನಿರಾಶೆ", "ದುಃಖ", "ಬೇಡ", "ಕೋಪ", "ಅಸಮಾಧಾನ", "ಕಿರಿಕಿರಿ",
    "ಹಾಳು", "ನಷ್ಟ", "ಸೋಲು", "ಭಯ", "ದ್ವೇಷ", "ಕ್ರೌರ್ಯ", "ದುರಂತ",
    "ಬೇಸರ", "ಅವಮಾನ", "ಕಷ್ಟ", "ಹಿಂಸೆ", "ದೂರು", "ಶೋಕ"
]

def has_words(text, word_list):
    return any(w in text for w in word_list)

def has_no_words(text, word_list):
    return not any(w in text for w in word_list)

print("Loading corpus...")
with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

print(f"Total sentences: {len(lines):,}")

positives = []
negatives = []

for line in lines:
    if len(line) < 20 or len(line) > 200:
        continue
    if has_words(line, positive_words) and has_no_words(line, negative_words):
        positives.append(line)
    elif has_words(line, negative_words) and has_no_words(line, positive_words):
        negatives.append(line)

print(f"Positive candidates: {len(positives):,}")
print(f"Negative candidates: {len(negatives):,}")

n = min(len(positives), len(negatives), 500)
random.seed(42)
pos_sample = random.sample(positives, n)
neg_sample = random.sample(negatives, n)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for text in pos_sample:
        f.write(json.dumps({"text": text, "label": 1}, ensure_ascii=False) + '\n')
    for text in neg_sample:
        f.write(json.dumps({"text": text, "label": 0}, ensure_ascii=False) + '\n')

print(f"Saved {n*2} labeled examples to {OUTPUT_FILE}")
print(f"  Positive: {n}")
print(f"  Negative: {n}")