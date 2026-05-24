"""Analyze tokenizer fertility vs multilingual tokenizers."""
import os
import random
from transformers import AutoTokenizer
from tqdm import tqdm

CLEANED_DIR = "data/cleaned"
TOKENIZER_DIR = "tokenizer"
CLEAN_FILE = os.path.join(CLEANED_DIR, "kannada_clean.txt")

# Load your tokenizer
print("Loading your BPE tokenizer...")
your_tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

# Load comparison tokenizers
print("Loading XLM-R tokenizer...")
xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

print("Loading mBERT tokenizer...")
mbert_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Sample test sentences
print("Loading test sentences...")
with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()

# Take 2000 random sentences for testing
random.seed(42)
test_sentences = random.sample(all_lines, min(2000, len(all_lines)))
test_sentences = [s.strip() for s in test_sentences if s.strip()]
print(f"Testing on {len(test_sentences)} sentences\n")

def fertility(tok, sentences):
    total_words = 0
    total_tokens = 0
    for sent in tqdm(sentences, desc=f"  {tok.name_or_path.split('/')[-1]:<30}", leave=False):
        words = sent.split()
        tokens = tok.tokenize(sent)
        total_words += len(words)
        total_tokens += len(tokens)
    return total_tokens / total_words

print("Calculating fertility (tokens per word)...")
fert_yours = fertility(your_tok, test_sentences)
fert_xlmr = fertility(xlmr_tok, test_sentences)
fert_mbert = fertility(mbert_tok, test_sentences)

print(f"\n{'='*50}")
print(f"{'Tokenizer':<30} {'Fertility':>10}")
print(f"{'='*50}")
print(f"{'Your Kannada BPE':<30} {fert_yours:>10.2f}")
print(f"{'XLM-R':<30} {fert_xlmr:>10.2f}")
print(f"{'mBERT':<30} {fert_mbert:>10.2f}")
print(f"{'='*50}")

improvement_xlmr = (fert_xlmr - fert_yours) / fert_xlmr * 100
improvement_mbert = (fert_mbert - fert_yours) / fert_mbert * 100
print(f"\nYour tokenizer is {improvement_xlmr:.1f}% more efficient than XLM-R")
print(f"Your tokenizer is {improvement_mbert:.1f}% more efficient than mBERT")