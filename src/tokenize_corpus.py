"""Tokenize corpus — balanced speed and RAM, with visible progress."""
import os
import gc
import shutil
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(PROJECT_ROOT, ".cache", "datasets")

TOKENIZER_DIR = "tokenizer"
CLEANED_DIR = "data/cleaned"
TOKENIZED_DIR = "data/tokenized"
CLEAN_FILE = os.path.join(CLEANED_DIR, "kannada_clean.txt")
BLOCK_SIZE = 512
BATCH_SIZE = 1_000_000  # Smaller batches for more frequent progress updates

# Clean previous
for d in ["data/tokenized/train", "data/tokenized/val"]:
    if os.path.exists(d):
        shutil.rmtree(d)

os.makedirs(os.path.join(TOKENIZED_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(TOKENIZED_DIR, "val"), exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

print("Loading text...")
with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
    all_lines = [l.strip() for l in f.readlines() if l.strip() and len(l.strip()) >= 10]

print(f"Total sentences: {len(all_lines):,}")

all_blocks = []
total_batches = (len(all_lines) + BATCH_SIZE - 1) // BATCH_SIZE

pbar = tqdm(total=len(all_lines), desc="Tokenizing", unit=" lines", unit_scale=True)

for batch_idx in range(total_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(all_lines))
    batch = all_lines[start:end]
    
    # Tokenize this batch
    encodings = tokenizer(
        batch,
        truncation=True,
        max_length=BLOCK_SIZE,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    
    # Flatten
    ids = []
    for x in encodings["input_ids"]:
        ids.extend(x)
    
    # Chunk into blocks
    for i in range(0, len(ids) - BLOCK_SIZE + 1, BLOCK_SIZE):
        block = ids[i : i + BLOCK_SIZE]
        all_blocks.append({"input_ids": block, "labels": block})
    
    pbar.update(len(batch))
    pbar.set_postfix(blocks=len(all_blocks))
    
    del batch, encodings, ids
    gc.collect()

pbar.close()
del all_lines
gc.collect()

print(f"\nTotal blocks: {len(all_blocks):,}")

# Dataset, shuffle, split
print("Creating dataset...")
dataset = Dataset.from_list(all_blocks)
del all_blocks
gc.collect()

print("Shuffling...")
dataset = dataset.shuffle(seed=42)

print("Splitting...")
split = dataset.train_test_split(test_size=0.01, seed=42)
print(f"Train: {len(split['train']):,}  |  Val: {len(split['test']):,}")

print("Saving train...")
split["train"].save_to_disk(os.path.join(TOKENIZED_DIR, "train"))
print("Saving val...")
split["test"].save_to_disk(os.path.join(TOKENIZED_DIR, "val"))
print("Done.")