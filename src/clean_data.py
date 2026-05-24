"""Clean Kannada text: filter non-Kannada, deduplicate, normalize."""
import re
import os
import hashlib
import unicodedata
from tqdm import tqdm

CLEANED_DIR = "data/cleaned"
RAW_FILE = os.path.join(CLEANED_DIR, "kannada_raw.txt")
CLEAN_FILE = os.path.join(CLEANED_DIR, "kannada_clean.txt")

# Kannada Unicode block: U+0C80 to U+0CFF
KANNADA_RE = re.compile(r'[\u0C80-\u0CFF]')

# Characters that should NOT dominate a Kannada sentence
# Latin, digits, common punctuation
NON_KANNADA_RE = re.compile(r'[a-zA-Z0-9]')

def is_valid(line):
    line = line.strip()
    
    # Length filter
    if len(line) < 10 or len(line) > 400:
        return False
    
    # Unicode normalize to NFC
    line = unicodedata.normalize('NFC', line)
    
    # Remove lines with URLs
    if re.search(r'https?://|www\.|<[^>]+>', line):
        return False
    
    # Count characters (excluding spaces)
    chars_no_space = line.replace(' ', '')
    if len(chars_no_space) == 0:
        return False
    
    # Count Kannada script characters
    kannada_chars = len(KANNADA_RE.findall(chars_no_space))
    non_kannada_chars = len(NON_KANNADA_RE.findall(chars_no_space))
    
    total_counted = kannada_chars + non_kannada_chars
    if total_counted == 0:
        return False
    
    # At least 60% of counted characters must be Kannada script
    kannada_ratio = kannada_chars / total_counted
    if kannada_ratio < 0.6:
        return False
    
    # Non-Kannada characters shouldn't exceed 30%
    if non_kannada_chars / len(chars_no_space) > 0.3:
        return False
    
    return True

print("Reading raw text...")
with open(RAW_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Raw lines: {len(lines):,}")

print("Cleaning...")
seen = set()
cleaned = []
stats = {"too_short": 0, "too_long": 0, "url": 0, "low_kannada": 0, "dup": 0, "kept": 0}

for line in tqdm(lines, desc="Filtering"):
    line = line.strip()
    
    if len(line) < 10:
        stats["too_short"] += 1
        continue
    if len(line) > 400:
        stats["too_long"] += 1
        continue
    if re.search(r'https?://|www\.|<[^>]+>', line):
        stats["url"] += 1
        continue
    
    chars_no_space = line.replace(' ', '')
    kannada_chars = len(KANNADA_RE.findall(chars_no_space))
    non_kannada_chars = len(NON_KANNADA_RE.findall(chars_no_space))
    total_counted = kannada_chars + non_kannada_chars
    
    if total_counted == 0 or (kannada_chars / total_counted) < 0.6:
        stats["low_kannada"] += 1
        continue
    
    if non_kannada_chars / len(chars_no_space) > 0.3:
        stats["low_kannada"] += 1
        continue
    
    # Dedup
    line = unicodedata.normalize('NFC', line)
    h = hashlib.md5(line.encode()).hexdigest()
    if h in seen:
        stats["dup"] += 1
        continue
    seen.add(h)
    
    cleaned.append(line)
    stats["kept"] += 1

print(f"\nFiltering stats:")
print(f"  Too short:       {stats['too_short']:>8,}")
print(f"  Too long:        {stats['too_long']:>8,}")
print(f"  Has URL:         {stats['url']:>8,}")
print(f"  Low Kannada %:   {stats['low_kannada']:>8,}")
print(f"  Duplicates:      {stats['dup']:>8,}")
print(f"  Kept:            {stats['kept']:>8,}")
print(f"  Keep rate:       {stats['kept']/len(lines)*100:.1f}%")

with open(CLEAN_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(cleaned))

print(f"\nSaved to {CLEAN_FILE}")