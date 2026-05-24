"""Extract text from downloaded parquet files into a single text file."""
import pandas as pd
import os
from tqdm import tqdm

RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"

os.makedirs(CLEANED_DIR, exist_ok=True)

parquet_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.parquet')])
output_file = os.path.join(CLEANED_DIR, "kannada_raw.txt")

print(f"Found {len(parquet_files)} parquet files")
print("Extracting text...")

total_rows = 0
with open(output_file, 'w', encoding='utf-8') as out:
    for file in tqdm(parquet_files, desc="Processing files"):
        df = pd.read_parquet(os.path.join(RAW_DIR, file))
        for text in df['text']:
            text = str(text).strip()
            if text:
                out.write(text + '\n')
                total_rows += 1

print(f"Done. {total_rows:,} documents saved to {output_file}")