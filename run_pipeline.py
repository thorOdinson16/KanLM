"""Run the complete Kannada LM pipeline end-to-end."""
import subprocess
import sys
import os

STEPS = [
    ("Extract text", "src/extract_text.py"),
    ("Clean data", "src/clean_data.py"),
    ("Train tokenizer", "src/train_tokenizer.py"),
    ("Analyze tokenizer", "src/analyze_tokenizer.py"),
    ("Tokenize corpus", "src/tokenize_corpus.py"),
    ("Build sentiment data", "src/build_sentiment_data.py"),
    ("Train model", "src/train.py"),
    ("Evaluate perplexity", "src/evaluate_perplexity.py"),
    ("Evaluate sentiment", "src/evaluate_sentiment.py"),
    ("Generate samples", "src/generate.py"),
]

SKIP_EXISTING = True  # Set to False to force rerun

def should_skip(script):
    """Check if output already exists so we don't rerun unnecessarily."""
    if not SKIP_EXISTING:
        return False
    checks = {
        "extract_text.py": "data/cleaned/kannada_raw.txt",
        "clean_data.py": "data/cleaned/kannada_clean.txt",
        "train_tokenizer.py": "tokenizer/vocab.json",
        "analyze_tokenizer.py": None,  # Always run
        "tokenize_corpus.py": "data/tokenized/train",
        "build_sentiment_data.py": "data/cleaned/kannada_sentiment.jsonl",
        "train.py": "checkpoints/final/model.safetensors",
        "evaluate_perplexity.py": "eval/perplexity_results.json",
        "evaluate_sentiment.py": "eval/sentiment_results.json",
        "generate.py": "eval/generated_samples.txt",
    }
    for key, path in checks.items():
        if key in script and path and os.path.exists(path):
            return True
    return False

for step_name, script in STEPS:
    if should_skip(script):
        print(f"\n[{step_name}] Already done, skipping.")
        continue
    
    print(f"\n{'='*60}")
    print(f"[{step_name}] Running {script}...")
    print(f"{'='*60}")
    
    result = subprocess.run([sys.executable, script], cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\nFailed at: {step_name}")
        sys.exit(1)

print("\n" + "="*60)
print("Pipeline complete!")
print("="*60)