# Kannada GPT-2 Small (kannada-gpt2-32m)

A **31.6M parameter GPT-2 style autoregressive language model** trained entirely from scratch on Kannada text. Everything — data pipeline, BPE tokenizer, model weights — built from the ground up on a single NVIDIA RTX 5070.

---

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("AbhiDS16/kannada-gpt2-32m")
tokenizer = AutoTokenizer.from_pretrained("AbhiDS16/kannada-gpt2-32m")

prompt = "ನಾನು ಇಂದು ಬೆಳಿಗ್ಗೆ"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**[🤗 Model on HuggingFace](https://huggingface.co/AbhiDS16/kannada-gpt2-32m)** | **[💻 Code on GitHub](https://github.com/thorOdinson16/KanLM)**

---

## Data Pipeline

### Source
**[CulturaX-Kn](https://huggingface.co/datasets/Kannada-LLM-Labs/CulturaX-Kn)** — 1.35M documents (~4GB) of Kannada web text sourced from mC4. Contains news articles, blog posts, and general web content.

### Cleaning Process

| Stage | Count |
|-------|-------|
| Raw sentences extracted | 20,243,314 |
| Too short (< 10 chars) | 1,051,647 |
| Too long (> 400 chars) | 2,349,165 |
| Contains URL/HTML | 67,789 |
| Low Kannada script ratio (< 60%) | 2,907,256 |
| Exact duplicates | 1,295,122 |
| **Final clean sentences** | **12,572,335** |
| **Keep rate** | **62.1%** |

### Filtering Rules
- Length: 10–400 characters
- Unicode normalization to NFC
- Remove lines with URLs or HTML tags
- At least 60% of characters must be Kannada script (U+0C80–U+0CFF)
- No more than 30% Latin/digit characters
- Exact deduplication via MD5 hashing

---

## Tokenizer

A **custom Byte-Pair Encoding (BPE) tokenizer** trained from scratch on the cleaned Kannada corpus. No pre-existing vocabulary — every subword was learned from Kannada text.

| Property | Value |
|----------|-------|
| Vocabulary size | 12,000 |
| Algorithm | BPE |
| Min token frequency | 3 |
| Normalization | Unicode NFC |
| Pre-tokenization | Whitespace |
| Special tokens | `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<mask>` |

### Fertility Analysis

*Tokens per word — lower is better. Measured on 2,000 held-out Kannada sentences.*

| Tokenizer | Fertility | Improvement |
|-----------|-----------|-------------|
| **Our BPE (Kannada)** | **1.91** | — |
| XLM-R (multilingual) | 2.43 | **21.5% more efficient** |
| mBERT (multilingual) | 4.00 | **52.2% more efficient** |

**Key insight:** Our tokenizer uses **half the tokens** of mBERT for the same Kannada text. Multilingual tokenizers drastically over-segment Dravidian scripts — a custom tokenizer effectively doubles the usable context window.

### Example Tokenization

```
Text: "ಕನ್ನಡ ಭಾಷೆಯ ಸೌಂದರ್ಯ ಅದ್ಭುತವಾಗಿದೆ"

Our BPE (5 tokens): ಕನ್ನಡ | ಭಾಷೆಯ | ಸೌಂದರ್ಯ | ಅದ್ಭುತ | ವಾಗಿದೆ
XLM-R  (6 tokens): ▁ಕನ್ನಡ | ▁ಭಾಷೆ | ಯ | ▁ಸೌಂದರ್ಯ | ▁ಅದ್ಭುತ | ವಾಗಿದೆ
```

---

## Model Architecture

Standard GPT-2 autoregressive transformer with causal language modeling head.

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-2 (decoder-only transformer) |
| Parameters | **31,626,240** |
| Layers | 8 |
| Hidden dimension | 512 |
| Attention heads | 8 |
| Feed-forward dimension | 2,048 |
| Vocabulary size | 12,000 |
| Context length | 512 tokens |
| Activation | GELU |
| Dropout | 0.1 (embeddings, attention, residual) |

---

## Training

### Configuration

| Parameter | Value |
|-----------|-------|
| Training blocks (512 tokens each) | 894,640 |
| Validation blocks | 9,037 |
| Effective training tokens | ~463M |
| Epochs | 3 |
| Batch size | 16 |
| Gradient accumulation | 2 (effective batch 32) |
| Learning rate | 5 × 10⁻⁴ |
| LR schedule | Cosine decay with 1,000 step warmup |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Precision | fp16 mixed |
| Hardware | **NVIDIA RTX 5070 (8GB VRAM)** |
| RAM | 32GB |
| **Training time** | **7 hours 16 minutes** |
| Total steps | 83,874 |

### Loss Progression

| Metric | Start | After 500 steps | After 5,000 steps | Final |
|--------|-------|-----------------|-------------------|-------|
| Training loss | 9.14 | 6.86 | 4.61 | **3.54** |
| Validation loss | — | 6.73 | 4.49 | **3.45** |

---

## Evaluation

### Perplexity

| Metric | Value |
|--------|-------|
| Validation loss | 3.4594 |
| **Perplexity** | **31.80** |
| Evaluation tokens | 4,626,944 |

For context: a 124M GPT-2 trained on English Wikipedia achieves ~35 perplexity. Our 32M model achieves 31.80 on Kannada web text — a strong result for the model size and data budget.

### Downstream Sentiment Classification

To prove the model learns transferable representations:

| Setup | Detail |
|-------|--------|
| Method | Frozen LM → mean pooling → Logistic Regression |
| Training examples | 800 |
| Test examples | 200 |
| Labels | Positive / Negative (keyword-derived) |
| **Accuracy** | **73.5%** |
| **F1 Score (macro)** | **0.735** |

The classifier uses **only frozen LM embeddings** — no fine-tuning. This demonstrates that the model has learned meaningful semantic representations of Kannada text that generalize to downstream tasks.

### Qualitative Generation

The model produces **grammatically correct Kannada** with proper syntax, verb agreement, and semantic coherence. Some repetition occurs in longer generations — expected for a 32M model.

---

## Project Structure

```
kanlm/
│
├── src/                           # All source code
│   ├── config.py                  # Hyperparameters and paths
│   ├── extract_text.py            # Parquet files → raw text
│   ├── clean_data.py              # Filter, normalize, deduplicate
│   ├── train_tokenizer.py         # Train BPE tokenizer from scratch
│   ├── analyze_tokenizer.py       # Fertility comparison vs XLM-R/mBERT
│   ├── tokenize_corpus.py         # Tokenize and chunk into 512-token blocks
│   ├── model.py                   # GPT-2 model definition
│   ├── train.py                   # Training loop with fp16 mixed precision
│   ├── generate.py                # Text generation with nucleus sampling
│   ├── evaluate_perplexity.py     # Perplexity on validation set
│   ├── evaluate_sentiment.py      # Downstream sentiment classification
│   └── build_sentiment_data.py    # Create labeled sentiment dataset
│
├── tokenizer/                     # Trained tokenizer files
│   ├── kannada-bpe.json           # Raw BPE tokenizer
│   ├── tokenizer.json             # HuggingFace format
│   └── tokenizer_config.json      # Tokenizer configuration
│
├── model/                         # Final trained model (for deployment)
│   ├── config.json                # Model architecture
│   ├── model.safetensors          # Weights (~126MB)
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── checkpoints/                   # Training checkpoints
│
├── data/
│   ├── raw/                       # Original parquet files (21 files, ~4GB)
│   ├── cleaned/
│   │   ├── kannada_raw.txt        # Extracted text (20.2M lines)
│   │   ├── kannada_clean.txt      # Filtered text (12.6M lines)
│   │   ├── kannada_sentiment.csv  # Legacy sentiment data
│   │   └── kannada_sentiment.jsonl # Labeled sentiment data (1,000 examples)
│   └── tokenized/
│       ├── train/                 # Training blocks (894,640 sequences)
│       └── val/                   # Validation blocks (9,037 sequences)
│
├── eval/                          # Evaluation results
│   ├── perplexity_results.json    # {"perplexity": 31.80}
│   ├── sentiment_results.json     # {"accuracy": 0.735, "f1_macro": 0.735}
│   ├── generated_samples.txt      # Prompt + output pairs
│   └── fertility_analysis.json    # Tokenizer comparison data
│
├── notebooks/                     # Jupyter notebooks for visualization
│   ├── 01_data_exploration.ipynb  # Sentence length distribution, character frequency
│   ├── 02_tokenizer_analysis.ipynb # Fertility bar chart, token comparison
│   └── 03_generations_quality.ipynb # Generation with different parameters
│
├── tests/                         # Unit tests
│   ├── test_tokenizer.py          # Tokenizer sanity checks
│   └── test_model.py              # Model loading, forward pass, generation
│
├── run_pipeline.py                # End-to-end pipeline runner
├── requirements.txt               # Python dependencies
├── .gitignore                     # Exclude data/, checkpoints/
└── README.md                      # This file
```

---

## Reproducing

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM
- 32GB system RAM recommended
- ~20GB free disk space

### Installation

```bash
pip install torch transformers datasets tokenizers pandas pyarrow tqdm scikit-learn matplotlib
```

### Step-by-Step

```bash
# 1. Download data (parquet files from HuggingFace)
#    Place in data/raw/ (21 parquet files)

# 2. Extract text from parquet files
python src/extract_text.py

# 3. Clean and filter
python src/clean_data.py

# 4. Train tokenizer
python src/train_tokenizer.py

# 5. Analyze tokenizer fertility
python src/analyze_tokenizer.py

# 6. Tokenize entire corpus
python src/tokenize_corpus.py

# 7. Train model (~7 hours on RTX 5070)
python src/train.py

# 8. Evaluate
python src/evaluate_perplexity.py
python src/build_sentiment_data.py
python src/evaluate_sentiment.py
python src/generate.py

# Or run everything at once:
python run_pipeline.py
```

### Verify

```bash
python tests/test_tokenizer.py
python tests/test_model.py
```

---

## Results Summary

| Component | Result |
|-----------|--------|
| **Training data** | 12.6M clean Kannada sentences |
| **Tokenizer** | Custom BPE, 12K vocab |
| **vs mBERT** | **52.2% more token-efficient** |
| **vs XLM-R** | **21.5% more token-efficient** |
| **Model parameters** | 31.6M |
| **Training time** | 7h 16m (RTX 5070) |
| **Validation perplexity** | **31.80** |
| **Sentiment accuracy** | **73.5%** (frozen LM) |
| **Generation quality** | Grammatically correct, coherent |

---

## Limitations

- **Small model:** 31.6M parameters — limited factual knowledge and reasoning ability
- **Repetition:** Tends to repeat phrases in longer generations (common in small autoregressive models)
- **Training data:** Web text (news, blogs) — reflects biases and code-mixing present in online Kannada
- **Not instruction-tuned:** Raw causal LM — not suitable for chat/QA without further fine-tuning
- **Data recency:** Training data from mC4 (2011–2022) — may not reflect current events

---

## Future Work

- Scale to 125M+ parameters with more data
- Train on cleaner, curated Kannada sources (books, Wikipedia, news)
- Instruction fine-tuning for conversational use
- Extend context length with RoPE or ALiBi
- Apply the same pipeline to other Dravidian languages (Tamil, Telugu, Malayalam)

---

## Acknowledgments

- **CulturaX-Kn** ([Kannada-LLM-Labs](https://huggingface.co/datasets/Kannada-LLM-Labs/CulturaX-Kn)) for the curated Kannada dataset
- **HuggingFace** for the transformers, tokenizers, and datasets libraries