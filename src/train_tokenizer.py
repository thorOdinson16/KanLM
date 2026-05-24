"""Train a BPE tokenizer on Kannada text."""
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from transformers import PreTrainedTokenizerFast

CLEANED_DIR = "data/cleaned"
TOKENIZER_DIR = "tokenizer"
CLEAN_FILE = os.path.join(CLEANED_DIR, "kannada_clean.txt")

os.makedirs(TOKENIZER_DIR, exist_ok=True)

# Initialize tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.normalizer = normalizers.NFC()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Special tokens
special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]

trainer = trainers.BpeTrainer(
    vocab_size=12000,
    min_frequency=3,
    special_tokens=special_tokens,
    show_progress=True,
)

print("Training tokenizer...")
tokenizer.train([CLEAN_FILE], trainer)

# Save raw tokenizer
tokenizer.save(os.path.join(TOKENIZER_DIR, "kannada-bpe.json"))
print(f"Raw tokenizer saved. Vocab size: {tokenizer.get_vocab_size()}")

# Wrap as HuggingFace tokenizer
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<bos>",
    eos_token="<eos>",
    pad_token="<pad>",
    unk_token="<unk>",
    mask_token="<mask>",
)
hf_tokenizer.save_pretrained(TOKENIZER_DIR)
print(f"HuggingFace tokenizer saved to {TOKENIZER_DIR}/")