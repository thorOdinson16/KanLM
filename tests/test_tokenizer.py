"""Sanity checks for the Kannada BPE tokenizer."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import TOKENIZER_DIR, VOCAB_SIZE
from transformers import AutoTokenizer

def test_tokenizer_loads():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    assert tokenizer is not None
    assert tokenizer.vocab_size == VOCAB_SIZE
    print("✓ Tokenizer loads with correct vocab size")

def test_special_tokens():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.bos_token == "<bos>"
    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.mask_token == "<mask>"
    print("✓ Special tokens present")

def test_encode_decode_roundtrip():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    text = "ನಾನು ಕನ್ನಡ ಕಲಿಯುತ್ತಿದ್ದೇನೆ"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert "ನಾನು" in decoded
    assert len(ids) > 1
    print(f"✓ Roundtrip works: '{text}' → {len(ids)} tokens → '{decoded[:40]}...'")

def test_fertility():
    """Test that tokenizer is efficient on Kannada text."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    sentences = [
        "ನಾನು ಇಂದು ಬೆಂಗಳೂರಿಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ",
        "ಕನ್ನಡ ಭಾಷೆ ತುಂಬಾ ಸುಂದರವಾಗಿದೆ",
        "ನಮ್ಮ ದೇಶದ ಸಂಸ್ಕೃತಿ ಶ್ರೇಷ್ಠವಾಗಿದೆ",
    ]
    for sent in sentences:
        tokens = tokenizer.tokenize(sent)
        words = sent.split()
        fertility = len(tokens) / len(words)
        assert fertility < 3.0, f"Fertility {fertility:.2f} too high for '{sent}'"
    print("✓ Fertility within expected range")

if __name__ == "__main__":
    test_tokenizer_loads()
    test_special_tokens()
    test_encode_decode_roundtrip()
    test_fertility()
    print("\nAll tokenizer tests passed!")