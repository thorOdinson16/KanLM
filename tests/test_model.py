"""Sanity checks for the trained Kannada model."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from src.config import CHECKPOINT_DIR, VOCAB_SIZE, BLOCK_SIZE
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_loads():
    tokenizer = AutoTokenizer.from_pretrained(f"{CHECKPOINT_DIR}/final")
    model = AutoModelForCausalLM.from_pretrained(f"{CHECKPOINT_DIR}/final")
    assert model is not None
    params = sum(p.numel() for p in model.parameters())
    assert params > 30_000_000
    print(f"✓ Model loads with {params:,} parameters")

def test_forward_pass():
    tokenizer = AutoTokenizer.from_pretrained(f"{CHECKPOINT_DIR}/final")
    model = AutoModelForCausalLM.from_pretrained(f"{CHECKPOINT_DIR}/final")
    
    text = "ನಾನು ಇಂದು ಬೆಳಿಗ್ಗೆ ಎದ್ದು"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    assert outputs.loss is not None
    assert outputs.loss.item() > 0
    assert outputs.loss.item() < 10
    print(f"✓ Forward pass works, loss = {outputs.loss.item():.4f}")

def test_generation():
    tokenizer = AutoTokenizer.from_pretrained(f"{CHECKPOINT_DIR}/final")
    model = AutoModelForCausalLM.from_pretrained(f"{CHECKPOINT_DIR}/final")
    
    prompt = "ಕನ್ನಡ ಭಾಷೆ"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert len(generated) > len(prompt)
    assert any('\u0C80' <= c <= '\u0CFF' for c in generated)  # Contains Kannada
    print(f"✓ Generation works: '{prompt}...' → '{generated[:60]}...'")

def test_vocab_size():
    model = AutoModelForCausalLM.from_pretrained(f"{CHECKPOINT_DIR}/final")
    # GPT2 has wte (word token embeddings) weight
    vocab_size = model.get_input_embeddings().weight.shape[0]
    assert vocab_size == VOCAB_SIZE
    print(f"✓ Model vocab size matches tokenizer: {vocab_size}")

if __name__ == "__main__":
    test_model_loads()
    test_forward_pass()
    test_generation()
    test_vocab_size()
    print("\nAll model tests passed!")