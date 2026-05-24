"""Define the GPT-2 style model architecture."""
from transformers import GPT2Config, GPT2LMHeadModel
from config import *

config = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    activation_function="gelu",
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)

model = GPT2LMHeadModel(config)
print(f"Parameters: {model.num_parameters():,}")
print(f"Layers: {N_LAYER}, Hidden: {N_EMBD}, Heads: {N_HEAD}")
print(f"Vocab size: {VOCAB_SIZE}, Block size: {BLOCK_SIZE}")