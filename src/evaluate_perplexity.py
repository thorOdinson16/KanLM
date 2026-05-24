"""Evaluate perplexity on the validation set."""
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
from config import CHECKPOINT_DIR, TOKENIZED_DIR

model_path = f"{CHECKPOINT_DIR}/final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
model.eval()

# Load validation data
val_data = load_from_disk(f"{TOKENIZED_DIR}/val")
print(f"Evaluating on {len(val_data)} sequences...")

total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for i in tqdm(range(len(val_data)), desc="Evaluating"):
        input_ids = torch.tensor([val_data[i]["input_ids"]]).cuda()
        labels = input_ids.clone()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item() * input_ids.size(1)
        total_tokens += input_ids.size(1)

avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)

print(f"\n{'='*40}")
print(f"Validation Loss: {avg_loss:.4f}")
print(f"Perplexity:      {perplexity:.2f}")
print(f"{'='*40}")

# Save results
import json
results = {
    "eval_loss": avg_loss,
    "perplexity": perplexity,
    "num_tokens": total_tokens,
}
with open("eval/perplexity_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to eval/perplexity_results.json")