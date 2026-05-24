"""Generate Kannada text from the trained model."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import CHECKPOINT_DIR

model_path = f"{CHECKPOINT_DIR}/final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
model.eval()

prompts = [
    "ನಾನು ಇಂದು ಬೆಳಿಗ್ಗೆ",
    "ಬೆಂಗಳೂರಿನಲ್ಲಿ ಮಳೆ ಬಂದಾಗ",
    "ಕನ್ನಡ ಭಾಷೆಯ ಸೌಂದರ್ಯ",
    "ನಮ್ಮ ದೇಶದ ಸಂಸ್ಕೃತಿ",
    "ವಿದ್ಯಾರ್ಥಿಗಳು ಪರೀಕ್ಷೆಯಲ್ಲಿ",
    "ಹೊಸ ವರ್ಷದ ಆಚರಣೆ",
]

with open("eval/generated_samples.txt", "w", encoding="utf-8") as f:
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        f.write(f"PROMPT: {prompt}\n")
        f.write(f"OUTPUT: {generated}\n")
        f.write("-" * 60 + "\n")
        print(f"Prompt: {prompt}")
        print(f"Output: {generated[:200]}...")
        print()

print("Saved to eval/generated_samples.txt")