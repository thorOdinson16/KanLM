"""Train the Kannada GPT-2 model."""
import os
import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel, 
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)
from datasets import load_from_disk
from config import *

def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache")

    # Load data
    print("Loading data...")
    train_data = load_from_disk(os.path.join(TOKENIZED_DIR, "train"))
    val_data = load_from_disk(os.path.join(TOKENIZED_DIR, "val"))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    print(f"Train blocks: {len(train_data):,}")
    print(f"Val blocks:   {len(val_data):,}")

    # Model
    print("Creating model...")
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

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=GRAD_CLIP,
        fp16=True,
        dataloader_num_workers=0,  # Must be 0 on Windows
        report_to="none",
        run_name="kannada-gpt2-80m",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving final model...")
    trainer.save_model(os.path.join(CHECKPOINT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(CHECKPOINT_DIR, "final"))
    print("Done.")

if __name__ == "__main__":
    main()