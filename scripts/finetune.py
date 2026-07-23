import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 1. Configuration
MODEL_ID = "unsloth/llama-3-8b-Instruct-bnb-4bit" # Ungated, pre-quantized 4-bit Llama 3! No HF token needed.
DATASET_PATH = "data/dataset.jsonl"
OUTPUT_DIR = "data/checkpoints/dev-ai-llama3-lora"

def train():
    print(f"🚀 DevAI: Starting QLoRA Fine-tuning on {MODEL_ID}")
    
    # 2. Load Dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Format conversations using the tokenizer's chat template
    def format_chat_template(example):
        # The dataset contains {"messages": [{"role": "user", "content": "..."}]}
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return example
        
    dataset = dataset.map(format_chat_template)

    # 4. Configure 4-bit Quantization (fits 8B model in 5GB VRAM)
    # bf16 compute dtype. Note: T4 has no bf16 Tensor Cores, so this runs slower
    # on T4 than fp16 would, but bf16 matches fp32's exponent range and is far
    # more numerically stable for LLM training (fp16 needs GradScaler loss-scaling
    # to avoid overflow, and that emulated path doesn't even support bf16 tensors --
    # which is what caused the earlier crash). Also: this checkpoint bakes its own
    # quantization_config (bf16 compute) into config.json, which transformers uses
    # instead of whatever we pass here anyway -- so bf16 is effectively the only
    # option unless we edit the checkpoint's config.json directly.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 5. Load Base Model
    print("Loading Base Model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    
    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)

    # 6. LoRA Adapter Configuration
    print("Injecting LoRA Adapters...")
    peft_config = LoraConfig(
        r=32,              # 32 is a great balance of speed/capacity for T4
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # NOTE: We no longer call get_peft_model() here. Newer versions of trl's
    # SFTTrainer inject the LoRA adapters itself when given `peft_config`,
    # and will raise a ValueError if it's handed an already-wrapped PeftModel.
    # We'll print trainable params after the trainer wraps the model instead.

    # 7. Training Arguments
    # SFTConfig replaces plain TrainingArguments and now also carries the
    # dataset/tokenization fields (dataset_text_field, max_length) that used
    # to be passed directly into SFTTrainer's constructor in older trl versions.
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        max_grad_norm=0.3,
        max_steps=500,
        fp16=False,
        bf16=True,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
        max_length=1024,
    )

    # 8. SFT Trainer
    # tokenizer= was removed in favor of processing_class=; dataset_text_field
    # and max_seq_length (now max_length) moved onto SFTConfig above.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )
    trainer.model.print_trainable_parameters()

    # 9. Train with Auto-Resume Support
    print("Starting Training Loop...")
    import os
    from transformers.trainer_utils import get_last_checkpoint
    
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    
    if last_checkpoint is not None:
        print(f"Attempting to resume training from checkpoint: {last_checkpoint}")
        try:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        except Exception as e:
            print(f"Failed to resume from checkpoint (likely a LoRA rank mismatch): {e}")
            print("Falling back to training from scratch...")
            trainer.train()
    else:
        trainer.train()

    # 10. Save Adapters
    print(f"Training Complete! Saving LoRA adapters to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done! You can now download the adapters and run the model locally.")

if __name__ == "__main__":
    train()
