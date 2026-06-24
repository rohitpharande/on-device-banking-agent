# Fine-Tuning Guide (Qwen3-0.6B)

This guide documents the fine-tuning workflow for adapting the **Qwen3-0.6B** base model to serve as the intent classifier and slot extractor for the banking voice assistant.

---

## 🔬 Core Methodology

To achieve low latency (<100ms inference) on consumer hardware, we fine-tune a tiny 0.6B parameters model rather than relying on a larger 7B+ model. 

- **Base Model**: `Qwen/Qwen3-0.6B` (or `Qwen/Qwen3-0.6B-Instruct`)
- **Optimization Stack**: [Unsloth AI](https://github.com/unslothai/unsloth) which provides 2x faster training and 50% less memory utilization using manual autograd backpropagation kernels.
- **Adapter Type**: LoRA (Low-Rank Adaptation)
- **Task**: Conversational Function Calling (structured JSON output)

---

## 📂 Training Data Format

The dataset contains multi-turn banking dialogues formatted as standard OpenAI messages. The assistant's turns contain function calls in JSON formatting rather than plain text.

### Example Training Sample (JSONL):
```json
{
  "messages": [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "greeting", "arguments": {}}}]},
    {"role": "user", "content": "I'd like to check my savings balance please"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "check_balance", "arguments": {"account_type": "savings"}}}]},
    {"role": "user", "content": "actually checking please"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "check_balance", "arguments": {"account_type": "checking"}}}]},
    {"role": "user", "content": "thank you"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "thank_you", "arguments": {}}}]}
  ]
}
```

---

## 🚀 Step-by-Step Fine-Tuning Workflow

All training code is contained in the Jupyter notebook `training/finetune_qwen3.ipynb`.

### Step 1: Install Training Dependencies
Install the package stack on a machine with an NVIDIA GPU (a single consumer GPU like RTX 3060/3070/4060 with ≥ 8GB VRAM is sufficient):
```bash
pip install -r requirements-training.txt
```

### Step 2: Configure LoRA Adapter
Inside the notebook, we wrap the base model using Unsloth's optimized PEFT configuration:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-0.6B",
    max_seq_length=2048,
    load_in_4bit=True,  # Set to False if you want FP16/BF16 training
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                         # LoRA Rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,               # Optimized for 0
    bias="none",                  # Optimized for "none"
    use_gradient_checkpointing=True,
)
```

### Step 3: Run SFTTrainer
We format the data using the Qwen Chat Template and run the Hugging Face Supervised Fine-Tuning Trainer:
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False, # Can make training faster for short sequences
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60, # Adjust based on data size (usually 1-2 epochs)
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
    ),
)
trainer.train()
```

### Step 4: Merge Weights & Export to GGUF
Unsloth provides a unified utility to export the fine-tuned model directly to 16-bit GGUF or quantized formats (e.g., Q8_0, Q4_K_M) for deployment with `llama.cpp`:

```python
# Save local 16bit GGUF model
model.save_pretrained_gguf(
    "models/qwen3_06b_voice_banking_f16", 
    tokenizer, 
    quantization_method="f16"
)

# Save quantized 8-bit version for ultra-low memory
model.save_pretrained_gguf(
    "models/qwen3_06b_voice_banking_q8_0", 
    tokenizer, 
    quantization_method="q8_0"
)
```

This merges the LoRA adapter back into the base Qwen weights, converts the model format to GGUF, and packages the tokenizer metadata.
