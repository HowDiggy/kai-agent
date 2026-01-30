from unsloth import FastLanguageModel
import torch

# 1. Load the Adapter Model
print("Loading Adapters...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. Merge LoRA into Base Model
print("Merging weights...")
model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit")
print("Success! Model saved to ./merged_model")
