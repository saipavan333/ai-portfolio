"""
qlora_finetune.py - PRODUCTION: QLoRA fine-tune a real LLM on a single GPU.

    python -m src.qlora_finetune --data train.jsonl --base Qwen/Qwen2.5-0.5B

WHAT : loads a small open model in 4-bit (the 'Q'), attaches LoRA adapters (the low-rank patches),
       and trains with TRL's SFTTrainer on instruction/response data; pushes the adapter to the Hub.
WHY  : this is how you fine-tune a multi-billion-parameter model on one free GPU.
HOW  : 4-bit base + LoRA config -> SFTTrainer -> push_to_hub. Always eval vs the base model.
WHERE: production layer; lora_demo.py explains WHY this works.

Requires: transformers, peft, trl, bitsandbytes, datasets, accelerate (Kaggle GPU).
"""
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl of instruction/response pairs")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype="bfloat16")        # 4-bit base = the 'Q' in QLoRA
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, device_map="auto")
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],   # the low-rank patches
                      lora_dropout=0.05, task_type="CAUSAL_LM")
    data = load_dataset("json", data_files=args.data)["train"]

    trainer = SFTTrainer(model=model, train_dataset=data, peft_config=lora,
                         args=SFTConfig(output_dir="out", num_train_epochs=2,
                                        per_device_train_batch_size=4, report_to="wandb"))
    trainer.train()
    trainer.model.push_to_hub("your-username/my-lora-adapter")        # adapter is only a few MB
    print("done - now evaluate base vs fine-tuned on a held-out set (and watch for forgetting)")


if __name__ == "__main__":
    main()
