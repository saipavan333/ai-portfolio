"""
vlm_infer.py - PRODUCTION multimodal: run a real open vision-language model.

    python -m src.vlm_infer --image invoice.png --question "What is the total?"

WHAT : loads an open VLM and answers a question about a real image (photo, document, chart).
WHY  : same input/output as vlm_demo.py, but a real model handles arbitrary images, not toy shapes.
HOW  : processor formats image+text -> model.generate -> decode the answer.
WHERE: production layer; vlm_demo.py teaches the patch-token idea. Deploy via app/app.py as a Space.

Requires: transformers, accelerate, pillow (Kaggle GPU).
"""
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, device_map="auto")

    image = Image.open(args.image)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.question}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(images=image, text=prompt, return_tensors="pt").to(model.device)
    print(proc.decode(model.generate(**inputs, max_new_tokens=100)[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
