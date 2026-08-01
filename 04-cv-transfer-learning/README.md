# Project 04 - Computer Vision with Transfer Learning + Live Demo

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented, runnable code in this folder.
> Fine-tune a pretrained vision model on your own image classes and ship it as a live Hugging Face Space anyone can try in the browser.
**Phase 2: Deep Learning Core**  |  **Difficulty:** Intermediate  |  **Est. time:** Week 4-7 (~45 hrs)

---

## Why (the point of this project)
Transfer learning is how real CV gets done: stand on a model pretrained on millions of images and adapt it to your problem with a few thousand. This project gives you your first DEPLOYED, clickable demo - dramatically more impressive than a notebook - and teaches the fine-tuning loop you will reuse for LLMs.

## What you will build
An image classifier fine-tuned from a pretrained backbone (ResNet or a Vision Transformer) on a focused dataset (e.g. plant disease, food types, or a custom set you assemble), plus a Gradio app deployed as a Hugging Face Space where users upload an image and get predictions.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **Transfer learning & fine-tuning** | Reuse a pretrained backbone's learned features; replace/retrain the head (and optionally unfreeze layers) for your classes. |
| **CNNs vs Vision Transformers** | CNNs exploit local spatial structure; ViTs treat image patches as tokens with attention. Both are available pretrained. |
| **Data augmentation** | Random crops/flips/color jitter synthetically expand the dataset and reduce overfitting. |
| **Transfer regimes** | Feature extraction (freeze backbone, train head) vs full fine-tuning (unfreeze with a small LR). Choose by dataset size. |
| **Deployment with Gradio/Spaces** | Wrap the model in a simple UI and host it for free so anyone can interact with it. |

## How - step by step
1. Choose a focused classification task and dataset (Kaggle has many; or assemble ~1-3k labeled images).
2. Set up a PyTorch (or timm) DataLoader with train/val/test splits and augmentation.
3. Load a pretrained backbone (resnet50 or a ViT via timm/transformers). Replace the classifier head.
4. Stage 1: freeze backbone, train the head. Stage 2: unfreeze top layers, fine-tune with a small LR.
5. Track loss/accuracy/confusion matrix in W&B; use early stopping on val loss.
6. Evaluate on the test set; inspect misclassified images to understand failure modes.
7. Build a Gradio app (image in -> top-k labels out) and deploy as a Hugging Face Space.
8. Push the trained model to the HF Hub with a model card.

## Tech stack
PyTorch, timm/torchvision, Hugging Face Hub, Gradio, Spaces, W&B

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **Kaggle** | Free GPU for fine-tuning; source the image dataset. |
| **Hugging Face** | Host the model (with model card) AND the live Gradio Space demo. |
| **GitHub** | Training code + the Space app code; link both in the README. |
| **Weights & Biases** | Track training, augmentation experiments, and the confusion matrix. |

## Portfolio artifact
Live HF Space (put the link everywhere) + HF model + GitHub repo + W&B report. Resume line: 'Fine-tuned a ViT/ResNet to XX% accuracy and deployed a public demo on Hugging Face Spaces.'

## Definition of Done
- [ ] Fine-tuned model clearly beats a from-scratch baseline on the test set.
- [ ] Live HF Space accepts an upload and returns predictions in the browser.
- [ ] Model pushed to the HF Hub with a complete model card.
- [ ] W&B shows the training curves and a confusion matrix.
- [ ] README documents data, augmentation, two-stage fine-tuning, and limitations.

## Stretch goals
- Add Grad-CAM heatmaps to show where the model looks.
- Compare a CNN vs a ViT backbone on the same data.
- Quantize/export to ONNX and measure latency improvement.

## Resources
- [timm (PyTorch Image Models)](https://huggingface.co/docs/timm/index)
- [Gradio quickstart](https://www.gradio.app/guides/quickstart)
- [HF Spaces docs](https://huggingface.co/docs/hub/spaces)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
