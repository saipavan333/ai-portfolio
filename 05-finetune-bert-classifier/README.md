# Project 05 - Fine-tune a Transformer for Text Classification (BERT)

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented, runnable code in this folder.
> Use the Hugging Face Transformers stack end-to-end: tokenize text, fine-tune a BERT-family model, evaluate properly, and ship a demo.
**Phase 3: NLP & Transformers**  |  **Difficulty:** Intermediate  |  **Est. time:** Week 7-9 (~40 hrs)

---

## Why (the point of this project)
This is your on-ramp to the Transformers ecosystem that every modern AI job assumes you know: the Trainer API, datasets, tokenizers, the Hub. Text classification (intent, sentiment, topic, spam, toxicity) is one of the most common production NLP tasks, and BERT-style encoders remain the efficient, accurate default for it in 2026.

## What you will build
A fine-tuned encoder model (DistilBERT/BERT/DeBERTa) on a labeled text dataset, evaluated with the right metrics, pushed to the HF Hub, and served through a small Gradio Space. You will also compare it against a strong classical baseline (TF-IDF + linear) to learn when transformers are worth it.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **Tokenization & subwords** | Text is split into subword tokens mapped to ids; handles unknown words and keeps vocab small. |
| **Pretraining vs fine-tuning** | BERT is pretrained on huge text (masked-language modeling), then you fine-tune cheaply on your labeled task. |
| **Encoder transformers & [CLS]** | Bidirectional attention builds context-aware embeddings; a pooled representation feeds a classification head. |
| **The HF Trainer & datasets** | High-level training loop with evaluation, checkpointing, and logging; datasets streams/maps data efficiently. |
| **Class imbalance & metrics** | Use weighted loss / macro-F1 when classes are skewed; accuracy alone hides minority-class failure. |

## How - step by step
1. Pick a text-classification dataset (e.g. emotion, AG News, toxic comments, or a domain set).
2. Build a TF-IDF + logistic-regression baseline first - fast, and the bar transformers must clear.
3. Tokenize with an AutoTokenizer; create a tokenized HF Dataset with proper train/val/test.
4. Fine-tune AutoModelForSequenceClassification with the Trainer; set metrics (accuracy + macro-F1).
5. Handle imbalance (class weights or focal loss) if needed; track everything in W&B.
6. Evaluate on test; produce a confusion matrix and inspect error examples.
7. Push the model + tokenizer to the HF Hub with a model card; deploy a Gradio Space.
8. Write up when the transformer beat the baseline and by how much (and the compute cost).

## Tech stack
Hugging Face Transformers, datasets, PyTorch, scikit-learn, Gradio, W&B

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **Hugging Face** | Datasets, model, Trainer, Hub hosting, and a Gradio Space demo. |
| **Kaggle** | Free GPU for fine-tuning; alternative dataset source. |
| **GitHub** | Training + app code, baseline comparison, results table. |
| **Weights & Biases** | Track fine-tuning runs and a hyperparameter sweep (LR, batch size, epochs). |

## Portfolio artifact
HF model + live Space + GitHub repo + W&B sweep report. Resume line: 'Fine-tuned a transformer text classifier (macro-F1 X.XX), beating a TF-IDF baseline, and deployed it.'

## Definition of Done
- [ ] Transformer beats the TF-IDF baseline on macro-F1 (or you explain why it doesn't).
- [ ] Model + tokenizer on the HF Hub with a model card.
- [ ] Live Space classifies user-entered text.
- [ ] W&B contains a sweep over at least 2 hyperparameters.
- [ ] README reports metrics, the baseline comparison, and error analysis.

## Stretch goals
- Add explainability (Captum / attention or SHAP for text).
- Distill to a smaller model and compare accuracy vs latency.
- Try a 2026 small instruct model as a zero-shot classifier and compare.

## Resources
- [HF Course - Fine-tuning](https://huggingface.co/learn/nlp-course)
- [Transformers Trainer docs](https://huggingface.co/docs/transformers/main_classes/trainer)
- [datasets docs](https://huggingface.co/docs/datasets)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
