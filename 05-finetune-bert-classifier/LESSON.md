# Lesson 05 - NLP & Transformers: Text Classification & Fine-tuning

> Read with `src/demo_classifier.py` open; run `notebooks/05_nlp_transformers.ipynb` for the
> word-importance and embedding diagrams.

## 0. What you'll be able to do after this
- Explain how text becomes numbers (tokenization, TF-IDF, embeddings).
- Build a strong text-classification **baseline** and run it (`python -m src.demo_classifier`).
- Read the production **BERT fine-tuning** code and know when a transformer is worth it.

## 1. The big picture (why this project exists)
Most production NLP is text classification: route a support ticket, flag toxic content, detect
intent. A simple TF-IDF model is a shockingly strong baseline; a fine-tuned transformer (BERT) goes
further on harder problems. You should always build the baseline first, then earn the transformer.

## 2. Foundations from scratch (the basics)
**Text -> numbers.** Computers need numbers, so we convert text first.
- **Tokenization:** split text into tokens (words or sub-words) and map each to an id.
- **TF-IDF:** represent a sentence by which words it contains, down-weighting common words ("the").
  Simple, fast, strong. `src/demo_classifier.py` uses it + logistic regression and classifies new
  sentences correctly.
- **Embeddings:** represent each word as a vector so *similar meaning = nearby* ("great" near
  "good"). Captures meaning that TF-IDF misses.

**Transformers (BERT).** Read a whole sentence at once and use **attention** to understand each
word *in context* ("bank" by a river vs a savings bank). They are **pretrained** on huge text
(unsupervised), then **fine-tuned** cheaply on your small labelled task. `src/finetune_bert.py`
does this with the Hugging Face `Trainer`.

**Pretraining vs fine-tuning.** Pretraining (done for you, once, expensively) teaches general
language. Fine-tuning (you, cheaply) adapts it to your task with a few thousand labelled examples.

**Metrics for imbalance.** If classes are skewed, use **macro-F1**, not accuracy (accuracy hides
poor performance on the minority class).

## 3. How the pieces fit - the flow
```
            (baseline - runs on CPU)                  (production - needs a GPU)
  text --> TF-IDF vectors --> logistic regression      text --> tokenizer --> BERT encoder
            (demo_classifier.py)  -> prediction                 (finetune_bert.py) -> Trainer
                                                                          |
                              ALWAYS compare:  transformer F1  >?  baseline F1
                                                                          v
                                                          push to Hub -> Gradio Space demo
```

## 4. Code reading order
1. `src/demo_classifier.py` - TF-IDF + logistic baseline (run it first).
2. `notebooks/05_nlp_transformers.ipynb` - tokenization, word importance, embeddings map.
3. `src/finetune_bert.py` - the production transformer fine-tune (tokenize -> Trainer -> push_to_hub).
4. `app/app.py` - the Gradio demo Space.

## 5. Common mistakes (and how to avoid them)
- **Skipping the baseline** -> you can't tell if the transformer actually helped.
- **Accuracy on imbalanced data** -> use macro-F1.
- **Tiny dataset, huge model** -> the baseline may win; that's a real finding, report it.
- **Data leakage** -> keep a clean train/validation/test split (same discipline as Project 01).

## 6. Check your understanding
1. Why is TF-IDF a strong baseline despite ignoring word order?
2. What do embeddings capture that TF-IDF cannot?
3. What does "pretraining" give BERT before you ever fine-tune it?
4. When might the simple baseline beat a fine-tuned transformer?

## 7. Mini-glossary (full versions in /GLOSSARY.md)
tokenization, vocabulary, TF-IDF, embedding, transformer, BERT, attention, pretraining,
fine-tuning, encoder, macro-F1.

## 8. Going deeper
- Hugging Face NLP Course: https://huggingface.co/learn/nlp-course
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
