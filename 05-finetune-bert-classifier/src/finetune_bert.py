"""
finetune_bert.py - PRODUCTION NLP: fine-tune a pretrained transformer (needs a GPU).

    python -m src.finetune_bert        # run on a Kaggle GPU notebook

WHAT : fine-tunes a DistilBERT/BERT encoder for text classification using the Hugging Face
       `Trainer`, then pushes the model to the Hub.
WHY  : a pretrained transformer already understands language, so it learns your task from
       relatively little data and usually beats the TF-IDF baseline on harder problems.
HOW  : tokenize -> AutoModelForSequenceClassification -> Trainer with accuracy+F1 -> push_to_hub.
WHERE: the production layer. demo_classifier.py is the baseline you must beat.

Requires: transformers, datasets, evaluate, accelerate, torch (preinstalled on Kaggle GPU).
"""
from __future__ import annotations


def main(model_name: str = "distilbert-base-uncased", dataset: str = "imdb"):
    import numpy as np, evaluate
    from datasets import load_dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)

    ds = load_dataset(dataset)
    tok = AutoTokenizer.from_pretrained(model_name)
    ds = ds.map(lambda b: tok(b["text"], truncation=True), batched=True)   # text -> token ids

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    acc, f1 = evaluate.load("accuracy"), evaluate.load("f1")

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": acc.compute(predictions=preds, references=p.label_ids)["accuracy"],
                "f1": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]}

    args = TrainingArguments(output_dir="out", per_device_train_batch_size=16,
                             num_train_epochs=2, eval_strategy="epoch", report_to="wandb")
    trainer = Trainer(model=model, args=args,
                      train_dataset=ds["train"].shuffle(seed=0).select(range(3000)),  # subset = fast
                      eval_dataset=ds["test"].select(range(1000)),
                      compute_metrics=metrics, tokenizer=tok)
    trainer.train()
    trainer.push_to_hub()          # share the model; compare its F1 to the TF-IDF baseline
    print("done - remember to report transformer F1 vs the baseline from demo_classifier.py")


if __name__ == "__main__":
    main()
