"""Gradio demo for the fine-tuned text classifier (deploy as a Hugging Face Space)."""
import gradio as gr
from transformers import pipeline

clf = pipeline("text-classification", model="distilbert-base-uncased")  # TODO: your Hub model id


def predict(text):
    return clf(text)[0]


gr.Interface(fn=predict, inputs="text", outputs="json",
             title="Sentiment Classifier").launch()
