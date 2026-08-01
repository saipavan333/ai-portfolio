"""Gradio demo for the vision-language app (deploy as a Hugging Face Space)."""
import gradio as gr
# from src.vlm_infer import answer   # wire your real VLM here


def vqa(image, question):
    # TODO: run the VLM on (image, question); placeholder so the Space boots:
    return f"(demo) you asked: {question}"


gr.Interface(fn=vqa, inputs=[gr.Image(type="pil"), "text"], outputs="text",
             title="Vision-Language Q&A").launch()
