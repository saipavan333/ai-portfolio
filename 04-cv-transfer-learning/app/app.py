"""Gradio demo for the fine-tuned classifier - deploy as a Hugging Face Space.
Run locally: python app/app.py   (needs gradio + torch + timm + your model.pt)"""
import gradio as gr


def predict(image):
    # TODO: load your model.pt, preprocess `image` (resize+normalize), return {label: prob}
    # placeholder so the Space boots before you wire the model in:
    return {"cat": 0.7, "dog": 0.3}


demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"),
                    outputs=gr.Label(num_top_classes=3),
                    title="Image Classifier", description="Upload an image to classify.")

if __name__ == "__main__":
    demo.launch()
