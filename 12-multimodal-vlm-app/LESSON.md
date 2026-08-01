# Lesson 12 - Multimodal AI: Vision-Language Models

> Read with `src/vlm_demo.py` open; run it (`python -m src.vlm_demo`). See
> `notebooks/12_multimodal_vlm.ipynb` for the patch-grid + VQA diagrams.

## 0. What you'll be able to do after this
- Explain how a model turns an **image into tokens** and fuses them with text.
- Run a toy but working **visual-Q&A** and read the production open-VLM code (`src/vlm_infer.py`).

## 1. The big picture (why this project exists)
Multimodal models understand **images + text together**: answer questions about a photo, read a
chart, extract fields from a scanned form. The newest open models ship native vision, and document
understanding is huge enterprise value.

## 2. Foundations from scratch (the basics)
- **An image is numbers** (a grid of pixels; colour = 3 grids R/G/B).
- **Patches as tokens:** a VLM chops the image into a grid of small **patches** and treats each as a
  token - so an image becomes a *sequence*, like a sentence. `to_patch_tokens()` shows a 64x64 image
  becoming 16 tokens.
- **One shared transformer:** image tokens are projected into the text token space, then a single
  transformer attends across image + question tokens to produce an answer.
- **Toy VQA:** `answer_color` + `answer_shape` read the image and answer correctly ("blue",
  "circle", asserted). A real VLM does this for arbitrary images.

## 3. How the pieces fit - the flow
```
        (concept - runs on CPU)                          (production - vlm_infer.py)
  image -> patches -> patch tokens                  image + question -> processor -> open VLM
            (vlm_demo.py)                                              -> generate -> answer
  toy VQA: dominant colour + shape -> answer         (Qwen2-VL etc.; deploy via app/app.py Space)
```

## 4. Code reading order
1. `src/vlm_demo.py` - `to_patch_tokens`, `answer_color`, `answer_shape` (run it first).
2. `notebooks/12_multimodal_vlm.ipynb` - patch grid + image+text->transformer schematic.
3. `src/vlm_infer.py` - real open-VLM inference; `app/app.py` - the Gradio Space.

## 5. Common mistakes (and how to avoid them)
- **Wrong image preprocessing** -> match the processor the VLM expects.
- **No output validation** -> validate structured outputs (reuse Project 09 guardrails).
- **No eval set** -> build labelled (image, question -> answer) pairs and measure accuracy.
- **Tiny text/dense tables** -> a known weak spot; test and document it.

## 6. Check your understanding
1. How does a VLM turn an image into something a transformer can read?
2. Why do image tokens and text tokens go into the SAME transformer?
3. What's the I/O of a visual-Q&A model?
4. Which Project-09 idea would you reuse to keep VLM outputs well-formed?

## 7. Mini-glossary (full versions in /GLOSSARY.md)
multimodal, VLM, patch, image encoder, projection, visual question answering, document understanding.

## 8. Going deeper
- HF visual question answering: https://huggingface.co/docs/transformers/tasks/visual_question_answering
- Open VLM leaderboard: https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
