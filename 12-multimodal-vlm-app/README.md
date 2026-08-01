# Project 12 - Multimodal App (Vision-Language Model)

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented, runnable code in this folder.
> Build an app that understands images AND text together - visual Q&A, document/screenshot understanding, or image-grounded chat.
**Phase 5: Agentic AI & Multimodal**  |  **Difficulty:** Advanced  |  **Est. time:** Week 20-22 (~40 hrs)

---

## Why (the point of this project)
'Multimodal systems (vision + language) are among the fastest-growing specializations in 2026,' and the newest open models ship native vision. Multimodal is where a lot of real enterprise value is (documents, screenshots, charts, medical imagery). This project proves you can work beyond text-only LLMs.

## What you will build
A vision-language application using an open VLM: e.g. a document/figure Q&A tool ('ask questions about this PDF page or chart'), a screenshot-to-structured-data extractor, or an image-grounded assistant - deployed as an HF Space with an evaluation set.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **Vision-language models (VLMs)** | Models that jointly encode images + text to answer questions, describe, or extract structured info. |
| **Image encoders + projection** | A vision encoder turns images into tokens projected into the LLM's space so it can 'read' them. |
| **Multimodal prompting** | Combine image(s) + instruction; prompt design and image preprocessing strongly affect results. |
| **Document/chart understanding** | OCR-free extraction of structure from documents, tables, and charts - high enterprise value. |
| **Multimodal evaluation** | Build a labeled set (image+question -> answer) and score accuracy; harder than text-only eval. |

## How - step by step
1. Choose a use case (PDF/figure Q&A, screenshot->JSON extraction, or image-grounded chat).
2. Select an open VLM appropriate to your compute (a 2026 small multimodal model).
3. Build the preprocessing (image loading, resizing, optional page rendering) - your data skills help.
4. Implement the inference pipeline (image + prompt -> structured/answer output).
5. Create a small labeled eval set; measure accuracy/exact-match; log to W&B.
6. Add output validation (e.g. JSON schema) reusing guardrails from P9.
7. Deploy a Gradio Space where users upload an image and ask questions.
8. Document strengths, failure cases (small text, dense tables), and mitigations.

## Tech stack
transformers (VLM), an open vision-language model, Pillow/pdf2image, Gradio, Docker, W&B

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **Hugging Face** | VLM + the live multimodal Space. |
| **Kaggle** | GPU for inference/eval; image datasets. |
| **GitHub** | Pipeline + eval set + results. |
| **Weights & Biases** | Track multimodal eval accuracy across prompts/models. |

## Portfolio artifact
Live HF Space (very demo-able) + GitHub repo + W&B eval. Resume line: 'Built and evaluated a vision-language app (document/figure Q&A) on an open VLM with a labeled eval set.'

## Definition of Done
- [ ] App answers questions about user-uploaded images in the browser.
- [ ] A labeled eval set produces an accuracy/exact-match score logged to W&B.
- [ ] Outputs are validated (schema) for the extraction use case.
- [ ] README documents the model, pipeline, and concrete failure cases.
- [ ] Runs reproducibly via Docker.

## Stretch goals
- Fine-tune the VLM (LoRA) on your domain images and compare.
- Add multi-image / multi-page reasoning.
- Compare two 2026 open VLMs on your eval set.

## Resources
- [HF multimodal/VLM docs](https://huggingface.co/docs/transformers/tasks/visual_question_answering)
- [Open VLM leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
- [Gradio multimodal](https://www.gradio.app/guides)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
