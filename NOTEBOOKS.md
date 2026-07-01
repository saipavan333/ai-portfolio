# Notebooks Index - 15 self-explanatory teaching notebooks

Every project ships an **executed, per-cell-explained Kaggle notebook** in its `notebooks/` folder.
Each one teaches the topic from scratch in plain English, has **real rendered diagrams**, and runs
its concept demo with lightweight tools (NumPy/scikit-learn) so it works anywhere. The heavy
GPU/LLM training is included as copy-paste-ready blocks for a Kaggle GPU.

**How to use:** on Kaggle -> New Notebook -> File -> Upload Notebook -> Run All. Or open locally in
Jupyter/VS Code.

| # | Topic | Notebook | Runnable demo you can see |
|---|-------|----------|----------------------------|
| 01 | Classical ML | `01-tabular-ml-pipeline/notebooks/01_classical_ml_churn.ipynb` | churn pipeline; class-balance, confusion-matrix, ROC diagrams |
| 02 | Data-Centric AI / MLOps | `02-feature-pipeline-mlops/notebooks/02_feature_engineering.ipynb` | +0.09 AUC from features; interaction heatmap |
| 03 | Neural Nets from scratch | `03-neural-net-from-scratch/notebooks/03_neural_net_from_scratch.ipynb` | NumPy backprop -> 97.5%; gradient-check PASS |
| 04 | Computer Vision | `04-cv-transfer-learning/notebooks/04_computer_vision.ipynb` | convolution feature maps; augmentation; transfer-learning schematic |
| 05 | NLP & Transformers | `05-finetune-bert-classifier/notebooks/05_nlp_transformers.ipynb` | working text classifier; word-importance; embeddings map |
| 06 | Mini-GPT | `06-mini-gpt-from-scratch/notebooks/06_mini_gpt_from_scratch.ipynb` | NumPy self-attention heatmap; tiny text generator |
| 07 | RAG | `07-production-rag/notebooks/07_rag_from_scratch.ipynb` | working retrieval; similarity bar; 2D knowledge map |
| 08 | LoRA / QLoRA | `08-lora-qlora-finetune/notebooks/08_lora_qlora_finetuning.ipynb` | 256x fewer params; low-rank + 4-bit quantization diagrams |
| 09 | LLM Eval & Guardrails | `09-llm-eval-guardrails/notebooks/09_llm_eval_guardrails.ipynb` | eval scorecard; PII redaction; injection blocking |
| 10 | Tool-using Agent | `10-tool-using-agent/notebooks/10_tool_using_agent.ipynb` | ReAct loop solving a 2-tool task; trace diagram |
| 11 | Multi-Agent | `11-multi-agent-workflow/notebooks/11_multi_agent_workflow.ipynb` | planner/researcher/writer/critic; critic loop improves coverage |
| 12 | Multimodal (VLM) | `12-multimodal-vlm-app/notebooks/12_multimodal_vlm.ipynb` | image->patch tokens; toy VQA answers "blue circle" |
| 13 | Reasoning & RL (GRPO) | `13-reasoning-model-rl/notebooks/13_reasoning_model_grpo.ipynb` | GRPO learns to reason; reward-hacking demo |
| 14 | Serving & MLOps | `14-llmops-deployment/notebooks/14_serving_mlops.ipynb` | batching throughput; p95 latency; rate limiter; architecture |
| 15 | Capstone | `15-capstone-agentic-product/notebooks/15_capstone_end_to_end.ipynb` | RAG+agent+guardrails+eval in one pipeline (100% end-to-end) |

All 15 were executed end-to-end with zero errors before shipping.
