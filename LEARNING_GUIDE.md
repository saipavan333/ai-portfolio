# LEARNING GUIDE - how to use this portfolio as a reference for years

This repo is built in **three layers** so it works both as a build plan *now* and a
textbook you can re-open in 2027, 2028, and beyond. Read this once.

## The three layers (open them in this order, every project)

1. **`README.md` - the blueprint (the WHAT and WHERE).**
   The 30,000-ft view: why the project matters, what you'll build, the step list, the
   tools, and the "Definition of Done". Read it first to know where you're going.

2. **`LESSON.md` - the teacher (the WHY and the basics).**
   This is the part that makes the repo a reference. It teaches the underlying concepts
   *from scratch* in plain English, draws the data/control flow as a diagram, tells you
   the order to read the code, and ends with a glossary + self-check questions. If you
   come back years later having forgotten everything, start here.

3. **The code in `src/` etc. - the worked example (the HOW).**
   Every file is complete, runnable, and **heavily commented**. Comments are written to be
   read like prose: each section says *why* it exists, not just *what* it does. Banners
   like `# === STEP 3: ... ===` mark the flow. `# Extend:` marks where to experiment.

> Rule of thumb: **README to decide, LESSON to understand, code to do.**

## How the code is commented (so future-you can navigate fast)

- **Module docstring** at the top of every file = WHAT this file is, WHY it exists, HOW it
  fits the pipeline, WHERE it sits in the flow.
- **`# === STEP n: ... ===`** banners = the flow, in order.
- **Inline `# why:`** comments = the reasoning behind a non-obvious choice.
- **`# Extend:`** = a safe place to tweak and learn by experiment.
- **`if __name__ == "__main__":`** = how to run the file directly.

## A study method that actually sticks

1. **Read LESSON.md actively** - don't just skim. Re-explain each concept out loud in your
   own words. If you can't, re-read that section.
2. **Run the code in `--demo` mode first** (foundational projects ship synthetic data so they
   run with zero setup). Watch it work before you understand every line.
3. **Read the code top-to-bottom in the "reading order"** LESSON.md gives you.
4. **Break it on purpose.** Change a number, delete a step, see the error. Understanding is
   built from broken things you fixed.
5. **Do the "Check your understanding" questions** at the end of each LESSON.
6. **Then do the real project** (real dataset, real training) and ship the artifact.
7. **Write the build-in-public post** - teaching others is the strongest memory anchor.

## Re-using this in future years

- Each LESSON.md is self-contained, so you can jump to any single concept (e.g. "how does
  backprop work again?" -> open `03-neural-net-from-scratch/LESSON.md`).
- `GLOSSARY.md` (repo root) is a master dictionary of every term used across all projects.
- The code runs offline in `--demo` mode, so the examples will still execute even if the
  exact datasets or model APIs have changed.

## Difficulty ramp (so you always know what "basics" you already have)

Phase 1 (P1-P2): supervised learning, pipelines, leakage, metrics, reproducibility.
Phase 2 (P3-P4): neurons, backprop, gradient descent, CNNs/transfer learning.
Phase 3 (P5-P6): tokenization, transformers, attention.
Phase 4 (P7-P9): embeddings, RAG, fine-tuning, evaluation.
Phase 5 (P10-P13): agents, multi-agent, multimodal, RL for reasoning.
Phase 6 (P14-P15): serving, MLOps, full-system design.

Every later phase assumes the vocabulary of the earlier ones - which is why LESSON.md files
keep re-pointing you back to the foundations when a term first appeared there.
