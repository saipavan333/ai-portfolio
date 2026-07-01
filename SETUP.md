# Setup Guide

## 0. One-time: create your dedicated AI branch
You asked specifically for a new GitHub branch for AI projects. Two clean options.

### Option A - a branch inside an existing repo
```bash
# from the root of an existing repo on your machine
git checkout main            # or master
git pull                     # get latest
git checkout -b ai-portfolio # create + switch to the new branch
# copy this whole bundle into the repo, then:
git add .
git commit -m "Start AI portfolio: 15-project blueprint"
git push -u origin ai-portfolio
```
Your branch URL becomes:
`https://github.com/<your-username>/<repo>/tree/ai-portfolio`

### Option B - a brand-new dedicated repo (recommended for a clean portfolio)
```bash
# create an empty repo named ai-portfolio on github.com first (no README), then:
cd ai-portfolio-bundle           # this unzipped folder
git init
git add .
git commit -m "Start AI portfolio: 15-project blueprint"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-portfolio.git
git push -u origin main
# then make the working branch you asked for:
git checkout -b ai-portfolio
git push -u origin ai-portfolio
```

Tip: do each project on its own short-lived branch (e.g. `git checkout -b p07-rag`),
open a Pull Request into `ai-portfolio`, and let GitHub Actions run your tests. That
mirrors real team workflow and makes your commit history tell a story.

## 1. Accounts & CLIs (free tiers are enough)
```bash
pip install --upgrade huggingface_hub wandb kaggle
huggingface-cli login     # paste token from huggingface.co/settings/tokens
wandb login               # paste key from wandb.ai/authorize
# Kaggle: download kaggle.json from kaggle.com/settings -> place at ~/.kaggle/kaggle.json
```

## 2. Python environment (per project)
```bash
cd 01-tabular-ml-pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Docker sanity check
```bash
docker --version
docker run --rm hello-world
```

## 4. Suggested per-project loop
1. Read the project README (why/what/how/where).
2. Create a branch: `git checkout -b pXX-shortname`.
3. Build in `src/`, experiment in `notebooks/`, track with W&B.
4. Fill the README results section + record the Definition-of-Done checklist.
5. Ship the public artifact (repo/Space/notebook/report/image).
6. Open a PR into `ai-portfolio`; write a short build-in-public post.
