# Deep Dive AI 🧠

A static, offline web app that teaches **machine learning, NLP, neural networks, and LLMs** from first principles — for a working developer who wants to *understand the math* and *build things from scratch*, all the way up to training your own GPT.

Unlike the companion review decks (which are fast refreshers), this is a **curriculum**: each section is a 40–60 minute lesson that goes intuition → history → the math (with derivations) → **from-scratch NumPy** → the production framework (scikit-learn / PyTorch / HuggingFace) → projects → references.

**Tracks** (in sidebar order — a deliberate learning path):

- **Math Foundations** — Linear Algebra · Calculus & Optimization · Probability & Statistics
- **Classical Machine Learning** — Foundations & workflow · Linear & Logistic Regression · Regularization · Trees & Ensembles · SVMs · kNN & Naive Bayes · Clustering · Dimensionality Reduction (PCA) · Model Evaluation · Feature Engineering
- **Neural Networks & Deep Learning** — NN foundations · **Backpropagation (full derivation)** · Training deep nets (optimizers, init, norm, regularization) · PyTorch · CNNs (computer vision) · RNNs & LSTMs
- **Natural Language Processing** — Foundations & tokenization · Text representation (TF-IDF, word2vec) · Seq2seq & attention · **Transformers from scratch**
- **Large Language Models** — What they are & how they work · **Build a GPT from scratch** · Pretraining & scaling laws · Using open models · Prompt engineering · RAG & embeddings · Fine-tuning (LoRA / RLHF) · Agents & tool use
- **Reinforcement Learning** — Foundations (MDPs, Q-learning) · Deep RL & the RLHF connection
- **MLOps & Deployment** — Serving & deployment · Production (tracking, quantization, GPUs, monitoring)
- **Practice** — Capstone project ideas across the whole stack

Related topics are grouped in the sidebar under a collapsible parent. Every section ends with **project suggestions** and a **"Go deeper"** list of the best books, courses, and papers.

## What makes this deck different

- **Real math rendering.** Equations are typeset with [KaTeX](https://katex.org/), vendored locally in `vendor/katex/` so it works fully offline (no CDN, no network). Derivations are shown in full, not hand-waved.
- **Build-it-from-scratch.** Every algorithm is implemented in plain NumPy first (so you understand the internals), then shown with the framework you'd actually use.
- **Rigorous but readable.** Intuition and callouts (💡 tips, ⚠️ gotchas, 📌 notes) carry you through the heavy parts.

## Run it

No build, no server, no dependencies. Just open the file:

```bash
# double-click index.html, or:
xdg-open index.html      # Linux
open index.html          # macOS
```

> Content and the math renderer load via plain `<script>`/`<link>` tags, so it works directly from `file://`.
> If your browser ever blocks local files, serve the folder instead:
> `python3 -m http.server` then visit http://localhost:8000

## Features

- 🧮 **Typeset math** (KaTeX, offline) — inline `$…$` and display equations, light/dark aware.
- 🌗 **Dark / light theme** — toggle with `t`, remembered across visits.
- 🔎 **Global search** (`/`) — jump to any concept, package, or cheat-card entry.
- 🃏 **Flashcards + quiz** (`f`) — active-recall practice per topic.
- 🧭 **Sticky nav + on-this-page TOC** with scroll-spy; the active group auto-expands.
- 📈 **Reading-progress bar** + **back-to-top** (`g` `g`).
- ↔ **Prev / Next** cards for linear study; 🔗 copyable section deep-links.
- 📋 Syntax-highlighted code (Python, Bash, JSON) with **copy** buttons.
- 🔀 **Cross-links** to the companion Backend and Frontend/GUI review decks.
- 🖨 **Print-friendly** (Ctrl/Cmd+P expands everything, equations included).

## Project structure

```
index.html            # shell (loads KaTeX, engine, content)
vendor/katex/         # vendored KaTeX (js, css, woff2 fonts) — offline math
css/                  # theme, layout, components, animations, math
js/                   # highlight, render (+ math block), nav, search, flashcards, theme, ux, app
content/              # one file per topic  (window.FRAMEWORKS.push({...}))
  _schema.md          # authoring guide: block types, math rules, escaping
```

## Editing or adding content

All content lives in `content/*.js`. Each file registers one topic object — see
[`content/_schema.md`](content/_schema.md) for the shape, the **math conventions**
(`String.raw` for display math, doubled backslashes for inline `$…$`), and escaping rules.

Validate a content file after editing:

```bash
node -e "global.window={FRAMEWORKS:[]};require('./content/nn-backprop.js');console.log('ok', window.FRAMEWORKS.at(-1).sections.length,'sections')"
```

To add a topic: create `content/<id>.js`, then add a matching `<script src="content/<id>.js"></script>`
line in `index.html`, **inside its track's group block** (nav groups must stay contiguous). The
sidebar, search, and flashcards pick it up automatically.
