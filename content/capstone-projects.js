(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "capstone-projects",
  name: "Capstone Projects & Learning Path",
  language: "Practice",
  group: "Practice",
  navLabel: "Capstone Projects",
  tagline: "A guided order to study this deck, plus portfolio-grade projects that force you to combine everything into things you can ship.",
  color: "#EAB308",
  readMinutes: 24,
  sections: [
    {
      id: "path",
      title: "The recommended learning path",
      level: "core",
      body: [
        { type: "p", text: "This deck is ordered as a curriculum — top to bottom is a deliberate sequence. You do **not** need to finish every math proof before touching code, but the dependencies are real: attention only makes sense after neural nets, which only make sense after linear algebra and calculus. Here is the path most people should take." },
        { type: "table",
          headers: ["Phase", "Read", "Goal", "You can then…"],
          rows: [
            ["1 · Foundations", "Math Foundations → Classical ML", "the mental model + your first models", "predict/classify on tabular data, evaluate honestly"],
            ["2 · Deep learning", "Neural Networks (all 6)", "build & train nets from scratch, then in PyTorch", "train a CNN on images, understand backprop cold"],
            ["3 · Language", "NLP (all 4)", "text → vectors → attention → Transformers", "implement self-attention; read any NLP paper"],
            ["4 · LLMs", "LLMs (all 8)", "build a GPT, then use/fine-tune/deploy big ones", "train a GPT, ship a RAG app, fine-tune with LoRA"],
            ["5 · Depth", "Reinforcement Learning + MLOps", "reward-based learning + shipping to production", "understand RLHF end-to-end; deploy + monitor"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Study loop that works:** read a section → do its derivations by hand → implement the from-scratch version → do ONE project from that section → move on. Active recall beats re-reading: use the flashcards (press `f`) after each section. Don't let the perfect (finishing every proof) be the enemy of momentum." },
        { type: "callout", variant: "note", text: "**If you only have a weekend per phase:** Phase 1 → do the Titanic/tabular project. Phase 2 → MNIST from scratch. Phase 3 → implement attention. Phase 4 → train the nanoGPT from the Build-a-GPT section and build a small RAG. That single spine (`tabular → MNIST → attention → GPT → RAG`) touches the whole field." },
      ]
    },
    {
      id: "classical",
      title: "Projects · Classical ML & data",
      level: "core",
      body: [
        { type: "p", text: "Build these with NumPy/pandas/scikit-learn. The point is the *whole pipeline* — framing, features, evaluation, leakage-avoidance — not just calling `.fit()`." },
        { type: "list", ordered: true, items: [
          "**Tabular predictor, done right.** Take a real dataset (Titanic, house prices, telco churn). Build a leak-free `Pipeline` + `ColumnTransformer`, compare logistic regression / random forest / gradient boosting with proper cross-validation, and write up which metric matters and why. This is the single most representative ML job task.",
          "**From-scratch model zoo.** Implement linear regression (normal eq + GD), logistic regression, k-NN, a decision tree, and k-means in pure NumPy, each with a gradient/logic check against scikit-learn. You will never fear these algorithms again.",
          "**Customer segmentation.** Cluster a customer/behavior dataset (k-means + GMM + DBSCAN), pick k with silhouette, visualize with PCA and t-SNE, and tell the story each cluster represents.",
          "**Anomaly / fraud detector.** Handle severe class imbalance: proper metrics (PR-AUC), resampling, threshold tuning, and an honest error analysis. Great for practicing evaluation.",
          "**A Kaggle-style feature-engineering bake-off.** Fix the model, and improve the score only by engineering features. Measure each feature's contribution. Internalize 'features > model' for tabular.",
        ]},
      ]
    },
    {
      id: "deep",
      title: "Projects · Deep learning & vision",
      level: "core",
      body: [
        { type: "list", ordered: true, items: [
          "**MNIST from scratch (rite of passage).** A 2-layer MLP with hand-written backprop in NumPy, reaching ~97% test accuracy — no framework. Then re-implement in PyTorch and match it.",
          "**Build micrograd.** A scalar autograd engine (`Value` class + `backward()`) that reproduces your MLP's gradients. The best exercise for *truly* understanding autodiff.",
          "**CIFAR-10 CNN + transfer learning.** Train a small CNN from scratch, then fine-tune a pretrained ResNet and compare. Plot learning curves; diagnose over/underfitting.",
          "**Char-RNN vs Transformer.** Train an LSTM to generate text, then the Build-a-GPT model on the same corpus. Compare quality, training speed, and code complexity — feel *why* Transformers won.",
          "**An optimizer/regularization ablation study.** Fix a model; sweep SGD vs Adam, dropout on/off, batchnorm on/off, init schemes. Produce the plots that make the Training-Deep-Nets section concrete.",
        ]},
        { type: "callout", variant: "good", text: "These five, done well and written up, are already a portfolio that clears most ML-engineer screening bars. Depth (one project explained thoroughly) beats breadth (ten half-finished notebooks)." },
      ]
    },
    {
      id: "llm",
      title: "Projects · LLMs & applied AI",
      level: "core",
      body: [
        { type: "p", text: "This is where the deck is aimed: being able to *build* with and *build* LLMs. Do at least the first two." },
        { type: "list", ordered: true, items: [
          "**Train your own GPT.** Follow Build-a-GPT: train the character-level Transformer on a corpus you care about (an author, your code, chat logs), then upgrade to a BPE tokenizer and scale it up. Generate samples and write up what changed with scale.",
          "**Chat with your documents (RAG).** Build a full RAG app: ingest → chunk → embed → vector index (FAISS/Chroma) → retrieve → cite → generate. Add hybrid search + a reranker and measure retrieval quality (recall@k). The most in-demand applied-LLM skill.",
          "**Fine-tune a small open model with LoRA/QLoRA.** Take Llama/Mistral/Qwen-small, SFT it on a custom style or task with `peft`+`trl`, then align preferences with DPO. Evaluate win-rate vs the base model.",
          "**Build an agent from scratch.** An LLM-in-a-loop with 2–3 tools (calculator, web/RAG search, code exec) and a ReAct loop — no framework first, then port to LangGraph. Add guardrails, retries, and a cost cap.",
          "**A local, private AI assistant.** Run an open model via Ollama/vLLM, wrap it in a FastAPI service with streaming + conversation memory + RAG over your notes. Fully offline, yours.",
          "**Prompt-eval harness.** Build a small eval set + LLM-as-judge to A/B test prompts and models quantitatively. Turns 'prompt engineering' from vibes into measurement.",
        ]},
      ]
    },
    {
      id: "capstone",
      title: "Grand capstones (combine everything)",
      level: "core",
      body: [
        { type: "p", text: "Each of these spans multiple tracks and is genuinely portfolio-defining. Pick one and go deep — ship it, deploy it (MLOps track), and write it up." },
        { type: "list", ordered: true, items: [
          "**End-to-end ML product.** Data → model → FastAPI + Docker → deployed API → monitoring dashboard with drift detection. Demonstrates the *entire* lifecycle, not just modeling.",
          "**A domain-specialized assistant.** RAG + a LoRA-fine-tuned model + an agent loop over a real domain (legal, medical notes, a codebase). Combines Phases 3–5.",
          "**Reproduce a paper.** Pick a tractable paper (a small Transformer variant, a DPO result, a CNN architecture) and reproduce its core claim from scratch. This is exactly what research engineers do.",
          "**From-scratch mini-GPT-2.** Use nanoGPT to reproduce GPT-2 (124M) on a subset of the web, then instruction-tune it. You will have built a real LLM end to end.",
          "**RLHF in miniature.** Train a tiny reward model from preference pairs, then improve a small LM's outputs with PPO or DPO. Makes the alignment pipeline tangible.",
        ]},
        { type: "callout", variant: "tip", text: "**Ship in public.** Put the code on GitHub with a real README (problem, approach, results, what you'd do next), write a short blog post, and record a 2-minute demo. A shipped, explained project is worth more than any certificate — it *is* the credential in this field." },
      ]
    },
    {
      id: "resources",
      title: "The canonical resources to keep nearby",
      level: "deep",
      body: [
        { type: "p", text: "This deck is self-contained, but these are the field's landmark references — worth owning/bookmarking for the whole journey:" },
        { type: "link", url: "https://www.deeplearningbook.org/", text: "Goodfellow, Bengio & Courville — Deep Learning (free; the foundational text)" },
        { type: "link", url: "https://d2l.ai/", text: "Dive into Deep Learning (free, interactive, code-first — an excellent companion)" },
        { type: "link", url: "https://github.com/rasbt/LLMs-from-scratch", text: "Raschka — Build a Large Language Model (From Scratch)" },
        { type: "link", url: "https://karpathy.ai/zero-to-hero.html", text: "Karpathy — Neural Networks: Zero to Hero (the best from-scratch video series)" },
        { type: "link", url: "https://huggingface.co/learn", text: "HuggingFace Courses (LLM, NLP, RL, Deep RL) — free and hands-on" },
        { type: "link", url: "https://www.statlearning.com/", text: "James et al. — An Introduction to Statistical Learning (free; the classical-ML bible)" },
        { type: "link", url: "https://incompleteideas.net/book/the-book-2nd.html", text: "Sutton & Barto — Reinforcement Learning: An Introduction (free)" },
        { type: "link", url: "https://huyenchip.com/2024/", text: "Chip Huyen — Designing Machine Learning Systems & AI Engineering (the production/MLOps references)" },
      ]
    },
  ],

  packages: [
    { name: "numpy · pandas", why: "the data-handling and from-scratch-math bedrock of every project" },
    { name: "scikit-learn", why: "classical ML, pipelines, evaluation — Phase 1 & 2 projects" },
    { name: "pytorch", why: "all deep-learning, NLP, and LLM projects" },
    { name: "transformers · peft · trl", why: "use and fine-tune open LLMs (Phase 4)" },
    { name: "sentence-transformers · faiss", why: "embeddings + vector search for the RAG project" },
    { name: "fastapi · docker", why: "ship any project as a real service (capstones + MLOps)" },
  ],

  gotchas: [
    "Finish and *write up* one project before starting the next — a portfolio of one deep project beats ten abandoned notebooks.",
    "Always hold out a real test set and avoid leakage; an impressive score from a leaked pipeline is worse than an honest lower one.",
    "Scope down aggressively: a small model on a small dataset that you fully understand teaches more than a giant one you can't debug.",
    "Version your data and experiments (Git + a tracking tool) from project one — future-you needs to reproduce results.",
    "For LLM projects, watch cost/token usage from the start; add caps before you run a loop that calls an API thousands of times.",
    "Reproducing a result (paper, benchmark) is harder and more educational than it looks — budget extra time and treat mismatches as learning.",
  ],

  flashcards: [
    { q: "What is the recommended study loop per section?", a: "Read → do the derivations by hand → implement the from-scratch version → do one project → active-recall with flashcards. Momentum over perfection." },
    { q: "What single project spine touches the whole field?", a: "Tabular predictor → MNIST from scratch → implement attention → train a nanoGPT → build a RAG app. One project per phase." },
    { q: "Why do depth over breadth on projects?", a: "One fully-built, well-explained, shipped project demonstrates the whole lifecycle and clears hiring bars; many half-finished notebooks demonstrate none of it." },
    { q: "What's the most in-demand applied-LLM skill to practice?", a: "Retrieval-Augmented Generation (RAG): ingest→chunk→embed→index→retrieve→cite→generate, with hybrid search + reranking and measured retrieval quality." },
    { q: "What makes an ML project 'portfolio-grade'?", a: "It's shipped in public: GitHub + a real README (problem/approach/results/next), a short write-up, and a demo. The shipped artifact is the credential." },
    { q: "How should you evaluate an ML project to avoid fooling yourself?", a: "A sacred held-out test set, leak-free pipelines, the metric that matches the business goal (not just accuracy), and an honest error analysis." },
  ],

  cheatsheet: [
    { label: "Phase 1", code: "Math + Classical ML → tabular project" },
    { label: "Phase 2", code: "Neural Nets → MNIST from scratch" },
    { label: "Phase 3", code: "NLP → implement attention" },
    { label: "Phase 4", code: "LLMs → train nanoGPT + build RAG" },
    { label: "Phase 5", code: "RL + MLOps → RLHF understanding + deploy" },
    { label: "Study loop", code: "read → derive → build → 1 project → recall" },
    { label: "Portfolio rule", code: "depth > breadth; ship in public" },
    { label: "First LLM build", code: "Build-a-GPT section → your own corpus" },
    { label: "Most-hireable app", code: "RAG over your documents" },
    { label: "Ship stack", code: "FastAPI + Docker + monitoring" },
  ],
});
