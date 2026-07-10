(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "llm-pretraining",
  name: "Pretraining & Scaling Laws",
  language: "LLMs",
  group: "Large Language Models",
  navLabel: "Pretraining & Scaling",
  tagline: "How frontier models are actually built — the expensive part: web-scale data, the $C\\approx6ND$ compute law, Chinchilla-optimal scaling, and the distributed-training machinery that turns a GPU cluster into GPT.",
  color: "#DB2777",
  readMinutes: 51,
  sections: [
    {
      id: "why",
      title: "What pretraining actually is",
      level: "core",
      body: [
        { type: "p", text: "You already know the transformer architecture (from the NLP track) and how to train a neural net (from the Neural Networks track). **Pretraining** is what happens when you take that architecture, make it very large, and run it against a substantial fraction of the public internet for months on thousands of GPUs. It is the step that turns a randomly-initialized pile of matrices into a base model that has absorbed the statistics of human text — grammar, facts, code, reasoning patterns — purely by learning to predict the next token." },
        { type: "p", text: "This is the single most expensive artifact humans routinely build in software. A frontier pretraining run costs tens of millions of dollars, burns megawatts, and cannot be casually repeated. So the engineering here is dominated by one question the rest of ML rarely asks seriously: **given a fixed compute budget, what is the best model you can build?** That question has a surprisingly clean, quantitative answer — the scaling laws — and this deck derives it." },
        { type: "table",
          headers: ["Phase", "What it does", "Cost", "Data"],
          rows: [
            ["**Pretraining**", "learn language from raw text (next-token)", "$10\\text{M}$–$100\\text{M}+$", "trillions of unlabeled tokens"],
            ["Continued pretraining", "adapt a base model to a new domain", "thousands–millions", "billions of domain tokens"],
            ["Fine-tuning / SFT", "teach instruction-following", "hundreds–thousands", "thousands–millions of examples"],
            ["RLHF / alignment", "make it helpful & safe", "moderate", "human preference pairs"],
          ]
        },
        { type: "p", text: "This section covers the first row — the foundation the other three build on. The output of pretraining is a **base model** (a.k.a. foundation model): it completes text but does not yet follow instructions. Turning it into a chatbot is the *post-training* story, covered in the alignment sections. Here we build the raw engine." },
        { type: "callout", variant: "note", text: "**How to read this deck.** Pretraining is where ML stops being a laptop activity and becomes a systems-engineering problem. We keep the math honest — you will derive the $6ND$ compute law and the Chinchilla optimum by hand — but the code is necessarily illustrative: nobody reproduces a frontier run in a code block. The last sections show what you *can* do at small scale (nanoGPT-style), which teaches every concept faithfully at 1/1{,}000{,}000 the cost." },
      ]
    },

    {
      id: "objective",
      title: "The objective: next-token prediction at scale",
      level: "core",
      body: [
        { type: "p", text: "The entire training signal is one idea: **predict the next token given all previous tokens.** No labels, no annotation — the text is its own supervision. This is **self-supervised learning**, and it is why pretraining can consume trillions of tokens: every position in every document is a free training example." },
        { type: "p", text: "Tokenize a document into a sequence $x_1, x_2, \\dots, x_T$ (using the BPE tokenizer from the tokenization section). The model, with parameters $\\theta$, defines a probability distribution over the next token conditioned on the prefix. We factor the joint probability of the whole sequence with the chain rule of probability:" },
        { type: "math", tex: String.raw`p_\theta(x_1, \dots, x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_1, \dots, x_{t-1})` },
        { type: "p", text: "This factorization is *exactly* what causal masking in the transformer enforces (position $t$ may only attend to positions $\\le t$). Maximizing the likelihood of the data is minimizing its negative log-likelihood, which is the **cross-entropy loss** averaged over every token:" },
        { type: "math", tex: String.raw`\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})` },
        { type: "p", text: "That is the whole objective. It is the same cross-entropy you met in classification — a $V$-way classification problem (one class per vocabulary token) solved at every position. The final linear layer produces logits $z \\in \\mathbb{R}^{V}$, softmax turns them into probabilities, and we penalize the log-probability assigned to the token that actually came next:" },
        { type: "math", tex: String.raw`p_\theta(x_t = w \mid x_{<t}) = \frac{e^{z_w}}{\sum_{v=1}^{V} e^{z_v}}, \qquad \mathcal{L}_t = -\log p_\theta(x_t \mid x_{<t})` },
        { type: "callout", variant: "tip", text: "**Why this simple objective produces intelligence.** To predict the next token well across *all* of human text, a model is implicitly forced to learn grammar, world facts, arithmetic, code semantics, and chains of reasoning — because all of those are needed to reduce next-token surprise on the relevant documents. Compression *is* understanding: the better you model the data, the more structure you must have internalized. This is the core empirical bet of the whole field." },
        { type: "code", lang: "py", code: "import torch, torch.nn.functional as F\n\n# The pretraining loss in three lines, given a batch of token ids.\n# tokens: (B, T+1) integer tensor from a tokenized, packed corpus.\ndef lm_loss(model, tokens):\n    inputs  = tokens[:, :-1]          # x_1 .. x_T      (B, T)\n    targets = tokens[:, 1:]           # x_2 .. x_{T+1}  (B, T)  -> shifted by one\n    logits  = model(inputs)           # (B, T, V)\n    # cross entropy over the vocabulary, averaged over all B*T positions\n    return F.cross_entropy(\n        logits.reshape(-1, logits.size(-1)),   # (B*T, V)\n        targets.reshape(-1),                    # (B*T,)\n    )\n\n# The 'shift by one' is the entire trick: the label for position t is the\n# input at position t+1. No human labelling anywhere in the pipeline." },
        { type: "callout", variant: "gotcha", text: "Documents are **packed**, not padded. Rather than one document per sequence (wasting compute on padding), the corpus is concatenated into one giant token stream and chopped into fixed-length blocks of, say, $T=4096$. A special end-of-document token marks boundaries; some implementations also block cross-document attention so a sequence spanning two documents can't leak context across the seam." },
      ]
    },

    {
      id: "data",
      title: "Data: the real moat",
      level: "core",
      body: [
        { type: "p", text: "Architecture is nearly commoditized — everyone uses a decoder-only transformer with minor variations (next section). What separates a good base model from a mediocre one at the same size is overwhelmingly the **data**: how much, how clean, how diverse, how deduplicated. Practitioners say *data is the moat*, and the leading labs treat their exact data mix as their most closely guarded secret." },
        { type: "heading", text: "Where the tokens come from" },
        { type: "p", text: "The raw material is the public web, plus curated high-value sources. The canonical open datasets tell the story of the field's progression:" },
        { type: "table",
          headers: ["Corpus", "Year", "Size", "What it is"],
          rows: [
            ["**Common Crawl**", "2008–", "petabytes of raw HTML", "the monthly web scrape everything is built from"],
            ["The Pile", "2020", "~825 GB / ~340B tokens", "EleutherAI's curated 22-source mix (books, code, arXiv, PubMed)"],
            ["C4", "2019", "~156B tokens", "cleaned Common Crawl used to train T5"],
            ["RedPajama", "2023", "~1.2T (v1), ~30T (v2)", "open reproduction of the LLaMA data recipe"],
            ["**FineWeb**", "2024", "**15T tokens**", "HuggingFace's rigorously filtered CommonCrawl (96 snapshots, 2013–2024)"],
            ["FineWeb-Edu", "2024", "~1.3T tokens", "FineWeb filtered by an educational-quality classifier"],
          ]
        },
        { type: "p", text: "A modern open model (Llama 3, for instance) is trained on **~15 trillion tokens**. To put that in perspective: at a human reading speed of 250 words/minute, reading 15T tokens would take on the order of 100{,}000 years. The model sees it in a couple of months." },
        { type: "heading", text: "The data pipeline: extraction → filter → dedup → mix" },
        { type: "p", text: "Turning raw Common Crawl WARC files into training tokens is a multi-stage funnel that typically discards **>90%** of the input. The stages, roughly in order:" },
        { type: "list", ordered: true, items: [
          "**Text extraction.** Pull readable text out of HTML (tools like trafilatura), stripping boilerplate, nav bars, and markup.",
          "**Language ID.** A fastText classifier keeps documents in the target language(s) above a confidence threshold.",
          "**Quality filtering.** Heuristics (Gopher/C4 rules): drop documents that are too short, have too few real words, too high a symbol-to-word ratio, don't end lines with punctuation, or contain excessive repetition. Increasingly, a **model-based** quality classifier scores each document (FineWeb-Edu uses an LLM-labeled educational-quality scorer).",
          "**Deduplication.** Near-duplicate removal with **MinHash** over n-grams. This is the highest-leverage step: the web is enormously redundant, and duplicated data both wastes compute and causes memorization.",
          "**Safety / PII filtering.** Remove adult content, remove or mask personal data.",
          "**Mixing & upsampling.** Blend sources in tuned proportions — high-quality domains (books, code, Wikipedia, arXiv) are *upsampled* relative to their raw web frequency because they punch above their weight.",
        ]},
        { type: "callout", variant: "note", text: "**Dedup is subtler than it looks.** FineWeb found that *global* deduplication across all 96 CommonCrawl snapshots actually **hurt** quality — it disproportionately deleted older, higher-quality pages that got re-crawled, leaving a residue of low-quality junk. Deduplicating *within each snapshot independently* worked better. The lesson: more aggressive filtering is not monotonically better, and every stage must be validated with ablations." },
        { type: "heading", text: "Data contamination — the benchmark trap" },
        { type: "p", text: "Because the training set is a large fraction of the web, it very often *contains the test sets* used to evaluate the model — MMLU questions, GSM8K problems, coding benchmarks all leak onto the internet. When evaluation data appears in training data, benchmark scores become inflated and meaningless. This is **data contamination** (or *test-set leakage*), and detecting/removing it (decontamination via n-gram overlap against known benchmarks) is now a mandatory pipeline stage. Suspiciously high scores on public benchmarks that don't transfer to private held-out evals are the classic symptom." },
        { type: "callout", variant: "gotcha", text: "**Tokens are counted after tokenization, not before.** '15 trillion tokens' means 15T BPE subword units, not 15T words or characters. A rough conversion for English: $1\\text{ token} \\approx 0.75\\text{ words} \\approx 4\\text{ characters}$. When you compare dataset sizes, make sure everyone is counting the same unit — a corpus quoted in GB, in words, and in GPT-2 tokens are three different numbers." },
      ]
    },

    {
      id: "scaling-laws",
      title: "Scaling laws: the equation that reshaped the field",
      level: "core",
      body: [
        { type: "p", text: "In 2020, Kaplan et al. (OpenAI) discovered something remarkable: the test loss of a language model is a **smooth power-law function** of scale, spanning many orders of magnitude. Loss decreases predictably as you increase any of three quantities — parameters $N$, dataset size $D$, or compute $C$ — as long as the other two are not bottlenecks. Empirically, holding the others fixed:" },
        { type: "math", tex: String.raw`L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}` },
        { type: "p", text: "The exponents are small (around $\\alpha_N \\approx 0.07$), which is *why scale is so relentless*: on a log-log plot, loss vs. compute is a straight line with no sign of bending. Doubling compute reliably shaves a predictable slice off the loss. This is what convinced the field that 'just make it bigger' was a viable multi-year strategy — you could **extrapolate** the loss of a model you hadn't built yet." },
        { type: "heading", text: "The Chinchilla correction (2022) — the important one" },
        { type: "p", text: "Kaplan's recipe, followed literally, said: given more compute, spend most of it on **more parameters** and relatively little on more data. This produced the era of giant, undertrained models — GPT-3 (175B params) was trained on only ~300B tokens. In 2022, Hoffmann et al. (DeepMind) re-ran the experiment more carefully and found Kaplan had under-weighted data. Their result — the **Chinchilla scaling law** — models loss as a sum of three terms:" },
        { type: "math", tex: String.raw`L(N, D) = \underbrace{E}_{\text{irreducible}} + \underbrace{\frac{A}{N^{\alpha}}}_{\text{finite model}} + \underbrace{\frac{B}{D^{\beta}}}_{\text{finite data}}` },
        { type: "p", text: "with fitted constants roughly $A \\approx 406$, $B \\approx 410$, $E \\approx 1.69$, $\\alpha \\approx 0.34$, $\\beta \\approx 0.28$ (nats/token). The first term $E$ is the **entropy of language itself** — the loss you cannot beat no matter how large the model. The other two terms shrink as you grow $N$ and $D$ respectively." },
        { type: "heading", text: "Deriving the compute-optimal point" },
        { type: "p", text: "Here is the payoff. Training compute is (from the next section) $C \\approx 6ND$. So for a **fixed compute budget** $C$, we have a constraint $D = C/(6N)$, and we want to choose $N$ to minimize the loss. Substitute the constraint into $L(N,D)$ and minimize over $N$:" },
        { type: "math", tex: String.raw`\min_{N}\; L\!\left(N,\, \tfrac{C}{6N}\right) = E + \frac{A}{N^{\alpha}} + B\left(\frac{6N}{C}\right)^{\beta}` },
        { type: "p", text: "Differentiate with respect to $N$, set to zero. The first data-independent term falls as $N$ grows; the last term rises as $N$ grows — there is a unique interior minimum where they trade off:" },
        { type: "math", tex: String.raw`-\alpha \frac{A}{N^{\alpha+1}} + \beta B\,\frac{6^{\beta}}{C^{\beta}}\,N^{\beta-1} = 0 \;\;\Longrightarrow\;\; N_{\text{opt}} \propto C^{\frac{\beta}{\alpha+\beta}}, \quad D_{\text{opt}} \propto C^{\frac{\alpha}{\alpha+\beta}}` },
        { type: "p", text: "Because $\\alpha \\approx \\beta$, the two exponents are both close to $1/2$: **optimal $N$ and optimal $D$ each scale as roughly $\\sqrt{C}$.** In plain terms — every time you get $4\\times$ the compute, you should roughly *double the model size and double the data*, not pour it all into parameters. Plugging in the fitted constants gives the famous rule of thumb:" },
        { type: "math", tex: String.raw`\boxed{\;D_{\text{opt}} \approx 20 \times N_{\text{opt}}\;}\qquad\text{(about 20 training tokens per parameter)}` },
        { type: "callout", variant: "note", text: "**Chinchilla in one sentence:** DeepMind trained a 70B model (Chinchilla) on 1.4T tokens and it *beat* the 280B Gopher trained on 300B tokens — same compute, smaller model, 4× more data, better results. Overnight, the industry stopped chasing raw parameter count and started chasing the $20{:}1$ token-to-parameter ratio. Every model from Llama onward is a child of this paper. A 2025 robustness re-analysis confirmed the $20{:}1$ ratio holds up under different fitting interpretations." },
        { type: "callout", variant: "tip", text: "**Why real models are trained *past* Chinchilla-optimal.** Chinchilla optimizes the cost of *training*. But a deployed model is run in **inference** billions of times, and a smaller model is cheaper to serve forever. So labs deliberately 'overtrain' a smaller model on far more than 20 tokens/param (Llama 3 8B saw ~15T tokens — roughly $1875{:}1$!) to get a model that is compute-suboptimal to *train* but wonderfully cheap to *run*. Compute-optimal $\\ne$ deployment-optimal." },
      ]
    },

    {
      id: "compute",
      title: "Compute: the 6ND law and what a run costs",
      level: "core",
      body: [
        { type: "p", text: "Everything above rests on one estimate: the total training compute (in floating-point operations, FLOPs) for a dense transformer with $N$ parameters trained on $D$ tokens is approximately" },
        { type: "math", tex: String.raw`\boxed{\; C \approx 6 N D \;}\qquad\text{(FLOPs)}` },
        { type: "p", text: "This little formula is the currency of the whole field — it converts model size and data into a dollar-and-months budget. Let's derive the factor of 6, because it demystifies the whole thing." },
        { type: "heading", text: "Deriving the factor of 6" },
        { type: "p", text: "The dominant cost of a transformer is its matrix multiplications, and every parameter participates in one multiply-accumulate (a multiply and an add) per token that flows through it. Count the FLOPs per token:" },
        { type: "list", ordered: false, items: [
          "**Forward pass:** each of the $N$ parameters does one multiply + one add per token $= 2N$ FLOPs/token.",
          "**Backward pass:** computing gradients costs about **twice** the forward pass — you propagate a gradient to the inputs *and* to the weights, roughly two multiply-adds per parameter $= 4N$ FLOPs/token.",
          "**Total:** $2N + 4N = 6N$ FLOPs per token. Multiply by $D$ tokens: $C \\approx 6ND$.",
        ]},
        { type: "p", text: "The approximation ignores attention's $O(T^2)$ term (small unless sequences are very long relative to width), layernorms, and the softmax — all sub-dominant. For back-of-the-envelope planning, $6ND$ is accurate to within a few percent, and it is what every capabilities team actually uses." },
        { type: "heading", text: "Putting real numbers in" },
        { type: "p", text: "Take a Chinchilla-optimal 70B model: $N = 7\\times10^{10}$, $D = 1.4\\times10^{12}$ tokens." },
        { type: "math", tex: String.raw`C \approx 6 \times (7\times10^{10}) \times (1.4\times10^{12}) \approx 5.9\times10^{23}\ \text{FLOPs}` },
        { type: "p", text: "Now divide by what your hardware delivers. An NVIDIA H100 does ~$10^{15}$ FLOP/s (1 PFLOP/s) at bf16 in theory, but real training runs at **30–50% Model FLOPs Utilization (MFU)** — the rest is lost to memory movement, communication, and pipeline bubbles. At 40% MFU that H100 delivers $4\\times10^{14}$ useful FLOP/s:" },
        { type: "code", lang: "py", code: "N = 70e9            # parameters\nD = 1.4e12          # training tokens (Chinchilla-optimal ~20*N)\nC = 6 * N * D       # total FLOPs\n\ngpu_peak = 1e15     # H100 bf16 peak FLOP/s (~1 PFLOP/s)\nmfu      = 0.40     # realistic model-flops-utilization\nn_gpus   = 1024\n\nuseful_flops_per_s = gpu_peak * mfu * n_gpus\nseconds = C / useful_flops_per_s\nprint(f'{C:.2e} FLOPs total')          # ~5.9e23\nprint(f'{seconds/86400:.1f} GPU-days on {n_gpus} GPUs')  # ~17 days\n\n# Cost: at ~$2/H100-hour rental\nhours = C / (gpu_peak * mfu) / 3600\nprint(f'${hours*2:,.0f} in GPU rental')  # ~a few hundred thousand $" },
        { type: "callout", variant: "gotcha", text: "**MFU is where budgets die.** Peak FLOPs on the spec sheet are a fantasy; you never get them. A well-tuned large run reaches 40–55% MFU, a poorly-tuned one 15–20% — a **3× difference in cost and time** for the identical model. Squeezing MFU (kernel fusion, FlashAttention, overlapping communication with compute, the right parallelism layout) is the core craft of a pretraining infra team." },
        { type: "callout", variant: "tip", text: "**The regulatory FLOP threshold.** $6ND$ escaped the lab: the 2023 US Executive Order and the EU AI Act both use training compute (around $10^{25}$–$10^{26}$ FLOPs) as a reporting trigger for 'frontier' models. When policymakers draw a line for 'dangerously capable AI,' they draw it in units of $6ND$." },
      ]
    },

    {
      id: "distributed",
      title: "Distributed training: how a model too big for one GPU trains anyway",
      level: "core",
      body: [
        { type: "p", text: "A 70B-parameter model in fp32 needs 280 GB just for weights — and training needs several times that for gradients and optimizer state (below). No single GPU (80 GB) holds it. Pretraining is therefore fundamentally a **distributed-systems** problem: split the work across thousands of GPUs and keep them fed. There are four axes of parallelism, and large runs combine all of them (**4D parallelism**)." },
        { type: "table",
          headers: ["Parallelism", "What is split", "Communication", "When to use"],
          rows: [
            ["**Data (DP)**", "the batch — each GPU holds a full model copy, processes different examples", "all-reduce gradients each step", "always; the first and cheapest axis"],
            ["**Tensor (TP)**", "individual weight matrices, split across GPUs within a layer", "all-reduce *inside* every layer (heavy)", "within a fast node (NVLink); model too big for one GPU"],
            ["**Pipeline (PP)**", "layers — GPU 0 holds layers 1–8, GPU 1 holds 9–16, …", "send activations between stages", "across nodes; very deep models"],
            ["**Sequence / context**", "the sequence length dimension of activations", "attention needs cross-GPU comms", "very long context (100k+ tokens)"],
          ]
        },
        { type: "heading", text: "The memory problem: where the VRAM goes" },
        { type: "p", text: "Training memory is dominated not by the weights but by the **optimizer state**. Training with Adam in mixed precision, per parameter you store: the fp16 weight, the fp16 gradient, and — in fp32 — a master weight copy, the first moment $m$, and the second moment $v$. That is roughly:" },
        { type: "math", tex: String.raw`\text{bytes/param} \approx \underbrace{2}_{\text{fp16 } w} + \underbrace{2}_{\text{fp16 } g} + \underbrace{4}_{\text{fp32 } w} + \underbrace{4}_{m} + \underbrace{4}_{v} = 16\ \text{bytes}` },
        { type: "p", text: "So a 70B model needs ~1.1 TB just for parameters + gradients + optimizer state, before a single activation is stored. This is what makes naive data-parallelism (a full copy per GPU) impossible." },
        { type: "heading", text: "ZeRO / FSDP — shard the state, not just the batch" },
        { type: "p", text: "**ZeRO** (Zero Redundancy Optimizer, DeepSpeed) and its PyTorch-native twin **FSDP** (Fully Sharded Data Parallel) fix this. Plain data-parallelism redundantly replicates the optimizer state on every GPU. ZeRO **shards** it: each GPU owns only $1/K$ of the parameters, gradients, and optimizer state. When a layer is needed for the forward pass, its parameters are gathered on-the-fly (all-gather), used, then freed. Three stages shard progressively more:" },
        { type: "list", ordered: false, items: [
          "**ZeRO-1:** shard optimizer state only (~4× memory saving).",
          "**ZeRO-2:** also shard gradients.",
          "**ZeRO-3 / FSDP:** also shard parameters themselves — memory per GPU falls ~linearly with GPU count, at the cost of more communication.",
        ]},
        { type: "heading", text: "Two more essential tricks" },
        { type: "p", text: "**Mixed precision** stores and computes in bf16/fp16 (halving memory and doubling throughput on tensor cores) while keeping an fp32 master copy of the weights for numerical stability. **bfloat16** is preferred over fp16 for its wider exponent range — it rarely overflows, so it needs no loss-scaling. **Gradient (activation) checkpointing** trades compute for memory: instead of storing every layer's activations for the backward pass, store only a few and *recompute* the rest on the fly — roughly a 30% compute increase to fit a much larger model." },
        { type: "code", lang: "py", code: "# Modern PyTorch: FSDP + mixed precision + activation checkpointing.\nimport torch\nfrom torch.distributed.fsdp import FullyShardedDataParallel as FSDP\nfrom torch.distributed.fsdp import MixedPrecision\nfrom torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (\n    checkpoint_wrapper)\n\nbf16 = MixedPrecision(param_dtype=torch.bfloat16,\n                      reduce_dtype=torch.float32,   # reduce grads in fp32\n                      buffer_dtype=torch.bfloat16)\n\n# Wrap each transformer block to recompute its activations in backward.\nfor i, block in enumerate(model.transformer.blocks):\n    model.transformer.blocks[i] = checkpoint_wrapper(block)\n\n# Shard params/grads/optimizer state across the whole world (ZeRO-3 style).\nmodel = FSDP(model, mixed_precision=bf16, device_id=torch.cuda.current_device())\n\n# From here training looks normal — FSDP hides the all-gather / reduce-scatter.\nopt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))" },
        { type: "callout", variant: "note", text: "**Communication is the hidden bottleneck.** All-reduce, all-gather, and reduce-scatter move terabytes of gradients across the cluster every step. This is why interconnect (NVLink within a node ~900 GB/s, InfiniBand across nodes) matters as much as raw FLOPs, and why parallelism layout is chosen to keep the chattiest axis (tensor-parallel) inside a single fast node. A frontier cluster is really a networking achievement wearing a GPU costume." },
      ]
    },

    {
      id: "optimization",
      title: "Optimization at scale: schedules, batch sizes & stability",
      level: "core",
      body: [
        { type: "p", text: "The optimizer is the same **AdamW** from the Neural Networks track, but at this scale the *schedule* and *stability tricks* matter more than the optimizer itself. A single divergence at step 200{,}000 can waste weeks of compute, so pretraining is obsessed with not blowing up." },
        { type: "heading", text: "AdamW and its hyperparameters at scale" },
        { type: "p", text: "The near-universal choice is AdamW with $\\beta_1 = 0.9$, $\\beta_2 = 0.95$ (lower than the classic $0.999$ — large-batch LM training likes faster second-moment adaptation), a small weight decay ($0.1$), and $\\epsilon = 10^{-8}$. Decoupled weight decay (the 'W') applies the regularizer directly to the weights rather than through the gradient, which interacts correctly with adaptive learning rates." },
        { type: "heading", text: "The learning-rate schedule: warmup then cosine decay" },
        { type: "p", text: "This is the single most important schedule in the field. **Linearly warm up** the LR from 0 over the first few thousand steps (a fresh model has garbage gradients; a big early LR diverges instantly), then **cosine-decay** it down to ~10% of its peak over the rest of training:" },
        { type: "math", tex: String.raw`\eta_t = \begin{cases} \eta_{\max}\,\dfrac{t}{t_{\text{warmup}}} & t \le t_{\text{warmup}} \\[2mm] \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\!\left(1 + \cos\dfrac{\pi\,(t - t_{\text{warmup}})}{t_{\text{total}} - t_{\text{warmup}}}\right) & t > t_{\text{warmup}} \end{cases}` },
        { type: "code", lang: "py", code: "import math\n\ndef lr_at(step, max_lr=3e-4, min_lr=3e-5, warmup=2000, total=300_000):\n    if step < warmup:                       # linear warmup\n        return max_lr * step / warmup\n    if step > total:                         # after training, hold at floor\n        return min_lr\n    # cosine decay from max_lr down to min_lr\n    frac = (step - warmup) / (total - warmup)\n    coeff = 0.5 * (1.0 + math.cos(math.pi * frac))   # 1 -> 0\n    return min_lr + coeff * (max_lr - min_lr)\n\nprint(lr_at(0), lr_at(2000), lr_at(150_000), lr_at(300_000))" },
        { type: "callout", variant: "gotcha", text: "**Warmup is not optional.** Skip it and a large-batch run almost always diverges in the first few hundred steps — the untrained model produces enormous, poorly-conditioned gradients that a full-size LR turns into NaNs. Warmup gives Adam's second-moment estimate time to calibrate before the LR reaches full strength." },
        { type: "heading", text: "Batch size and the critical batch size" },
        { type: "p", text: "Pretraining uses gigantic batches — millions of tokens per step (e.g. 4M for GPT-3) — because large batches give a low-variance gradient estimate and parallelize across thousands of GPUs. But there is a **critical batch size**: below it, doubling the batch nearly halves the steps to a target loss (great); above it, you get diminishing returns and are wasting data-parallel compute. Batches are assembled with **gradient accumulation** — sum gradients over many micro-batches before stepping — to hit a target token count that no single GPU could hold." },
        { type: "heading", text: "Stability tricks that keep a run alive" },
        { type: "list", ordered: false, items: [
          "**Gradient clipping** to a global norm of $1.0$ — the single most important stabilizer; caps the occasional exploding gradient before it corrupts the weights.",
          "**bfloat16** everywhere (wide exponent range) instead of fp16 (needs finicky dynamic loss-scaling).",
          "**Z-loss / logit soft-capping** — a small auxiliary penalty keeping the softmax logits from drifting to huge magnitudes.",
          "**Careful initialization** — scale residual-branch weights by $1/\\sqrt{2L}$ (for $L$ layers) so the residual stream variance doesn't explode with depth.",
          "**QK-normalization** — normalizing queries and keys before attention, increasingly standard in 2024–2025 models to prevent attention-logit blowups at scale.",
        ]},
        { type: "callout", variant: "tip", text: "**Read the loss curve like an ECG.** A healthy pretraining loss falls fast then follows a smooth power-law glide. A sudden **spike** signals a bad batch or numerical instability — modern runs auto-detect spikes, roll back to the last checkpoint, skip the offending data shard, and resume. Frontier teams watch this curve 24/7; a silent divergence overnight can cost a week." },
      ]
    },

    {
      id: "architecture",
      title: "Modern architecture choices: what 2025 models actually use",
      level: "core",
      body: [
        { type: "p", text: "The 2017 transformer has been quietly upgraded. Essentially every open model since Llama (2023) shares the same handful of departures from the original — each a small, well-motivated change that improves stability, quality, or inference cost. If you build a modern LM, this is the default recipe." },
        { type: "table",
          headers: ["Component", "Original (2017)", "Modern default", "Why"],
          rows: [
            ["Positional info", "sinusoidal / learned absolute", "**RoPE**", "relative positions, extrapolates to longer context"],
            ["Normalization", "LayerNorm, post-block", "**RMSNorm**, pre-block", "cheaper, more stable at depth"],
            ["FFN activation", "ReLU / GeLU", "**SwiGLU**", "gated activation, better quality per FLOP"],
            ["Attention (KV)", "multi-head (MHA)", "**GQA**", "shrinks the KV-cache for cheap inference"],
            ["Bias terms", "yes", "mostly removed", "no quality loss, slightly more stable"],
            ["Scaling capacity", "dense", "**MoE** (frontier)", "more parameters at fixed FLOPs/token"],
          ]
        },
        { type: "heading", text: "RoPE — rotary position embeddings" },
        { type: "p", text: "Instead of *adding* a position vector to the token embedding, **RoPE** *rotates* each query and key vector by an angle proportional to its position. The dot product between a query at position $m$ and a key at position $n$ then depends only on their **relative** distance $m-n$ — exactly what attention wants. This also lets a model trained on 4k context be extended to longer context by interpolating the rotation frequencies." },
        { type: "math", tex: String.raw`\langle \mathrm{RoPE}(q, m),\, \mathrm{RoPE}(k, n)\rangle = g(q, k,\, m-n)` },
        { type: "heading", text: "RMSNorm — a cheaper LayerNorm" },
        { type: "p", text: "LayerNorm subtracts the mean and divides by the standard deviation. **RMSNorm** drops the mean-centering entirely and just divides by the root-mean-square, with a learned scale $g$ — fewer operations, no measurable quality loss, and it is applied *before* each sub-block (pre-norm) for stable deep-network gradients:" },
        { type: "math", tex: String.raw`\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot g` },
        { type: "heading", text: "SwiGLU — the gated feed-forward" },
        { type: "p", text: "The FFN's ReLU is replaced by a **gated linear unit** using the SiLU/swish activation. Two projections are computed; one gates the other elementwise. It consistently beats plain ReLU/GeLU FFNs at equal compute, so the hidden dimension is scaled down (to $\\frac{2}{3}\\times$) to keep the parameter count matched:" },
        { type: "math", tex: String.raw`\mathrm{SwiGLU}(x) = \big(\mathrm{SiLU}(xW_1) \odot xW_3\big)W_2, \qquad \mathrm{SiLU}(z) = z\,\sigma(z)` },
        { type: "heading", text: "GQA — grouped-query attention" },
        { type: "p", text: "At inference, generating each token requires caching the keys and values of every previous token (the **KV-cache**), and that cache — not the weights — dominates memory for long contexts. **Multi-query attention** shares a *single* K/V head across all query heads (tiny cache, slight quality hit); **grouped-query attention** is the sweet spot: split the query heads into $g$ groups that each share one K/V head. Llama 2/3 use GQA. It shrinks the KV-cache by the grouping factor with almost no quality loss." },
        { type: "heading", text: "Mixture-of-Experts — more parameters, same FLOPs" },
        { type: "p", text: "The frontier's answer to 'how do we get more capacity without more compute per token?' is **Mixture-of-Experts**. Replace each dense FFN with $E$ parallel expert FFNs plus a small **router** that, per token, selects only the top-$k$ (usually 1 or 2) experts to run. A token thus touches only a fraction of the total parameters:" },
        { type: "math", tex: String.raw`y = \sum_{i \in \mathrm{TopK}(g(x))} g_i(x)\, \mathrm{Expert}_i(x), \qquad g(x) = \mathrm{softmax}(x W_r)` },
        { type: "p", text: "A model like Mixtral 8×7B has ~47B *total* parameters but activates only ~13B per token — you pay 13B-model FLOPs to serve 47B-model quality. The catch: all 47B must still fit in memory, the router can collapse (a **load-balancing auxiliary loss** forces even expert usage), and the all-to-all token routing across GPUs is a communication headache. DeepSeek-V3, Mixtral, and many 2024–2025 frontier models are MoEs." },
        { type: "callout", variant: "tip", text: "**The MoE scaling-law wrinkle.** Because active params $\\ne$ total params, MoEs break the simple $6ND$ accounting — the right cost model uses *active* parameters for FLOPs but *total* parameters for memory. Recent (2025) work fits separate scaling laws for the number of experts and the sparsity ratio; the emerging finding is that fairly sparse MoEs (activating a small fraction) are compute-efficient, especially for knowledge-heavy tasks." },
      ]
    },

    {
      id: "evaluation",
      title: "Evaluating a base model",
      level: "core",
      body: [
        { type: "p", text: "A base model is not a chatbot — it completes text, it doesn't answer instructions. So you evaluate it in two complementary ways: the **intrinsic** loss metric (perplexity) and a battery of **extrinsic** benchmarks." },
        { type: "heading", text: "Perplexity — the intrinsic metric" },
        { type: "p", text: "**Perplexity** is just the exponentiated cross-entropy loss. If the average per-token loss (in nats) is $\\mathcal{L}$, then:" },
        { type: "math", tex: String.raw`\mathrm{PPL} = \exp(\mathcal{L}) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})\right)` },
        { type: "p", text: "Intuitively, perplexity is the model's **effective branching factor** — how many equally-likely tokens it feels it is choosing between at each step. A perplexity of 1 is perfect (it always knows the next token); uniform guessing over a 50k vocabulary is a perplexity of 50{,}000. Lower is better, and it is the number the scaling laws predict. But it is only comparable *across the same tokenizer and test set* — a model with a bigger vocabulary has a different per-token perplexity for free." },
        { type: "heading", text: "Benchmarks — the extrinsic metrics" },
        { type: "p", text: "Perplexity on held-out text doesn't tell you if the model can *do* anything, so base models are scored on standardized benchmarks, usually in a **few-shot** setting (prepend a handful of examples, then measure accuracy on the answer):" },
        { type: "table",
          headers: ["Benchmark", "Tests", "Format"],
          rows: [
            ["**MMLU**", "broad knowledge across 57 subjects", "4-way multiple choice"],
            ["**HellaSwag**", "commonsense sentence completion", "pick the plausible ending"],
            ["**ARC**", "grade-school science reasoning", "multiple choice"],
            ["**WinoGrande**", "pronoun-resolution commonsense", "binary choice"],
            ["**GSM8K**", "grade-school math word problems", "generate the answer"],
            ["**HumanEval**", "Python code generation", "pass@k on unit tests"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**Benchmark scores are softer than they look.** They are sensitive to prompt phrasing, few-shot example choice, and the exact scoring code (do you compare log-likelihoods of answer choices, or parse generated text?). Two labs reporting 'MMLU 70%' may have measured differently. Worse, **contamination** (from the data section) silently inflates scores. Treat a leaderboard number as a rough signal, always verify on a private held-out eval, and prefer the LM-Eval-Harness for apples-to-apples comparison." },
        { type: "callout", variant: "note", text: "**Base vs. instruct benchmarks differ.** MMLU/HellaSwag suit base models. Chatbot-style evals (MT-Bench, Arena Elo, IFEval) require an *instruction-tuned* model and measure a different thing — helpfulness and instruction-following — which pretraining alone does not provide. Don't judge a raw base model on a chat leaderboard." },
      ]
    },

    {
      id: "wont-pretrain",
      title: "Why you (probably) won't pretrain from scratch",
      level: "core",
      body: [
        { type: "p", text: "Here is the honest advice: **almost nobody should pretrain a frontier model from scratch.** It costs millions, requires a cluster you don't have, and — critically — an excellent open base model (Llama, Qwen, Mistral, Gemma, DeepSeek) already exists that you can fine-tune for a rounding error of the cost. The knowledge in this section is for *understanding* what those models are and how to reason about them, not usually for reproducing them." },
        { type: "p", text: "The decision tree in practice:" },
        { type: "table",
          headers: ["Your situation", "What to do", "Cost"],
          rows: [
            ["Need a task-specific model", "**fine-tune** an open base model (LoRA/QLoRA)", "$–$$"],
            ["Have a large private domain corpus (legal, bio, code)", "**continued pretraining** on top of an open base", "$$–$$$"],
            ["Need a new language / modality poorly covered by open models", "continued pretraining, maybe from scratch if truly novel", "$$$"],
            ["You are a frontier lab", "pretrain from scratch", "$$$$$"],
          ]
        },
        { type: "heading", text: "When continued pretraining makes sense" },
        { type: "p", text: "**Continued (or continual) pretraining** takes an existing base model and keeps running the same next-token objective on a new corpus — a domain (medicine, law, a proprietary codebase), a new language, or simply fresher data. It's the right tool when you have *billions* of unlabeled domain tokens and the base model's knowledge is genuinely thin there. Use a **lower learning rate** than the original run and **replay** a slice of general web data to avoid *catastrophic forgetting* (the model overfitting the new domain and losing general ability). If you only have thousands of labeled examples, skip this and go straight to fine-tuning — continued pretraining needs a lot of raw text to pay off." },
        { type: "callout", variant: "tip", text: "**What the 'build a GPT' section does instead.** You *should* pretrain — just at a scale that fits on one GPU. The next section (see the **Build a GPT** / nanoGPT track) trains a ~10M-parameter character-level or BPE transformer on a small corpus (TinyShakespeare, a slice of FineWeb) in minutes to hours. It exercises *every* concept here — the next-token loss, the AdamW + warmup-cosine schedule, the loss curve, perplexity, even a mini scaling-law sweep — faithfully, at 1/1{,}000{,}000 the cost. Understanding scales down perfectly even when compute doesn't." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "You cannot afford a frontier run, but every idea here is reproducible on a single GPU (even a free Colab/T4) or a laptop. Do at least two — the scaling-law reproduction in particular will make the theory *click*." },
        { type: "list", ordered: true, items: [
          "**Train a small LM end-to-end.** Clone nanoGPT (or write ~300 lines yourself), tokenize TinyShakespeare with a BPE tokenizer, and pretrain a 6-layer, ~10M-param model. Implement the shift-by-one next-token loss, AdamW with $\\beta_2=0.95$, gradient clipping, and a warmup+cosine LR. Watch the loss fall and sample text at checkpoints — it goes from noise to Shakespeare-ish in minutes.",
          "**Reproduce a mini scaling-law curve.** Train the same architecture at 5 sizes (e.g. 0.5M, 1M, 2M, 4M, 8M params), each Chinchilla-optimally (~20 tokens/param), and plot final loss vs. compute ($6ND$) on log-log axes. You should see a straight line. Extrapolate it to predict the loss of a 16M model, then train that model and check your prediction.",
          "**Find the compute-optimal split yourself.** Fix a small compute budget $C$. Train several models trading $N$ against $D$ (small-model/more-data vs. big-model/less-data) all at the same $6ND$. Plot loss vs. $N$; the U-shaped curve's bottom is your empirical Chinchilla point. Compare its token/param ratio to 20.",
          "**Build the data pipeline.** Take a raw slice of Common Crawl (or FineWeb's raw variant), and implement the funnel: language ID, C4/Gopher quality heuristics, and MinHash near-dedup. Measure what fraction survives each stage (you'll lose >90%) and train two identical models — one on raw, one on filtered — to *see* filtering improve loss.",
          "**Measure MFU.** Instrument your training loop: compute achieved FLOP/s ($6N \\times$ tokens/sec) and divide by your GPU's peak. Then add gradient checkpointing and mixed precision and watch memory drop and MFU change. Try to push MFU above 40%.",
          "**Compute a contamination check.** Take a public benchmark (e.g. HellaSwag), n-gram-hash its examples, and scan your training corpus for overlaps. Quantify how much leaks in — this is exactly what real decontamination stages do.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The primary sources here are unusually readable — the scaling-law papers in particular are worth reading in full. In recommended order:" },
        { type: "link", url: "https://arxiv.org/abs/2001.08361", text: "Kaplan et al. 2020 — Scaling Laws for Neural Language Models (the original power laws)" },
        { type: "link", url: "https://arxiv.org/abs/2203.15556", text: "Hoffmann et al. 2022 — Training Compute-Optimal LLMs (the Chinchilla paper; derive the 20:1 rule)" },
        { type: "link", url: "https://arxiv.org/abs/2407.21783", text: "Dubey et al. 2024 — The Llama 3 Herd of Models (the most detailed modern pretraining report)" },
        { type: "link", url: "https://arxiv.org/abs/2406.17557", text: "Penedo et al. 2024 — The FineWeb Datasets (state-of-the-art open data curation, with ablations)" },
        { type: "link", url: "https://github.com/karpathy/nanoGPT", text: "Karpathy — nanoGPT (the cleanest from-scratch GPT pretraining code; start here to build)" },
        { type: "link", url: "https://www.youtube.com/watch?v=l8pRSuU81PU", text: "Karpathy — 'Let's reproduce GPT-2 (124M)' (4-hour walkthrough of a real small pretraining run)" },
        { type: "link", url: "https://arxiv.org/abs/1910.02054", text: "Rajbhandari et al. 2020 — ZeRO: Memory Optimizations for Training Trillion-Parameter Models" },
        { type: "link", url: "https://arxiv.org/abs/2101.03961", text: "Fedus et al. 2021 — Switch Transformers (the Mixture-of-Experts scaling reference)" },
      ]
    },
  ],

  packages: [
    { name: "pytorch", why: "the framework every pretraining stack is built on (autograd, FSDP, AMP)" },
    { name: "torch.distributed.fsdp", why: "shard params/grads/optimizer state across GPUs (ZeRO-3 in native PyTorch)" },
    { name: "deepspeed", why: "ZeRO stages, pipeline parallelism, and large-scale training utilities" },
    { name: "megatron-lm", why: "NVIDIA's tensor/pipeline-parallel library used for the largest runs" },
    { name: "flash-attention", why: "IO-aware fused attention kernel — the single biggest MFU win" },
    { name: "datasets", why: "HuggingFace streaming/tokenization of trillion-token corpora (FineWeb etc.)" },
    { name: "nanoGPT", why: "Karpathy's minimal, hackable from-scratch GPT training code" },
    { name: "lm-eval-harness", why: "standardized base-model benchmarking (MMLU, HellaSwag, ...)" },
  ],

  gotchas: [
    "The compute law is $C \\approx 6ND$: $2N$ forward + $4N$ backward FLOPs per token. Use it for every budget estimate.",
    "Chinchilla-optimal is ~20 tokens per parameter, but that optimizes *training* cost. For deployment, overtrain a smaller model far past 20:1 to make inference cheap.",
    "Peak FLOPs are fiction — real runs hit 30–55% MFU. A poorly-tuned run costs 3× more for the identical model.",
    "Optimizer state, not weights, dominates training memory: Adam mixed-precision is ~$16$ bytes/param. Shard it with ZeRO/FSDP.",
    "Warmup is mandatory: skip it and large-batch training diverges in the first few hundred steps as untrained gradients blow up.",
    "Deduplicate the corpus (MinHash) — but not too aggressively; global dedup can *hurt* by deleting re-crawled high-quality pages.",
    "Data contamination inflates benchmarks: test sets leak onto the web and into training data. Decontaminate and verify on private evals.",
    "Perplexity only compares across the *same tokenizer and test set* — a larger vocab changes per-token perplexity for free.",
    "'15T tokens' means BPE subwords ($\\approx 0.75$ words each), not words or characters. Always check the unit before comparing corpora.",
  ],

  flashcards: [
    { q: "What is the pretraining objective, in one equation?", a: "Minimize next-token cross-entropy: $\\mathcal{L} = -\\frac{1}{T}\\sum_t \\log p_\\theta(x_t \\mid x_{<t})$. Self-supervised — the text is its own label (shift by one)." },
    { q: "State the training-compute law and derive its constant.", a: "$C \\approx 6ND$. Each parameter costs $2$ FLOPs/token forward (multiply-add) and $\\approx 4$ backward, so $6N$ FLOPs/token times $D$ tokens." },
    { q: "What did Chinchilla (2022) change?", a: "It showed prior models were undertrained. Compute-optimal is $N_{opt}, D_{opt} \\propto \\sqrt{C}$ — roughly **20 tokens per parameter** — so scale data and model together, not just parameters." },
    { q: "Write the Chinchilla loss decomposition.", a: "$L(N,D) = E + A/N^{\\alpha} + B/D^{\\beta}$: irreducible entropy $E$, plus finite-model and finite-data terms. Minimizing under $D = C/6N$ gives the $\\sqrt{C}$ optimum." },
    { q: "Why is compute-optimal not the same as deployment-optimal?", a: "Chinchilla minimizes *training* cost. A deployed model runs billions of inferences, so labs overtrain a *smaller* model far past 20:1 tokens/param to cut serving cost forever." },
    { q: "What does ZeRO / FSDP do?", a: "Shards optimizer state (stage 1), gradients (2), and parameters (3) across GPUs instead of replicating them — memory per GPU falls ~linearly with GPU count." },
    { q: "Why the warmup + cosine learning-rate schedule?", a: "Warmup avoids early divergence while Adam's second moment calibrates; cosine decay to ~10% of peak improves final loss. Skipping warmup usually causes NaNs." },
    { q: "Name the four modern architecture upgrades over the 2017 transformer.", a: "RoPE (relative positions), RMSNorm (cheap pre-norm), SwiGLU (gated FFN), GQA (small KV-cache). Frontier models add MoE for capacity at fixed FLOPs." },
    { q: "What is Mixture-of-Experts and its trade-off?", a: "Replace the FFN with $E$ experts + a router selecting top-$k$ per token. More total params at fixed FLOPs/token, but all experts must fit in memory and routing needs a load-balancing loss." },
    { q: "What is perplexity?", a: "$\\exp$ of the cross-entropy loss — the model's effective branching factor per token. Lower is better; comparable only within the same tokenizer and test set." },
    { q: "What is data contamination and why does it matter?", a: "Test-set examples leaking into training data (they're on the web). It inflates benchmark scores meaninglessly; decontamination via n-gram overlap is a required pipeline stage." },
    { q: "When does continued pretraining make sense over fine-tuning?", a: "When you have *billions* of unlabeled domain tokens and the base model is thin there. Use a low LR and replay general data to avoid catastrophic forgetting." },
  ],

  cheatsheet: [
    { label: "Compute law", code: "C = 6 * N * D   # FLOPs" },
    { label: "Chinchilla ratio", code: "D_opt ~= 20 * N   # tokens per parameter" },
    { label: "Next-token loss", code: "F.cross_entropy(logits[:, :-1], tokens[:, 1:])" },
    { label: "AdamW (LM)", code: "AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)" },
    { label: "Grad clip", code: "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)" },
    { label: "Mixed precision", code: "torch.autocast('cuda', dtype=torch.bfloat16)" },
    { label: "FSDP (ZeRO-3)", code: "model = FSDP(model, mixed_precision=bf16)" },
    { label: "Grad checkpoint", code: "checkpoint_wrapper(block)  # recompute in backward" },
    { label: "Perplexity", code: "ppl = math.exp(loss.item())" },
    { label: "Grad accumulation", code: "for mb in micro: (loss/steps).backward()" },
    { label: "Adam memory", code: "~16 bytes/param  (w, g, fp32 w, m, v)" },
    { label: "MFU", code: "achieved_flops / (n_gpus * peak_flops)" },
  ],
});
