(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "llm-build-gpt",
  name: "Build a GPT From Scratch",
  language: "LLMs",
  group: "Large Language Models",
  navLabel: "Build a GPT (from scratch)",
  tagline: "Assemble everything — tokens, embeddings, attention, blocks, training, sampling — into a working, trainable GPT in ~200 lines of PyTorch.",
  color: "#F43F5E",
  readMinutes: 62,
  sections: [
    {
      id: "what",
      title: "What we're building (and why it's the same thing as GPT-4)",
      level: "core",
      body: [
        { type: "p", text: "This is the capstone. You have learned the pieces — tokenization (NLP Foundations), embeddings (Representations), self-attention and the transformer block (Transformers), the training loop and optimizers (Training Deep Networks), and sampling (How LLMs Work). Here we bolt them together into a **decoder-only transformer language model** that trains on raw text and generates new text, one token at a time." },
        { type: "p", text: "We build a **character-level GPT** trained on a small corpus (the classic choice is the concatenated works of Shakespeare, ~1 MB). It is small enough to train on a laptop in minutes, yet it is *architecturally identical* to GPT-3 and Llama: the only differences are the tokenizer (characters vs BPE subwords), the scale (thousands vs billions of parameters), and the data (1 MB vs 15 trillion tokens). The code you write here is the code that runs the world's frontier models, minus the zeros." },
        { type: "callout", variant: "note", text: "**Lineage.** This section follows Andrej Karpathy's *nanoGPT* / \"Let's build GPT\" — the canonical from-scratch build. We reproduce and explain it end to end. After this you should be able to read the real nanoGPT repo and the GPT-2 code and recognize every line. The full runnable references are linked at the bottom." },
        { type: "callout", variant: "tip", text: "Type every line yourself; do not copy-paste. The goal is that by the end you could rewrite this from a blank file. That is the difference between having *read* about transformers and being able to *build* one." },
      ]
    },
    {
      id: "data",
      title: "Step 1 — Data and a character tokenizer",
      level: "core",
      body: [
        { type: "p", text: "A language model needs (a) a way to turn text into integer token IDs and back, and (b) batches of contiguous token sequences to train on. For a character-level model the tokenizer is trivial — the vocabulary is just the set of distinct characters." },
        { type: "code", lang: "py", code: "import torch\n\nwith open('input.txt', 'r', encoding='utf-8') as f:\n    text = f.read()               # e.g. the tiny-shakespeare corpus\n\nchars = sorted(set(text))\nvocab_size = len(chars)           # ~65 for shakespeare\nstoi = {c: i for i, c in enumerate(chars)}   # char -> id\nitos = {i: c for i, c in enumerate(chars)}   # id -> char\nencode = lambda s: [stoi[c] for c in s]\ndecode = lambda ids: ''.join(itos[i] for i in ids)\n\ndata = torch.tensor(encode(text), dtype=torch.long)\nn = int(0.9 * len(data))\ntrain_data, val_data = data[:n], data[n:]   # never train on val\nprint(vocab_size, data.shape)" },
        { type: "p", text: "Training operates on fixed-length windows. `block_size` (the **context length**) is the maximum number of previous tokens the model conditions on. We draw a random batch of windows; the target for each position is simply the *next* character — the model learns to predict token $t{+}1$ from tokens $\\le t$." },
        { type: "code", lang: "py", code: "torch.manual_seed(1337)\nblock_size = 256      # context length\nbatch_size = 64\n\ndef get_batch(split):\n    d = train_data if split == 'train' else val_data\n    ix = torch.randint(len(d) - block_size, (batch_size,))\n    x = torch.stack([d[i:i+block_size]     for i in ix])   # (B, T)\n    y = torch.stack([d[i+1:i+1+block_size] for i in ix])   # (B, T) shifted by 1\n    return x, y\n\nxb, yb = get_batch('train')\nprint(xb.shape, yb.shape)   # torch.Size([64, 256]) torch.Size([64, 256])" },
        { type: "callout", variant: "note", text: "One window of length $T$ actually contains $T$ training examples: predicting token 1 from token 0, token 2 from tokens 0–1, …, token $T$ from tokens 0..$T{-}1$. The causal mask (below) is what lets all $T$ predictions be computed and trained **in parallel** from a single sequence." },
      ]
    },
    {
      id: "bigram",
      title: "Step 2 — A bigram baseline to establish the skeleton",
      level: "core",
      body: [
        { type: "p", text: "Before attention, build the simplest possible model so the training/eval/generation scaffolding is in place. A **bigram model** predicts the next token from only the current one via a lookup table of logits — an `(vocab, vocab)` embedding. It is bad, but it proves the loop works." },
        { type: "code", lang: "py", code: "import torch.nn as nn\nfrom torch.nn import functional as F\n\nclass BigramLM(nn.Module):\n    def __init__(self, vocab_size):\n        super().__init__()\n        self.token_emb = nn.Embedding(vocab_size, vocab_size)\n\n    def forward(self, idx, targets=None):\n        logits = self.token_emb(idx)          # (B, T, vocab)\n        loss = None\n        if targets is not None:\n            B, T, C = logits.shape\n            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))\n        return logits, loss\n\n    @torch.no_grad()\n    def generate(self, idx, max_new_tokens):\n        for _ in range(max_new_tokens):\n            logits, _ = self(idx)\n            logits = logits[:, -1, :]                 # last step (B, vocab)\n            probs = F.softmax(logits, dim=-1)\n            nxt = torch.multinomial(probs, num_samples=1)\n            idx = torch.cat([idx, nxt], dim=1)        # append and continue\n        return idx" },
        { type: "p", text: "The loss at initialization should be about $\\ln(\\text{vocab\\_size})$ — the cross-entropy of a uniform guess (for vocab 65, that is $\\approx 4.17$). Seeing that number confirms the model starts maximally uncertain, exactly as theory predicts (ref the cross-entropy derivation in Probability)." },
      ]
    },
    {
      id: "attention",
      title: "Step 3 — One head of self-attention",
      level: "core",
      body: [
        { type: "p", text: "Now the heart. Each head projects tokens to queries, keys, and values, scores all query–key pairs, masks the future, softmaxes, and averages the values (the full derivation is in the Transformers section — here we implement it). The causal mask is a lower-triangular matrix registered as a buffer." },
        { type: "code", lang: "py", code: "class Head(nn.Module):\n    def __init__(self, n_embd, head_size, block_size, dropout=0.2):\n        super().__init__()\n        self.key   = nn.Linear(n_embd, head_size, bias=False)\n        self.query = nn.Linear(n_embd, head_size, bias=False)\n        self.value = nn.Linear(n_embd, head_size, bias=False)\n        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))\n        self.dropout = nn.Dropout(dropout)\n\n    def forward(self, x):\n        B, T, C = x.shape\n        k, q = self.key(x), self.query(x)              # (B, T, hs)\n        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # (B,T,T) scaled scores\n        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # causal mask\n        wei = F.softmax(wei, dim=-1)                    # attention weights\n        wei = self.dropout(wei)\n        v = self.value(x)                               # (B, T, hs)\n        return wei @ v                                  # (B, T, hs)" },
        { type: "callout", variant: "gotcha", text: "The scaling `* k.shape[-1]**-0.5` (the $1/\\sqrt{d_k}$) is essential — without it the softmax saturates and gradients vanish (ref Transformers). The `masked_fill(...-inf)` before softmax is what makes this a *decoder* (causal) head; drop it and you have BERT-style bidirectional attention that cheats by seeing the future." },
      ]
    },
    {
      id: "block",
      title: "Step 4 — Multi-head attention, feed-forward, and the block",
      level: "core",
      body: [
        { type: "p", text: "Run several heads in parallel and concatenate; follow with a position-wise feed-forward network (the $4\\times$ expansion from the Transformers section). Then wrap both sub-layers with **residual connections** and **pre-LayerNorm** — the arrangement that lets us stack many blocks without vanishing gradients." },
        { type: "code", lang: "py", code: "class MultiHead(nn.Module):\n    def __init__(self, n_heads, n_embd, head_size, block_size, dropout=0.2):\n        super().__init__()\n        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout)\n                                    for _ in range(n_heads)])\n        self.proj = nn.Linear(n_heads * head_size, n_embd)\n        self.dropout = nn.Dropout(dropout)\n\n    def forward(self, x):\n        out = torch.cat([h(x) for h in self.heads], dim=-1)\n        return self.dropout(self.proj(out))\n\nclass FeedForward(nn.Module):\n    def __init__(self, n_embd, dropout=0.2):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(n_embd, 4 * n_embd),   # expand\n            nn.GELU(),\n            nn.Linear(4 * n_embd, n_embd),   # project back\n            nn.Dropout(dropout),\n        )\n    def forward(self, x): return self.net(x)\n\nclass Block(nn.Module):\n    def __init__(self, n_embd, n_heads, block_size, dropout=0.2):\n        super().__init__()\n        self.sa  = MultiHead(n_heads, n_embd, n_embd // n_heads, block_size, dropout)\n        self.ff  = FeedForward(n_embd, dropout)\n        self.ln1 = nn.LayerNorm(n_embd)\n        self.ln2 = nn.LayerNorm(n_embd)\n\n    def forward(self, x):\n        x = x + self.sa(self.ln1(x))   # pre-norm + residual (attention)\n        x = x + self.ff(self.ln2(x))   # pre-norm + residual (feed-forward)\n        return x" },
        { type: "callout", variant: "tip", text: "`x = x + sublayer(ln(x))` is the single most important line for trainability. The `x +` is the residual highway that gives gradients a direct path back through every block (ref the backprop vanishing-gradient section); the `ln(...)` stabilizes the scale of activations entering each sub-layer." },
      ]
    },
    {
      id: "full-model",
      title: "Step 5 — The full GPT",
      level: "core",
      body: [
        { type: "p", text: "Assemble it: a **token embedding** (what each token means) plus a **positional embedding** (where it sits — attention has no built-in order), a stack of blocks, a final LayerNorm, and an unembedding to vocab-sized logits. This is a complete GPT." },
        { type: "code", lang: "py", code: "class GPT(nn.Module):\n    def __init__(self, vocab_size, n_embd=384, n_heads=6, n_layer=6,\n                 block_size=256, dropout=0.2):\n        super().__init__()\n        self.block_size = block_size\n        self.token_emb = nn.Embedding(vocab_size, n_embd)\n        self.pos_emb   = nn.Embedding(block_size, n_embd)\n        self.blocks = nn.Sequential(*[\n            Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layer)])\n        self.ln_f = nn.LayerNorm(n_embd)\n        self.head = nn.Linear(n_embd, vocab_size)\n\n    def forward(self, idx, targets=None):\n        B, T = idx.shape\n        tok = self.token_emb(idx)                                   # (B,T,C)\n        pos = self.pos_emb(torch.arange(T, device=idx.device))      # (T,C)\n        x = tok + pos                                               # broadcast add\n        x = self.blocks(x)\n        x = self.ln_f(x)\n        logits = self.head(x)                                       # (B,T,vocab)\n        loss = None\n        if targets is not None:\n            B, T, C = logits.shape\n            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))\n        return logits, loss\n\n    @torch.no_grad()\n    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):\n        for _ in range(max_new_tokens):\n            idx_cond = idx[:, -self.block_size:]        # crop to context window\n            logits, _ = self(idx_cond)\n            logits = logits[:, -1, :] / temperature     # focus on last step\n            if top_k is not None:\n                v, _ = torch.topk(logits, top_k)\n                logits[logits < v[:, [-1]]] = -float('inf')\n            probs = F.softmax(logits, dim=-1)\n            nxt = torch.multinomial(probs, num_samples=1)\n            idx = torch.cat([idx, nxt], dim=1)\n        return idx" },
        { type: "p", text: "With $n\\_embd=384$, $6$ heads, $6$ layers and block size $256$, this is roughly a **10-million-parameter** model — about 1/12500th of GPT-3's 175 B, running the identical computation. Count them with `sum(p.numel() for p in model.parameters())`." },
      ]
    },
    {
      id: "training",
      title: "Step 6 — Train it",
      level: "core",
      body: [
        { type: "p", text: "The training loop is the canonical one from the Training Deep Networks section: sample a batch, forward to get the loss, zero grads, backward, step. We use **AdamW** and periodically estimate loss on both splits to watch for overfitting." },
        { type: "code", lang: "py", code: "device = 'cuda' if torch.cuda.is_available() else 'cpu'\nmodel = GPT(vocab_size, block_size=block_size).to(device)\noptimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)\n\n@torch.no_grad()\ndef estimate_loss(iters=200):\n    out = {}\n    model.eval()\n    for split in ('train', 'val'):\n        losses = torch.zeros(iters)\n        for k in range(iters):\n            x, y = get_batch(split)\n            _, loss = model(x.to(device), y.to(device))\n            losses[k] = loss.item()\n        out[split] = losses.mean().item()\n    model.train()\n    return out\n\nfor step in range(5000):\n    if step % 500 == 0:\n        l = estimate_loss()\n        print(f\"step {step:4d} | train {l['train']:.3f} | val {l['val']:.3f}\")\n    xb, yb = get_batch('train')\n    _, loss = model(xb.to(device), yb.to(device))\n    optimizer.zero_grad(set_to_none=True)\n    loss.backward()\n    optimizer.step()" },
        { type: "p", text: "On a GPU this reaches a validation loss around **1.5** in a few minutes — from the initial $\\approx 4.17$. That drop is the model going from uniform guessing to genuinely modeling English character statistics." },
        { type: "callout", variant: "gotcha", text: "Watch the gap between train and val loss. If val stops improving while train keeps dropping, you are overfitting — raise dropout, shrink the model, or get more data (ref the bias–variance dashboard in Training Deep Networks). On the tiny-shakespeare set this small model is roughly balanced." },
      ]
    },
    {
      id: "generate",
      title: "Step 7 — Generate",
      level: "core",
      body: [
        { type: "p", text: "Sampling starts from a single token (or a prompt) and autoregressively extends: predict the next-token distribution, sample from it (with temperature / top-k, ref the sampling section), append, repeat. Each new token attends back over the growing context." },
        { type: "code", lang: "py", code: "start = torch.zeros((1, 1), dtype=torch.long, device=device)   # the newline token\nout = model.generate(start, max_new_tokens=500, temperature=0.8, top_k=200)\nprint(decode(out[0].tolist()))\n\n# After training on Shakespeare you get lines like fake-Shakespearean verse:\n#   'MENENIUS: Why, then the world hath not a heart to give...'\n# It's not memorized — it's newly generated from the learned distribution." },
        { type: "callout", variant: "tip", text: "Temperature < 1 sharpens the distribution (safer, more repetitive); > 1 flattens it (more surprising, more errors). top-k truncates to the k most likely tokens before sampling, cutting off the long tail of nonsense. These same knobs control ChatGPT's outputs — they are exactly the ones in the How-LLMs-Work section." },
      ]
    },
    {
      id: "scaling-up",
      title: "Step 8 — From this to a real LLM",
      level: "deep",
      body: [
        { type: "p", text: "You now have the *complete* architecture of a frontier model. The path from this 10 M-parameter toy to GPT-4 is not a different design — it is **more of the same**, plus engineering. What actually changes:" },
        { type: "table",
          headers: ["Dimension", "This toy", "A frontier LLM", "Same idea?"],
          rows: [
            ["Tokenizer", "characters (~65)", "BPE subwords (~100k–200k)", "yes — ref NLP Foundations"],
            ["Parameters", "~10 million", "billions to trillions", "identical blocks, just more"],
            ["Data", "1 MB Shakespeare", "10–15 trillion tokens of web text", "same next-token objective"],
            ["Context", "256 tokens", "128k–1M+ tokens", "same attention, + RoPE, FlashAttention"],
            ["Norm / act", "LayerNorm / GELU", "RMSNorm / SwiGLU", "minor refinements"],
            ["Attention", "vanilla multi-head", "grouped-query + FlashAttention", "same formula, cheaper"],
            ["Training", "AdamW, 1 GPU, minutes", "AdamW, thousands of GPUs, months", "same optimizer, distributed"],
            ["After pretraining", "— (base only)", "SFT + RLHF/DPO alignment", "ref Fine-Tuning & Alignment"],
          ]
        },
        { type: "p", text: "The next sections make each of these real: **Pretraining & Scaling Laws** (how much data/compute for a given size), **Using LLMs** (run the big open ones), **Fine-Tuning & Alignment** (turn a base model into an assistant), **RAG** (give it knowledge), and **Agents** (let it act). But the engine — the thing doing the thinking — is the one you just built." },
        { type: "callout", variant: "good", text: "**You have built a GPT.** Everything else in the LLM track is scaling, data, alignment, and application around this exact core. When someone says a model is \"a 70B decoder-only transformer,\" you now know precisely what every word means and could, given the compute, build it." },
      ]
    },
    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Do project 1 fully — a trained model you generate from — before anything else. The rest deepen and extend it." },
        { type: "list", ordered: true, items: [
          "**Train your own GPT.** Get the tiny-shakespeare corpus (or your own text — your chat logs, code, a favorite author), train the model above, and generate. Tune `n_layer`, `n_head`, `n_embd`, `dropout` and watch train/val loss respond. This is the whole point of the section.",
          "**Swap in a BPE tokenizer.** Replace the character tokenizer with `tiktoken` (GPT-2's BPE). Notice the vocab jumps to ~50k, sequences get shorter, and generation quality improves for the same context length. You have just closed most of the gap to a real GPT.",
          "**Scale it and reproduce a loss curve.** Train three sizes (small/medium/large `n_embd`) on the same data and plot final val loss vs parameter count — a hand-made mini scaling law (ref Pretraining & Scaling Laws).",
          "**Read and diff nanoGPT.** Clone Karpathy's nanoGPT and line-by-line compare it to your code. Find what it adds (weight tying, better init, learning-rate schedule, gradient clipping, mixed precision) and fold those in.",
          "**Fine-tune your model.** Take your trained base model and continue-train it on a narrower corpus (one author, one genre) and watch its style shift — a from-scratch preview of the Fine-Tuning section.",
          "**Instrument attention.** Extract and visualize the attention weights of your trained model for a sample string. See which characters attend to which — your own interpretability experiment.",
        ]},
      ]
    },
    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "These are the canonical runnable companions — the code you just wrote, in fuller form:" },
        { type: "link", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", text: "Andrej Karpathy — \"Let's build GPT: from scratch, in code, spelled out\" (this section IS this video; watch and code along)" },
        { type: "link", url: "https://github.com/karpathy/nanoGPT", text: "nanoGPT — the minimal, production-lean GPT training repo (train GPT-2 scale)" },
        { type: "link", url: "https://github.com/karpathy/minGPT", text: "minGPT — the readable, educational predecessor" },
        { type: "link", url: "https://github.com/karpathy/build-nanogpt", text: "build-nanoGPT — reproducing GPT-2 (124M) from scratch, step by step" },
        { type: "link", url: "https://github.com/rasbt/LLMs-from-scratch", text: "Sebastian Raschka — Build a Large Language Model (From Scratch), full book code" },
        { type: "link", url: "https://jalammar.github.io/illustrated-gpt2/", text: "The Illustrated GPT-2 — a visual walkthrough of exactly this architecture" },
      ]
    },
  ],

  packages: [
    { name: "torch", why: "the entire model — nn.Module, nn.Embedding, nn.Linear, AdamW, autograd" },
    { name: "torch.nn.functional", why: "cross_entropy, softmax — the loss and sampling core" },
    { name: "tiktoken", why: "GPT-2/GPT-4 BPE tokenizer — the upgrade from char-level" },
    { name: "nanoGPT", why: "Karpathy's reference implementation to train GPT-2 scale" },
    { name: "datasets", why: "HuggingFace — load real text corpora at scale for bigger runs" },
    { name: "wandb", why: "track loss curves across runs when you start scaling (ref MLOps)" },
  ],

  gotchas: [
    "Initial loss should be $\\approx \\ln(\\text{vocab\\_size})$ (~4.17 for vocab 65). If it's wildly different, your logits or cross-entropy wiring is wrong.",
    "In `generate`, crop the context to the last `block_size` tokens (`idx[:, -block_size:]`) — the positional embedding table only has `block_size` rows and will index-error otherwise.",
    "The causal mask must be applied **before** softmax with `-inf`, not after. Forgetting it lets the model see the future and 'train' to a suspiciously low loss that won't generate.",
    "`cross_entropy` expects logits of shape `(N, C)` and targets `(N,)` — reshape `(B,T,C)`→`(B*T,C)` and `(B,T)`→`(B*T,)`. A shape mismatch here is the most common bug.",
    "Token + position embeddings are **added**, not concatenated: `tok_emb + pos_emb`, relying on broadcasting `(B,T,C) + (T,C)`.",
    "Pre-norm (`x + sa(ln(x))`) trains far more stably than post-norm at depth — use it, as GPT-2 onward does.",
    "Move both model and every batch to the same `device`; a CPU/GPU mismatch throws at the first matmul.",
  ],

  flashcards: [
    { q: "What makes a transformer a 'GPT' (decoder-only)?", a: "Causal (masked) self-attention — each position attends only to itself and the past — trained on next-token prediction. That mask is the whole difference from BERT." },
    { q: "Why add positional embeddings?", a: "Self-attention is permutation-invariant; without position info the model can't tell token order. GPT adds a learned positional embedding to each token embedding." },
    { q: "What is `block_size` / context length?", a: "The maximum number of previous tokens the model conditions on. It caps the positional embedding table and must be respected when generating (crop the context)." },
    { q: "Why is initial loss ≈ ln(vocab_size)?", a: "An untrained model outputs a near-uniform distribution; cross-entropy of a uniform guess over V classes is ln(V) (ref Probability). It's a correctness check." },
    { q: "How does generation work?", a: "Autoregressive sampling: forward pass → take last-position logits → apply temperature/top-k → softmax → sample one token → append → repeat, cropping context to block_size." },
    { q: "Why residual + LayerNorm around each sub-layer?", a: "The residual gives gradients a direct path back through all blocks (prevents vanishing gradients at depth); LayerNorm stabilizes activation scale. Modern GPTs use pre-norm." },
    { q: "How does this toy differ from GPT-4 architecturally?", a: "It barely does: BPE vs char tokenizer, billions vs millions of params, trillions vs 1MB tokens, RoPE/RMSNorm/GQA/FlashAttention refinements, and post-training alignment. Same core blocks." },
    { q: "What does the FeedForward sub-layer do?", a: "A per-token MLP (expand to 4×, non-linearity, project back). Attention mixes info across tokens; the FFN processes each token and stores much of the model's knowledge." },
  ],

  cheatsheet: [
    { label: "Char encode/decode", code: "encode=lambda s:[stoi[c] for c in s]" },
    { label: "Batch of windows", code: "x=d[i:i+T]; y=d[i+1:i+1+T]" },
    { label: "Scaled scores", code: "wei = q@k.transpose(-2,-1)*k.shape[-1]**-0.5" },
    { label: "Causal mask", code: "wei.masked_fill(tril==0, float('-inf'))" },
    { label: "Attention out", code: "F.softmax(wei,-1) @ v" },
    { label: "Block (pre-norm)", code: "x=x+sa(ln1(x)); x=x+ff(ln2(x))" },
    { label: "Token+pos emb", code: "x = tok_emb(idx) + pos_emb(arange(T))" },
    { label: "LM loss", code: "F.cross_entropy(logits.view(B*T,C), y.view(B*T))" },
    { label: "Optimizer", code: "torch.optim.AdamW(model.parameters(), lr=3e-4)" },
    { label: "Train step", code: "opt.zero_grad(); loss.backward(); opt.step()" },
    { label: "Crop context", code: "idx_cond = idx[:, -block_size:]" },
    { label: "Param count", code: "sum(p.numel() for p in model.parameters())" },
  ],
});
