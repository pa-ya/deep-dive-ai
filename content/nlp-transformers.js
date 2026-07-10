(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "nlp-transformers",
  name: "Transformers",
  language: "NLP",
  group: "Natural Language Processing",
  navLabel: "Transformers (from scratch)",
  tagline: "The architecture behind every modern LLM — self-attention derived from scratch, built in NumPy and PyTorch, and explained until it is obvious.",
  color: "#0891B2",
  readMinutes: 58,
  sections: [
    {
      id: "why",
      title: "Why transformers replaced everything",
      level: "core",
      body: [
        { type: "p", text: "In 2017 a team at Google published *\"Attention Is All You Need\"* and quietly ended the RNN era. Within five years the transformer became the substrate of GPT, BERT, Llama, Stable Diffusion, AlphaFold, and essentially every frontier model. If you understand this one architecture deeply, you understand the engine of modern AI. This section builds it from nothing." },
        { type: "p", text: "Recall the two fatal problems of recurrent networks (from the RNN section):" },
        { type: "list", ordered: false, items: [
          "**No parallelism.** An RNN processes token $t$ only after token $t-1$ — training is inherently sequential, so it cannot exploit a GPU's thousands of cores across the time axis.",
          "**Long-range forgetting.** Information from token 1 must survive being repeatedly transformed to influence token 1000. Even LSTMs strain over long distances.",
        ]},
        { type: "p", text: "The transformer's answer is radical: **throw away recurrence entirely.** Let every token look *directly* at every other token in a single parallel operation called **self-attention**. There is no distance — token 1 and token 1000 are one hop apart. And because every position is computed independently, the whole sequence processes at once." },
        { type: "callout", variant: "note", text: "**The bet that paid off.** Removing recurrence meant the model had *no* built-in notion of order or locality — priors that CNNs and RNNs bake in. The transformer makes up for this weak inductive bias with **scale**: more data and parameters. That trade — less structure, more scale — turned out to be exactly what unlocked the LLM era. This section is the technical prerequisite for the entire LLM track." },
      ]
    },
    {
      id: "intuition",
      title: "The core idea: attention as soft, content-based lookup",
      level: "core",
      body: [
        { type: "p", text: "Start with intuition before math. Consider the sentence *\"The animal didn't cross the street because it was too tired.\"* What does *\"it\"* refer to? To resolve *\"it\"*, the model should **attend** to *\"animal\"*. Attention is the mechanism that lets each word gather information from the other words that are relevant to it." },
        { type: "p", text: "Think of it as a **soft dictionary lookup**. A normal dictionary matches a query key exactly and returns one value. Attention instead compares a query against *all* keys, produces a similarity weight for each, and returns a **weighted average** of all values. Every token builds a query ('what am I looking for?'), and every token offers a key ('what do I contain?') and a value ('what will I contribute if attended to?')." },
        { type: "math", tex: String.raw`\text{output}_i = \sum_{j} \underbrace{\alpha_{ij}}_{\text{how much } i \text{ attends to } j} \; v_j, \qquad \sum_j \alpha_{ij} = 1` },
        { type: "p", text: "The weights $\\alpha_{ij}$ are computed from the compatibility between token $i$'s query and token $j$'s key. High compatibility → high weight → token $j$'s value dominates token $i$'s output. That is the entire concept; the rest is making it precise and fast." },
      ]
    },
    {
      id: "qkv",
      title: "Scaled dot-product attention, derived",
      level: "core",
      body: [
        { type: "p", text: "Let the input be a sequence of $n$ token embeddings stacked as rows of $X \\in \\mathbb{R}^{n\\times d}$. We produce three projections of each token using learned weight matrices — the **Query**, **Key**, and **Value**:" },
        { type: "math", tex: String.raw`Q = X W^Q, \qquad K = X W^K, \qquad V = X W^V` },
        { type: "p", text: "with $W^Q, W^K \\in \\mathbb{R}^{d\\times d_k}$ and $W^V \\in \\mathbb{R}^{d\\times d_v}$. Now measure how much every token should attend to every other token. The compatibility of query $i$ with key $j$ is their **dot product** $q_i^\\top k_j$ — large when the two vectors are aligned (recall the dot product as alignment from Linear Algebra). All pairs at once is a single matrix product:" },
        { type: "math", tex: String.raw`S = Q K^\top \in \mathbb{R}^{n\times n}, \qquad S_{ij} = q_i^\top k_j` },
        { type: "heading", text: "Why divide by √dₖ" },
        { type: "p", text: "Here is the one subtle step. If the entries of $q_i$ and $k_j$ are independent with mean 0 and variance 1, then their dot product $q_i^\\top k_j = \\sum_{l=1}^{d_k} q_{il}k_{jl}$ is a sum of $d_k$ such terms, so it has variance $d_k$ and standard deviation $\\sqrt{d_k}$. For large $d_k$ these scores become huge, pushing softmax into a saturated region where one weight is ≈1 and the rest ≈0 — and there the gradient nearly vanishes. Dividing by $\\sqrt{d_k}$ renormalizes the variance back to 1:" },
        { type: "math", tex: String.raw`\operatorname{Var}\!\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1` },
        { type: "p", text: "Now normalize each row into a probability distribution with softmax, and use those weights to average the values. That is the celebrated formula in full:" },
        { type: "math", tex: String.raw`\boxed{\;\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\;}` },
        { type: "p", text: "Read it right-to-left one more time: score all query–key pairs ($QK^\\top$), scale, softmax each row into attention weights, then multiply by $V$ to get, for each token, the weighted blend of every token's value. Output shape is $n\\times d_v$ — one context-aware vector per input token." },
        { type: "callout", variant: "tip", text: "Softmax over a row is $\\operatorname{softmax}(z)_j = e^{z_j}/\\sum_l e^{z_l}$. For numerical stability always subtract the row max before exponentiating: $\\operatorname{softmax}(z) = \\operatorname{softmax}(z - \\max z)$. Every real implementation does this; skipping it overflows on long sequences." },
      ]
    },
    {
      id: "self-vs-cross",
      title: "Self-attention, cross-attention, and the causal mask",
      level: "core",
      body: [
        { type: "p", text: "**Self-attention** is when $Q$, $K$, $V$ all come from the *same* sequence — tokens attend to each other. **Cross-attention** is when queries come from one sequence and keys/values from another (a decoder attending to an encoder's output — the basis of translation)." },
        { type: "heading", text: "Causal masking — the trick that makes GPT possible" },
        { type: "p", text: "A language model predicts the next token, so when computing the representation of position $i$ it must **not** be allowed to peek at positions $> i$ (the future). We enforce this by adding a mask to the scores *before* softmax: set the disallowed entries to $-\\infty$ so their softmax weight becomes exactly 0." },
        { type: "math", tex: String.raw`S^{\text{masked}}_{ij} = \begin{cases} S_{ij} & j \le i \\ -\infty & j > i \end{cases} \qquad\Longrightarrow\qquad \operatorname{softmax}(-\infty) = 0` },
        { type: "callout", variant: "note", text: "This single mask is the difference between **BERT** (bidirectional — every token sees all others, great for understanding) and **GPT** (causal — each token sees only the past, which is what lets it *generate* left-to-right). Same attention math, one triangular mask apart." },
      ]
    },
    {
      id: "multihead",
      title: "Multi-head attention",
      level: "core",
      body: [
        { type: "p", text: "One attention operation forces every token to blend everything through a single set of weights — one 'relationship type.' But language has many simultaneous relationships: syntactic agreement, coreference, positional adjacency. **Multi-head attention** runs $h$ attention operations in parallel, each with its own projections, letting each **head** specialize." },
        { type: "math", tex: String.raw`\operatorname{head}_l = \operatorname{Attention}(X W^Q_l,\, X W^K_l,\, X W^V_l), \quad l = 1,\dots,h` },
        { type: "math", tex: String.raw`\operatorname{MultiHead}(X) = \operatorname{Concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)\,W^O` },
        { type: "p", text: "Each head projects to a smaller dimension $d_k = d/h$, so the total compute is roughly the same as one full-width head. The heads' outputs are concatenated back to width $d$ and mixed by a final projection $W^O \\in \\mathbb{R}^{d\\times d}$. Interpretability studies find heads that track exactly the linguistic relations above — some attend to the previous token, some to the syntactic head of a phrase, some to coreferent nouns." },
        { type: "callout", variant: "tip", text: "In modern LLMs the number of heads is a key config (e.g. GPT-2 small: $d=768$, $h=12$, so $d_k=64$). Variants like **multi-query** and **grouped-query attention** share keys/values across heads to shrink the memory of the KV-cache during generation — you will meet these in the LLM inference section." },
      ]
    },
    {
      id: "positional",
      title: "Positional encoding — putting order back in",
      level: "core",
      body: [
        { type: "p", text: "Attention has a curious property: it is **permutation-equivariant**. Shuffle the input tokens and the outputs shuffle identically — the mechanism has *no inherent notion of order*. But *\"dog bites man\"* and *\"man bites dog\"* must differ. So we explicitly inject position information by adding a **positional encoding** to each token embedding." },
        { type: "p", text: "The original paper uses fixed **sinusoidal** encodings — a different frequency per dimension, so each position gets a unique, smoothly varying fingerprint:" },
        { type: "math", tex: String.raw`PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)` },
        { type: "p", text: "The clever part: for any fixed offset $k$, $PE_{pos+k}$ is a **linear function** of $PE_{pos}$ (a rotation), so the model can learn to attend by *relative* position easily. And because it is defined by a formula, it extrapolates to sequences longer than any seen in training." },
        { type: "table",
          headers: ["Scheme", "Idea", "Used by"],
          rows: [
            ["Sinusoidal (fixed)", "hand-designed sin/cos frequencies", "original Transformer"],
            ["Learned absolute", "a trainable vector per position", "BERT, GPT-2"],
            ["Rotary (RoPE)", "rotate Q,K by a position-dependent angle", "Llama, most modern LLMs"],
            ["ALiBi", "add a distance penalty to attention scores", "some long-context models"],
          ]
        },
        { type: "callout", variant: "note", text: "**RoPE** (rotary position embedding) is what most current LLMs use because it encodes *relative* position directly inside the dot product and extends gracefully to long contexts. It is covered in the LLM architecture section — but notice it is a direct descendant of the rotation property of these sinusoids." },
      ]
    },
    {
      id: "block",
      title: "The full transformer block",
      level: "core",
      body: [
        { type: "p", text: "A transformer is a stack of identical **blocks**. Each block has two sub-layers — multi-head self-attention and a position-wise feed-forward network — and each sub-layer is wrapped with a **residual connection** and **layer normalization**. Writing the modern **pre-norm** arrangement (norm before the sub-layer, used by GPT-2 onward because it trains more stably):" },
        { type: "math", tex: String.raw`\begin{aligned}
z &= x + \operatorname{MultiHead}(\operatorname{LN}(x)) \\
y &= z + \operatorname{FFN}(\operatorname{LN}(z))
\end{aligned}` },
        { type: "p", text: "The **feed-forward network** is applied to each position independently — two linear layers with a non-linearity, expanding to a wider hidden dimension (typically $4d$) and back:" },
        { type: "math", tex: String.raw`\operatorname{FFN}(x) = W_2\,\phi(W_1 x + b_1) + b_2, \qquad W_1 \in \mathbb{R}^{4d\times d},\; W_2 \in \mathbb{R}^{d\times 4d}` },
        { type: "p", text: "with $\\phi$ a GELU (modern) or ReLU (original). Intuition for the division of labor: **attention mixes information across tokens**; the **FFN processes each token's gathered information** (and holds much of the model's factual knowledge). Together they are one block; stacking $N$ of them ($N=12$ for GPT-2 small, $96$ for GPT-3) builds depth." },
        { type: "callout", variant: "tip", text: "**Why residuals are non-negotiable here** (ref the backprop vanishing-gradient section): the $x + \\text{sublayer}(x)$ form gives the gradient a direct identity path back through every one of the $N$ blocks. Without it, a 96-layer transformer simply would not train. Residual connections are what make depth possible." },
      ]
    },
    {
      id: "architectures",
      title: "Three architectures from one block",
      level: "core",
      body: [
        { type: "p", text: "The same block yields three model families depending on how you wire and mask it:" },
        { type: "table",
          headers: ["Family", "Attention", "Trained to", "Examples", "Best at"],
          rows: [
            ["**Encoder-only**", "bidirectional", "fill in masked tokens (MLM)", "BERT, RoBERTa", "understanding: classification, retrieval, NER"],
            ["**Decoder-only**", "causal (masked)", "predict the next token", "GPT, Llama, Mistral", "generation: chat, code, everything LLM"],
            ["**Encoder–decoder**", "encoder bi-, decoder causal + cross-attn", "map a sequence to a sequence", "T5, BART, original Transformer", "translation, summarization"],
          ]
        },
        { type: "p", text: "The industry has largely converged on **decoder-only** models for general-purpose LLMs, because next-token prediction on huge corpora turns out to be a stunningly general training objective — it forces the model to learn grammar, facts, reasoning, and translation all at once, with no labels. That realization is the seed of the entire LLM track, where you will build exactly this architecture into a working GPT." },
      ]
    },
    {
      id: "from-scratch",
      title: "Self-attention from scratch",
      level: "core",
      body: [
        { type: "p", text: "First the raw mechanism in NumPy — no framework — so nothing is hidden. This is causal single-head self-attention, the heart of a GPT:" },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef softmax(x, axis=-1):\n    x = x - x.max(axis=axis, keepdims=True)   # numerical stability\n    e = np.exp(x)\n    return e / e.sum(axis=axis, keepdims=True)\n\ndef causal_self_attention(X, Wq, Wk, Wv):\n    n, d = X.shape\n    Q, K, V = X @ Wq, X @ Wk, X @ Wv          # (n, d_k) each\n    dk = Q.shape[1]\n    scores = (Q @ K.T) / np.sqrt(dk)          # (n, n) all pairs\n    mask = np.triu(np.ones((n, n)), k=1).astype(bool)  # upper triangle = future\n    scores[mask] = -np.inf                    # forbid attending ahead\n    A = softmax(scores, axis=-1)              # (n, n) attention weights\n    return A @ V, A                           # (n, d_v) outputs, and weights\n\nnp.random.seed(0)\nn, d, dk = 4, 8, 8\nX = np.random.randn(n, d)\nWq, Wk, Wv = (np.random.randn(d, dk) for _ in range(3))\nout, A = causal_self_attention(X, Wq, Wk, Wv)\nprint(A.round(2))   # lower-triangular: row i only attends to columns <= i\nprint(out.shape)    # (4, 8)" },
        { type: "p", text: "Now the same thing the way you'd actually write it — PyTorch, multi-head, using the fused kernel that ships with modern PyTorch:" },
        { type: "code", lang: "py", code: "import torch, torch.nn as nn, torch.nn.functional as F\n\nclass MultiHeadSelfAttention(nn.Module):\n    def __init__(self, d, n_heads):\n        super().__init__()\n        assert d % n_heads == 0\n        self.h, self.dk = n_heads, d // n_heads\n        self.qkv = nn.Linear(d, 3 * d)     # fused Q,K,V projection\n        self.proj = nn.Linear(d, d)        # output projection W^O\n\n    def forward(self, x):                  # x: (B, T, d)\n        B, T, d = x.shape\n        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)\n        q, k, v = qkv[0], qkv[1], qkv[2]   # each (B, h, T, dk)\n        # fused scaled-dot-product attention with a causal mask:\n        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)\n        y = y.transpose(1, 2).reshape(B, T, d)   # recombine heads\n        return self.proj(y)\n\nx = torch.randn(2, 16, 128)               # batch 2, seq 16, dim 128\nmha = MultiHeadSelfAttention(d=128, n_heads=8)\nprint(mha(x).shape)                        # torch.Size([2, 16, 128])" },
        { type: "callout", variant: "good", text: "`F.scaled_dot_product_attention` is the exact formula from the derivation, but backed by **FlashAttention** — a fused, memory-efficient kernel that never materializes the full $n\\times n$ score matrix. Understanding the naive NumPy version is what lets you read, debug, and modify the fast one." },
      ]
    },
    {
      id: "complexity",
      title: "The quadratic cost of attention",
      level: "deep",
      body: [
        { type: "p", text: "Self-attention compares every token with every token, so the score matrix $S = QK^\\top$ is $n\\times n$. Both compute and memory scale **quadratically** with sequence length:" },
        { type: "math", tex: String.raw`\text{cost} = O(n^2 d)` },
        { type: "p", text: "Doubling the context length quadruples the attention cost. This is *the* central engineering constraint of LLMs — it is why context windows were historically short, why long-context models are a research frontier, and why so much effort goes into cheaper attention." },
        { type: "table",
          headers: ["Approach", "Idea", "Cost"],
          rows: [
            ["FlashAttention", "fuse the kernel; recompute instead of storing $S$", "$O(n^2)$ compute, $O(n)$ memory"],
            ["Sparse / windowed", "each token attends to a local window only", "$O(n\\cdot w)$"],
            ["Linear attention", "approximate softmax with a kernel feature map", "$O(n)$"],
            ["KV-cache (inference)", "reuse past keys/values during generation", "$O(n)$ per new token"],
          ]
        },
        { type: "callout", variant: "note", text: "The **KV-cache** is why chatbots stay fast as a conversation grows: during generation you cache each token's $K$ and $V$ so a new token attends to stored vectors instead of recomputing them. It is also the main consumer of GPU memory at inference — covered in the LLM serving section." },
      ]
    },
    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "You cannot understand LLMs without implementing attention at least once. Do project 1 no matter what; it is the direct prerequisite for the 'Build a GPT' section." },
        { type: "list", ordered: true, items: [
          "**Attention by hand.** Implement single-head causal self-attention in pure NumPy (as above), then extend to multi-head. Feed it a tiny sequence and *print the attention matrix* — verify it is lower-triangular and each row sums to 1. Visualize the weights as a heatmap.",
          "**A full transformer block.** Assemble multi-head attention + residual + LayerNorm + FFN into a `Block` module in PyTorch. Stack a few and confirm shapes flow through unchanged. You now have the body of a GPT.",
          "**Visualize what heads learn.** Take a pretrained small model (e.g. GPT-2 via HuggingFace), extract attention weights for a sentence, and plot per-head heatmaps. Find a head that tracks the previous token and one that does coreference.",
          "**Positional encoding ablation.** Train a tiny transformer on a copy/sort task with and without positional encodings. Watch it fail without them — proving attention's permutation-equivariance to yourself.",
          "**Sinusoid explorer.** Plot the sinusoidal positional-encoding matrix as an image and verify the relative-position (rotation) property numerically. Then swap in learned embeddings and compare.",
        ]},
      ]
    },
    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "This section is self-contained, but these are the canonical companions — read/watch them in this order:" },
        { type: "link", url: "https://jalammar.github.io/illustrated-transformer/", text: "Jay Alammar — The Illustrated Transformer (the best visual explanation; read it right after this)" },
        { type: "link", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", text: "Andrej Karpathy — Let's build GPT: from scratch, in code (build a working transformer line by line — the perfect next step)" },
        { type: "link", url: "https://arxiv.org/abs/1706.03762", text: "Vaswani et al. (2017) — Attention Is All You Need (the original paper; now readable after this section)" },
        { type: "link", url: "https://nlp.seas.harvard.edu/annotated-transformer/", text: "The Annotated Transformer (Harvard) — the paper reimplemented in runnable PyTorch, line by line" },
        { type: "link", url: "https://arxiv.org/abs/2104.09864", text: "Su et al. (2021) — RoFormer / RoPE (rotary position embeddings, used by modern LLMs)" },
        { type: "link", url: "https://arxiv.org/abs/2205.14135", text: "Dao et al. (2022) — FlashAttention (the fused kernel behind fast attention)" },
      ]
    },
  ],

  packages: [
    { name: "torch.nn.functional.scaled_dot_product_attention", why: "the fused, FlashAttention-backed implementation of the core formula" },
    { name: "torch.nn.MultiheadAttention", why: "batteries-included multi-head attention layer" },
    { name: "torch.nn.TransformerEncoderLayer", why: "a full pre-built block (attention + FFN + norm + residual)" },
    { name: "transformers", why: "HuggingFace — pretrained transformers (BERT/GPT/T5) and the tooling around them" },
    { name: "einops", why: "readable tensor reshaping (rearrange) — invaluable for multi-head bookkeeping" },
    { name: "bertviz", why: "visualize attention heads to see what they learn" },
  ],

  gotchas: [
    "Attention is **permutation-equivariant** — without positional encodings the model literally cannot tell word order. Never forget to add them.",
    "Always subtract the row max before softmax; on long sequences the raw scores overflow `exp` otherwise.",
    "The $1/\\sqrt{d_k}$ scaling is not cosmetic — omit it and for large $d_k$ softmax saturates and gradients vanish.",
    "Causal masking must set future scores to $-\\infty$ **before** softmax (so they become exactly 0), not zero them after.",
    "Attention memory/compute is $O(n^2)$ — long contexts are expensive; this is the field's central constraint, not a minor detail.",
    "Pre-norm (LN before sub-layer) vs post-norm changes training stability; modern GPTs use pre-norm. Don't copy the original post-norm blindly.",
    "Multi-head splits the dimension ($d_k = d/h$); the heads don't add compute, they *reallocate* it. Getting the reshape/transpose order wrong is the classic bug.",
  ],

  flashcards: [
    { q: "Write the scaled dot-product attention formula.", a: "$\\operatorname{Attention}(Q,K,V)=\\operatorname{softmax}(QK^\\top/\\sqrt{d_k})\\,V$. Score all query-key pairs, scale, softmax to weights, average the values." },
    { q: "Why divide the scores by $\\sqrt{d_k}$?", a: "Dot products of $d_k$-dim vectors have variance $d_k$; unscaled they grow large and saturate softmax (vanishing gradient). Dividing by $\\sqrt{d_k}$ restores unit variance." },
    { q: "What are Q, K, and V?", a: "Learned linear projections of the input: Query = what a token is looking for, Key = what a token offers for matching, Value = what it contributes if attended to." },
    { q: "What does the causal mask do and why?", a: "Sets scores for future positions ($j>i$) to $-\\infty$ so their softmax weight is 0, preventing a token from seeing the future — required for next-token generation (GPT)." },
    { q: "Why multi-head instead of one big attention?", a: "Different heads specialize in different relationships (syntax, coreference, position) in parallel subspaces, then concatenate — richer than a single averaging." },
    { q: "Why do transformers need positional encodings?", a: "Self-attention is permutation-equivariant (no built-in order). PEs inject position so the model can distinguish 'dog bites man' from 'man bites dog'." },
    { q: "What are the two sub-layers of a transformer block?", a: "Multi-head self-attention (mixes info across tokens) and a position-wise feed-forward network (processes each token), each wrapped with residual + LayerNorm." },
    { q: "Encoder-only vs decoder-only vs encoder-decoder?", a: "Encoder (bidirectional, BERT, understanding); decoder (causal, GPT, generation); encoder-decoder (T5, seq-to-seq like translation)." },
    { q: "What is the computational complexity of attention?", a: "$O(n^2 d)$ in sequence length $n$ — quadratic. This is why long context is expensive and drives FlashAttention, sparse/linear attention, and KV-caching." },
    { q: "What is the KV-cache?", a: "During generation, cache each token's keys and values so each new token attends to stored vectors instead of recomputing — makes autoregressive decoding $O(n)$ per token." },
  ],

  cheatsheet: [
    { label: "Attention", code: "softmax(Q @ K.T / sqrt(dk)) @ V" },
    { label: "Stable softmax", code: "e=exp(x-x.max(-1,keepdims)); e/e.sum(-1,keepdims)" },
    { label: "Causal mask", code: "scores.masked_fill(tri_upper, -inf)" },
    { label: "Fused attn (torch)", code: "F.scaled_dot_product_attention(q,k,v,is_causal=True)" },
    { label: "Head split", code: "d_k = d // n_heads" },
    { label: "FFN width", code: "hidden = 4 * d" },
    { label: "Pre-norm block", code: "x = x + attn(LN(x)); x = x + ffn(LN(x))" },
    { label: "Sinusoid PE", code: "sin(pos / 10000**(2i/d))" },
    { label: "MHA layer", code: "nn.MultiheadAttention(d, n_heads, batch_first=True)" },
    { label: "Encoder block", code: "nn.TransformerEncoderLayer(d, n_heads)" },
    { label: "Load pretrained", code: "AutoModel.from_pretrained('bert-base-uncased')" },
    { label: "Attention cost", code: "O(n^2 * d)" },
  ],
});
