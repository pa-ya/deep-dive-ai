(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "nlp-seq2seq",
  name: "Seq2Seq & the Birth of Attention",
  language: "NLP",
  group: "Natural Language Processing",
  navLabel: "Seq2Seq & Attention",
  tagline: "How a fixed-vector bottleneck killed early neural translation — and how **attention** was invented to fix it, the idea that (three years later) became the Transformer.",
  color: "#0E7490",
  readMinutes: 44,
  sections: [
    {
      id: "problem",
      title: "The sequence-to-sequence problem",
      level: "core",
      body: [
        { type: "p", text: "So far the neural nets you have built map a **fixed-size input to a fixed-size output**: an image to a class, a feature vector to a price. But the most valuable language tasks are **sequence-to-sequence** — the input is a sequence of variable length and so is the output, and the two lengths need not match:" },
        { type: "table",
          headers: ["Task", "Input sequence", "Output sequence"],
          rows: [
            ["Machine translation", "*\"the cat sat\"* (English, 3 tokens)", "*\"le chat s'est assis\"* (French, 5 tokens)"],
            ["Summarization", "a 900-word article", "a 2-sentence abstract"],
            ["Question answering", "a question + passage", "an answer span or sentence"],
            ["Speech recognition", "a waveform of $T$ frames", "a transcript of $T'$ words"],
            ["Code generation", "a docstring", "a function body"],
          ]
        },
        { type: "p", text: "A plain classifier cannot do this. Its output layer has a fixed number of units, so it can only ever emit a fixed-length answer. Translation of *\"the cat sat\"* is 5 French tokens; translation of the next sentence might be 40. You need an architecture whose output length is **decided at run time** and whose every output token can depend on the **entire** input." },
        { type: "callout", variant: "note", text: "**Why this section exists.** Before 2014, machine translation was dominated by statistical phrase tables (Moses-style SMT) — huge, hand-engineered pipelines. Then two papers (Sutskever et al. and Cho et al.) showed a single neural network could learn to translate end-to-end. That work introduced the **encoder-decoder**, hit a wall called the **bottleneck**, and the fix for that wall — **attention** — is the single most important idea in modern AI. This is the historical bridge from RNNs to Transformers; by the end you will have *derived* attention, and the [Transformers section](#nlp-transformers) will take it the last step." },
      ]
    },
    {
      id: "encoder-decoder",
      title: "The RNN encoder–decoder (Sutskever et al., 2014)",
      level: "core",
      body: [
        { type: "p", text: "The founding idea is disarmingly simple. Use **two** RNNs. The first, the **encoder**, reads the whole source sentence and compresses it into a single fixed-length vector. The second, the **decoder**, is a conditional language model that generates the target sentence one token at a time, starting from that vector." },
        { type: "heading", text: "The encoder: read, then summarize" },
        { type: "p", text: "The encoder is an RNN (in practice an LSTM or GRU — see the RNN section for why the gates matter). It ingests the source tokens $x_1,\\dots,x_T$ and updates a hidden state at each step. Its **final** hidden state is taken as a summary of the entire sentence — the **context vector** $c$:" },
        { type: "math", tex: String.raw`h_t = f(h_{t-1},\, x_t), \qquad c = h_T \in \mathbb{R}^{d}` },
        { type: "p", text: "That is the crux: $c$ is a *single fixed-size vector* that must stand in for the whole source sentence, however long it was. Hold that thought — it is the seed of the coming problem." },
        { type: "heading", text: "The decoder: a language model conditioned on c" },
        { type: "p", text: "The decoder is a second RNN that generates the target sequence $y_1,\\dots,y_{T'}$. It factorizes the joint probability of the output using the chain rule — each token is predicted from all previously generated tokens **and** the context vector:" },
        { type: "math", tex: String.raw`p(y_1,\dots,y_{T'} \mid x) = \prod_{t=1}^{T'} p(y_t \mid y_1,\dots,y_{t-1},\, c)` },
        { type: "p", text: "Concretely, at each decoding step the decoder updates its own hidden state $s_t$ and emits a distribution over the vocabulary via a softmax:" },
        { type: "math", tex: String.raw`s_t = g(s_{t-1},\, y_{t-1},\, c), \qquad p(y_t \mid \cdot) = \operatorname{softmax}(W_o\, s_t + b_o)` },
        { type: "p", text: "The decoder's first hidden state $s_0$ is initialized from $c$ (often $s_0 = c$), and generation starts from a special `<sos>` (start-of-sequence) token and continues until the model emits `<eos>` (end-of-sequence). That `<eos>` token is how the network learns to **choose its own output length** — solving the variable-length problem." },
        { type: "heading", text: "Training objective & teacher forcing" },
        { type: "p", text: "Training is maximum likelihood: over a corpus of source–target pairs, minimize the negative log-likelihood of the correct target tokens (this is exactly cross-entropy, summed over output positions):" },
        { type: "math", tex: String.raw`\mathcal{L}(\theta) = -\sum_{(x,y)}\sum_{t=1}^{T'} \log p_\theta(y_t \mid y_{1:t-1},\, x)` },
        { type: "p", text: "There is a subtlety in *what you feed the decoder* at step $t$. You could feed its own previous prediction $\\hat y_{t-1}$ — but early in training those predictions are garbage, so errors compound and learning stalls. Instead, during training you feed the **ground-truth** previous token $y_{t-1}$ regardless of what the model predicted. This is **teacher forcing**, and it makes training fast and stable." },
        { type: "callout", variant: "gotcha", text: "**Teacher forcing creates a train/test mismatch (\"exposure bias\").** At training time the decoder always sees correct history; at inference it sees its *own* (possibly wrong) history, so a single mistake can knock it onto a path it never trained on. Mitigations include *scheduled sampling* (randomly feed the model's own predictions during training) — but the mismatch is inherent to autoregressive generation and never fully disappears." },
        { type: "callout", variant: "tip", text: "**Sutskever's famous trick: reverse the source.** The team found that feeding the source sentence *backwards* ($x_T,\\dots,x_1$) sharply improved BLEU. Why? It shortens the distance between the first source words and the first target words they generate, easing the RNN's short-term-memory burden. That this hack helped so much was a giant flashing clue that the fixed-vector bottleneck was the real enemy." },
      ]
    },
    {
      id: "bottleneck",
      title: "The bottleneck problem",
      level: "core",
      body: [
        { type: "p", text: "Look again at what the architecture demands. The **entire** meaning of the source sentence — every noun, every clause, every negation — must be crammed into one fixed vector $c \\in \\mathbb{R}^d$. For *\"the cat sat\"* that is easy. For a 50-word sentence with subordinate clauses, that single vector is a hopeless straw through which the whole meaning must be sucked. This is the **information bottleneck**." },
        { type: "math", tex: String.raw`\underbrace{x_1, x_2, \dots, x_T}_{\text{whole sentence}} \;\longrightarrow\; \underbrace{c \in \mathbb{R}^{d}}_{\text{one fixed vector}} \;\longrightarrow\; \underbrace{y_1, y_2, \dots, y_{T'}}_{\text{whole translation}}` },
        { type: "p", text: "There is also a **recency bias**: because $c = h_T$ is the *last* hidden state, it disproportionately reflects the *end* of the sentence. Words near the start have been overwritten by dozens of subsequent updates. (Reversing the input, per Sutskever, just moves the problem to the other end.)" },
        { type: "heading", text: "The empirical smoking gun: BLEU collapses with length" },
        { type: "p", text: "Cho et al. (2014) measured this directly. They plotted translation quality (BLEU) against source-sentence length. For short sentences the encoder-decoder was competitive; but past roughly **20–30 words, BLEU fell off a cliff** — the fixed vector simply could not hold a long sentence. The model was not failing to *learn*; it was failing to *store*." },
        { type: "table",
          headers: ["Source length", "Fixed-vector encoder-decoder", "Diagnosis"],
          rows: [
            ["short (< 10 words)", "good BLEU", "fits comfortably in $c$"],
            ["medium (10–20)", "decent, slipping", "$c$ getting crowded"],
            ["long (30–50+)", "**sharp BLEU drop**", "bottleneck overflows; early words lost"],
          ]
        },
        { type: "callout", variant: "note", text: "**The key realization.** Why force the decoder to work from one frozen summary at all? The encoder computed a hidden state $h_j$ at *every* source position — a rich, per-word representation — and we threw all of them away except the last. What if, at each step of decoding, the model could **look back at all of them** and pull out the few that matter for the word it is generating *right now*? That question is attention." },
      ]
    },
    {
      id: "bahdanau",
      title: "Bahdanau (additive) attention — the birth of attention",
      level: "core",
      body: [
        { type: "p", text: "In *\"Neural Machine Translation by Jointly Learning to Align and Translate\"* (Bahdanau, Cho & Bengio, 2014), the fix is introduced. **Keep every encoder hidden state.** Then, at each decoding step, compute a *fresh* context vector $c_t$ that is a **weighted average of all encoder states**, with the weights chosen to emphasize the source words relevant to the current target word. No single frozen summary — a new, focused summary for every output token." },
        { type: "heading", text: "Step 1 — keep all the encoder states (annotations)" },
        { type: "p", text: "Bahdanau uses a **bidirectional** encoder so each state summarizes context from both directions. The forward and backward hidden states at position $j$ are concatenated into an **annotation** $h_j$:" },
        { type: "math", tex: String.raw`h_j = \big[\,\overrightarrow{h_j}\, ;\, \overleftarrow{h_j}\,\big], \qquad j = 1,\dots,T` },
        { type: "p", text: "We now have a *matrix* of annotations $\\{h_1,\\dots,h_T\\}$, one column per source word, instead of a single vector. Nothing is discarded." },
        { type: "heading", text: "Step 2 — score how well each source word matches the current decoder state" },
        { type: "p", text: "At decoding step $t$, the decoder's previous state $s_{t-1}$ represents *\"what I am about to generate.\"* We measure the compatibility of that intent with each source annotation $h_j$ using a small learned feed-forward network — the **alignment model** $a$. Because it *adds* two projections inside a $\\tanh$, this is called **additive** (or **concat**) attention:" },
        { type: "math", tex: String.raw`e_{tj} = a(s_{t-1}, h_j) = v_a^\top \tanh\!\big(W_a\, s_{t-1} + U_a\, h_j\big)` },
        { type: "p", text: "Here $W_a, U_a, v_a$ are learned. Intuitively $e_{tj}$ answers *\"how relevant is source word $j$ to the target word I am generating at step $t$?\"* — a single scalar per source position." },
        { type: "heading", text: "Step 3 — softmax the scores into attention weights" },
        { type: "p", text: "Normalize the scores across all source positions so they form a probability distribution — a soft selection over the source. These are the **attention weights** $\\alpha_{tj}$:" },
        { type: "math", tex: String.raw`\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T}\exp(e_{tk})}, \qquad \sum_{j=1}^{T}\alpha_{tj} = 1` },
        { type: "heading", text: "Step 4 — context = weighted sum of annotations" },
        { type: "p", text: "The context vector for this step is the attention-weighted blend of all source annotations. Source words with high weight dominate; irrelevant words contribute almost nothing:" },
        { type: "math", tex: String.raw`c_t = \sum_{j=1}^{T} \alpha_{tj}\, h_j` },
        { type: "p", text: "Finally, the decoder uses this **step-specific** context (instead of a fixed $c$) to update its state and predict the token:" },
        { type: "math", tex: String.raw`s_t = g(s_{t-1},\, y_{t-1},\, c_t), \qquad p(y_t \mid \cdot) = \operatorname{softmax}\big(W_o\,[\,s_t; c_t\,] + b_o\big)` },
        { type: "callout", variant: "good", text: "**Why this dissolves the bottleneck.** Information no longer has to flow through one $d$-dimensional straw. Every output word gets direct, weighted access to *every* input word — the path length from any source word to any target word is now constant, not $O(T)$. Bahdanau's model (\"RNNsearch\") held its BLEU **flat** as sentences grew past 30, 40, 50 words, exactly where the fixed-vector model collapsed. The plot in that paper is one of the most consequential figures in deep learning." },
        { type: "callout", variant: "note", text: "**\"Align and translate.\"** The title is precise: the $\\alpha_{tj}$ weights *are* a learned soft **alignment** between source and target words — the thing classical SMT needed a whole separate alignment model (GIZA++) to estimate. Bahdanau's network learns to align and to translate *jointly*, end-to-end, from nothing but sentence pairs. We visualize these alignments in a later section." },
      ]
    },
    {
      id: "luong",
      title: "Luong (multiplicative / dot) attention",
      level: "core",
      body: [
        { type: "p", text: "One year later, Luong, Pham & Manning (2015), *\"Effective Approaches to Attention-based NMT,\"* simplified and generalized the scoring step. Their central observation: the additive network $v_a^\\top\\tanh(\\cdots)$ is expressive but expensive. If the decoder and encoder states live in compatible spaces, you can score compatibility with a plain **dot product** — the alignment operation we already know measures similarity (recall the dot product as alignment from Linear Algebra)." },
        { type: "heading", text: "Three scoring functions" },
        { type: "p", text: "Luong uses the *current* decoder state $s_t$ (a small but important change from Bahdanau's $s_{t-1}$) and proposes three ways to score it against each annotation $h_j$:" },
        { type: "table",
          headers: ["Name", "Score $\\;\\text{score}(s_t, h_j)$", "Character"],
          rows: [
            ["dot", "$s_t^\\top h_j$", "no parameters; needs matching dims"],
            ["general", "$s_t^\\top W_a\\, h_j$", "one learned matrix bridges the spaces"],
            ["concat (additive)", "$v_a^\\top \\tanh(W_a[s_t; h_j])$", "Bahdanau-style; most expressive, slowest"],
          ]
        },
        { type: "p", text: "The **dot** and **general** forms are called **multiplicative** attention. Everything downstream is identical to Bahdanau — softmax the scores into weights $\\alpha_{tj}$, then $c_t = \\sum_j \\alpha_{tj} h_j$ — only the scoring function changed. Luong then combines the context with the decoder state into an *attentional hidden state* $\\tilde h_t = \\tanh(W_c[c_t; s_t])$ and predicts from that." },
        { type: "callout", variant: "tip", text: "**Additive vs multiplicative — the practical trade-off.** They perform comparably in quality, but multiplicative attention is far faster and more memory-efficient, because a batch of dot products is a single matrix multiply $S = QK^\\top$ that runs at full speed on a GPU's matmul units. The additive network cannot be fused into one matmul. This efficiency is exactly why the dot-product variant won — and why it became the template for what came next." },
        { type: "heading", text: "The direct ancestor of the Transformer" },
        { type: "p", text: "Look hard at Luong's **general** score $s_t^\\top W_a h_j$. Fold the learned matrix into the two vectors — project the decoder state into a **query** $q = W^Q s_t$ and each annotation into a **key** $k_j = W^K h_j$ — and the score is simply $q^\\top k_j$. Do the same to the values ($v_j = W^V h_j$) that get averaged. Add one scaling factor $1/\\sqrt{d_k}$ and you have **scaled dot-product attention**, the beating heart of the Transformer:" },
        { type: "math", tex: String.raw`\underbrace{s_t^\top W_a\, h_j}_{\text{Luong general (2015)}} \;\;\longrightarrow\;\; \underbrace{\frac{q^\top k_j}{\sqrt{d_k}}}_{\text{scaled dot-product (2017)}}, \qquad q = W^Q s_t,\;\; k_j = W^K h_j` },
        { type: "callout", variant: "note", text: "We stop the derivation exactly here on purpose. The **Query/Key/Value** framing, the $1/\\sqrt{d_k}$ scaling and *why* it is needed, multi-head attention, and what happens when a sequence attends *to itself* are developed in full in the [Transformers section](#nlp-transformers). For now, the one thing to carry forward: the Transformer's attention is **Luong dot-product attention with learned Q/K/V projections and a scaling factor** — nothing more exotic. You have already met its parent." },
      ]
    },
    {
      id: "alignment",
      title: "Attention as soft alignment (and how to read the heatmaps)",
      level: "core",
      body: [
        { type: "p", text: "The attention weights $\\alpha_{tj}$ are not just an implementation detail — they are **interpretable**. Stack them into a matrix $A$ with target words down the rows and source words across the columns. Each entry $A_{tj} = \\alpha_{tj}$ is how much target word $t$ drew on source word $j$. Plotting $A$ as a heatmap literally *shows you what the model looked at* to produce each word — the soft alignment between the two languages." },
        { type: "heading", text: "What the pictures reveal" },
        { type: "list", ordered: false, items: [
          "**A bright diagonal** means monotonic, word-for-word alignment — common between similar languages that share word order.",
          "**Off-diagonal blocks** reveal **reordering**: translating English *\"European Economic Area\"* to French *\"zone économique européenne\"* flips the adjective order, and the attention matrix shows a clean crossing pattern exactly there — the model learned the reordering with no explicit rule.",
          "**One target word attending to several source words** (a horizontal smear) captures many-to-one alignments, like a French contraction covering an English phrase.",
          "**`<eos>` attention** often parks on sentence-final punctuation — the model learning where to stop.",
        ]},
        { type: "callout", variant: "tip", text: "**Soft vs hard attention.** What we have built is **soft** attention: a differentiable weighted *average* over all source positions, so gradients flow to every annotation and you can train it with plain backprop. **Hard** attention (Xu et al., *\"Show, Attend and Tell,\"* 2015) instead *samples one* position to attend to — sharper and cheaper at inference, but non-differentiable, so it needs reinforcement-learning-style estimators to train. Soft attention won for exactly this reason: it drops straight into gradient descent. Every attention you meet after this is soft." },
        { type: "callout", variant: "good", text: "**This interpretability is rare and precious.** Most of a neural net is an inscrutable tangle of weights. Attention hands you a human-readable map of the model's reasoning for free. The same trick — plotting attention weights — is how researchers later probed what *Transformer* heads learn (previous-token heads, coreference heads). The habit starts here, on a translation alignment heatmap." },
      ]
    },
    {
      id: "leap",
      title: "The leap: \"Attention Is All You Need\"",
      level: "core",
      body: [
        { type: "p", text: "By 2016 attention was everywhere — but always as an *add-on* bolted to an RNN encoder-decoder. The RNN did the sequence modeling; attention just let the decoder peek back at the encoder. Then someone asked the question that defined the next decade of AI: **what does the RNN still buy us?**" },
        { type: "p", text: "Reconsider what attention already does. It connects any two positions in **one hop**, regardless of distance — the very thing RNNs struggle with. It has no recency bias. And it is a pure matrix multiply, so it **parallelizes** perfectly across positions. The RNN, by contrast, is inherently *sequential* — it must process token $t$ before token $t+1$ — which is a catastrophe for GPU utilization and the main reason these models were slow to train." },
        { type: "table",
          headers: ["Property", "RNN encoder-decoder", "Attention"],
          rows: [
            ["Path length between distant tokens", "$O(T)$ steps", "$O(1)$ — one hop"],
            ["Parallelism across the sequence", "**none** (sequential)", "**full** (one matmul)"],
            ["Long-range dependencies", "strained even with LSTM gates", "direct, distance-free"],
            ["Notion of order", "built in (recurrence)", "**none** — must be added back"],
          ]
        },
        { type: "p", text: "So Vaswani et al. (2017) took the radical step: **keep the attention, throw away the recurrence entirely.** Replace the RNN encoder with layers of tokens attending *to each other* (**self-attention**), replace the RNN decoder likewise, and connect them with the cross-attention we just built. With no recurrence there is no inherent order — so they add a **positional encoding** back in (the one thing recurrence gave for free). The result trained faster *and* translated better. The paper's title says it plainly: *Attention Is All You Need*." },
        { type: "callout", variant: "note", text: "**You are now one step from the Transformer.** Everything in this section — dot-product scoring, softmax weights, context as a weighted sum of values, cross-attention between two sequences — carries over unchanged. What is new in the [Transformers section](#nlp-transformers) is: (1) letting a sequence attend to *itself* (self-attention), (2) the Q/K/V projections and $1/\\sqrt{d_k}$ scaling, (3) multiple heads in parallel, and (4) positional encodings to replace the lost recurrence. Go there next; you have already built its foundation." },
      ]
    },
    {
      id: "from-scratch",
      title: "From scratch: attention scoring and a mini seq2seq",
      level: "core",
      body: [
        { type: "p", text: "Nothing makes attention concrete like watching the weights form. First, both scoring functions in pure NumPy on a toy setup: 5 encoder annotations and one decoder state. We compute Bahdanau (additive) and Luong (dot) scores, softmax them, and read off the context." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef softmax(z):\n    z = z - z.max()                 # numerical stability (see Transformers)\n    e = np.exp(z)\n    return e / e.sum()\n\nnp.random.seed(0)\nT, d = 5, 4                          # 5 source words, 4-dim states\nH = np.random.randn(T, d)           # encoder annotations h_1..h_5  (T, d)\ns = np.random.randn(d)              # current decoder state s_t     (d,)\n\n# --- Luong 'dot' attention -------------------------------------------\nscores_dot = H @ s                  # e_j = h_j . s        -> (T,)\nalpha_dot  = softmax(scores_dot)    # attention weights    -> (T,)\nc_dot      = alpha_dot @ H          # context = weighted sum of h_j -> (d,)\n\n# --- Bahdanau 'additive' attention -----------------------------------\nda = 6                              # alignment hidden size\nWa = np.random.randn(da, d)        # projects decoder state\nUa = np.random.randn(da, d)        # projects each annotation\nva = np.random.randn(da)           # scores the tanh\nscores_add = np.tanh(H @ Ua.T + (Wa @ s)) @ va   # e_j = va^T tanh(Wa s + Ua h_j)\nalpha_add  = softmax(scores_add)\nc_add      = alpha_add @ H\n\nprint('dot weights     ', alpha_dot.round(3), 'sum =', alpha_dot.sum())\nprint('additive weights', alpha_add.round(3))\nprint('context (dot)   ', c_dot.round(3))\n# Both weight vectors are non-negative and sum to 1: a soft selection over source words." },
        { type: "callout", variant: "tip", text: "Notice the dot-product path is a single matrix-vector product `H @ s`. Batch the decoder states into a matrix $Q$ and it becomes `Q @ H.T` — one matmul for *all* query-source pairs. That is the exact shape of the Transformer's $QK^\\top$; you are already writing its inner loop." },
        { type: "p", text: "Now a minimal but complete **seq2seq-with-attention** in PyTorch, trained on a toy **sequence-reversal** task (output = input reversed). It exercises every piece: a GRU encoder that keeps all states, a Bahdanau attention module, and a GRU decoder that builds a fresh context each step." },
        { type: "code", lang: "py", code: "import torch, torch.nn as nn, torch.nn.functional as F\n\nclass Encoder(nn.Module):\n    def __init__(self, vocab, d):\n        super().__init__()\n        self.emb = nn.Embedding(vocab, d)\n        self.rnn = nn.GRU(d, d, batch_first=True)\n    def forward(self, x):\n        H, hn = self.rnn(self.emb(x))   # H: (B,T,d) ALL states; hn: (1,B,d)\n        return H, hn.squeeze(0)          # keep every annotation, not just the last\n\nclass BahdanauAttention(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.Wa, self.Ua, self.va = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, 1, bias=False)\n    def forward(self, s, H):             # s:(B,d) decoder state, H:(B,T,d) annotations\n        e = self.va(torch.tanh(self.Wa(s).unsqueeze(1) + self.Ua(H)))  # (B,T,1) scores\n        alpha = F.softmax(e.squeeze(-1), dim=1)                        # (B,T) weights\n        c = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)               # (B,d) context\n        return c, alpha\n\nclass Decoder(nn.Module):\n    def __init__(self, vocab, d):\n        super().__init__()\n        self.emb  = nn.Embedding(vocab, d)\n        self.attn = BahdanauAttention(d)\n        self.rnn  = nn.GRUCell(2 * d, d)      # input = [emb(y_{t-1}); context]\n        self.out  = nn.Linear(2 * d, vocab)   # predict from [s_t; c_t]\n    def step(self, y_prev, s, H):\n        c, alpha = self.attn(s, H)\n        s = self.rnn(torch.cat([self.emb(y_prev), c], dim=-1), s)\n        return self.out(torch.cat([s, c], dim=-1)), s, alpha\n\ndef reverse_batch(B, T, vocab):\n    x = torch.randint(3, vocab, (B, T))       # ids 0,1,2 reserved: pad/sos/eos\n    return x, torch.flip(x, dims=[1])          # target is the reversed source\n\nvocab, d = 20, 32\nenc, dec = Encoder(vocab, d), Decoder(vocab, d)\nopt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=1e-2)\n\nfor step in range(400):\n    x, y = reverse_batch(64, 6, vocab)\n    H, s = enc(x)\n    y_prev = torch.ones(64, dtype=torch.long)  # <sos> = id 1\n    loss = 0.0\n    for t in range(y.size(1)):\n        logits, s, _ = dec.step(y_prev, s, H)\n        loss = loss + F.cross_entropy(logits, y[:, t])\n        y_prev = y[:, t]                        # teacher forcing: feed the truth\n    opt.zero_grad(); loss.backward(); opt.step()\n    if step % 100 == 0:\n        print(f'step {step:3d}  loss/token {loss.item()/y.size(1):.3f}')\n# Loss falls toward 0 as the model learns to copy-in-reverse by attending to the mirror position." },
        { type: "callout", variant: "gotcha", text: "Two beginner traps live in this code. (1) The encoder must return **all** hidden states `H`, not just the final one — returning only `hn` rebuilds the very bottleneck we are escaping. (2) The inner loop uses **teacher forcing** (`y_prev = y[:, t]`); to *generate* at inference you instead feed the model's own `argmax` and stop at `<eos>`, which is where exposure bias creeps in." },
      ]
    },
    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Attention is best understood by watching its weights. Do at least project 1 and project 3 — printing and plotting the $\\alpha$ matrix is the fastest path to the intuition the rest of the NLP track assumes." },
        { type: "list", ordered: true, items: [
          "**Reversal, then copy, then sort.** Extend the toy PyTorch model above to three tasks: reverse a sequence, copy it verbatim, and sort it. For each, extract and plot the attention matrix $\\alpha$. You should *see* the alignment change — a clean anti-diagonal for reversal, a diagonal for copy, a scramble for sort. This is the single most instructive exercise in the section.",
          "**Additive vs multiplicative bake-off.** Swap the Bahdanau attention module for Luong dot and Luong general scoring (three variants). Compare convergence speed and final loss on the reversal task, and time a training step for each. Confirm the dot variant is fastest — the reason the Transformer chose it.",
          "**Real translation alignments.** Train a small GRU seq2seq-with-attention on a real parallel corpus (e.g. the Multi30k or Tatoeba EN-FR pairs). Pick a sentence with adjective-noun reordering and plot its alignment heatmap; find the crossing pattern where the languages disagree on word order.",
          "**Kill the attention, feel the bottleneck.** Replace attention with the original fixed context vector $c = h_T$ and retrain on progressively longer sequences. Plot accuracy vs. length for both models — reproduce Cho et al.'s BLEU-vs-length collapse on your own machine.",
          "**Greedy vs beam search.** Implement inference (no teacher forcing): greedy `argmax` decoding, then beam search with width 3–5. Measure how often beam search fixes an early mistake that greedy decoding cannot recover from — a concrete look at exposure bias.",
          "**Trace the lineage to Q/K/V.** Rewrite Luong's `general` score $s_t^\\top W_a h_j$ by factoring $W_a$ into separate query and key projections, then add $1/\\sqrt{d_k}$ scaling. Confirm numerically it equals scaled dot-product attention — bridging this section to the Transformer by hand.",
        ]},
      ]
    },
    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The four papers below are the actual birth of attention, in order; the last two are the best explainers. Read Bahdanau 2014 in full — it is short, lucid, and historic." },
        { type: "link", url: "https://arxiv.org/abs/1409.3215", text: "Sutskever, Vinyals & Le (2014) — Sequence to Sequence Learning with Neural Networks (the RNN encoder-decoder; the reverse-the-source trick)" },
        { type: "link", url: "https://arxiv.org/abs/1409.0473", text: "Bahdanau, Cho & Bengio (2014) — Neural MT by Jointly Learning to Align and Translate (the birth of attention; additive scoring)" },
        { type: "link", url: "https://arxiv.org/abs/1508.04025", text: "Luong, Pham & Manning (2015) — Effective Approaches to Attention-based NMT (dot/general/concat scores; global vs local)" },
        { type: "link", url: "https://arxiv.org/abs/1406.1078", text: "Cho et al. (2014) — Learning Phrase Representations with RNN Encoder-Decoder (introduces the GRU; documents the length bottleneck)" },
        { type: "link", url: "https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/", text: "Jay Alammar — Visualizing seq2seq Models with Attention (the illustrated companion to this whole section)" },
        { type: "link", url: "https://arxiv.org/abs/1706.03762", text: "Vaswani et al. (2017) — Attention Is All You Need (where this story leads; read the Transformers section first, then this)" },
      ]
    },
  ],

  packages: [
    { name: "torch.nn.GRU / nn.LSTM", why: "the recurrent encoder/decoder cells — return all hidden states, not just the last" },
    { name: "torch.nn.GRUCell", why: "single-step recurrence for a hand-rolled decoder loop with attention" },
    { name: "torch.bmm", why: "batched matrix multiply — the weighted sum $\\sum_j \\alpha_{tj} h_j$ as one op" },
    { name: "torch.nn.functional.softmax", why: "turn alignment scores into attention weights over source positions" },
    { name: "torchtext / datasets", why: "parallel corpora (Multi30k, Tatoeba) and tokenization for real translation" },
    { name: "sacrebleu", why: "standard BLEU scoring to reproduce the quality-vs-length curves" },
    { name: "matplotlib", why: "plot the $\\alpha$ alignment matrix as a heatmap — the payoff of the section" },
  ],

  gotchas: [
    "The encoder must expose **all** hidden states $h_1..h_T$; keeping only $h_T$ rebuilds the exact bottleneck attention exists to remove.",
    "Attention weights $\\alpha_{tj}$ must be softmax-normalized **across source positions** (they sum to 1 per target step) — normalizing the wrong axis silently breaks alignment.",
    "**Teacher forcing** (feed ground-truth $y_{t-1}$) is a training-only convenience; at inference you feed the model's own output, causing exposure bias.",
    "Bahdanau uses the *previous* decoder state $s_{t-1}$ to score; Luong uses the *current* $s_t$. Small change, different information flow — don't mix them up.",
    "**Dot** attention $s_t^\\top h_j$ requires encoder and decoder states to share a dimension; use Luong **general** ($s_t^\\top W_a h_j$) when they don't.",
    "BLEU with the fixed-vector model collapses past ~30 words — that is the bottleneck, not a bug in your training loop.",
    "Soft attention is differentiable and trains with plain backprop; **hard** attention samples a position and needs RL-style estimators — don't reach for it by default.",
    "Remember `<sos>`/`<eos>` tokens: `<eos>` is how the decoder learns its own output length, and forgetting it makes generation never stop.",
  ],

  flashcards: [
    { q: "What is the sequence-to-sequence problem, and why can't a plain classifier solve it?", a: "Map a variable-length input sequence to a variable-length output sequence (translation, summarization). A classifier has a fixed-size output layer, so it cannot emit a run-time-decided length." },
    { q: "Describe the RNN encoder-decoder (Sutskever 2014).", a: "An encoder RNN compresses the source into a fixed context vector $c = h_T$; a decoder RNN generates the target autoregressively, conditioned on $c$, factorized as $\\prod_t p(y_t \\mid y_{<t}, c)$." },
    { q: "What is teacher forcing and its downside?", a: "During training, feed the decoder the ground-truth previous token (not its own prediction) for fast, stable learning. Downside: exposure bias — a train/test mismatch since inference feeds the model's own (possibly wrong) tokens." },
    { q: "What is the bottleneck problem?", a: "Cramming an entire source sentence into one fixed vector $c$ loses information; BLEU collapses past ~30 words. Early words are also overwritten (recency bias)." },
    { q: "Derive Bahdanau (additive) attention in four steps.", a: "(1) keep all annotations $h_j$; (2) score $e_{tj}=v_a^\\top\\tanh(W_a s_{t-1}+U_a h_j)$; (3) $\\alpha_{tj}=\\operatorname{softmax}_j(e_{tj})$; (4) $c_t=\\sum_j \\alpha_{tj} h_j$ — a fresh context per output token." },
    { q: "How does Luong dot attention differ, and why did it win?", a: "It scores with a dot product $s_t^\\top h_j$ (or $s_t^\\top W_a h_j$) instead of a $\\tanh$ network. Comparable quality, but a single fused matmul — far faster on GPUs." },
    { q: "How is Luong attention the ancestor of the Transformer?", a: "Project the decoder state to a query and annotations to keys/values, and $s_t^\\top W_a h_j$ becomes $q^\\top k_j$; add $1/\\sqrt{d_k}$ and you have scaled dot-product attention." },
    { q: "What do attention weights represent, and how do you read them?", a: "A soft alignment between target and source words. Plot $\\alpha_{tj}$ as a heatmap: a diagonal is monotonic alignment; crossings reveal reordering." },
    { q: "Soft vs hard attention?", a: "Soft = differentiable weighted average over all positions (trains with backprop). Hard = sample one position (sharper, cheaper, but non-differentiable; needs RL-style training)." },
    { q: "What did 'Attention Is All You Need' remove, and what did it add back?", a: "Removed recurrence entirely (keeping only attention, now including self-attention) for O(1) paths and full parallelism; added positional encodings to restore the order that recurrence had provided." },
  ],

  cheatsheet: [
    { label: "Encoder-decoder factorization", code: "p(y|x) = prod_t p(y_t | y_<t, c)" },
    { label: "Context vector (fixed)", code: "c = h_T   # the bottleneck" },
    { label: "Bahdanau score", code: "e_tj = va @ tanh(Wa @ s + Ua @ h_j)" },
    { label: "Luong dot score", code: "e_tj = s_t @ h_j" },
    { label: "Luong general score", code: "e_tj = s_t @ Wa @ h_j" },
    { label: "Attention weights", code: "alpha = softmax(e, axis=source)" },
    { label: "Context (attention)", code: "c_t = sum_j alpha_tj * h_j" },
    { label: "Batched dot scores", code: "S = Q @ H.T   # -> Transformer's QK^T" },
    { label: "Weighted sum (torch)", code: "torch.bmm(alpha[:,None,:], H).squeeze(1)" },
    { label: "Teacher forcing", code: "y_prev = y[:, t]   # feed the truth" },
    { label: "Keep ALL encoder states", code: "H, hn = self.rnn(emb(x))  # use H" },
    { label: "Plot alignment", code: "plt.imshow(alpha)  # target x source" },
  ],
});
