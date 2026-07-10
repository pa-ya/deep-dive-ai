(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "nn-rnn",
  name: "Recurrent Neural Networks",
  language: "Neural Networks",
  group: "Neural Networks",
  navLabel: "RNNs & LSTMs",
  tagline: "How neural nets learned to read sequences — vanilla RNNs, backprop through time, the vanishing-gradient wall, and the **LSTM/GRU** gates that broke through it, all derived and built from scratch.",
  color: "#FBBF24",
  readMinutes: 50,
  sections: [
    {
      id: "why",
      title: "Why sequences need a different architecture",
      level: "core",
      body: [
        { type: "p", text: "Every network you have built so far is **feed-forward**: a fixed-size input goes in, a fixed-size output comes out, and each input is processed independently of the last. That is fine for a house-price vector or a $28\\times28$ image. But an enormous share of the world's data is **sequential** — text, speech, sensor streams, stock ticks, DNA — where the order carries the meaning and the length is not fixed." },
        { type: "p", text: "Sequences break the feed-forward assumptions in two specific ways:" },
        { type: "table",
          headers: ["Problem", "Why an MLP/CNN fails", "What we need"],
          rows: [
            ["Variable length", "the input layer has a fixed number of units", "a model that ingests one element at a time, for as many steps as needed"],
            ["Order matters", "*\"dog bites man\"* ≠ *\"man bites dog\"*, but a bag-of-features loses order", "a state that remembers what came before"],
            ["Long-range dependency", "*\"The **keys** … are on the table\"* — verb agrees with a noun 6 words back", "memory that carries information across many steps"],
            ["Parameter sharing", "a fixed-window model relearns *\"cat\"* at every position", "the **same** weights applied at every timestep"],
          ]
        },
        { type: "p", text: "The **recurrent neural network** (RNN) answers all four with one idea: process the sequence one element at a time, and maintain a **hidden state** $h_t$ — a running summary of everything seen so far — that is fed back into the network at the next step. The same weights are reused at every step, so the model handles any length and shares what it learns across positions." },
        { type: "callout", variant: "note", text: "**History.** The recurrent idea goes back to the Hopfield network (1982) and Jordan/Elman networks (late 1980s). RNNs were notoriously hard to train until Hochreiter & Schmidhuber's **LSTM** (1997) tamed the vanishing gradient. RNNs then dominated sequence modeling from roughly 2013–2017 — powering the first neural machine translation, speech recognition, and Karpathy's viral *char-RNN* — before the **Transformer** (2017) displaced them. This section is that whole arc: why RNNs work, why they were hard, how LSTMs fixed them, and why attention eventually won." },
        { type: "callout", variant: "tip", text: "**How to read this.** Everything builds on the [Backpropagation](#nn-backprop) section — an RNN is just a very deep feed-forward net with **shared weights**, and \"backprop through time\" is ordinary backprop on that unrolled net. If the chain-rule recurrence there is fresh, this will feel familiar." },
      ]
    },
    {
      id: "recurrence",
      title: "The vanilla RNN: recurrence and unrolling",
      level: "core",
      body: [
        { type: "p", text: "An RNN reads a sequence $x_1, x_2, \\dots, x_T$ one vector at a time. At each step it combines the **current input** $x_t$ with the **previous hidden state** $h_{t-1}$ to produce a **new hidden state** $h_t$. That is the entire recurrence:" },
        { type: "math", tex: String.raw`h_t = \phi\!\left(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h\right)` },
        { type: "p", text: "Here $x_t \\in \\mathbb{R}^{n}$ is the input at step $t$, $h_t \\in \\mathbb{R}^{d}$ is the hidden state, $\\phi$ is a non-linearity (classically $\\tanh$), and the three parameters are shared across **all** timesteps: $W_{xh}\\in\\mathbb{R}^{d\\times n}$ maps input→hidden, $W_{hh}\\in\\mathbb{R}^{d\\times d}$ maps hidden→hidden (the recurrent connection), and $b_h\\in\\mathbb{R}^{d}$ is the bias. The state is usually seeded at $h_0 = \\mathbf{0}$." },
        { type: "p", text: "To read anything *out* of the network — a prediction at some or every step — a separate output head maps the hidden state to the output space:" },
        { type: "math", tex: String.raw`\hat y_t = W_{hy}\, h_t + b_y \qquad (\text{then softmax / sigmoid / identity as needed})` },
        { type: "heading", text: "Unrolling: an RNN is a deep net with tied weights" },
        { type: "p", text: "The recurrence looks compact, but if you **unroll** it across time — write out $h_1, h_2, \\dots, h_T$ explicitly — you see the true structure: a $T$-layer feed-forward network where every layer uses the *same* weight matrices. Depth in an RNN is depth in **time**." },
        { type: "code", lang: "text", code: "x1        x2        x3        x4\n |         |         |         |\n v         v         v         v\nh0 --> [ h1 ] --> [ h2 ] --> [ h3 ] --> [ h4 ]\n         |         |         |         |\n         v         v         v         v\n        y1        y2        y3        y4\n\nSame W_xh, W_hh, W_hy reused at every step  (weight sharing)" },
        { type: "p", text: "This unrolled view is the key mental model. It tells you two things at once: (1) how to compute — a forward loop over $t$; and (2) how to train — ordinary backpropagation over the unrolled graph, which is called **backpropagation through time**. It also foreshadows the trouble: a sequence of length 100 is a *100-layer-deep* network, and we already saw in the backprop section what depth does to gradients." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef rnn_forward(xs, h0, Wxh, Whh, Why, bh, by, phi=np.tanh):\n    \"\"\"Run a vanilla RNN over a sequence. xs: list of (n,) input vectors.\"\"\"\n    h = h0\n    hs, ys = [], []\n    for x in xs:\n        h = phi(Whh @ h + Wxh @ x + bh)   # the recurrence\n        y = Why @ h + by                  # output head\n        hs.append(h); ys.append(y)\n    return hs, ys\n\nn, d, k = 5, 8, 3                          # input, hidden, output dims\nrng = np.random.default_rng(0)\nWxh = rng.normal(0, 0.1, (d, n))\nWhh = rng.normal(0, 0.1, (d, d))\nWhy = rng.normal(0, 0.1, (k, d))\nbh, by = np.zeros(d), np.zeros(k)\n\nxs = [rng.normal(size=n) for _ in range(4)]   # a length-4 sequence\nhs, ys = rnn_forward(xs, np.zeros(d), Wxh, Whh, Why, bh, by)\nprint(len(hs), hs[-1].shape)   # 4 states, each (8,)" },
        { type: "callout", variant: "gotcha", text: "The hidden state $h_t$ is the RNN's **only** memory — everything the network knows about steps $1\\dots t$ must fit in that one $d$-dimensional vector. This is the same *information-bottleneck* pressure that will later doom the seq2seq context vector and motivate attention (see [Seq2Seq & Attention](#nlp-seq2seq))." },
      ]
    },
    {
      id: "bptt",
      title: "Backpropagation through time (BPTT), derived",
      level: "core",
      body: [
        { type: "p", text: "Training an RNN means computing $\\partial\\mathcal{L}/\\partial W$ for the shared matrices and running gradient descent. Because the unrolled network is just a deep feed-forward net, we use plain backprop — but with one twist created by weight sharing. The recipe has three moves: **unroll**, **backprop through the unrolled graph**, then **sum the gradients** of the shared weights across all timesteps." },
        { type: "p", text: "Take a total loss that sums a per-step loss over the sequence (e.g. cross-entropy at each position):" },
        { type: "math", tex: String.raw`\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t, \qquad \mathcal{L}_t = \ell(\hat y_t, y_t)` },
        { type: "heading", text: "The hidden-state error flows backward in time" },
        { type: "p", text: "The crucial quantity, exactly as in feed-forward backprop, is the error on each hidden state, $\\partial\\mathcal{L}/\\partial h_t$. But $h_t$ influences the loss through **two** paths: directly via its own output $\\hat y_t$, and indirectly via the *next* hidden state $h_{t+1}$ (since $h_{t+1}$ depends on $h_t$). The multivariable chain rule sums both paths:" },
        { type: "math", tex: String.raw`\frac{\partial \mathcal{L}}{\partial h_t} = \underbrace{W_{hy}^{\top}\frac{\partial \mathcal{L}_t}{\partial \hat y_t}}_{\text{local output}} + \underbrace{W_{hh}^{\top}\left(\frac{\partial \mathcal{L}}{\partial h_{t+1}}\odot \phi'(z_{t+1})\right)}_{\text{error from the future}}` },
        { type: "p", text: "where $z_{t+1} = W_{hh}h_t + W_{xh}x_{t+1} + b_h$ is the pre-activation. Read the second term carefully: the error at step $t$ receives a contribution from step $t+1$, pulled back through the transposed recurrent matrix $W_{hh}^\\top$ and the activation slope $\\phi'$. Errors literally propagate **backward through time**, step by step, from $T$ down to $1$ — hence the name." },
        { type: "heading", text: "Shared weights: sum the per-step gradients" },
        { type: "p", text: "Now the twist. Because the *same* $W_{hh}$ appears at every timestep, it has many \"copies\" in the unrolled graph. The rule for a shared parameter is: compute the gradient at each place it is used and **add them all up** (the multivariate chain rule again — a parameter influencing the loss through many paths sums over those paths). Define the per-step pre-activation error $\\delta_t = (\\partial\\mathcal{L}/\\partial h_t)\\odot\\phi'(z_t)$; then:" },
        { type: "math", tex: String.raw`\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \delta_t\, h_{t-1}^{\top}, \qquad \frac{\partial \mathcal{L}}{\partial W_{xh}} = \sum_{t=1}^{T} \delta_t\, x_t^{\top}, \qquad \frac{\partial \mathcal{L}}{\partial b_h} = \sum_{t=1}^{T} \delta_t` },
        { type: "callout", variant: "note", text: "**This summation is the entire difference between BPTT and ordinary backprop.** In a normal deep net each layer has its own weights, so each gradient is used once. In an RNN the weights are tied across time, so you accumulate the gradient over every timestep before taking a single update. Forget the sum and you train as if only the last step existed." },
        { type: "heading", text: "Truncated BPTT — you cannot backprop through 10,000 steps" },
        { type: "p", text: "For a long sequence (a whole book, an hour of audio), unrolling the full length is impossible: storing every activation blows up memory, and the gradient path is astronomically deep. The practical fix is **truncated BPTT**: process the sequence in chunks of $k$ steps, carry the hidden state *forward* between chunks (so the model still has long-range memory in its state), but only backpropagate the gradient $k$ steps back before cutting it off." },
        { type: "code", lang: "py", code: "# Truncated BPTT sketch: forward the state, backprop only k steps.\nh = np.zeros(d)\nfor chunk in chunks_of(sequence, k=35):        # e.g. 35-step windows\n    h = h.detach() if hasattr(h, 'detach') else h.copy()  # cut the graph\n    hs, ys, cache = forward(chunk, h)          # h carries memory across chunks\n    loss = sequence_loss(ys, targets)\n    grads = bptt_backward(cache)                # gradient only spans this chunk\n    clip_gradients(grads, max_norm=5.0)         # (see next section: exploding grads)\n    sgd_step(params, grads)\n    h = hs[-1]                                  # keep the last state, drop its history" },
        { type: "callout", variant: "gotcha", text: "Truncation trades correctness for tractability: dependencies **longer than the truncation window $k$ get no gradient signal**, so the model can only *learn* patterns up to that horizon (even though the forwarded hidden state can still *carry* information further). Choosing $k$ is a memory-vs-reach knob. This limitation is one more reason Transformers, which attend to the whole context at once, eventually won." },
      ]
    },
    {
      id: "vanishing",
      title: "Vanishing & exploding gradients — in time",
      level: "core",
      body: [
        { type: "p", text: "The backprop section showed that gradients in a deep net are a **product** of per-layer factors, and if those factors are consistently $<1$ or $>1$ the gradient vanishes or explodes over depth. An RNN is depth-in-time, so the *same* pathology appears — but it is arguably worse, because the repeated factor is the **same** matrix $W_{hh}$ multiplied by itself over and over." },
        { type: "p", text: "Trace the sensitivity of a late loss to an early state. Applying the recurrence's Jacobian at each step, the gradient that reaches step $t$ from step $T$ is a long product:" },
        { type: "math", tex: String.raw`\frac{\partial \mathcal{L}_T}{\partial h_t} = \frac{\partial \mathcal{L}_T}{\partial h_T}\prod_{\tau=t+1}^{T}\frac{\partial h_\tau}{\partial h_{\tau-1}}, \qquad \frac{\partial h_\tau}{\partial h_{\tau-1}} = \operatorname{diag}\!\big(\phi'(z_\tau)\big)\, W_{hh}` },
        { type: "p", text: "That product has $T-t$ factors, each of which is essentially $W_{hh}$ scaled by activation slopes. Its magnitude is governed by the **largest singular value** (spectral norm) of $W_{hh}$, call it $\\gamma$, together with the bound $|\\phi'|\\le 1$ for $\\tanh$:" },
        { type: "math", tex: String.raw`\left\|\frac{\partial \mathcal{L}_T}{\partial h_t}\right\| \;\lesssim\; \gamma^{\,T-t}\quad\Longrightarrow\quad \begin{cases}\gamma < 1: & \text{vanishes exponentially}\\[2pt] \gamma > 1: & \text{explodes exponentially}\end{cases}` },
        { type: "p", text: "Because the exponent is the **time gap** $T-t$, information from far in the past decays (or blows up) geometrically. A gap of 50 steps with $\\gamma = 0.9$ scales the gradient by $0.9^{50}\\approx 0.005$ — the early input has essentially zero influence on the late loss. This is precisely why a vanilla RNN **cannot learn long-range dependencies**: the learning signal linking distant events never survives the trip back." },
        { type: "table",
          headers: ["Symptom", "Cause", "Standard fix"],
          rows: [
            ["Loss plateaus; long-range patterns never learned", "$\\gamma<1$ → gradients **vanish** over time", "**gated units (LSTM/GRU)**, ReLU-ish activations, careful init"],
            ["Loss suddenly $\\to$ NaN; wild spikes", "$\\gamma>1$ → gradients **explode**", "**gradient clipping** (rescale if $\\|g\\|$ exceeds a threshold)"],
            ["Training unstable near start", "poor recurrent-weight scaling", "orthogonal / identity init of $W_{hh}$"],
          ]
        },
        { type: "code", lang: "py", code: "# Gradient clipping: the standard defense against exploding gradients.\ndef clip_gradients(grads, max_norm=5.0):\n    total = np.sqrt(sum((g**2).sum() for g in grads))\n    if total > max_norm:\n        scale = max_norm / (total + 1e-6)\n        grads = [g * scale for g in grads]      # rescale, preserve direction\n    return grads\n\n# PyTorch one-liner (call between loss.backward() and optimizer.step()):\n# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)" },
        { type: "callout", variant: "tip", text: "**Clipping fixes explosion, not vanishing.** Rescaling a huge gradient keeps training from diverging, but it does nothing for a gradient that has already decayed to zero — you cannot amplify signal that is gone. Vanishing needs an **architectural** fix, which is exactly what the LSTM provides: a memory path where the repeated multiplication is replaced by repeated *addition*." },
      ]
    },
    {
      id: "lstm",
      title: "The LSTM cell, fully derived",
      level: "core",
      body: [
        { type: "p", text: "The **Long Short-Term Memory** cell (Hochreiter & Schmidhuber, 1997) is the architectural cure for vanishing gradients. Its central innovation is a second state — the **cell state** $c_t$ — that acts as a protected memory *highway* running straight through time, modified only by cheap additive and gated operations. Alongside it flows the familiar hidden state $h_t$, now the cell's controlled *output*." },
        { type: "p", text: "Three **gates** — vectors in $(0,1)$ produced by sigmoids — regulate the flow of information. Each gate reads the previous hidden state and the current input, and decides how much of something to let through (0 = block, 1 = pass):" },
        { type: "math", tex: String.raw`\begin{aligned}
f_t &= \sigma\!\left(W_f[h_{t-1},\, x_t] + b_f\right) & &\text{forget gate — what to erase from memory}\\
i_t &= \sigma\!\left(W_i[h_{t-1},\, x_t] + b_i\right) & &\text{input gate — what to write}\\
o_t &= \sigma\!\left(W_o[h_{t-1},\, x_t] + b_o\right) & &\text{output gate — what to read out}\\
\tilde c_t &= \tanh\!\left(W_c[h_{t-1},\, x_t] + b_c\right) & &\text{candidate — the new content to maybe write}
\end{aligned}` },
        { type: "p", text: "Here $[h_{t-1}, x_t]$ is the concatenation of previous hidden state and current input, so each $W$ is a single matrix acting on that stacked vector. The gates and candidate then combine to update the two states:" },
        { type: "math", tex: String.raw`\boxed{\,c_t = f_t \odot c_{t-1} \;+\; i_t \odot \tilde c_t\,}, \qquad h_t = o_t \odot \tanh(c_t)` },
        { type: "p", text: "Read the boxed cell-state update as a **memory edit**: the forget gate $f_t$ elementwise decides how much of the old memory $c_{t-1}$ to keep, and the input gate $i_t$ decides how much of the fresh candidate $\\tilde c_t$ to add. The new hidden state $h_t$ is a *gated view* of that memory, squashed through $\\tanh$ and filtered by the output gate." },
        { type: "heading", text: "Why additive state fixes the vanishing gradient" },
        { type: "p", text: "This is the crux — and it is worth deriving. In the vanilla RNN the state was **multiplied** by $W_{hh}$ every step, giving that fatal $\\gamma^{T-t}$ product. In the LSTM the cell state is **added** to, so the gradient flowing along the memory highway is governed by the derivative of $c_t$ with respect to $c_{t-1}$:" },
        { type: "math", tex: String.raw`\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t) \quad\Longrightarrow\quad \frac{\partial c_T}{\partial c_t} = \prod_{\tau=t+1}^{T}\operatorname{diag}(f_\tau)` },
        { type: "p", text: "The per-step factor is just the **forget gate** — no repeated multiplication by a weight matrix, no activation-slope shrinkage. If the network learns to keep a memory ($f_\\tau \\approx 1$), the product stays $\\approx 1$ across arbitrarily many steps, and the gradient reaches far into the past **undiminished**. This is the *constant error carousel* the LSTM authors designed: an unobstructed additive path that lets error flow backward for hundreds of steps." },
        { type: "callout", variant: "good", text: "**The one sentence to remember:** the LSTM replaces the vanilla RNN's *repeated matrix multiplication* of the state (which vanishes) with a *gated additive* update (whose backward path is a product of forget gates near 1). Addition preserves gradients; multiplication destroys them. Every gated architecture since — GRUs, highway networks, even the residual connections in Transformers — is a variation on this single trick." },
        { type: "callout", variant: "tip", text: "**Init the forget-gate bias to +1 (or higher).** A positive $b_f$ makes $f_t \\approx 1$ at the start of training, so the cell *defaults to remembering* and gradients flow freely from step one. This tiny trick (Jozefowicz et al., 2015) noticeably improves LSTM training and is baked into most modern implementations." },
        { type: "heading", text: "The whole cell in ~10 lines of NumPy" },
        { type: "code", lang: "py", code: "import numpy as np\ndef sigmoid(z): return 1.0 / (1.0 + np.exp(-z))\n\ndef lstm_step(x, h_prev, c_prev, P):\n    \"\"\"One LSTM timestep. P holds stacked weights for [f,i,o,c].\"\"\"\n    z = np.concatenate([h_prev, x])              # [h_{t-1}, x_t]\n    f = sigmoid(P['Wf'] @ z + P['bf'])           # forget gate\n    i = sigmoid(P['Wi'] @ z + P['bi'])           # input gate\n    o = sigmoid(P['Wo'] @ z + P['bo'])           # output gate\n    g = np.tanh(P['Wc'] @ z + P['bc'])           # candidate  (c-tilde)\n    c = f * c_prev + i * g                        # cell-state highway (additive!)\n    h = o * np.tanh(c)                            # gated readout\n    return h, c\n\nd, n = 16, 10\nrng = np.random.default_rng(0)\nP = {k: rng.normal(0, 0.1, (d, d + n)) for k in ['Wf','Wi','Wo','Wc']}\nP.update({b: np.zeros(d) for b in ['bi','bo','bc']}); P['bf'] = np.ones(d)  # forget bias = 1\n\nh, c = np.zeros(d), np.zeros(d)\nfor x in rng.normal(size=(5, n)):                # a length-5 sequence\n    h, c = lstm_step(x, h, c, P)\nprint(h.shape, c.shape)   # (16,) (16,) — hidden and cell state" },
      ]
    },

    {
      id: "gru",
      title: "The GRU, and LSTM vs GRU",
      level: "core",
      body: [
        { type: "p", text: "The **Gated Recurrent Unit** (Cho et al., 2014) is a streamlined gated cell. It drops the separate cell state — there is only $h_t$ — and uses **two** gates instead of three, so it has roughly 25% fewer parameters than an LSTM while keeping the additive-update trick that defeats vanishing gradients." },
        { type: "math", tex: String.raw`\begin{aligned}
r_t &= \sigma\!\left(W_r[h_{t-1},\, x_t] + b_r\right) & &\text{reset gate — how much past to use}\\
z_t &= \sigma\!\left(W_z[h_{t-1},\, x_t] + b_z\right) & &\text{update gate — keep vs. refresh}\\
\tilde h_t &= \tanh\!\left(W_h[\,r_t \odot h_{t-1},\, x_t\,] + b_h\right) & &\text{candidate state}\\
h_t &= (1 - z_t)\odot h_{t-1} \;+\; z_t \odot \tilde h_t & &\text{convex blend of old and new}
\end{aligned}` },
        { type: "p", text: "The final line is the heart of it: the **update gate** $z_t$ performs a convex interpolation between carrying the old state forward ($z_t\\to 0$: copy $h_{t-1}$ verbatim, giving a gradient-preserving skip) and replacing it with the candidate ($z_t\\to 1$). The **reset gate** $r_t$ lets the cell ignore the past when forming the candidate — useful when a new segment begins. Notice the LSTM's forget and input gates are here fused into the single coupled term $(1-z_t)$ and $z_t$." },
        { type: "table",
          headers: ["", "LSTM", "GRU"],
          rows: [
            ["States", "two: hidden $h_t$ + cell $c_t$", "one: hidden $h_t$"],
            ["Gates", "three (forget, input, output)", "two (reset, update)"],
            ["Parameters", "more (~33% larger)", "fewer, faster to train"],
            ["Long-range memory", "slightly stronger; separate protected cell", "competitive on most tasks"],
            ["When to prefer", "very long dependencies, large data", "smaller data, faster iteration"],
          ]
        },
        { type: "callout", variant: "note", text: "**Which should you use?** Empirically they are close — the large-scale comparison by Jozefowicz et al. (2015) found no consistent winner, and Chung et al. (2014) reported GRUs matching LSTMs on many sequence tasks with less compute. Rule of thumb: **start with a GRU** (cheaper), reach for an LSTM if you need the extra capacity for very long-range structure. In 2026 the honest answer is often \"use a Transformer\" — but the gated-RNN intuition still underpins how you reason about memory." },
      ]
    },

    {
      id: "architectures",
      title: "Sequence architectures: many-to-one, many-to-many, encoder–decoder",
      level: "core",
      body: [
        { type: "p", text: "The same recurrent cell can be wired into several **shapes** depending on how inputs and outputs line up in time. Karpathy's famous diagram names them by the input→output cardinality:" },
        { type: "table",
          headers: ["Shape", "Reads / emits", "Example task"],
          rows: [
            ["one-to-many", "single input → sequence out", "image captioning (image → words)"],
            ["many-to-one", "sequence in → single output", "sentiment classification, next-value forecast"],
            ["many-to-many (aligned)", "output at every input step", "part-of-speech tagging, per-frame labeling"],
            ["many-to-many (seq2seq)", "read all, then generate", "translation, summarization (lengths differ)"],
          ]
        },
        { type: "p", text: "**Many-to-one** simply ignores all outputs except the last: run the RNN over the whole sequence and feed the final hidden state $h_T$ — the summary of everything read — into a classifier head. **Aligned many-to-many** attaches an output head at every step, producing one label per input token, and sums the per-step losses (exactly the BPTT loss from earlier)." },
        { type: "p", text: "The most important shape is the **encoder–decoder** (seq2seq): one RNN (the *encoder*) reads the entire input into a context vector, and a second RNN (the *decoder*) generates the output sequence from it — the two can have different lengths, which is what translation demands." },
        { type: "math", tex: String.raw`c = h_T^{\text{enc}} \quad\longrightarrow\quad s_0 = c,\quad s_t = g(s_{t-1}, y_{t-1}, c),\quad p(y_t\mid\cdot) = \operatorname{softmax}(W_o s_t)` },
        { type: "callout", variant: "note", text: "That single context vector $c$ is the same bottleneck the vanilla RNN hidden state has — and cramming a 50-word sentence into one fixed vector is exactly where early neural translation broke. The fix, **attention**, and the full seq2seq story are derived in [Seq2Seq & the Birth of Attention](#nlp-seq2seq). This section is the machinery that makes that one possible." },
      ]
    },

    {
      id: "variants",
      title: "Bidirectional & stacked RNNs",
      level: "core",
      body: [
        { type: "heading", text: "Bidirectional RNNs — read the future too" },
        { type: "p", text: "A plain RNN at step $t$ has only seen tokens $1\\dots t$. But for many tasks the *right* context disambiguates the present — *\"I read a **book**\"* vs *\"please **book** a flight\"*. A **bidirectional RNN** runs two independent RNNs: one left-to-right and one right-to-left, then concatenates their hidden states so every position sees the whole sequence:" },
        { type: "math", tex: String.raw`\overrightarrow{h}_t = f(\overrightarrow{h}_{t-1}, x_t), \quad \overleftarrow{h}_t = f(\overleftarrow{h}_{t+1}, x_t), \quad h_t = [\,\overrightarrow{h}_t\, ;\, \overleftarrow{h}_t\,]` },
        { type: "callout", variant: "gotcha", text: "Bidirectionality needs the **entire** sequence up front, so it is fine for classification, tagging, and encoders — but **impossible for autoregressive generation** (you cannot read the future you are still generating) and for real-time streaming. Use it on the encoder side; never on a decoder." },
        { type: "heading", text: "Stacked (deep) RNNs — depth in features, not just time" },
        { type: "p", text: "Stacking RNN layers gives depth in the *representation* dimension: the hidden-state sequence output by layer 1 becomes the input sequence to layer 2, and so on. Lower layers capture local patterns, higher layers more abstract structure — the same hierarchy CNNs exploit spatially. Two or three layers is typical; beyond that, residual connections between layers become necessary to keep gradients healthy." },
        { type: "math", tex: String.raw`h_t^{(\ell)} = f^{(\ell)}\!\left(h_{t-1}^{(\ell)},\; h_t^{(\ell-1)}\right), \qquad h_t^{(0)} \equiv x_t` },
        { type: "callout", variant: "tip", text: "A production sequence model circa 2016 was typically a **stacked bidirectional LSTM** with dropout between layers — e.g. Google's Neural Machine Translation system used 8 stacked LSTM layers with residual connections. In PyTorch this is just `nn.LSTM(..., num_layers=8, bidirectional=True, dropout=0.3)`." },
      ]
    },

    {
      id: "from-scratch",
      title: "From scratch: a char-RNN in NumPy, then nn.LSTM in PyTorch",
      level: "core",
      body: [
        { type: "p", text: "Nothing cements the recurrence like Karpathy's **char-RNN**: a tiny character-level language model that reads text one character at a time and learns to predict the next one. Trained on Shakespeare it produces (nonsense but) Shakespeare-shaped text. Here is the core — forward over a sequence, cross-entropy loss, and BPTT with the shared-weight gradient sum — in pure NumPy." },
        { type: "code", lang: "py", code: "import numpy as np\nrng = np.random.default_rng(0)\n\n# --- toy data: predict next char in a repeating string ---\ndata = \"hello world \" * 50\nchars = sorted(set(data)); V = len(chars)\nstoi = {c: i for i, c in enumerate(chars)}\n\nd = 64                                   # hidden size\nWxh = rng.normal(0, 0.01, (d, V))\nWhh = rng.normal(0, 0.01, (d, d))\nWhy = rng.normal(0, 0.01, (V, d))\nbh, by = np.zeros(d), np.zeros(V)\n\ndef loss_and_grads(inputs, targets, hprev):\n    xs, hs, ps = {}, {-1: hprev}, {}\n    loss = 0\n    # ---- forward: unroll over the sequence ----\n    for t, ix in enumerate(inputs):\n        xs[t] = np.zeros(V); xs[t][ix] = 1            # one-hot input\n        hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t-1] + bh)\n        y = Why @ hs[t] + by\n        ps[t] = np.exp(y) / np.exp(y).sum()          # softmax\n        loss += -np.log(ps[t][targets[t]])           # cross-entropy\n    # ---- backward through time ----\n    dWxh, dWhh, dWhy = (np.zeros_like(W) for W in (Wxh, Whh, Why))\n    dbh, dby = np.zeros_like(bh), np.zeros_like(by)\n    dhnext = np.zeros(d)\n    for t in reversed(range(len(inputs))):\n        dy = ps[t].copy(); dy[targets[t]] -= 1        # softmax+CE: yhat - y\n        dWhy += np.outer(dy, hs[t]); dby += dy\n        dh = Why.T @ dy + dhnext                      # error: output + future\n        dz = (1 - hs[t]**2) * dh                       # tanh'(z)\n        dbh += dz\n        dWxh += np.outer(dz, xs[t])\n        dWhh += np.outer(dz, hs[t-1])                  # SUM over timesteps\n        dhnext = Whh.T @ dz                            # pass error back in time\n    for g in (dWxh, dWhh, dWhy, dbh, dby):\n        np.clip(g, -5, 5, out=g)                       # guard against explosion\n    return loss, (dWxh, dWhh, dWhy, dbh, dby), hs[len(inputs)-1]\n\n# ---- Adagrad training loop over sliding windows ----\nseq_len, lr = 25, 0.1\nmem = [np.zeros_like(W) for W in (Wxh, Whh, Why, bh, by)]\nhprev = np.zeros(d); p = 0\nfor step in range(2000):\n    if p + seq_len + 1 >= len(data): p, hprev = 0, np.zeros(d)\n    inputs  = [stoi[c] for c in data[p:p+seq_len]]\n    targets = [stoi[c] for c in data[p+1:p+seq_len+1]]\n    loss, grads, hprev = loss_and_grads(inputs, targets, hprev)\n    for param, g, m in zip((Wxh, Whh, Why, bh, by), grads, mem):\n        m += g*g; param -= lr * g / (np.sqrt(m) + 1e-8)   # Adagrad step\n    p += seq_len\n    if step % 500 == 0: print(f\"step {step:4d}  loss {loss/seq_len:.3f}\")" },
        { type: "callout", variant: "good", text: "Study the backward loop: `dhnext = Whh.T @ dz` is the error flowing **backward in time**, and `dWhh += np.outer(dz, hs[t-1])` is the shared-weight gradient being **summed** across steps — the two ideas from the BPTT derivation, in code. If you can write this loop from memory, you understand RNN training." },
        { type: "heading", text: "The framework way: nn.LSTM in PyTorch" },
        { type: "p", text: "In practice you never hand-roll BPTT — `nn.LSTM` implements a fused, cuDNN-accelerated cell and autograd handles the backward pass. A complete character LSTM is a few lines:" },
        { type: "code", lang: "py", code: "import torch, torch.nn as nn\n\nclass CharLSTM(nn.Module):\n    def __init__(self, vocab, d=128, layers=2):\n        super().__init__()\n        self.embed = nn.Embedding(vocab, d)\n        self.lstm  = nn.LSTM(d, d, num_layers=layers, batch_first=True, dropout=0.2)\n        self.head  = nn.Linear(d, vocab)\n    def forward(self, x, state=None):\n        e = self.embed(x)                 # (B, T) -> (B, T, d)\n        out, state = self.lstm(e, state)  # out: (B, T, d); state carries (h, c)\n        return self.head(out), state      # logits (B, T, vocab)\n\nmodel = CharLSTM(vocab=65)\nopt = torch.optim.Adam(model.parameters(), lr=3e-3)\nx = torch.randint(0, 65, (32, 100))       # batch of 32 sequences, length 100\ny = torch.randint(0, 65, (32, 100))\n\nlogits, _ = model(x)\nloss = nn.functional.cross_entropy(logits.reshape(-1, 65), y.reshape(-1))\nopt.zero_grad(); loss.backward()\nnn.utils.clip_grad_norm_(model.parameters(), 5.0)   # exploding-gradient guard\nopt.step()\nprint(loss.item())" },
        { type: "callout", variant: "tip", text: "`nn.LSTM` returns `(output, (h_n, c_n))`. Pass that `state` tuple back in on the next chunk to implement **truncated BPTT** across long documents — detach it first (`h.detach()`) to cut the graph, exactly as in the truncation sketch earlier. `batch_first=True` makes tensors `(batch, time, features)`, which is far less error-prone than the default `(time, batch, features)`." },
      ]
    },

    {
      id: "transformers",
      title: "Why Transformers replaced RNNs",
      level: "core",
      body: [
        { type: "p", text: "By 2017 gated RNNs were the state of the art in NLP — and then the Transformer made them nearly obsolete within two years. The reasons are structural, and every one traces back to the recurrence you just derived." },
        { type: "table",
          headers: ["RNN limitation", "Root cause", "How Transformers escape it"],
          rows: [
            ["No parallelism across time", "$h_t$ needs $h_{t-1}$ — the loop is inherently **sequential**", "self-attention processes all positions **at once**; fully parallel on GPUs"],
            ["Long-range paths are long", "info from step 1 to step 100 traverses 100 cells", "attention connects any two positions in **one** step ($O(1)$ path length)"],
            ["Residual vanishing/decay", "even gated memory attenuates over hundreds of steps", "direct all-to-all attention has no decay over distance"],
            ["Fixed-vector bottleneck (seq2seq)", "whole input squeezed into one $c$", "decoder attends over **all** encoder states"],
          ]
        },
        { type: "p", text: "The decisive one is **parallelism**. An RNN's forward pass is a `for` loop over time that cannot be parallelized — step $t$ literally waits for step $t-1$. On modern GPUs, which crave massive parallelism, this is a fatal inefficiency: you cannot make an RNN much faster by adding hardware. Self-attention, by contrast, is a batch of matrix multiplications over the whole sequence simultaneously — it saturates a GPU and scales to enormous datasets, which is precisely what made GPT-scale pretraining feasible." },
        { type: "callout", variant: "note", text: "**The trade-off Transformers took on.** Attention's cost is $O(T^2)$ in sequence length (every token attends to every other), versus the RNN's $O(T)$. For a long time that quadratic cost capped context windows — the entire field of efficient/long-context attention exists to claw it back. But the ability to parallelize training won decisively. The full derivation of self-attention and why it beat recurrence is in [Seq2Seq & Attention](#nlp-seq2seq) and the Transformers section." },
        { type: "callout", variant: "tip", text: "**RNNs are not dead.** They remain excellent where sequences are streamed, latency and memory are tight, or context is naturally bounded — on-device speech, time-series forecasting, control. And the pendulum swings: modern **state-space models** (S4, Mamba, 2023–2024) are a sophisticated return to the recurrent idea, achieving linear-time sequence modeling competitive with Transformers. Understanding RNNs is understanding the lineage those models revive." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "RNNs are best learned by watching one train. Implement at least the first two in pure NumPy before reaching for a framework — the debugging *is* the learning." },
        { type: "list", ordered: true, items: [
          "**Char-RNN from scratch.** Extend the NumPy char-RNN above: train it on a real text file (Shakespeare, code, song lyrics), add a `sample()` function that generates text one character at a time, and watch coherence emerge as loss drops. This is the canonical rite of passage — Karpathy's blog is your reference.",
          "**Measure vanishing gradients.** Build a vanilla RNN on a *copy task* (output a token seen $\\tau$ steps ago). Plot the gradient norm $\\|\\partial\\mathcal{L}/\\partial h_t\\|$ against the time gap for $\\tau = 5, 20, 50$ and watch it collapse. Swap in an LSTM and watch the curve flatten — you will have *reproduced* the result that justified the LSTM.",
          "**Build an LSTM cell by hand + gradient-check.** Implement the forget/input/output gate forward pass and its backward, and verify against `torch.nn.LSTMCell` (or numerical gradients). Confirm the $\\partial c_t/\\partial c_{t-1} = \\operatorname{diag}(f_t)$ path is what preserves the gradient.",
          "**Sentiment classifier (many-to-one).** Train a bidirectional GRU on IMDB reviews: embed tokens, run the biRNN, take the final concatenated state, and classify. Compare against a bag-of-words baseline to feel what order buys you.",
          "**Toy seq2seq without attention.** Build an encoder–decoder to reverse or sort short number sequences. Then plot accuracy vs. input length and *watch the bottleneck appear* — accuracy falls off a cliff as sequences grow. This motivates the attention section directly.",
          "**Truncated BPTT on a long document.** Train a word-level LSTM language model on WikiText-2, carrying and detaching hidden state across chunks. Report perplexity and experiment with the truncation window $k$.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The two blog posts below are, together, the best RNN/LSTM tutorial ever written — read them in order. The rest are the primary sources." },
        { type: "link", url: "https://karpathy.github.io/2015/05/21/rnn-effectiveness/", text: "Andrej Karpathy — \"The Unreasonable Effectiveness of Recurrent Neural Networks\" (the char-RNN post; intuition + code, start here)" },
        { type: "link", url: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/", text: "Christopher Olah — \"Understanding LSTM Networks\" (the clearest visual walkthrough of the gates ever made)" },
        { type: "link", url: "https://www.bioinf.jku.at/publications/older/2604.pdf", text: "Hochreiter & Schmidhuber (1997) — \"Long Short-Term Memory\", the original LSTM paper" },
        { type: "link", url: "https://arxiv.org/abs/1406.1078", text: "Cho et al. (2014) — \"Learning Phrase Representations using RNN Encoder–Decoder\" (introduces the GRU)" },
        { type: "link", url: "https://arxiv.org/abs/1211.5063", text: "Pascanu, Mikolov & Bengio (2013) — \"On the difficulty of training RNNs\" (the vanishing/exploding-gradient analysis + gradient clipping)" },
        { type: "link", url: "https://www.deeplearningbook.org/contents/rnn.html", text: "Goodfellow, Bengio & Courville — Deep Learning, Ch. 10 (sequence modeling, formal treatment)" },
        { type: "link", url: "https://arxiv.org/abs/2312.00752", text: "Gu & Dao (2023) — \"Mamba\" (the modern linear-time state-space return to recurrence)" },
      ]
    },
  ],

  packages: [
    { name: "numpy", why: "hand-roll the recurrence and BPTT — the only way to truly learn how state flows in time" },
    { name: "torch.nn.LSTM", why: "fused, cuDNN-accelerated LSTM; handles stacking, bidirectionality, and dropout" },
    { name: "torch.nn.GRU", why: "the lighter gated cell — same API, ~25% fewer parameters" },
    { name: "torch.nn.RNN", why: "vanilla $\\tanh$/ReLU RNN, for teaching and short sequences" },
    { name: "torch.nn.utils.clip_grad_norm_", why: "the standard exploding-gradient defense; call before optimizer.step()" },
    { name: "torch.nn.utils.rnn.pack_padded_sequence", why: "efficient batching of variable-length sequences without wasting compute on padding" },
    { name: "torch.nn.Embedding", why: "learnable token→vector lookup that feeds the recurrent layers" },
  ],

  gotchas: [
    "An RNN's memory is **only** the hidden state $h_t$ — one fixed vector must hold everything seen so far. This bottleneck is the seed of the seq2seq problem attention later solves.",
    "BPTT must **sum** the shared-weight gradients over all timesteps ($\\partial\\mathcal{L}/\\partial W_{hh}=\\sum_t \\delta_t h_{t-1}^\\top$). Forgetting the sum trains as if only the last step existed.",
    "Vanilla RNN gradients scale like $\\gamma^{T-t}$ (spectral norm of $W_{hh}$ to the time gap): they **vanish** for $\\gamma<1$ and **explode** for $\\gamma>1$ — the core reason long-range learning fails.",
    "**Gradient clipping fixes explosion, not vanishing.** You can rescale a huge gradient but not resurrect one that has decayed to zero — vanishing needs the LSTM's additive path.",
    "The LSTM's power is the **additive** cell update: $\\partial c_t/\\partial c_{t-1}=\\operatorname{diag}(f_t)$, no repeated matrix multiply. Initialize the forget-gate bias to $+1$ so it defaults to remembering.",
    "**Bidirectional RNNs cannot generate** — they need the whole sequence up front. Use them only on encoders/classifiers, never on an autoregressive decoder or a real-time stream.",
    "**Truncated BPTT** only learns dependencies up to the truncation window $k$; the forwarded hidden state carries information further but gets no gradient beyond $k$ steps.",
    "In PyTorch, remember to `detach()` the hidden state between truncated-BPTT chunks, or the graph (and memory) grows without bound. And prefer `batch_first=True` to keep tensor axes straight.",
  ],

  flashcards: [
    { q: "Write the vanilla RNN recurrence.", a: "$h_t = \\phi(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$ — combine previous state and current input, same weights every step. Output: $\\hat y_t = W_{hy}h_t + b_y$." },
    { q: "What is 'unrolling' an RNN?", a: "Writing the recurrence out across time as a $T$-layer feed-forward net with **tied** weights. It reveals that RNN depth is depth in time and that training is just backprop on that graph." },
    { q: "What makes BPTT different from ordinary backprop?", a: "The weights are shared across timesteps, so you **sum** each parameter's gradient over all $t$ before updating: $\\partial\\mathcal{L}/\\partial W_{hh}=\\sum_t\\delta_t h_{t-1}^\\top$." },
    { q: "What is truncated BPTT and why use it?", a: "Backpropagate only $k$ steps while carrying the hidden state forward across chunks. It makes long sequences tractable in memory, at the cost of not learning dependencies longer than $k$." },
    { q: "Why do vanilla RNNs vanish/explode over time?", a: "The gradient to step $t$ is a product of $\\approx W_{hh}$ Jacobians, magnitude $\\sim\\gamma^{T-t}$ (spectral norm $\\gamma$). $<1$ vanishes, $>1$ explodes — exponentially in the time gap." },
    { q: "Give the LSTM cell-state update and its two gates.", a: "$c_t = f_t\\odot c_{t-1} + i_t\\odot\\tilde c_t$: forget gate $f_t$ keeps old memory, input gate $i_t$ writes the candidate $\\tilde c_t$. Then $h_t = o_t\\odot\\tanh(c_t)$." },
    { q: "Why does the LSTM fix vanishing gradients?", a: "The cell update is **additive**, so $\\partial c_t/\\partial c_{t-1}=\\operatorname{diag}(f_t)$ — no repeated weight multiply. With $f_t\\approx1$ the gradient flows across hundreds of steps undiminished (the 'constant error carousel')." },
    { q: "How does a GRU differ from an LSTM?", a: "One state instead of two, two gates (reset, update) instead of three, ~25% fewer parameters. $h_t=(1-z_t)\\odot h_{t-1}+z_t\\odot\\tilde h_t$ is a convex blend of old and new." },
    { q: "What is a bidirectional RNN, and when can't you use one?", a: "Two RNNs (forward + backward) whose states are concatenated so each position sees the whole sequence. Unusable for generation or streaming — it needs the full input up front." },
    { q: "Give the four reasons Transformers replaced RNNs.", a: "No cross-time parallelism (sequential loop), long-range paths grow with distance, gated memory still decays, and the seq2seq fixed-vector bottleneck. Attention makes all path lengths $O(1)$ and fully parallel." },
    { q: "Why can't RNNs be parallelized across time?", a: "$h_t$ depends on $h_{t-1}$, so timesteps must be computed in order. GPUs crave parallelism, so this is a fatal efficiency limit — the main reason attention won for large-scale training." },
  ],

  cheatsheet: [
    { label: "RNN step", code: "h = np.tanh(Whh @ h + Wxh @ x + bh)" },
    { label: "Output head", code: "y = Why @ h + by" },
    { label: "BPTT weight grad", code: "dWhh += np.outer(dz, h_prev)  # sum over t" },
    { label: "Error back in time", code: "dhnext = Whh.T @ dz" },
    { label: "tanh derivative", code: "dz = (1 - h**2) * dh" },
    { label: "Clip gradients", code: "nn.utils.clip_grad_norm_(p, 5.0)" },
    { label: "LSTM cell update", code: "c = f*c_prev + i*g;  h = o*np.tanh(c)" },
    { label: "GRU update", code: "h = (1-z)*h_prev + z*h_tilde" },
    { label: "nn.LSTM", code: "nn.LSTM(d, d, num_layers=2, batch_first=True)" },
    { label: "Bidirectional", code: "nn.LSTM(..., bidirectional=True)" },
    { label: "Carry/detach state", code: "out, (h, c) = lstm(x, (h.detach(), c.detach()))" },
    { label: "Pack padded seq", code: "nn.utils.rnn.pack_padded_sequence(x, lengths)" },
    { label: "Forget-bias init", code: "P['bf'] = np.ones(d)  # default to remember" },
  ],
});

