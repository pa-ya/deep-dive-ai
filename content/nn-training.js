(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "nn-training",
  name: "Training Deep Networks",
  language: "Neural Networks",
  group: "Neural Networks",
  navLabel: "Training Deep Nets",
  tagline: "Everything between *\"I can compute gradients\"* and *\"my deep net actually trains well\"* — optimizers, schedules, init, normalization, regularization, and a battle-tested recipe.",
  color: "#FB923C",
  readMinutes: 54,
  sections: [
    {
      id: "loop",
      title: "The training loop, and how data flows through it",
      level: "core",
      body: [
        { type: "p", text: "In the Backpropagation track you learned to compute the gradient $\\nabla_\\theta \\mathcal{L}$ of a scalar loss with respect to every parameter, exactly, in one backward pass. That is the engine. **Training** is what you build around that engine so it actually converges to a network that generalizes. This whole track lives in the gap between *\"I can compute gradients\"* and *\"my deep net trains well.\"*" },
        { type: "p", text: "Every training run — from a 2-layer MLP to a trillion-parameter LLM — is the same five-step loop repeated millions of times:" },
        { type: "list", ordered: true, items: [
          "**Sample** a batch of examples from the training set.",
          "**Forward** — run the batch through the network to get predictions and the loss $\\mathcal{L}$.",
          "**Backward** — backprop to fill in $\\nabla_\\theta \\mathcal{L}$ for every parameter.",
          "**Step** — the *optimizer* updates the parameters using the gradient.",
          "**Repeat**, occasionally evaluating on a held-out validation set.",
        ]},
        { type: "callout", variant: "note", text: "Backprop gives you step 3. This track is about steps 1, 4, and the machinery (schedules, init, normalization, regularization) that decides whether those millions of steps land you in a good place or a bad one. The loss function is fixed; everything here shapes the *path* your parameters take through it." },
        { type: "heading", text: "Batch vs. mini-batch vs. stochastic — how many examples per step?" },
        { type: "p", text: "The gradient of the *full* training loss is an average over all $N$ examples. Computing it exactly every step (**full-batch gradient descent**) is accurate but slow — one update per pass over the entire dataset. The opposite extreme, **stochastic gradient descent (SGD)**, uses a single example per step: cheap and noisy. In practice everyone uses the middle ground, **mini-batch SGD**, with a batch of $m$ examples ($m$ typically 32–512):" },
        { type: "math", tex: String.raw`\nabla_\theta \mathcal{L} \;\approx\; \frac{1}{m}\sum_{i \in \mathcal{B}} \nabla_\theta \ell\!\left(f_\theta(x^{(i)}), y^{(i)}\right)` },
        { type: "p", text: "The mini-batch gradient is an **unbiased estimate** of the true gradient — its expectation over random batches is the full gradient — with variance that shrinks like $1/m$. That noise is not purely a nuisance: it helps escape sharp minima and saddle points, and mild noise is part of why SGD-trained nets generalize well." },
        { type: "table",
          headers: ["Regime", "Batch size", "Update noise", "Steps / epoch", "Verdict"],
          rows: [
            ["Full-batch GD", "$N$ (all)", "none", "$1$", "smooth but slow; stuck in sharp minima"],
            ["Mini-batch SGD", "$32{-}512$", "moderate", "$N/m$", "the universal default"],
            ["Stochastic (pure) SGD", "$1$", "very high", "$N$", "too noisy; poor hardware use"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Batch size is a hardware-and-statistics knob.** Bigger batches = less gradient noise, better GPU utilization, but each step costs more memory and (past a point) *worse* generalization (\"the large-batch generalization gap\"). Rule of thumb: pick the largest batch that fits in memory, then tune the learning rate to match — roughly, scale the LR *linearly* with batch size (the **linear scaling rule**)." },
        { type: "heading", text: "Epochs and the data pipeline" },
        { type: "p", text: "One **epoch** is one full pass over the training set — $N/m$ mini-batches. You train for many epochs (or, at LLM scale, a fixed number of *steps* / tokens rather than epochs). Each epoch you **shuffle** the data so batches differ, which decorrelates consecutive gradients." },
        { type: "code", lang: "py", code: "import torch\nfrom torch.utils.data import DataLoader, TensorDataset\n\n# A DataLoader IS the data pipeline: batching + shuffling + parallel loading.\nds = TensorDataset(X, y)                 # X: (N, d) features, y: (N,) targets\nloader = DataLoader(\n    ds,\n    batch_size=64,        # m — examples per step\n    shuffle=True,         # reshuffle every epoch (train only!)\n    num_workers=4,        # parallel CPU workers prefetch batches\n    pin_memory=True,      # faster host->GPU copies\n    drop_last=True,       # drop a ragged final batch (keeps shapes stable)\n)\n\nfor epoch in range(num_epochs):\n    for xb, yb in loader:            # one mini-batch per iteration\n        xb, yb = xb.to(device), yb.to(device)\n        ...                          # forward / backward / step" },
        { type: "callout", variant: "gotcha", text: "**Never shuffle the validation/test loader**, and never let data augmentation or dropout run during evaluation. Also: if data loading is your bottleneck (GPU sits idle), raise `num_workers` and use `pin_memory=True` — a starved GPU is the most common silent cause of slow training." },
      ]
    },

    {
      id: "optimizers",
      title: "Optimizers, derived from SGD to AdamW",
      level: "core",
      body: [
        { type: "p", text: "An **optimizer** is the rule that turns a gradient into a parameter update. Plain SGD takes a step directly downhill; every optimizer after it is a fix for a specific failure of that naive step. We derive the whole family — each one motivated by the problem the previous one leaves unsolved." },
        { type: "heading", text: "SGD — the baseline" },
        { type: "p", text: "The simplest rule: step against the gradient, scaled by the **learning rate** $\\eta$." },
        { type: "math", tex: String.raw`\theta_{t+1} = \theta_t - \eta\, g_t, \qquad g_t \equiv \nabla_\theta \mathcal{L}(\theta_t)` },
        { type: "p", text: "SGD's weakness is **ravines** — loss surfaces that curve much more steeply in some directions than others (a high condition number, from the Linear Algebra track). Plain SGD bounces across the steep walls of the ravine while crawling along its gentle floor, wasting most of its motion." },
        { type: "heading", text: "Momentum — accumulate velocity to power through ravines" },
        { type: "p", text: "The fix: don't step by the raw gradient, step by an **exponentially-decaying running average** of past gradients — a *velocity*. Oscillating components (which flip sign each step) cancel out in the average; the consistent downhill component (which points the same way each step) accumulates. Physically, it is a heavy ball rolling downhill with inertia." },
        { type: "math", tex: String.raw`\begin{aligned}
v_t &= \beta\, v_{t-1} + g_t & &\text{(velocity: running sum of gradients)}\\
\theta_{t+1} &= \theta_t - \eta\, v_t & &\text{(step along the velocity)}
\end{aligned}` },
        { type: "p", text: "Unrolling the recurrence shows *why* it accelerates: $v_t = \\sum_{k=0}^{t}\\beta^{k} g_{t-k}$, a weighted sum of all past gradients with geometrically-decaying weights. In a direction where the gradient is persistent, the sum of a geometric series gives an effective step of $\\frac{1}{1-\\beta}$ times the plain-SGD step. With the usual $\\beta = 0.9$ that is a **10× effective speed-up** along consistent directions, while oscillating directions stay damped." },
        { type: "callout", variant: "tip", text: "$\\beta$ (momentum coefficient, typically $0.9$) is the memory length: the average spans roughly $\\frac{1}{1-\\beta}$ past gradients. Higher $\\beta$ = smoother, more inertia, but slower to react to a change of direction. Momentum is *the* single most valuable addition to plain SGD." },
        { type: "heading", text: "Nesterov momentum — look before you leap" },
        { type: "p", text: "Nesterov Accelerated Gradient makes one refinement: since momentum will carry you to $\\theta_t - \\eta\\beta v_{t-1}$ regardless, evaluate the gradient at that *look-ahead* point rather than where you are now. This lets the update start correcting *before* it overshoots, giving a provably better convergence rate on convex problems." },
        { type: "math", tex: String.raw`v_t = \beta\, v_{t-1} + \nabla_\theta \mathcal{L}\!\left(\theta_t - \eta\beta\, v_{t-1}\right), \qquad \theta_{t+1} = \theta_t - \eta\, v_t` },
        { type: "callout", variant: "note", text: "The look-ahead is subtle in code, so frameworks use an algebraically-equivalent reformulation (a change of variables). In PyTorch it is just `nesterov=True` on `SGD`. On many vision benchmarks, SGD + Nesterov momentum still *beats* Adam for final accuracy — it is not obsolete." },
        { type: "heading", text: "AdaGrad — a per-parameter learning rate" },
        { type: "p", text: "So far one global $\\eta$ scales every parameter. But a rare feature might need big steps while a common one needs small ones. **AdaGrad** gives each parameter its own rate by accumulating the *sum of squared gradients* and dividing by its square root:" },
        { type: "math", tex: String.raw`G_t = G_{t-1} + g_t^{\odot 2}, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \varepsilon}\odot g_t` },
        { type: "p", text: "Parameters with large historical gradients get their effective rate shrunk; rarely-updated parameters keep a large rate. Great for sparse features — but $G_t$ only ever *grows*, so the effective learning rate decays monotonically to zero and training **stalls** on long runs. That flaw is exactly what RMSProp fixes." },
        { type: "heading", text: "RMSProp — forget the distant past" },
        { type: "p", text: "Replace AdaGrad's ever-growing *sum* with an **exponential moving average** of squared gradients. Now old gradients decay away, so the denominator stabilizes instead of exploding, and training never stalls:" },
        { type: "math", tex: String.raw`s_t = \rho\, s_{t-1} + (1-\rho)\, g_t^{\odot 2}, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \varepsilon}\odot g_t` },
        { type: "p", text: "$s_t$ tracks the *recent* magnitude (variance) of each parameter's gradient with decay $\\rho \\approx 0.9$. Dividing by $\\sqrt{s_t}$ normalizes every parameter's step to roughly unit scale — automatic per-parameter adaptive rates that don't decay to zero. RMSProp has momentum's smoothing in the *magnitude*; Adam adds it back in the *direction* too." },
        { type: "heading", text: "Adam — momentum + RMSProp, with bias correction" },
        { type: "p", text: "**Adam** (Adaptive Moment Estimation, Kingma & Ba 2014) is the default optimizer of deep learning. It keeps *two* exponential moving averages: the mean of gradients (like momentum, the **first moment** $m_t$) and the mean of squared gradients (like RMSProp, the **second moment** $v_t$)." },
        { type: "math", tex: String.raw`\begin{aligned}
m_t &= \beta_1\, m_{t-1} + (1-\beta_1)\, g_t & &\text{(1st moment — direction)}\\
v_t &= \beta_2\, v_{t-1} + (1-\beta_2)\, g_t^{\odot 2} & &\text{(2nd moment — magnitude)}
\end{aligned}` },
        { type: "p", text: "There is a problem: $m_0 = v_0 = 0$, so early in training both averages are biased toward zero. Here is the derivation of the correction. Assume the gradient is roughly stationary with true second moment $\\mathbb{E}[g^2]$. Unrolling the EMA for $v_t$:" },
        { type: "math", tex: String.raw`v_t = (1-\beta_2)\sum_{k=1}^{t}\beta_2^{\,t-k}\, g_k^{2} \;\;\Longrightarrow\;\; \mathbb{E}[v_t] = \mathbb{E}[g^2]\,(1-\beta_2)\sum_{k=1}^{t}\beta_2^{\,t-k} = \mathbb{E}[g^2]\left(1-\beta_2^{\,t}\right)` },
        { type: "p", text: "So $v_t$ underestimates the true second moment by exactly the factor $(1-\\beta_2^{t})$. Divide it out (and likewise for $m_t$) to get **bias-corrected** estimates, then take an RMSProp-style step with the corrected moments:" },
        { type: "math", tex: String.raw`\hat m_t = \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}}, \qquad \boxed{\;\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat v_t} + \varepsilon}\,\hat m_t\;}` },
        { type: "p", text: "As $t$ grows, $\\beta^{t}\\to 0$ and the correction factors approach 1 — it only matters for the first few hundred steps, but without it the initial updates are wildly too small (or, for $v$, too large a step). Defaults that work almost everywhere: $\\beta_1 = 0.9,\\; \\beta_2 = 0.999,\\; \\varepsilon = 10^{-8},\\; \\eta = 3\\times10^{-4}$." },
        { type: "callout", variant: "good", text: "**Why Adam is the default:** it needs little LR tuning (the per-parameter normalization makes it robust across scales), converges fast, and handles sparse and noisy gradients well. Karpathy's half-joke — `3e-4` is the best learning rate for Adam — is a genuinely good starting point." },
        { type: "heading", text: "AdamW — fix Adam's broken weight decay" },
        { type: "p", text: "Standard L2 regularization adds $\\lambda\\theta$ to the gradient. In Adam that term gets divided by $\\sqrt{\\hat v_t}$ like everything else, so parameters with large gradients get *less* regularization — coupling that nobody wants. **AdamW** (Loshchilov & Hutter 2017) *decouples* weight decay: apply the Adam step, then shrink the weights separately, outside the adaptive-rate machinery:" },
        { type: "math", tex: String.raw`\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} + \lambda\,\theta_t\right)` },
        { type: "p", text: "The $\\lambda\\theta_t$ term is applied uniformly, unscaled by $\\hat v_t$ — true weight decay again. This small change measurably improves generalization, and **AdamW is now the standard optimizer for training Transformers and LLMs.** When in doubt, reach for AdamW, not Adam." },
        { type: "heading", text: "The comparison, at a glance" },
        { type: "table",
          headers: ["Optimizer", "Core idea", "Adds over previous", "Typical use"],
          rows: [
            ["SGD", "step downhill", "—", "baseline; simple/convex problems"],
            ["+ Momentum", "accumulate velocity", "escapes ravines, 1/(1−β) speed-up", "CNNs, still SOTA for vision"],
            ["+ Nesterov", "look-ahead gradient", "corrects before overshoot", "when you want SGD's best"],
            ["AdaGrad", "per-param rate from $\\sum g^2$", "adaptive rates", "sparse features (NLP, old)"],
            ["RMSProp", "EMA of $g^2$", "no rate decay-to-zero", "RNNs, RL"],
            ["Adam", "momentum + RMSProp + bias-corr.", "robust, fast, low tuning", "the default everywhere"],
            ["AdamW", "Adam + decoupled decay", "correct regularization", "Transformers, LLMs (SOTA)"],
          ]
        },
        { type: "code", lang: "py", code: "import torch\n\n# All optimizers share the same interface: zero_grad -> backward -> step.\nsgd    = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)\nrms    = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.9)\nadam   = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))\nadamw  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)  # <- prefer this\n\n# The step, spelled out (Adam) — this is the derivation above, in code:\n#   m = b1*m + (1-b1)*g\n#   v = b2*v + (1-b2)*g*g\n#   mhat = m / (1 - b1**t);  vhat = v / (1 - b2**t)\n#   theta -= lr * mhat / (vhat.sqrt() + eps)" },
        { type: "callout", variant: "gotcha", text: "Put weight decay in the **optimizer** (`weight_decay=`), not by hand-adding $\\lambda\\|\\theta\\|^2$ to the loss — and use `AdamW`, not `Adam`, if you want that decay to behave correctly. Also, common practice is to exclude bias and normalization parameters from weight decay entirely." },
      ]
    },

    {
      id: "schedules",
      title: "Learning-rate schedules and finding the LR",
      level: "core",
      body: [
        { type: "p", text: "The learning rate is the **single most important hyperparameter**. Too high and training diverges (loss → NaN); too low and it crawls or gets stuck. And the *best* LR changes over the course of training: you want large steps early to make fast progress, and small steps late to settle precisely into a minimum. A **schedule** varies $\\eta$ over time to get both." },
        { type: "heading", text: "Step decay — the classic" },
        { type: "p", text: "Drop the LR by a constant factor at fixed milestones (e.g. ×0.1 every 30 epochs). Simple, effective, and for years the standard for training ImageNet CNNs. You can *see* it in the loss curve: each drop produces a sudden step down as the optimizer settles into a finer minimum." },
        { type: "math", tex: String.raw`\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor} \qquad (\text{e.g. } \gamma = 0.1,\; s = 30\text{ epochs})` },
        { type: "heading", text: "Cosine annealing — the modern default" },
        { type: "p", text: "Instead of discrete drops, decay the LR smoothly along a half-cosine from $\\eta_{\\max}$ down to $\\eta_{\\min}\\approx 0$ over the training run. The smooth decay avoids the shock of step drops and empirically reaches better minima; it is the standard schedule for training Transformers and modern vision models." },
        { type: "math", tex: String.raw`\eta_t = \eta_{\min} + \tfrac{1}{2}\left(\eta_{\max} - \eta_{\min}\right)\left(1 + \cos\!\frac{t\pi}{T}\right)` },
        { type: "heading", text: "Warmup — start slow, especially for Transformers" },
        { type: "p", text: "At step 0 the weights are random and the adaptive-optimizer moment estimates ($\\hat v_t$) are noisy and poorly-calibrated — a full-size LR step can knock the network into a bad region it never recovers from. **Warmup** ramps the LR *linearly from ~0 up to* $\\eta_{\\max}$ over the first few hundred to few thousand steps, then hands off to the main (usually cosine) decay:" },
        { type: "math", tex: String.raw`\eta_t = \begin{cases} \dfrac{t}{T_{\text{warm}}}\,\eta_{\max}, & t \le T_{\text{warm}} \\[1.2ex] \text{cosine decay to } \eta_{\min}, & t > T_{\text{warm}} \end{cases}` },
        { type: "callout", variant: "tip", text: "**Why warmup matters so much for Transformers:** early in training, Adam's second-moment estimate $\\hat v_t$ has enormous variance (few samples), so its adaptive step sizes are unreliable. Big early steps + LayerNorm + residual connections can blow up the scale of activations. Warmup keeps steps tiny until the moment estimates stabilize. Essentially every Transformer (BERT, GPT, ViT) uses **linear warmup + cosine decay** — it is not optional." },
        { type: "heading", text: "One-cycle — warmup up then all the way down" },
        { type: "p", text: "Leslie Smith's **one-cycle** policy runs a single up-then-down triangle over the *whole* training run: LR climbs from low to a high peak, then descends below where it started, while momentum moves inversely (high→low→high). The high-LR middle phase acts as a strong regularizer, often reaching good accuracy in far fewer epochs (\"**super-convergence**\")." },
        { type: "heading", text: "The LR range test — how to *find* a good LR" },
        { type: "p", text: "Don't guess the learning rate; measure it. In the **LR range test** (Smith), start from a tiny LR and *exponentially increase* it every batch while recording the loss. Plot loss vs. LR (log scale). The loss stays flat, then drops steeply, then explodes upward. Pick an LR near the steepest descent — roughly an order of magnitude below where the loss starts diverging." },
        { type: "code", lang: "py", code: "import torch, math\n\ndef lr_range_test(model, loader, opt, loss_fn, lr_start=1e-7, lr_end=1.0):\n    \"\"\"Exponentially ramp LR over one pass; record (lr, loss).\"\"\"\n    n = len(loader)\n    mult = (lr_end / lr_start) ** (1 / n)   # per-step multiplier\n    lr = lr_start\n    history = []\n    for xb, yb in loader:\n        for grp in opt.param_groups:\n            grp['lr'] = lr\n        opt.zero_grad()\n        loss = loss_fn(model(xb), yb)\n        loss.backward()\n        opt.step()\n        history.append((lr, loss.item()))\n        if loss.item() > 4 * history[0][1]:  # diverged — stop early\n            break\n        lr *= mult\n    return history  # plot loss vs lr (log-x); pick lr ~1 decade before the min\n\n# --- schedules in PyTorch ---\nsched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)\n# or warmup + cosine via SequentialLR:\nwarm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=500)\ncos  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - 500)\nsched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[500])\n# one-cycle:\n# sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=total_steps)" },
        { type: "callout", variant: "gotcha", text: "Call `scheduler.step()` **once per iteration** for step-based schedules (warmup, one-cycle, most Transformer training) but **once per epoch** for epoch-based ones (`CosineAnnealingLR` with `T_max` in epochs). Mixing these up silently ruins your schedule. Always call `optimizer.step()` *before* `scheduler.step()`." },
      ]
    },

    {
      id: "init",
      title: "Weight initialization: keeping signal alive",
      level: "core",
      body: [
        { type: "p", text: "Before the first gradient ever flows, you must choose the *initial* weights. This sounds trivial and is not: initialize too small and the signal shrinks to zero as it passes through layers (vanishing activations, then vanishing gradients); too large and it blows up. The goal of principled init is to keep the **variance of activations and gradients roughly constant across depth** — so a 50-layer net behaves like a 5-layer one at step 0." },
        { type: "callout", variant: "gotcha", text: "**Never initialize all weights to the same value (e.g. all zeros).** Every neuron in a layer would compute the identical output and receive the identical gradient, so they update identically and stay identical forever — the layer collapses to a single neuron. You *must* break symmetry with random init. (Biases, however, are fine to start at 0.)" },
        { type: "heading", text: "The variance-propagation argument" },
        { type: "p", text: "Consider one linear layer $z = \\sum_{j=1}^{n_{\\text{in}}} w_j x_j$ with inputs and weights drawn independently, zero-mean. The variance of a single output, assuming independence, is a sum of $n_{\\text{in}}$ independent terms:" },
        { type: "math", tex: String.raw`\mathrm{Var}(z) = \sum_{j=1}^{n_{\text{in}}} \mathrm{Var}(w_j)\,\mathrm{Var}(x_j) = n_{\text{in}}\,\mathrm{Var}(w)\,\mathrm{Var}(x)` },
        { type: "p", text: "For the output variance to *equal* the input variance — so signal neither grows nor shrinks layer to layer — we need the bracketed factor to be 1:" },
        { type: "math", tex: String.raw`n_{\text{in}}\,\mathrm{Var}(w) = 1 \quad\Longrightarrow\quad \mathrm{Var}(w) = \frac{1}{n_{\text{in}}}` },
        { type: "heading", text: "Xavier / Glorot init — for tanh and sigmoid" },
        { type: "p", text: "Glorot & Bengio (2010) noted you want variance preserved in *both* the forward pass ($n_{\\text{in}}$ terms) and the backward pass ($n_{\\text{out}}$ terms). Since you generally can't satisfy both, take the harmonic-mean compromise — the average of $\\frac{1}{n_{\\text{in}}}$ and $\\frac{1}{n_{\\text{out}}}$:" },
        { type: "math", tex: String.raw`\mathrm{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}} \qquad\Longrightarrow\qquad w \sim \mathcal{U}\!\left[-\sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}},\; \sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}}\right]` },
        { type: "p", text: "(The uniform bound $\\sqrt{6/(n_{\\text{in}}+n_{\\text{out}})}$ is chosen so a uniform distribution has exactly that variance.) Xavier assumes the activation is roughly *linear and symmetric* around 0 — true for tanh near the origin, which is why it was designed for tanh/sigmoid nets." },
        { type: "heading", text: "He / Kaiming init — for ReLU" },
        { type: "p", text: "ReLU zeroes out half its inputs, which halves the output variance. He et al. (2015) corrected the variance-preservation calculation for that factor of 2 — you need *twice* the weight variance to compensate. This is the standard init for ReLU networks (and their cousins):" },
        { type: "math", tex: String.raw`\mathrm{Var}(w) = \frac{2}{n_{\text{in}}} \qquad\Longrightarrow\qquad w \sim \mathcal{N}\!\left(0,\; \tfrac{2}{n_{\text{in}}}\right)` },
        { type: "callout", variant: "tip", text: "**The rule you actually use:** ReLU/GELU networks → **He (Kaiming)** init. tanh/sigmoid networks → **Xavier (Glorot)**. Frameworks do this for you (PyTorch `Linear`/`Conv` default to a Kaiming-uniform variant), but for deep custom nets you should set it explicitly — a wrong init is a common reason a deep net simply won't train." },
        { type: "code", lang: "py", code: "import torch, torch.nn as nn\n\n# From scratch — the two formulas, verified to preserve variance:\nn_in, n_out = 512, 512\nW_xavier = torch.randn(n_out, n_in) * (2.0 / (n_in + n_out)) ** 0.5\nW_he     = torch.randn(n_out, n_in) * (2.0 / n_in) ** 0.5\n\nx = torch.randn(4096, n_in)\nprint(x.var().item(), (x @ W_he.T).var().item())   # ~1.0  ~1.0  (He, no ReLU: ~2x)\n# with ReLU the He version keeps var ~1: (torch.relu(x @ W_he.T)).var() ~ 1.0\n\n# In practice — apply an init to a whole model:\ndef init_weights(m):\n    if isinstance(m, nn.Linear):\n        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # He\n        if m.bias is not None:\n            nn.init.zeros_(m.bias)\nmodel.apply(init_weights)" },
      ]
    },

    {
      id: "normalization",
      title: "Normalization: Batch Norm and Layer Norm",
      level: "core",
      body: [
        { type: "p", text: "Good init keeps activations well-scaled at step 0, but as weights update, the distribution of each layer's inputs drifts — the original Batch Norm paper called this **internal covariate shift**. Normalization layers fix the scale *continuously* during training by re-standardizing activations inside the network. They are among the highest-impact tricks in deep learning: they let you train deeper nets, faster, with higher learning rates." },
        { type: "heading", text: "Batch Norm — normalize each feature across the batch" },
        { type: "p", text: "Batch Normalization (Ioffe & Szegedy, 2015) standardizes each feature (channel) over the current mini-batch, then applies a learnable scale and shift. For a feature with batch values $x_1,\\dots,x_m$:" },
        { type: "math", tex: String.raw`\begin{aligned}
\mu_{\mathcal{B}} &= \frac{1}{m}\sum_{i=1}^{m} x_i, & \sigma_{\mathcal{B}}^2 &= \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_{\mathcal{B}})^2 \\[0.6ex]
\hat x_i &= \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \varepsilon}}, & y_i &= \gamma\,\hat x_i + \beta
\end{aligned}` },
        { type: "p", text: "The first line standardizes to zero-mean/unit-variance; the second line's learnable $\\gamma$ (scale) and $\\beta$ (shift) let the network *undo* the normalization if that is what's optimal — so BN never reduces representational power. Crucially, $\\mu_\\mathcal{B}$ and $\\sigma_\\mathcal{B}$ depend on the batch, so they are part of the computational graph and gradients flow through them." },
        { type: "p", text: "**Why it helps** (the modern understanding): beyond reducing covariate shift, BN *smooths the loss landscape* — it makes the gradients more predictable and stable, which is what actually permits larger learning rates and faster convergence. It also adds mild noise (batch statistics vary), giving a small regularizing effect." },
        { type: "heading", text: "The train/eval difference — the #1 BN footgun" },
        { type: "p", text: "At **test time** you often have a single example (no batch to compute statistics from), and predictions shouldn't depend on which other examples happen to share the batch. So BN keeps an **exponential moving average** of the mean and variance during training, and *uses those fixed running statistics* at eval time:" },
        { type: "math", tex: String.raw`\mu_{\text{run}} \leftarrow (1-\alpha)\,\mu_{\text{run}} + \alpha\,\mu_{\mathcal{B}}, \qquad \sigma^2_{\text{run}} \leftarrow (1-\alpha)\,\sigma^2_{\text{run}} + \alpha\,\sigma^2_{\mathcal{B}}` },
        { type: "callout", variant: "gotcha", text: "**You MUST call `model.train()` and `model.eval()`.** In `train()` mode BN uses batch stats and updates its running averages; in `eval()` mode it uses the frozen running stats (and Dropout turns off). Forgetting `model.eval()` before validation/inference is the most common deep-learning bug there is — your accuracy will be silently wrong. BN also behaves badly with very small batches (noisy statistics) — use GroupNorm or LayerNorm there." },
        { type: "heading", text: "Layer Norm — normalize across features, per example" },
        { type: "p", text: "Batch Norm's dependence on the batch is a problem for sequences and small batches. **Layer Norm** (Ba, Kiros & Hinton, 2016) normalizes over the *feature* dimension of each example *independently* — no batch statistics at all:" },
        { type: "math", tex: String.raw`\mu_i = \frac{1}{d}\sum_{j=1}^{d} x_{ij}, \quad \sigma_i^2 = \frac{1}{d}\sum_{j=1}^{d}(x_{ij}-\mu_i)^2, \quad y_{ij} = \gamma_j\,\frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2+\varepsilon}} + \beta_j` },
        { type: "p", text: "Because each example is normalized on its own, LayerNorm behaves *identically* at train and eval time (no running stats, no batch-size sensitivity), which makes it ideal for variable-length sequences. **This is why every Transformer uses LayerNorm, not BatchNorm** — attention operates per-token on batches of wildly varying sequence lengths where per-batch statistics make no sense. (Modern LLMs often use the simpler **RMSNorm** variant, which skips the mean-subtraction.)" },
        { type: "table",
          headers: ["", "Batch Norm", "Layer Norm"],
          rows: [
            ["Normalizes over", "the batch (per feature)", "the features (per example)"],
            ["Depends on batch?", "yes — needs running stats", "no — self-contained"],
            ["Train vs eval", "differ (batch vs running stats)", "identical"],
            ["Small-batch behavior", "poor (noisy stats)", "unaffected"],
            ["Home turf", "CNNs / vision", "Transformers / sequences"],
          ]
        },
        { type: "code", lang: "py", code: "import torch, torch.nn as nn\n\n# From scratch — BatchNorm forward (training mode), one feature dim:\ndef batchnorm_fwd(x, gamma, beta, eps=1e-5):\n    mu  = x.mean(0)                       # over the batch\n    var = x.var(0, unbiased=False)\n    xhat = (x - mu) / torch.sqrt(var + eps)\n    return gamma * xhat + beta            # learnable scale & shift\n\n# From scratch — LayerNorm forward, over the feature dim:\ndef layernorm_fwd(x, gamma, beta, eps=1e-5):\n    mu  = x.mean(-1, keepdim=True)        # over features, per example\n    var = x.var(-1, keepdim=True, unbiased=False)\n    return gamma * (x - mu) / torch.sqrt(var + eps) + beta\n\n# Framework:\nbn = nn.BatchNorm1d(256)   # CNN/MLP; tracks running mean/var\nln = nn.LayerNorm(256)     # Transformers; no running stats" },
      ]
    },

    {
      id: "regularization",
      title: "Regularization: making the net generalize",
      level: "core",
      body: [
        { type: "p", text: "A network with millions of parameters can memorize the training set outright — zero training loss, useless on new data. **Regularization** is any technique that trades a little training-fit for better *generalization*. Deep learning has its own toolbox beyond the classical L2 penalty." },
        { type: "heading", text: "L2 / weight decay — penalize large weights" },
        { type: "p", text: "Add a penalty on the squared norm of the weights. Smaller weights mean a smoother, less wiggly function that is less prone to overfit:" },
        { type: "math", tex: String.raw`\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2}\|\theta\|_2^2 \qquad\Longrightarrow\qquad \nabla_\theta \mathcal{L}_{\text{reg}} = \nabla_\theta \mathcal{L} + \lambda\theta` },
        { type: "p", text: "The extra $\\lambda\\theta$ pulls every weight toward zero each step — hence the name **weight decay**. As shown in the optimizers section, with adaptive optimizers you want the *decoupled* AdamW form, not the loss-penalty form. Typical $\\lambda$: $10^{-4}$ to $10^{-2}$." },
        { type: "heading", text: "Dropout — randomly delete neurons during training" },
        { type: "p", text: "Dropout (Srivastava et al., 2014) randomly zeroes each activation with probability $1-p$ (keep probability $p$) on every forward pass during training. The network can't rely on any single neuron, so it learns redundant, robust features — it's like training an ensemble of exponentially many sub-networks that share weights." },
        { type: "p", text: "The subtlety: at test time we want the *full* network with no dropout, but then each neuron receives more total input than it did during training (nothing was dropped), shifting the scale. We must correct for this. **Inverted dropout** does the correction at *train* time so test time is a clean no-op. Draw a mask $r_j \\sim \\text{Bernoulli}(p)$ and divide by $p$:" },
        { type: "math", tex: String.raw`\tilde a_j = \frac{r_j}{p}\,a_j, \qquad r_j \sim \text{Bernoulli}(p) \quad (\text{train}); \qquad \tilde a_j = a_j \quad (\text{test})` },
        { type: "p", text: "**The expectation argument** — why divide by $p$: we want the expected activation during training to equal the plain activation used at test. Under the mask, $\\mathbb{E}[r_j] = p$, so:" },
        { type: "math", tex: String.raw`\mathbb{E}\!\left[\tilde a_j\right] = \mathbb{E}\!\left[\frac{r_j}{p}\right]a_j = \frac{p}{p}\,a_j = a_j \;\checkmark` },
        { type: "p", text: "The scaling makes the training-time activation *unbiased* for the test-time activation, so no rescaling is needed at inference — test is just a forward pass with dropout disabled. Typical drop rate $1-p$: 0.1–0.5." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef dropout_forward(a, p_keep, training):\n    \"\"\"Inverted dropout: scale at train time so test time is identity.\"\"\"\n    if not training:\n        return a                                   # test: full network, no scaling\n    mask = (np.random.rand(*a.shape) < p_keep) / p_keep   # 0 or 1/p\n    return a * mask                                 # E[a*mask] == a  (unbiased)\n\n# Framework — nn.Dropout implements inverted dropout, gated by model.train/eval:\n# self.drop = nn.Dropout(p=0.5)   # p is the DROP prob in PyTorch (not keep!)" },
        { type: "callout", variant: "gotcha", text: "In PyTorch `nn.Dropout(p=0.5)` — `p` is the **drop** probability (so keep $=1-p$), the opposite of the $p$ in most textbooks. And like BatchNorm, dropout is controlled by `model.train()`/`model.eval()` — it must be **off at inference**. Don't stack heavy dropout *and* BatchNorm on the same layer; they can interact badly." },
        { type: "heading", text: "Early stopping — quit while you're ahead" },
        { type: "p", text: "Track validation loss each epoch; when it stops improving for `patience` epochs, stop training and restore the best checkpoint. The training loss would keep dropping, but that is memorization — the validation loss marks the moment the net starts overfitting. It's free regularization and you should always use it." },
        { type: "heading", text: "Data augmentation — manufacture more training data" },
        { type: "p", text: "Apply *label-preserving* random transformations to inputs so the model sees a different version each epoch, teaching invariance and dramatically enlarging the effective dataset. Images: random crop, flip, color jitter, rotation, Cutout, MixUp. Text: synonym swap, back-translation. Audio: time/pitch shift, SpecAugment. For vision it is often the single most effective regularizer." },
        { type: "heading", text: "Label smoothing — don't be overconfident" },
        { type: "p", text: "Hard one-hot targets push the network to make the correct logit $+\\infty$ and the rest $-\\infty$ — overconfident, poorly-calibrated, and prone to overfit. **Label smoothing** softens the target: give the true class $1-\\alpha$ and spread $\\alpha$ over the other $K-1$ classes:" },
        { type: "math", tex: String.raw`y_k^{\text{LS}} = (1-\alpha)\,y_k + \frac{\alpha}{K} \qquad (\text{typical } \alpha = 0.1)` },
        { type: "p", text: "This caps how confident the network is allowed to be, improving calibration and generalization. It is standard in training image classifiers and Transformers." },
        { type: "callout", variant: "note", text: "Normalization (BatchNorm/LayerNorm) also has a mild regularizing side-effect from its batch/statistical noise. In modern practice you rarely rely on one technique: a typical recipe stacks weight decay + dropout (or just norm) + data augmentation + label smoothing + early stopping. Add them one at a time and watch the validation curve." },
      ]
    },

    {
      id: "gradients",
      title: "Vanishing & exploding gradients, revisited",
      level: "core",
      body: [
        { type: "p", text: "In the Backpropagation track we derived *why* deep nets are hard to train: the backward recurrence multiplies by a weight matrix and an activation slope at every layer, so the gradient reaching layer 1 is a product of $L$ terms." },
        { type: "math", tex: String.raw`\delta^{[1]} \sim \left(\prod_{l=2}^{L} (W^{[l]})^{\top}\,\mathrm{diag}\!\left(\phi'(z^{[l-1]})\right)\right)\delta^{[L]}` },
        { type: "p", text: "Factors below 1 make the product decay to zero (**vanishing** — early layers barely learn); factors above 1 make it explode (**exploding** — NaN losses). We've already met most of the fixes; here are the two we haven't derived, both aimed squarely at this product." },
        { type: "heading", text: "Gradient clipping — cap the exploding side" },
        { type: "p", text: "If the total gradient norm exceeds a threshold, rescale the *whole* gradient vector down to that threshold — preserving its direction but capping its magnitude. This makes an occasional exploding step harmless, and is essential for RNNs and Transformers where sudden gradient spikes are common:" },
        { type: "math", tex: String.raw`\text{if } \|g\| > \tau: \quad g \leftarrow \tau\,\frac{g}{\|g\|}` },
        { type: "code", lang: "py", code: "# After backward(), before step(): clip the global grad norm.\nloss.backward()\ntorch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # tau = 1.0\noptimizer.step()" },
        { type: "heading", text: "Residual connections — give the gradient a shortcut home" },
        { type: "p", text: "The most important fix. A **residual (skip) connection** adds the block's input back to its output: instead of learning $H(x)$, the block learns the *residual* $F(x)$ and outputs $F(x) + x$." },
        { type: "math", tex: String.raw`y = F(x) + x \qquad\Longrightarrow\qquad \frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I` },
        { type: "p", text: "That $+I$ is everything. In the backward pass the gradient flows through the identity path *undiminished* — even if $\\partial F/\\partial x$ is tiny, the gradient still reaches earlier layers via the $I$ term. The product-of-many-terms becomes a *sum* that always includes a clean 1, so gradients can't vanish across the skip. This is why **ResNets** scale to 100+ layers and why every **Transformer** wraps attention and MLP blocks in residual connections." },
        { type: "callout", variant: "good", text: "Residual connections (He et al., 2015) are arguably the single most important architectural idea enabling *deep* deep learning. Combined with normalization, they turn the vanishing-gradient product into a well-behaved sum. Every modern deep architecture — ResNet, Transformer, U-Net, diffusion models — is built on them." },
        { type: "table",
          headers: ["Problem", "Fix", "Mechanism"],
          rows: [
            ["Vanishing", "ReLU/GELU activations", "slope = 1 for positive inputs (no saturation)"],
            ["Vanishing", "He/Xavier init", "keeps the per-layer factor near 1 at start"],
            ["Both", "Batch/Layer norm", "re-scales activations, smooths landscape"],
            ["Vanishing", "Residual connections", "identity path: $\\partial y/\\partial x = \\partial F/\\partial x + I$"],
            ["Exploding", "Gradient clipping", "caps the global gradient norm at $\\tau$"],
          ]
        },
      ]
    },

    {
      id: "diagnosing",
      title: "Diagnosing training: reading the loss curves",
      level: "core",
      body: [
        { type: "p", text: "You cannot train well what you cannot see. Always plot **training and validation loss (and metric) vs. epoch** on the same axes. The *gap* and *shape* of those two curves diagnose almost every problem, mapping directly onto the bias–variance framework from Classical ML." },
        { type: "heading", text: "The four signatures" },
        { type: "table",
          headers: ["Train loss", "Val loss", "Diagnosis", "What to do"],
          rows: [
            ["high", "high (≈ train)", "**underfitting** (high bias)", "bigger model, train longer, higher LR, less regularization"],
            ["low", "much higher", "**overfitting** (high variance)", "more data/aug, more regularization, smaller model, early stop"],
            ["low", "low (small gap)", "**good fit**", "ship it (or push capacity for more)"],
            ["not decreasing", "not decreasing", "**not learning**", "LR too low/high, bug, bad init, dead ReLUs"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**Loss goes to NaN** almost always means the LR is too high (or you forgot to clip gradients, or have a `log(0)`/`0/0` in the loss). Lower the LR by 10×, add gradient clipping, add $\\varepsilon$ inside logs. **Loss is flat from step 0** usually means the LR is too *low*, a broken data pipeline (labels shuffled off their inputs), or all-zero init. Run the LR range test." },
        { type: "heading", text: "The bias–variance dashboard" },
        { type: "p", text: "Turn the diagnosis into a decision procedure. Check these in order — each points to a *different* fix, so you never guess:" },
        { type: "list", ordered: true, items: [
          "**Is training loss low enough?** If not → high bias (underfitting). The model can't even fit the data it sees: increase capacity, train longer, raise the LR, or remove regularization. *Fixing variance here is wasted effort.*",
          "**Is the train↔val gap small?** If the gap is large → high variance (overfitting): add data/augmentation, add regularization (weight decay, dropout, label smoothing), or shrink the model.",
          "**Are both good?** You have a well-fit model. If you want more, *now* increase capacity — a bigger model on a well-regularized pipeline is the path to better performance.",
        ]},
        { type: "callout", variant: "tip", text: "**Sanity check before any real run: overfit a single batch.** Take one mini-batch and train until the loss is ~0. If you *can't* drive a handful of examples to near-zero loss, you have a bug (wrong loss, detached graph, broken labels) — no amount of hyperparameter tuning will help. This 60-second test, from Karpathy's recipe, catches most implementation bugs instantly." },
        { type: "code", lang: "py", code: "# The single most valuable debugging test: overfit one batch to ~0 loss.\nxb, yb = next(iter(train_loader))\nfor step in range(200):\n    opt.zero_grad()\n    loss = loss_fn(model(xb), yb)\n    loss.backward()\n    opt.step()\n    if step % 25 == 0:\n        print(f\"step {step:3d}  loss {loss.item():.5f}\")\n# loss should march to ~0. If it plateaus high -> there is a bug, not a tuning issue." },
      ]
    },

    {
      id: "recipe",
      title: "A recipe for training a net that works",
      level: "core",
      body: [
        { type: "p", text: "Deep-net training rewards *discipline*, not cleverness. This checklist distills the whole track (and Karpathy's \"A Recipe for Training Neural Networks\") into an order of operations. Follow it top to bottom; change **one thing at a time**." },
        { type: "list", ordered: true, items: [
          "**Know your data first.** Look at examples by hand, check label balance, spot corrupt samples and duplicates. Most \"model\" bugs are data bugs.",
          "**Build a dumb baseline.** Get an end-to-end skeleton training (fixed seeds, tiny model). Verify input shapes, loss at init (for $K$-class cross-entropy it should be $\\approx \\ln K$), and that `model.eval()` is wired in.",
          "**Overfit one batch to ~0.** The non-negotiable sanity check. If it fails, stop and fix the bug.",
          "**Get the LR right.** Run an LR range test. Start with **AdamW at 3e-4**; add **linear warmup + cosine decay**.",
          "**Set init & normalization.** He init for ReLU/GELU; add BatchNorm (CNN) or LayerNorm (Transformer); residual connections for depth.",
          "**Train and watch the curves.** Plot train/val loss. Underfitting? Add capacity / train longer. Not moving? Recheck LR and pipeline.",
          "**Once it overfits, regularize.** Add data augmentation → weight decay → dropout → label smoothing → early stopping, one at a time, watching the val gap close.",
          "**Then scale capacity.** With a well-regularized pipeline, a bigger model + more data is the reliable path to better numbers.",
          "**Tune last, and cheaply.** Hyperparameter search (LR, weight decay, dropout) matters far less than the steps above. Prefer random search over grid.",
        ]},
        { type: "callout", variant: "note", text: "**Order matters.** Fixing overfitting on a model that underfits is wasted work; regularizing before you can overfit hides bugs. Establish that the model *can* fit (drive train loss down), *then* close the generalization gap. Bias first, variance second." },
        { type: "callout", variant: "warn", text: "**Reproducibility:** set seeds (`torch.manual_seed`, `numpy`, Python `random`), log every hyperparameter, and checkpoint regularly. A result you can't reproduce is a result you can't build on — and non-determinism will mask real regressions." },
      ]
    },

    {
      id: "pytorch-loop",
      title: "The complete PyTorch training loop",
      level: "core",
      body: [
        { type: "p", text: "Everything in this track, assembled into one real, runnable training loop: AdamW + weight decay, warmup→cosine schedule, BatchNorm, Dropout, gradient clipping, and early stopping with best-checkpoint restore. This is the template you can adapt to essentially any supervised deep-learning task." },
        { type: "code", lang: "py", code: "import copy, torch, torch.nn as nn\n\ndevice = 'cuda' if torch.cuda.is_available() else 'cpu'\n\nclass MLP(nn.Module):\n    def __init__(self, d_in, d_hidden, d_out, p_drop=0.3):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(d_in, d_hidden),\n            nn.BatchNorm1d(d_hidden),      # normalize -> higher LR, faster\n            nn.ReLU(),\n            nn.Dropout(p_drop),            # regularize (train-only, auto-gated)\n            nn.Linear(d_hidden, d_hidden),\n            nn.BatchNorm1d(d_hidden),\n            nn.ReLU(),\n            nn.Dropout(p_drop),\n            nn.Linear(d_hidden, d_out),    # logits (no softmax; CE does it)\n        )\n        self.apply(self._init)            # He init for the ReLU stack\n\n    @staticmethod\n    def _init(m):\n        if isinstance(m, nn.Linear):\n            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')\n            nn.init.zeros_(m.bias)\n\n    def forward(self, x):\n        return self.net(x)\n\nmodel   = MLP(d_in=784, d_hidden=256, d_out=10).to(device)\nloss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)         # softmax+CE, smoothed\nopt     = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)\n\ntotal_steps = num_epochs * len(train_loader)\nwarm  = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=500)\ncos   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - 500)\nsched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[500])\n\nbest_val, best_state, patience, wait = float('inf'), None, 5, 0\n\nfor epoch in range(num_epochs):\n    # ---------- train ----------\n    model.train()                                  # BN uses batch stats, dropout ON\n    for xb, yb in train_loader:\n        xb, yb = xb.to(device), yb.to(device)\n        opt.zero_grad()\n        loss = loss_fn(model(xb), yb)\n        loss.backward()\n        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # tame exploding grads\n        opt.step()\n        sched.step()                               # per-STEP schedule (warmup/cosine)\n\n    # ---------- validate ----------\n    model.eval()                                   # BN uses running stats, dropout OFF\n    val_loss, correct, n = 0.0, 0, 0\n    with torch.no_grad():\n        for xb, yb in val_loader:\n            xb, yb = xb.to(device), yb.to(device)\n            logits = model(xb)\n            val_loss += loss_fn(logits, yb).item() * len(xb)\n            correct  += (logits.argmax(1) == yb).sum().item()\n            n += len(xb)\n    val_loss /= n\n    print(f\"epoch {epoch:2d}  val_loss {val_loss:.4f}  acc {correct/n:.3f}\")\n\n    # ---------- early stopping ----------\n    if val_loss < best_val - 1e-4:\n        best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0\n    else:\n        wait += 1\n        if wait >= patience:\n            print(f\"early stop at epoch {epoch}\")\n            break\n\nmodel.load_state_dict(best_state)   # restore the best checkpoint\n" },
        { type: "callout", variant: "good", text: "Read this loop against the checklist: init and normalization (model), AdamW + weight decay (optimizer), warmup→cosine (scheduler, stepped per-iteration), gradient clipping (before `step`), the `train()`/`eval()` toggle (BN + dropout correctness), and early stopping with best-state restore. If you internalize one code block from this track, make it this one." },
        { type: "callout", variant: "gotcha", text: "The three lines beginners forget, each silently wrong: (1) `opt.zero_grad()` — PyTorch *accumulates* gradients, so skipping this sums gradients across steps. (2) `model.eval()` before validation — otherwise BN/dropout corrupt your metrics. (3) `with torch.no_grad()` during eval — otherwise you build a useless graph and may OOM." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Training intuition is earned by *running experiments and reading curves*, not by reading about them. Do at least the first two; instrument everything and plot everything." },
        { type: "list", ordered: true, items: [
          "**Optimizer bake-off.** On the same small net (MNIST or Fashion-MNIST), train with SGD, SGD+momentum, RMSProp, Adam, and AdamW. Plot all five loss curves on one axis. Then implement Adam *from scratch* (the four update lines + bias correction) and confirm it matches `torch.optim.Adam` step-for-step.",
          "**Schedule shoot-out.** Fix the optimizer and compare constant LR, step decay, cosine, and warmup+cosine. Then implement the **LR range test**, plot loss-vs-LR, pick the LR it suggests, and verify it beats your guess.",
          "**Init & the vanishing gradient, measured.** Build a 20-layer MLP. Log the per-layer gradient norm at init under (a) tiny-random, (b) Xavier, (c) He init, with sigmoid vs. ReLU. Reproduce the classic plot where bad init vanishes and He + ReLU stays flat.",
          "**Regularization ladder.** Take a model that clearly overfits (big net, small data). Add — one at a time — weight decay, dropout, data augmentation, label smoothing, early stopping, and record how each moves the train/val gap. Build intuition for which does the most.",
          "**BatchNorm train/eval trap.** Train a net with BatchNorm, then evaluate *once with* `model.eval()` and *once without*. Measure the accuracy difference. Internalize, permanently, why the mode toggle matters.",
          "**The full recipe on a real dataset.** Take CIFAR-10, follow the checklist end-to-end (overfit one batch → LR test → train → regularize → scale), and push past 90% test accuracy. Keep a lab notebook of what each change did.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "This track is self-contained, but these are the canonical sources — read Karpathy's recipe first, it is the practitioner's bible for this material:" },
        { type: "link", url: "https://karpathy.github.io/2019/04/25/recipe/", text: "Andrej Karpathy — \"A Recipe for Training Neural Networks\" (the single best practical guide; the checklist in this track distills it)" },
        { type: "link", url: "https://cs231n.github.io/neural-networks-3/", text: "Stanford CS231n — Neural Networks Part 3: learning & evaluation (init, optimizers, schedules, babysitting training)" },
        { type: "link", url: "https://arxiv.org/abs/1412.6980", text: "Kingma & Ba (2014) — Adam: A Method for Stochastic Optimization (the optimizer, with the bias-correction derivation)" },
        { type: "link", url: "https://arxiv.org/abs/1711.05101", text: "Loshchilov & Hutter (2017) — Decoupled Weight Decay Regularization (AdamW; why Adam's L2 was broken)" },
        { type: "link", url: "https://www.deeplearningbook.org/contents/optimization.html", text: "Goodfellow, Bengio & Courville — Deep Learning, Ch. 8 (optimization for training deep models — the formal treatment)" },
        { type: "link", url: "https://www.deeplearningbook.org/contents/regularization.html", text: "Deep Learning, Ch. 7 (regularization — weight decay, dropout, early stopping, augmentation, derived)" },
        { type: "link", url: "https://arxiv.org/abs/1502.03167", text: "Ioffe & Szegedy (2015) — Batch Normalization (the original paper)" },
        { type: "link", url: "https://arxiv.org/abs/1506.01186", text: "Leslie Smith (2015) — Cyclical Learning Rates & the LR range test (and the one-cycle policy)" },
      ]
    },
  ],

  packages: [
    { name: "torch.optim", why: "SGD, RMSprop, Adam, AdamW — the whole optimizer family behind one `.step()` API" },
    { name: "torch.optim.lr_scheduler", why: "step / cosine / warmup / one-cycle schedules (`CosineAnnealingLR`, `OneCycleLR`, `SequentialLR`)" },
    { name: "torch.nn.init", why: "Kaiming (He) & Xavier (Glorot) weight initialization" },
    { name: "nn.BatchNorm1d / nn.LayerNorm", why: "normalization layers — faster training, higher LRs, deeper nets" },
    { name: "nn.Dropout", why: "inverted-dropout regularization, auto-gated by train/eval mode" },
    { name: "nn.utils.clip_grad_norm_", why: "gradient clipping to tame exploding gradients (essential for RNNs/Transformers)" },
    { name: "torch.utils.data.DataLoader", why: "the data pipeline: batching, shuffling, parallel prefetch" },
  ],

  gotchas: [
    "Call `model.train()` before training and `model.eval()` before validation/inference — otherwise **BatchNorm** and **Dropout** behave wrongly and your metrics are silently corrupted.",
    "`optimizer.zero_grad()` every step: PyTorch **accumulates** gradients, so forgetting it sums gradients across iterations.",
    "Use **AdamW** (decoupled weight decay), not Adam-with-L2-in-the-loss — in Adam the $\\lambda\\theta$ term wrongly gets divided by $\\sqrt{\\hat v_t}$.",
    "Loss → **NaN** ⇒ LR too high (or missing gradient clip, or `log(0)` in the loss). Loss **flat from step 0** ⇒ LR too low, all-zero init, or a broken data pipeline.",
    "**Never initialize all weights equal** (e.g. zeros): every neuron in a layer stays identical forever. Random init breaks the symmetry; biases at 0 are fine.",
    "Step-based schedules (warmup, one-cycle) call `scheduler.step()` **per iteration**; epoch-based (`CosineAnnealingLR` with `T_max` in epochs) call it **per epoch**. Always after `optimizer.step()`.",
    "In `nn.Dropout(p)`, `p` is the **drop** probability (keep $=1-p$) — the opposite of most textbooks' $p$.",
    "Fix bias before variance: regularizing a model that still **underfits** is wasted effort. First drive train loss down, then close the train↔val gap.",
  ],

  flashcards: [
    { q: "Why use mini-batches instead of the full-batch gradient?", a: "The mini-batch gradient is an unbiased estimate of the true gradient (variance $\\sim 1/m$); it's far cheaper per step, uses hardware well, and its noise helps escape saddles/sharp minima." },
    { q: "Derive the momentum update and its speed-up.", a: "$v_t = \\beta v_{t-1} + g_t$, $\\theta \\mathrel{-}= \\eta v_t$. Unrolling gives a geometric sum, so a persistent direction gets an effective step $\\frac{1}{1-\\beta}$ larger — ~10× at $\\beta=0.9$ — while oscillations cancel." },
    { q: "Write the full Adam update, including bias correction.", a: "$m_t=\\beta_1 m_{t-1}+(1-\\beta_1)g_t$, $v_t=\\beta_2 v_{t-1}+(1-\\beta_2)g_t^2$, $\\hat m=m_t/(1-\\beta_1^t)$, $\\hat v=v_t/(1-\\beta_2^t)$, $\\theta\\mathrel{-}=\\eta\\,\\hat m/(\\sqrt{\\hat v}+\\varepsilon)$." },
    { q: "Why does Adam need bias correction?", a: "$m_0=v_0=0$, so early EMAs underestimate the true moments by $(1-\\beta^t)$. Dividing by $(1-\\beta^t)$ removes the bias; the correction fades as $t$ grows." },
    { q: "What does AdamW fix versus Adam?", a: "It decouples weight decay: apply $-\\eta\\lambda\\theta$ separately rather than adding $\\lambda\\theta$ to the gradient, so decay isn't scaled by $\\sqrt{\\hat v}$. Standard for Transformers/LLMs." },
    { q: "Why do Transformers need learning-rate warmup?", a: "Early on, Adam's second-moment estimate $\\hat v$ is high-variance, so adaptive step sizes are unreliable; big early steps can blow up activations. Warmup keeps steps tiny until the moments stabilize." },
    { q: "State He vs. Xavier init and when to use each.", a: "He: $\\mathrm{Var}(w)=2/n_{in}$ for ReLU (corrects for ReLU zeroing half the variance). Xavier: $\\mathrm{Var}(w)=2/(n_{in}+n_{out})$ for tanh/sigmoid. Goal: preserve activation/gradient variance across depth." },
    { q: "Give the BatchNorm forward pass and its train/eval difference.", a: "$\\hat x=(x-\\mu_\\mathcal{B})/\\sqrt{\\sigma_\\mathcal{B}^2+\\varepsilon}$, $y=\\gamma\\hat x+\\beta$. Train: batch stats (+ update running EMA). Eval: use frozen running mean/var. Must toggle `train()`/`eval()`." },
    { q: "Why LayerNorm in Transformers instead of BatchNorm?", a: "LayerNorm normalizes over features per example — no batch dependence, identical at train/eval, unaffected by batch size or variable sequence length. BatchNorm's per-batch stats make no sense for sequences." },
    { q: "Derive why inverted dropout scales by $1/p$.", a: "With mask $r\\sim\\text{Bernoulli}(p)$, $\\mathbb{E}[(r/p)a]=(p/p)a=a$, so the training activation is unbiased for the test activation — no rescaling needed at inference (dropout just turns off)." },
    { q: "How do residual connections fight vanishing gradients?", a: "$y=F(x)+x$ gives $\\partial y/\\partial x=\\partial F/\\partial x+I$. The $+I$ identity path lets gradient flow undiminished to earlier layers, turning a product of small factors into a sum containing a clean 1." },
    { q: "How do you tell underfitting from overfitting?", a: "Underfitting: train loss high (val ≈ train) → add capacity / train longer / less regularization. Overfitting: train low but val much higher → more data/aug, more regularization, or smaller model." },
  ],

  cheatsheet: [
    { label: "SGD + momentum", code: "SGD(params, lr=0.1, momentum=0.9, nesterov=True)" },
    { label: "AdamW (default)", code: "AdamW(params, lr=3e-4, weight_decay=1e-2)" },
    { label: "Adam step (by hand)", code: "m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g; mh=m/(1-b1**t); vh=v/(1-b2**t); p-=lr*mh/(vh.sqrt()+eps)" },
    { label: "Cosine schedule", code: "lr_scheduler.CosineAnnealingLR(opt, T_max=T)" },
    { label: "Warmup + cosine", code: "SequentialLR(opt, [LinearLR(...), CosineAnnealingLR(...)], milestones=[warm])" },
    { label: "He init", code: "nn.init.kaiming_normal_(w, nonlinearity='relu')" },
    { label: "Xavier init", code: "nn.init.xavier_uniform_(w)" },
    { label: "Batch/Layer norm", code: "nn.BatchNorm1d(d)  /  nn.LayerNorm(d)" },
    { label: "Dropout (p = drop)", code: "nn.Dropout(p=0.5)" },
    { label: "Label smoothing", code: "nn.CrossEntropyLoss(label_smoothing=0.1)" },
    { label: "Gradient clip", code: "nn.utils.clip_grad_norm_(params, 1.0)" },
    { label: "Train / eval mode", code: "model.train()  /  model.eval()" },
    { label: "The step order", code: "zero_grad(); loss.backward(); clip; opt.step(); sched.step()" },
    { label: "Overfit-one-batch test", code: "train on next(iter(loader)) until loss ~ 0" },
  ],
});
