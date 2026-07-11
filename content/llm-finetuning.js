(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "llm-finetuning",
  name: "Fine-Tuning, LoRA & RLHF",
  language: "LLMs",
  group: "Large Language Models",
  navLabel: "Fine-Tuning & Alignment",
  tagline: "How to specialize and *align* a base model — SFT, **LoRA/QLoRA**, the reward model & PPO, and the **DPO** shortcut that skips it, derived from scratch and built in HuggingFace.",
  color: "#9D174D",
  readMinutes: 54,
  sections: [
    {
      id: "pipeline",
      title: "The alignment pipeline: from raw predictor to helpful assistant",
      level: "core",
      body: [
        { type: "p", text: "A freshly pretrained LLM is a **magnificent autocomplete and nothing more**. It has read a large fraction of the internet and can continue any text plausibly, but it has no idea it is supposed to *answer your question* rather than, say, generate five more questions like it. Turning that raw next-token predictor into ChatGPT is a second training phase called **post-training** or **alignment**, and it is where fine-tuning lives." },
        { type: "p", text: "The modern recipe has three stages. Each one changes the model less than the last, but changes *what it is for* more:" },
        { type: "table",
          headers: ["Stage", "What it does", "Data", "Scale"],
          rows: [
            ["**Pretraining**", "learn language & world knowledge by predicting the next token", "trillions of tokens of raw text", "$$$$ (months, thousands of GPUs)"],
            ["**Supervised fine-tuning (SFT)**", "teach the *format* of following instructions", "10k–1M curated (prompt, response) pairs", "$$ (hours–days, few GPUs)"],
            ["**Preference optimization** (RLHF / DPO)", "align *behavior* with human taste — helpful, harmless, honest", "human preference pairs (A vs B)", "$$ (hours–days)"],
          ]
        },
        { type: "p", text: "That progression — **pretraining → SFT → preference optimization** — is the backbone of this entire topic. Pretraining is covered in the Transformers track; here we own the last two stages, plus the parameter-efficient tricks (LoRA, QLoRA) that make them affordable on a single GPU." },
        { type: "heading", text: "Base vs instruct vs chat models" },
        { type: "p", text: "When you download a model you will see two flavors. Know the difference before you waste a weekend:" },
        { type: "table",
          headers: ["Model", "What it is", "How it behaves"],
          rows: [
            ["**Base** (e.g. `Llama-3-8B`)", "pretrained only", "raw completion; will happily continue your prompt instead of answering it"],
            ["**Instruct / Chat** (e.g. `Llama-3-8B-Instruct`)", "base + SFT + preference optimization", "follows instructions, uses a chat template, refuses unsafe requests"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Which do you fine-tune?** For a new *task/style*, start from the **instruct** model — you inherit its instruction-following and only teach the delta. For deep domain adaptation where you'll do your own full alignment (or want no built-in refusals), start from the **base** model. Fine-tuning an instruct model on raw completions is a classic way to accidentally destroy its chat abilities." },
        { type: "heading", text: "The real question: fine-tune, prompt, or RAG?" },
        { type: "p", text: "Before you fine-tune anything, be honest about whether you should. Fine-tuning is the *most* expensive and *least* reversible option, and beginners reach for it far too early. Use this decision order:" },
        { type: "table",
          headers: ["You want to…", "Reach for", "Why not fine-tune"],
          rows: [
            ["change *behavior/format/persona*", "**fine-tuning (SFT/LoRA)**", "this is exactly what FT is for"],
            ["inject *facts / private documents*", "**RAG** (retrieval)", "FT bakes facts in badly & staleley; retrieval stays fresh and citable"],
            ["tweak tone or add a few examples", "**prompting / few-shot**", "free, instant, no training loop"],
            ["hit a *latency/cost* target", "**fine-tune a smaller model** (distillation)", "FT lets a 7B match a 70B on *your* task"],
          ]
        },
        { type: "callout", variant: "note", text: "**The one-sentence rule.** *Fine-tuning teaches a model a new **skill or style**; RAG gives it new **knowledge**; prompting **steers** what it already has.* Most production systems that look like fine-tuning are actually RAG + a good system prompt. Fine-tune when you need consistent behavior that a prompt can't reliably produce, or when you're shrinking a model to cut cost." },
        { type: "callout", variant: "gotcha", text: "Fine-tuning is **not** a good way to teach facts. If you fine-tune `Llama` on your company wiki, it will learn the *style* of your wiki and hallucinate confident wrong answers about its contents. Facts belong in a retrieval index, not in the weights. This mistake is made in roughly every first fine-tuning project." },
      ]
    },

    {
      id: "sft",
      title: "Supervised fine-tuning (SFT): teaching the shape of an answer",
      level: "core",
      body: [
        { type: "p", text: "**SFT** is ordinary supervised learning applied to a language model. You show it thousands of `(instruction, ideal response)` pairs and continue the *same* next-token cross-entropy objective it was pretrained with — the only thing that changes is the data. After enough examples it internalizes the *pattern* 'when I see an instruction, I produce a helpful completion in this format.'" },
        { type: "math", tex: String.raw`\mathcal{L}_{\text{SFT}}(\theta) = -\,\mathbb{E}_{(x,y)\sim\mathcal{D}}\!\left[\sum_{t} \log \pi_\theta\big(y_t \mid x,\, y_{<t}\big)\right]` },
        { type: "p", text: "Read that as: for each token $y_t$ of the *response*, maximize the model's log-probability of the correct next token given the prompt $x$ and everything written so far $y_{<t}$. It is the same loss as pretraining; the magic is entirely in curating $\\mathcal{D}$." },
        { type: "heading", text: "Instruction datasets & chat templates" },
        { type: "p", text: "An instruction dataset is a list of conversations. Each model family expects a specific **chat template** — a fixed set of special tokens that mark where the system prompt, the user turn, and the assistant turn begin and end. Get this wrong and the model learns nothing useful, because at inference time the harness will wrap prompts in tokens your fine-tune never saw." },
        { type: "code", lang: "text", code: "# The Llama-3 chat template (rendered). The special tokens ARE the interface.\n<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe capital of France is Paris.<|eot_id|>" },
        { type: "callout", variant: "tip", text: "**Never hand-format the template yourself.** Every tokenizer ships one: `tokenizer.apply_chat_template(messages, tokenize=False)`. Use it for both training and inference so the tokens match exactly. `trl`'s `SFTTrainer` applies it automatically when your data is in the `messages` format." },
        { type: "heading", text: "Loss masking — train only on the completion" },
        { type: "p", text: "Here is the single most important SFT detail, and the one tutorials skip. You do **not** want to train the model to *generate the user's question* — only to generate the *answer*. So you mask the loss on the prompt tokens: set their labels to $-100$ (PyTorch's `ignore_index`), and cross-entropy skips them. The gradient flows only through the assistant's tokens." },
        { type: "math", tex: String.raw`\mathcal{L}_{\text{SFT}}(\theta) = -\,\mathbb{E}\!\left[\sum_{t}\; \mathbb{1}[\,y_t \in \text{completion}\,]\;\log \pi_\theta\big(y_t \mid x, y_{<t}\big)\right]` },
        { type: "code", lang: "py", code: "import torch\n\n# Build labels that ignore the prompt. prompt_ids = the templated question,\n# response_ids = the assistant answer (+ EOS).\nprompt_ids   = tokenizer(prompt,   add_special_tokens=False).input_ids\nresponse_ids = tokenizer(response, add_special_tokens=False).input_ids\n\ninput_ids = prompt_ids + response_ids\nlabels    = [-100] * len(prompt_ids) + response_ids   # -100 == ignored by loss\n\ninput_ids = torch.tensor(input_ids)\nlabels    = torch.tensor(labels)\n# HF models compute cross-entropy internally, skipping every label == -100." },
        { type: "callout", variant: "gotcha", text: "Forget to mask and the model spends half its capacity learning to *parrot user prompts*, which dilutes the signal you care about and can make it echo questions back at you. `trl`'s `DataCollatorForCompletionOnlyLM` and the `SFTTrainer` handle this for you — but only if you configure the response template correctly." },
        { type: "heading", text: "Full-parameter fine-tuning and why it hurts" },
        { type: "p", text: "The obvious way to do SFT is to update **all** the weights — full fine-tuning. It works and is the quality ceiling, but the memory bill is brutal. With the Adam optimizer you must hold, for every one of $N$ parameters: the weight, its gradient, and *two* optimizer moments — and for stable training, master copies in fp32." },
        { type: "math", tex: String.raw`\text{Mem} \approx \underbrace{2N}_{\text{fp16 weights}} + \underbrace{2N}_{\text{grads}} + \underbrace{12N}_{\text{Adam }m,v,\text{fp32 master}} \;\approx\; 16N \text{ bytes}` },
        { type: "p", text: "For a 7-billion-parameter model that is roughly **112 GB** of static training memory (fp16 weights + gradients + optimizer state) — before activations — which does not fit on a single 80 GB A100, let alone a consumer GPU. This memory wall is the entire reason parameter-efficient fine-tuning was invented, and it is where we go next." },
        { type: "callout", variant: "note", text: "**When full FT is still right:** when you have the hardware and want maximum quality, when adapting a *small* model (≤1–2B), or when doing large-scale continued pretraining. For everything else on one or two GPUs, LoRA/QLoRA is the default in 2025–2026." },
      ]
    },

    {
      id: "peft-lora",
      title: "PEFT & LoRA: fine-tuning a giant by learning a tiny low-rank patch",
      level: "core",
      body: [
        { type: "p", text: "**Parameter-efficient fine-tuning (PEFT)** is a family of methods that freeze the pretrained weights and train only a small number of *new* parameters — often <1% of the model. You get most of the quality of full fine-tuning at a fraction of the memory, and you end up with a few-megabyte 'adapter' you can swap in and out. The dominant PEFT method, by a wide margin, is **LoRA**." },
        { type: "heading", text: "The idea, and where it comes from" },
        { type: "p", text: "LoRA (Hu et al., 2021, *Low-Rank Adaptation*) starts from an empirical observation: the *update* a model needs to adapt to a new task has **low intrinsic rank**. You are not changing what each weight matrix fundamentally does; you are nudging it in a few directions. And a low-rank matrix can be written as a skinny product — exactly the low-rank factorization you met in the **SVD / low-rank approximation section of the Linear Algebra track**." },
        { type: "p", text: "Recall from that section: the best rank-$r$ approximation of any matrix is a sum of $r$ outer products, $A_k = \\sum_{i=1}^{r} \\sigma_i u_i v_i^\\top$, which factors as a tall matrix times a wide matrix. LoRA turns that *analysis* tool into a *training* trick: instead of finding a low-rank approximation of an existing matrix, it **learns a low-rank *update* from scratch**." },
        { type: "heading", text: "The derivation" },
        { type: "p", text: "Full fine-tuning replaces a frozen pretrained weight $W_0 \\in \\mathbb{R}^{d \\times k}$ with $W_0 + \\Delta W$, where $\\Delta W$ is the learned change. LoRA's bet is that $\\Delta W$ can be **constrained to rank $r \\ll \\min(d,k)$**, so we factor it as a product of two skinny matrices:" },
        { type: "math", tex: String.raw`W = W_0 + \Delta W = W_0 + BA, \qquad B \in \mathbb{R}^{d\times r},\; A \in \mathbb{R}^{r\times k},\; r \ll \min(d,k)` },
        { type: "p", text: "We **freeze $W_0$** and train only $A$ and $B$. The forward pass adds the two paths (and scales the update by $\\alpha/r$, see below):" },
        { type: "math", tex: String.raw`h = W_0 x + \frac{\alpha}{r}\, B A x` },
        { type: "p", text: "The parameter count drops from $d\\cdot k$ to $r(d + k)$. For a $4096\\times4096$ attention projection ($16.7$M params) at rank $r=8$, that is $2\\cdot 8\\cdot 4096 = 65{,}536$ trainable params — a **256× reduction** for that matrix." },
        { type: "callout", variant: "tip", text: "**Initialization matters and is elegant.** $A$ is initialized from a random Gaussian and $B$ is initialized to **zero**. So at step 0, $BA = 0$ and the model is *exactly* the pretrained model — training starts from the original behavior and only departs from it as $B$ learns. No warm-up shock, no risk of a random adapter wrecking the base." },
        { type: "heading", text: "The three knobs: rank r, alpha, and target modules" },
        { type: "table",
          headers: ["Hyperparameter", "What it controls", "Sensible default (2025)"],
          rows: [
            ["**rank $r$**", "capacity of the update; higher = more expressive but more params & overfitting risk", "$8$–$32$ (start at $16$)"],
            ["**$\\alpha$ (alpha)**", "scales the update by $\\alpha/r$; effectively the adapter's learning-rate multiplier", "set $\\alpha = 2r$ (e.g. $r{=}16,\\alpha{=}32$)"],
            ["**target_modules**", "which weight matrices get an adapter", "all linear layers (attn **and** MLP)"],
            ["**dropout**", "regularizes the adapter", "$0.05$–$0.1$"],
          ]
        },
        { type: "p", text: "Why the $\\alpha/r$ scaling? It **decouples** the update magnitude from the rank. If you double $r$ to get more capacity, the scaling keeps the *effective* step size roughly constant, so you don't have to re-tune the learning rate every time you change rank. The rule of thumb $\\alpha = 2r$ has held up well in practice." },
        { type: "callout", variant: "note", text: "**Which layers?** The original paper adapted only the attention $q$ and $v$ projections. The current consensus (and the QLoRA paper) is that adapting **all linear layers** — $q,k,v,o$ *and* the MLP `gate/up/down` projections — measurably improves quality for a small extra cost. When in doubt, target them all." },
        { type: "heading", text: "LoRA from scratch in PyTorch — it's ~15 lines" },
        { type: "code", lang: "py", code: "import torch, torch.nn as nn\n\nclass LoRALinear(nn.Module):\n    \"\"\"Wrap a frozen nn.Linear with a trainable low-rank update B@A.\"\"\"\n    def __init__(self, base: nn.Linear, r=8, alpha=16):\n        super().__init__()\n        self.base = base\n        for p in self.base.parameters():\n            p.requires_grad = False           # freeze W0\n        d_out, d_in = base.weight.shape\n        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)  # Gaussian init\n        self.B = nn.Parameter(torch.zeros(d_out, r))        # zero init -> dW=0\n        self.scaling = alpha / r\n\n    def forward(self, x):\n        return self.base(x) + self.scaling * (x @ self.A.T @ self.B.T)\n\n# Only A and B carry gradients; the base weight never moves.\nlayer = LoRALinear(nn.Linear(4096, 4096), r=8, alpha=16)\ntrainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)\nprint(trainable)   # 65,536  -- vs 16.7M for the full matrix" },
        { type: "callout", variant: "good", text: "**Adapters are composable and cheap to ship.** A LoRA adapter for a 7B model is typically **10–200 MB**. You can host one base model in memory and hot-swap dozens of task-specific adapters, or **merge** an adapter back into the weights (`W_0 + BA`) for zero inference overhead. This is why LoRA is the backbone of the fine-tuning economy." },
      ]
    },

    {
      id: "qlora",
      title: "QLoRA & friends: fine-tuning a 70B model on one GPU",
      level: "core",
      body: [
        { type: "p", text: "LoRA slashes the *optimizer* memory, but you still have to hold the frozen base weights in GPU memory. A 70B model in fp16 is ~140 GB — still out of reach. **QLoRA** (Dettmers et al., 2023) closes the gap and is the single innovation that *democratized* fine-tuning: it made 65B fine-tuning fit on a single 48 GB GPU, and 7B fit on a free Colab." },
        { type: "heading", text: "The idea: a 4-bit frozen base + fp16 LoRA on top" },
        { type: "p", text: "QLoRA is LoRA with one change: **quantize the frozen base model to 4-bit** and keep only the small LoRA adapters in higher precision. Since $W_0$ is frozen, its precision only needs to be good enough for the *forward pass* — no gradients ever update it — so aggressive quantization costs almost nothing in quality. The trainable path ($A$, $B$) stays in bf16 where precision actually matters." },
        { type: "table",
          headers: ["QLoRA innovation", "What it does"],
          rows: [
            ["**4-bit NormalFloat (NF4)**", "an information-theoretically optimal 4-bit datatype for the (roughly Gaussian) pretrained weights"],
            ["**Double quantization**", "quantizes the quantization constants themselves — saves another ~0.4 bits/param"],
            ["**Paged optimizers**", "pages optimizer state to CPU RAM on memory spikes, preventing OOM crashes"],
          ]
        },
        { type: "math", tex: String.raw`\underbrace{W_0}_{\text{4-bit NF4, frozen}}\;x \;+\; \underbrace{\tfrac{\alpha}{r}\,B A}_{\text{bf16, trainable}}\;x \;=\; h` },
        { type: "p", text: "During the forward pass each 4-bit weight block is **dequantized on the fly** to bf16, used in the matmul, and discarded — so peak memory holds the base at ~4 bits/param, not 16. The result: a 7B model fine-tunes in **under 10 GB**, and a 70B in ~48 GB." },
        { type: "callout", variant: "good", text: "**Why this mattered.** Before QLoRA, fine-tuning a serious model required a cluster and a budget. After QLoRA, a student with one gaming GPU could specialize a 7B–13B model overnight. The entire open fine-tuning ecosystem — thousands of community models on the Hub — rests on this trick." },
        { type: "callout", variant: "gotcha", text: "QLoRA trades a little **speed** for the memory win: on-the-fly dequantization makes each step slower than plain LoRA on unquantized weights. If the fp16 base *fits*, plain LoRA trains faster. QLoRA is the tool for when it *doesn't* fit — which, for large models on one GPU, is most of the time." },
        { type: "heading", text: "The rest of the PEFT zoo (briefly)" },
        { type: "table",
          headers: ["Method", "Trainable part", "Note"],
          rows: [
            ["**Adapters** (Houlsby, 2019)", "small bottleneck MLPs inserted between layers", "the original PEFT; adds inference latency (extra layers)"],
            ["**Prefix / P-tuning**", "trainable 'virtual token' vectors prepended to the keys/values", "no weight change; steers attention"],
            ["**Prompt tuning**", "a handful of learned soft-prompt embeddings", "tiniest footprint; weaker for hard tasks"],
            ["**DoRA / rsLoRA**", "weight-decomposed & rank-stabilized LoRA variants", "small quality bumps over vanilla LoRA"],
          ]
        },
        { type: "callout", variant: "tip", text: "**In practice you will almost always reach for LoRA or QLoRA.** The alternatives are worth knowing for interviews and for edge cases (prefix-tuning shines when you must keep the weights byte-identical), but LoRA won the ecosystem: best quality-per-parameter, mergeable, and supported everywhere." },
      ]
    },

    {
      id: "rlhf",
      title: "RLHF: the reward model, Bradley–Terry, and PPO with a KL leash",
      level: "core",
      body: [
        { type: "p", text: "SFT teaches a model to *imitate* good answers. But 'good' is subjective and hard to demonstrate — it is far easier for a human to say '*answer B is better than answer A*' than to write the perfect answer from scratch. **RLHF** (Reinforcement Learning from Human Feedback), the technique behind InstructGPT and ChatGPT, turns those cheap pairwise comparisons into a training signal. It has two steps: **train a reward model**, then **optimize the policy against it with RL**." },
        { type: "heading", text: "Step 1 — the reward model & the Bradley–Terry loss" },
        { type: "p", text: "We collect prompts, sample two responses $y_w$ (winner) and $y_l$ (loser) from the SFT model, and have a human pick the better one. We want a **reward model** $r_\\phi(x, y)$ — a copy of the LLM with a scalar head — that scores good responses higher. But 'higher by how much'? We need a probabilistic model linking a *scalar reward gap* to the *probability a human prefers one response*. That model is **Bradley–Terry** (1952), the classic model for pairwise comparisons:" },
        { type: "math", tex: String.raw`P(y_w \succ y_l \mid x) = \frac{\exp\big(r_\phi(x,y_w)\big)}{\exp\big(r_\phi(x,y_w)\big) + \exp\big(r_\phi(x,y_l)\big)}` },
        { type: "p", text: "Divide numerator and denominator by $\\exp(r_\\phi(x,y_w))$ and the right-hand side collapses into a **sigmoid of the reward difference** — the same logistic form as binary classification:" },
        { type: "math", tex: String.raw`P(y_w \succ y_l \mid x) = \frac{1}{1 + \exp\big(-(r_\phi(x,y_w) - r_\phi(x,y_l))\big)} = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)` },
        { type: "p", text: "Now train $\\phi$ by **maximum likelihood** on the human labels: minimize the negative log-likelihood that the model agrees with the human's choice. This is the reward-model loss, derived — not memorized:" },
        { type: "math", tex: String.raw`\mathcal{L}_{\text{RM}}(\phi) = -\,\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big]` },
        { type: "code", lang: "py", code: "import torch, torch.nn.functional as F\n\ndef reward_model_loss(r_chosen, r_rejected):\n    \"\"\"Bradley-Terry / pairwise logistic loss. r_* are scalar rewards, shape (B,).\"\"\"\n    # -log sigmoid(r_w - r_l)  ==  softplus(-(r_w - r_l))\n    return -F.logsigmoid(r_chosen - r_rejected).mean()\n\nr_chosen   = torch.tensor([2.0, 0.5, 3.1])\nr_rejected = torch.tensor([1.0, 0.7, 1.0])\nprint(reward_model_loss(r_chosen, r_rejected).item())   # small when chosen > rejected" },
        { type: "callout", variant: "note", text: "Only the *gap* $r(x,y_w) - r(x,y_l)$ is identified by the data — the absolute scale of the reward is arbitrary (add a constant to every reward and the loss is unchanged). This is why raw reward values are meaningless to compare across reward models, and why the RL step must be *regularized* (next)." },
        { type: "heading", text: "Step 2 — PPO fine-tuning against the reward, with a KL penalty" },
        { type: "p", text: "Now we have a learned reward $r_\\phi$. Treat the LLM as a **policy** $\\pi_\\theta$ that, given a prompt, generates a response (a sequence of token 'actions'), and use reinforcement learning — specifically **PPO** (see the RL track) — to maximize the expected reward. But naive reward maximization is a disaster: the policy will find gibberish that happens to score high on the imperfect reward model (**reward hacking**). The fix is a **KL-divergence penalty** that leashes the policy to the original SFT model $\\pi_{\\text{ref}}$:" },
        { type: "math", tex: String.raw`\max_{\theta}\; \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot\mid x)}\Big[\, r_\phi(x,y) \;-\; \beta\, \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big) \Big]` },
        { type: "p", text: "In practice the KL term is applied **per-token** as a penalty subtracted from the reward, giving a shaped reward that PPO then optimizes with its clipped objective:" },
        { type: "math", tex: String.raw`R(x,y) = \underbrace{r_\phi(x,y)}_{\text{be good}} \;-\; \beta\,\underbrace{\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}}_{\text{don't drift from SFT}}` },
        { type: "callout", variant: "warn", text: "**Why the KL leash is non-negotiable.** The reward model is a flawed proxy for human preference. Without the KL term, PPO will exploit its blind spots — producing repetitive, sycophantic, or degenerate text that scores high but reads terribly. $\\beta$ is the safety dial: too low and the model reward-hacks and forgets how to write; too high and it never improves. Tuning $\\beta$ is most of the art of RLHF." },
        { type: "callout", variant: "gotcha", text: "**RLHF is operationally heavy.** At training time you hold *four* models in memory — the policy, the frozen reference, the reward model, and (for PPO) a value/critic model — plus an online generation loop. It is powerful but finicky and expensive, which is exactly the pain that DPO was invented to remove." },
      ]
    },

    {
      id: "dpo",
      title: "DPO: skip the reward model, optimize preferences directly",
      level: "core",
      body: [
        { type: "p", text: "**Direct Preference Optimization** (Rafailov et al., 2023) is the insight that you don't actually *need* the reward model or the RL loop. The reward model, the KL-constrained objective, and the policy are all coupled by a clean bit of math — and when you follow that math through, the reward model **cancels out**, leaving a simple classification loss you can train with plain backprop. It has largely replaced PPO for open-model alignment because it is dramatically simpler." },
        { type: "heading", text: "The derivation (the part worth understanding)" },
        { type: "p", text: "Start from the RLHF objective. It is a known result that the policy maximizing 'expected reward minus $\\beta\\cdot$KL-to-reference' has a **closed-form optimum** — the reference distribution reweighted by the exponentiated reward:" },
        { type: "math", tex: String.raw`\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\tfrac{1}{\beta}\, r(x,y)\Big)` },
        { type: "p", text: "where $Z(x) = \\sum_y \\pi_{\\text{ref}}(y\\mid x)\\exp(r(x,y)/\\beta)$ is an intractable normalizer. Now **invert** this: solve for the reward in terms of the optimal policy. Take logs and rearrange:" },
        { type: "math", tex: String.raw`r(x,y) = \beta\,\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\,\log Z(x)` },
        { type: "p", text: "This is the trick: **the reward is just a scaled log-ratio between the policy and the reference.** Now substitute this expression into the Bradley–Terry preference model from the RLHF section. In the reward *difference* $r(x,y_w) - r(x,y_l)$, the ugly $\\beta\\log Z(x)$ term is identical for both responses and **cancels**:" },
        { type: "math", tex: String.raw`\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log \sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]` },
        { type: "p", text: "Look at what happened. There is **no reward model** and **no RL** — just a loss over the policy and a frozen reference, computed directly from the preference pairs. It says: *raise the log-probability of the chosen response and lower the log-probability of the rejected one, relative to the reference, and pass the gap through a sigmoid.* The KL constraint is baked in through the $\\pi_{\\text{ref}}$ ratio; $\\beta$ still controls how far you drift." },
        { type: "code", lang: "py", code: "import torch, torch.nn.functional as F\n\ndef dpo_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta=0.1):\n    \"\"\"All args are sum-of-log-probs of the response under policy / reference.\"\"\"\n    pi_logratios  = pi_logp_w  - pi_logp_l     # policy prefers chosen by how much\n    ref_logratios = ref_logp_w - ref_logp_l    # reference's baseline preference\n    logits = beta * (pi_logratios - ref_logratios)\n    return -F.logsigmoid(logits).mean()\n# That's the whole algorithm. No reward net, no critic, no rollouts." },
        { type: "callout", variant: "good", text: "**DPO vs PPO in one line.** DPO gets you ~90% of RLHF's benefit with ~10% of the complexity: a single supervised-style loss, two models in memory (policy + frozen ref) instead of four, and no unstable RL loop. For most open-model alignment in 2025–2026, DPO (or a variant) is the default; PPO/GRPO win when you have an *online* signal or a *verifiable* reward (math, code)." },
        { type: "heading", text: "The 2025–2026 variant landscape" },
        { type: "table",
          headers: ["Method", "Key idea", "Use when"],
          rows: [
            ["**DPO**", "offline pairwise loss from chosen/rejected", "the reliable default with preference pairs"],
            ["**ORPO**", "merges SFT + preference into one stage via an odds-ratio term; **no reference model**", "you want one-shot SFT+align with no separate SFT stage and half the memory"],
            ["**KTO**", "learns from *unpaired* thumbs-up/down binary feedback", "you have thumbs up/down data, not matched preference pairs"],
            ["**GRPO**", "online, critic-free PPO; advantage from *group* of samples' relative scores", "verifiable rewards / reasoning (used by DeepSeek for math & code)"],
            ["**SimPO**", "reference-free, length-normalized reward", "cutting the reference model, avoiding length bias"],
          ]
        },
        { type: "callout", variant: "tip", text: "**GRPO is why 2025's reasoning models exist.** DeepSeek-R1 showed that if your reward is *verifiable* (did the code pass? is the math answer correct?), you can drop the reward model entirely and use GRPO — sample a group of answers per prompt, reward the correct ones, and use the group's mean as the baseline. It is the engine behind the reasoning-model wave. See the RL track for PPO/GRPO mechanics." },
      ]
    },

    {
      id: "data",
      title: "Data for fine-tuning: quality beats quantity, every time",
      level: "core",
      body: [
        { type: "p", text: "The most important finding in the fine-tuning literature is almost embarrassingly simple: **a small set of excellent examples beats a large set of mediocre ones.** Meta's LIMA paper aligned a strong assistant with just **1,000** carefully curated examples. Your effort belongs in curation, not collection." },
        { type: "callout", variant: "tip", text: "**A practical starting budget.** For SFT of a task/style: **500–5,000** hand-checked examples often suffice with LoRA. For preference tuning: **1,000–10,000** pairs. Diminishing returns kick in fast — spend your time removing bad examples, not adding more. One wrong-format or toxic example can teach the model more than a hundred good ones un-teach." },
        { type: "heading", text: "Synthetic data & distillation" },
        { type: "p", text: "You rarely need to hand-write everything. **Distillation** — generating training data from a stronger 'teacher' model (e.g. use GPT-4-class outputs to fine-tune a 7B 'student') — is how most open instruction models are built. Combined with careful filtering, synthetic data scales curation cheaply." },
        { type: "callout", variant: "warn", text: "**Two cautions with synthetic data.** (1) **Legal/ToS:** many providers forbid using their outputs to train competing models — check the terms. (2) **Model collapse:** train repeatedly on a model's own outputs and quality degrades — errors and biases compound, diversity shrinks. Always mix in real human data and filter aggressively." },
        { type: "heading", text: "Catastrophic forgetting" },
        { type: "p", text: "**Catastrophic forgetting** is the tendency of a network to *lose* previously learned abilities when trained on new data — fine-tune a chat model hard on legal documents and it may forget how to do arithmetic or hold a normal conversation. It is the alignment world's version of overfitting-to-the-new." },
        { type: "table",
          headers: ["Defense", "How"],
          rows: [
            ["**Use LoRA, not full FT**", "the frozen base *can't* forget — only the small adapter changes; LoRA 'learns less and forgets less'"],
            ["**Mix in general data**", "blend a slice of general instruction data with your task data so old skills stay warm"],
            ["**Fewer epochs, lower LR**", "1–3 epochs is usually enough; over-training is the main cause of forgetting"],
            ["**Keep the chat template**", "train in the same format the base was aligned in, so you don't overwrite its instruction-following"],
          ]
        },
        { type: "callout", variant: "note", text: "This is a concrete reason LoRA is often *preferred over* full fine-tuning even when you can afford the latter: because the base weights are frozen, the model's general capabilities are structurally protected. You are adding a skill, not overwriting the brain." },
      ]
    },

    {
      id: "evaluation",
      title: "Evaluating a fine-tuned & aligned model",
      level: "core",
      body: [
        { type: "p", text: "Fine-tuning without evaluation is guessing. And evaluating aligned models is genuinely hard — 'better' is multi-dimensional and partly subjective. You need to measure the thing you improved *and* watch for the things you may have broken." },
        { type: "heading", text: "The four things to measure" },
        { type: "table",
          headers: ["Dimension", "How to measure it"],
          rows: [
            ["**Task performance**", "held-out task metrics: accuracy/F1 for classification, exact-match/pass@k for code, ROUGE/BLEU only as rough proxies for generation"],
            ["**Preference win-rate**", "the standard alignment metric: have a judge (human or a strong LLM) compare your model vs a baseline, report the **% of prompts your model wins**"],
            ["**Safety / refusals**", "does it refuse genuinely harmful requests *and* still comply with benign ones? (watch **over-refusal**)"],
            ["**Regressions**", "run general benchmarks (MMLU, GSM8K, coding) before *and* after to catch capabilities you silently broke"],
          ]
        },
        { type: "heading", text: "LLM-as-a-judge & win-rate" },
        { type: "p", text: "Since human evaluation is slow and expensive, the field standardized on **LLM-as-a-judge**: a strong model (e.g. Claude or GPT-4-class) scores or ranks responses. Frameworks like MT-Bench and AlpacaEval report a **win-rate** — the fraction of head-to-head comparisons your model wins against a reference — which correlates well with human preference at a fraction of the cost." },
        { type: "callout", variant: "gotcha", text: "**Judges are biased — control for it.** LLM judges favor **longer** answers, answers in **position A**, and answers in their **own style**. Mitigate: randomize which response is A/B and average both orders, cap/normalize length, and use a *different* model family as judge than the one you trained from. Never let the model judge its own outputs unchecked." },
        { type: "callout", variant: "warn", text: "**The regression trap.** Your model can win on your target task while quietly losing 5 points of MMLU and forgetting how to code. This 'alignment tax' is invisible unless you measure it. **Always** run a fixed suite of general benchmarks before and after fine-tuning and diff the numbers — a fine-tune that improves one thing and breaks three is a net loss." },
      ]
    },

    {
      id: "practical",
      title: "Practical: a full QLoRA SFT run, then DPO",
      level: "core",
      body: [
        { type: "p", text: "Time to build it. The HuggingFace stack — `transformers` + `peft` + `trl` + `bitsandbytes` — turns everything above into a few dozen lines. We'll do a QLoRA supervised fine-tune first, then align it further with DPO. This is the exact template real practitioners start from." },
        { type: "heading", text: "Part 1 — QLoRA SFT with `SFTTrainer`" },
        { type: "code", lang: "py", code: "import torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\nfrom peft import LoraConfig\nfrom trl import SFTTrainer, SFTConfig\nfrom datasets import load_dataset\n\nmodel_id = \"meta-llama/Llama-3.2-3B-Instruct\"\n\n# 1) 4-bit quantization config -- this is the 'Q' in QLoRA.\nbnb = BitsAndBytesConfig(\n    load_in_4bit=True,\n    bnb_4bit_quant_type=\"nf4\",              # NormalFloat-4, optimal for weights\n    bnb_4bit_compute_dtype=torch.bfloat16,  # dequantize to bf16 for the matmul\n    bnb_4bit_use_double_quant=True,         # double quantization\n)\n\ntokenizer = AutoTokenizer.from_pretrained(model_id)\nmodel = AutoModelForCausalLM.from_pretrained(\n    model_id, quantization_config=bnb, device_map=\"auto\", torch_dtype=torch.bfloat16,\n)\n\n# 2) LoRA config: rank 16, alpha 32, all linear layers.\npeft_config = LoraConfig(\n    r=16, lora_alpha=32, lora_dropout=0.05, bias=\"none\", task_type=\"CAUSAL_LM\",\n    target_modules=[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\n                    \"gate_proj\",\"up_proj\",\"down_proj\"],\n)" },
        { type: "code", lang: "py", code: "# 3) Data in chat format: each row has a 'messages' list. SFTTrainer applies\n#    the chat template AND masks the prompt tokens for you.\ndataset = load_dataset(\"HuggingFaceH4/ultrachat_200k\", split=\"train_sft[:2000]\")\n\nargs = SFTConfig(\n    output_dir=\"./llama-qlora-sft\",\n    num_train_epochs=1,                # 1-3 epochs; more overfits\n    per_device_train_batch_size=2,\n    gradient_accumulation_steps=8,     # effective batch = 16\n    learning_rate=2e-4,                # LoRA likes a higher LR than full FT\n    lr_scheduler_type=\"cosine\",\n    bf16=True,\n    max_length=1024,\n    logging_steps=10,\n)\n\ntrainer = SFTTrainer(\n    model=model, args=args, train_dataset=dataset,\n    peft_config=peft_config, processing_class=tokenizer,\n)\ntrainer.train()\ntrainer.save_model(\"./llama-qlora-sft\")   # saves ONLY the ~50MB adapter" },
        { type: "callout", variant: "tip", text: "Note the learning rate: **`2e-4`** for LoRA vs the `~1e-5` you'd use for full fine-tuning. Because you're training a tiny, zero-initialized adapter rather than nudging billions of pretrained weights, you can — and should — use a much larger step size. Cosine schedule + 1–3 epochs is the standard recipe." },
        { type: "heading", text: "Part 2 — align it further with `DPOTrainer`" },
        { type: "p", text: "After SFT, sharpen behavior on preference data. The dataset needs three columns — `prompt`, `chosen`, `rejected` — and `DPOTrainer` implements the exact loss we derived. With PEFT, the frozen base *is* the reference model, so you don't even load a separate copy." },
        { type: "code", lang: "py", code: "from trl import DPOTrainer, DPOConfig\nfrom datasets import load_dataset\n\n# Preference pairs: {'prompt': ..., 'chosen': ..., 'rejected': ...}\npref = load_dataset(\"trl-lib/ultrafeedback_binarized\", split=\"train[:2000]\")\n\ndpo_args = DPOConfig(\n    output_dir=\"./llama-dpo\",\n    beta=0.1,                          # the KL strength from the derivation\n    learning_rate=5e-6,                # DPO wants a SMALL lr -- it's delicate\n    per_device_train_batch_size=2,\n    gradient_accumulation_steps=8,\n    num_train_epochs=1,\n    bf16=True,\n)\n\ntrainer = DPOTrainer(\n    model=model,               # the SFT model (with LoRA adapters)\n    ref_model=None,            # None -> uses the frozen base as reference\n    args=dpo_args, train_dataset=pref, processing_class=tokenizer,\n    peft_config=peft_config,\n)\ntrainer.train()" },
        { type: "callout", variant: "gotcha", text: "**DPO is not SFT — turn the learning rate *down*.** DPO is far more sensitive; a large LR or too many epochs makes the model collapse into terse, weird, or degenerate text (it over-optimizes the preference gap). Start at `beta=0.1` and `lr≈5e-6`, run **one** epoch, and watch the chosen/rejected reward-margin metric — if it explodes, you've gone too far." },
        { type: "heading", text: "Loading the finished adapter for inference" },
        { type: "code", lang: "py", code: "from peft import PeftModel\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\nbase = AutoModelForCausalLM.from_pretrained(model_id, device_map=\"auto\")\nmodel = PeftModel.from_pretrained(base, \"./llama-dpo\")     # attach adapter\nmodel = model.merge_and_unload()   # fold BA into W0 -> zero inference overhead\n\ntok = AutoTokenizer.from_pretrained(model_id)\nmsgs = [{\"role\": \"user\", \"content\": \"Explain LoRA in one sentence.\"}]\nids = tok.apply_chat_template(msgs, return_tensors=\"pt\", add_generation_prompt=True)\nprint(tok.decode(model.generate(ids.to(model.device), max_new_tokens=80)[0]))" },
      ]
    },

    {
      id: "failures",
      title: "When fine-tuning goes wrong",
      level: "core",
      body: [
        { type: "p", text: "Fine-tuning fails in a small set of recognizable ways. Knowing them saves you days of confusion — most 'my fine-tune got worse' reports are one of these four." },
        { type: "heading", text: "Overfitting" },
        { type: "p", text: "Too many epochs or too small a dataset and the model **memorizes** your examples instead of learning the pattern. Symptoms: training loss keeps dropping while the model gets *worse* on anything slightly different, and it starts reproducing training examples verbatim. Fix: fewer epochs (1–3), more/more-diverse data, a held-out eval set you actually watch, and early stopping." },
        { type: "heading", text: "Format lock-in / mode collapse" },
        { type: "p", text: "Train on a narrow format and the model **can only produce that format**. Fine-tune exclusively on `{\"answer\": ...}` JSON and it will wrap *everything* — including 'hello' — in JSON, forever. Over-aggressive DPO similarly collapses outputs into short, samey responses. Fix: include format diversity, mix in general data, and don't over-optimize." },
        { type: "heading", text: "Catastrophic forgetting" },
        { type: "p", text: "Covered earlier: the model loses general skills as it gains the new one. Symptom: great on your task, suddenly bad at math/coding/conversation. Fix: LoRA over full FT, blend in general data, lower LR/fewer epochs — and *measure regressions* so you catch it." },
        { type: "heading", text: "The alignment tax" },
        { type: "p", text: "**Alignment tax** is the well-documented phenomenon that making a model safer and more helpful can slightly *reduce* raw capability on some benchmarks — the model spends capacity on hedging, refusing, and formatting. A little tax is acceptable; a large one means your reward model or preference data is pushing the wrong behavior (e.g. rewarding length or sycophancy)." },
        { type: "callout", variant: "warn", text: "**Over-refusal is the alignment tax's ugly cousin.** Push safety training too hard and the model refuses harmless requests — 'how do I *kill* a Linux process?' → a lecture on violence. It reads as safe on a refusal benchmark but is useless in production. Always evaluate benign-compliance alongside harmful-refusal; you want a sharp boundary, not a scared model." },
        { type: "callout", variant: "tip", text: "**The universal debugging move:** before blaming the algorithm, **print your actual training examples after templating and masking**. A huge fraction of failed fine-tunes are silent data bugs — wrong chat template, unmasked prompts, a stray newline, labels off by one. Look at the bytes the model actually sees." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Fine-tuning is a craft you learn by burning a few GPU-hours. Start absurdly small — a 0.5B–3B model, a few hundred examples — and get the *loop* working end to end before you scale. Free-tier Colab or Kaggle GPUs are enough for everything below." },
        { type: "list", ordered: true, items: [
          "**LoRA a small model on your own style.** Collect 200–500 examples in a distinctive voice (your writing, a character, a coding style) as `messages`, then QLoRA-fine-tune `Llama-3.2-1B-Instruct` with `SFTTrainer`. Chat with it before and after. This is the fastest way to *feel* what fine-tuning does.",
          "**Implement LoRA from scratch.** Write the `LoRALinear` wrapper from the LoRA section, inject it into a small pretrained model's attention layers, freeze everything else, and confirm the trainable-parameter count drops ~100×. Train it on a toy task and verify it learns — then compare against `peft`.",
          "**Reward model + Bradley–Terry.** Build a tiny reward model (a small LM + a scalar head) and train it on a preference dataset with the pairwise logistic loss you derived. Check that it scores `chosen > rejected` on held-out pairs — you've built the heart of RLHF.",
          "**SFT then DPO, and measure the delta.** Take an SFT model and align it with `DPOTrainer` on `ultrafeedback_binarized`. Compute an LLM-as-a-judge **win-rate** vs the pre-DPO model, controlling for position and length bias. Confirm DPO actually helped — and check MMLU didn't regress.",
          "**Break it on purpose.** Deliberately overfit (20 epochs on 50 examples) and watch it memorize; train only on JSON and watch format lock-in; crank the DPO learning rate and watch mode collapse. Learning the failure signatures firsthand makes you diagnose them instantly later.",
          "**QLoRA a 7B on one GPU.** Fine-tune a 7B–8B model in 4-bit on a real task dataset, then `merge_and_unload` the adapter and serve it. Confirm it fits in <12 GB and that inference works after merging — the full democratized-fine-tuning pipeline.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The primary sources here are unusually readable — the LoRA, QLoRA, and DPO papers are all short and worth reading in full. Start with InstructGPT for the *why*, then the method papers, then the HuggingFace docs for the *how*." },
        { type: "link", url: "https://arxiv.org/abs/2203.02155", text: "Ouyang et al. — InstructGPT / RLHF (the paper that defined SFT + reward model + PPO; read this first)" },
        { type: "link", url: "https://arxiv.org/abs/2106.09685", text: "Hu et al. — LoRA: Low-Rank Adaptation of Large Language Models (the original; short and clear)" },
        { type: "link", url: "https://arxiv.org/abs/2305.14314", text: "Dettmers et al. — QLoRA: Efficient Finetuning of Quantized LLMs (NF4, double quant, paged optimizers)" },
        { type: "link", url: "https://arxiv.org/abs/2305.18290", text: "Rafailov et al. — Direct Preference Optimization (the full derivation of the DPO loss)" },
        { type: "link", url: "https://arxiv.org/abs/2402.03300", text: "Shao et al. — DeepSeekMath / GRPO (critic-free RL for verifiable-reward reasoning)" },
        { type: "link", url: "https://huggingface.co/docs/peft", text: "HuggingFace PEFT docs — LoRA, QLoRA, adapters, and how to load/merge them" },
        { type: "link", url: "https://huggingface.co/docs/trl", text: "HuggingFace TRL docs — SFTTrainer, DPOTrainer, and the full alignment toolkit" },
      ]
    },
  ],

  packages: [
    { name: "transformers", why: "load base models, tokenizers, and chat templates" },
    { name: "peft", why: "LoRA/QLoRA adapters — `LoraConfig`, load & `merge_and_unload`" },
    { name: "trl", why: "the alignment trainers: `SFTTrainer`, `DPOTrainer`, reward modeling" },
    { name: "bitsandbytes", why: "4-bit NF4 quantization — the 'Q' in QLoRA" },
    { name: "datasets", why: "load & stream instruction / preference datasets from the Hub" },
    { name: "accelerate", why: "multi-GPU / mixed-precision launch plumbing under the trainers" },
    { name: "flash-attn", why: "fast, memory-efficient attention kernels for training long sequences" },
  ],

  gotchas: [
    "Fine-tuning teaches **skills/style**, not **facts** — put knowledge in RAG, not the weights, or you'll get confident hallucinations.",
    "Mask the prompt tokens (label $-100$); train the loss **only on the completion**, or the model wastes capacity parroting questions.",
    "Always use `tokenizer.apply_chat_template` for both train and inference — a mismatched template silently ruins the fine-tune.",
    "Set LoRA $\\alpha = 2r$ and use a **high** LR (`~2e-4`); for **DPO** flip to a **low** LR (`~5e-6`) — it collapses easily.",
    "Without the **KL penalty** ($\\beta$) in RLHF, PPO reward-hacks the imperfect reward model into degenerate text.",
    "In DPO the frozen base is your reference model (`ref_model=None` with PEFT) — no separate copy needed.",
    "1–3 epochs is usually enough; more causes overfitting, format lock-in, and catastrophic forgetting.",
    "Measure **regressions** (MMLU, coding) before/after — a win on your task can hide an alignment-tax loss everywhere else.",
  ],

  flashcards: [
    { q: "What are the three stages of the alignment pipeline?", a: "**Pretraining** (learn language) → **SFT** (learn to follow instructions) → **preference optimization** (RLHF/DPO — align behavior with human taste)." },
    { q: "Why mask the prompt tokens during SFT?", a: "You want the model to learn to generate the *answer*, not the *question*. Set prompt labels to $-100$ so cross-entropy skips them; train only on the completion." },
    { q: "State the LoRA update and its parameters.", a: "$W = W_0 + \\frac{\\alpha}{r}BA$ with $W_0$ frozen, $B\\in\\mathbb{R}^{d\\times r}$ (init 0), $A\\in\\mathbb{R}^{r\\times k}$ (init Gaussian), $r \\ll \\min(d,k)$. Only $A,B$ train." },
    { q: "Why does LoRA connect to SVD / low-rank approximation?", a: "The task-adaptation update $\\Delta W$ has low intrinsic rank, so it factors as a skinny product $BA$ — the same low-rank factorization the SVD gives (Linear Algebra track)." },
    { q: "What is QLoRA and why did it matter?", a: "LoRA on top of a **4-bit (NF4) quantized** frozen base. It cut fine-tuning memory ~4× (7B in <10GB, 70B on one 48GB GPU), democratizing fine-tuning." },
    { q: "Write the reward-model (Bradley–Terry) loss.", a: "$\\mathcal{L}_{RM} = -\\mathbb{E}[\\log\\sigma(r_\\phi(x,y_w) - r_\\phi(x,y_l))]$ — a pairwise logistic loss on chosen vs rejected." },
    { q: "Why does RLHF need a KL penalty to the reference model?", a: "The reward model is an imperfect proxy; without $-\\beta\\,\\mathrm{KL}(\\pi_\\theta\\|\\pi_{ref})$ the policy reward-hacks it into degenerate text. $\\beta$ leashes it to the SFT model." },
    { q: "What is DPO's key insight?", a: "The optimal RLHF policy gives reward $r=\\beta\\log\\frac{\\pi}{\\pi_{ref}}+\\beta\\log Z$. Sub into Bradley–Terry and $Z$ cancels — so you optimize preferences directly, no reward model, no RL." },
    { q: "Write the DPO loss.", a: "$-\\mathbb{E}[\\log\\sigma(\\beta\\log\\frac{\\pi_\\theta(y_w)}{\\pi_{ref}(y_w)} - \\beta\\log\\frac{\\pi_\\theta(y_l)}{\\pi_{ref}(y_l)})]$ — raise chosen, lower rejected log-prob vs the reference." },
    { q: "Fine-tune, prompt, or RAG — how to choose?", a: "Fine-tune for **behavior/style/skill**; RAG for **knowledge/facts**; prompting to **steer** what the model already has. Most 'fine-tune' needs are really RAG." },
    { q: "How do ORPO, KTO, and GRPO differ from DPO?", a: "**ORPO** merges SFT+alignment, no ref model. **KTO** uses unpaired thumbs up/down. **GRPO** is online, critic-free RL for *verifiable* rewards (math/code, DeepSeek-R1)." },
    { q: "What is catastrophic forgetting and how does LoRA help?", a: "Losing prior skills while learning new ones. LoRA freezes the base weights, so general capabilities are structurally protected — it 'learns less and forgets less'." },
  ],

  cheatsheet: [
    { label: "4-bit config", code: "BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')" },
    { label: "LoRA config", code: "LoraConfig(r=16, lora_alpha=32, target_modules='all-linear', task_type='CAUSAL_LM')" },
    { label: "SFT trainer", code: "SFTTrainer(model, args, train_dataset=ds, peft_config=cfg).train()" },
    { label: "DPO trainer", code: "DPOTrainer(model, ref_model=None, args=DPOConfig(beta=0.1), train_dataset=pref)" },
    { label: "Apply chat template", code: "tokenizer.apply_chat_template(messages, tokenize=False)" },
    { label: "Mask prompt in labels", code: "labels = [-100]*len(prompt_ids) + response_ids" },
    { label: "Reward / DPO loss core", code: "-F.logsigmoid(chosen - rejected).mean()" },
    { label: "Load + merge adapter", code: "PeftModel.from_pretrained(base, path).merge_and_unload()" },
    { label: "LoRA update", code: "h = W0 @ x + (alpha/r) * (B @ A @ x)" },
    { label: "SFT loss (completion only)", code: "CE over response tokens; prompt labels = -100" },
    { label: "Save adapter only", code: "trainer.save_model(dir)   # ~50MB, not the full model" },
    { label: "LoRA LR vs DPO LR", code: "SFT/LoRA ~2e-4   |   DPO ~5e-6" },
  ],
});
