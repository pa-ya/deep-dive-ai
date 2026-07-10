(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "llm-intro",
  name: "What Are LLMs & How They Work",
  language: "LLMs",
  group: "Large Language Models",
  navLabel: "LLMs: How They Work",
  tagline: "A large language model is a decoder-only transformer trained to predict the next token — this section shows *why* that one objective produces something that writes code, reasons, and translates.",
  color: "#EC4899",
  readMinutes: 50,
  sections: [
    {
      id: "what-is",
      title: "What an LLM actually is",
      level: "core",
      body: [
        { type: "p", text: "Strip away the hype and a **large language model** is a startlingly simple object: a **decoder-only transformer** (the exact architecture you built in the Transformers section) trained on one objective — **predict the next token** — over an enormous slice of the internet. That is the whole recipe. Everything else — chat, code, translation, chain-of-thought reasoning — is an *emergent consequence* of doing that one thing well enough, at enough scale." },
        { type: "p", text: "It is worth sitting with how surprising this is. Nobody told the model the rules of Python, the capital of France, or how to write a sonnet. We only ever asked it: *given this text, what token comes next?* The claim of the entire field — validated empirically over the last decade — is that **next-token prediction is a sufficiently rich objective that solving it forces the model to learn grammar, facts, arithmetic, code semantics, and rudimentary reasoning**, because all of those are needed to predict the next word accurately across billions of documents." },
        { type: "callout", variant: "note", text: "**The one-sentence definition.** An LLM is a probability distribution $p_\\theta(x_t \\mid x_{<t})$ over the next token, parameterized by a large transformer with weights $\\theta$, fit by maximizing the likelihood of a huge text corpus. To *generate*, you sample from it repeatedly. Hold onto that sentence — the next three sections unpack every piece of it." },
        { type: "p", text: "\"Large\" is doing real work in the name. A model with 100 million parameters trained the same way is grammatical but dim; the same architecture at 100 **billion** parameters is a different kind of thing entirely. The abilities that make LLMs useful appear only past certain scales of parameters, data, and compute — a phenomenon called **emergence** (section 6). This is why the field is obsessed with scale." },
        { type: "table",
          headers: ["What people call it", "What it really is", "Where it is built"],
          rows: [
            ["\"The model\" / \"the AI\"", "a transformer with weights $\\theta$", "Transformers section"],
            ["\"It predicts text\"", "$p_\\theta(x_t \\mid x_{<t})$, a next-token distribution", "section 2"],
            ["\"A token\"", "a sub-word unit from a BPE vocabulary", "section 3"],
            ["\"Generating\" / \"inference\"", "sampling tokens one at a time", "sections 2 & 5"],
            ["\"It learned that\"", "SGD on cross-entropy over a corpus", "section 7 + Neural Nets track"],
          ]
        },
        { type: "p", text: "This section is the **opener of the LLM track**. It assumes you have done the Transformers section (self-attention, the causal mask, the block, decoder-only architectures). Here we zoom out from *the architecture* to *the whole system*: how a stack of transformer blocks becomes ChatGPT — the objective, the tokenizer, the decoding loop, the training pipeline, the limits, and the 2025-26 landscape. Later LLM sections drill into each: building a GPT from scratch, scaling laws, fine-tuning (SFT/RLHF/DPO), and serving." },
      ]
    },

    {
      id: "autoregressive",
      title: "The autoregressive factorization — how generation works",
      level: "core",
      body: [
        { type: "p", text: "An LLM models the probability of a whole sequence of text. A sequence $x = (x_1, x_2, \\dots, x_T)$ of tokens has some joint probability $p(x)$. Modeling a joint distribution over thousands of positions directly is hopeless — but the **probability chain rule** (from the Probability section) lets us factor *any* joint distribution into a product of conditionals, exactly, with no approximation:" },
        { type: "math", tex: String.raw`p(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1}) = \prod_{t=1}^{T} p(x_t \mid x_{<t})` },
        { type: "p", text: "This is the **autoregressive** factorization: the probability of the sequence is the product of the probability of each token given everything before it. \"Auto-regressive\" literally means *regressing on itself* — each token is predicted from the model's own earlier tokens. An LLM is nothing more than a parameterized estimate of the single factor $p_\\theta(x_t \\mid x_{<t})$, applied at every position." },
        { type: "heading", text: "Why decoder-only transformers are built for exactly this" },
        { type: "p", text: "Recall the **causal mask** from the Transformers section: it forbids position $t$ from attending to any position $> t$. That mask is not an incidental detail — it is what makes the transformer compute $p_\\theta(x_t \\mid x_{<t})$ for *every* $t$ **simultaneously** in one forward pass, without any position cheating by looking at its own future. During training this is a massive efficiency win: one pass over a length-$T$ sequence yields $T$ next-token predictions and $T$ loss terms at once." },
        { type: "math", tex: String.raw`\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \qquad\text{(the cross-entropy / negative log-likelihood)}` },
        { type: "p", text: "Minimizing this loss *is* maximizing the likelihood the model assigns to real text. It is plain cross-entropy — the same loss from the classification sections — where the \"class\" being predicted is which of the $|V|$ vocabulary tokens comes next." },
        { type: "heading", text: "Generation: the autoregressive loop" },
        { type: "p", text: "Training predicts all positions in parallel; **generation is inherently sequential**. You feed in a prompt, get the distribution for the next token, pick one, *append it to the input*, and repeat. Each new token becomes part of the context for the next step:" },
        { type: "code", lang: "text", code: "prompt:  \"The capital of France is\"\n  step 1: p(next | \"The capital of France is\")      -> pick \" Paris\"\n  step 2: p(next | \"...France is Paris\")            -> pick \".\"\n  step 3: p(next | \"...is Paris.\")                  -> pick <eos>  (stop)\n\noutput:  \"The capital of France is Paris.\"" },
        { type: "callout", variant: "note", text: "**This is why chatbots stream word by word.** They are *literally* computing one token at a time in a loop; the streaming UI is just showing you the loop as it runs. It is also why generation cost grows with output length, and why the **KV-cache** (Transformers section) exists — to avoid recomputing attention over the whole prefix at every single step." },
        { type: "callout", variant: "gotcha", text: "An LLM does not \"plan\" a sentence and then write it. At each step it only ever produces a distribution over the *very next* token given the text so far. Any apparent long-range coherence emerges from conditioning on its own growing output — not from an internal outline. This is the root of several limits in section 8." },
      ]
    },

    {
      id: "tokenization",
      title: "Tokens: the atoms an LLM actually sees",
      level: "core",
      body: [
        { type: "p", text: "An LLM never sees letters or words — it sees **tokens**, integer IDs into a fixed **vocabulary** of sub-word units (typically 32k-256k entries). The mapping text ↔ tokens is done by a **tokenizer**, and the dominant algorithm is **Byte-Pair Encoding (BPE)** (covered in depth in NLP Foundations). We recap it here because *everything an LLM costs and can perceive is measured in tokens*." },
        { type: "p", text: "BPE starts from raw bytes and **greedily merges the most frequent adjacent pair** over and over, learning a vocabulary where common words become single tokens and rare words split into pieces. This is the sweet spot between two bad extremes: a character vocabulary (tiny vocab, but sequences become enormously long) and a word vocabulary (short sequences, but millions of words and no way to handle a word never seen in training)." },
        { type: "table",
          headers: ["Text", "Tokenizes to (roughly)", "# tokens"],
          rows: [
            ["\" the\"", "`[\" the\"]` — a common word is one token", "1"],
            ["\" tokenization\"", "`[\" token\", \"ization\"]`", "2"],
            ["\" antidisestablishment\"", "`[\" anti\", \"dis\", \"establishment\"]`", "3"],
            ["\"1234567\"", "`[\"123\", \"45\", \"67\"]` — digits split oddly", "3"],
            ["\"🦙\"", "`[bytes]` — emoji become several byte tokens", "2-4"],
          ]
        },
        { type: "p", text: "A useful rule of thumb for English: **~1 token ≈ 4 characters ≈ 0.75 words**, so ~750 words ≈ 1000 tokens. This is the unit of billing for APIs, the unit of the context window, and the unit of the model's attention cost." },
        { type: "heading", text: "The context window — the model's working memory" },
        { type: "p", text: "The **context window** is the maximum number of tokens the model can attend to at once — prompt plus generated output combined. It is a hard architectural limit: GPT-2 had 1,024; GPT-3 had 2,048; today's models run 128k-1M+ (section 9). Because attention is $O(n^2)$ in sequence length $n$ (Transformers section), a longer window is genuinely expensive, which is why context length is a headline spec and an active research frontier." },
        { type: "code", lang: "py", code: "# See exactly how a real LLM tokenizer chops your text.\nfrom transformers import AutoTokenizer\n\ntok = AutoTokenizer.from_pretrained(\"gpt2\")\ntext = \"LLMs read tokens, not words.\"\nids = tok.encode(text)\nprint(ids)                       # [3069, 10128, 1100, 16326, 11, 407, 2456, 13]\nprint(len(ids), \"tokens\")        # 8 tokens\nprint([tok.decode([i]) for i in ids])\n# ['LL', 'Ms', ' read', ' tokens', ',', ' not', ' words', '.']\n#  ^^^ note 'LLMs' is TWO tokens 'LL'+'Ms' — the model never sees the word whole" },
        { type: "callout", variant: "gotcha", text: "**Tokenization is the hidden cause of a whole class of LLM failures.** \"How many r's in strawberry?\" is hard partly because the model sees `straw`+`berry`, not letters. Arithmetic is shaky because numbers tokenize inconsistently. Reversing a string is hard for the same reason. When an LLM does something dumb with characters or digits, suspect the tokenizer first (section 8)." },
        { type: "callout", variant: "tip", text: "Every model family has its *own* tokenizer and vocabulary — a token count from GPT-2's tokenizer will not match Llama's or Qwen's. Always use the tokenizer that ships **with** the model. Use OpenAI's `tiktoken` for GPT-family counting and each model's `AutoTokenizer` otherwise." },
      ]
    },

    {
      id: "anatomy",
      title: "Model anatomy: from token IDs to a probability over the vocabulary",
      level: "core",
      body: [
        { type: "p", text: "Let us trace a single forward pass end to end. Everything here is assembled from parts you already have — this section is the *wiring diagram* that connects them into a language model." },
        { type: "list", ordered: true, items: [
          "**Embedding.** Each token ID indexes a row of an embedding matrix $E \\in \\mathbb{R}^{|V| \\times d}$, turning the integer into a $d$-dimensional vector. (Positional information is added here too — RoPE in most modern models.) Input `(T,)` integers → `(T, d)` floats.",
          "**N transformer blocks.** The `(T, d)` sequence flows through a stack of $N$ identical decoder blocks — causal multi-head self-attention + a feed-forward network, each with residual connections and normalization. This is the bulk of the parameters and the compute. Shape stays `(T, d)` throughout.",
          "**Unembedding → logits.** The final `(T, d)` states are multiplied by an output matrix (often the transpose of $E$, *tied weights*) to produce **logits** — one raw score per vocabulary token, at every position. Shape `(T, |V|)`.",
          "**Softmax.** Applying softmax to the last position's logits turns them into $p_\\theta(x_t \\mid x_{<t})$ — a probability distribution over the whole vocabulary. This is the distribution section 5 samples from.",
        ]},
        { type: "math", tex: String.raw`\underbrace{\text{ids}}_{(T,)} \xrightarrow{\;E\;} \underbrace{h_0}_{(T,d)} \xrightarrow{\;\times N \text{ blocks}\;} \underbrace{h_N}_{(T,d)} \xrightarrow{\;E^\top\;} \underbrace{\text{logits}}_{(T,|V|)} \xrightarrow{\;\text{softmax}\;} \underbrace{p(x_t\mid x_{<t})}_{(T,|V|)}` },
        { type: "heading", text: "Where do the parameters live?" },
        { type: "p", text: "For a decoder-only transformer with $N$ layers, hidden size $d$, and vocabulary $|V|$, the parameter count is dominated by the blocks. Per block: attention has $4d^2$ weights (the $Q,K,V,O$ projections), and the feed-forward network has $\\approx 8d^2$ (up to $4d$ and back). So each block is $\\approx 12d^2$, and the embedding/unembedding add $\\approx 2\\,|V|\\,d$:" },
        { type: "math", tex: String.raw`\#\text{params} \;\approx\; \underbrace{N \cdot 12 d^2}_{\text{transformer blocks}} \;+\; \underbrace{2\,|V|\,d}_{\text{(un)embedding}}` },
        { type: "p", text: "Plug in GPT-2 small ($N{=}12$, $d{=}768$, $|V|{=}50{,}257$): the blocks give $12 \\cdot 12 \\cdot 768^2 \\approx 85$M and the embeddings $\\approx 77$M, totaling ~124M — the published number. The formula is worth internalizing: **it is almost all $d^2$**, which is why widening a model is so expensive and why FLOPs scale the way scaling laws say they do (later section)." },
        { type: "heading", text: "What scale looks like across the eras" },
        { type: "table",
          headers: ["Model (era)", "Params", "Layers $N$", "Hidden $d$", "Context"],
          rows: [
            ["GPT-2 small (2019)", "124M", "12", "768", "1,024"],
            ["GPT-2 XL (2019)", "1.5B", "48", "1,600", "1,024"],
            ["GPT-3 (2020)", "175B", "96", "12,288", "2,048"],
            ["Llama 3 8B (2024)", "8B", "32", "4,096", "128k"],
            ["Llama 3.1 405B (2024)", "405B", "126", "16,384", "128k"],
            ["Qwen3 / Mistral / GPT-4-era (2025-26)", "frontier: MoE, 100s of B", "many", "large", "128k-1M+"],
          ]
        },
        { type: "callout", variant: "note", text: "**Mixture-of-Experts (MoE).** Most 2025-26 frontier open models (DeepSeek-V3, Llama 4, Qwen3-235B) are MoE: they have a huge *total* parameter count but activate only a fraction per token by routing to a few \"expert\" FFNs. E.g. a 235B-total model may use only ~22B *active* parameters per token — big model capacity at small-model inference cost. You will meet MoE properly in the architecture section." },
        { type: "callout", variant: "tip", text: "A fast napkin estimate: a model in `fp16` needs **~2 bytes per parameter** just to hold the weights. So a 7B model needs ~14 GB of VRAM, a 70B model ~140 GB. **Quantization** to 4-bit cuts that ~4x (a 7B model fits in ~4-5 GB), which is exactly what makes local models (section 10) runnable on a laptop." },
      ]
    },

    {
      id: "decoding",
      title: "Decoding & sampling strategies, derived",
      level: "core",
      body: [
        { type: "p", text: "At each step the model hands you a probability distribution over ~50k+ tokens. **Decoding** is the policy for turning that distribution into an actual chosen token. This is *not* part of the model — it is a knob you control at generation time, and it has an enormous effect on output quality. Getting it right is one of the highest-leverage, least-understood skills in using LLMs, so we derive each strategy." },
        { type: "p", text: "Let $z \\in \\mathbb{R}^{|V|}$ be the logits at the current step, and $p_i = \\operatorname{softmax}(z)_i$ the probabilities." },
        { type: "heading", text: "Greedy decoding" },
        { type: "p", text: "The simplest policy: always take the single most probable token." },
        { type: "math", tex: String.raw`x_t = \arg\max_i \; p_i` },
        { type: "p", text: "Deterministic and locally optimal, but globally myopic — a high-probability token now can lead into a low-probability dead end, and greedy output is notoriously **repetitive and bland** (\"the the the\" loops). Best for tasks with one right answer: classification, extraction, short factual answers." },
        { type: "heading", text: "Beam search" },
        { type: "p", text: "Keep the $b$ most probable *partial sequences* (\"beams\") at every step instead of one, expand them all, and re-prune to the top $b$ by cumulative log-probability. It approximates finding the globally most-likely sequence $\\arg\\max_x \\prod_t p(x_t \\mid x_{<t})$." },
        { type: "callout", variant: "gotcha", text: "Beam search shines in **low-entropy** tasks like translation and speech recognition, where there really is a single best sequence. For **open-ended** generation (chat, stories) it is a trap: maximizing sequence probability produces generic, repetitive, \"safe\" text, because the most *probable* continuation is rarely the most *interesting* one. Modern chat models almost never use beam search." },
        { type: "heading", text: "Temperature — reshaping the distribution" },
        { type: "p", text: "Before sampling, divide the logits by a **temperature** $T > 0$, then softmax. This is the master dial for the randomness/creativity trade-off:" },
        { type: "math", tex: String.raw`p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}` },
        { type: "p", text: "As $T \\to 0$ the distribution sharpens to a spike on the top token (→ greedy). As $T \\to \\infty$ it flattens toward uniform (pure randomness). $T = 1$ leaves the model's native distribution untouched. Intuition: $T$ stretches ($T{>}1$) or compresses ($T{<}1$) the *gaps* between logits before they are exponentiated." },
        { type: "table",
          headers: ["Temperature", "Effect", "Use for"],
          rows: [
            ["$T \\to 0$ (≈ greedy)", "deterministic, picks top token", "code, math, extraction, JSON"],
            ["$T \\approx 0.7$", "focused but varied", "general chat, Q&A (a common default)"],
            ["$T \\approx 1.0$", "the model's raw distribution", "balanced creative writing"],
            ["$T > 1.2$", "wild, surprising, error-prone", "brainstorming, poetry, deliberate diversity"],
          ]
        },
        { type: "heading", text: "Top-k sampling" },
        { type: "p", text: "Sampling from the *full* distribution risks occasionally drawing a bizarre low-probability token (the long tail of 50k options has a lot of garbage). **Top-k** truncates to the $k$ most probable tokens, renormalizes, and samples from those. It kills the tail but uses a fixed $k$ regardless of how peaked or flat the distribution is." },
        { type: "math", tex: String.raw`V_k = \{\text{top-}k \text{ tokens by } p_i\}, \qquad \tilde p_i = \frac{p_i}{\sum_{j \in V_k} p_j} \; \text{ for } i \in V_k` },
        { type: "heading", text: "Top-p (nucleus) sampling — the modern default" },
        { type: "p", text: "**Top-p** (Holtzman et al., 2019) fixes top-k's rigidity: instead of a fixed count, keep the smallest set of tokens whose cumulative probability first exceeds $p$, then sample from that renormalized \"nucleus.\"" },
        { type: "math", tex: String.raw`V_p = \text{smallest set with} \sum_{i \in V_p} p_i \ge p, \qquad \text{sample from renormalized } V_p` },
        { type: "p", text: "The key virtue is that the nucleus is **adaptive**: when the model is confident (peaked distribution) the set is small — maybe 2-3 tokens; when it is uncertain (flat distribution) the set grows to include more options. This tracks the model's own uncertainty. A typical modern default is $p = 0.9$-$0.95$ combined with temperature $\\approx 0.7$-$1.0$." },
        { type: "heading", text: "Repetition penalty" },
        { type: "p", text: "Even good sampling can loop. A **repetition (or frequency/presence) penalty** discourages tokens already generated by downweighting their logits before softmax — subtracting a constant per prior occurrence, or dividing the logit by a factor $r > 1$:" },
        { type: "math", tex: String.raw`z_i \;\leftarrow\; z_i - \alpha \cdot \operatorname{count}(i \text{ so far}) \qquad\text{(frequency penalty, } \alpha > 0)` },
        { type: "callout", variant: "tip", text: "**A practical recipe.** Deterministic tasks (code, extraction, tool-calls, evals): `temperature=0` (greedy). General chat: `temperature≈0.7, top_p≈0.9`. Creative writing: `temperature≈1.0, top_p≈0.95`, maybe a small repetition penalty. Change **one** knob at a time — temperature and top-p interact, and cranking both at once gives incoherent output." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef sample_next(logits, temperature=1.0, top_p=0.9):\n    \"\"\"Temperature + nucleus (top-p) sampling, from scratch.\"\"\"\n    logits = logits / max(temperature, 1e-8)          # 1) temperature\n    probs = np.exp(logits - logits.max())\n    probs /= probs.sum()                              # softmax\n\n    order = np.argsort(probs)[::-1]                   # 2) sort desc\n    cumulative = np.cumsum(probs[order])\n    cutoff = np.searchsorted(cumulative, top_p) + 1   # smallest nucleus\n    keep = order[:cutoff]\n\n    nucleus = probs[keep] / probs[keep].sum()         # 3) renormalize\n    return np.random.choice(keep, p=nucleus)          # 4) sample\n\nrng = np.random.seed(0)\nlogits = np.array([2.0, 1.0, 0.1, -1.0, -3.0])       # 5-token toy vocab\nprint(sample_next(logits, temperature=0.7, top_p=0.9))" },
        { type: "callout", variant: "note", text: "In HuggingFace `transformers`, all of this is exposed on `model.generate(...)`: `do_sample=True`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `num_beams`. Setting `do_sample=False` (the default) gives greedy/beam. You will use these constantly — they are the difference between a model that feels sharp and one that feels broken." },
      ]
    },

    {
      id: "emergence",
      title: "Emergent abilities & in-context learning",
      level: "core",
      body: [
        { type: "p", text: "Here is the phenomenon that turned language models from a curiosity into a revolution. As you scale a model, some capabilities improve smoothly — but others appear to switch on **abruptly** past a certain scale, absent in smaller models and present in larger ones. These are called **emergent abilities**: multi-step arithmetic, following instructions, chain-of-thought reasoning, and the headline one below." },
        { type: "heading", text: "In-context learning — \"learning\" without training" },
        { type: "p", text: "The signature emergent ability, discovered with GPT-3 (2020), is **in-context learning**: the model performs a *new* task after seeing a few examples **in its prompt**, with no gradient updates and no weight changes at all. You are not training it — you are *conditioning* it. This is what \"few-shot prompting\" actually is." },
        { type: "code", lang: "text", code: "Prompt (2-shot, given entirely at inference time):\n  Translate English to French.\n  sea otter  => loutre de mer\n  cheese     => fromage\n  car        =>            <- model completes: \" voiture\"\n\nNo weights changed. The examples in the context steered p(next token)." },
        { type: "p", text: "Why does this work? A compelling view: to minimize next-token loss over the whole internet, the model had to become good at **recognizing and continuing patterns** — and \"a list of `input => output` pairs\" is a pattern. The prompt examples let the model *infer the task* and continue the pattern. In effect, next-token prediction at scale produced a general-purpose pattern-completer, and prompting is how you point it at your task." },
        { type: "table",
          headers: ["Prompting style", "What you give", "When to use"],
          rows: [
            ["**Zero-shot**", "just the instruction/question", "capable instruct models, simple tasks"],
            ["**Few-shot**", "instruction + a few examples", "custom formats, nudging output shape"],
            ["**Chain-of-thought**", "\"think step by step\" / worked examples", "arithmetic, logic, multi-step reasoning"],
          ]
        },
        { type: "callout", variant: "note", text: "**Chain-of-thought (CoT)** is a special case worth naming: prompting the model to produce intermediate reasoning steps *before* its final answer dramatically improves accuracy on hard problems. Mechanistically it makes sense — each generated reasoning token becomes context the model conditions on, giving it more \"compute steps\" to reach the answer. The 2024-26 \"reasoning models\" (DeepSeek-R1, the o-series) are trained to do long CoT automatically." },
        { type: "callout", variant: "gotcha", text: "**Emergence is contested and partly a measurement artifact.** Some 2023-24 work (Schaeffer et al.) argues certain \"sudden\" jumps are an illusion created by harsh all-or-nothing metrics — under smoother metrics the improvement is gradual. The practical takeaway is unchanged: **bigger models can do qualitatively new things**, but be skeptical of dramatic \"emergence at exactly N parameters\" claims." },
        { type: "p", text: "*Why* scale produces this is the subject of **scaling laws** — the empirical power-law relationship between loss, parameters, data, and compute (Kaplan 2020; Chinchilla 2022). That is a section of its own in this track (forward reference: *Pretraining & Scaling Laws*). For now, the one fact to carry: **loss falls predictably as a power law in compute**, and capability rides on top of falling loss." },
      ]
    },

    {
      id: "training-pipeline",
      title: "The training pipeline: base → instruct → chat",
      level: "core",
      body: [
        { type: "p", text: "The ChatGPT you talk to is not a raw language model — it is a raw model that has been through a **multi-stage pipeline**. Understanding the stages explains most of the difference between \"a model that completes text\" and \"an assistant that answers you.\" Each stage gets a full treatment in the *Fine-tuning* section; here is the map." },
        { type: "table",
          headers: ["Stage", "Data", "What it produces", "Cost"],
          rows: [
            ["**1. Pretraining**", "trillions of tokens of raw web/text", "a **base model** (a raw next-token predictor)", "months, $$$M, most of the compute"],
            ["**2. SFT**", "curated (prompt, ideal response) pairs", "an **instruct model** that follows instructions", "days, moderate"],
            ["**3. Alignment (RLHF/DPO)**", "human preference comparisons", "a **chat model** tuned to be helpful/harmless", "days, moderate"],
          ]
        },
        { type: "heading", text: "1. Pretraining" },
        { type: "p", text: "Next-token prediction (section 2) over a giant corpus. This is where essentially *all* the knowledge and capability is acquired, and where ~99% of the compute is spent. The result is a **base model**: it can continue text brilliantly but does not \"answer questions\" — give it *\"What is the capital of France?\"* and it might reply with *more questions*, because on the internet that string often appears in a list of quiz questions. It completes; it does not assist." },
        { type: "heading", text: "2. Supervised fine-tuning (SFT)" },
        { type: "p", text: "Continue training the base model on a curated dataset of **(instruction, high-quality response)** pairs written or vetted by humans. Same next-token loss, but now on demonstrations of *being a helpful assistant*. This is what teaches the model the **format** of a response: read an instruction, produce a direct, helpful answer. The output is an **instruct** (or **-it**) model." },
        { type: "heading", text: "3. Preference alignment: RLHF and DPO" },
        { type: "p", text: "SFT teaches *a* good answer; alignment teaches *which of two answers humans prefer*. **RLHF** (Reinforcement Learning from Human Feedback) trains a reward model on human A-vs-B comparisons, then optimizes the LLM against that reward with PPO. **DPO** (Direct Preference Optimization, 2023) reaches a similar place with a simpler, RL-free classification-style loss directly on the preference pairs — which is why much of the field moved to it. Both are derived in the *Fine-tuning* section (forward reference)." },
        { type: "callout", variant: "note", text: "**Base vs Instruct/Chat — pick the right one.** Repos ship both, e.g. `Llama-3.1-8B` (base) and `Llama-3.1-8B-Instruct` (chat). Use the **Instruct/Chat** variant for anything conversational or instruction-following — it is what you almost always want. Use the **base** model only when you plan to fine-tune it yourself or want raw completion. Prompting a base model like a chatbot gives frustrating, rambly results." },
        { type: "heading", text: "Chat templates — the hidden formatting layer" },
        { type: "p", text: "Chat models are trained on a specific **template** with special tokens delimiting roles (system / user / assistant). Your nice `messages=[...]` list is rendered into one flat string *in exactly that format* before hitting the model — and using the wrong template noticeably degrades quality. Different families use different templates (ChatML, Llama's, etc.), which is why you must apply the model's own:" },
        { type: "code", lang: "py", code: "from transformers import AutoTokenizer\ntok = AutoTokenizer.from_pretrained(\"Qwen/Qwen2.5-7B-Instruct\")\n\nmessages = [\n    {\"role\": \"system\",    \"content\": \"You are a helpful assistant.\"},\n    {\"role\": \"user\",      \"content\": \"Name one prime number.\"},\n]\n# The tokenizer knows THIS model's chat template — always use it:\nprompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\nprint(prompt)\n# <|im_start|>system\\nYou are a helpful assistant.<|im_end|>\n# <|im_start|>user\\nName one prime number.<|im_end|>\n# <|im_start|>assistant\\n        <- add_generation_prompt cues the model to reply" },
        { type: "callout", variant: "gotcha", text: "Never hand-format chat prompts as plain strings for an instruct model. Call `apply_chat_template` (or use an API's `messages` interface, which does it for you). A mismatched or missing template is one of the most common reasons a \"good\" model gives bad answers locally." },
      ]
    },

    {
      id: "limits",
      title: "Capabilities and hard limits",
      level: "core",
      body: [
        { type: "p", text: "LLMs are genuinely remarkable and genuinely limited, and the failures are not random — they follow directly from the mechanics in this section. Knowing *why* each limit exists is what separates someone who uses LLMs well from someone who trusts them blindly." },
        { type: "heading", text: "Hallucination" },
        { type: "p", text: "An LLM will state false things **fluently and confidently**. This is not a bug to be fully patched — it is intrinsic to the objective. The model is trained to produce *plausible* continuations, not *true* ones; \"the most likely next token\" and \"the truth\" usually coincide but are not the same target. When the model lacks the fact, it generates something that *sounds* like the fact's shape. **Never trust an unverified factual claim, citation, or quote from an LLM.**" },
        { type: "heading", text: "Knowledge cutoff" },
        { type: "p", text: "A model only knows what was in its pretraining data, which ends at a **knowledge cutoff** date. It has no awareness of events after that, and — because it does not know what it does not know — it will often confabulate rather than say \"I can't know that.\" This is the core motivation for **RAG** (retrieval-augmented generation) and tool/web access: inject fresh, authoritative context at inference time (covered in the RAG section)." },
        { type: "heading", text: "Context limits & \"lost in the middle\"" },
        { type: "p", text: "The context window (section 3) is a hard ceiling: text beyond it simply does not exist to the model. Even *within* the window, models attend unevenly — information at the **start and end** of a long context is used more reliably than information buried in the **middle** (the \"lost in the middle\" effect). A 1M-token window does not guarantee 1M tokens of *effective* recall." },
        { type: "heading", text: "No guaranteed reasoning" },
        { type: "p", text: "An LLM does not execute a verified algorithm; it pattern-matches plausible token sequences. Chain-of-thought *helps* (section 6) but does not *guarantee* correctness — the reasoning trace can be fluent and wrong, and the model can even produce a correct-looking derivation with a false conclusion. For anything requiring guaranteed correctness (arithmetic, dates, lookups), **give the model tools** (a calculator, code execution, a database) rather than trusting its internal computation." },
        { type: "heading", text: "Tokenization artifacts" },
        { type: "p", text: "As shown in section 3, the model sees tokens, not characters — so character-level tasks (counting letters, reversing strings, some spelling) and digit-level arithmetic are systematically shaky, independent of how \"smart\" the model is. Recognizing a tokenization failure saves you from concluding a model is dumb when it is merely blind at the character level." },
        { type: "callout", variant: "warn", text: "**The mental model that keeps you safe:** an LLM is a fluent, well-read, extremely fast intern with no memory of anything after its cutoff, no calculator, a tendency to make up citations rather than admit ignorance, and total confidence at all times. You would verify that intern's factual claims and give them tools for exact work. Do exactly that with an LLM." },
        { type: "table",
          headers: ["Limit", "Root cause", "Mitigation"],
          rows: [
            ["Hallucination", "trained for plausibility, not truth", "RAG, citations, verification, tools"],
            ["Knowledge cutoff", "static pretraining data", "web/tool access, retrieval"],
            ["Context ceiling / mid-loss", "$O(n^2)$ attention; uneven use", "chunking, put key info at the ends"],
            ["Unreliable reasoning", "pattern-matching, not execution", "CoT, tool use, code execution, self-check"],
            ["Char/digit errors", "sub-word tokenization", "tools for exact string/number work"],
          ]
        },
      ]
    },

    {
      id: "landscape",
      title: "The 2025-26 landscape: open-weight vs closed",
      level: "core",
      body: [
        { type: "p", text: "The ecosystem splits into two camps. **Closed / API models** are served only through an endpoint — you never see the weights, you get top capability and zero ops burden, and you pay per token. **Open-weight models** publish their weights for download — you can run them locally, fine-tune them, and inspect them, at the cost of running your own infrastructure. Both matter, and the frontier gap between them has narrowed sharply." },
        { type: "callout", variant: "warn", text: "**This landscape moves faster than any other topic in this deck.** Specific version numbers below are a snapshot from late-2025/early-2026 and *will* be stale — treat them as an illustration of the shape of the field, not a live leaderboard. For current rankings check the live sources in the references. What *won't* go stale is the taxonomy and the how-to-choose logic." },
        { type: "heading", text: "Closed / API frontier" },
        { type: "table",
          headers: ["Family", "Maker", "Notable for"],
          rows: [
            ["**GPT** (GPT-4o, GPT-5 era)", "OpenAI", "broad capability, ecosystem, tool use"],
            ["**Claude** (Sonnet, Opus 4.x)", "Anthropic", "coding, long agentic tasks, long context"],
            ["**Gemini** (2.x Pro/Flash)", "Google", "native multimodal, very long context"],
          ]
        },
        { type: "heading", text: "Open-weight families" },
        { type: "table",
          headers: ["Family", "Maker", "Typical sizes", "License"],
          rows: [
            ["**Llama** (3.x, 4)", "Meta", "8B-405B; Llama 4 MoE (Scout/Maverick)", "Llama Community (near-open)"],
            ["**Qwen** (2.5, 3)", "Alibaba", "0.5B-235B, dense + MoE", "Apache 2.0 (permissive)"],
            ["**Mistral** (7B, Small, Large)", "Mistral AI", "7B-100s of B, strong MoE (Mixtral)", "Apache 2.0 (mostly)"],
            ["**Gemma** (2, 3)", "Google", "1B-27B, strong small models", "Gemma (permissive-ish)"],
            ["**DeepSeek** (V3, R1)", "DeepSeek", "large MoE; R1 = open reasoning model", "MIT (very permissive)"],
          ]
        },
        { type: "heading", text: "How to choose" },
        { type: "list", ordered: false, items: [
          "**Prototyping / max quality / no ops:** start with a closed API (GPT/Claude/Gemini). Fastest path to a working product; you pay per token.",
          "**Privacy, offline, or high volume:** open-weight, self-hosted. Your data never leaves your machine, and at scale it can be far cheaper than per-token API pricing.",
          "**Fine-tuning on your own data:** open-weight is the natural home (LoRA/QLoRA on a Llama/Qwen/Mistral base — see the Fine-tuning section).",
          "**On a laptop / edge:** a small strong model — Gemma 3 4B, Qwen3 4-8B, Llama 3.x 8B — quantized to 4-bit via Ollama or llama.cpp (section 10).",
          "**Reasoning-heavy tasks:** a reasoning-tuned model (DeepSeek-R1 open, or the closed o-series/Gemini-thinking variants).",
        ]},
        { type: "callout", variant: "gotcha", text: "**\"Open-weight\" is not the same as \"open-source.\"** Most released models publish *weights* but not the training data or full training code, and some licenses carry restrictions (e.g. Llama's community license has an acceptable-use policy and a large-company clause). Truly permissive licenses — **Apache 2.0** (Qwen, Mistral) and **MIT** (DeepSeek) — let you do essentially anything commercially. **Always read the license before shipping.**" },
      ]
    },

    {
      id: "hands-on",
      title: "A first hands-on: local and via API",
      level: "core",
      body: [
        { type: "p", text: "Enough theory — let us actually run a model. Three ways, from most-local to most-hosted. All three do the same thing: build a prompt, run the autoregressive loop, decode tokens." },
        { type: "heading", text: "1. HuggingFace transformers (a small open model, in Python)" },
        { type: "p", text: "The `pipeline` API hides the tokenize → generate → decode loop behind one call. This downloads a small instruct model and runs it on your CPU or GPU:" },
        { type: "code", lang: "py", code: "# pip install \"transformers>=4.44\" torch\nfrom transformers import pipeline\n\ngen = pipeline(\n    \"text-generation\",\n    model=\"Qwen/Qwen2.5-0.5B-Instruct\",   # ~1 GB, runs on a laptop CPU\n    device_map=\"auto\",\n)\n\nmessages = [{\"role\": \"user\", \"content\": \"Explain what a token is in one sentence.\"}]\nout = gen(\n    messages,                 # pipeline applies the chat template for you\n    max_new_tokens=64,\n    do_sample=True,\n    temperature=0.7,\n    top_p=0.9,\n)\nprint(out[0][\"generated_text\"][-1][\"content\"])" },
        { type: "callout", variant: "tip", text: "Start with a **tiny** model (0.5B-1B) to get the plumbing working, then scale up. A 0.5B model is not smart, but it proves your environment runs. Only jump to 7B+ once the pipeline works, and quantize (`load_in_4bit` via `bitsandbytes`) if you are short on VRAM." },
        { type: "heading", text: "2. Ollama (the easiest local path — no Python)" },
        { type: "p", text: "**Ollama** wraps `llama.cpp` and makes running quantized open models a one-liner. It handles download, quantization, chat templating, and a local server. This is the fastest way to get a capable model chatting on your own machine:" },
        { type: "code", lang: "bash", code: "# Install from https://ollama.com, then:\nollama run llama3.2         # pulls a quantized Llama 3.2 and starts a chat\n\n# It also serves an OpenAI-compatible API on localhost:11434\ncurl http://localhost:11434/api/generate -d '{\n  \"model\": \"llama3.2\",\n  \"prompt\": \"Name one prime number.\",\n  \"stream\": false\n}'" },
        { type: "heading", text: "3. Via a hosted API (OpenAI-compatible)" },
        { type: "p", text: "For frontier quality with zero ops, call a hosted endpoint. The `messages` interface handles chat templating server-side; the decoding knobs from section 5 are the same parameters. Note this exact client shape (base URL + key) also works against Ollama, vLLM, and most providers — the OpenAI schema is the de-facto standard:" },
        { type: "code", lang: "py", code: "# pip install openai   (works for OpenAI, and any OpenAI-compatible endpoint)\nfrom openai import OpenAI\n\nclient = OpenAI()   # reads OPENAI_API_KEY; set base_url=... to point elsewhere\n\nresp = client.chat.completions.create(\n    model=\"gpt-4o-mini\",\n    messages=[\n        {\"role\": \"system\", \"content\": \"You are concise.\"},\n        {\"role\": \"user\",   \"content\": \"Explain what a token is in one sentence.\"},\n    ],\n    temperature=0.7,\n    top_p=0.9,\n    max_tokens=64,\n)\nprint(resp.choices[0].message.content)\nprint(resp.usage)   # prompt_tokens, completion_tokens — this is what you pay for" },
        { type: "callout", variant: "note", text: "Notice the three interfaces converge on the same shape: a `messages` list, decoding parameters (`temperature`, `top_p`, `max_tokens`), and token usage in the response. Learn this shape once and it transfers across every provider and every local runtime. If you are calling a Claude/Anthropic endpoint, load the `claude-api` skill for that SDK's exact parameters and model IDs." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "You learn LLMs by *running the loop yourself*. Do project 1 and 2 no matter what — they make the autoregressive loop and decoding knobs concrete in a way no amount of reading can." },
        { type: "list", ordered: true, items: [
          "**Implement the generation loop by hand.** Load a small model via `transformers` with `output_scores`, and instead of `.generate()`, write the autoregressive loop yourself: forward pass → take last-position logits → apply *your own* temperature+top-p sampler (section 5) → append the token → repeat until `<eos>`. This one exercise demystifies inference completely.",
          "**A decoding-strategy playground.** Fix a prompt and generate with greedy, then `temperature` ∈ {0.3, 0.7, 1.0, 1.5}, then `top_p` ∈ {0.5, 0.9, 1.0}. Print the outputs side by side. *Feel* how each knob changes coherence vs. diversity — this intuition is worth more than any table.",
          "**Tokenizer autopsy.** Load 3 different tokenizers (GPT-2, Llama, Qwen) and tokenize the same paragraph, some numbers, an emoji, and a rare word. Compare token counts and splits. Then reproduce the \"how many r's in strawberry\" failure and explain it from the token IDs.",
          "**Base vs Instruct, head to head.** Load `Llama-3.2-1B` and `Llama-3.2-1B-Instruct`. Give both the same question with and without a chat template. Watch the base model ramble and the instruct model answer — the SFT/alignment effect (section 7) made visible.",
          "**Few-shot in-context learning.** Pick a made-up task (e.g. translate to pig-latin, or a custom label format). Show the model 0, 1, 3, and 5 examples in the prompt and measure accuracy climbing. You just reproduced GPT-3's headline result on your laptop.",
          "**A tiny local chatbot.** Wire Ollama (or the API client) into a loop that keeps a running `messages` list, so the assistant remembers the conversation. Then watch it break when the history exceeds the context window — and add truncation. You have now met the core engineering problem of every chat app.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "This section is the map of the LLM track; these are the best companions for going deeper, roughly in order of accessibility. The Karpathy talk is the ideal thing to watch immediately after this section." },
        { type: "link", url: "https://www.youtube.com/watch?v=zjkBMFhNj_g", text: "Andrej Karpathy — Intro to Large Language Models (the perfect 1-hour overview to watch right after this section)" },
        { type: "link", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", text: "Andrej Karpathy — Let's build GPT: from scratch (build the decoder-only model from this section, line by line)" },
        { type: "link", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", text: "Radford et al. (2019) — Language Models are Unsupervised Multitask Learners (the GPT-2 paper)" },
        { type: "link", url: "https://arxiv.org/abs/2005.14165", text: "Brown et al. (2020) — Language Models are Few-Shot Learners (GPT-3; the in-context learning result)" },
        { type: "link", url: "https://arxiv.org/abs/1904.09751", text: "Holtzman et al. (2019) — The Curious Case of Neural Text Degeneration (introduces nucleus / top-p sampling)" },
        { type: "link", url: "https://huggingface.co/learn/llm-course", text: "Hugging Face — the LLM Course (hands-on with transformers, the practical companion to this track)" },
        { type: "link", url: "https://artificialanalysis.ai/leaderboards/models", text: "Artificial Analysis — live LLM leaderboard (use this for the *current* model landscape, since section 9 dates fast)" },
      ]
    },
  ],

  packages: [
    { name: "transformers", why: "HuggingFace — load any open model + tokenizer; `pipeline`, `generate`, chat templates" },
    { name: "torch", why: "the tensor/autograd backend transformers runs on" },
    { name: "tokenizers / tiktoken", why: "fast BPE tokenization; `tiktoken` for counting GPT-family tokens" },
    { name: "ollama", why: "one-command local inference for quantized open models (wraps llama.cpp)" },
    { name: "openai", why: "the de-facto client shape — works for OpenAI and any OpenAI-compatible endpoint" },
    { name: "vllm", why: "high-throughput local serving with paged KV-cache (production inference)" },
    { name: "accelerate / bitsandbytes", why: "device mapping and 4-bit quantization to fit big models on small GPUs" },
    { name: "huggingface_hub", why: "download weights/tokenizers and manage the local model cache" },
  ],

  gotchas: [
    "LLMs optimize for **plausibility, not truth** — they hallucinate confidently. Verify every factual claim, citation, and quote.",
    "The model sees **tokens, not characters** — letter-counting, string reversal, and digit arithmetic fail for tokenization reasons, not stupidity.",
    "Use the **Instruct/Chat** variant for conversation and `apply_chat_template`; prompting a **base** model like a chatbot gives rambly garbage.",
    "Decoding knobs are not the model — change **one at a time**. Cranking `temperature` and `top_p` together yields incoherent output.",
    "Beam search is wrong for open-ended chat — maximizing sequence probability produces bland, repetitive text. Use sampling (top-p) instead.",
    "The **context window** is prompt + output combined, a hard ceiling; and models recall the **middle** of a long context worse than the ends.",
    "**Open-weight ≠ open-source**, and licenses differ — Apache 2.0 (Qwen/Mistral) and MIT (DeepSeek) are permissive; read the license before shipping.",
    "`fp16` weights cost **~2 bytes/param** (7B ≈ 14 GB VRAM); quantize to 4-bit (~4x smaller) to run locally.",
  ],

  flashcards: [
    { q: "In one sentence, what is an LLM?", a: "A decoder-only transformer that models $p_\\theta(x_t \\mid x_{<t})$ — the next-token distribution — fit by maximizing the likelihood of a huge text corpus; you generate by sampling from it repeatedly." },
    { q: "State the autoregressive factorization.", a: "$p(x_1,\\dots,x_T) = \\prod_{t=1}^{T} p(x_t \\mid x_{<t})$ — the chain rule of probability. An LLM parameterizes the single factor $p_\\theta(x_t \\mid x_{<t})$ and applies it at every position." },
    { q: "Why does the causal mask matter for training efficiency?", a: "It lets one forward pass over a length-$T$ sequence produce all $T$ next-token predictions at once without any position seeing its future — $T$ loss terms per pass." },
    { q: "What does temperature $T$ do to the next-token distribution?", a: "Divides logits by $T$ before softmax: $T\\to0$ sharpens toward greedy, $T\\to\\infty$ flattens toward uniform, $T=1$ is the model's raw distribution." },
    { q: "How does top-p (nucleus) sampling differ from top-k?", a: "Top-k keeps a fixed number of tokens; top-p keeps the smallest set whose cumulative probability exceeds $p$ — an *adaptive* set that shrinks when the model is confident and grows when uncertain." },
    { q: "What is in-context (few-shot) learning?", a: "The model performs a new task from examples placed **in the prompt**, with no weight updates — an emergent ability of scale (GPT-3). Prompting conditions the distribution; it does not train the model." },
    { q: "Name the three training stages and what each produces.", a: "Pretraining → base model (raw next-token predictor); SFT → instruct model (follows instructions); RLHF/DPO → chat model (aligned to human preferences)." },
    { q: "Why do LLMs hallucinate?", a: "They are trained to produce *plausible* continuations, not *true* ones. When the fact is absent, the model generates something shaped like the fact. Truth and 'most likely next token' are different targets." },
    { q: "Base model vs Instruct model — which do you use for chat?", a: "The **Instruct/Chat** variant, with the model's chat template. A base model completes text but doesn't 'answer'; prompting it conversationally gives poor results." },
    { q: "What is a Mixture-of-Experts (MoE) model?", a: "A model with huge *total* parameters that routes each token to only a few 'expert' FFNs, so *active* params per token are far fewer — big capacity at small-model inference cost (DeepSeek-V3, Llama 4, Qwen3-235B)." },
    { q: "Why are character-level tasks (counting letters, reversing) hard for LLMs?", a: "The model sees sub-word **tokens**, not characters ('strawberry' = 'straw'+'berry'), so it has no direct view of individual letters or consistent digit boundaries." },
    { q: "Open-weight vs open-source — what's the distinction?", a: "Open-weight publishes the weights (runnable, fine-tunable) but usually not the training data/code, and licenses vary. Apache 2.0/MIT are permissive; some (Llama) add usage restrictions." },
  ],

  cheatsheet: [
    { label: "Next-token dist", code: "logits = model(ids); p = softmax(logits[-1])" },
    { label: "Autoregressive law", code: "p(x) = prod_t p(x_t | x_<t)" },
    { label: "Count tokens (GPT)", code: "tiktoken.encoding_for_model('gpt-4o').encode(text)" },
    { label: "Load model+tokenizer", code: "AutoModelForCausalLM.from_pretrained(id); AutoTokenizer..." },
    { label: "Chat template", code: "tok.apply_chat_template(messages, add_generation_prompt=True)" },
    { label: "Generate (sampling)", code: "model.generate(**x, do_sample=True, temperature=0.7, top_p=0.9)" },
    { label: "Generate (greedy)", code: "model.generate(**x, do_sample=False)" },
    { label: "Quick pipeline", code: "pipeline('text-generation', model=id)(messages)" },
    { label: "Run local model", code: "ollama run llama3.2" },
    { label: "API call", code: "client.chat.completions.create(model=..., messages=...)" },
    { label: "4-bit load", code: "from_pretrained(id, load_in_4bit=True)" },
    { label: "VRAM rule (fp16)", code: "~2 bytes/param  (7B ~ 14 GB)" },
    { label: "Token rule of thumb", code: "1 token ~ 4 chars ~ 0.75 words" },
  ],
});
