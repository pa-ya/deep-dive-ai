(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "llm-using",
  name: "Using Open & Hosted LLMs",
  language: "LLMs",
  group: "Large Language Models",
  navLabel: "Using LLMs (local & API)",
  tagline: "Actually *run* the models — Ollama and llama.cpp on your own GPU, HuggingFace `transformers` from scratch, hosted APIs, decoding controls, quantization, and a streaming chatbot you build yourself.",
  color: "#E11D48",
  readMinutes: 48,
  sections: [
    {
      id: "options-map",
      title: "The options map: hosted APIs vs open weights you run yourself",
      level: "core",
      body: [
        { type: "p", text: "You understand *how* a transformer generates text (see the LLM-intro topic). Now you need to **make one produce tokens on demand**. There are two worlds, and a professional should be fluent in both:" },
        { type: "table",
          headers: ["", "Hosted API", "Open weights, run locally"],
          rows: [
            ["Examples", "OpenAI GPT-4o/o-series, Anthropic Claude, Google Gemini", "Llama 3.3, Qwen2.5/Qwen3, Gemma 3, Mistral, Phi-4, DeepSeek-R1, gpt-oss"],
            ["You send", "text over HTTPS to someone else's GPU", "text to a process on **your** machine"],
            ["Cost model", "**per token** (pay as you go)", "**per hour of hardware** (electricity + the GPU you own/rent)"],
            ["Privacy", "data leaves your machine", "data never leaves; works fully offline"],
            ["Control", "the provider's knobs only", "every weight, every sampler, fine-tuning, LoRA"],
            ["Best when", "you want the strongest model with zero ops", "privacy, cost-at-scale, offline, or you want to *learn*"],
          ]
        },
        { type: "p", text: "This topic is deliberately biased toward the **open-weights, run-it-yourself** path, because that is where you learn the most and pay the least — but the final skill (calling an OpenAI-compatible endpoint) is *identical* for a hosted API and for your own local server, so you get both for the price of one." },
        { type: "callout", variant: "note", text: "**\"Open weights\" ≠ \"open source.\"** Llama, Gemma, and Qwen ship the trained *parameters* under a license, but not always the training data or a fully OSI-compliant license. You can download, run, quantize, and fine-tune them. That is what matters here. Truly open-source-license models exist too (OLMo, gpt-oss is Apache-2.0), but the practical workflow is the same." },
        { type: "heading", text: "The three tradeoffs, quantified" },
        { type: "list", ordered: false, items: [
          "**Cost.** A hosted frontier model might cost a few dollars per million tokens. A 7–8B open model on a GPU you already own costs ~nothing per token after the electricity. At high volume, or for background/batch work, local wins by orders of magnitude. For occasional access to the single strongest model, hosted wins.",
          "**Privacy.** Regulated data (health, legal, internal code) often *cannot* leave your network. Local inference is the only option that is auditable and airgappable. This alone decides the choice for many companies.",
          "**Control.** Locally you choose the exact model, the quantization, the sampler, the system prompt, and you can fine-tune. You can pin a version forever. Hosted models change under you and can be deprecated.",
        ]},
        { type: "callout", variant: "tip", text: "**The pragmatic default for a developer learning ML:** install **Ollama**, pull an 8B model, and you have a private ChatGPT-like assistant and an OpenAI-compatible API on `localhost` in about five minutes. Everything else in this topic builds on that. We start there." },
      ]
    },

    {
      id: "running-locally",
      title: "Running locally: Ollama, llama.cpp, LM Studio, vLLM",
      level: "core",
      body: [
        { type: "p", text: "There are four tools you should know, arranged from *easiest* to *most production-grade*. They all ultimately run the same open weights; they differ in packaging, speed, and how many concurrent users they serve." },
        { type: "heading", text: "Ollama — the easiest on-ramp" },
        { type: "p", text: "**Ollama** wraps `llama.cpp` behind a one-line CLI and a background server. It downloads a quantized model, manages a local model registry, and exposes both its own API and an **OpenAI-compatible** endpoint on `http://localhost:11434`. This is the fastest way to a working local LLM." },
        { type: "code", lang: "bash", code: "# Install (macOS/Linux): see ollama.com/download\n# Then pull-and-run a model. First run downloads the weights (a few GB).\nollama run llama3.2:3b          # tiny, runs on almost anything (CPU ok)\nollama run llama3.1:8b          # the reliable default on an 8GB+ GPU\nollama run qwen2.5:14b          # stronger, needs ~10-12GB VRAM at Q4\n\n# List what you have, remove one, show model details:\nollama list\nollama show llama3.1:8b\nollama rm qwen2.5:14b\n\n# Ollama is also a server the moment it's installed:\ncurl http://localhost:11434/api/generate -d '{\n  \"model\": \"llama3.1:8b\",\n  \"prompt\": \"Explain a transformer in one sentence.\",\n  \"stream\": false\n}'" },
        { type: "callout", variant: "tip", text: "The tag after the colon (`:8b`, `:3b`, `:14b`) is the **parameter count**, and by default Ollama pulls a `Q4_K_M` (4-bit) quant. `ollama run llama3.1:8b-instruct-q8_0` picks a specific quant. Bigger number = smarter but slower and more VRAM. Start small; size up until it stops fitting." },
        { type: "heading", text: "llama.cpp + GGUF — the engine underneath" },
        { type: "p", text: "**llama.cpp** is the C/C++ inference engine that started the local-LLM revolution. It runs models in the **GGUF** file format on CPU, GPU (CUDA/Metal/Vulkan/ROCm), or a split of both. Ollama and LM Studio are convenience layers *over* llama.cpp. Using it directly gives you maximum control and is worth doing once." },
        { type: "code", lang: "bash", code: "# Build (needs a C++ toolchain; add -DGGML_CUDA=ON for NVIDIA GPUs)\ngit clone https://github.com/ggml-org/llama.cpp && cd llama.cpp\ncmake -B build && cmake --build build -j\n\n# Download a GGUF from HuggingFace (search '<model> GGUF' — e.g. bartowski/TheBloke)\n# then run interactive chat, offloading 35 layers to the GPU:\n./build/bin/llama-cli -m ./models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \\\n    --n-gpu-layers 35 --ctx-size 8192 -p \"Hello\"\n\n# Or start an OpenAI-compatible server on :8080\n./build/bin/llama-server -m ./models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \\\n    --n-gpu-layers 35 --ctx-size 8192 --host 0.0.0.0 --port 8080" },
        { type: "p", text: "**GGUF quantization levels** decode as `Q<bits>_<K><size>`. The K-quants use *mixed* precision — sensitive layers (attention, embeddings) get more bits than the feed-forward layers — which is why they beat naive uniform quantization. The one you will use 90% of the time is **`Q4_K_M`**." },
        { type: "table",
          headers: ["Quant", "Bits/weight (approx)", "Quality vs FP16", "When to use"],
          rows: [
            ["`Q8_0`", "~8.5", "≈ lossless", "when you have VRAM to spare"],
            ["`Q6_K`", "~6.6", "near-lossless", "quality-sensitive, still compact"],
            ["`Q5_K_M`", "~5.7", "very good", "a notch above the default"],
            ["`Q4_K_M`", "~4.8", "1–3% loss, **the standard**", "the default balance — pick this first"],
            ["`Q3_K_M`", "~3.9", "noticeable loss", "squeezing a bigger model into small VRAM"],
            ["`Q2_K`", "~3.4", "degraded", "last resort to fit at all"],
          ]
        },
        { type: "heading", text: "LM Studio — a desktop GUI" },
        { type: "p", text: "**LM Studio** is a polished desktop app (Windows/macOS/Linux) that searches HuggingFace, downloads GGUFs with a click, gives you a chat window, and can start the same OpenAI-compatible local server. It is the friendliest option for non-CLI users and for browsing which models fit your hardware — it warns you before you download something too big for your VRAM." },
        { type: "heading", text: "vLLM — production serving" },
        { type: "p", text: "**vLLM** is the serious *serving* engine. It runs the **full-precision** (or AWQ/GPTQ/FP8-quantized) HuggingFace weights on GPUs and is built for **throughput**: its *PagedAttention* manages the KV-cache like virtual memory, so it can batch many concurrent requests and serve them far faster than llama.cpp when a GPU is saturated. It exposes an OpenAI-compatible server out of the box." },
        { type: "code", lang: "bash", code: "pip install vllm\n\n# Serve a model with an OpenAI-compatible API on :8000\nvllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto\n\n# Serve a pre-quantized model to fit a smaller GPU:\nvllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --quantization awq\n\n# Spread a big model across 2 GPUs:\nvllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2" },
        { type: "callout", variant: "note", text: "**Which do I use?** *Ollama/LM Studio* for personal use and prototyping (one user, easy). *llama.cpp* directly when you want CPU/low-VRAM flexibility and full control. *vLLM* when you are serving an app to many users on real GPUs and need throughput. All four can speak the OpenAI API, so your client code (Section 6) never has to change." },
        { type: "heading", text: "Hardware: how big a model can I run?" },
        { type: "p", text: "The dominant cost is holding the **weights** in memory. A rough rule: memory for weights ≈ (parameters) × (bytes per parameter). FP16 is 2 bytes; a 4-bit quant is ~0.5–0.6 bytes. On top of that you pay for the **KV-cache** (grows with context length — Section 7) and some overhead." },
        { type: "table",
          headers: ["Model size", "FP16 weights", "Q4 (GGUF/AWQ)", "Fits on"],
          rows: [
            ["3B", "~6 GB", "~2 GB", "any modern laptop, even CPU"],
            ["7–8B", "~15 GB", "~5 GB", "8GB GPU (RTX 3060/4060), or CPU slowly"],
            ["13–14B", "~28 GB", "~9 GB", "12GB GPU"],
            ["32–34B", "~68 GB", "~20 GB", "24GB GPU (RTX 3090/4090)"],
            ["70B", "~140 GB", "~40 GB", "2×24GB, or 1×48GB (A6000), or offload+slow"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**No GPU? You can still run everything here.** llama.cpp and Ollama run on CPU + system RAM — a 7B Q4 model needs ~6GB of RAM and generates a handful of tokens/second on a modern laptop. Slow but real. A GPU mainly buys you *speed* and the ability to hold *bigger* models. Start on whatever you have." },
      ]
    },

    {
      id: "transformers",
      title: "HuggingFace Transformers: generate text from scratch",
      level: "core",
      body: [
        { type: "p", text: "Ollama hides the model behind a server. **HuggingFace `transformers`** is the library that lets you load the raw weights *in your own Python process* and drive every step — tokenize, forward pass, sample, decode. This is the layer you must understand to fine-tune, inspect activations, or do anything non-standard. It is also how nearly all research code is written." },
        { type: "heading", text: "The three objects: tokenizer, model, config" },
        { type: "p", text: "Every model on the HuggingFace Hub is loaded by its **repo id** (e.g. `meta-llama/Llama-3.1-8B-Instruct`). `AutoTokenizer` and `AutoModelForCausalLM` read the repo's config and instantiate the right classes automatically — you rarely name a specific architecture." },
        { type: "code", lang: "py", code: "import torch\nfrom transformers import AutoTokenizer, AutoModelForCausalLM\n\nmodel_id = \"Qwen/Qwen2.5-1.5B-Instruct\"   # small enough for most machines\n\ntok = AutoTokenizer.from_pretrained(model_id)\nmodel = AutoModelForCausalLM.from_pretrained(\n    model_id,\n    torch_dtype=\"auto\",   # use the dtype the weights were saved in (bf16/fp16)\n    device_map=\"auto\",    # place layers on GPU if present, else CPU\n)\n\nprint(model.config.num_hidden_layers, \"layers\",\n      model.num_parameters() / 1e9, \"B params\")" },
        { type: "callout", variant: "tip", text: "**`device_map=\"auto\"`** uses the `accelerate` library to spread the model across whatever devices exist — one GPU, several GPUs, or GPU+CPU+disk offload for a model too big to fit. **`torch_dtype=\"auto\"`** avoids the classic beginner mistake of loading bf16 weights into fp32 and using 2× the memory. Install with `pip install transformers accelerate torch`." },
        { type: "heading", text: "Tokenize → generate → decode, the long way" },
        { type: "p", text: "`generate()` is the autoregressive loop from the LLM-intro topic, implemented for you: it runs the forward pass, samples the next token, appends it, and repeats until it hits a stop condition. You feed it token **ids** and get token ids back." },
        { type: "code", lang: "py", code: "prompt = \"The three laws of thermodynamics are:\"\ninputs = tok(prompt, return_tensors=\"pt\").to(model.device)\n\nwith torch.no_grad():\n    out_ids = model.generate(\n        **inputs,\n        max_new_tokens=120,\n        do_sample=True,        # sample instead of greedy argmax\n        temperature=0.7,\n        top_p=0.9,\n    )\n\n# out_ids includes the prompt; slice it off to get only the new tokens\nnew_tokens = out_ids[0, inputs[\"input_ids\"].shape[1]:]\nprint(tok.decode(new_tokens, skip_special_tokens=True))" },
        { type: "callout", variant: "gotcha", text: "**`generate()` returns the prompt tokens too.** If you decode `out_ids[0]` directly you will see your prompt echoed back. Slice off the first `input_ids.shape[1]` tokens (as above) to get only the completion. This trips up everyone once." },
        { type: "heading", text: "The one-liner: `pipeline`" },
        { type: "p", text: "For quick use, `pipeline` wraps tokenizer + model + decoding into a single callable. It even applies the chat template (Section 5) for you when you pass a list of messages." },
        { type: "code", lang: "py", code: "from transformers import pipeline\n\npipe = pipeline(\"text-generation\", model=\"Qwen/Qwen2.5-1.5B-Instruct\",\n                torch_dtype=\"auto\", device_map=\"auto\")\n\nmessages = [\n    {\"role\": \"system\", \"content\": \"You are a terse assistant.\"},\n    {\"role\": \"user\", \"content\": \"Name three sorting algorithms.\"},\n]\nout = pipe(messages, max_new_tokens=64, do_sample=True, temperature=0.7)\nprint(out[0][\"generated_text\"][-1][\"content\"])   # the assistant's reply" },
        { type: "callout", variant: "note", text: "**`transformers` vs `llama.cpp`.** `transformers` runs the *original* PyTorch weights (fp16/bf16, or bitsandbytes-quantized — Section 7). `llama.cpp` runs *GGUF* files. Same model, two ecosystems. Use `transformers` when you want Python-level control, training, or research; use llama.cpp/Ollama when you want a fast, low-VRAM, deploy-anywhere binary." },
      ]
    },

    {
      id: "decoding",
      title: "Decoding controls: turning logits into text you like",
      level: "core",
      body: [
        { type: "p", text: "At each step the model outputs a **logit** for every token in the vocabulary. *How* you turn that distribution into one chosen token is **decoding**, and it changes the output character more than almost anything else. These are the same knobs whether you use `generate()`, Ollama, or a hosted API — only the names shift slightly. (The math of the softmax and sampling lives in the LLM-intro topic; here we build intuition for the effect.)" },
        { type: "heading", text: "Temperature — the creativity dial" },
        { type: "p", text: "Temperature $T$ divides the logits before the softmax: $p_i = \\mathrm{softmax}(z_i / T)$. $T<1$ sharpens the distribution (safer, more repetitive); $T>1$ flattens it (more surprising, more mistakes); $T\\to 0$ becomes greedy argmax — deterministic. Set `do_sample=True` or temperature is ignored." },
        { type: "math", tex: String.raw`p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} \qquad T \downarrow \Rightarrow \text{peaky, safe} \qquad T \uparrow \Rightarrow \text{flat, wild}` },
        { type: "heading", text: "top-p (nucleus) and top-k — truncating the tail" },
        { type: "p", text: "Even at moderate temperature the model can occasionally pick a garbage token from the long tail. **top-k** keeps only the $k$ most likely tokens; **top-p** (nucleus sampling) keeps the smallest set whose cumulative probability reaches $p$ (e.g. 0.9), then renormalizes and samples from that set. top-p adapts its cutoff to the model's confidence, which is why it is the more popular of the two." },
        { type: "table",
          headers: ["Setting", "Effect", "Good default"],
          rows: [
            ["`temperature`", "randomness / creativity", "$0.7$ chat, $0.0$ facts/code"],
            ["`top_p`", "keep top mass, cut the tail", "$0.9$–$0.95$"],
            ["`top_k`", "keep top $k$ tokens", "$40$ (or off if using top_p)"],
            ["`max_new_tokens`", "hard cap on output length", "task-dependent"],
            ["`repetition_penalty`", "down-weight already-seen tokens", "$1.1$ if it loops"],
            ["`stop` / `eos`", "strings/ids that end generation", "model's chat stop tokens"],
            ["`seed`", "make sampling reproducible", "any fixed int for tests"],
          ]
        },
        { type: "heading", text: "Repetition penalty & stop sequences" },
        { type: "p", text: "Small models love to **loop** (\"...and that is why, and that is why, and that is why...\"). `repetition_penalty` > 1 divides the logits of tokens already in the context, discouraging repeats. **Stop sequences** cut generation the moment a given string appears — essential when you inject your own turn markers or want to stop at `\\n\\n`." },
        { type: "code", lang: "py", code: "from transformers import set_seed\nset_seed(0)   # reproducible sampling across a run\n\nout = model.generate(\n    **inputs,\n    max_new_tokens=200,\n    do_sample=True,\n    temperature=0.8,\n    top_p=0.9,\n    top_k=40,\n    repetition_penalty=1.15,      # kill the loops\n    no_repeat_ngram_size=3,       # hard ban on repeating any 3-gram\n    eos_token_id=tok.eos_token_id # stop at end-of-sequence\n)" },
        { type: "callout", variant: "gotcha", text: "**Greedy is not \"the best\" answer.** `do_sample=False` (greedy/argmax) is deterministic but often *duller* and more repetitive than sampling — always picking the single top token is a local, not global, optimum. For anything factual or code, low temperature (0–0.3) beats pure greedy; for chat and brainstorming, 0.7–1.0 with top-p 0.9 is the sweet spot." },
        { type: "callout", variant: "tip", text: "**Reproducibility for tests.** To get identical output every run, fix the seed *and* pin `temperature`, `top_p`, `top_k`. Even then, GPU non-determinism (parallel reductions) can cause tiny differences — set `temperature=0` (greedy) when you need a truly stable golden output for an assertion." },
      ]
    },

    {
      id: "chat-templates",
      title: "Chat templates, roles & structured output",
      level: "core",
      body: [
        { type: "p", text: "A raw language model only continues text. To make it a *chat assistant*, the fine-tuning process taught it a specific **conversation format** with special tokens marking whose turn it is. Getting this format exactly right is the difference between coherent replies and gibberish — and it is the single most common local-LLM bug." },
        { type: "heading", text: "Base vs Instruct/Chat models" },
        { type: "p", text: "Every model family ships (at least) two variants. A **base** model is the raw next-token predictor from pretraining — it *completes* text and does not follow instructions. An **instruct** (or **chat**) model has been further trained (SFT + RLHF/DPO) to follow instructions in a role format. **For chatbots and assistants you almost always want the `-Instruct` / `-Chat` / `-it` variant.** The base model is for fine-tuning or raw completion." },
        { type: "table",
          headers: ["You want to…", "Use the…", "Example id"],
          rows: [
            ["chat / follow instructions", "**Instruct** model", "`Llama-3.1-8B-Instruct`"],
            ["fine-tune on your own data", "**Base** model", "`Llama-3.1-8B`"],
            ["raw text completion / autocomplete", "**Base** model", "`Qwen2.5-7B`"],
          ]
        },
        { type: "heading", text: "The three roles and `apply_chat_template`" },
        { type: "p", text: "A conversation is a list of messages, each with a **role**: `system` (persistent instructions/persona), `user` (the human), and `assistant` (the model's prior replies). You do **not** format the special tokens by hand — the tokenizer knows this model's exact template and `apply_chat_template` renders it for you." },
        { type: "code", lang: "py", code: "messages = [\n    {\"role\": \"system\", \"content\": \"You are a helpful assistant who answers in one sentence.\"},\n    {\"role\": \"user\", \"content\": \"Why is the sky blue?\"},\n]\n\n# Render to the model's exact chat format. add_generation_prompt=True appends\n# the tokens that cue the model to start the assistant's turn.\nprompt_ids = tok.apply_chat_template(\n    messages, add_generation_prompt=True, return_tensors=\"pt\"\n).to(model.device)\n\nout = model.generate(prompt_ids, max_new_tokens=80, do_sample=True, temperature=0.7)\nreply = tok.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)\nprint(reply)" },
        { type: "code", lang: "text", code: "# What apply_chat_template actually produces for a Llama-3 model\n# (the special tokens are model-specific — never type these by hand):\n<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant who answers in one sentence.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\nWhy is the sky blue?<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n" },
        { type: "callout", variant: "gotcha", text: "**Mismatched templates = broken model.** If you hand-write the wrong special tokens, or feed an Instruct prompt to a base model, you get rambling or nonsense. Always use `apply_chat_template` (or let Ollama/the API do it). Set `add_generation_prompt=True` when you want a *reply*, and `False` when you are building training targets." },
        { type: "heading", text: "Structured output: JSON mode & function calling" },
        { type: "p", text: "Apps need machine-readable output, not prose. Three escalating techniques: (1) **prompt** for JSON and hope; (2) **JSON mode / constrained decoding**, where the runtime masks logits so only tokens that keep the output valid JSON (or a specific schema) can be sampled — this *guarantees* parseable output; (3) **function/tool calling**, where the model is given tool schemas and emits a structured call your code executes." },
        { type: "code", lang: "py", code: "# Grammar-constrained JSON with an OpenAI-compatible server (Ollama/vLLM/llama.cpp\n# all support a 'response_format' / json_schema). Guarantees valid JSON out.\nfrom openai import OpenAI\nclient = OpenAI(base_url=\"http://localhost:11434/v1\", api_key=\"ollama\")\n\nresp = client.chat.completions.create(\n    model=\"llama3.1:8b\",\n    messages=[{\"role\": \"user\", \"content\": \"Extract name and age: 'Ada, 36, engineer'\"}],\n    response_format={\"type\": \"json_object\"},   # force valid JSON\n)\nimport json\nprint(json.loads(resp.choices[0].message.content))   # {'name': 'Ada', 'age': 36}" },
        { type: "callout", variant: "note", text: "**Function/tool calling is the seed of agents.** When a model can emit a structured `call(get_weather, city=\"Paris\")` and you run it and feed the result back, you have the core loop of an *agent*. That whole topic — tools, planning, multi-step — is covered in the Agents track. Here, just know that structured output is where it begins." },
      ]
    },
    {
      id: "hosted-apis",
      title: "Calling hosted APIs — and your local server with the same code",
      level: "core",
      body: [
        { type: "p", text: "The industry has converged on one request shape: the **OpenAI Chat Completions API**. OpenAI, and — crucially — Ollama, vLLM, llama.cpp's server, LM Studio, and most hosting providers all speak it. Learn it once and the *identical* client code talks to a $20B hosted model or to a 3B model on your laptop. Only `base_url`, `api_key`, and `model` change." },
        { type: "code", lang: "py", code: "from openai import OpenAI   # pip install openai\n\n# --- Hosted (OpenAI) ---\nclient = OpenAI(api_key=\"sk-...\")               # base_url defaults to OpenAI\n\n# --- Local (Ollama), SAME class, SAME methods ---\nclient = OpenAI(base_url=\"http://localhost:11434/v1\", api_key=\"ollama\")\n\n# --- Local (vLLM) ---\nclient = OpenAI(base_url=\"http://localhost:8000/v1\", api_key=\"none\")\n\nresp = client.chat.completions.create(\n    model=\"llama3.1:8b\",     # or \"gpt-4o-mini\", etc.\n    messages=[\n        {\"role\": \"system\", \"content\": \"You are concise.\"},\n        {\"role\": \"user\", \"content\": \"What is a KV-cache?\"},\n    ],\n    temperature=0.7,\n    max_tokens=200,\n)\nprint(resp.choices[0].message.content)" },
        { type: "callout", variant: "tip", text: "**One codebase, any backend.** Because Ollama and vLLM expose `/v1/chat/completions`, you can prototype against a free local model and switch to a hosted frontier model for production by changing three strings. Keep `base_url` and `model` in config, never hard-coded." },
        { type: "heading", text: "Streaming — tokens as they are produced" },
        { type: "p", text: "For anything interactive you want tokens to appear as they are generated, not after a 10-second wait. Set `stream=True` and iterate over server-sent chunks. This is exactly how the typewriter effect in every chat UI works." },
        { type: "code", lang: "py", code: "stream = client.chat.completions.create(\n    model=\"llama3.1:8b\",\n    messages=[{\"role\": \"user\", \"content\": \"Write a haiku about GPUs.\"}],\n    stream=True,\n)\nfor chunk in stream:\n    delta = chunk.choices[0].delta.content\n    if delta:\n        print(delta, end=\"\", flush=True)   # print each token as it arrives\nprint()" },
        { type: "heading", text: "Token accounting, cost & rate limits" },
        { type: "p", text: "Hosted APIs bill by **tokens** — both the **prompt** (input) and the **completion** (output), usually at different prices, output being more expensive. Every response carries a `usage` object. Budgeting means counting tokens *before* you send (with `tiktoken` for OpenAI, or the model's own tokenizer) so a runaway prompt doesn't surprise you." },
        { type: "code", lang: "py", code: "print(resp.usage)\n# CompletionUsage(prompt_tokens=23, completion_tokens=57, total_tokens=80)\n\n# Estimate cost (example rates — always check current pricing):\nrate_in, rate_out = 0.15, 0.60         # $ per 1M tokens (input, output)\ncost = (resp.usage.prompt_tokens * rate_in\n        + resp.usage.completion_tokens * rate_out) / 1e6\nprint(f\"${cost:.6f}\")\n\n# Count tokens BEFORE sending (OpenAI models):\nimport tiktoken\nenc = tiktoken.encoding_for_model(\"gpt-4o-mini\")\nprint(len(enc.encode(\"How many tokens is this sentence?\")))" },
        { type: "callout", variant: "gotcha", text: "**Rate limits are per-minute quotas on requests (RPM) and tokens (TPM).** Hammer an API and you get `429 Too Many Requests`. The fix is **exponential backoff with jitter**: on a 429, wait $2^n$ seconds (plus randomness) and retry. Batch and cache where you can. Local servers have no token cost but are limited by your GPU's throughput instead." },
        { type: "callout", variant: "note", text: "For Anthropic's Claude specifically, the request shape differs slightly (a top-level `system` field, `messages` with `role`/`content`) via the `anthropic` SDK — but the *concepts* (roles, streaming, `usage`, temperature/top_p) are identical. Consult the provider's own docs for exact model ids and pricing rather than trusting a hard-coded table." },
      ]
    },

    {
      id: "quantization",
      title: "Quantization for inference — trading precision for memory",
      level: "core",
      body: [
        { type: "p", text: "Weights are trained in 16-bit floats, but you rarely need that precision to *run* the model. **Quantization** stores each weight in fewer bits — 8, 4, even 2 — shrinking memory 2–8× and often speeding things up, at a small and usually acceptable quality cost. This is *the* technique that lets a 70B model run on a gaming GPU." },
        { type: "math", tex: String.raw`\text{VRAM}_{\text{weights}} \;\approx\; (\text{\# params}) \times (\text{bytes/param}), \qquad \text{fp16}=2,\; \text{int8}\approx1,\; \text{int4}\approx0.5` },
        { type: "heading", text: "The main families" },
        { type: "table",
          headers: ["Method", "Bits", "Ecosystem", "Notes"],
          rows: [
            ["**bitsandbytes**", "8 / 4 (NF4)", "`transformers` / PyTorch", "load on the fly, no pre-conversion; great for QLoRA fine-tuning"],
            ["**GGUF (k-quants)**", "2–8", "llama.cpp / Ollama", "CPU+GPU, the local-inference standard (Section 2)"],
            ["**GPTQ**", "3–4", "GPU (`transformers`/vLLM)", "post-training, calibration-based, fast kernels (Marlin)"],
            ["**AWQ**", "4", "GPU (`transformers`/vLLM)", "activation-aware; strong quality, popular for vLLM serving"],
            ["**FP8**", "8", "modern GPUs / vLLM", "hardware-accelerated on H100-class cards"],
          ]
        },
        { type: "heading", text: "4-bit in `transformers` with bitsandbytes" },
        { type: "p", text: "The easiest quantization in Python: pass a `BitsAndBytesConfig` and the model loads in 4-bit directly, no separate conversion step. This is what makes **QLoRA** (fine-tuning a 4-bit model with small trainable adapters) possible on a single consumer GPU." },
        { type: "code", lang: "py", code: "import torch\nfrom transformers import AutoModelForCausalLM, BitsAndBytesConfig\n\nbnb = BitsAndBytesConfig(\n    load_in_4bit=True,\n    bnb_4bit_quant_type=\"nf4\",              # NormalFloat4 — best for LLM weights\n    bnb_4bit_compute_dtype=torch.bfloat16,  # math still done in bf16\n    bnb_4bit_use_double_quant=True,         # quantize the quant constants too\n)\nmodel = AutoModelForCausalLM.from_pretrained(\n    \"meta-llama/Llama-3.1-8B-Instruct\",\n    quantization_config=bnb, device_map=\"auto\",\n)   # ~5GB instead of ~16GB" },
        { type: "callout", variant: "gotcha", text: "**What you trade.** 8-bit is essentially free (imperceptible quality loss). 4-bit costs a small, usually-fine amount (a point or two on benchmarks) and is the standard for local use. 3-bit and below degrade noticeably. Rule of thumb: **a bigger model at 4-bit usually beats a smaller model at 8-bit** for the same VRAM. Prefer more parameters over more bits." },
        { type: "heading", text: "The other memory cost: the KV-cache" },
        { type: "p", text: "Weights are not the whole story. During generation the model **caches** the key and value vectors of every past token so it doesn't recompute them each step (this is the KV-cache from the transformers topic). This cache grows **linearly with context length** and can rival the weights for long contexts — it is why a huge context window costs so much memory." },
        { type: "math", tex: String.raw`\text{KV bytes} \;=\; 2 \times L_{\text{layers}} \times n_{\text{ctx}} \times d_{\text{model}} \times b \times (\text{batch})` },
        { type: "p", text: "The factor $2$ is keys **and** values, $L$ is layers, $n_{\\text{ctx}}$ is the number of tokens in context, $d_{\\text{model}}$ the hidden size, and $b$ the bytes per element. For an 8B model at 32K context this is several GB *on top of* the weights — often the reason you run out of memory before the weights do." },
        { type: "callout", variant: "tip", text: "**Levers when you hit an OOM.** (1) Quantize the weights harder (Q4). (2) Shrink the context window / `--ctx-size` — the KV-cache scales with it. (3) Reduce batch/concurrent requests. (4) Use models with **Grouped-Query Attention** (Llama 3, Qwen2.5), which share KV heads and cut the cache several-fold. (5) In vLLM, PagedAttention already minimizes cache waste for you." },
      ]
    },

    {
      id: "embeddings-multimodal",
      title: "Embeddings and multimodal models",
      level: "core",
      body: [
        { type: "p", text: "Generation is only one thing an LLM stack does. Two others you will reach for constantly: turning text into **vectors** (embeddings), and models that see **images** (multimodal)." },
        { type: "heading", text: "Embeddings — text as vectors" },
        { type: "p", text: "An **embedding model** maps a chunk of text to a fixed-length vector such that *similar meanings land close together* (cosine similarity — see the Linear Algebra topic). This is the engine of semantic search, clustering, deduplication, and **RAG**. The same servers expose an `/v1/embeddings` endpoint." },
        { type: "code", lang: "py", code: "# Local embeddings via Ollama (pull one first: `ollama pull nomic-embed-text`)\nfrom openai import OpenAI\nclient = OpenAI(base_url=\"http://localhost:11434/v1\", api_key=\"ollama\")\n\ne = client.embeddings.create(model=\"nomic-embed-text\",\n                             input=[\"a cat sat on a mat\", \"a feline rested on a rug\"])\nv1, v2 = e.data[0].embedding, e.data[1].embedding\n\nimport numpy as np\nv1, v2 = np.array(v1), np.array(v2)\ncos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))\nprint(round(float(cos), 3))   # high — the two sentences mean the same thing" },
        { type: "code", lang: "py", code: "# Or fully in-process with sentence-transformers (no server):\nfrom sentence_transformers import SentenceTransformer\nembedder = SentenceTransformer(\"BAAI/bge-small-en-v1.5\")\nvecs = embedder.encode([\"a cat sat on a mat\", \"a feline rested on a rug\"],\n                       normalize_embeddings=True)\nprint(float(vecs[0] @ vecs[1]))   # cosine similarity" },
        { type: "callout", variant: "note", text: "**Embeddings are the whole first half of RAG** (Retrieval-Augmented Generation): embed your documents, store the vectors, embed the user's question, retrieve the nearest chunks, and stuff them into the prompt. That pipeline — chunking, vector databases, retrieval — gets a dedicated topic. Here, just know where vectors come from." },
        { type: "heading", text: "Multimodal — models that see" },
        { type: "p", text: "**Vision-language models (VLMs)** accept images alongside text: Llama 3.2 Vision, Qwen2.5-VL, Gemma 3, LLaVA, and the hosted GPT-4o/Gemini. Under the hood an image encoder turns the picture into embedding tokens that are concatenated with the text tokens — from the transformer's point of view it is all one sequence. The API just lets a message `content` be a list of text and image parts." },
        { type: "code", lang: "py", code: "# Vision via an OpenAI-compatible server (e.g. ollama run llama3.2-vision)\nresp = client.chat.completions.create(\n    model=\"llama3.2-vision\",\n    messages=[{\"role\": \"user\", \"content\": [\n        {\"type\": \"text\", \"text\": \"What is in this image?\"},\n        {\"type\": \"image_url\",\n         \"image_url\": {\"url\": \"data:image/jpeg;base64,\" + b64_of_my_image}},\n    ]}],\n)\nprint(resp.choices[0].message.content)" },
        { type: "callout", variant: "tip", text: "Multimodal models are hungrier — a single high-res image can expand to hundreds or thousands of tokens. Watch your context budget and cost. For OCR-heavy or document tasks, a purpose-built VLM (Qwen2.5-VL) often beats a general chat model." },
      ]
    },

    {
      id: "build-chatbot",
      title: "Build it: a streaming local chatbot with memory",
      level: "core",
      body: [
        { type: "p", text: "Time to assemble everything into a real, useful program: a terminal chatbot that (1) runs against a **local** model, (2) **streams** its reply token-by-token, and (3) keeps **conversation history** so it remembers earlier turns. This is ~40 lines and it is genuinely the core of every chat app." },
        { type: "code", lang: "py", code: "\"\"\"A minimal streaming chatbot with memory. Requires a local OpenAI-compatible\nserver, e.g.:  ollama run llama3.1:8b   (then run this script).\"\"\"\nfrom openai import OpenAI\n\nclient = OpenAI(base_url=\"http://localhost:11434/v1\", api_key=\"ollama\")\nMODEL = \"llama3.1:8b\"\n\n# History IS the memory: we resend the whole list every turn.\nhistory = [{\"role\": \"system\", \"content\": \"You are a helpful, concise assistant.\"}]\n\nprint(\"Chat (Ctrl-C to quit)\\n\")\nwhile True:\n    try:\n        user = input(\"you> \").strip()\n    except (EOFError, KeyboardInterrupt):\n        break\n    if not user:\n        continue\n\n    history.append({\"role\": \"user\", \"content\": user})\n\n    print(\"bot> \", end=\"\", flush=True)\n    reply = \"\"\n    stream = client.chat.completions.create(\n        model=MODEL, messages=history, stream=True, temperature=0.7,\n    )\n    for chunk in stream:\n        piece = chunk.choices[0].delta.content\n        if piece:\n            print(piece, end=\"\", flush=True)\n            reply += piece\n    print(\"\\n\")\n\n    history.append({\"role\": \"assistant\", \"content\": reply})   # remember the reply" },
        { type: "callout", variant: "gotcha", text: "**The model is stateless — *you* hold the memory.** The server remembers nothing between calls; \"memory\" is just you resending the growing `history` list every turn. That means every turn re-processes the whole conversation, so cost and latency grow as it gets longer." },
        { type: "callout", variant: "tip", text: "**Context windows are finite** (a few K to hundreds of K tokens). When `history` outgrows the window you must **trim** or **summarize** old turns — a common strategy is to keep the system prompt plus the last N turns, or periodically ask the model to summarize the older conversation into one message. Real assistants do exactly this." },
        { type: "p", text: "That is the whole loop. To make it a proper local product you would add: a token-count guard that trims history, a `--model` flag, and persistence to disk. Everything else — tools, retrieval, personas — is layered on top of this exact structure." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "You only learn this by running models on your own machine. Every project below works with a free, local model — no API key required. Start with a small model (`llama3.2:3b` or `Qwen2.5-1.5B-Instruct`) so iteration is fast." },
        { type: "list", ordered: true, items: [
          "**The five-minute private assistant.** Install Ollama, `ollama run llama3.1:8b`, and chat with it offline. Then hit its OpenAI endpoint with `curl` and with the Python `openai` client. Confirm the *same* code works if you swap `base_url` to a hosted provider. Goal: internalize that local and hosted are one interface.",
          "**Decoding lab.** Take one fixed prompt and generate 5 completions each at `temperature` ∈ {0, 0.3, 0.7, 1.2} and `top_p` ∈ {0.5, 0.9, 1.0}. Line them up. *See* determinism collapse into chaos, and watch a small model start looping — then fix the loop with `repetition_penalty=1.15`.",
          "**From-scratch generate().** Load `Qwen2.5-1.5B-Instruct` with `transformers`, apply the chat template by hand, and write your *own* sampling loop: call the model for logits, apply temperature + top-p yourself, sample one token, append, repeat. Match `model.generate()`'s behavior. This demystifies the whole stack.",
          "**Quantization shootout.** Run the same 7–8B model at Q8, Q4_K_M, and Q3 (via Ollama tags or bitsandbytes). Measure VRAM, tokens/sec, and quality on 10 questions you grade yourself. Confirm the rule: a bigger model at Q4 beats a smaller one at Q8 for equal memory.",
          "**Structured extraction service.** Build a function that takes messy text (an invoice, a résumé line) and returns validated JSON using `response_format={\"type\":\"json_object\"}`. Add a Pydantic schema and re-ask on validation failure. This is the bridge to agents.",
          "**The chatbot, finished.** Extend Section 9's loop: add history trimming when you near the context limit, persist conversations to a JSON file, and add a `/system` command to change the persona live. Then point it at a vision model and let it describe images.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The tooling here moves fast — always check the official docs for current model ids, flags, and pricing rather than trusting any cached table. These are the primary sources:" },
        { type: "link", url: "https://github.com/ollama/ollama", text: "Ollama — GitHub + docs (the fastest path to a local model and an OpenAI-compatible endpoint)" },
        { type: "link", url: "https://huggingface.co/docs/transformers/en/llm_tutorial", text: "HuggingFace Transformers — the text-generation / LLM tutorial (generate, chat templates, decoding)" },
        { type: "link", url: "https://github.com/ggml-org/llama.cpp", text: "llama.cpp — the GGUF inference engine, quantization guide, and llama-server docs" },
        { type: "link", url: "https://docs.vllm.ai/en/latest/", text: "vLLM docs — high-throughput serving, PagedAttention, AWQ/GPTQ/FP8 quantization, OpenAI server" },
        { type: "link", url: "https://platform.openai.com/docs/api-reference/chat", text: "OpenAI Chat Completions API — the request/response shape everyone implements (streaming, usage, tools)" },
        { type: "link", url: "https://huggingface.co/docs/transformers/en/quantization/bitsandbytes", text: "bitsandbytes quantization in Transformers — 8-bit and 4-bit (NF4), the basis of QLoRA" },
        { type: "link", url: "https://lmstudio.ai/docs", text: "LM Studio — the desktop GUI for discovering, downloading, and serving local GGUF models" },
      ]
    },
  ],

  packages: [
    { name: "ollama", why: "one-command local models + an OpenAI-compatible server on :11434 — the easiest start" },
    { name: "transformers", why: "load raw HuggingFace weights in Python; `AutoModelForCausalLM`, `generate`, chat templates" },
    { name: "accelerate", why: "powers `device_map=\"auto\"` — multi-GPU and CPU/disk offload placement" },
    { name: "openai", why: "the client for hosted *and* local OpenAI-compatible endpoints (Ollama, vLLM, llama.cpp)" },
    { name: "vllm", why: "production serving engine — PagedAttention, batching, AWQ/GPTQ/FP8, OpenAI API" },
    { name: "bitsandbytes", why: "on-the-fly 8-bit / 4-bit (NF4) quantization for `transformers`; enables QLoRA" },
    { name: "sentence-transformers", why: "in-process embedding models for semantic search and RAG" },
    { name: "tiktoken", why: "count tokens before you send — budget cost and stay under the context limit" },
  ],

  gotchas: [
    "`generate()` returns the **prompt plus** the completion. Slice off `input_ids.shape[1]` tokens or you echo the prompt back.",
    "Use the **`-Instruct`/`-Chat`** variant for chat; a **base** model does not follow instructions — it just completes text.",
    "Always format conversations with `apply_chat_template`, never hand-typed special tokens. A wrong template gives gibberish.",
    "**Greedy is not best.** `do_sample=False` is deterministic but dull and loop-prone; low-temperature sampling usually reads better.",
    "The model is **stateless** — *you* resend the whole `history` each turn. Memory grows cost and latency; trim or summarize old turns.",
    "Weights aren't the only memory cost: the **KV-cache** grows linearly with context length and can exceed the weights at long $n_{\\text{ctx}}$.",
    "Prefer a **bigger model at 4-bit** over a smaller model at 8-bit for the same VRAM — parameters buy more than precision.",
    "Hosted APIs enforce **RPM/TPM rate limits**; on `429` back off exponentially with jitter and retry rather than hammering.",
  ],

  flashcards: [
    { q: "When should you run open weights locally instead of calling a hosted API?", a: "For privacy (data can't leave), cost at scale/batch, offline use, full control (sampler, quantization, fine-tuning), or to learn. Hosted wins when you want the single strongest model with zero ops." },
    { q: "What is the fastest way to a working local LLM with an OpenAI-compatible API?", a: "Install Ollama and `ollama run llama3.1:8b`. It downloads a Q4 quant and exposes `/v1/chat/completions` on `localhost:11434`." },
    { q: "What does `Q4_K_M` mean and why is it the default?", a: "A 4-bit GGUF k-quant with *mixed* precision (more bits for sensitive layers). Best balance: ~1–3% quality loss for ~3–4× smaller than fp16." },
    { q: "Base vs Instruct model — which for a chatbot?", a: "**Instruct** (also `-Chat`/`-it`). It was fine-tuned to follow instructions in a role format. The base model only continues text; use it for fine-tuning or raw completion." },
    { q: "What does temperature do to the logits?", a: "Divides them before softmax: $p_i=\\mathrm{softmax}(z_i/T)$. $T<1$ sharpens (safe), $T>1$ flattens (creative), $T\\to0$ is greedy/deterministic." },
    { q: "Difference between top-k and top-p (nucleus) sampling?", a: "top-k keeps the $k$ most likely tokens; top-p keeps the smallest set whose cumulative probability reaches $p$, adapting the cutoff to the model's confidence." },
    { q: "Why can the same client code hit both OpenAI and a local model?", a: "Ollama, vLLM, and llama.cpp all implement the OpenAI Chat Completions API. Only `base_url`, `api_key`, and `model` change." },
    { q: "What is the KV-cache and why does it matter for memory?", a: "Cached key/value vectors for past tokens so they aren't recomputed each step. It grows linearly with context length and can rival the weights in size." },
    { q: "What does `bitsandbytes` `load_in_4bit` with NF4 give you?", a: "On-the-fly 4-bit loading of HF weights (~3–4× less VRAM), math still done in bf16. It's the basis of QLoRA fine-tuning on consumer GPUs." },
    { q: "How does a chatbot 'remember' earlier turns?", a: "It doesn't — the server is stateless. You resend the whole growing `messages`/history list every request." },
    { q: "How do you get *guaranteed* valid JSON out of a model?", a: "Constrained decoding / JSON mode (`response_format`), where the runtime masks logits so only tokens that keep the output valid schema can be sampled." },
    { q: "Rule of thumb for choosing model size vs quantization for fixed VRAM?", a: "A bigger model at 4-bit generally beats a smaller model at 8-bit. Parameters matter more than bits, down to ~4-bit; below 3-bit quality degrades." },
  ],

  cheatsheet: [
    { label: "Run a model (Ollama)", code: "ollama run llama3.1:8b" },
    { label: "Local OpenAI client", code: "OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')" },
    { label: "Load in transformers", code: "AutoModelForCausalLM.from_pretrained(id, torch_dtype='auto', device_map='auto')" },
    { label: "Generate", code: "model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)" },
    { label: "Chat template", code: "tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt')" },
    { label: "Pipeline one-liner", code: "pipeline('text-generation', model=id)(messages, max_new_tokens=64)" },
    { label: "Chat completion", code: "client.chat.completions.create(model=m, messages=msgs, temperature=0.7)" },
    { label: "Stream tokens", code: "for c in client.chat.completions.create(..., stream=True): print(c.choices[0].delta.content or '', end='')" },
    { label: "Force JSON out", code: "response_format={'type': 'json_object'}" },
    { label: "Token usage", code: "resp.usage.prompt_tokens, resp.usage.completion_tokens" },
    { label: "Count tokens", code: "tiktoken.encoding_for_model('gpt-4o-mini').encode(text)" },
    { label: "4-bit load", code: "BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')" },
    { label: "Serve with vLLM", code: "vllm serve meta-llama/Llama-3.1-8B-Instruct" },
    { label: "Local embeddings", code: "client.embeddings.create(model='nomic-embed-text', input=[...])" },
  ],
});
