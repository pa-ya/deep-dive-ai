(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "mlops-serving",
  name: "Serving & Deploying Models",
  language: "MLOps",
  group: "MLOps & Deployment",
  navLabel: "Serving & Deployment",
  tagline: "Turning *\"I trained a model\"* into *\"anyone can call it\"* — serialize, wrap in an API, containerize, optimize, and scale.",
  color: "#64748B",
  readMinutes: 46,
  sections: [
    {
      id: "gap",
      title: "The deployment gap: from notebook to production",
      level: "core",
      body: [
        { type: "p", text: "You trained a model. It sits in a Jupyter notebook as a Python variable named `model`. Right now exactly one person on earth can use it — you, with that kernel alive. **Deployment** is the engineering work of turning that ephemeral object into a service other people (and other programs) can call reliably, at scale, without you in the loop. This is where most ML projects quietly die: a [2020 survey and the folklore since](https://venturebeat.com/ai/why-do-87-of-data-science-projects-never-make-it-into-production/) put the fraction of models that never ship at well over half." },
        { type: "p", text: "The gap is real because a notebook and a production service optimize for opposite things. A notebook is interactive, stateful, single-user, and forgiving. A service must be stateless, concurrent, observable, versioned, and *fast enough* under load. Nothing about training prepares the model object for that." },
        { type: "table",
          headers: ["Concern", "In the notebook", "In production"],
          rows: [
            ["Who calls it", "you, by hand", "apps, cron jobs, other services"],
            ["Lifetime", "until the kernel dies", "months; survives restarts"],
            ["Input", "a clean DataFrame you made", "untrusted JSON from the internet"],
            ["Latency", "\"a few seconds is fine\"", "p99 under a budget (e.g. 200 ms)"],
            ["Failure", "you see the traceback", "must log, alert, and degrade gracefully"],
            ["Environment", "whatever is installed", "pinned, reproducible, containerized"],
          ]
        },
        { type: "heading", text: "Three shapes of inference" },
        { type: "p", text: "Before writing any code, decide *how* predictions are consumed. This single choice dictates your entire architecture — the wrong pick means you build the wrong thing." },
        { type: "table",
          headers: ["Pattern", "How it works", "Latency budget", "Use when"],
          rows: [
            ["**Batch / offline**", "score a whole table on a schedule, write results to a DB", "minutes–hours", "recommendations refreshed nightly, risk scores, churn"],
            ["**Online / real-time**", "an HTTP request in, one prediction out, synchronously", "milliseconds", "fraud check at checkout, search ranking, a chatbot reply"],
            ["**Streaming**", "consume an event stream (Kafka/Kinesis), emit predictions continuously", "sub-second, continuous", "clickstream personalization, anomaly detection on sensors"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Batch is dramatically easier — prefer it when you can.** If the business can tolerate predictions that are a few hours stale, a nightly batch job (a Python script + a cron trigger + a table) removes almost every hard problem in this deck: no latency budget, no autoscaling, no online serving stack. Only reach for real-time when *freshness at request time* genuinely matters. Most teams build a real-time service they didn't need." },
        { type: "callout", variant: "note", text: "**This file is the online/real-time path**, because it is the one that needs real engineering. We build up: serialize the model → wrap it in an API → containerize it → pick a serving framework → optimize it → scale it. Each section is a layer you add only when the previous one is solid." },
      ]
    },

    {
      id: "serialize",
      title: "Serializing models: pickle, joblib, torch, safetensors, ONNX",
      level: "core",
      body: [
        { type: "p", text: "**Serialization** is converting the in-memory model into bytes on disk that a *different* process can load back. The format you choose determines security, portability, and load speed. There are two things worth saving: the **weights** (the learned numbers) and, sometimes, the **computation graph** (the architecture). Frameworks disagree on which they store." },
        { type: "heading", text: "scikit-learn: joblib (not raw pickle)" },
        { type: "p", text: "A fitted sklearn estimator is a plain Python object, so it serializes with `pickle`. Use `joblib` instead — it is a drop-in that handles the large NumPy arrays inside estimators far more efficiently." },
        { type: "code", lang: "py", code: "import joblib\nfrom sklearn.ensemble import RandomForestClassifier\n\nclf = RandomForestClassifier().fit(X_train, y_train)\n\njoblib.dump(clf, \"model.joblib\")           # save\nclf = joblib.load(\"model.joblib\")          # load in another process\nprint(clf.predict(X_test[:5]))" },
        { type: "callout", variant: "warn", text: "**Pickle/joblib execute arbitrary code on load.** Unpickling runs `__reduce__` methods, so a malicious `.joblib` file is remote code execution the instant you `load()` it. [Hugging Face regularly finds malicious pickled models in the wild.](https://huggingface.co/docs/hub/security-pickle) **Never load a pickle from a source you don't trust**, and never accept one over the network as user input. This single fact is why `safetensors` exists." },
        { type: "heading", text: "PyTorch: state_dict, not the whole model" },
        { type: "p", text: "`torch.save(model, path)` pickles the entire object graph — including your source class — which is brittle (it breaks the moment your code moves) and unsafe. The idiomatic way is to save only the **`state_dict`**: an ordered dict mapping parameter names to tensors. You reconstruct the architecture in code, then pour the weights in." },
        { type: "code", lang: "py", code: "import torch\n\n# SAVE: just the learned tensors, not the Python object\ntorch.save(model.state_dict(), \"weights.pt\")\n\n# LOAD: rebuild the architecture, then load weights into it\nmodel = MyNet()                              # same class definition\nmodel.load_state_dict(torch.load(\"weights.pt\", weights_only=True))\nmodel.eval()                                 # turn off dropout / batchnorm updates" },
        { type: "callout", variant: "gotcha", text: "**Forgetting `model.eval()` is a classic silent bug.** In eval mode dropout is disabled and batch-norm uses running statistics. Leave the model in `train()` mode at inference and your predictions become noisy and batch-dependent — the model \"works\" but is quietly wrong. Pair it with `torch.no_grad()` (or `torch.inference_mode()`) to skip building the autograd graph and save memory." },
        { type: "heading", text: "safetensors: the safe, fast weight format" },
        { type: "p", text: "`safetensors` (from Hugging Face, now under the [PyTorch Foundation](https://huggingface.co/docs/safetensors)) stores *only* raw tensor data plus a small JSON header of names, shapes, and dtypes. It executes **no code** on load, so it is immune to the pickle attack, and it supports zero-copy memory-mapping, making it up to ~76× faster to load on CPU. It is now the default for essentially every model on the HF Hub." },
        { type: "code", lang: "py", code: "from safetensors.torch import save_file, load_file\n\n# save a state_dict safely\nsave_file(model.state_dict(), \"model.safetensors\")\n\n# load — no arbitrary code can run\nstate = load_file(\"model.safetensors\")\nmodel.load_state_dict(state)" },
        { type: "heading", text: "ONNX: framework-agnostic graph + weights" },
        { type: "p", text: "The four formats above are all *ecosystem-locked* — a `.joblib` needs sklearn, a `.pt` needs PyTorch. **ONNX** (Open Neural Network Exchange) is a portable standard that captures the **entire computation graph** *and* the weights as a language-neutral protobuf. Export once, then run the exact same model in C++, C#, Java, Rust, the browser, or on a microcontroller — via [ONNX Runtime](https://onnxruntime.ai/), TensorRT, or CoreML — with no Python at all." },
        { type: "code", lang: "py", code: "import torch\n\nmodel.eval()\ndummy = torch.randn(1, 3, 224, 224)          # a representative input\n\ntorch.onnx.export(\n    model, dummy, \"model.onnx\",\n    input_names=[\"input\"], output_names=[\"logits\"],\n    dynamic_axes={\"input\": {0: \"batch\"}},      # let batch size vary\n    opset_version=17,\n)\n\n# Run it anywhere, no PyTorch needed:\nimport onnxruntime as ort, numpy as np\nsess = ort.InferenceSession(\"model.onnx\")\nout = sess.run(None, {\"input\": np.random.randn(1, 3, 224, 224).astype(np.float32)})\nprint(out[0].shape)" },
        { type: "callout", variant: "tip", text: "**Why ONNX matters for serving:** decoupling the model from Python unlocks the fast runtimes. ONNX Runtime and TensorRT apply graph optimizations (operator fusion, constant folding) and hardware kernels that often make the same model **2–5× faster** than eager PyTorch — for free, without retraining. It is the standard bridge from \"trained in PyTorch\" to \"served fast in production.\"" },
        { type: "heading", text: "The reproducibility problem" },
        { type: "p", text: "A serialized weight file is worthless if you can't rebuild the environment that runs it. Six months later, a new NumPy releases, an operator's behavior shifts, and your saved model produces different numbers — or won't load at all. Serialization is only half of reproducibility; **pinning the environment** is the other half." },
        { type: "list", ordered: false, items: [
          "**Pin every dependency** to an exact version — `torch==2.5.1`, not `torch>=2`. Commit a `requirements.txt` (or `poetry.lock` / `uv.lock`) alongside the weights.",
          "**Record framework + Python + CUDA versions** in a metadata file saved *with* the model. A `.safetensors` plus a `meta.json` is a complete, loadable artifact.",
          "**Set and log seeds** (`torch.manual_seed`, `np.random.seed`, `random.seed`) so training is reproducible — though note that full bit-exactness on GPU also needs deterministic kernels.",
          "**Version the artifact itself.** Store models in an artifact registry (MLflow, W&B, or even versioned S3) keyed by a hash, so \"which model is in prod\" is always answerable.",
        ]},
        { type: "callout", variant: "gotcha", text: "**Train/serve skew is the deployment bug that hurts most.** The model is fine; the *preprocessing* differs. If training scaled features with a `StandardScaler` fit on the train set, serving must apply that *exact same* fitted scaler — not a fresh one. Serialize the **whole pipeline** (preprocessing + model) as one artifact, or you will silently feed the model inputs it never saw in training." },
      ]
    },

    {
      id: "fastapi",
      title: "Wrapping a model in an API with FastAPI",
      level: "core",
      body: [
        { type: "p", text: "An API turns your model into a network service: a client sends JSON over HTTP, your code runs the model, and JSON comes back. **[FastAPI](https://fastapi.tiangolo.com/)** is the de-facto standard for ML services in Python — it is fast (built on Starlette/ASGI), gives you automatic request validation and interactive docs via [Pydantic](https://docs.pydantic.dev/), and is natively async. It replaced Flask as the default for the same reason PyTorch replaced TensorFlow 1.x: the ergonomics are simply better." },
        { type: "heading", text: "Typed request & response schemas" },
        { type: "p", text: "The first thing an API needs is a contract: what does a valid request look like, and what comes back? You declare these as Pydantic models. FastAPI then validates every incoming request against the schema automatically — malformed input gets a clear `422` error *before* your model ever sees it." },
        { type: "code", lang: "py", code: "from pydantic import BaseModel, Field\n\nclass PredictRequest(BaseModel):\n    # a house with 4 numeric features; Field adds validation + docs\n    sqft: float = Field(gt=0, description=\"square footage\")\n    bedrooms: int = Field(ge=0, le=20)\n    bathrooms: float = Field(ge=0)\n    age_years: int = Field(ge=0)\n\nclass PredictResponse(BaseModel):\n    price: float\n    model_version: str" },
        { type: "heading", text: "Load the model once, at startup" },
        { type: "p", text: "The single most important performance rule for an ML API: **load the model once when the process boots, not on every request.** Loading is slow (disk + deserialization); inference is fast. Use FastAPI's `lifespan` to load into a shared object that every request reuses." },
        { type: "code", lang: "py", code: "from contextlib import asynccontextmanager\nfrom fastapi import FastAPI\nimport joblib\n\nml = {}   # shared state\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    ml[\"model\"] = joblib.load(\"model.joblib\")   # loaded ONCE at boot\n    ml[\"version\"] = \"1.3.0\"\n    yield                                        # app runs here\n    ml.clear()                                   # cleanup on shutdown\n\napp = FastAPI(title=\"House Price API\", lifespan=lifespan)" },
        { type: "callout", variant: "gotcha", text: "**Loading the model inside the endpoint function is the #1 rookie mistake.** It re-reads and re-deserializes the model on *every single request*, turning a 5 ms prediction into a 500 ms one and exhausting memory under load. Load at startup; the endpoint should only ever *call* the already-loaded model." },
        { type: "heading", text: "The predict endpoint" },
        { type: "p", text: "Now the endpoint itself: it receives an already-validated `PredictRequest`, builds the feature vector in the *exact order the model was trained on*, runs inference, and returns a typed response. FastAPI serializes it to JSON automatically." },
        { type: "code", lang: "py", code: "import numpy as np\nfrom fastapi import HTTPException\n\n@app.post(\"/predict\", response_model=PredictResponse)\nasync def predict(req: PredictRequest):\n    # feature order MUST match training exactly\n    x = np.array([[req.sqft, req.bedrooms, req.bathrooms, req.age_years]])\n    try:\n        price = float(ml[\"model\"].predict(x)[0])\n    except Exception as e:\n        raise HTTPException(status_code=500, detail=f\"inference failed: {e}\")\n    return PredictResponse(price=price, model_version=ml[\"version\"])\n\n@app.get(\"/health\")\nasync def health():\n    return {\"status\": \"ok\", \"model_loaded\": \"model\" in ml}" },
        { type: "callout", variant: "tip", text: "**Always ship a `/health` (liveness) and ideally a `/ready` (readiness) endpoint.** Load balancers and Kubernetes poll these to decide whether to send traffic to a replica. A pod that is still loading a 10 GB model should report *not ready* so it doesn't receive requests it will drop. This one endpoint is the difference between a rolling deploy that's invisible to users and one that returns 500s." },
        { type: "heading", text: "Batching requests for throughput" },
        { type: "p", text: "Running the model on one row at a time wastes the vectorized hardware. If your clients can send multiple items per call, accept a list — one HTTP round-trip and one `model.predict` on a whole matrix is far cheaper than N separate calls." },
        { type: "code", lang: "py", code: "from typing import List\n\nclass BatchRequest(BaseModel):\n    items: List[PredictRequest] = Field(..., max_length=1000)  # cap the batch\n\n@app.post(\"/predict_batch\")\nasync def predict_batch(req: BatchRequest):\n    X = np.array([[i.sqft, i.bedrooms, i.bathrooms, i.age_years]\n                  for i in req.items])\n    prices = ml[\"model\"].predict(X)                 # one vectorized call\n    return {\"prices\": prices.tolist(), \"model_version\": ml[\"version\"]}" },
        { type: "heading", text: "async, and the CPU-bound trap" },
        { type: "p", text: "FastAPI is async, which is a huge win for **I/O-bound** work — while one request waits on a database or a downstream API, the event loop serves others. But model inference is **CPU-bound** (or GPU-bound): it doesn't `await` anything, it just computes. A heavy synchronous computation inside an `async def` endpoint **blocks the entire event loop**, freezing every other request." },
        { type: "callout", variant: "gotcha", text: "**The async CPU trap:** if your prediction is a blocking, multi-hundred-millisecond compute, either (a) make the endpoint a plain `def` — FastAPI runs those in a threadpool automatically, keeping the loop free — or (b) offload to a process pool / a dedicated serving framework. An `async def` endpoint doing heavy CPU work is *slower* under load than a sync one, which surprises everyone. Reserve `async def` for endpoints that genuinely `await`." },
        { type: "code", lang: "bash", code: "# run it locally (dev)\nuvicorn app:app --reload\n\n# production: multiple worker processes behind an ASGI server\nuvicorn app:app --host 0.0.0.0 --port 8000 --workers 4\n# interactive docs are auto-generated at http://localhost:8000/docs" },
      ]
    },

    {
      id: "docker",
      title: "Containerizing with Docker",
      level: "core",
      body: [
        { type: "p", text: "\"Works on my machine\" is the enemy of deployment. **Docker** packages your code, the Python interpreter, every pinned dependency, and the model into a single **image** — a frozen filesystem that runs identically on your laptop, a colleague's machine, and a Kubernetes cluster. A running instance of an image is a **container**. This is how you make the reproducible environment from the serialization section *actually* travel." },
        { type: "heading", text: "A minimal Dockerfile for an ML service" },
        { type: "p", text: "A `Dockerfile` is a recipe, built top to bottom into cached **layers**. The key trick: order instructions from least- to most-frequently-changing, so Docker reuses cached layers. Dependencies change rarely; your code changes constantly — so copy and install requirements *before* copying the source." },
        { type: "code", lang: "text", code: "# Dockerfile\nFROM python:3.11-slim                      # small base, not full python:3.11\n\nWORKDIR /app\n\n# 1) deps first — this layer is cached until requirements.txt changes\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n\n# 2) then code + model (changes often, so it comes last)\nCOPY app.py model.joblib ./\n\n# document the port and run as non-root for safety\nEXPOSE 8000\nRUN useradd -m appuser\nUSER appuser\n\nCMD [\"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]" },
        { type: "code", lang: "bash", code: "docker build -t house-api:1.3.0 .\ndocker run -p 8000:8000 house-api:1.3.0\ncurl localhost:8000/health          # -> {\"status\":\"ok\",\"model_loaded\":true}" },
        { type: "callout", variant: "gotcha", text: "**`--host 0.0.0.0` is mandatory inside a container.** The default `127.0.0.1` binds only to the container's own loopback, so `-p 8000:8000` maps to nothing and you get \"connection refused\" from outside. Bind to `0.0.0.0` so the container accepts traffic on its published port. This one flag causes a genuinely enormous share of first-time Docker-ML confusion." },
        { type: "heading", text: "Keeping the image small" },
        { type: "p", text: "A naive ML image can hit several gigabytes, which slows every build, push, pull, and cold start. Size is money and latency." },
        { type: "list", ordered: false, items: [
          "**Start from `-slim`** (or `distroless`) base images, not the full OS image — hundreds of MB saved instantly.",
          "**`pip install --no-cache-dir`** so pip's download cache isn't baked into the layer.",
          "**Use a `.dockerignore`** to exclude `.git`, data, notebooks, and virtualenvs from the build context.",
          "**Multi-stage builds:** compile/install in a fat `builder` stage, then `COPY --from=builder` only the finished artifacts into a clean final stage — the toolchain never ships.",
          "**Install CPU-only PyTorch** when you don't need CUDA in the image (`--index-url .../cpu`); the CUDA wheels are multiple GB on their own.",
        ]},
        { type: "code", lang: "text", code: "# Multi-stage: build deps in one stage, copy only what runs\nFROM python:3.11-slim AS builder\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir --prefix=/install -r requirements.txt\n\nFROM python:3.11-slim\nCOPY --from=builder /install /usr/local     # only the installed packages\nWORKDIR /app\nCOPY app.py model.safetensors ./\nEXPOSE 8000\nCMD [\"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]" },
        { type: "heading", text: "GPU containers" },
        { type: "p", text: "To use a GPU inside a container you need two things: an image built on **NVIDIA's CUDA base image** (so the CUDA libraries are present), and the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)** on the host (so the container can see the physical GPU). You do *not* install the GPU driver in the image — the driver lives on the host and is injected at runtime." },
        { type: "code", lang: "text", code: "# GPU image: base on CUDA runtime, then add Python + your app\nFROM nvidia/cuda:12.4.1-runtime-ubuntu22.04\nRUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY app.py weights.safetensors ./\nCMD [\"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]" },
        { type: "code", lang: "bash", code: "# --gpus all exposes the host GPUs to the container (needs the toolkit)\ndocker run --gpus all -p 8000:8000 my-gpu-api:latest\n# inside the container this should now print True:\n#   python -c \"import torch; print(torch.cuda.is_available())\"" },
        { type: "callout", variant: "tip", text: "**Match CUDA versions across the stack.** The CUDA runtime in your base image must be compatible with the framework's CUDA build (the PyTorch `cu124` wheel) and with the host's driver. Mismatches produce the infamous `CUDA error: no kernel image is available` at runtime — the container builds fine and fails only when it touches the GPU. Pin all three deliberately." },
      ]
    },

    {
      id: "frameworks",
      title: "Serving frameworks: when a plain API isn't enough",
      level: "core",
      body: [
        { type: "p", text: "A FastAPI + Docker service is perfect for a classical model or moderate traffic. But it makes *you* implement everything a serious serving stack needs: dynamic batching, multi-model hosting, GPU scheduling, metrics, model versioning, and canary rollout. **Serving frameworks** are purpose-built servers that give you these for free. Reach for one when you outgrow the hand-rolled API — not before." },
        { type: "table",
          headers: ["Framework", "Built by / for", "Sweet spot"],
          rows: [
            ["**TorchServe**", "PyTorch team", "serving PyTorch models with handlers; being wound down — check status before adopting"],
            ["**Triton Inference Server**", "NVIDIA", "the heavyweight: multi-framework (ONNX/TRT/PyTorch/TF), dynamic batching, GPU concurrency, model ensembles"],
            ["**BentoML**", "BentoML", "Pythonic packaging: wrap any model as a `Service`, get an API + Docker image + K8s deploy; best DX for mixed/custom models"],
            ["**Ray Serve**", "Anyscale / Ray", "Python-native scaling; compose multi-model pipelines and scale each stage independently across a cluster"],
            ["**vLLM**", "vLLM project (open)", "LLM inference: PagedAttention + continuous batching → the throughput default for self-hosted LLMs"],
            ["**TGI**", "Hugging Face", "LLM serving; strong long-prompt prefix caching, but moved to maintenance mode in Dec 2025 — new work should use vLLM/SGLang"],
          ]
        },
        { type: "heading", text: "BentoML: the Pythonic middle ground" },
        { type: "p", text: "BentoML is the gentlest step up from FastAPI. You decorate a class as a `Service`; it handles the API, dependency packaging, containerization, and adaptive batching. You keep writing Python, but get production plumbing." },
        { type: "code", lang: "py", code: "import bentoml, numpy as np\n\n@bentoml.service(resources={\"cpu\": \"2\"}, traffic={\"timeout\": 10})\nclass HousePricer:\n    # BentoML loads the model once and can batch across requests for you\n    def __init__(self):\n        self.model = bentoml.sklearn.load_model(\"house_rf:latest\")\n\n    @bentoml.api(batchable=True)                 # server groups calls into batches\n    def predict(self, features: np.ndarray) -> np.ndarray:\n        return self.model.predict(features)\n# `bentoml serve` runs it; `bentoml build` + `containerize` makes a Docker image." },
        { type: "heading", text: "Serving LLMs is a different sport" },
        { type: "p", text: "Large language models break the classical serving model. Generation is **autoregressive** — one token at a time, each depending on the last — so a single request is a *loop*, requests finish at wildly different lengths, and the GPU sits idle waiting on memory. This is why LLMs get their own engines (covered in depth in the **LLMs / Using LLMs** track); the two ideas that changed everything:" },
        { type: "list", ordered: false, items: [
          "**PagedAttention (vLLM):** manages the attention KV-cache like an OS manages virtual memory — in fixed pages — eliminating the memory fragmentation that used to waste most of the GPU. This is vLLM's core trick.",
          "**Continuous (in-flight) batching:** instead of waiting for a whole batch to finish, the server swaps a *new* request into a slot the instant one sequence completes. GPU utilization goes from dismal to near-full — the single biggest throughput win for LLM serving.",
        ]},
        { type: "code", lang: "bash", code: "# Serve any HF model with an OpenAI-compatible API in one command:\nvllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000\n# then call it exactly like the OpenAI API:\ncurl localhost:8000/v1/completions -H 'Content-Type: application/json' \\\n  -d '{\"model\":\"meta-llama/Llama-3.1-8B-Instruct\",\"prompt\":\"Hello\",\"max_tokens\":32}'" },
        { type: "callout", variant: "note", text: "**Landscape as of 2026:** vLLM is the broad default for self-hosted LLM serving; **SGLang** has emerged as a top-tier peer (often winning on throughput/TTFT and constrained decoding, and powering serving at several major labs); **TensorRT-LLM** (via Triton) squeezes the most out of NVIDIA hardware at the cost of a compile step; **TGI** has moved to maintenance mode. For non-LLM models, Triton (max performance) and BentoML/Ray Serve (max flexibility) remain the mainstays." },
        { type: "callout", variant: "tip", text: "**Decision rule.** Classical model, modest load → FastAPI + Docker. Need dynamic batching / multi-model / GPU concurrency without writing it yourself → Triton (NVIDIA) or BentoML/Ray Serve (Python-first). Serving an LLM → vLLM or SGLang, full stop. Don't put an LLM behind a naive FastAPI loop — you will get a fraction of the possible throughput." },
      ]
    },

    {
      id: "optimize",
      title: "Optimizing inference: latency vs throughput",
      level: "core",
      body: [
        { type: "p", text: "Two metrics, often in tension, govern serving performance. **Latency** is how long *one* request takes (usually reported at the tail: p95/p99, because the worst case is what users feel). **Throughput** is how many requests you serve per second across all clients. Batching trades one for the other: bigger batches raise throughput (better hardware use) but raise each request's latency (it waits for batch-mates)." },
        { type: "callout", variant: "note", text: "**Optimize for the metric your product actually cares about.** An interactive chatbot lives or dies by p99 *latency* — a user won't wait. A nightly batch scoring job cares only about total *throughput* — no human is waiting, so max out batch size and GPU use. Picking the wrong target leads you to tune the wrong knobs." },
        { type: "heading", text: "The optimization toolbox, cheapest first" },
        { type: "table",
          headers: ["Technique", "What it does", "Typical win"],
          rows: [
            ["**Dynamic batching**", "server briefly buffers incoming requests, runs them as one batch", "large throughput gain, small latency cost"],
            ["**Quantization**", "store/compute weights in int8/fp8/int4 instead of fp32", "2–4× smaller, 2–4× faster, tiny accuracy loss"],
            ["**Graph runtime (ONNX/TensorRT)**", "operator fusion + hardware kernels on a frozen graph", "2–5× vs eager PyTorch, no retraining"],
            ["**Caching**", "memoize identical requests / reuse LLM KV-cache", "eliminates repeat compute entirely"],
            ["**Right hardware**", "GPU for big parallel models, CPU for small/light ones", "avoid paying for a GPU you don't need"],
          ]
        },
        { type: "heading", text: "Quantization" },
        { type: "p", text: "**Quantization** represents weights (and sometimes activations) in fewer bits — int8, fp8, or even int4 — instead of 32-bit floats. Memory and bandwidth are the bottleneck in modern inference, so using 4× fewer bytes is roughly 4× less to move and store, with accuracy loss that is usually negligible when done well. It is the single highest-leverage optimization for large models (derived in depth in the **quantization** material of the LLM track)." },
        { type: "code", lang: "py", code: "# Post-training dynamic quantization of a PyTorch model to int8 (CPU)\nimport torch\n\nqmodel = torch.quantization.quantize_dynamic(\n    model, {torch.nn.Linear}, dtype=torch.qint8    # quantize the Linear layers\n)\ntorch.save(qmodel.state_dict(), \"model_int8.pt\")\n# Smaller on disk, faster on CPU. For LLMs, use GPTQ/AWQ/bitsandbytes instead." },
        { type: "heading", text: "Compile the graph: ONNX Runtime & TensorRT" },
        { type: "p", text: "Eager PyTorch dispatches each operation from Python one at a time. A **graph runtime** first captures the whole computation, then **fuses** operations (e.g. a conv + bias + ReLU become one kernel), folds constants, and picks hardware-tuned kernels. [ONNX Runtime](https://onnxruntime.ai/) does this portably across CPU/GPU; **TensorRT** does it with NVIDIA-specific aggression (and a compile step) for the fastest GPU inference." },
        { type: "code", lang: "py", code: "import onnxruntime as ort\n\n# Pick the fastest available backend; ORT applies graph optimizations on load\nsess = ort.InferenceSession(\n    \"model.onnx\",\n    providers=[\"CUDAExecutionProvider\", \"CPUExecutionProvider\"],  # GPU, else CPU\n)\nout = sess.run(None, {\"input\": x})\n# Same weights as the PyTorch model, commonly 2-5x faster with no accuracy change." },
        { type: "heading", text: "Caching & the right hardware" },
        { type: "p", text: "If identical requests recur (the same search query, the same prompt prefix), **cache the result** — the cheapest possible optimization is not computing at all. For LLMs, prompt-prefix / KV caching reuses the attention state of a shared prefix across requests. On hardware: a **GPU** wins for large, densely parallel models (deep nets, transformers); a **CPU** is often cheaper *and* lower-latency for small models (linear/tree models, tiny nets) where GPU data-transfer overhead dominates the actual compute." },
        { type: "callout", variant: "gotcha", text: "**Measure before and after every optimization — on realistic load.** \"It should be faster\" is not data. A GPU can be *slower* than a CPU for a tiny model because moving inputs across the PCIe bus costs more than the multiply-adds saved. Benchmark with production-shaped inputs and concurrency, look at p99 not just the mean, and keep the change only if the number moved." },
      ]
    },

    {
      id: "scaling",
      title: "Scaling & infrastructure",
      level: "core",
      body: [
        { type: "p", text: "One container serves one machine's worth of traffic. Real load needs many replicas, a way to spread requests across them, and a controller that adds/removes replicas as demand changes. This is the infrastructure layer — mostly generic web-service scaling, with a few ML-specific twists (GPUs are expensive and slow to start)." },
        { type: "heading", text: "Load balancing & horizontal scaling" },
        { type: "p", text: "Because a well-built service is **stateless** (the model is read-only; each request is independent), you scale *horizontally*: run N identical replicas and put a **load balancer** in front to distribute requests across them. This is also how you get zero-downtime deploys (roll replicas one at a time) and fault tolerance (one replica dies, the balancer routes around it)." },
        { type: "heading", text: "Autoscaling" },
        { type: "p", text: "**Autoscaling** adjusts the replica count automatically from a signal — CPU/GPU utilization, request rate, or queue depth. Traffic doubles at 9am, scale up; it drops overnight, scale down and stop paying. The ML wrinkle is **slow cold starts**: a replica that must pull a multi-GB image and load a large model onto a GPU can take minutes to become ready, so reactive autoscaling lags behind spikes." },
        { type: "callout", variant: "tip", text: "**Scale-to-zero vs. warm pools.** Scaling GPU replicas to zero when idle saves serious money but adds a multi-minute cold start to the next request. For spiky-but-latency-sensitive traffic, keep a small **warm pool** of always-on replicas and autoscale *above* that floor. The right floor is a direct cost-vs-tail-latency dial — tune it to the traffic shape, don't leave it at the default." },
        { type: "heading", text: "Kubernetes, briefly" },
        { type: "p", text: "**Kubernetes (K8s)** is the industry-standard orchestrator: you declare the *desired state* (\"I want 3 replicas of this image, each with 1 GPU, behind this service\") and K8s continuously makes reality match — restarting crashed pods, rolling out new versions, and scaling. The core objects you'll touch:" },
        { type: "table",
          headers: ["K8s object", "Role"],
          rows: [
            ["**Pod**", "the smallest unit: one (or a few) containers running together"],
            ["**Deployment**", "manages a set of identical pods; handles rollouts & self-healing"],
            ["**Service**", "a stable network address + load balancing across a Deployment's pods"],
            ["**HPA** (HorizontalPodAutoscaler)", "adds/removes pods based on a metric (CPU, custom, queue depth)"],
            ["**Ingress**", "routes external HTTP traffic to Services (paths, TLS, hostnames)"],
          ]
        },
        { type: "code", lang: "text", code: "# deployment.yaml — 3 replicas of the API, each requesting a slice of a GPU\napiVersion: apps/v1\nkind: Deployment\nmetadata: { name: house-api }\nspec:\n  replicas: 3\n  selector: { matchLabels: { app: house-api } }\n  template:\n    metadata: { labels: { app: house-api } }\n    spec:\n      containers:\n        - name: api\n          image: house-api:1.3.0\n          ports: [ { containerPort: 8000 } ]\n          readinessProbe:              # don't send traffic until ready\n            httpGet: { path: /health, port: 8000 }\n          resources:\n            limits: { nvidia.com/gpu: 1 }" },
        { type: "callout", variant: "note", text: "**You rarely hand-write this for ML.** Higher-level layers sit on K8s and generate it for you: **KServe** and **Seldon Core** give you an `InferenceService` CRD with autoscaling, canary rollout, and pluggable runtimes (vLLM/Triton/etc.) out of the box. Learn the K8s primitives so you can debug, but adopt a serving layer so you're not maintaining YAML by hand." },
        { type: "heading", text: "Serverless inference & cost" },
        { type: "p", text: "**Serverless** endpoints (SageMaker Serverless, Modal, Replicate, Baseten, cloud-run-style GPU) flip the model: you hand over a container and they run it only when called, billing per-request or per-second with autoscaling — including to zero. Great for spiky or low-volume workloads where a 24/7 GPU would idle expensively; the trade-off is cold-start latency and less control. **The cost lens dominates every decision here:** GPUs are the biggest line item in most ML systems, so the real optimization goal is *maximizing GPU utilization* — batching, right-sizing, and scaling to zero all serve that one end." },
      ]
    },

    {
      id: "edge",
      title: "Edge & browser inference",
      level: "deep",
      body: [
        { type: "p", text: "Not every model runs in a data center. Running inference **on-device** — in a browser, on a phone, on a Raspberry Pi — removes network latency, works offline, and keeps data private (nothing leaves the device). The cost is tight compute and memory budgets, which is exactly where the portable formats and quantization from earlier pay off." },
        { type: "table",
          headers: ["Runtime", "Runs where", "Use for"],
          rows: [
            ["**ONNX Runtime Web**", "the browser (WASM / WebGPU)", "run an exported `.onnx` model client-side, no server"],
            ["**TensorFlow.js**", "browser & Node.js", "JS-native models; WebGL/WebGPU acceleration"],
            ["**llama.cpp + GGUF**", "laptops, phones, edge CPUs", "quantized LLMs on-device (GGUF is its quantized weight format)"],
            ["**Core ML / TFLite**", "iOS / Android", "native mobile inference using the phone's NPU"],
          ]
        },
        { type: "p", text: "The common thread: **export to a portable format** (ONNX or GGUF), **quantize hard** (int8/int4 — device memory is scarce), and run on a lightweight runtime with no Python. `llama.cpp` in particular made it routine to run multi-billion-parameter LLMs on a laptop CPU by combining an efficient C++ engine with aggressive GGUF quantization." },
        { type: "callout", variant: "tip", text: "**The browser is now a real inference target.** With WebGPU, ONNX Runtime Web and Transformers.js run genuine models (embeddings, small LLMs, vision) entirely client-side — zero server cost, zero data egress, instant offline. For privacy-sensitive or cost-sensitive features, shipping the model *to* the user instead of the data *to* the model is increasingly the right call." },
      ]
    },

    {
      id: "worked-example",
      title: "End-to-end: train → save → serve → containerize → test",
      level: "core",
      body: [
        { type: "p", text: "Everything above, assembled into one runnable pipeline. We train a tiny regression model, save the *whole pipeline* (so no train/serve skew), wrap it in FastAPI, put it in Docker, and hit it with `curl`. This is a complete, minimal production service — the skeleton you can grow from." },
        { type: "heading", text: "1. Train and save the whole pipeline" },
        { type: "code", lang: "py", code: "# train.py\nimport joblib\nfrom sklearn.datasets import fetch_california_housing\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import GradientBoostingRegressor\nfrom sklearn.model_selection import train_test_split\n\nX, y = fetch_california_housing(return_X_y=True)\nXtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)\n\n# scaler + model as ONE object -> the scaler is fit on train and travels with it\npipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42))\npipe.fit(Xtr, ytr)\nprint(\"R^2:\", pipe.score(Xte, yte))\n\njoblib.dump(pipe, \"model.joblib\")     # one artifact: preprocessing + model\nprint(\"saved model.joblib\")" },
        { type: "heading", text: "2. The FastAPI service" },
        { type: "code", lang: "py", code: "# app.py\nfrom contextlib import asynccontextmanager\nfrom fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel, Field\nimport joblib, numpy as np\n\nml = {}\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    ml[\"pipe\"] = joblib.load(\"model.joblib\")   # load once at boot\n    ml[\"version\"] = \"1.0.0\"\n    yield\n    ml.clear()\n\napp = FastAPI(title=\"California Housing\", lifespan=lifespan)\n\nclass House(BaseModel):                          # 8 California-housing features\n    features: list[float] = Field(..., min_length=8, max_length=8)\n\n@app.get(\"/health\")\nasync def health():\n    return {\"status\": \"ok\", \"model_loaded\": \"pipe\" in ml}\n\n@app.post(\"/predict\")\ndef predict(h: House):                            # sync def -> runs in threadpool\n    try:\n        x = np.array(h.features).reshape(1, -1)\n        price = float(ml[\"pipe\"].predict(x)[0])   # pipeline scales THEN predicts\n    except Exception as e:\n        raise HTTPException(500, f\"inference failed: {e}\")\n    return {\"median_house_value\": price, \"model_version\": ml[\"version\"]}" },
        { type: "heading", text: "3. Pin the environment & containerize" },
        { type: "code", lang: "text", code: "# requirements.txt  (pin exact versions for reproducibility)\nfastapi==0.115.6\nuvicorn==0.34.0\nscikit-learn==1.6.1\njoblib==1.4.2\nnumpy==2.2.1" },
        { type: "code", lang: "text", code: "# Dockerfile\nFROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY app.py model.joblib ./\nEXPOSE 8000\nCMD [\"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]" },
        { type: "heading", text: "4. Build, run, and test with curl" },
        { type: "code", lang: "bash", code: "python train.py                       # -> model.joblib\ndocker build -t housing:1.0.0 .\ndocker run -d -p 8000:8000 housing:1.0.0\n\ncurl localhost:8000/health\n# {\"status\":\"ok\",\"model_loaded\":true}\n\ncurl -X POST localhost:8000/predict \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"features\": [8.3, 41, 6.9, 1.02, 322, 2.55, 37.88, -122.23]}'\n# {\"median_house_value\": 4.13, \"model_version\": \"1.0.0\"}" },
        { type: "callout", variant: "good", text: "**That's a real deployment.** A model trained, serialized as a self-contained pipeline, served behind a validated API, frozen into a reproducible image, and callable by anyone over HTTP. From here the path forward is *operational*, not conceptual: push the image to a registry, deploy it to K8s or a serverless endpoint, add metrics/logging, and put autoscaling in front. The hard part — crossing the gap — is done." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Deployment is a skill you can only learn by *shipping*. Reading about Docker teaches recognition; getting a container to actually serve traffic teaches fluency. Do at least two of these end-to-end — the goal each time is a URL someone else can hit." },
        { type: "list", ordered: true, items: [
          "**Deploy a model as a public API.** Take any model you've trained, wrap it in FastAPI (health + predict endpoints, Pydantic validation), containerize it, and deploy to a free/cheap host (Render, Railway, Fly.io, Hugging Face Spaces, or a cloud run service). Share the URL and hit it from a different machine. This is the capstone — everything else supports it.",
          "**Serialize four ways, compare.** Save the same model as joblib, `torch.save`, safetensors, and ONNX. Measure file size and load time for each, then load the ONNX version with ONNX Runtime and confirm the predictions match the original (within float tolerance).",
          "**Benchmark the optimization ladder.** Take one model and measure p50/p99 latency and throughput as: (a) eager PyTorch, (b) ONNX Runtime, (c) int8-quantized. Plot latency vs. batch size for each. See the throughput/latency trade-off with your own numbers.",
          "**Shrink the image.** Start from a naive `python:3.11` Dockerfile, record the image size, then apply `-slim`, `--no-cache-dir`, a `.dockerignore`, and a multi-stage build. Track the size after each change and aim to cut it by more than half.",
          "**Serve an LLM with vLLM.** Run `vllm serve` on a small open model (locally or on a rented GPU), then call its OpenAI-compatible endpoint from Python. Fire many concurrent requests and watch continuous batching keep throughput high — compare against a naive one-at-a-time loop.",
          "**Add load-based autoscaling.** Deploy your API to Kubernetes (kind/minikube locally is fine) with a Deployment, Service, readiness probe, and an HPA. Drive load with a tool like `hey` or `locust` and watch replicas scale up under load and back down when it stops.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "Serving tooling moves fast; these are the sources worth tracking. Start with Made With ML and Full Stack Deep Learning for the *system-level* view, then the framework docs for depth." },
        { type: "link", url: "https://madewithml.com/", text: "Made With ML (Goku Mohandas) — the best free, code-first course on MLOps: from model to production, end to end" },
        { type: "link", url: "https://fullstackdeeplearning.com/course/", text: "Full Stack Deep Learning — the canonical course on building & deploying real ML products (deployment, testing, monitoring)" },
        { type: "link", url: "https://fastapi.tiangolo.com/", text: "FastAPI documentation — the reference for the API layer; the tutorial covers async, validation, and deployment" },
        { type: "link", url: "https://onnx.ai/", text: "ONNX + ONNX Runtime docs — the portable model format and its fast cross-platform runtime" },
        { type: "link", url: "https://docs.bentoml.com/", text: "BentoML documentation — Pythonic model packaging, serving, and containerization" },
        { type: "link", url: "https://docs.vllm.ai/", text: "vLLM documentation — the standard for high-throughput LLM serving (PagedAttention, continuous batching)" },
        { type: "link", url: "https://huggingface.co/docs/safetensors", text: "safetensors — the safe, fast weight-serialization format (now under the PyTorch Foundation)" },
        { type: "link", url: "https://kserve.github.io/website/", text: "KServe — standardized model serving on Kubernetes (InferenceService CRD, autoscaling, canary rollout)" },
      ]
    },
  ],

  packages: [
    { name: "fastapi", why: "the standard framework for ML APIs — async, auto-validation via Pydantic, auto docs" },
    { name: "uvicorn", why: "the ASGI server that runs FastAPI apps in dev and production (`--workers`)" },
    { name: "pydantic", why: "typed request/response schemas; validates untrusted input before the model sees it" },
    { name: "joblib", why: "efficient serialization of scikit-learn estimators and pipelines (better than raw pickle)" },
    { name: "safetensors", why: "safe, fast, code-free weight format — the HF/PyTorch default, immune to pickle attacks" },
    { name: "onnx / onnxruntime", why: "framework-agnostic graph + weights, and a fast portable runtime (2–5× vs eager)" },
    { name: "bentoml", why: "Pythonic packaging → API + Docker image + K8s deploy, with adaptive batching" },
    { name: "vllm", why: "high-throughput LLM serving via PagedAttention + continuous batching" },
  ],

  gotchas: [
    "**Never unpickle/`joblib.load` an untrusted file** — it runs arbitrary code on load. Use `safetensors` for weights you receive over a network.",
    "Load the model **once at startup** (FastAPI `lifespan`), never inside the endpoint — loading per-request turns a 5 ms prediction into 500 ms.",
    "Forgetting `model.eval()` + `torch.no_grad()` at inference leaves dropout/batchnorm active — predictions become noisy and batch-dependent, silently wrong.",
    "Inside a container you **must** bind `--host 0.0.0.0`; the default `127.0.0.1` makes `-p 8000:8000` map to nothing (connection refused).",
    "**Train/serve skew:** serialize the whole preprocessing+model pipeline as one artifact, or serving applies different scaling than training and the model sees inputs it never saw.",
    "Heavy CPU work in an `async def` endpoint blocks the entire event loop — use a plain `def` (threadpool) or a real serving framework instead.",
    "GPU CUDA versions must match across driver, base image, and framework wheel, or you get `CUDA error: no kernel image is available` only at runtime.",
    "A GPU can be **slower** than a CPU for tiny models — PCIe transfer overhead exceeds the compute saved. Measure p99 on realistic load before assuming.",
  ],

  flashcards: [
    { q: "Batch vs online vs streaming inference — when do you pick each?", a: "Batch: score a table on a schedule (freshness in hours is fine) — much simpler. Online: synchronous HTTP, millisecond latency at request time. Streaming: continuous predictions off an event stream." },
    { q: "Why prefer safetensors over pickle/`torch.save`?", a: "Pickle executes arbitrary code on load (a supply-chain RCE risk). safetensors stores only raw tensors + a JSON header — no code runs, and it memory-maps for ~76× faster CPU loads." },
    { q: "What does ONNX give you that a `.pt` or `.joblib` doesn't?", a: "Framework independence: it stores the full computation graph + weights as a portable protobuf, runnable in C++/JS/Rust/edge via ONNX Runtime or TensorRT — often 2–5× faster than eager PyTorch." },
    { q: "Where should you load the model in a FastAPI app, and why?", a: "Once at startup via `lifespan`, into shared state. Loading inside the endpoint re-deserializes on every request, destroying latency and memory." },
    { q: "Why can `async def` hurt a CPU-bound predict endpoint?", a: "CPU work doesn't `await`, so it blocks the single event loop and freezes all other requests. Use a plain `def` (FastAPI runs it in a threadpool) or offload it." },
    { q: "What two ideas make LLM serving (vLLM) fast?", a: "PagedAttention — manage the KV-cache in fixed pages like OS virtual memory, ending fragmentation. Continuous batching — swap a new request into a slot the instant one sequence finishes, keeping the GPU full." },
    { q: "Latency vs throughput — how does batching relate them?", a: "Bigger batches raise throughput (better hardware utilization) but raise each request's latency (it waits for batch-mates). Interactive → optimize latency (p99); offline → optimize throughput." },
    { q: "What is train/serve skew and how do you prevent it?", a: "Serving preprocesses inputs differently than training did (e.g. a fresh scaler vs the fitted one). Prevent it by serializing the entire preprocessing+model pipeline as one artifact." },
    { q: "Why is `--host 0.0.0.0` required in a container?", a: "The default `127.0.0.1` binds only the container's loopback, so published-port mapping reaches nothing. `0.0.0.0` accepts traffic on all interfaces." },
    { q: "What do a Deployment, Service, and HPA do in Kubernetes?", a: "Deployment manages a self-healing set of identical pods and rollouts; Service gives a stable address + load balancing across them; HPA adds/removes pods based on a metric." },
  ],

  cheatsheet: [
    { label: "Save sklearn", code: "joblib.dump(pipe, 'model.joblib')" },
    { label: "Save torch weights", code: "torch.save(model.state_dict(), 'w.pt')" },
    { label: "Save safetensors", code: "save_file(model.state_dict(), 'm.safetensors')" },
    { label: "Export ONNX", code: "torch.onnx.export(model, dummy, 'm.onnx', opset_version=17)" },
    { label: "Run ONNX", code: "ort.InferenceSession('m.onnx').run(None, {'input': x})" },
    { label: "Eval mode", code: "model.eval(); torch.no_grad()" },
    { label: "Run FastAPI (prod)", code: "uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4" },
    { label: "Build image", code: "docker build -t api:1.0 ." },
    { label: "Run container", code: "docker run -p 8000:8000 api:1.0" },
    { label: "GPU container", code: "docker run --gpus all -p 8000:8000 api:1.0" },
    { label: "Quantize (int8)", code: "torch.quantization.quantize_dynamic(model, {nn.Linear}, torch.qint8)" },
    { label: "Serve an LLM", code: "vllm serve meta-llama/Llama-3.1-8B-Instruct" },
    { label: "K8s scale", code: "kubectl scale deploy/house-api --replicas=5" },
    { label: "Test endpoint", code: "curl -X POST localhost:8000/predict -d '{...}'" },
  ],
});
