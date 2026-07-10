(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "nn-pytorch",
  name: "PyTorch in Depth",
  language: "Neural Networks",
  group: "Neural Networks",
  navLabel: "PyTorch",
  tagline: "The framework you'll build everything in — tensors, autograd (reverse-mode AD, for real), `nn.Module`, the training loop, and GPU/mixed-precision, from first principles.",
  color: "#EA580C",
  readMinutes: 50,
  sections: [
    {
      id: "why",
      title: "Why PyTorch: define-by-run vs. static graphs",
      level: "core",
      body: [
        { type: "p", text: "You have already built a neural net and backprop **from scratch in NumPy**. That was the point: you now know that a network is matrices, non-linearities, and a chain-rule pass that flows gradients backward. PyTorch is the tool that does the mechanical parts — the gradient bookkeeping, the GPU kernels, the batching — so you can spend your attention on the model, not the calculus. Everything you learned transfers directly; PyTorch's `tensor` is NumPy's `ndarray` with two superpowers bolted on: **it can live on a GPU**, and **it remembers how it was computed so it can differentiate itself**." },
        { type: "p", text: "PyTorch was released by Meta (then Facebook) AI Research in 2016 and, by around 2019, had become the dominant framework in research — and increasingly in production. The reason is one design decision: it is **define-by-run** (a *dynamic* computational graph). The graph of operations is built on the fly, as your ordinary Python executes, and thrown away after each backward pass." },
        { type: "table",
          headers: ["", "Define-and-run (static)", "Define-by-run (dynamic)"],
          rows: [
            ["Examples", "TensorFlow 1.x, Theano, Caffe", "PyTorch, TF2 eager, JAX"],
            ["When the graph is built", "once, up front, then executed", "every forward pass, as code runs"],
            ["Control flow (`if`, `while`)", "special graph ops (`tf.cond`)", "plain Python — just works"],
            ["Debugging", "opaque; errors at graph-run time", "`print()`/`pdb` mid-model; Python stack traces"],
            ["Variable-length inputs (RNNs, trees)", "awkward", "natural"],
          ]
        },
        { type: "p", text: "The trade was historically speed: a static graph can be compiled and optimized whole. PyTorch has since largely closed that gap — `torch.compile` (introduced in PyTorch 2.0, 2023) traces your dynamic code and JIT-compiles fused kernels, giving you static-graph speed without giving up define-by-run ergonomics." },
        { type: "heading", text: "The ecosystem you'll actually use" },
        { type: "table",
          headers: ["Library", "What it gives you"],
          rows: [
            ["**torchvision / torchaudio / torchtext**", "datasets, pretrained models, and domain transforms"],
            ["**Hugging Face** (`transformers`, `datasets`)", "pretrained LLMs/transformers + a training `Trainer`; the NLP/LLM tracks live here"],
            ["**PyTorch Lightning / `accelerate`**", "removes training-loop boilerplate, multi-GPU, mixed precision for free"],
            ["**`timm`, `einops`**", "SOTA vision backbones; readable tensor reshaping"],
          ]
        },
        { type: "callout", variant: "note", text: "**Install & GPU check.** Install the build matched to your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) (it picks the right wheel for you). Then the first thing you ever run: `import torch; print(torch.cuda.is_available())`. If that prints `False` on a machine with a GPU, your CUDA/driver/PyTorch versions are mismatched — fix that before anything else, because CPU training of anything real is painfully slow. On Apple Silicon the device is `\"mps\"` instead of `\"cuda\"`." },
        { type: "code", lang: "py", code: "import torch\nprint(torch.__version__)              # e.g. 2.5.1+cu124\nprint(torch.cuda.is_available())      # True on a working CUDA box\nprint(torch.cuda.get_device_name(0))  # e.g. 'NVIDIA A100-SXM4-40GB'\n\n# The one-liner every script starts with: pick a device once, reuse it.\ndevice = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\nprint(\"training on:\", device)" },
      ]
    },

    {
      id: "tensors",
      title: "Tensors: the array with a GPU and a gradient",
      level: "core",
      body: [
        { type: "p", text: "A **tensor** is an $n$-dimensional array — the exact same object as a NumPy `ndarray`, and the API is deliberately almost identical. A scalar is a 0-D tensor, a vector is 1-D, a matrix is 2-D, and a batch of RGB images is a 4-D tensor of shape $(N, C, H, W)$. Everything you know about NumPy broadcasting and axes carries over unchanged." },
        { type: "heading", text: "Creation, dtypes, shapes" },
        { type: "code", lang: "py", code: "import torch\n\ntorch.tensor([[1., 2.], [3., 4.]])   # from data; infers dtype float32\ntorch.zeros(2, 3)                    # (2,3) of 0.0\ntorch.ones(2, 3)\ntorch.arange(0, 10, 2)               # tensor([0,2,4,6,8])\ntorch.linspace(0, 1, 5)              # 5 points from 0..1\ntorch.randn(2, 3)                    # standard-normal samples\ntorch.eye(3)                         # identity\ntorch.full((2, 2), 7.0)\ntorch.zeros_like(x)                  # match shape+dtype+device of x\n\nx = torch.randn(3, 4)\nprint(x.shape)    # torch.Size([3, 4])  -> also x.size()\nprint(x.dtype)    # torch.float32\nprint(x.device)   # cpu\nprint(x.ndim)     # 2" },
        { type: "callout", variant: "gotcha", text: "**Dtype matters more than in NumPy.** Model weights and inputs are `float32` by default; **integer** tensors cannot have gradients. A classic beginner error is building an input with `torch.tensor([1, 2, 3])` (inferred `int64`) and feeding it to a `float32` layer — you get `RuntimeError: expected scalar type Long but found Float`. Fix with `.float()` or `dtype=torch.float32`. Class **labels** for cross-entropy, conversely, must stay `int64` (`long`)." },
        { type: "heading", text: "Indexing, slicing, broadcasting" },
        { type: "p", text: "Indexing is NumPy indexing: slices, boolean masks, fancy indexing, and `None`/`newaxis` to insert dimensions. Broadcasting follows the identical right-aligned rule from the Linear Algebra section — align shapes from the right, dimensions are compatible if equal or one is $1$." },
        { type: "code", lang: "py", code: "x = torch.arange(12).reshape(3, 4)\nx[0]           # first row -> shape (4,)\nx[:, 1]        # second column -> shape (3,)\nx[x > 5]       # boolean mask -> 1-D of matching elements\nx[:, None].shape   # (3, 1, 4)  -> None inserts an axis\n\n# Broadcasting: add a per-column bias to every row, no loop.\nX = torch.randn(1000, 4)\nb = torch.tensor([1., 2., 3., 4.])   # (4,)\n(X + b).shape                        # (1000, 4)  -> b stretches down rows" },
        { type: "heading", text: "reshape vs. view — the contiguity trap" },
        { type: "p", text: "Both `.view()` and `.reshape()` change a tensor's shape without moving data. The difference is subtle and bites everyone once: `.view()` requires the tensor to be **contiguous** in memory and returns a *view* (shares storage, zero-copy); `.reshape()` returns a view when it can and silently makes a *copy* when it can't. Operations like transpose (`.T`, `.transpose`, `.permute`) produce non-contiguous tensors." },
        { type: "code", lang: "py", code: "x = torch.arange(6)\nx.view(2, 3)          # ok: contiguous\n\ny = torch.randn(3, 4).T   # transpose -> NON-contiguous\n# y.view(-1)            # RuntimeError: view size is not compatible ...\ny.reshape(-1)          # works (copies)\ny.contiguous().view(-1)  # explicit: make contiguous, then view\n\n# -1 means 'infer this dimension'. Only one -1 allowed.\ntorch.randn(4, 3, 2).view(4, -1).shape   # (4, 6)\ntorch.randn(2, 3).flatten().shape        # (6,)\ntorch.randn(2, 3).unsqueeze(0).shape     # (1, 2, 3) -> add batch dim\ntorch.randn(1, 3).squeeze().shape        # (3,) -> drop size-1 dims" },
        { type: "callout", variant: "tip", text: "**Rule of thumb:** reach for `.reshape()` by default (it always works); use `.view()` only in hot loops where you *want* the error that tells you a copy would happen. Use `.unsqueeze(dim)` / `.squeeze(dim)` to add/drop size-1 axes — far more readable than `[None]` indexing when adding a batch dimension." },
        { type: "heading", text: "Devices: moving to the GPU" },
        { type: "p", text: "A tensor lives on exactly one device. Operations require all operands on the **same** device, or you get `RuntimeError: Expected all tensors to be on the same device`. You move with `.to(device)` (or `.cuda()` / `.cpu()`)." },
        { type: "code", lang: "py", code: "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n\nx = torch.randn(3, 3)         # starts on CPU\nx = x.to(device)              # move to GPU (returns a NEW tensor)\n\n# Create directly on device (avoids a CPU->GPU copy):\ny = torch.randn(3, 3, device=device)\nz = x @ y                     # both on GPU -> result on GPU\n\n# .to() is a no-op if already there, so 'x = x.to(device)' is always safe." },
        { type: "callout", variant: "gotcha", text: "**`.to(device)` returns a new tensor; it does NOT move in place.** `x.to(device)` alone does nothing — you must write `x = x.to(device)`. (`nn.Module.to()` *is* in-place for a model's parameters — one of PyTorch's few asymmetries. So `model.to(device)` needs no assignment, but `tensor.to(device)` does.)" },
        { type: "heading", text: "NumPy interop — and the shared-memory gotcha" },
        { type: "code", lang: "py", code: "import numpy as np, torch\n\na = np.array([1., 2., 3.])\nt = torch.from_numpy(a)     # zero-copy: SHARES memory with a\nt2 = t.numpy()              # zero-copy back (CPU tensors only)\n\nt.add_(100)                 # in-place add\nprint(a)                    # [101. 102. 103.] -> a changed too!\n\n# A GPU tensor must come home first:\ng = torch.randn(3, device=device)\n# g.numpy()                 # RuntimeError\ng.cpu().numpy()             # correct: move to CPU, then convert" },
        { type: "callout", variant: "warn", text: "`torch.from_numpy` and `.numpy()` **share the underlying buffer** for CPU tensors — mutating one mutates the other. That is a feature (no copy) and a footgun (spooky action at a distance). Call `.clone()` if you want an independent copy." },
      ]
    },

    {
      id: "autograd",
      title: "Autograd: this IS reverse-mode automatic differentiation",
      level: "core",
      body: [
        { type: "p", text: "This is the section where the framework earns its keep. In the from-scratch backprop section you hand-derived $\\partial \\mathcal{L}/\\partial W$ for every layer and coded the backward pass yourself. **Autograd does exactly that, automatically, for any computation you write.** It is not magic and it is not numerical (finite-difference) differentiation — it is *reverse-mode automatic differentiation*, the precise algorithm you already implemented. PyTorch just builds the graph and applies the chain rule for you." },
        { type: "heading", text: "requires_grad and the computational graph" },
        { type: "p", text: "Set `requires_grad=True` on a tensor (leaf) and PyTorch starts **recording** every operation that touches it, building a directed graph of the computation as it runs. Each node stores the local derivative rule (its `grad_fn`) needed to backprop through it. This graph is the exact same DAG from the autodiff section." },
        { type: "code", lang: "py", code: "import torch\n\nx = torch.tensor(3.0, requires_grad=True)\ny = x**2 + 2*x + 1        # y = (x+1)^2 ;  dy/dx = 2x + 2\nprint(y)                  # tensor(16., grad_fn=<AddBackward0>)\nprint(y.grad_fn)          # <AddBackward0> -> the recorded op\n\ny.backward()              # reverse-mode pass: fill in gradients\nprint(x.grad)             # tensor(8.)   == 2*3 + 2  ✓" },
        { type: "p", text: "The `grad_fn` on `y` is the tail of a linked list of backward functions — `AddBackward`, `PowBackward`, etc. Calling `.backward()` walks that graph from output to inputs, multiplying local Jacobians via the chain rule, and **accumulates** the result into each leaf's `.grad`. For the general case where the output is a vector, `.backward()` needs a vector to seed the reverse pass — but for a **scalar loss** (the usual case) it defaults to $1$, which is exactly why loss functions return a single number." },
        { type: "math", tex: String.raw`\text{forward: } \; x \to \cdots \to \mathcal{L} \qquad\quad \text{backward: } \; \frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial u}\,\frac{\partial u}{\partial x} \;\text{ chained back through the graph}` },
        { type: "callout", variant: "gotcha", text: "**Gradients ACCUMULATE — they are not overwritten.** Every `.backward()` *adds* into `.grad`. If you don't zero them between steps, gradients from all previous iterations pile up and training diverges. This is precisely why the training loop has `optimizer.zero_grad()` — remember it as a non-negotiable ritual. (The accumulation behavior is actually useful for gradient accumulation across micro-batches, which is why it's the default.)" },
        { type: "heading", text: "no_grad and detach — turning recording off" },
        { type: "p", text: "Recording the graph costs memory (every intermediate tensor is kept so its gradient can be computed later — remember this when we hit CUDA OOM). When you are **not** going to call `.backward()` — inference, evaluation, or a manual parameter update — you should switch recording off." },
        { type: "code", lang: "py", code: "# 1) torch.no_grad(): a context that disables graph-building entirely.\nwith torch.no_grad():\n    preds = model(x_eval)      # forward only, no memory for backward\n\n# 2) .detach(): returns a tensor sharing data but CUT from the graph.\nz = y.detach()                 # z has requires_grad=False\n\n# 3) The manual SGD update MUST be under no_grad, or you'd record\n#    the update itself into the graph:\nwith torch.no_grad():\n    for p in model.parameters():\n        p -= lr * p.grad" },
        { type: "table",
          headers: ["Tool", "What it does", "Use when"],
          rows: [
            ["`with torch.no_grad():`", "disables graph recording in the block", "eval / inference / manual updates"],
            ["`x.detach()`", "one tensor, forgets its history", "logging a value, stopping a gradient path"],
            ["`x.item()`", "extract a Python number from a 1-elem tensor", "printing/accumulating the loss scalar"],
            ["`model.eval()`", "switches BatchNorm/Dropout to eval mode", "**orthogonal** to no_grad — see §7"],
          ]
        },
        { type: "callout", variant: "warn", text: "**`model.eval()` and `torch.no_grad()` are different things and you usually want both.** `eval()` changes *layer behavior* (Dropout off, BatchNorm uses running stats); `no_grad()` changes *whether gradients are tracked*. Neither implies the other. Evaluation code should be inside `with torch.no_grad():` **and** after `model.eval()`." },
        { type: "callout", variant: "tip", text: "Accumulating a running loss with `total += loss` secretly keeps the **entire graph** of every batch alive (a slow memory leak that ends in OOM). Always accumulate the detached scalar: `total += loss.item()`." },
      ]
    },

    {
      id: "nn-module",
      title: "nn.Module: how models are built",
      level: "core",
      body: [
        { type: "p", text: "You *could* keep raw weight tensors with `requires_grad=True` and update them by hand — and doing it once is a great exercise. But real models have thousands of parameters across dozens of layers, and you need them all tracked, moved to the GPU together, saved together, and handed to an optimizer together. That is what **`nn.Module`** provides: a container that auto-registers any `nn.Parameter` or sub-module you assign as an attribute." },
        { type: "heading", text: "The two ways to define a model" },
        { type: "p", text: "**`nn.Sequential`** is a shortcut for a straight-line stack of layers — concise but rigid (no branching, no reused inputs). **Subclassing `nn.Module`** is the real, general way: you define layers in `__init__` and describe the data flow in `forward`. Anything with a skip connection, multiple inputs, or conditional logic needs subclassing." },
        { type: "code", lang: "py", code: "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n# (a) Sequential — good for simple stacks.\nmlp = nn.Sequential(\n    nn.Linear(784, 256),\n    nn.ReLU(),\n    nn.Linear(256, 10),\n)\n\n# (b) Subclass — the general pattern you'll use everywhere.\nclass MLP(nn.Module):\n    def __init__(self, in_dim=784, hidden=256, out_dim=10):\n        super().__init__()                 # ALWAYS call this first\n        self.fc1 = nn.Linear(in_dim, hidden)\n        self.fc2 = nn.Linear(hidden, out_dim)\n        self.dropout = nn.Dropout(0.2)\n\n    def forward(self, x):\n        x = F.relu(self.fc1(x))            # (B, hidden)\n        x = self.dropout(x)\n        return self.fc2(x)                 # (B, out_dim) logits\n\nmodel = MLP()\nlogits = model(torch.randn(32, 784))       # call model(x), NOT model.forward(x)\nprint(logits.shape)                        # torch.Size([32, 10])" },
        { type: "callout", variant: "gotcha", text: "**Call `model(x)`, never `model.forward(x)`.** The `__call__` wrapper runs registered forward/backward *hooks* and other bookkeeping around your `forward`; calling `forward` directly skips them and will silently break things like hooks and some quantization/compile paths. Also: `nn.Linear`, `nn.ReLU`, etc. are **classes you instantiate**; their functional twins `F.relu`, `F.linear` are stateless functions you call inline. ReLU has no parameters, so `F.relu` in `forward` is common; layers *with* weights (Linear, Conv) must be created in `__init__` so their parameters are registered." },
        { type: "heading", text: "Parameters are auto-registered and auto-tracked" },
        { type: "code", lang: "py", code: "# Any nn.Parameter or sub-Module assigned to self is discovered:\nfor name, p in model.named_parameters():\n    print(name, tuple(p.shape), p.requires_grad)\n# fc1.weight (256, 784) True\n# fc1.bias   (256,)     True\n# fc2.weight (10, 256)  True\n# fc2.bias   (10,)      True\n\ntotal = sum(p.numel() for p in model.parameters())\nprint(f\"{total:,} trainable parameters\")   # 203,530\n\nmodel.to(device)   # moves ALL parameters to the GPU in one call (in place)" },
        { type: "heading", text: "The common layers you'll reach for" },
        { type: "table",
          headers: ["Layer", "Shape map", "Used for"],
          rows: [
            ["`nn.Linear(in, out)`", "$(B, \\text{in}) \\to (B, \\text{out})$", "fully-connected / MLP"],
            ["`nn.Conv2d(cin, cout, k)`", "$(B, C_{in}, H, W) \\to (B, C_{out}, H', W')$", "images (see CNN section)"],
            ["`nn.Embedding(V, d)`", "int ids $(B, L) \\to (B, L, d)$", "tokens → vectors (NLP)"],
            ["`nn.LayerNorm(d)` / `nn.BatchNorm1d`", "shape-preserving", "normalization / stable training"],
            ["`nn.Dropout(p)`", "shape-preserving", "regularization (train only)"],
            ["`nn.ReLU` / `nn.GELU`", "elementwise", "non-linearity"],
          ]
        },
        { type: "callout", variant: "tip", text: "`nn.Linear(in, out)` stores its weight with shape `(out, in)` and computes $y = xW^\\top + b$ — the transpose is why a batch $x$ of shape $(B, \\text{in})$ maps cleanly to $(B, \\text{out})$. This matches the $Wx+b$ layer from the Linear Algebra section, just batched." },
      ]
    },

    {
      id: "loss-optim",
      title: "Losses, optimizers & the canonical training loop",
      level: "core",
      body: [
        { type: "p", text: "A model turns inputs into predictions; a **loss** turns (prediction, target) into a single scalar measuring wrongness; an **optimizer** turns gradients into a parameter update. PyTorch gives you a library of each so you rarely code them by hand — but you already derived the important ones (MSE, cross-entropy, SGD) from scratch, so none of this is a black box." },
        { type: "heading", text: "Loss functions (torch.nn)" },
        { type: "table",
          headers: ["Loss", "Task", "Note"],
          rows: [
            ["`nn.MSELoss()`", "regression", "mean squared error $\\frac1n\\sum(\\hat y - y)^2$"],
            ["`nn.CrossEntropyLoss()`", "multi-class classification", "takes **raw logits** + int class labels"],
            ["`nn.BCEWithLogitsLoss()`", "binary / multi-label", "takes logits; sigmoid built in (stable)"],
            ["`nn.NLLLoss()`", "classification", "takes `log_softmax` output (rarely needed directly)"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**`nn.CrossEntropyLoss` expects raw logits, NOT probabilities — do not put a `softmax` before it.** It fuses `log_softmax + NLLLoss` internally for numerical stability. Adding your own softmax first double-applies it and quietly wrecks training. Its targets are **integer class indices** of dtype `int64` and shape `(B,)` — *not* one-hot vectors. This pair of mistakes is the single most common PyTorch classification bug." },
        { type: "heading", text: "Optimizers (torch.optim)" },
        { type: "p", text: "You hand the optimizer the model's parameters once; thereafter it holds references to them and their `.grad` buffers, and `step()` applies the update rule you derived in the optimization section (SGD, momentum, Adam)." },
        { type: "code", lang: "py", code: "import torch.optim as optim\n\nopt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)\n# or, the modern default that 'just works' on most problems:\nopt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)\n# AdamW decouples weight decay from the gradient step (preferred for transformers):\nopt = optim.AdamW(model.parameters(), lr=3e-4)" },
        { type: "heading", text: "The training loop, annotated line by line" },
        { type: "p", text: "Every PyTorch training loop, from a toy MLP to a billion-parameter transformer, is this same five-move dance. Memorize its shape; you will type it hundreds of times." },
        { type: "code", lang: "py", code: "model.train()                          # (0) enable Dropout / BatchNorm training mode\nfor epoch in range(num_epochs):\n    for xb, yb in train_loader:        # one mini-batch at a time\n        xb, yb = xb.to(device), yb.to(device)   # move data to model's device\n\n        opt.zero_grad()                # (1) clear old gradients (they accumulate!)\n        logits = model(xb)             # (2) FORWARD: build the graph, get predictions\n        loss = loss_fn(logits, yb)     # (3) compute the scalar loss\n        loss.backward()                # (4) BACKWARD: reverse-mode AD fills every .grad\n        opt.step()                     # (5) UPDATE: optimizer nudges params down-gradient\n\n        running += loss.item()         # .item(): detach the scalar, avoid a memory leak" },
        { type: "table",
          headers: ["Step", "Call", "What happens under the hood"],
          rows: [
            ["1", "`opt.zero_grad()`", "sets every `param.grad` to zero (or `None`) so this batch starts clean"],
            ["2", "`model(xb)`", "runs `forward`, recording the computational graph"],
            ["3", "`loss_fn(logits, yb)`", "reduces to one scalar — the root of the backward graph"],
            ["4", "`loss.backward()`", "walks the graph output→input, accumulating `∂loss/∂param` into each `.grad`"],
            ["5", "`opt.step()`", "reads `.grad`, applies the update rule, mutates params in place"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Order is sacred: `zero_grad → forward → loss → backward → step`.** The two classic swaps: calling `step()` before `backward()` (updates with stale/no gradients), and forgetting `zero_grad()` (gradients accumulate across batches and the loss explodes). `zero_grad(set_to_none=True)` is the default in modern PyTorch and is slightly faster/more memory-friendly than zeroing." },
      ]
    },

    {
      id: "data",
      title: "Dataset & DataLoader: feeding the loop",
      level: "core",
      body: [
        { type: "p", text: "The training loop's `for xb, yb in train_loader` hides the entire data pipeline: how examples are stored, batched, shuffled, and (optionally) loaded in parallel. PyTorch splits this into two abstractions. A **`Dataset`** answers 'give me example $i$' — it defines `__len__` and `__getitem__`. A **`DataLoader`** wraps a `Dataset` and handles batching, shuffling, and multiprocessing. This separation is the whole design: your `Dataset` worries about *one* example; the `DataLoader` worries about *throughput*." },
        { type: "code", lang: "py", code: "from torch.utils.data import Dataset, DataLoader\nimport torch\n\nclass ToyDataset(Dataset):\n    def __init__(self, X, y):\n        self.X = torch.as_tensor(X, dtype=torch.float32)\n        self.y = torch.as_tensor(y, dtype=torch.long)\n    def __len__(self):\n        return len(self.X)                 # how many examples total\n    def __getitem__(self, i):\n        return self.X[i], self.y[i]        # ONE (features, label) pair\n\nds = ToyDataset(X_train, y_train)\nloader = DataLoader(\n    ds,\n    batch_size=64,      # examples per batch -> DataLoader stacks them\n    shuffle=True,       # reshuffle every epoch (train only; False for val/test)\n    num_workers=4,      # subprocesses prefetching batches in parallel\n    pin_memory=True,    # faster CPU->GPU copies when training on CUDA\n    drop_last=False,    # keep the ragged final batch\n)\n\nfor xb, yb in loader:\n    print(xb.shape, yb.shape)   # (64, n_features) (64,)\n    break" },
        { type: "callout", variant: "tip", text: "**`shuffle=True` on train, `False` on validation/test.** Shuffling breaks any accidental ordering the model could memorize; on eval it's pointless and makes logs harder to compare. The `DataLoader` automatically **collates** — stacks — the per-example tensors from `__getitem__` into a batched tensor with a leading batch dimension. For variable-length data (text) you pass a custom `collate_fn` to pad within each batch." },
        { type: "heading", text: "num_workers and transforms" },
        { type: "p", text: "`num_workers > 0` spawns background processes that fetch and preprocess the *next* batches while the GPU is busy on the current one — this overlaps data loading with compute and is often the difference between 30% and 95% GPU utilization. **Transforms** are preprocessing functions applied in `__getitem__` (via torchvision they compose cleanly): normalize, augment, tensor-ify." },
        { type: "code", lang: "py", code: "from torchvision import datasets, transforms\n\ntfm = transforms.Compose([\n    transforms.ToTensor(),                         # PIL/np -> (C,H,W) float in [0,1]\n    transforms.Normalize((0.5,), (0.5,)),          # (x - mean) / std  -> ~[-1, 1]\n])\n\ntrain_ds = datasets.FashionMNIST(\n    root=\"./data\", train=True, download=True, transform=tfm)\ntrain_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)" },
        { type: "callout", variant: "gotcha", text: "**More workers is not always faster**, and on Windows/notebooks `num_workers>0` can hang or error unless the entry point is guarded by `if __name__ == '__main__':`. Start with `num_workers=4`, `pin_memory=True` on GPU, and tune from there. If the very first batch takes forever, your `__getitem__` is doing too much work per call — move heavy one-time preprocessing into `__init__`." },
      ]
    },

    {
      id: "state",
      title: "Train/eval modes, saving, checkpointing & seeds",
      level: "core",
      body: [
        { type: "heading", text: "model.train() vs model.eval()" },
        { type: "p", text: "Some layers behave differently during training and inference. **Dropout** randomly zeros activations while training but must be a no-op at test time; **BatchNorm** uses the *current batch's* statistics during training but *running averages* during eval. `model.train()` and `model.eval()` flip a single boolean (`self.training`) that every such layer reads. Forget `model.eval()` at test time and your accuracy will be mysteriously noisy and wrong." },
        { type: "code", lang: "py", code: "model.train()   # dropout ON, batchnorm uses batch stats\n# ... training loop ...\n\nmodel.eval()    # dropout OFF, batchnorm uses running stats\nwith torch.no_grad():          # AND turn off gradient tracking\n    for xb, yb in val_loader:\n        preds = model(xb.to(device)).argmax(dim=1)" },
        { type: "heading", text: "Saving & loading: state_dict, not the whole object" },
        { type: "p", text: "A model's learnable state is its **`state_dict`** — an ordered dict mapping parameter names to tensors. The recommended way to save is to persist *this dict*, not the pickled model object (which brittly couples the file to your exact class definition and file layout)." },
        { type: "code", lang: "py", code: "# SAVE (recommended: weights only)\ntorch.save(model.state_dict(), \"model.pt\")\n\n# LOAD: re-create the architecture, then pour the weights in.\nmodel = MLP()                                   # same class/shape as saved\nmodel.load_state_dict(torch.load(\"model.pt\", map_location=device))\nmodel.eval()                                    # remember to switch to eval\n\n# CHECKPOINT: to resume training you need more than weights.\ncheckpoint = {\n    \"epoch\": epoch,\n    \"model\": model.state_dict(),\n    \"optim\": opt.state_dict(),      # optimizer momentum/Adam buffers matter!\n    \"loss\": best_loss,\n}\ntorch.save(checkpoint, \"ckpt.pt\")\n\nckpt = torch.load(\"ckpt.pt\", map_location=device)\nmodel.load_state_dict(ckpt[\"model\"])\nopt.load_state_dict(ckpt[\"optim\"])\nstart_epoch = ckpt[\"epoch\"] + 1" },
        { type: "callout", variant: "gotcha", text: "**Save the optimizer state too, or resumed training will stutter.** Adam/momentum keep per-parameter running buffers; restart without them and the first steps after resuming are effectively cold. Also, for inference-only loads, prefer `torch.load(..., weights_only=True)` (the safe default in recent PyTorch) — plain pickle loading can execute arbitrary code from an untrusted file." },
        { type: "heading", text: "Reproducibility: seeds" },
        { type: "p", text: "Random weight init, shuffling, and dropout all draw from RNGs. To make a run reproducible, seed **every** source. Full bitwise determinism on GPU additionally needs deterministic algorithms enabled (and is sometimes slower)." },
        { type: "code", lang: "py", code: "import torch, numpy as np, random\n\ndef set_seed(seed=42):\n    random.seed(seed)\n    np.random.seed(seed)\n    torch.manual_seed(seed)\n    torch.cuda.manual_seed_all(seed)\n    # For strict determinism (optional, can be slower):\n    torch.backends.cudnn.deterministic = True\n    torch.backends.cudnn.benchmark = False\n\nset_seed(42)" },
      ]
    },

    {
      id: "gpu",
      title: "GPU training, mixed precision & why CUDA runs out of memory",
      level: "core",
      body: [
        { type: "p", text: "Training on a GPU is mostly the loop you already have, plus one discipline: **model and data must be on the same device**. You move the model once (`model.to(device)`) and every batch inside the loop (`xb.to(device)`)." },
        { type: "heading", text: "Mixed precision (autocast + GradScaler)" },
        { type: "p", text: "By default everything is `float32` (32-bit). **Automatic mixed precision (AMP)** runs the heavy matmuls/convolutions in `float16`/`bfloat16` — roughly halving memory and often doubling throughput on modern GPUs — while keeping numerically-sensitive parts in `float32`. `autocast` picks the dtype per-op automatically. A **`GradScaler`** multiplies the loss by a large factor before `backward()` so tiny `float16` gradients don't underflow to zero, then unscales before the optimizer step." },
        { type: "code", lang: "py", code: "from torch.amp import autocast, GradScaler\n\nscaler = GradScaler(\"cuda\")\nfor xb, yb in train_loader:\n    xb, yb = xb.to(device), yb.to(device)\n    opt.zero_grad()\n    with autocast(\"cuda\", dtype=torch.float16):   # ops run in fp16 where safe\n        logits = model(xb)\n        loss = loss_fn(logits, yb)\n    scaler.scale(loss).backward()   # scale up so small grads don't underflow\n    scaler.step(opt)                # unscale + optimizer.step()\n    scaler.update()                 # adapt the scale factor for next iter" },
        { type: "callout", variant: "tip", text: "`bfloat16` (on Ampere+/A100/H100) has the same exponent range as `float32`, so it rarely underflows and often needs **no** `GradScaler` at all — many modern recipes just use `autocast(dtype=torch.bfloat16)`. Use `float16` + `GradScaler` on older GPUs where bf16 isn't supported." },
        { type: "heading", text: "Why you get CUDA out-of-memory — and it's not the weights" },
        { type: "p", text: "The most common shock for beginners: a model with a few million parameters (tens of MB) OOMs on a 24 GB GPU. The parameters are almost never the problem. GPU memory during training is dominated by **activations cached for the backward pass**. Recall from the autograd section: to compute $\\partial \\mathcal{L}/\\partial W$, the backward pass needs the *inputs* to each layer's forward — so the graph keeps every intermediate tensor alive until `backward()` runs. Memory therefore scales with **batch size × model depth × activation size**, not just parameter count." },
        { type: "table",
          headers: ["What eats GPU memory", "Scales with", "Lever to reduce it"],
          rows: [
            ["Parameters", "model size", "smaller model (usually not the issue)"],
            ["**Activations (cached for backward)**", "**batch × depth × width**", "smaller `batch_size`; gradient checkpointing"],
            ["Optimizer state (Adam: 2× params)", "params × optimizer", "SGD, or 8-bit optimizers"],
            ["Gradients", "params", "gradient accumulation over micro-batches"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**First aid for `CUDA out of memory`:** (1) lower `batch_size` — it's the biggest lever and directly shrinks cached activations; (2) wrap eval in `torch.no_grad()` so no activations are cached at all; (3) use mixed precision (halves activation memory); (4) accumulate `loss.item()`, not `loss`, so old graphs get freed; (5) enable **gradient checkpointing** to trade compute for memory (recompute activations in backward instead of storing them). `torch.cuda.empty_cache()` frees PyTorch's cached allocator blocks but does *not* fix a genuine leak." },
      ]
    },

    {
      id: "end-to-end",
      title: "End-to-end: FashionMNIST from data to saved model",
      level: "core",
      body: [
        { type: "p", text: "Everything above, assembled into one runnable script: load a real dataset, define a model, train it, evaluate held-out accuracy, and save. FashionMNIST is 60k 28×28 grayscale clothing images in 10 classes — a drop-in, harder replacement for MNIST that still trains in a couple of minutes on a CPU." },
        { type: "code", lang: "py", code: "import torch, torch.nn as nn, torch.nn.functional as F\nfrom torch.utils.data import DataLoader\nfrom torchvision import datasets, transforms\n\ndevice = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\ntorch.manual_seed(42)\n\n# 1) DATA -----------------------------------------------------------\ntfm = transforms.Compose([transforms.ToTensor(),\n                          transforms.Normalize((0.2860,), (0.3530,))])\ntrain_ds = datasets.FashionMNIST(\"./data\", train=True,  download=True, transform=tfm)\ntest_ds  = datasets.FashionMNIST(\"./data\", train=False, download=True, transform=tfm)\ntrain_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)\ntest_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)\n\n# 2) MODEL ----------------------------------------------------------\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(28*28, 256)\n        self.fc2 = nn.Linear(256, 128)\n        self.fc3 = nn.Linear(128, 10)\n        self.drop = nn.Dropout(0.2)\n    def forward(self, x):\n        x = x.view(x.size(0), -1)          # flatten (B,1,28,28) -> (B,784)\n        x = F.relu(self.fc1(x))\n        x = self.drop(F.relu(self.fc2(x)))\n        return self.fc3(x)                 # logits (B,10)\n\nmodel = Net().to(device)\nloss_fn = nn.CrossEntropyLoss()\nopt = torch.optim.Adam(model.parameters(), lr=1e-3)\n\n# 3) TRAIN ----------------------------------------------------------\ndef train_one_epoch():\n    model.train()\n    total = 0.0\n    for xb, yb in train_loader:\n        xb, yb = xb.to(device), yb.to(device)\n        opt.zero_grad()\n        loss = loss_fn(model(xb), yb)\n        loss.backward()\n        opt.step()\n        total += loss.item() * xb.size(0)\n    return total / len(train_ds)\n\n# 4) EVALUATE -------------------------------------------------------\n@torch.no_grad()\ndef evaluate(loader):\n    model.eval()\n    correct = 0\n    for xb, yb in loader:\n        xb, yb = xb.to(device), yb.to(device)\n        preds = model(xb).argmax(dim=1)\n        correct += (preds == yb).sum().item()\n    return correct / len(loader.dataset)\n\nfor epoch in range(1, 6):\n    tr_loss = train_one_epoch()\n    acc = evaluate(test_loader)\n    print(f\"epoch {epoch}  train_loss {tr_loss:.4f}  test_acc {acc:.3%}\")\n\n# 5) SAVE -----------------------------------------------------------\ntorch.save(model.state_dict(), \"fashion_mlp.pt\")\nprint(\"saved -> fashion_mlp.pt\")\n# Expected: test accuracy climbs to ~88-89% in 5 epochs." },
        { type: "callout", variant: "good", text: "Notice how little of this is new: it is §2–§7 clicked together. The `@torch.no_grad()` **decorator** on `evaluate` is a tidy alternative to the `with` block — the whole function runs without graph tracking. `argmax(dim=1)` turns logits into predicted class indices; `(preds == yb).sum().item()` counts hits. Swap the MLP for a small CNN (two `Conv2d` + `MaxPool` layers) and the same loop takes you past ~91%." },
      ]
    },

    {
      id: "debugging",
      title: "Debugging: shape errors, .item(), and the usual pitfalls",
      level: "core",
      body: [
        { type: "p", text: "90% of PyTorch bugs are one of three things: a **shape mismatch**, a **device mismatch**, or a **dtype mismatch**. The define-by-run design is your friend here — you can drop a `print(x.shape)` or a breakpoint anywhere in `forward` and it just runs, because it *is* ordinary Python." },
        { type: "heading", text: "Read the error, then print the shape" },
        { type: "code", lang: "py", code: "# The single most useful debugging habit: print shapes through forward.\ndef forward(self, x):\n    print(\"in     \", x.shape)          # (B, 1, 28, 28)\n    x = x.view(x.size(0), -1)\n    print(\"flat   \", x.shape)          # (B, 784)\n    x = F.relu(self.fc1(x))\n    print(\"hidden \", x.shape)          # (B, 256)\n    return self.fc2(x)\n\n# 'mat1 and mat2 shapes cannot be multiplied (32x784 and 256x10)'\n#  -> a Linear's in_features doesn't match the incoming last dim. Trace it." },
        { type: "table",
          headers: ["Error message (paraphrased)", "Real cause", "Fix"],
          rows: [
            ["`shapes cannot be multiplied (Axb and cxd)`", "a `Linear`'s `in_features` ≠ incoming feature dim", "match `nn.Linear(in, ..)` to the flattened size"],
            ["`Expected all tensors on same device`", "model on GPU, batch on CPU (or vice-versa)", "`xb = xb.to(device)` in the loop"],
            ["`expected scalar type Long but found Float`", "int input to a float layer, or float labels", "`.float()` inputs; labels stay `.long()`"],
            ["`element 0 ... does not require grad`", "input has no grad path / under `no_grad`", "check `requires_grad`; not inside `no_grad`"],
            ["loss is `nan`", "lr too high, or `log(0)` / bad normalization", "lower `lr`; check for `softmax`+`CrossEntropy`"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**`.item()` only works on a one-element tensor** and pulls a Python `float`/`int` out of it — use it for logging (`loss.item()`), never inside the computation (it breaks the graph). Its cousin `.tolist()` handles bigger tensors. And a subtle one: an **in-place** op (`x += 1`, `x.relu_()`, trailing underscore) on a tensor that autograd still needs raises `a leaf Variable that requires grad is being used in an in-place operation` — prefer out-of-place ops inside `forward`." },
        { type: "callout", variant: "tip", text: "**Fast sanity checks before a long run:** (1) overfit a *single batch* on purpose — a correct model should drive its loss to ~0 in a few dozen steps; if it can't, the bug is in the model/loss, not the data. (2) Verify `loss` is a scalar (`.shape == torch.Size([])`) before `.backward()`. (3) Check `param.grad` is not `None` after `backward()` — if it is, that parameter isn't connected to the loss. (4) Print `loss.item()` every few steps — it should trend down." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Do at least two of these end-to-end. Reading PyTorch builds recognition; typing the training loop from memory builds fluency. Aim to write the loop *without* looking it up." },
        { type: "list", ordered: true, items: [
          "**Re-implement your from-scratch net in PyTorch.** Take the NumPy MLP + backprop you wrote earlier and rebuild it as an `nn.Module`. Train both on the same data and confirm the losses track each other — this proves autograd is doing exactly what your hand-coded backward pass did.",
          "**FashionMNIST, MLP → CNN.** Start from the §9 script. First reproduce ~88% with the MLP, then replace it with a small CNN (two `nn.Conv2d` + `nn.MaxPool2d`, then `Linear`) and push past 91%. Plot train vs. test accuracy per epoch and identify where overfitting begins.",
          "**Overfit-a-single-batch harness.** Write a `sanity_check(model, batch)` helper that trains on one batch for 200 steps and asserts the loss drops below 0.01. Make it the first thing you run for every new model. Deliberately introduce a bug (e.g. a `softmax` before `CrossEntropyLoss`) and watch it fail.",
          "**Checkpoint & resume.** Add full checkpointing (model + optimizer + epoch) to a training run. Kill it mid-training, then resume from the last checkpoint and confirm the loss curve continues smoothly rather than jumping — proving you saved the optimizer state correctly.",
          "**Mixed-precision speed-up.** Take a CNN that nearly fills your GPU. Measure images/sec and peak memory (`torch.cuda.max_memory_allocated()`) in `float32`, then add `autocast` + `GradScaler`. Report the throughput and memory change, and confirm final accuracy is unchanged.",
          "**A custom `Dataset` from raw files.** Point a `Dataset` at a folder of your own images or a CSV, implement `__getitem__` with transforms, wrap it in a `DataLoader`, and train a classifier — the skill you'll use on every real project where the data isn't a tidy built-in.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "PyTorch is best learned by building; these are the resources that will take you from 'I can follow the loop' to 'I can design the model,' in recommended order:" },
        { type: "link", url: "https://pytorch.org/tutorials/beginner/basics/intro.html", text: "Official PyTorch tutorials — 'Learn the Basics' (tensors → autograd → training loop; the canonical starting point)" },
        { type: "link", url: "https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html", text: "Deep Learning with PyTorch: A 60 Minute Blitz — the fastest end-to-end intro" },
        { type: "link", url: "https://www.manning.com/books/deep-learning-with-pytorch", text: "Stevens, Antiga & Viehmann — 'Deep Learning with PyTorch' (the book; free PDF from the authors, deep on tensors & autograd)" },
        { type: "link", url: "https://karpathy.ai/zero-to-hero.html", text: "Andrej Karpathy — Neural Networks: Zero to Hero (build micrograd, then GPT; the best intuition for what autograd is doing)" },
        { type: "link", url: "https://pytorch.org/docs/stable/notes/autograd.html", text: "PyTorch docs — Autograd mechanics (the precise semantics of the computational graph, in-place ops, and no_grad)" },
        { type: "link", url: "https://pytorch.org/docs/stable/amp.html", text: "PyTorch docs — Automatic Mixed Precision (autocast + GradScaler reference)" },
        { type: "link", url: "https://lightning.ai/docs/pytorch/stable/", text: "PyTorch Lightning docs — when you're ready to stop writing the loop boilerplate by hand" },
      ]
    },
  ],

  packages: [
    { name: "torch", why: "tensors, autograd, `nn`, `optim` — the whole framework" },
    { name: "torch.nn", why: "layers, losses, and the `Module` base class for models" },
    { name: "torch.optim", why: "SGD, Adam, AdamW and the `.step()`/`.zero_grad()` API" },
    { name: "torch.utils.data", why: "`Dataset` + `DataLoader` for batching, shuffling, parallel loading" },
    { name: "torchvision", why: "datasets (FashionMNIST, CIFAR), transforms, pretrained vision models" },
    { name: "torch.amp", why: "`autocast` + `GradScaler` for mixed-precision training" },
    { name: "transformers (HF)", why: "pretrained transformers/LLMs + `Trainer` — the NLP/LLM tracks" },
    { name: "pytorch-lightning", why: "removes training-loop boilerplate; free multi-GPU + AMP" },
  ],

  gotchas: [
    "Gradients **accumulate**; you must call `optimizer.zero_grad()` every step or the loss diverges.",
    "`nn.CrossEntropyLoss` takes **raw logits + int64 labels** — never put a `softmax` before it, and labels are class indices, not one-hot.",
    "`tensor.to(device)` returns a **new** tensor (assign it: `x = x.to(device)`); `model.to(device)` is in place.",
    "Use `loss.item()` when accumulating/printing — `total += loss` keeps the whole graph alive and leaks memory to OOM.",
    "`model.eval()` and `torch.no_grad()` are different and both needed at test time: one changes Dropout/BatchNorm, the other stops grad tracking.",
    "CUDA OOM is usually **activations cached for backward** (scale with batch×depth), not the parameters — lower `batch_size` first.",
    "`.view()` needs a contiguous tensor and fails after a transpose; use `.reshape()` (or `.contiguous().view(...)`) when unsure.",
    "`torch.from_numpy(a)` and `t.numpy()` **share memory** with the array on CPU — mutating one mutates the other; `.clone()` to break the link.",
  ],

  flashcards: [
    { q: "What kind of differentiation does autograd perform, and how does it relate to backprop?", a: "**Reverse-mode automatic differentiation** — the exact algorithm as hand-coded backprop. It records a computational graph on the forward pass, then `.backward()` applies the chain rule output→input, accumulating `∂loss/∂param` into each `.grad`." },
    { q: "Name the five steps of the training loop in order.", a: "`optimizer.zero_grad()` → `logits = model(x)` (forward) → `loss = loss_fn(logits, y)` → `loss.backward()` (reverse-mode AD) → `optimizer.step()` (update)." },
    { q: "Why must you call `optimizer.zero_grad()` each iteration?", a: "Gradients **accumulate** into `.grad` rather than overwrite; without zeroing, gradients from all past batches pile up and training diverges." },
    { q: "What input does `nn.CrossEntropyLoss` expect?", a: "**Raw logits** (no softmax — it fuses `log_softmax`+`NLLLoss` internally) and **integer class indices** (`int64`, shape `(B,)`), not one-hot vectors." },
    { q: "Difference between `model.eval()` and `torch.no_grad()`?", a: "`eval()` changes **layer behavior** (Dropout off, BatchNorm uses running stats); `no_grad()` stops **gradient tracking** to save memory. They're orthogonal — use both at test time." },
    { q: "Your tiny model OOMs on a big GPU. Most likely cause?", a: "**Activations cached for the backward pass**, which scale with batch × depth, not parameter count. Lower `batch_size`, use `no_grad()` for eval, or gradient checkpointing." },
    { q: "`.view()` vs `.reshape()`?", a: "`.view()` needs a contiguous tensor and returns a zero-copy view (fails after transpose); `.reshape()` returns a view when possible and silently copies otherwise. Default to `.reshape()`." },
    { q: "What do `autocast` and `GradScaler` do in mixed precision?", a: "`autocast` runs ops in fp16/bf16 where numerically safe (halving memory, speeding up matmuls); `GradScaler` scales the loss up before backward so small fp16 gradients don't underflow, then unscales before the step." },
    { q: "Why save `state_dict()` instead of the whole model, and what else does a resume checkpoint need?", a: "The `state_dict` (name→tensor) decouples weights from the class/file layout and is safer to load. To *resume* training you also need the **optimizer** state (Adam/momentum buffers) and the epoch." },
    { q: "What does `loss.item()` do and why use it?", a: "Extracts a Python scalar from a 1-element tensor, detached from the graph — safe for logging/accumulation. Using `loss` directly in a sum keeps the graph alive and leaks memory." },
    { q: "How do you check a GPU is available and move computation to it?", a: "`torch.cuda.is_available()`; then `device = torch.device('cuda' if ... else 'cpu')`, `model.to(device)` once, and `xb = xb.to(device)` per batch. All operands must share a device." },
  ],

  cheatsheet: [
    { label: "Pick device", code: "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')" },
    { label: "Move to GPU", code: "x = x.to(device)   # model.to(device) is in place" },
    { label: "Track gradient", code: "x = torch.tensor(3., requires_grad=True)" },
    { label: "Backward pass", code: "loss.backward()   # then read x.grad" },
    { label: "No-grad block", code: "with torch.no_grad(): preds = model(x)" },
    { label: "Define model", code: "class Net(nn.Module):\n    def __init__(self): super().__init__(); ...\n    def forward(self, x): ..." },
    { label: "Loss + optimizer", code: "loss_fn = nn.CrossEntropyLoss(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)" },
    { label: "Training step", code: "opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()" },
    { label: "DataLoader", code: "DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)" },
    { label: "Train / eval mode", code: "model.train()  /  model.eval()" },
    { label: "Save / load weights", code: "torch.save(model.state_dict(), 'm.pt'); model.load_state_dict(torch.load('m.pt'))" },
    { label: "Predicted class", code: "logits.argmax(dim=1)" },
    { label: "Scalar for logging", code: "loss.item()" },
    { label: "Mixed precision", code: "with autocast('cuda'): loss = loss_fn(model(x), y)" },
  ],
});
