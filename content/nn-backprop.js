(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "nn-backprop",
  name: "Backpropagation",
  language: "Neural Networks",
  group: "Neural Networks",
  navLabel: "Backpropagation",
  tagline: "The algorithm that trains every neural network — derived in full, then built from scratch in NumPy and checked numerically.",
  color: "#F97316",
  readMinutes: 50,
  sections: [
    {
      id: "problem",
      title: "The problem backprop solves",
      level: "core",
      body: [
        { type: "p", text: "A modern neural network is a function with millions to billions of tunable parameters. Training means finding parameter values that minimize a loss $\\mathcal{L}$. From the Calculus track you know the tool: **gradient descent**, which needs the gradient $\\nabla_\\theta \\mathcal{L}$ — the partial derivative of the loss with respect to *every single parameter*." },
        { type: "p", text: "So the entire game reduces to one question: **how do you compute the gradient with respect to millions of parameters efficiently?** The naive answer — nudge each parameter, re-run the network, see how the loss changes — costs one full forward pass *per parameter*. For a million parameters that is a million forward passes per training step. Hopeless." },
        { type: "p", text: "**Backpropagation** computes the *exact* gradient with respect to all parameters in a **single backward pass** — roughly the same cost as one forward pass. That is the whole reason deep learning is computationally feasible. It is not an approximation and it is not new: it is just the **chain rule from calculus applied in reverse**, organized so that no work is repeated." },
        { type: "callout", variant: "note", text: "**History.** Reverse-mode automatic differentiation was described by Seppo Linnainmaa in 1970. Its application to neural networks was popularized by Rumelhart, Hinton & Williams in their 1986 *Nature* paper \"Learning representations by back-propagating errors,\" which reignited neural-network research. Every framework — PyTorch, TensorFlow, JAX — is a general-purpose engine for exactly this algorithm." },
      ]
    },
    {
      id: "setup",
      title: "A concrete network to differentiate",
      level: "core",
      body: [
        { type: "p", text: "We will derive backprop on a fully-connected two-layer network (one hidden layer). Everything generalizes to arbitrary depth by induction. Fix notation for a single input vector $x \\in \\mathbb{R}^{n}$:" },
        { type: "math", tex: String.raw`\begin{aligned}
z^{[1]} &= W^{[1]} x + b^{[1]} & &\text{(hidden pre-activation)}\\
a^{[1]} &= \phi\!\left(z^{[1]}\right) & &\text{(hidden activation)}\\
z^{[2]} &= W^{[2]} a^{[1]} + b^{[2]} & &\text{(output pre-activation)}\\
\hat{y} &= \sigma\!\left(z^{[2]}\right) & &\text{(prediction)}\\
\mathcal{L} &= \ell(\hat{y}, y) & &\text{(scalar loss)}
\end{aligned}` },
        { type: "p", text: "Here $\\phi$ is a hidden non-linearity (ReLU, tanh, …), $\\sigma$ is the output map (sigmoid for binary, softmax for multiclass, identity for regression), and $\\ell$ is the loss. The shapes: if the hidden layer has $h$ units, then $W^{[1]}\\in\\mathbb{R}^{h\\times n}$, $b^{[1]}\\in\\mathbb{R}^{h}$, $W^{[2]}\\in\\mathbb{R}^{k\\times h}$, $b^{[2]}\\in\\mathbb{R}^{k}$ for $k$ outputs." },
        { type: "p", text: "The parameters we must differentiate are $\\theta = \\{W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}\\}$. Our goal: compute $\\dfrac{\\partial \\mathcal{L}}{\\partial W^{[1]}}, \\dfrac{\\partial \\mathcal{L}}{\\partial b^{[1]}}, \\dfrac{\\partial \\mathcal{L}}{\\partial W^{[2]}}, \\dfrac{\\partial \\mathcal{L}}{\\partial b^{[2]}}$." },
        { type: "callout", variant: "tip", text: "The forward pass is a **chain of functions**: $x \\to z^{[1]} \\to a^{[1]} \\to z^{[2]} \\to \\hat y \\to \\mathcal{L}$. Backprop walks this chain **right to left**, and the whole trick is that each step reuses the gradient computed by the step after it." },
      ]
    },
    {
      id: "chain-rule",
      title: "The engine: the multivariable chain rule",
      level: "core",
      body: [
        { type: "p", text: "Backprop is one idea applied repeatedly. If a scalar $\\mathcal{L}$ depends on a variable $u$ only *through* intermediate variables $v_1, \\dots, v_m$, the chain rule sums the contributions along every path:" },
        { type: "math", tex: String.raw`\frac{\partial \mathcal{L}}{\partial u} = \sum_{j=1}^{m} \frac{\partial \mathcal{L}}{\partial v_j}\,\frac{\partial v_j}{\partial u}` },
        { type: "p", text: "Read it as: *the sensitivity of the loss to $u$ equals, for each thing $u$ influences, (how much the loss cares about that thing) × (how much $u$ moves that thing), summed up.* In matrix form, if $v = f(u)$ with Jacobian $\\frac{\\partial v}{\\partial u}$, then gradients pull back by the **transpose** of the Jacobian:" },
        { type: "math", tex: String.raw`\nabla_u \mathcal{L} = \left(\frac{\partial v}{\partial u}\right)^{\!\top} \nabla_v \mathcal{L}` },
        { type: "p", text: "That transpose is why data flows *forward* through $W$ but gradients flow *backward* through $W^\\top$. Keep this one fact and the rest is mechanical." },
      ]
    },
    {
      id: "delta",
      title: "The key object: the error signal δ",
      level: "core",
      body: [
        { type: "p", text: "The insight that makes backprop efficient is to compute the gradient with respect to each **pre-activation** $z^{[l]}$ and reuse it. Define the **error signal** of layer $l$:" },
        { type: "math", tex: String.raw`\delta^{[l]} \;\equiv\; \frac{\partial \mathcal{L}}{\partial z^{[l]}}` },
        { type: "p", text: "Once you know $\\delta^{[l]}$, the parameter gradients for that layer fall out immediately, because $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$ is a simple linear function of $W^{[l]}$ and $b^{[l]}$:" },
        { type: "math", tex: String.raw`\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} \left(a^{[l-1]}\right)^{\top}, \qquad \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}` },
        { type: "p", text: "So the problem shrinks to: **find $\\delta$ at the output, then propagate it backward layer by layer.** Two rules do everything." },
      ]
    },
    {
      id: "derivation",
      title: "Full derivation of the two backprop rules",
      level: "core",
      body: [
        { type: "heading", text: "Rule 1 — the output-layer error" },
        { type: "p", text: "At the output, $\\delta^{[2]} = \\partial\\mathcal{L}/\\partial z^{[2]}$. By the chain rule through $\\hat y = \\sigma(z^{[2]})$:" },
        { type: "math", tex: String.raw`\delta^{[2]} = \frac{\partial \mathcal{L}}{\partial \hat y} \odot \sigma'\!\left(z^{[2]}\right)` },
        { type: "p", text: "where $\\odot$ is elementwise multiplication. Here is the beautiful part: for the **natural pairings** of output map and loss, this collapses to something trivial. For sigmoid + binary cross-entropy, and for softmax + categorical cross-entropy, and for identity + mean-squared-error, all the messy terms cancel and you get:" },
        { type: "math", tex: String.raw`\boxed{\;\delta^{[2]} = \hat y - y\;}` },
        { type: "p", text: "The gradient at the output is simply **prediction minus target**. (This cancellation is not luck — it is why these loss/activation pairs are the canonical choices; the Probability track shows they are all maximum-likelihood estimators.) Let us verify it for sigmoid + BCE with a single output:" },
        { type: "math", tex: String.raw`\mathcal{L} = -\big[y\log\hat y + (1-y)\log(1-\hat y)\big], \quad \hat y = \sigma(z), \quad \sigma'(z) = \hat y(1-\hat y)` },
        { type: "math", tex: String.raw`\frac{\partial \mathcal{L}}{\partial \hat y} = -\frac{y}{\hat y} + \frac{1-y}{1-\hat y} = \frac{\hat y - y}{\hat y(1-\hat y)}` },
        { type: "math", tex: String.raw`\delta = \frac{\partial \mathcal{L}}{\partial \hat y}\cdot \sigma'(z) = \frac{\hat y - y}{\hat y(1-\hat y)}\cdot \hat y(1-\hat y) = \hat y - y \;\checkmark` },
        { type: "heading", text: "Rule 2 — propagating error to the previous layer" },
        { type: "p", text: "Now push $\\delta^{[2]}$ back to the hidden layer. The loss depends on $z^{[1]}$ only through $z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$ and then $a^{[1]} = \\phi(z^{[1]})$. Apply the chain rule in two hops. First from $z^{[2]}$ to $a^{[1]}$ (Jacobian is $W^{[2]}$, so pull back by its transpose), then from $a^{[1]}$ to $z^{[1]}$ (elementwise, so multiply by $\\phi'$):" },
        { type: "math", tex: String.raw`\delta^{[1]} = \underbrace{\left(W^{[2]}\right)^{\top}\delta^{[2]}}_{\text{error pulled back through the weights}} \;\odot\; \underbrace{\phi'\!\left(z^{[1]}\right)}_{\text{local slope of the activation}}` },
        { type: "p", text: "That is the entire algorithm. For a network of depth $L$, the general recurrence is identical:" },
        { type: "math", tex: String.raw`\delta^{[L]} = \nabla_{\hat y}\mathcal{L}\odot\sigma'(z^{[L]}), \qquad \delta^{[l]} = \left(W^{[l+1]}\right)^{\top}\delta^{[l+1]} \odot \phi'\!\left(z^{[l]}\right)` },
        { type: "callout", variant: "note", text: "**Why it is called back-propagation:** the error $\\delta$ literally propagates backward through the same weights used in the forward pass, but transposed. Each layer receives the downstream error, multiplies by its local activation slope, and passes it further back. No path is ever recomputed — that reuse is what turns an exponential cost into a linear one." },
      ]
    },
    {
      id: "algorithm",
      title: "The backpropagation algorithm (batched)",
      level: "core",
      body: [
        { type: "p", text: "In practice we process a **minibatch** of $m$ examples at once. Stack them as rows of $X\\in\\mathbb{R}^{m\\times n}$; every activation becomes a matrix, and the per-example gradients are **averaged**. The algorithm in full:" },
        { type: "list", ordered: true, items: [
          "**Forward pass** — compute and *store* $Z^{[1]}, A^{[1]}, Z^{[2]}, \\hat Y$. (You must cache activations; the backward pass needs them.)",
          "**Loss** — evaluate $\\mathcal{L} = \\frac{1}{m}\\sum \\ell(\\hat y_i, y_i)$.",
          "**Output error** — $\\Delta^{[2]} = \\hat Y - Y$ (shape $m\\times k$).",
          "**Output grads** — $\\nabla_{W^{[2]}} = \\frac{1}{m}(A^{[1]})^{\\top}\\Delta^{[2]}$, $\\;\\nabla_{b^{[2]}} = \\frac{1}{m}\\sum_i \\Delta^{[2]}_i$.",
          "**Backprop to hidden** — $\\Delta^{[1]} = \\left(\\Delta^{[2]} (W^{[2]})^{\\top}\\right)\\odot \\phi'(Z^{[1]})$.",
          "**Hidden grads** — $\\nabla_{W^{[1]}} = \\frac{1}{m}X^{\\top}\\Delta^{[1]}$, $\\;\\nabla_{b^{[1]}} = \\frac{1}{m}\\sum_i \\Delta^{[1]}_i$.",
          "**Update** — gradient-descent step $\\theta \\leftarrow \\theta - \\eta\\,\\nabla_\\theta \\mathcal{L}$ on every parameter.",
        ]},
        { type: "callout", variant: "gotcha", text: "With rows-as-examples the transposes flip versus the single-example formulas: forward uses $XW^\\top$-style products and the weight gradient is $A^\\top\\Delta$. Getting these transposes right is 90% of debugging a from-scratch net — always sanity-check that each gradient has the **same shape** as the parameter it updates." },
      ]
    },
    {
      id: "from-scratch",
      title: "From scratch in NumPy: an MLP that learns XOR",
      level: "core",
      body: [
        { type: "p", text: "XOR is the classic test: it is *not* linearly separable, so a linear model cannot solve it, but a single hidden layer can. Here is a complete implementation — forward, backward, and training loop — in pure NumPy, no autograd." },
        { type: "code", lang: "py", code: "import numpy as np\nrng = np.random.default_rng(0)\n\n# XOR dataset: 4 points, not linearly separable\nX = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)\ny = np.array([[0],[1],[1],[0]], dtype=float)\n\ndef sigmoid(z): return 1.0 / (1.0 + np.exp(-z))\n\n# --- parameters (He-ish init) ---\nh = 8\nW1 = rng.normal(0, 1, (2, h)) * np.sqrt(2/2)\nb1 = np.zeros((1, h))\nW2 = rng.normal(0, 1, (h, 1)) * np.sqrt(2/h)\nb2 = np.zeros((1, 1))\n\nlr = 0.5\nfor epoch in range(5000):\n    # ---- forward (rows = examples) ----\n    Z1 = X @ W1 + b1          # (4,h)\n    A1 = np.tanh(Z1)          # hidden activation\n    Z2 = A1 @ W2 + b2         # (4,1)\n    Yh = sigmoid(Z2)          # prediction\n\n    # ---- loss: binary cross-entropy ----\n    eps = 1e-9\n    loss = -np.mean(y*np.log(Yh+eps) + (1-y)*np.log(1-Yh+eps))\n\n    # ---- backward ----\n    d2 = (Yh - y) / len(X)            # delta^[2] = yhat - y   (BCE+sigmoid)\n    dW2 = A1.T @ d2                   # (h,1)\n    db2 = d2.sum(axis=0, keepdims=True)\n    d1 = (d2 @ W2.T) * (1 - A1**2)    # tanh'(z) = 1 - tanh(z)^2\n    dW1 = X.T @ d1                    # (2,h)\n    db1 = d1.sum(axis=0, keepdims=True)\n\n    # ---- gradient-descent update ----\n    for p, g in [(W1,dW1),(b1,db1),(W2,dW2),(b2,db2)]:\n        p -= lr * g\n\n    if epoch % 1000 == 0:\n        print(f\"epoch {epoch:4d}  loss {loss:.4f}\")\n\nprint(\"predictions:\", sigmoid((np.tanh(X@W1+b1))@W2+b2).round(2).ravel())\n# -> ~[0.02, 0.98, 0.98, 0.02]  : it learned XOR" },
        { type: "callout", variant: "good", text: "Notice the backward block is *five lines* and mirrors the forward block exactly, in reverse. That symmetry is the signature of a correct backprop implementation. If you can write this from memory, you understand neural networks at the level that lets you debug anything." },
      ]
    },
    {
      id: "grad-check",
      title: "Gradient checking — never trust a gradient you haven't verified",
      level: "core",
      body: [
        { type: "p", text: "A subtle sign error in backprop still 'trains' (loss wobbles down slowly) but is wrong. The gold-standard test is to compare your analytic gradient against a **numerical** one from the limit definition of a derivative:" },
        { type: "math", tex: String.raw`\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta + \varepsilon e_i) - \mathcal{L}(\theta - \varepsilon e_i)}{2\varepsilon}` },
        { type: "p", text: "The two-sided (central) difference has error $O(\\varepsilon^2)$, far better than the one-sided version. Use $\\varepsilon \\approx 10^{-5}$ and require the relative error below $10^{-6}$." },
        { type: "code", lang: "py", code: "def numerical_grad(f, theta, eps=1e-5):\n    g = np.zeros_like(theta)\n    it = np.nditer(theta, flags=['multi_index'])\n    while not it.finished:\n        idx = it.multi_index\n        old = theta[idx]\n        theta[idx] = old + eps; fp = f()\n        theta[idx] = old - eps; fm = f()\n        theta[idx] = old\n        g[idx] = (fp - fm) / (2*eps)   # central difference\n        it.iternext()\n    return g\n\n# relative error: ||analytic - numeric|| / (||analytic|| + ||numeric||)\ndef rel_err(a, b):\n    return np.linalg.norm(a-b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)\n# A correct backprop gives rel_err < 1e-7 for every parameter tensor." },
        { type: "callout", variant: "tip", text: "Gradient checking is $O(\\text{params})$ forward passes — far too slow for training, but perfect as a **unit test** on a tiny network. Every from-scratch layer you ever write should ship with a gradient check. This one habit will save you days." },
      ]
    },
    {
      id: "vanishing",
      title: "Why deep networks are hard: vanishing & exploding gradients",
      level: "deep",
      body: [
        { type: "p", text: "The backward recurrence $\\delta^{[l]} = (W^{[l+1]})^\\top\\delta^{[l+1]}\\odot\\phi'(z^{[l]})$ multiplies by a weight matrix and an activation slope at *every* layer. Over many layers these factors compound multiplicatively — the gradient is a product of $L$ terms:" },
        { type: "math", tex: String.raw`\delta^{[1]} \sim \left(\prod_{l=2}^{L} (W^{[l]})^{\top}\,\mathrm{diag}\!\left(\phi'(z^{[l-1]})\right)\right)\delta^{[L]}` },
        { type: "p", text: "If the typical factor magnitude is below 1, the product decays exponentially — the **vanishing gradient** problem, and early layers barely learn. Above 1, it blows up — **exploding gradients**. Sigmoid/tanh make this worse because their derivative saturates near 0 for large inputs ($\\sigma'\\le 0.25$)." },
        { type: "p", text: "This single equation motivates a huge fraction of modern deep learning:" },
        { type: "table",
          headers: ["Fix", "What it does", "Attacks"],
          rows: [
            ["ReLU activations", "slope is exactly 1 for positive inputs", "vanishing"],
            ["He / Xavier init", "keeps signal variance ~constant across layers", "both"],
            ["Batch / layer norm", "renormalizes activations each layer", "both"],
            ["Residual connections", "adds an identity path so gradient can skip layers", "vanishing"],
            ["Gradient clipping", "caps the gradient norm", "exploding"],
          ]
        },
        { type: "callout", variant: "note", text: "These are covered in depth in the **Training Deep Networks** section. The point here: every one of them is a direct response to the compounding product in the backprop recurrence. Residual connections — the reason ResNets and Transformers can be hundreds of layers deep — exist precisely so that $\\delta$ has a shortcut home." },
      ]
    },
    {
      id: "autodiff",
      title: "From hand-derived backprop to autograd",
      level: "deep",
      body: [
        { type: "p", text: "Deriving $\\delta$ by hand for every architecture would be miserable. Frameworks generalize backprop into **reverse-mode automatic differentiation**: you build a **computational graph** of primitive operations during the forward pass, and each primitive knows its own local derivative (its *vector-Jacobian product*). Calling `.backward()` walks the graph in reverse topological order, applying the chain rule automatically." },
        { type: "code", lang: "py", code: "import torch\n\n# The same MLP gradient, computed by autograd instead of by hand.\nX = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)\ny = torch.tensor([[0.],[1.],[1.],[0.]])\n\nW1 = torch.randn(2, 8, requires_grad=True)\nb1 = torch.zeros(8, requires_grad=True)\nW2 = torch.randn(8, 1, requires_grad=True)\nb2 = torch.zeros(1, requires_grad=True)\n\nA1 = torch.tanh(X @ W1 + b1)\nYh = torch.sigmoid(A1 @ W2 + b2)\nloss = torch.nn.functional.binary_cross_entropy(Yh, y)\n\nloss.backward()          # reverse-mode autodiff fills in every .grad\nprint(W1.grad.shape)     # torch.Size([2, 8]) — same shape as W1\n# W1.grad here equals the dW1 you computed by hand above." },
        { type: "callout", variant: "tip", text: "Backprop *is* reverse-mode autodiff specialized to scalar loss. Understanding the hand derivation is what lets you read framework error messages, write custom layers with a correct `backward`, and reason about memory (activations must be kept until the backward pass — the origin of 'CUDA out of memory' during training). Andrej Karpathy's **micrograd** (~100 lines) is the canonical 'build autograd yourself' exercise — do it." },
      ]
    },
    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Backprop is a *doing* skill, not a reading skill. Implement at least the first two before moving on." },
        { type: "list", ordered: true, items: [
          "**MLP from scratch + gradient check.** Extend the XOR network to a general $n$-layer MLP class (`forward`, `backward`, `step`) in pure NumPy. Add a gradient checker and prove every layer's gradient matches numerically to $<10^{-7}$. Train it on `sklearn.datasets.make_moons`.",
          "**Build micrograd.** Implement a scalar-valued autograd engine: a `Value` class that records operations and a `backward()` that topologically sorts and applies the chain rule. Reproduce your XOR result through it. This is the single best exercise for *truly* understanding autodiff.",
          "**Break it on purpose.** Flip one sign in your backward pass and watch gradient checking catch it. Then remove the activation-derivative term and see training silently degrade — internalize what a subtle backprop bug looks like.",
          "**Vanishing gradients, measured.** Build a 20-layer sigmoid MLP and log the gradient norm at each layer. Plot it; watch it vanish. Swap sigmoid→ReLU and He init and watch the plot flatten. You will have *reproduced* the core result that unlocked deep learning.",
          "**MNIST from scratch.** Train a 2-layer MLP on MNIST (784→128→10, softmax + cross-entropy) with only NumPy backprop. Reaching ~97% test accuracy with no framework is a rite of passage.",
        ]},
      ]
    },
    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "This section is self-contained, but these are the best places to reinforce it:" },
        { type: "link", url: "https://www.youtube.com/watch?v=VMj-3S1tku0", text: "Andrej Karpathy — \"The spelled-out intro to neural networks and backpropagation: building micrograd\" (watch and code along; the best backprop lesson on the internet)" },
        { type: "link", url: "https://cs231n.github.io/optimization-2/", text: "Stanford CS231n — Backpropagation, intuitions (the classic course notes)" },
        { type: "link", url: "http://neuralnetworksanddeeplearning.com/chap2.html", text: "Michael Nielsen — Neural Networks and Deep Learning, Ch. 2 (the four backprop equations, derived carefully)" },
        { type: "link", url: "https://www.nature.com/articles/323533a0", text: "Rumelhart, Hinton & Williams (1986) — the original \"Learning representations by back-propagating errors\"" },
        { type: "link", url: "https://www.deeplearningbook.org/contents/mlp.html", text: "Goodfellow, Bengio & Courville — Deep Learning, Ch. 6.5 (back-propagation, formal treatment)" },
      ]
    },
  ],

  packages: [
    { name: "numpy", why: "implement forward/backward by hand — the only way to truly learn it" },
    { name: "torch.autograd", why: "reverse-mode autodiff; `.backward()` generalizes hand-derived backprop" },
    { name: "torch.nn.functional", why: "ready-made losses (cross_entropy, binary_cross_entropy) with fused stable gradients" },
    { name: "micrograd", why: "Karpathy's ~100-line scalar autograd engine — the canonical build-it-yourself" },
    { name: "torch.autograd.gradcheck", why: "framework-grade numerical gradient checking for custom layers" },
  ],

  gotchas: [
    "You must **cache forward activations** — the backward pass needs $a^{[l-1]}$ and $z^{[l]}$. This is why training uses far more memory than inference.",
    "Every parameter gradient must have the **same shape** as the parameter. A shape mismatch means a transpose is wrong.",
    "$\\delta^{[2]} = \\hat y - y$ only holds for the **matched** loss/activation pairs (softmax+CE, sigmoid+BCE, identity+MSE). Mix them and you must keep the full $\\odot\\,\\sigma'$ term.",
    "Forgetting the activation-derivative factor $\\phi'(z^{[1]})$ in $\\delta^{[1]}$ is the most common silent bug — training still limps along, masking it. Gradient-check.",
    "With rows-as-examples, remember to **average** gradients over the batch ($\\frac{1}{m}$) or your effective learning rate scales with batch size.",
    "`np.exp` in sigmoid/softmax overflows for large inputs — subtract the max (softmax) or clip, and add $\\varepsilon$ inside logs.",
    "Numerical gradient checking is only for **tiny** nets as a unit test — it costs one forward pass per parameter.",
  ],

  flashcards: [
    { q: "In one sentence, what is backpropagation?", a: "The chain rule applied in reverse over the computational graph, computing the exact gradient of a scalar loss w.r.t. all parameters in a single backward pass." },
    { q: "Define the error signal $\\delta^{[l]}$.", a: "$\\delta^{[l]} = \\partial\\mathcal{L}/\\partial z^{[l]}$, the gradient of the loss w.r.t. layer $l$'s pre-activation. Knowing it gives the weight/bias gradients immediately." },
    { q: "Given $\\delta^{[l]}$, what are the parameter gradients?", a: "$\\partial\\mathcal{L}/\\partial W^{[l]} = \\delta^{[l]}(a^{[l-1]})^\\top$ and $\\partial\\mathcal{L}/\\partial b^{[l]} = \\delta^{[l]}$." },
    { q: "What is the backprop recurrence for a hidden layer?", a: "$\\delta^{[l]} = (W^{[l+1]})^\\top\\delta^{[l+1]} \\odot \\phi'(z^{[l]})$ — pull error back through the transposed weights, then scale by the local activation slope." },
    { q: "Why does $\\delta = \\hat y - y$ at the output?", a: "For the canonical loss/activation pairs (sigmoid+BCE, softmax+cross-entropy, identity+MSE) the $\\sigma'$ term cancels the loss derivative. They're the maximum-likelihood choices." },
    { q: "How do you verify a backprop implementation?", a: "Numerical gradient checking: compare against the central difference $(\\mathcal{L}(\\theta+\\varepsilon)-\\mathcal{L}(\\theta-\\varepsilon))/2\\varepsilon$; require relative error $<10^{-6}$." },
    { q: "What causes vanishing gradients?", a: "The backward pass multiplies by a weight matrix and activation slope at every layer; if these factors are <1 they compound to near-zero over depth. Fixed by ReLU, good init, normalization, residuals." },
    { q: "Why does training use more memory than inference?", a: "The backward pass needs the forward activations, so they must be cached until `.backward()` runs — the usual cause of out-of-memory during training but not inference." },
    { q: "What is reverse-mode autodiff?", a: "The generalization of backprop: build a graph of primitive ops (each knowing its local vector-Jacobian product) and traverse it in reverse topological order applying the chain rule." },
  ],

  cheatsheet: [
    { label: "Output error", code: "d2 = Yhat - Y" },
    { label: "Weight grad", code: "dW = A_prev.T @ delta" },
    { label: "Bias grad", code: "db = delta.sum(0, keepdims=True)" },
    { label: "Backprop delta", code: "d1 = (d2 @ W2.T) * act_deriv(Z1)" },
    { label: "tanh derivative", code: "1 - np.tanh(z)**2" },
    { label: "ReLU derivative", code: "(z > 0).astype(z.dtype)" },
    { label: "sigmoid derivative", code: "s*(1-s)  # s=sigmoid(z)" },
    { label: "GD update", code: "p -= lr * grad" },
    { label: "Central diff", code: "(f(t+e)-f(t-e))/(2*e)" },
    { label: "Autograd (torch)", code: "loss.backward(); W.grad" },
    { label: "Zero grads", code: "opt.zero_grad()" },
    { label: "No-grad eval", code: "with torch.no_grad(): ..." },
  ],
});
