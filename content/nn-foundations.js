(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "nn-foundations",
  name: "Neural Network Foundations",
  language: "Neural Networks",
  group: "Neural Networks",
  navLabel: "NN Foundations",
  tagline: "What a neural network *is* — the perceptron, the multilayer forward pass $a^{[l]}=\\phi(W^{[l]}a^{[l-1]}+b^{[l]})$, activation functions, and why non-linearity is the whole point.",
  color: "#F59E0B",
  readMinutes: 48,
  sections: [
    {
      id: "what",
      title: "What a neural network actually is",
      level: "core",
      body: [
        { type: "p", text: "Strip away the mystique and a neural network is a **parameterized function** $f_\\theta : \\mathbb{R}^n \\to \\mathbb{R}^k$ built by alternating two operations you already know: a **linear map** (matrix multiply + bias, from the Linear Algebra track) and a **non-linear squashing function** applied elementwise. Stack a few of these and you get something that can approximate essentially any function — which is why the same recipe powers image classifiers, language models, and game-playing agents." },
        { type: "p", text: "This section is about the *architecture* — how a network is wired and how a prediction flows through it (the **forward pass**). We deliberately do **not** cover how it learns here: computing the gradient and updating the weights is **backpropagation**, the very next section. Think of this as learning the anatomy before the physiology." },
        { type: "table",
          headers: ["Ingredient", "What it is", "Where it comes from"],
          rows: [
            ["A layer's linear step", "$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$", "matrix–vector product (Linear Algebra)"],
            ["A layer's non-linearity", "$a^{[l]} = \\phi(z^{[l]})$", "activation function (this section)"],
            ["The whole network", "$f_\\theta = \\phi\\circ W^{[L]}\\cdots\\phi\\circ W^{[1]}$", "composition of layers"],
            ["The parameters", "$\\theta = \\{W^{[l]}, b^{[l]}\\}_{l=1}^{L}$", "what training will tune (next section)"],
          ]
        },
        { type: "callout", variant: "note", text: "**How to read this deck.** Every idea appears three ways: the **intuition** (what it means), the **math** (the exact statement), and the **code** (NumPy from scratch, then the framework one-liner). The forward pass you build here is literally the first half of the backprop code in the next section — nothing is wasted." },
      ]
    },

    {
      id: "history",
      title: "A short history: from a 1943 neuron to AlexNet and beyond",
      level: "core",
      body: [
        { type: "p", text: "Neural networks are not new — the field is 80 years old and has died and been resurrected at least twice. Knowing the arc tells you *why* the modern pieces exist, because almost every one of them was invented to fix a specific historical failure." },
        { type: "heading", text: "1943 — McCulloch & Pitts: the first artificial neuron" },
        { type: "p", text: "Warren McCulloch (a neurophysiologist) and Walter Pitts (a logician) proposed a **binary threshold unit**: inputs are 0/1, weights are fixed, and the neuron fires (outputs 1) if the weighted sum exceeds a threshold. They proved networks of these units could compute any logical proposition. There was no *learning* — weights were hand-set — but the core idea (neuron = weighted sum + threshold) was born." },
        { type: "heading", text: "1958 — Rosenblatt's perceptron: a neuron that learns" },
        { type: "p", text: "Frank Rosenblatt added a **learning rule** that adjusts the weights from examples, built it in hardware (the *Mark I Perceptron*), and demonstrated it learning to classify images. The press ran wild — the New York Times reported the Navy expected a machine that would 'walk, talk, see, write, reproduce itself.' The first AI hype cycle had begun." },
        { type: "heading", text: "1969 — the XOR winter: Minsky & Papert" },
        { type: "p", text: "In their book *Perceptrons*, Marvin Minsky and Seymour Papert proved a fatal limitation: a single-layer perceptron **cannot** compute the XOR function, because XOR is not linearly separable. Everyone knew multilayer networks could fix this, but nobody had a way to *train* the hidden layers. Funding evaporated and the first **AI winter** set in for over a decade." },
        { type: "heading", text: "1986 — the connectionist revival: backprop popularized" },
        { type: "p", text: "Rumelhart, Hinton & Williams published *Learning representations by back-propagating errors* in *Nature*, showing that the **backpropagation** algorithm (reverse-mode differentiation, actually described by Linnainmaa in 1970) could train multilayer networks and *learn* hidden features — including a solution to XOR. Neural networks were back, and the 'connectionist' school flourished through the early 1990s (LeCun's convolutional nets read handwritten ZIP codes on real mail)." },
        { type: "heading", text: "2012–now — the deep learning era" },
        { type: "p", text: "A second winter followed (SVMs and other methods outperformed neural nets on the data and hardware of the 1990s–2000s). The thaw came from three things arriving together: **big labeled datasets** (ImageNet), **GPUs** (cheap parallel matrix multiplication), and a few **engineering fixes** (ReLU, better initialization, dropout). In 2012 **AlexNet** (Krizhevsky, Sutskever & Hinton) crushed the ImageNet competition, halving the error rate. That result triggered the deep-learning explosion that runs unbroken through ResNets (2015), the Transformer (2017), and the large language models of today." },
        { type: "callout", variant: "note", text: "**The through-line.** The perceptron's linear-separability limit (1969) is fixed by hidden layers; hidden layers are only trainable because of backprop (1986); backprop only works at scale because of GPUs and non-saturating activations like ReLU (2012). Every section below is a chapter of this same story." },
      ]
    },

    {
      id: "biology",
      title: "The biological analogy — and where it breaks",
      level: "core",
      body: [
        { type: "p", text: "The vocabulary (neuron, activation, firing) is borrowed from neuroscience, and the loose analogy is genuinely useful. A biological neuron collects electrical inputs through its **dendrites**, sums them in the **cell body**, and if the total crosses a threshold it **fires** a spike down its **axon** to other neurons through **synapses** of varying strength." },
        { type: "table",
          headers: ["Biological neuron", "Artificial unit", "Role"],
          rows: [
            ["Dendrite inputs", "input values $a^{[l-1]}_j$", "signals arriving from other units"],
            ["Synaptic strength", "weight $W^{[l]}_{ij}$", "how much each input counts"],
            ["Cell body summation", "$z = \\sum_j W_{ij}a_j + b_i$", "weighted sum + bias"],
            ["Firing threshold", "activation $\\phi$", "non-linear decision to 'fire'"],
            ["Axon output", "activation $a^{[l]}_i$", "signal passed to the next layer"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**Do not take the analogy too far.** Real neurons communicate with *discrete spikes over time* (rate/timing codes), not continuous real numbers; the brain does not do backpropagation (there is no known biological mechanism for the exact reverse-mode gradient); real synapses have complex dynamics; and the brain runs on ~20 watts. An artificial 'neuron' is a mathematical convenience — $\\phi(w^\\top x + b)$ — not a model of biology. Treat the analogy as *inspiration*, not *explanation*." },
        { type: "p", text: "The honest framing: neural networks are **differentiable function approximators** that happen to have a neuron-flavored history. Everything that follows can be understood purely as linear algebra and calculus — no neuroscience required." },
      ]
    },

    {
      id: "perceptron",
      title: "The perceptron: a single linear unit",
      level: "core",
      body: [
        { type: "p", text: "The **perceptron** is one neuron: it takes an input vector $x \\in \\mathbb{R}^n$, forms a weighted sum plus a bias, and thresholds it to a binary decision. It is the atom from which everything else is built." },
        { type: "math", tex: String.raw`\hat y = \operatorname{step}\!\left(w^\top x + b\right), \qquad \operatorname{step}(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}` },
        { type: "p", text: "Geometrically, $w^\\top x + b = 0$ is a **hyperplane** (a line in 2-D, a plane in 3-D) that splits the input space in two. The perceptron outputs 1 on one side and 0 on the other. The weight vector $w$ is the *normal* to this boundary; $b$ shifts it away from the origin. So a perceptron is nothing more than a **linear decision boundary**." },
        { type: "heading", text: "The perceptron learning rule" },
        { type: "p", text: "Rosenblatt's rule is beautifully simple: loop over training examples, and whenever the perceptron misclassifies one, nudge the weights toward the correct answer. For example $(x, y)$ with prediction $\\hat y$:" },
        { type: "math", tex: String.raw`w \leftarrow w + \eta\,(y - \hat y)\,x, \qquad b \leftarrow b + \eta\,(y - \hat y)` },
        { type: "p", text: "Read the update: if the prediction is correct, $y - \\hat y = 0$ and nothing changes. If the true label is 1 but we predicted 0, we *add* $\\eta x$ to $w$, pushing the sum $w^\\top x$ up for this input next time; if we over-predicted, we subtract. The **Perceptron Convergence Theorem** (Novikoff, 1962) guarantees this finds a separating boundary in *finitely many* steps — **if one exists**. That caveat is everything." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef perceptron_fit(X, y, epochs=100, lr=1.0):\n    w = np.zeros(X.shape[1])\n    b = 0.0\n    for _ in range(epochs):\n        errors = 0\n        for xi, yi in zip(X, y):\n            yhat = 1.0 if (w @ xi + b) >= 0 else 0.0\n            update = lr * (yi - yhat)      # 0 if correct, +/-lr if wrong\n            w += update * xi\n            b += update\n            errors += int(update != 0)\n        if errors == 0:                    # converged: no mistakes this pass\n            break\n    return w, b\n\n# AND is linearly separable -> the perceptron nails it\nX = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)\ny_and = np.array([0, 0, 0, 1], dtype=float)\nw, b = perceptron_fit(X, y_and)\npred = ((X @ w + b) >= 0).astype(int)\nprint(pred)          # [0 0 0 1]  -> learned AND" },
        { type: "heading", text: "What a single unit cannot do: XOR" },
        { type: "p", text: "AND, OR, and NOT are all **linearly separable** — you can draw one straight line with all the 1s on one side. **XOR** is not: its true cases $(0,1)$ and $(1,0)$ sit on one diagonal, its false cases $(0,0)$ and $(1,1)$ on the other. No single straight line separates the diagonals, so no perceptron — no matter how you set $w$ and $b$ — can compute XOR." },
        { type: "table",
          headers: ["$x_1$", "$x_2$", "AND", "OR", "XOR"],
          rows: [
            ["0", "0", "0", "0", "0"],
            ["0", "1", "0", "1", "1"],
            ["1", "0", "0", "1", "1"],
            ["1", "1", "1", "1", "0"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**This is the 1969 result that caused an AI winter.** A single perceptron can only carve the input space with *one* hyperplane. XOR needs a *non-convex* region, which requires at least two boundaries — i.e. a **hidden layer**. The fix is not a better learning rule; it is more layers plus a non-linearity, which is exactly the next section." },
      ]
    },

    {
      id: "mlp",
      title: "The multilayer perceptron & the forward pass",
      level: "core",
      body: [
        { type: "p", text: "Stack neurons into **layers** and connect every unit of one layer to every unit of the next: that is a **multilayer perceptron (MLP)**, also called a *fully-connected* or *dense* feedforward network. Between the input and output layers sit one or more **hidden layers** whose job is to learn intermediate features. Two boundaries in a hidden layer, combined by the output, are enough to carve out XOR." },
        { type: "p", text: "We use the standard bracketed notation (identical to the Backpropagation section, so the two read as one continuous story). Layer $l$ has weight matrix $W^{[l]}$, bias $b^{[l]}$, pre-activation $z^{[l]}$, and activation $a^{[l]}$, with $a^{[0]} \\equiv x$ the input:" },
        { type: "math", tex: String.raw`\boxed{\; z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \qquad a^{[l]} = \phi\!\left(z^{[l]}\right) \;}` },
        { type: "p", text: "Running this from $l = 1$ to $l = L$ is the **forward pass**: data flows left to right, each layer transforming the previous layer's activations, until $a^{[L]} = \\hat y$ is the prediction." },
        { type: "math", tex: String.raw`x = a^{[0]} \;\longrightarrow\; z^{[1]}, a^{[1]} \;\longrightarrow\; z^{[2]}, a^{[2]} \;\longrightarrow\; \cdots \;\longrightarrow\; a^{[L]} = \hat y` },
        { type: "heading", text: "Shapes — get these right and everything else follows" },
        { type: "p", text: "If layer $l$ has $n_l$ units, then $W^{[l]} \\in \\mathbb{R}^{n_l \\times n_{l-1}}$ (out × in), $b^{[l]} \\in \\mathbb{R}^{n_l}$, and $a^{[l]} \\in \\mathbb{R}^{n_l}$. The matrix maps an $n_{l-1}$-vector to an $n_l$-vector — the arithmetic *is* the matrix–vector product from the Linear Algebra track." },
        { type: "table",
          headers: ["Object", "Shape (single example)", "Shape (batch of $m$)"],
          rows: [
            ["Input $a^{[0]} = x$", "$(n_0,)$", "$(m, n_0)$"],
            ["Weights $W^{[l]}$", "$(n_l, n_{l-1})$", "$(n_l, n_{l-1})$ — shared"],
            ["Bias $b^{[l]}$", "$(n_l,)$", "$(n_l,)$ — broadcast"],
            ["Pre-activation $z^{[l]}$", "$(n_l,)$", "$(m, n_l)$"],
            ["Activation $a^{[l]}$", "$(n_l,)$", "$(m, n_l)$"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Rows-as-examples convention.** In code we stack a batch as rows of $X \\in \\mathbb{R}^{m\\times n}$ and write the layer as `Z = X @ W.T + b` (or store $W$ already transposed as `(in, out)` and write `X @ W + b`). The bias broadcasts across the $m$ rows. This is the exact layout the Backpropagation section differentiates — so learn it here once." },
        { type: "callout", variant: "note", text: "**The count of parameters** in a dense layer is $n_l \\cdot n_{l-1}$ weights plus $n_l$ biases. A modest 784→128→10 MNIST network already has $784{\\cdot}128 + 128 + 128{\\cdot}10 + 10 = 101{,}770$ parameters. This is why we need an *efficient* way to compute all their gradients — the motivation for backprop." },
      ]
    },

    {
      id: "activations",
      title: "Activation functions",
      level: "core",
      body: [
        { type: "p", text: "The activation $\\phi$ is the non-linear squashing applied after each linear step. It is the single most consequential architectural choice, and the history of deep learning is partly a history of better activations. Here are the ones you must know cold, with their formulas and derivatives (the derivative is what backprop will need next section)." },
        { type: "heading", text: "Sigmoid (logistic)" },
        { type: "math", tex: String.raw`\sigma(z) = \frac{1}{1 + e^{-z}} \in (0,1), \qquad \sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)` },
        { type: "p", text: "Squashes any real number into $(0,1)$, so it reads as a probability — still the right choice for a **binary output** unit. As a *hidden* activation it has fallen out of favor: its derivative peaks at just $0.25$ (at $z=0$) and vanishes for large $|z|$, so gradients shrink layer by layer (the **vanishing-gradient** problem)." },
        { type: "heading", text: "Tanh (hyperbolic tangent)" },
        { type: "math", tex: String.raw`\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} \in (-1,1), \qquad \tanh'(z) = 1 - \tanh^2(z)` },
        { type: "p", text: "A rescaled sigmoid that is **zero-centered** ($\\tanh(0)=0$), which makes optimization better-behaved than sigmoid, so it was the default hidden activation through the 1990s–2000s. It still saturates for large $|z|$, so it too suffers from vanishing gradients in very deep nets." },
        { type: "heading", text: "ReLU (Rectified Linear Unit)" },
        { type: "math", tex: String.raw`\operatorname{ReLU}(z) = \max(0, z), \qquad \operatorname{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}` },
        { type: "p", text: "The workhorse of modern deep learning and the **default hidden activation** since AlexNet (2012). Its derivative is exactly $1$ for positive inputs — it does not saturate on the positive side — so gradients flow undiminished through active units. It is also trivially cheap (a `max`). The catch is the **dead-ReLU problem**: a unit whose input is always negative outputs $0$, has gradient $0$, and can never recover — it is dead for the rest of training. A too-large learning rate or bad init can kill a big fraction of units." },
        { type: "heading", text: "LeakyReLU (and PReLU)" },
        { type: "math", tex: String.raw`\operatorname{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \le 0 \end{cases}, \qquad \alpha \approx 0.01` },
        { type: "p", text: "Fixes dead ReLUs by giving negative inputs a small non-zero slope $\\alpha$, so the gradient is never exactly zero and a unit can climb back to life. **PReLU** makes $\\alpha$ a learnable parameter." },
        { type: "heading", text: "GELU (Gaussian Error Linear Unit)" },
        { type: "math", tex: String.raw`\operatorname{GELU}(z) = z\,\Phi(z) \approx 0.5\,z\left(1 + \tanh\!\left[\sqrt{\tfrac{2}{\pi}}\,(z + 0.044715\,z^3)\right]\right)` },
        { type: "p", text: "A smooth activation where $\\Phi$ is the standard-normal CDF; it weights each input by the probability that a Gaussian is below it, giving a soft, differentiable version of ReLU. **GELU is the default in Transformers** (BERT, GPT), which is why it matters far beyond its modest appearance." },
        { type: "heading", text: "Softmax — for the output layer, not hidden layers" },
        { type: "math", tex: String.raw`\operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}, \qquad \sum_i \operatorname{softmax}(z)_i = 1` },
        { type: "p", text: "Softmax turns a vector of $k$ real scores (**logits**) into a probability distribution over $k$ classes. It is used **only at the output** of a multiclass classifier, paired with cross-entropy loss. Unlike the others it is not applied elementwise — every output depends on every input through the shared denominator." },
        { type: "table",
          headers: ["Activation", "Range", "Pros", "Cons / when NOT to use"],
          rows: [
            ["Sigmoid", "$(0,1)$", "probabilistic output", "saturates, vanishing grads — avoid in hidden layers"],
            ["Tanh", "$(-1,1)$", "zero-centered", "still saturates in deep nets"],
            ["ReLU", "$[0,\\infty)$", "no positive saturation, cheap, default", "dead units for always-negative inputs"],
            ["LeakyReLU", "$(-\\infty,\\infty)$", "no dead units", "extra hyperparameter $\\alpha$"],
            ["GELU", "$(-\\approx0,\\infty)$", "smooth, SOTA in Transformers", "slightly more compute than ReLU"],
            ["Softmax", "$(0,1)$, sums to 1", "multiclass probabilities", "output layer only, never hidden"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Which to use, in one line.** Hidden layers: start with **ReLU**; switch to **LeakyReLU/GELU** if you see dead units or you're building a Transformer. Output layer: **identity** for regression, **sigmoid** for binary, **softmax** for multiclass. That covers 95% of real networks." },
        { type: "callout", variant: "gotcha", text: "**Numerical stability of softmax.** `np.exp(z)` overflows for large logits. Always subtract the max first: $\\operatorname{softmax}(z) = \\operatorname{softmax}(z - \\max_j z_j)$, which is mathematically identical (the constant cancels top and bottom) but never overflows. Frameworks do this internally; from-scratch code must do it by hand." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef sigmoid(z):    return 1.0 / (1.0 + np.exp(-z))\ndef relu(z):       return np.maximum(0.0, z)\ndef leaky_relu(z, a=0.01): return np.where(z > 0, z, a*z)\ndef gelu(z):\n    return 0.5*z*(1 + np.tanh(np.sqrt(2/np.pi)*(z + 0.044715*z**3)))\n\ndef softmax(z):\n    z = z - z.max(axis=-1, keepdims=True)   # stability: subtract row max\n    e = np.exp(z)\n    return e / e.sum(axis=-1, keepdims=True)\n\nz = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])\nprint(relu(z))                 # [0.  0.  0.  0.5 2. ]\nprint(softmax(z).round(3))     # sums to 1.0\nprint(softmax(z).sum())        # 1.0" },
      ]
    },

    {
      id: "why-nonlinear",
      title: "Why non-linearity is essential (a proof)",
      level: "core",
      body: [
        { type: "p", text: "Why bother with $\\phi$ at all? Because **without it, depth is an illusion**. A stack of linear layers with no activation collapses into a single linear layer — no matter how many you stack, you can only ever represent one linear map, which we already know cannot do XOR. Here is the proof; it takes three lines." },
        { type: "p", text: "Take two layers with **no** activation (i.e. $\\phi = $ identity):" },
        { type: "math", tex: String.raw`a^{[1]} = W^{[1]} x + b^{[1]}, \qquad a^{[2]} = W^{[2]} a^{[1]} + b^{[2]}` },
        { type: "p", text: "Substitute the first into the second and regroup using the associativity of matrix multiplication:" },
        { type: "math", tex: String.raw`a^{[2]} = W^{[2]}\!\left(W^{[1]} x + b^{[1]}\right) + b^{[2]} = \underbrace{\left(W^{[2]}W^{[1]}\right)}_{W'} x + \underbrace{\left(W^{[2]}b^{[1]} + b^{[2]}\right)}_{b'}` },
        { type: "math", tex: String.raw`\boxed{\; a^{[2]} = W' x + b' \;}` },
        { type: "p", text: "The two-layer network is *exactly equal* to a single linear layer with weights $W' = W^{[2]}W^{[1]}$ and bias $b' = W^{[2]}b^{[1]} + b^{[2]}$. By induction, **any** number of linear layers collapses to one. All that depth bought you nothing — the composition of linear functions is linear." },
        { type: "callout", variant: "good", text: "**This is the whole reason activations exist.** Inserting a non-linear $\\phi$ between the matrix multiplies breaks the collapse: $\\phi(W^{[2]}\\phi(W^{[1]}x + b^{[1]}) + b^{[2]})$ cannot be flattened into one linear map, so each layer adds genuinely new expressive power. Non-linearity is not a detail — it is what makes a *deep* network more than a *wide* linear one." },
        { type: "code", lang: "py", code: "import numpy as np\nrng = np.random.default_rng(0)\n\n# Two 'linear-only' layers collapse to one linear layer:\nW1, b1 = rng.normal(size=(4,3)), rng.normal(size=4)\nW2, b2 = rng.normal(size=(2,4)), rng.normal(size=2)\nx = rng.normal(size=3)\n\ntwo_layer   = W2 @ (W1 @ x + b1) + b2\none_layer   = (W2 @ W1) @ x + (W2 @ b1 + b2)\nprint(np.allclose(two_layer, one_layer))   # True -> identical map\n\n# Add a non-linearity and the equivalence breaks:\nrelu = lambda z: np.maximum(0, z)\nnonlinear   = W2 @ relu(W1 @ x + b1) + b2\nprint(np.allclose(nonlinear, one_layer))    # False -> new expressive power" },
      ]
    },

    {
      id: "uat",
      title: "The Universal Approximation Theorem",
      level: "core",
      body: [
        { type: "p", text: "If one hidden layer plus a non-linearity is enough to break linearity, *how much* can such a network represent? The answer is the celebrated **Universal Approximation Theorem** (Cybenko 1989 for sigmoids; Hornik 1991 generalized it)." },
        { type: "callout", variant: "note", text: "**Statement (informal).** A feedforward network with a **single hidden layer** containing a **finite** number of neurons and a non-constant, bounded, continuous activation can approximate **any** continuous function on a compact subset of $\\mathbb{R}^n$ to **any** desired accuracy $\\varepsilon > 0$ — provided the hidden layer is wide enough." },
        { type: "math", tex: String.raw`\forall\, \varepsilon > 0,\; \exists\, N,\, \{w_i, b_i, v_i\} : \quad \left| f(x) - \sum_{i=1}^{N} v_i\,\phi\!\left(w_i^\top x + b_i\right) \right| < \varepsilon \quad \forall x \in K` },
        { type: "heading", text: "The intuition" },
        { type: "p", text: "Each hidden unit $\\phi(w_i^\\top x + b_i)$ is a soft step/bump in input space. With enough bumps you can tile the domain and sum them (weighted by $v_i$) to trace out any continuous shape — the same way a Riemann sum of many thin rectangles approximates any curve. Width buys resolution." },
        { type: "heading", text: "What it does NOT promise (read this twice)" },
        { type: "p", text: "The theorem is an **existence** result, and existence is not the same as attainability. It is routinely over-quoted; here is what it genuinely guarantees versus what it does not." },
        { type: "table",
          headers: ["It DOES say", "It does NOT say"],
          rows: [
            ["A weight setting achieving accuracy $\\varepsilon$ *exists*", "That gradient descent will *find* those weights"],
            ["One hidden layer suffices *in principle*", "That one layer is *efficient* — width may be astronomical"],
            ["The function can be *represented*", "That it can be *learned from finite data* (generalization)"],
            ["Approximation on a *compact* set", "Anything about extrapolation outside that set"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**Existence, not learnability.** The theorem says the needle is in the haystack; it says nothing about whether SGD can find it, how much data you'd need, or whether the required width is $10$ or $10^{40}$. This gap between *representation* and *optimization + generalization* is what the rest of deep learning is actually about." },
        { type: "heading", text: "Why depth beats width" },
        { type: "p", text: "So why isn't every network one fat hidden layer? Because for many important functions a **shallow** network needs *exponentially* more units than a **deep** one to reach the same accuracy. Deep networks compose features hierarchically — edges → shapes → objects — reusing lower-level pieces, so they represent structured functions **exponentially more parameter-efficiently**. Depth is the practical realization of what the theorem only guarantees in principle. Empirically, depth is what made ImageNet-scale learning work." },
      ]
    },

    {
      id: "outputs-losses",
      title: "Output layers & losses by task",
      level: "core",
      body: [
        { type: "p", text: "The hidden layers are the same everywhere; what changes with the task is the **output activation** and the **loss**. These come in matched pairs — the pairing is not arbitrary, it is what makes the output gradient collapse to the clean $\\hat y - y$ you will see in the next section, and (as the Probability track shows) each pair is a maximum-likelihood estimator." },
        { type: "table",
          headers: ["Task", "Output units", "Output activation", "Loss"],
          rows: [
            ["Regression", "$1$ (or $k$)", "identity (none)", "mean squared error (MSE)"],
            ["Binary classification", "$1$", "sigmoid", "binary cross-entropy (BCE)"],
            ["Multiclass (1 label)", "$k$", "softmax", "categorical cross-entropy"],
            ["Multilabel (many)", "$k$", "sigmoid (per unit)", "sum of binary cross-entropies"],
          ]
        },
        { type: "math", tex: String.raw`\text{MSE: } \tfrac{1}{m}\sum_i (\hat y_i - y_i)^2 \qquad \text{BCE: } -\big[y\log\hat y + (1{-}y)\log(1{-}\hat y)\big]` },
        { type: "math", tex: String.raw`\text{Categorical CE: } -\sum_{c=1}^{k} y_c \log \hat y_c \quad (\text{with } \hat y = \operatorname{softmax}(z^{[L]}))` },
        { type: "callout", variant: "tip", text: "**Logits, not probabilities, into the loss.** Framework losses like `CrossEntropyLoss` (PyTorch) expect **raw logits** $z^{[L]}$ and apply softmax *internally* using the log-sum-exp trick for numerical stability. Do **not** apply softmax yourself and then pass it in — you'll double-apply it and get wrong, unstable gradients. This is one of the most common beginner bugs." },
        { type: "callout", variant: "note", text: "**Why these pairs?** Each output activation matches the natural output space of the task (real line → identity, $[0,1]$ → sigmoid, simplex → softmax), and each loss is the negative log-likelihood under the matching distribution (Gaussian → MSE, Bernoulli → BCE, Categorical → cross-entropy). The gradients are derived in full in the **Backpropagation** section." },
      ]
    },

    {
      id: "forward-scratch",
      title: "A forward-pass MLP from scratch (and in PyTorch)",
      level: "core",
      body: [
        { type: "p", text: "Let us build the architecture and run a forward pass — **no training yet** (that is the next section). This is the anatomy: initialized weights, a stack of layers, and data flowing through to a prediction. Note how the code is a direct transcription of $a^{[l]} = \\phi(W^{[l]}a^{[l-1]} + b^{[l]})$." },
        { type: "code", lang: "py", code: "import numpy as np\nrng = np.random.default_rng(0)\n\ndef relu(z):    return np.maximum(0.0, z)\ndef softmax(z):\n    z = z - z.max(axis=-1, keepdims=True)\n    e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)\n\nclass MLP:\n    \"\"\"A dense feedforward net. Architecture only — forward pass, no learning.\"\"\"\n    def __init__(self, sizes):            # e.g. [784, 128, 64, 10]\n        self.W, self.b = [], []\n        for n_in, n_out in zip(sizes[:-1], sizes[1:]):\n            # He initialization: keeps activation variance stable through ReLU\n            self.W.append(rng.normal(0, np.sqrt(2.0/n_in), (n_in, n_out)))\n            self.b.append(np.zeros(n_out))\n\n    def forward(self, X):                 # X: (m, n_0), rows = examples\n        A = X\n        L = len(self.W)\n        for l in range(L):\n            Z = A @ self.W[l] + self.b[l]   # z = W a + b  (batched)\n            A = softmax(Z) if l == L-1 else relu(Z)   # last layer: softmax\n        return A\n\nnet = MLP([784, 128, 64, 10])            # a classic MNIST-shaped classifier\nX = rng.normal(size=(4, 784))            # a batch of 4 fake 28x28 images\nprobs = net.forward(X)\nprint(probs.shape)                       # (4, 10)\nprint(probs.sum(axis=1).round(6))        # [1. 1. 1. 1.] -> valid distributions\nprint(probs[0].argmax())                 # predicted class for example 0\n# Untrained -> predictions are random. Learning them is the next section." },
        { type: "callout", variant: "tip", text: "**Why He initialization?** Weights are drawn with variance $2/n_{\\text{in}}$ (He et al. 2015) so the signal's variance stays roughly constant as it passes through each ReLU layer, avoiding activations that blow up or shrink to zero across depth. Poor init is a leading cause of a network that 'won't train' — the deeper story is in the Training Deep Networks material." },
        { type: "heading", text: "The same architecture in torch.nn.Sequential" },
        { type: "p", text: "Frameworks give you the identical network in a few lines — `nn.Linear` holds a $W$ and $b$, `nn.ReLU` is the activation, and `Sequential` chains them into the forward pass. (No softmax at the end: `CrossEntropyLoss` applies it internally on the raw logits, per the earlier gotcha.)" },
        { type: "code", lang: "py", code: "import torch\nimport torch.nn as nn\n\nnet = nn.Sequential(\n    nn.Linear(784, 128),   # W:(128,784) b:(128,) — created & inited for you\n    nn.ReLU(),\n    nn.Linear(128, 64),\n    nn.ReLU(),\n    nn.Linear(64, 10),     # 10 logits; softmax lives inside the loss\n)\n\nX = torch.randn(4, 784)          # batch of 4\nlogits = net(X)                  # forward pass\nprint(logits.shape)              # torch.Size([4, 10])\nprint(net(X).softmax(dim=1).sum(dim=1))   # tensor([1., 1., 1., 1.])\n\n# Count parameters — matches the from-scratch net exactly\nprint(sum(p.numel() for p in net.parameters()))   # 109386" },
        { type: "callout", variant: "note", text: "**This is the whole forward pass.** Calling `net(X)` runs $a^{[l]} = \\phi(W^{[l]}a^{[l-1]}+b^{[l]})$ for every layer — nothing more. What's missing is turning the prediction error into weight updates, which is `loss.backward()` — the subject of the **Backpropagation** section that follows." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Architecture is a *building* skill. Do at least the first two by hand before starting Backpropagation — you'll reuse this exact forward-pass code there." },
        { type: "list", ordered: true, items: [
          "**Perceptron on logic gates.** Implement the perceptron learning rule and train it on AND, OR, and NOT — watch it converge. Then try it on **XOR** and watch it *never* converge no matter how many epochs. Plot the four points and the decision line to *see* why. This reproduces the 1969 Minsky–Papert result with your own hands.",
          "**Activation zoo.** Plot sigmoid, tanh, ReLU, LeakyReLU, and GELU and their derivatives on the same axes over $[-5, 5]$. Mark where each derivative saturates (goes to ~0). This single plot explains the entire vanishing-gradient story visually.",
          "**Prove the linear collapse.** Build a 5-layer *linear-only* network and verify numerically it equals a single matrix $W' = W^{[5]}\\cdots W^{[1]}$. Then insert ReLUs and confirm the equality breaks. Internalize *why* non-linearity is mandatory.",
          "**Forward-pass MLP class.** Extend the from-scratch `MLP` to accept any activation per layer and any output type (identity / sigmoid / softmax). Run forward passes on random data and assert every shape and that softmax outputs sum to 1. This class becomes your backprop starting point next section.",
          "**Universal approximation demo.** With a *fixed* random single-hidden-layer network of increasing width (10 → 100 → 1000 units), fit the coefficients $v_i$ by least squares to approximate $\\sin(x)$ on $[-\\pi,\\pi]$. Watch the fit improve with width — you'll *see* the theorem, and *feel* how inefficient pure width is.",
          "**Count the parameters.** For a 784→256→128→10 MLP, compute the parameter count by hand, then confirm with `sum(p.numel() for p in net.parameters())`. Understand exactly where every number lives.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "This section is self-contained, but these are the best places to reinforce the foundations before moving on to Backpropagation:" },
        { type: "link", url: "http://neuralnetworksanddeeplearning.com/chap1.html", text: "Michael Nielsen — Neural Networks and Deep Learning, Ch. 1 (perceptrons, sigmoid neurons, and a network for MNIST — the best free intro)" },
        { type: "link", url: "https://www.3blue1brown.com/topics/neural-networks", text: "3Blue1Brown — Neural Networks series (the visual intuition for what a layer computes; watch this first)" },
        { type: "link", url: "https://www.deeplearningbook.org/contents/mlp.html", text: "Goodfellow, Bengio & Courville — Deep Learning, Ch. 6 (deep feedforward networks; the formal treatment of MLPs and activations)" },
        { type: "link", url: "http://www.cs.cmu.edu/~./epxing/Class/10715/reading/Cybenko.pdf", text: "Cybenko (1989) — Approximation by superpositions of a sigmoidal function (the original Universal Approximation Theorem)" },
        { type: "link", url: "https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html", text: "Krizhevsky, Sutskever & Hinton (2012) — ImageNet Classification with Deep CNNs (AlexNet: the paper that started the deep-learning era)" },
        { type: "link", url: "https://cs231n.github.io/neural-networks-1/", text: "Stanford CS231n — Neural Networks Part 1 (neurons, activation functions, architectures — excellent course notes)" },
      ]
    },
  ],

  packages: [
    { name: "numpy", why: "build the forward pass by hand — `@` for the linear step, elementwise ufuncs for activations" },
    { name: "torch.nn", why: "`Linear`, `ReLU`, `Sequential` — the same MLP in a few lines, with weights inited for you" },
    { name: "torch.nn.functional", why: "stateless `relu`, `gelu`, `softmax`, `sigmoid` for functional-style forward passes" },
    { name: "scikit-learn", why: "`Perceptron` and `MLPClassifier` for a quick baseline and to compare against your from-scratch code" },
    { name: "matplotlib", why: "plot activations, their derivatives, and perceptron decision boundaries — seeing is understanding" },
  ],

  gotchas: [
    "A **single perceptron** can only draw one hyperplane, so it cannot learn XOR (not linearly separable). The fix is a hidden layer, not a better rule.",
    "**Stacking linear layers is pointless** without activations: they collapse to one linear map $W' = W^{[2]}W^{[1]}$. Non-linearity is mandatory, not optional.",
    "Weight matrix shape is **out × in**: $W^{[l]} \\in \\mathbb{R}^{n_l \\times n_{l-1}}$. Every gradient must match its parameter's shape — a mismatch means a transpose is wrong.",
    "**Softmax overflows** for large logits — subtract the row max first: $\\operatorname{softmax}(z-\\max_j z_j)$. Mathematically identical, numerically safe.",
    "Pass **raw logits** (not softmax outputs) to `CrossEntropyLoss` — it applies softmax internally. Double-applying it is a classic silent bug.",
    "The **Universal Approximation Theorem** promises *existence* of good weights, not that training will *find* them or that width is practical. Don't over-quote it.",
    "**ReLU can die**: a unit stuck with negative input outputs 0 with gradient 0 forever. Use LeakyReLU/GELU or a smaller learning rate if many units die.",
    "Sigmoid/tanh **saturate** ($\\phi'\\to 0$ for large $|z|$), causing vanishing gradients in deep nets — that's why ReLU replaced them as the hidden default.",
  ],

  flashcards: [
    { q: "What is a perceptron and what is its fundamental limitation?", a: "A single linear unit $\\hat y = \\operatorname{step}(w^\\top x + b)$ — one hyperplane. It can only classify **linearly separable** data, so it cannot compute XOR." },
    { q: "State the perceptron learning rule.", a: "$w \\leftarrow w + \\eta(y - \\hat y)x$ (and $b \\leftarrow b + \\eta(y-\\hat y)$): no change when correct, nudge toward the right answer when wrong. Converges iff the data is separable." },
    { q: "Write the forward-pass equation for layer $l$.", a: "$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$, then $a^{[l]} = \\phi(z^{[l]})$, with $a^{[0]} = x$ and $a^{[L]} = \\hat y$." },
    { q: "Why can't you build a deep network from linear layers alone?", a: "They collapse: $W^{[2]}(W^{[1]}x + b^{[1]}) + b^{[2]} = W'x + b'$. Any stack of linear maps equals one linear map. A non-linear $\\phi$ between them prevents the collapse." },
    { q: "Give the derivatives of sigmoid, tanh, and ReLU.", a: "$\\sigma'=\\sigma(1-\\sigma)$; $\\tanh'=1-\\tanh^2$; $\\operatorname{ReLU}'=1$ for $z>0$ else $0$." },
    { q: "What is the dead-ReLU problem and one fix?", a: "A unit whose input is always negative outputs 0 with gradient 0 and never recovers. Fix: LeakyReLU (small negative slope $\\alpha$), GELU, or a smaller learning rate." },
    { q: "State the Universal Approximation Theorem and its key caveat.", a: "A single-hidden-layer net with a non-linear activation can approximate any continuous function on a compact set to arbitrary accuracy given enough width. Caveat: it guarantees *existence*, not *learnability* or *efficiency*." },
    { q: "Why does depth help if one hidden layer is universal?", a: "Deep nets compose features hierarchically and can represent many structured functions with *exponentially* fewer units than a shallow net needs. Width is universal but often astronomically inefficient." },
    { q: "Which output activation and loss go with each task?", a: "Regression: identity + MSE. Binary: sigmoid + BCE. Multiclass: softmax + cross-entropy. These matched pairs make the output gradient collapse to $\\hat y - y$." },
    { q: "Why subtract the max before softmax?", a: "`exp` of large logits overflows. Subtracting $\\max_j z_j$ is mathematically identical (constant cancels) but numerically stable." },
    { q: "What was the significance of AlexNet (2012)?", a: "It won ImageNet by a huge margin using a deep CNN on GPUs with ReLU, halving the error rate and launching the modern deep-learning era." },
    { q: "What is GELU and where is it used?", a: "$\\operatorname{GELU}(z)=z\\Phi(z)$, a smooth ReLU-like activation weighting inputs by the normal CDF. It's the default in Transformers (BERT, GPT)." },
  ],

  cheatsheet: [
    { label: "Perceptron", code: "yhat = (w @ x + b) >= 0" },
    { label: "Perceptron update", code: "w += lr*(y-yhat)*x" },
    { label: "Layer forward", code: "Z = A @ W + b; A = phi(Z)" },
    { label: "Sigmoid", code: "1/(1+np.exp(-z))" },
    { label: "Tanh", code: "np.tanh(z)" },
    { label: "ReLU", code: "np.maximum(0, z)" },
    { label: "LeakyReLU", code: "np.where(z>0, z, 0.01*z)" },
    { label: "Softmax (stable)", code: "e=np.exp(z-z.max(-1,keepdims=True)); e/e.sum(-1,keepdims=True)" },
    { label: "He init", code: "rng.normal(0, np.sqrt(2/n_in), (n_in,n_out))" },
    { label: "torch MLP", code: "nn.Sequential(nn.Linear(a,b), nn.ReLU(), nn.Linear(b,c))" },
    { label: "torch forward", code: "logits = net(X)" },
    { label: "Param count", code: "sum(p.numel() for p in net.parameters())" },
  ],
});
