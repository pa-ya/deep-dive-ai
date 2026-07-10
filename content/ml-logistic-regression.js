(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "ml-logistic-regression",
  name: "Logistic Regression & Classification",
  language: "Classical ML",
  group: "Classical Machine Learning",
  navLabel: "Logistic Regression",
  tagline: "The gateway from regression to classification — sigmoid, cross-entropy, and the clean $\\hat y - y$ gradient that *is* a neural net's output layer.",
  color: "#0D9488",
  readMinutes: 50,
  sections: [
    {
      id: "setup",
      title: "From regression to classification: odds & log-odds",
      level: "core",
      body: [
        { type: "p", text: "Linear regression predicts a **number**: house price, temperature, revenue. But a huge share of real problems ask a **yes/no** question instead — is this email spam, will this loan default, does this scan show a tumour? That is **binary classification**, and the honest answer is not a hard 0/1 but a **probability**: 'there's an 87% chance this is spam.' Logistic regression is the simplest, most important model that outputs calibrated probabilities — and, as you will see in the last section, it is quite literally the output neuron of every classifier neural network." },
        { type: "p", text: "The setup: each example is a feature vector $x \\in \\mathbb{R}^n$ with a **binary label** $y \\in \\{0, 1\\}$. We want a function that maps $x$ to $p(y=1 \\mid x) \\in [0, 1]$." },
        { type: "heading", text: "Why not just use linear regression?" },
        { type: "p", text: "The tempting first idea is to fit $\\hat y = w^\\top x + b$ and call anything above $0.5$ a positive. This fails for three concrete reasons:" },
        { type: "list", ordered: false, items: [
          "**It leaves the $[0,1]$ box.** A linear function is unbounded, so it happily predicts $\\hat y = 1.7$ or $\\hat y = -0.3$. Those are not probabilities, and there is no principled way to interpret them.",
          "**The wrong loss.** Squared error assumes Gaussian noise around a continuous target. A 0/1 label is not Gaussian — it is **Bernoulli**. Fitting the wrong noise model gives you the wrong estimator.",
          "**Outliers rotate the boundary.** A single far-away positive example drags the least-squares line toward it, moving the $0.5$ crossing and misclassifying points that were fine. Classification should not care *how far* past the boundary a point is, only *which side* it is on.",
        ]},
        { type: "callout", variant: "note", text: "**History.** The logistic function was studied by Pierre François Verhulst in the 1830s–40s to model population growth. Its use for classification is due to statistician **David Cox (1958)**, building on the *probit* model and Ronald Fisher's maximum-likelihood framework. It has survived seventy years and the entire deep-learning revolution because it is the *right* model for Bernoulli data — the sigmoid is not a hack, it falls out of the math." },
        { type: "heading", text: "Odds, and why we take their log" },
        { type: "p", text: "Probabilities live in $[0,1]$, which is an awkward, bounded interval to fit a linear model against. The fix is a change of variable that stretches $[0,1]$ onto the whole real line, in two steps. First, the **odds** — the ratio of success to failure, familiar from betting:" },
        { type: "math", tex: String.raw`\text{odds}(p) = \frac{p}{1-p} \in [0, \infty)` },
        { type: "p", text: "Odds of $3$ ('3 to 1') means $p = 0.75$. Odds fix the upper bound but are still one-sided — they cannot go negative. So take the logarithm, giving the **log-odds** or **logit**, which ranges over *all* real numbers:" },
        { type: "math", tex: String.raw`\operatorname{logit}(p) = \ln\!\left(\frac{p}{1-p}\right) \in (-\infty, +\infty)` },
        { type: "p", text: "Now we have something a linear model can predict without ever leaving its natural range. The central modelling assumption of logistic regression is exactly this — **the log-odds are linear in the features**:" },
        { type: "math", tex: String.raw`\ln\!\left(\frac{p}{1-p}\right) = w^\top x + b` },
        { type: "callout", variant: "tip", text: "This one assumption *is* the model. Everything else — the sigmoid, the loss, the gradient — is a consequence. When you interpret a fitted logistic model, a weight $w_j$ is 'the change in log-odds per unit of feature $j$', and $e^{w_j}$ is the **odds ratio**: how many times the odds multiply when $x_j$ increases by one. That interpretability is why logistic regression still dominates in medicine, credit scoring, and epidemiology." },
      ]
    },

    {
      id: "sigmoid",
      title: "The sigmoid function: inverting the log-odds",
      level: "core",
      body: [
        { type: "p", text: "We assumed the log-odds are linear: $\\ln\\!\\big(p/(1-p)\\big) = z$ where $z = w^\\top x + b$. But we want $p$, not the log-odds. So **solve for $p$** — this derivation is worth doing once by hand, because the famous S-curve simply pops out." },
        { type: "math", tex: String.raw`\ln\!\left(\frac{p}{1-p}\right) = z \;\Longrightarrow\; \frac{p}{1-p} = e^{z} \;\Longrightarrow\; p = e^{z}(1-p)` },
        { type: "math", tex: String.raw`p\,(1 + e^{z}) = e^{z} \;\Longrightarrow\; p = \frac{e^{z}}{1 + e^{z}} = \frac{1}{1 + e^{-z}}` },
        { type: "p", text: "That last expression is the **sigmoid** (a.k.a. logistic) function $\\sigma$. It is the inverse of the logit — the function that squashes any real number back into a valid probability:" },
        { type: "math", tex: String.raw`\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma : \mathbb{R} \to (0, 1)` },
        { type: "heading", text: "Properties you should know cold" },
        { type: "table",
          headers: ["Property", "Statement", "Why it matters"],
          rows: [
            ["Range", "$\\sigma(z) \\in (0,1)$, never exactly $0$ or $1$", "output is always a valid probability"],
            ["Midpoint", "$\\sigma(0) = 0.5$", "the decision boundary sits at $z = 0$"],
            ["Symmetry", "$\\sigma(-z) = 1 - \\sigma(z)$", "$p(y{=}0) = 1 - p(y{=}1)$ comes for free"],
            ["Saturation", "$\\sigma \\to 1$ as $z\\to\\infty$, $\\to 0$ as $z\\to-\\infty$", "confident predictions; but flat tails = vanishing gradients"],
            ["Monotonic", "strictly increasing", "larger score $\\Rightarrow$ larger probability"],
          ]
        },
        { type: "heading", text: "The derivative — the identity that makes everything clean" },
        { type: "p", text: "The single most useful fact about the sigmoid is that its derivative is expressible in terms of *itself*. This is why the backprop of a logistic output is so cheap. Let $s = \\sigma(z)$. Differentiate $\\sigma(z) = (1 + e^{-z})^{-1}$ with the chain rule:" },
        { type: "math", tex: String.raw`\sigma'(z) = \frac{d}{dz}\big(1 + e^{-z}\big)^{-1} = -\big(1 + e^{-z}\big)^{-2}\cdot(-e^{-z}) = \frac{e^{-z}}{\big(1 + e^{-z}\big)^{2}}` },
        { type: "p", text: "Now split the fraction and recognise the pieces — $\\tfrac{1}{1+e^{-z}} = \\sigma(z)$ and $\\tfrac{e^{-z}}{1+e^{-z}} = 1 - \\sigma(z)$:" },
        { type: "math", tex: String.raw`\sigma'(z) = \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}} = \sigma(z)\big(1 - \sigma(z)\big)` },
        { type: "math", tex: String.raw`\boxed{\;\sigma'(z) = \sigma(z)\,\big(1 - \sigma(z)\big)\;}` },
        { type: "callout", variant: "tip", text: "This derivative is maximal ($0.25$) at $z=0$ and shrinks to zero in both tails. That is the seed of the **vanishing-gradient problem**: a neuron that is confidently saturated ($z$ far from $0$) learns almost nothing, because $\\sigma' \\approx 0$ kills its gradient. It is the reason deep networks eventually replaced sigmoid hidden units with ReLU — a story you meet in the Neural Networks track." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef sigmoid(z):\n    # Numerically stable: never call exp on a large positive number.\n    z = np.asarray(z, dtype=float)\n    out = np.empty_like(z)\n    pos, neg = z >= 0, z < 0\n    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))\n    ez = np.exp(z[neg])\n    out[neg] = ez / (1.0 + ez)\n    return out\n\ndef sigmoid_prime(z):\n    s = sigmoid(z)\n    return s * (1 - s)\n\nprint(sigmoid(0.0))        # 0.5\nprint(sigmoid(1e3))        # 1.0  (no overflow warning)\nprint(sigmoid(-1e3))       # 0.0\nprint(sigmoid_prime(0.0))  # 0.25  -> steepest here" },
        { type: "callout", variant: "gotcha", text: "The naive `1/(1+np.exp(-z))` overflows for very negative `z` (`np.exp(1000)` = `inf`) and spams `RuntimeWarning`. The branch above evaluates the mathematically-equivalent $e^{z}/(1+e^{z})$ form when $z<0$, so `exp` only ever sees non-positive arguments. Production libraries do exactly this." },
      ]
    },

    {
      id: "model",
      title: "The model & its linear decision boundary",
      level: "core",
      body: [
        { type: "p", text: "Putting the pieces together, logistic regression is one clean equation. The probability of the positive class is the sigmoid of a linear score:" },
        { type: "math", tex: String.raw`p(y=1 \mid x) = \sigma\!\big(w^\top x + b\big) = \frac{1}{1 + e^{-(w^\top x + b)}}` },
        { type: "p", text: "and the negative class is whatever is left over, $p(y=0\\mid x) = 1 - \\sigma(w^\\top x + b)$. The vector $w$ says how strongly each feature pushes toward class 1; the bias $b$ sets the baseline log-odds when all features are zero." },
        { type: "heading", text: "The decision boundary is a hyperplane" },
        { type: "p", text: "You predict class 1 when $p \\ge 0.5$. Because $\\sigma$ is monotonic and $\\sigma(0) = 0.5$, that threshold on $p$ corresponds to a threshold on the *score*: $p \\ge 0.5 \\iff w^\\top x + b \\ge 0$. So the surface separating the two predicted classes is" },
        { type: "math", tex: String.raw`w^\top x + b = 0` },
        { type: "p", text: "which is a **hyperplane** — a line in 2-D, a plane in 3-D, a flat sheet in general. This is the crucial structural fact: **logistic regression is a linear classifier.** The sigmoid bends the *probabilities* into an S-curve, but the *boundary* is dead straight. Data that cannot be separated by a straight line (the classic XOR pattern) cannot be perfectly classified by logistic regression — you must either engineer non-linear features or move to a neural network." },
        { type: "callout", variant: "note", text: "**Distance from the boundary = confidence.** The score $w^\\top x + b$ is proportional to the signed distance from the hyperplane (scaled by $\\|w\\|$). Points far on the positive side get $p \\approx 1$; points near the boundary get $p \\approx 0.5$. The sigmoid is the dial that converts 'how far, which side' into 'how sure'." },
        { type: "callout", variant: "gotcha", text: "**Always fold the bias into the weights, or standardize your features — but do not forget the bias entirely.** Without $b$, the boundary is forced through the origin, which is almost never right. In code, either carry a separate `b` scalar (clearest) or prepend a column of ones to $X$ and let $w_0$ be the bias." },
      ]
    },

    {
      id: "loss",
      title: "The loss: binary cross-entropy from maximum likelihood",
      level: "core",
      body: [
        { type: "p", text: "We have a model $p_\\theta(x) = \\sigma(w^\\top x + b)$ with parameters $\\theta = (w, b)$. To fit it we need a loss. Rather than guess one, we **derive** it from a principle: choose the parameters that make the observed labels most probable. This is **maximum likelihood estimation (MLE)**, and for binary labels it produces one specific loss — binary cross-entropy." },
        { type: "heading", text: "Step 1 — the Bernoulli likelihood of one example" },
        { type: "p", text: "A single label $y \\in \\{0,1\\}$ is a **Bernoulli** random variable with success probability $\\hat y = p_\\theta(x)$. Its probability mass function can be written as one slick expression (check: plug in $y=1$ to get $\\hat y$, plug in $y=0$ to get $1-\\hat y$):" },
        { type: "math", tex: String.raw`p(y \mid x) = \hat y^{\,y}\,(1 - \hat y)^{\,1-y}` },
        { type: "heading", text: "Step 2 — likelihood of the whole dataset" },
        { type: "p", text: "Assuming examples are independent, the probability of *all* $m$ labels is the product of the individual probabilities:" },
        { type: "math", tex: String.raw`\mathcal{L}(\theta) = \prod_{i=1}^{m} \hat y_i^{\,y_i}\,(1 - \hat y_i)^{\,1 - y_i}` },
        { type: "heading", text: "Step 3 — take the log, then negate" },
        { type: "p", text: "Products of many small numbers underflow and are painful to differentiate. Taking the logarithm turns the product into a sum (monotonic, so the maximizer is unchanged). Maximizing the log-likelihood is the same as minimizing its negative, and dividing by $m$ gives a per-example average — the **binary cross-entropy (BCE)**, also called **log-loss**:" },
        { type: "math", tex: String.raw`\mathcal{L}_{\text{BCE}}(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\Big[\,y_i \ln \hat y_i + (1 - y_i)\ln(1 - \hat y_i)\,\Big]` },
        { type: "p", text: "Read it intuitively: for a positive example ($y=1$) only the $\\ln \\hat y$ term survives, so the loss is $-\\ln \\hat y$ — zero when you predict $\\hat y = 1$, exploding to $+\\infty$ as $\\hat y \\to 0$. The loss punishes **confident wrongness** without mercy, and rewards confident correctness. That asymmetry is exactly what you want from a probability model." },
        { type: "callout", variant: "note", text: "**Why it's called cross-entropy.** In information theory, cross-entropy $H(P, Q) = -\\sum P \\log Q$ measures the extra bits needed to encode outcomes drawn from the true distribution $P$ using a code built for your model $Q$. Minimizing BCE minimizes exactly that gap. Same formula, deeper name — and it generalizes directly to the softmax loss in the next section." },
        { type: "heading", text: "Why NOT mean squared error?" },
        { type: "p", text: "You *could* write $\\tfrac{1}{m}\\sum(\\hat y_i - y_i)^2$ with $\\hat y = \\sigma(z)$. People try it; it is a mistake, for two rigorous reasons:" },
        { type: "list", ordered: true, items: [
          "**Non-convexity.** BCE composed with the sigmoid is **convex** in $(w,b)$ — one global minimum, gradient descent cannot get stuck. MSE composed with the sigmoid is **non-convex**: it has flat regions and local minima, so optimization can stall in a bad spot depending on initialization.",
          "**Vanishing gradients on wrong, confident predictions.** With MSE the gradient carries a factor of $\\sigma'(z) = \\hat y(1-\\hat y)$. When the model is confidently *wrong* (say $\\hat y \\approx 0.99$ but $y = 0$), $\\sigma' \\approx 0.01$ throttles the gradient to near-zero — the model barely corrects the very mistake it should fix fastest. With BCE that $\\sigma'$ term **cancels exactly** (next section), so the gradient stays strong precisely when the error is large.",
        ]},
        { type: "callout", variant: "gotcha", text: "Numerically, $\\ln \\hat y$ is $-\\infty$ when $\\hat y$ hits exactly $0$ or $1$. Never feed raw probabilities into `log`. Either clip with a small $\\varepsilon$ (`np.clip(yhat, 1e-12, 1-1e-12)`), or — far better — compute the loss directly from the logits $z$ using the log-sum-exp trick (`BCEWithLogitsLoss` in PyTorch, `from_logits=True` in Keras). Frameworks fuse the sigmoid and the log for exactly this stability." },
      ]
    },

    {
      id: "gradient",
      title: "The gradient: the clean $\\hat y - y$ result, and GD from scratch",
      level: "core",
      body: [
        { type: "p", text: "To minimize the loss we need $\\nabla_\\theta \\mathcal{L}$. The derivation is short and the payoff is one of the most beautiful cancellations in ML — the gradient of BCE-through-sigmoid is just **prediction minus target**, with no leftover $\\sigma'$ term." },
        { type: "heading", text: "Derivation for a single example" },
        { type: "p", text: "Let $z = w^\\top x + b$, $\\hat y = \\sigma(z)$, and the per-example loss $\\ell = -[y\\ln\\hat y + (1-y)\\ln(1-\\hat y)]$. Apply the chain rule $\\tfrac{\\partial \\ell}{\\partial z} = \\tfrac{\\partial \\ell}{\\partial \\hat y}\\cdot\\tfrac{\\partial \\hat y}{\\partial z}$. First factor:" },
        { type: "math", tex: String.raw`\frac{\partial \ell}{\partial \hat y} = -\frac{y}{\hat y} + \frac{1 - y}{1 - \hat y} = \frac{\hat y - y}{\hat y\,(1 - \hat y)}` },
        { type: "p", text: "Second factor is the sigmoid identity from before, $\\tfrac{\\partial \\hat y}{\\partial z} = \\hat y(1 - \\hat y)$. Multiply, and watch the denominator annihilate:" },
        { type: "math", tex: String.raw`\frac{\partial \ell}{\partial z} = \frac{\hat y - y}{\hat y\,(1-\hat y)} \cdot \hat y\,(1 - \hat y) = \hat y - y` },
        { type: "p", text: "Then $z = w^\\top x + b$ is linear, so $\\tfrac{\\partial z}{\\partial w} = x$ and $\\tfrac{\\partial z}{\\partial b} = 1$. Chaining once more and averaging over the dataset gives the full gradients in clean matrix form (with $X \\in \\mathbb{R}^{m\\times n}$, predictions $\\hat y \\in \\mathbb{R}^m$):" },
        { type: "math", tex: String.raw`\nabla_w \mathcal{L} = \frac{1}{m}\,X^\top(\hat y - y), \qquad \nabla_b \mathcal{L} = \frac{1}{m}\sum_{i=1}^{m}(\hat y_i - y_i)` },
        { type: "callout", variant: "tip", text: "**Remember this shape:** $X^\\top(\\hat y - y)$. It is *identical* to the least-squares gradient from the Linear Algebra track — only the definition of $\\hat y$ changed (from $Xw$ to $\\sigma(Xw)$). And the term $\\delta = \\hat y - y$ is exactly the **output-layer error** in backprop. Linear regression, logistic regression, and a neural net's last layer all share this one gradient. Learn it once, use it everywhere." },
        { type: "heading", text: "Gradient descent from scratch" },
        { type: "p", text: "There is no closed-form solution like the normal equation here — the sigmoid makes it nonlinear — so we iterate. Because the loss is convex, plain gradient descent is guaranteed to reach the global optimum." },
        { type: "code", lang: "py", code: "import numpy as np\nrng = np.random.default_rng(0)\n\ndef sigmoid(z):\n    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))\n\ndef bce_loss(y, yhat, eps=1e-12):\n    yhat = np.clip(yhat, eps, 1 - eps)\n    return -np.mean(y*np.log(yhat) + (1-y)*np.log(1-yhat))\n\ndef fit_logreg(X, y, lr=0.1, epochs=2000):\n    m, n = X.shape\n    w, b = np.zeros(n), 0.0\n    for epoch in range(epochs):\n        z = X @ w + b\n        yhat = sigmoid(z)              # forward pass\n        error = yhat - y              # <-- the clean delta\n        grad_w = X.T @ error / m      # nabla_w = X^T (yhat - y) / m\n        grad_b = error.mean()        # nabla_b\n        w -= lr * grad_w             # gradient-descent step\n        b -= lr * grad_b\n        if epoch % 400 == 0:\n            print(f\"epoch {epoch:4d}  loss {bce_loss(y, yhat):.4f}\")\n    return w, b\n\n# Toy: two Gaussian blobs, linearly separable-ish\nX = np.vstack([rng.normal(-1, 1, (100, 2)),\n               rng.normal(+2, 1, (100, 2))])\ny = np.array([0]*100 + [1]*100, dtype=float)\n\nw, b = fit_logreg(X, y, lr=0.5, epochs=2000)\nprob = sigmoid(X @ w + b)\nacc = ((prob > 0.5) == y).mean()\nprint(f\"weights {w.round(2)}  bias {b:.2f}  accuracy {acc:.1%}\")" },
        { type: "callout", variant: "gotcha", text: "Notice there is **no learning-rate-sensitive closed form** and no matrix inversion — logistic regression is trained iteratively even by scikit-learn (which uses smarter second-order solvers like L-BFGS and `newton-cg`, but the same convex objective). If your from-scratch loss goes up instead of down, your learning rate is too big or you flipped a sign on the gradient." },
      ]
    },

    {
      id: "multiclass",
      title: "Multiclass: softmax regression & categorical cross-entropy",
      level: "core",
      body: [
        { type: "p", text: "Real problems often have more than two classes: digit 0–9, one of 1000 ImageNet categories, next-token over a 50k vocabulary. Two standard ways to get there — and one of them scales all the way to GPT's final layer." },
        { type: "heading", text: "Option A — One-vs-Rest (OvR)" },
        { type: "p", text: "Train $K$ separate binary logistic classifiers, each answering 'class $k$ or not?'. At prediction time, run all $K$ and pick the one with the highest score. Simple, embarrassingly parallel, and the default fallback for many libraries. Its weakness: the $K$ probabilities are produced independently and **do not sum to 1**, so they are not a coherent distribution and can disagree at the margins." },
        { type: "heading", text: "Option B — Softmax (multinomial) regression" },
        { type: "p", text: "The principled generalization. Give each class its own weight vector — stack them into $W \\in \\mathbb{R}^{K\\times n}$ and biases $b \\in \\mathbb{R}^{K}$ — producing a vector of $K$ scores (**logits**) $z = Wx + b$. Then the **softmax** turns those logits into a proper probability distribution over the classes:" },
        { type: "math", tex: String.raw`p(y = k \mid x) = \operatorname{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}` },
        { type: "p", text: "Every entry is positive and they sum to exactly 1 by construction. Softmax is the multi-class sibling of the sigmoid — in fact for $K=2$ softmax reduces to the sigmoid. The exponentials exaggerate differences (the largest logit dominates), which is where the 'soft-max' name comes from: a smooth, differentiable approximation to picking the argmax." },
        { type: "heading", text: "Categorical cross-entropy" },
        { type: "p", text: "With the true label one-hot encoded as $y \\in \\{0,1\\}^K$ (a 1 in the correct class slot), the MLE loss generalizes BCE to the **categorical cross-entropy**:" },
        { type: "math", tex: String.raw`\mathcal{L}_{\text{CE}} = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{ik}\,\ln \hat y_{ik} = -\frac{1}{m}\sum_{i=1}^{m}\ln \hat y_{i, c_i}` },
        { type: "p", text: "The right-hand form shows what it really does: for each example it is just $-\\ln$ of the probability the model assigned to the *correct* class $c_i$. Maximize the probability of the truth; nothing else matters." },
        { type: "callout", variant: "tip", text: "**The magic cancellation carries over.** Differentiating categorical cross-entropy through the softmax gives, once again, $\\nabla_z \\mathcal{L} = \\hat y - y$ — predicted distribution minus the one-hot target, in matrix form $\\tfrac{1}{m}X^\\top(\\hat Y - Y)$. Sigmoid+BCE and softmax+CE are the *same object* at different sizes, and both hand you the frictionless $\\hat y - y$ output gradient that makes backprop's last step trivial." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef softmax(Z):\n    # Z: (m, K) logits. Subtract row-max for numerical stability.\n    Z = Z - Z.max(axis=1, keepdims=True)\n    E = np.exp(Z)\n    return E / E.sum(axis=1, keepdims=True)\n\ndef fit_softmax(X, Y, K, lr=0.1, epochs=1000):\n    # X: (m, n), Y: (m, K) one-hot\n    m, n = X.shape\n    W = np.zeros((n, K))\n    b = np.zeros(K)\n    for _ in range(epochs):\n        P = softmax(X @ W + b)        # (m, K) predicted distribution\n        dZ = (P - Y) / m              # <-- yhat - y, again\n        W -= lr * (X.T @ dZ)          # (n, K)\n        b -= lr * dZ.sum(axis=0)\n    return W, b\n\n# Fake 3-class data\nrng = np.random.default_rng(0)\nX = rng.normal(size=(300, 2))\ncls = ((X[:,0] + X[:,1]) > 0).astype(int) + (X[:,0] > 1).astype(int)\nY = np.eye(3)[cls]                     # one-hot, K=3\nW, b = fit_softmax(X, Y, K=3, lr=0.5, epochs=2000)\npred = softmax(X @ W + b).argmax(1)\nprint(\"accuracy:\", (pred == cls).mean().round(3))" },
        { type: "callout", variant: "gotcha", text: "**Softmax is shift-invariant:** adding any constant $c$ to every logit leaves the output unchanged, because $e^{z_k+c}/\\sum e^{z_j+c} = e^{z_k}/\\sum e^{z_j}$. That is *why* subtracting the row-max is safe and mandatory — it prevents `np.exp` from overflowing on large logits without changing a single probability. Forgetting it is the classic softmax bug." },
      ]
    },

    {
      id: "threshold",
      title: "Thresholds, class imbalance & calibration",
      level: "core",
      body: [
        { type: "p", text: "A logistic model outputs a probability; turning that into a *decision* requires a **threshold**, and $0.5$ is only the default, not a law. Choosing it well is where classification meets the real world." },
        { type: "heading", text: "Moving the threshold trades precision for recall" },
        { type: "p", text: "Lower the threshold (say to $0.2$) and you flag more positives — you catch more true positives (**higher recall**) but also more false alarms (**lower precision**). Raise it and the trade reverses. The right operating point depends on the *cost of each error type*: for cancer screening a missed tumour is catastrophic, so you accept many false positives and set a low threshold; for a spam filter, dumping a real email into spam is worse than letting one spam through, so you set a high one." },
        { type: "table",
          headers: ["Threshold", "Effect", "Use when"],
          rows: [
            ["Low (e.g. $0.2$)", "high recall, low precision", "missing a positive is very costly (disease, fraud)"],
            ["Default $0.5$", "balanced (only if classes balanced)", "symmetric error costs"],
            ["High (e.g. $0.8$)", "high precision, low recall", "a false positive is very costly (spam, auto-actions)"],
          ]
        },
        { type: "callout", variant: "tip", text: "Choose the threshold by plotting the **precision–recall curve** (or ROC curve) on a validation set and reading off the point that meets your business constraint — do not hard-code $0.5$. The threshold is a *deployment* decision, entirely separate from *training* the model, and you can retune it without retraining." },
        { type: "heading", text: "Class imbalance" },
        { type: "p", text: "When positives are rare (fraud is <1% of transactions), a model can hit 99% accuracy by predicting 'negative' for everything — useless. Accuracy is the wrong metric; use precision, recall, F1, or AUC. Fixes: **class weighting** (scale each class's loss by its inverse frequency, `class_weight='balanced'`), **resampling** (oversample the minority with SMOTE, or undersample the majority), and always **moving the threshold** to reflect the true base rate." },
        { type: "heading", text: "Calibration — are the probabilities honest?" },
        { type: "p", text: "A model is **calibrated** if, among all cases where it says '$70\\%$', roughly $70\\%$ are actually positive. Logistic regression is naturally *well-calibrated* because it optimizes a proper scoring rule (log-loss) — one of its underrated superpowers, and a reason it is used as a final calibration layer on top of other models. Contrast: SVMs and naive Bayes produce poorly-calibrated scores that need post-hoc **Platt scaling** (which is literally fitting a 1-D logistic regression on the scores) or **isotonic regression**." },
        { type: "callout", variant: "note", text: "Check calibration with a **reliability diagram**: bin predictions by their stated probability and plot the observed positive rate in each bin against the bin's midpoint. A perfectly calibrated model traces the diagonal. `sklearn.calibration.calibration_curve` and `CalibratedClassifierCV` do this for you." },
      ]
    },

    {
      id: "regularization",
      title: "Regularized logistic regression (L2 & L1)",
      level: "core",
      body: [
        { type: "p", text: "On separable or high-dimensional data, unregularized logistic regression will happily drive weights toward $\\pm\\infty$ to squeeze the last drop of log-loss — overfitting and numerical blow-up. The cure is the same **penalty** idea from linear regression: add a term that discourages large weights, trading a little training fit for much better generalization." },
        { type: "math", tex: String.raw`\mathcal{L}_{\text{reg}} = \underbrace{-\frac{1}{m}\sum_i \big[y_i\ln\hat y_i + (1-y_i)\ln(1-\hat y_i)\big]}_{\text{cross-entropy}} \;+\; \underbrace{\lambda\, R(w)}_{\text{penalty}}` },
        { type: "table",
          headers: ["Penalty", "$R(w)$", "Effect", "Analogue"],
          rows: [
            ["L2 (ridge)", "$\\tfrac{1}{2}\\|w\\|_2^2$", "shrinks weights smoothly toward 0; the default", "ridge regression"],
            ["L1 (lasso)", "$\\|w\\|_1$", "drives some weights **exactly** to 0 -> feature selection", "lasso regression"],
            ["Elastic net", "$\\alpha\\|w\\|_1 + \\tfrac{1-\\alpha}{2}\\|w\\|_2^2$", "sparsity + stability with correlated features", "elastic net"],
          ]
        },
        { type: "p", text: "L2 only changes the gradient by a single friendly term — add $\\tfrac{\\lambda}{m} w$ to $\\nabla_w$ (equivalently, multiply $w$ by $(1 - \\eta\\lambda/m)$ each step, which is why L2 is also called **weight decay**):" },
        { type: "math", tex: String.raw`\nabla_w \mathcal{L}_{\text{reg}} = \frac{1}{m}X^\top(\hat y - y) + \frac{\lambda}{m}\,w` },
        { type: "callout", variant: "gotcha", text: "**Never regularize the bias**, and **always standardize features first**. The penalty assumes every weight is on the same scale; if one feature is in dollars and another in kilometres, L2 will unfairly crush the small-magnitude one. And scikit-learn parameterizes strength by $C = 1/\\lambda$ — so *smaller* `C` means *stronger* regularization, the opposite of what beginners expect." },
        { type: "callout", variant: "note", text: "**Forward reference.** L1 vs L2 geometry — why the L1 diamond's corners produce exact zeros while the L2 circle does not — is developed fully in the Regularization & Model Selection topic. The same penalties reappear as **weight decay** in the Training Deep Networks track. It is one idea wearing three different names across the curriculum." },
      ]
    },

    {
      id: "sklearn",
      title: "scikit-learn in practice (vs. from scratch)",
      level: "core",
      body: [
        { type: "p", text: "You now understand every line of what the library does. In production you call the one-liner — but knowing the internals means you can read every argument, debug convergence warnings, and trust the output. Here is a complete, honest workflow on a real dataset, plus a check that our from-scratch model agrees." },
        { type: "code", lang: "py", code: "import numpy as np\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import classification_report, roc_auc_score\n\nX, y = load_breast_cancer(return_X_y=True)   # 569 samples, 30 features\nXtr, Xte, ytr, yte = train_test_split(\n    X, y, test_size=0.25, stratify=y, random_state=0)\n\n# Standardize: essential so regularization treats features fairly.\nscaler = StandardScaler().fit(Xtr)\nXtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)\n\nclf = LogisticRegression(\n    penalty=\"l2\",      # default; use 'l1' with solver='liblinear'/'saga'\n    C=1.0,             # inverse reg strength: smaller C = stronger penalty\n    solver=\"lbfgs\",    # quasi-Newton; robust default for L2\n    max_iter=1000,\n)\nclf.fit(Xtr_s, ytr)\n\nproba = clf.predict_proba(Xte_s)[:, 1]         # P(malignant)\npred  = (proba > 0.5).astype(int)\nprint(classification_report(yte, pred, digits=3))\nprint(\"ROC-AUC:\", round(roc_auc_score(yte, proba), 4))   # ~0.997\nprint(\"n weights:\", clf.coef_.shape, \"bias:\", clf.intercept_.round(3))" },
        { type: "heading", text: "Sanity-check against the hand-rolled version" },
        { type: "code", lang: "py", code: "# Our fit_logreg from the gradient section, on the same standardized data.\nw, b = fit_logreg(Xtr_s, ytr.astype(float), lr=0.5, epochs=5000)\nour_proba = sigmoid(Xte_s @ w + b)\nour_acc   = ((our_proba > 0.5) == yte).mean()\n\nsk_acc = (pred == yte).mean()\nprint(f\"from-scratch acc {our_acc:.3f}   sklearn acc {sk_acc:.3f}\")\n# The two agree to within a fraction of a percent. Same model, same math;\n# sklearn just uses a smarter (2nd-order) optimizer and reaches it faster.\nprint(\"weight correlation:\",\n      round(np.corrcoef(w, clf.coef_.ravel())[0, 1], 3))  # ~0.99" },
        { type: "callout", variant: "tip", text: "**Reading the arguments through your new eyes:** `predict_proba` returns the sigmoid outputs you derived; `coef_` is $w$, `intercept_` is $b$; `C` is $1/\\lambda$; `solver='lbfgs'` is a second-order method that approximates the Hessian to converge in far fewer steps than plain GD. `multi_class='multinomial'` switches on the softmax loss from the multiclass section. Nothing here is a black box anymore." },
        { type: "callout", variant: "gotcha", text: "If you see `ConvergenceWarning: lbfgs failed to converge`, the fix is almost always (1) **standardize your features** — unscaled inputs make the loss surface ill-conditioned — or (2) raise `max_iter`. It is rarely a reason to distrust the model; it means the optimizer ran out of iterations on a badly-scaled problem." },
      ]
    },

    {
      id: "nn-connection",
      title: "Logistic regression IS a one-layer neural network",
      level: "core",
      body: [
        { type: "p", text: "This section is the whole reason logistic regression sits where it does in this curriculum. Look again at the model and compare it to a single artificial neuron:" },
        { type: "math", tex: String.raw`\underbrace{\hat y = \sigma\!\big(w^\top x + b\big)}_{\text{logistic regression}} \qquad\Longleftrightarrow\qquad \underbrace{a = \sigma\!\big(W x + b\big)}_{\text{one neuron with sigmoid activation}}` },
        { type: "p", text: "They are the **same equation**. Logistic regression is a neural network with **zero hidden layers** — an input layer wired straight to a single sigmoid output unit. Everything you learned here is the atom from which deep networks are built:" },
        { type: "table",
          headers: ["Logistic regression concept", "Neural network concept"],
          rows: [
            ["Linear score $w^\\top x + b$", "a **pre-activation** $z$ (one unit)"],
            ["Sigmoid $\\sigma(z)$", "the **activation function**"],
            ["Weights $w$, bias $b$", "layer **parameters** $W, b$"],
            ["Binary cross-entropy loss", "the **output loss** for binary classification"],
            ["Gradient $X^\\top(\\hat y - y)$", "the **output-layer delta** $\\delta = \\hat y - y$ in backprop"],
            ["Gradient descent on $\\mathcal{L}$", "one step of **training** the network"],
            ["Softmax regression", "the **softmax output layer** (image nets, LLMs)"],
          ]
        },
        { type: "p", text: "Stack a hidden layer of these neurons with a nonlinearity in between, and you get a **multi-layer perceptron** that can carve *curved* decision boundaries — escaping logistic regression's straight-line limitation. Add convolutions and you get a CNN; add attention and you get a Transformer. But the last layer of virtually every classifier — a sigmoid for binary, a softmax for multiclass, trained with cross-entropy — is *exactly this topic*. GPT's final layer is a softmax regression over the vocabulary." },
        { type: "callout", variant: "good", text: "**The single most valuable takeaway of the Classical ML track:** master the $\\hat y - y$ gradient here, and the hardest part of backpropagation — the output layer — is already behind you. The Neural Networks track only adds one new idea on top of it: the chain rule pushing that same $\\delta$ backward through hidden layers. You have built the seed of a neural net from first principles." },
        { type: "callout", variant: "note", text: "**Forward reference.** Continue to **Backpropagation & Automatic Differentiation** in the Neural Networks track, where $\\delta = \\hat y - y$ becomes $\\delta^{[L]}$, the error at the output layer, and the recurrence $\\delta^{[l]} = (W^{[l+1]})^\\top\\delta^{[l+1]}\\odot\\phi'(z^{[l]})$ carries it back through the network." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Build at least two of these end-to-end. The goal is fluency: you should be able to write the sigmoid, the BCE loss, and the $\\hat y - y$ gradient from memory, and explain why MSE would be worse. Everything except the last uses only NumPy." },
        { type: "list", ordered: true, items: [
          "**Logistic regression from scratch, fully.** Implement `sigmoid`, `bce_loss`, and gradient descent on a 2-D toy dataset. Plot the data, the sigmoid probability surface as a heatmap, and the linear decision boundary $w^\\top x + b = 0$. Then feed it XOR-patterned data and *watch it fail* — proving to yourself that the boundary is linear.",
          "**MSE vs BCE, empirically.** Train the same model twice, once minimizing BCE and once MSE-through-sigmoid, from several random initializations. Plot both loss curves; show MSE stalling on confidently-wrong points while BCE powers through. This makes the 'why not MSE' argument concrete.",
          "**Softmax regression on MNIST.** Extend to multiclass and classify handwritten digits (784 features -> 10 classes) with softmax + categorical cross-entropy in pure NumPy. You should reach ~92% test accuracy — and you have just built the output layer of every image classifier.",
          "**Threshold & imbalance lab.** Take an imbalanced dataset (e.g. credit-card fraud). Plot the precision–recall curve, pick a threshold for a target recall, then compare `class_weight='balanced'` vs SMOTE oversampling. Report F1 and AUC, not accuracy.",
          "**Calibration check.** Fit logistic regression and an SVM on the same data; draw reliability diagrams for both. Confirm logistic regression hugs the diagonal while the raw SVM scores do not, then fix the SVM with `CalibratedClassifierCV` (Platt scaling).",
          "**Interpretability report.** On a tabular dataset, fit L1-regularized logistic regression, then rank features by weight and report the **odds ratios** $e^{w_j}$. Write two sentences per top feature explaining its effect — the skill that makes logistic regression beloved in regulated industries.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The best places to reinforce and extend this material, in recommended order:" },
        { type: "link", url: "https://cs229.stanford.edu/main_notes.pdf", text: "Stanford CS229 lecture notes (Andrew Ng) — the canonical derivation of logistic regression, MLE, and softmax; sections 1–3 are the gold standard" },
        { type: "link", url: "https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression", text: "scikit-learn User Guide — LogisticRegression: solvers, penalties, multiclass options, and the C/λ convention explained" },
        { type: "link", url: "https://www.statlearning.com/", text: "An Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani) — Chapter 4, free PDF; the clearest textbook treatment with the odds-ratio interpretation" },
        { type: "link", url: "https://web.stanford.edu/~jurafsky/slp3/5.pdf", text: "Jurafsky & Martin, Speech and Language Processing — Chapter 5 is an outstanding, self-contained logistic-regression + cross-entropy tutorial" },
        { type: "link", url: "https://www.deeplearningbook.org/contents/mlp.html", text: "Goodfellow, Bengio & Courville, Deep Learning — see how sigmoid/softmax + cross-entropy become the output layer of a deep net (the connection this topic previews)" },
        { type: "link", url: "https://scikit-learn.org/stable/modules/calibration.html", text: "scikit-learn — Probability calibration: reliability diagrams, Platt scaling, and isotonic regression, with worked examples" },
      ]
    },
  ],

  packages: [
    { name: "numpy", why: "build the sigmoid, BCE loss, and $X^\\top(\\hat y - y)$ gradient from scratch" },
    { name: "sklearn.linear_model.LogisticRegression", why: "the production one-liner — L1/L2, multinomial softmax, robust solvers" },
    { name: "sklearn.preprocessing.StandardScaler", why: "standardize features so regularization is fair and solvers converge" },
    { name: "sklearn.metrics", why: "`classification_report`, `roc_auc_score`, `precision_recall_curve` — the right metrics for classification" },
    { name: "sklearn.calibration", why: "`calibration_curve`, `CalibratedClassifierCV` — check and fix probability honesty" },
    { name: "torch.nn.BCEWithLogitsLoss", why: "numerically-stable fused sigmoid + cross-entropy; the same loss inside a neural net" },
    { name: "imbalanced-learn", why: "SMOTE and resampling utilities for skewed class distributions" },
  ],

  gotchas: [
    "Don't use MSE with a sigmoid: the loss becomes **non-convex** and the $\\sigma'$ factor kills the gradient exactly when the model is confidently wrong. Use cross-entropy.",
    "Never `log(yhat)` on raw probabilities — clip to $[\\varepsilon, 1-\\varepsilon]$, or compute the loss from logits with `BCEWithLogitsLoss` / `from_logits=True`.",
    "scikit-learn's `C` is **inverse** regularization strength: smaller `C` = *stronger* penalty. This trips up almost everyone.",
    "**Standardize features** before regularized logistic regression, and **never penalize the bias** — otherwise the penalty is applied unequally across differently-scaled weights.",
    "Softmax must subtract the row-max before `exp` (`z - z.max()`); it's shift-invariant, so this changes nothing but prevents overflow.",
    "The decision boundary is **linear** ($w^\\top x + b = 0$). Non-linearly-separable data (XOR) is impossible to fit perfectly — engineer features or use a neural net.",
    "`0.5` is just the default threshold. Tune it on a validation PR/ROC curve to match your error costs, especially under class imbalance.",
    "Accuracy is misleading on imbalanced data — a 99%-negative dataset scores 99% by predicting all-negative. Use precision, recall, F1, or AUC.",
  ],

  flashcards: [
    { q: "Why can't linear regression output probabilities?", a: "Its output is unbounded (can exceed $[0,1]$), it assumes Gaussian not Bernoulli noise, and outliers rotate the boundary. Logistic regression fixes all three." },
    { q: "What is the modelling assumption of logistic regression?", a: "The **log-odds are linear in the features**: $\\ln\\!\\big(p/(1-p)\\big) = w^\\top x + b$. The sigmoid is the consequence of solving that for $p$." },
    { q: "Write the sigmoid and its derivative.", a: "$\\sigma(z) = 1/(1+e^{-z})$, and $\\sigma'(z) = \\sigma(z)\\,(1-\\sigma(z))$ — the derivative expressed in terms of the function itself." },
    { q: "Where does binary cross-entropy come from?", a: "It is the **negative log-likelihood of a Bernoulli**: $-\\tfrac1m\\sum[y\\ln\\hat y + (1-y)\\ln(1-\\hat y)]$. Maximizing likelihood = minimizing log-loss." },
    { q: "Why NOT mean squared error for classification?", a: "MSE-through-sigmoid is **non-convex**, and its gradient carries a $\\sigma'$ factor that vanishes on confident wrong predictions. BCE is convex and cancels that factor." },
    { q: "What is the gradient of BCE w.r.t. the weights?", a: "$\\nabla_w \\mathcal{L} = \\tfrac1m X^\\top(\\hat y - y)$ — the $\\sigma'$ term cancels, leaving the clean 'prediction minus target' form." },
    { q: "What does softmax compute, and its loss?", a: "$\\operatorname{softmax}(z)_k = e^{z_k}/\\sum_j e^{z_j}$, a distribution over $K$ classes; trained with categorical cross-entropy $-\\sum_k y_k\\ln\\hat y_k$. Gradient is again $\\hat y - y$." },
    { q: "How is logistic regression a neural network?", a: "It's a **one-layer net**: input -> single sigmoid unit. The output-layer backprop error $\\delta = \\hat y - y$ is exactly logistic regression's gradient." },
    { q: "What is an odds ratio in a fitted model?", a: "$e^{w_j}$ — the multiplicative change in the odds of $y=1$ when feature $j$ increases by one unit. The basis of logistic regression's interpretability." },
    { q: "Why is logistic regression well-calibrated?", a: "It minimizes log-loss, a proper scoring rule, so its probabilities are honest. That's why it's used for Platt scaling on top of other models." },
    { q: "In scikit-learn, what does `C` control?", a: "Inverse regularization strength, $C = 1/\\lambda$. Smaller `C` means stronger L2/L1 penalty — the opposite of most people's intuition." },
  ],

  cheatsheet: [
    { label: "Sigmoid", code: "1 / (1 + np.exp(-z))" },
    { label: "Sigmoid derivative", code: "s * (1 - s)  # s = sigmoid(z)" },
    { label: "BCE loss", code: "-np.mean(y*np.log(p) + (1-y)*np.log(1-p))" },
    { label: "Weight gradient", code: "X.T @ (yhat - y) / m" },
    { label: "Bias gradient", code: "(yhat - y).mean()" },
    { label: "GD step", code: "w -= lr * grad_w;  b -= lr * grad_b" },
    { label: "Softmax (stable)", code: "e = np.exp(Z - Z.max(1, keepdims=True)); e / e.sum(1, keepdims=True)" },
    { label: "L2 gradient term", code: "+ (lam / m) * w  # never on the bias" },
    { label: "Fit (sklearn)", code: "LogisticRegression(C=1.0, max_iter=1000).fit(Xs, y)" },
    { label: "Probabilities", code: "clf.predict_proba(Xs)[:, 1]" },
    { label: "Custom threshold", code: "(proba > 0.3).astype(int)" },
    { label: "Multiclass softmax", code: "LogisticRegression(multi_class='multinomial')" },
    { label: "Standardize", code: "StandardScaler().fit_transform(X_train)" },
    { label: "Stable loss (logits)", code: "torch.nn.BCEWithLogitsLoss()(z, y)" },
  ],
});
