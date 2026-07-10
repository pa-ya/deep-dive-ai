(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "ml-linear-regression",
  name: "Linear Regression",
  language: "Classical ML",
  group: "Classical Machine Learning",
  navLabel: "Linear Regression",
  tagline: "The first real learning algorithm — derived from geometry *and* from probability, then built from scratch and in scikit-learn.",
  color: "#059669",
  readMinutes: 48,
  sections: [
    {
      id: "setup",
      title: "The model & problem setup",
      level: "core",
      body: [
        { type: "p", text: "**Linear regression** is the *Hello, World* of machine learning — and, unusually for a Hello-World, it is also genuinely useful and forms the skeleton of almost everything that comes later. Logistic regression, neural-network layers, and the attention mechanism all start from the same object: a **weighted sum of inputs**. Learn this one deeply and the rest of ML is variations on a theme." },
        { type: "p", text: "The setup: you have $m$ examples, each described by $n$ features, and a real-valued target you want to predict — a house's price from its size and age, a patient's blood pressure from their weight and diet. The model assumes the target is (approximately) a **linear function** of the features:" },
        { type: "math", tex: String.raw`\hat y \;=\; w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b \;=\; w^\top x + b` },
        { type: "p", text: "Here $x \\in \\mathbb{R}^n$ is one example's feature vector, $w \\in \\mathbb{R}^n$ are the **weights** (one per feature — how much that feature moves the prediction), and $b$ is the **bias** or intercept (the prediction when every feature is zero). The hat on $\\hat y$ signals *prediction*, distinct from the true label $y$. Learning means finding the $w$ and $b$ that make $\\hat y$ close to $y$ across the whole dataset." },
        { type: "heading", text: "The bias trick — folding $b$ into $w$" },
        { type: "p", text: "Carrying $b$ around separately is annoying. The standard move is to prepend a constant $1$ to every feature vector; then the bias becomes just another weight $w_0$ and the whole model is a single dot product. Nearly all the math below uses this convention." },
        { type: "math", tex: String.raw`\tilde x = \begin{bmatrix} 1 \\ x_1 \\ \vdots \\ x_n \end{bmatrix}, \quad \tilde w = \begin{bmatrix} b \\ w_1 \\ \vdots \\ w_n \end{bmatrix} \;\Longrightarrow\; \hat y = \tilde w^\top \tilde x` },
        { type: "p", text: "Stacking all $m$ examples as rows of a **design matrix** $X \\in \\mathbb{R}^{m \\times n}$ (with that leading column of ones), every prediction happens at once as a single matrix–vector product — the same $Xw$ you met in the Linear Algebra page's least-squares section:" },
        { type: "math", tex: String.raw`\hat y = X w \in \mathbb{R}^m, \qquad X \in \mathbb{R}^{m \times n},\; w \in \mathbb{R}^n` },
        { type: "heading", text: "Regression vs. classification" },
        { type: "p", text: "The target's *type* decides the problem. **Regression** predicts a continuous number (price, temperature, time-to-failure); **classification** predicts a discrete category (spam / not-spam, which of ten digits). Same feature-vector input, different output space — and the loss you use differs accordingly. Linear regression is the archetypal regressor; its cousin logistic regression (next track) reuses this exact machinery for classification by squashing $w^\\top x$ through a sigmoid." },
        { type: "table",
          headers: ["", "Regression", "Classification"],
          rows: [
            ["Target $y$", "continuous, $y \\in \\mathbb{R}$", "discrete class label"],
            ["Example", "predict house price", "predict spam / not spam"],
            ["Typical loss", "mean squared error", "cross-entropy"],
            ["This page", "**yes**", "see Logistic Regression"],
          ]
        },
        { type: "heading", text: "When a linear model is the right call" },
        { type: "p", text: "Linear regression is not a fallback you outgrow — it is often the *correct* choice. Reach for it when:" },
        { type: "list", items: [
          "You need an **interpretable** model: each weight $w_j$ is literally 'how much $\\hat y$ changes per unit of feature $j$, holding the others fixed.' Regulators and doctors love this; a random forest cannot offer it.",
          "You have **few examples relative to features**, or a genuinely near-linear relationship. Complex models overfit small data; a linear model's strong bias is a feature, not a bug.",
          "You want a **fast, strong baseline**. Fit it in milliseconds; if a deep net cannot beat it, the deep net is broken or the signal is weak.",
          "The relationship is non-linear *in the inputs* but you can make it **linear in the parameters** via basis functions (section 6) — polynomials, splines, interactions. This is more powerful than it first looks.",
        ]},
        { type: "callout", variant: "note", text: "**How this page is built.** We derive linear regression **two independent ways** — as geometry (project $y$ onto a column space, the normal equation) and as probability (maximize likelihood under Gaussian noise, which *is* minimizing squared error) — and show they land on the identical answer. Seeing one result fall out of two unrelated arguments is the moment ML stops feeling arbitrary." },
      ]
    },

    {
      id: "cost",
      title: "The cost function: mean squared error",
      level: "core",
      body: [
        { type: "p", text: "A model is only as good as the yardstick you judge it by. The **cost function** (loss, objective) turns a choice of $w$ into a single number measuring how wrong the predictions are; learning is then just *minimize this number*. For regression the default is **mean squared error (MSE)** — average the squared gap between prediction and truth:" },
        { type: "math", tex: String.raw`J(w) \;=\; \frac{1}{m}\sum_{i=1}^{m}\big(\hat y^{(i)} - y^{(i)}\big)^2 \;=\; \frac{1}{m}\,\|Xw - y\|_2^2` },
        { type: "p", text: "The superscript $(i)$ indexes examples. Some texts write $\\tfrac{1}{2m}$ instead of $\\tfrac{1}{m}$; the $\\tfrac12$ is a cosmetic trick that cancels the $2$ from differentiating the square, and scaling the loss by a constant never changes *where* the minimum is. Both are 'MSE'." },
        { type: "heading", text: "Why *squared*? Three answers, from weakest to deepest" },
        { type: "p", text: "**(1) It is smooth and convex.** Squaring gives a differentiable, bowl-shaped surface (a positive-semidefinite quadratic form, in the Linear Algebra page's language) with a single global minimum — exactly the well-behaved landscape gradient descent and the closed form both need. Absolute error $|\\hat y - y|$ has a kink at zero and no clean derivative there." },
        { type: "p", text: "**(2) It punishes large errors disproportionately.** An error of $10$ costs $100$; an error of $1$ costs $1$. The model will bend over backwards to eliminate big misses, which is usually what you want — but note the flip side: it is **sensitive to outliers** (one wild point can dominate the sum). Absolute-error loss is the robust alternative." },
        { type: "heading", text: "(3) The deep answer: MSE is maximum likelihood under Gaussian noise" },
        { type: "p", text: "This is the argument from the Probability page, restated here because it is the reason squared error is not an arbitrary choice but *the* principled one. Assume the world generates targets as a linear function plus **independent Gaussian noise**:" },
        { type: "math", tex: String.raw`y^{(i)} = w^\top x^{(i)} + \varepsilon^{(i)}, \qquad \varepsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)\ \text{i.i.d.}` },
        { type: "p", text: "Then each target is Gaussian-distributed around the model's prediction, so the probability (density) of observing $y^{(i)}$ given the inputs and weights is:" },
        { type: "math", tex: String.raw`p\big(y^{(i)} \mid x^{(i)}; w\big) = \frac{1}{\sqrt{2\pi\sigma^2}}\,\exp\!\left(-\frac{\big(y^{(i)} - w^\top x^{(i)}\big)^2}{2\sigma^2}\right)` },
        { type: "p", text: "By independence, the **likelihood** of the whole dataset is the product over examples. Products of exponentials are miserable to differentiate, so we take the **log-likelihood** (monotonic — same maximizer) which turns the product into a sum and the exp into its exponent:" },
        { type: "math", tex: String.raw`\log \mathcal{L}(w) = \sum_{i=1}^{m} \log p\big(y^{(i)}\mid x^{(i)};w\big) = -\frac{1}{2\sigma^2}\sum_{i=1}^{m}\big(y^{(i)} - w^\top x^{(i)}\big)^2 \;-\; m\log\sqrt{2\pi\sigma^2}` },
        { type: "p", text: "The second term does not depend on $w$, and the leading $-\\tfrac{1}{2\\sigma^2}$ is a positive constant. So **maximizing** the log-likelihood over $w$ is identical to **minimizing** $\\sum_i (y^{(i)} - w^\\top x^{(i)})^2$ — which is exactly the sum of squared errors. That is the whole point:" },
        { type: "math", tex: String.raw`\underbrace{\arg\max_w\; \log\mathcal{L}(w)}_{\text{maximum likelihood}} \;=\; \underbrace{\arg\min_w\; \sum_{i=1}^m \big(y^{(i)} - w^\top x^{(i)}\big)^2}_{\text{least squares}}` },
        { type: "callout", variant: "good", text: "**Squared error is not a convenient hack — it is the maximum-likelihood estimator under the assumption of Gaussian noise.** This reframing pays off constantly: swap the noise assumption and you derive a *different* loss. Laplacian noise gives absolute-error (robust) regression; Bernoulli 'noise' gives the cross-entropy of logistic regression. Every loss function is a likelihood in disguise." },
        { type: "callout", variant: "gotcha", text: "The Gaussian assumption also implies **constant** noise variance $\\sigma^2$ across all examples (homoscedasticity). When that fails — noise grows with the target's magnitude — plain MSE is no longer the right likelihood, and residual plots (section 7) will show it as a tell-tale funnel shape." },
      ]
    },

    {
      id: "normal-equation",
      title: "The closed form: normal equation",
      level: "core",
      body: [
        { type: "p", text: "Because MSE is a convex quadratic in $w$, its minimum can be found *exactly*, in one shot, by setting the gradient to zero — no iteration required. This is the **normal equation**, and it is the first learning algorithm most people ever meet that has a clean closed-form solution." },
        { type: "p", text: "The full derivation lives in the Linear Algebra page (least-squares section); the short version is: expand $J(w) = (Xw - y)^\\top(Xw - y)$, differentiate using $\\nabla_w\\, w^\\top A w = 2Aw$ and $\\nabla_w\\, b^\\top w = b$, set to zero, and solve." },
        { type: "math", tex: String.raw`\nabla_w J = 2X^\top X w - 2X^\top y \overset{!}{=} 0 \;\Longrightarrow\; X^\top X\, w = X^\top y` },
        { type: "math", tex: String.raw`\boxed{\; w = (X^\top X)^{-1} X^\top y \;}` },
        { type: "p", text: "That is the entire training procedure: two matrix multiplies, one solve. The matrix $(X^\\top X)^{-1}X^\\top$ is the **Moore–Penrose pseudo-inverse** $X^{+}$, and geometrically $\\hat y = Xw$ is the **orthogonal projection** of the target vector $y$ onto the column space of $X$ — the closest point in 'everything the model can express' to the truth. Least squares *is* projection." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef fit_normal_equation(X, y):\n    \"\"\"Closed-form least squares. X already has a bias column of ones.\"\"\"\n    # Solve (X^T X) w = X^T y directly — never invert explicitly.\n    return np.linalg.solve(X.T @ X, X.T @ y)\n\n# Synthetic data: y = 3 + 2*x + noise\nrng = np.random.default_rng(0)\nx = rng.uniform(0, 10, size=200)\ny = 3 + 2 * x + rng.normal(0, 1.5, size=200)\n\nX = np.column_stack([np.ones_like(x), x])   # (200, 2): [1, x]\nw = fit_normal_equation(X, y)\nprint(w)                    # ~[3.0, 2.0]  -> [bias, slope]\n\n# The production one-liner (SVD under the hood, survives rank deficiency):\nw_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)\nprint(np.allclose(w, w_lstsq))   # True" },
        { type: "callout", variant: "gotcha", text: "**Solve, don't invert.** `np.linalg.solve(X.T@X, X.T@y)` is faster and far more numerically stable than `np.linalg.inv(X.T@X) @ X.T@y`. Better still, prefer `np.linalg.lstsq(X, y)`, which works directly on $X$ via SVD and never forms $X^\\top X$ at all — forming that product *squares the condition number* and throws away precision." },
        { type: "heading", text: "When the closed form fails" },
        { type: "p", text: "The normal equation needs $X^\\top X$ to be **invertible**, and it is not always. Two failure modes:" },
        { type: "list", items: [
          "**Singular $X^\\top X$ (rank deficiency).** If two features are perfectly collinear — say `size_in_m2` and `size_in_ft2`, or a redundant one-hot column — the columns of $X$ are linearly dependent, $X^\\top X$ is singular, and $(X^\\top X)^{-1}$ does not exist. The inverse-based method explodes; `lstsq`/pseudo-inverse survive by returning the minimum-norm solution.",
          "**Too many features ($n > m$).** More parameters than examples makes $X^\\top X$ (an $n\\times n$ matrix) automatically rank-deficient — infinitely many perfect fits exist. This is the regime where regularization (section 9) stops being optional.",
          "**Sheer size.** Forming and factorizing $X^\\top X$ costs $O(n^3)$ in the number of features. When $n$ reaches tens of thousands, the closed form becomes slower than iterative gradient descent, which is $O(mn)$ per step.",
        ]},
        { type: "callout", variant: "tip", text: "**Rule of thumb.** Few features and a modest dataset? The normal equation (via `lstsq`) is exact and instant — use it. Many features, huge or streaming data, or you want the same code to later fit a neural net? Use gradient descent. Ridge regression (section 9) fixes the singularity by adding $\\lambda I$, which makes $X^\\top X + \\lambda I$ invertible *always*." },
      ]
    },

    {
      id: "gradient-descent",
      title: "Gradient descent from scratch",
      level: "core",
      body: [
        { type: "p", text: "The closed form is a luxury unique to linear regression — the moment you add a non-linearity (logistic regression, any neural net), no such formula exists and you must **search** for the minimum iteratively. That search algorithm is **gradient descent**, the single most important optimization method in all of ML. Linear regression is the perfect place to learn it, because you can check its answer against the exact one." },
        { type: "p", text: "The idea: the gradient $\\nabla_w J$ points in the direction of *steepest increase* of the loss. So to *decrease* the loss, step in the opposite direction. Repeat until you stop moving." },
        { type: "math", tex: String.raw`w \;\leftarrow\; w - \eta\, \nabla_w J(w)` },
        { type: "p", text: "where $\\eta > 0$ is the **learning rate** (step size). We just need the gradient of MSE." },
        { type: "heading", text: "Deriving the gradient of MSE" },
        { type: "p", text: "Write the loss as $J(w) = \\tfrac{1}{m}\\|Xw - y\\|^2$ and let the residual be $r = Xw - y$. Differentiate: the chain rule gives $\\nabla_w \\|r\\|^2 = 2 X^\\top r$, so" },
        { type: "math", tex: String.raw`\nabla_w J(w) = \frac{2}{m}\, X^\top (Xw - y) = \frac{2}{m}\sum_{i=1}^{m}\big(\hat y^{(i)} - y^{(i)}\big)\,x^{(i)}` },
        { type: "p", text: "Read the right-hand side out loud: **each example contributes its own error times its own feature vector.** An example the model under-predicts pulls the weights up along its features; an example it over-predicts pushes them down. The average of those tugs is the gradient. This exact structure — error $\\times$ input — reappears as the delta rule in backpropagation." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef mse(X, y, w):\n    r = X @ w - y\n    return (r @ r) / len(y)\n\ndef gradient_descent(X, y, eta=0.01, n_iters=1000):\n    m, n = X.shape\n    w = np.zeros(n)                    # start at the origin\n    history = []\n    for t in range(n_iters):\n        y_hat = X @ w\n        grad = (2 / m) * X.T @ (y_hat - y)   # <- the derived gradient\n        w -= eta * grad                       # <- the update step\n        history.append(mse(X, y, w))\n    return w, history\n\n# Data: y = 3 + 2*x + noise, with x standardized so GD converges fast.\nrng = np.random.default_rng(0)\nx = rng.uniform(0, 10, size=200)\ny = 3 + 2 * x + rng.normal(0, 1.5, size=200)\nx_z = (x - x.mean()) / x.std()                # feature scaling — important!\nX = np.column_stack([np.ones_like(x_z), x_z])\n\nw, hist = gradient_descent(X, y, eta=0.1, n_iters=500)\nprint(w)                    # weights in *standardized* space\nprint(f\"final MSE: {hist[-1]:.3f}\")\n\n# Sanity check against the closed form:\nw_exact, *_ = np.linalg.lstsq(X, y, rcond=None)\nprint(np.allclose(w, w_exact, atol=1e-3))   # True — GD found the same minimum" },
        { type: "heading", text: "Learning rate: the one knob that makes or breaks it" },
        { type: "p", text: "$\\eta$ controls step size, and it is unforgiving on both ends:" },
        { type: "table",
          headers: ["Learning rate", "Behaviour", "Symptom"],
          rows: [
            ["Too small", "creeps toward the minimum", "loss falls, but painfully slowly"],
            ["Just right", "steady, fast descent", "loss drops smoothly to a floor"],
            ["Too large", "overshoots the minimum", "loss oscillates or **diverges to NaN**"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Always plot the loss curve** (the `history` above) vs. iteration. A healthy run drops fast then flattens. A curve that rises, oscillates, or goes NaN means $\\eta$ is too big — cut it by 10x. A curve still steeply falling at the end means more iterations (or a bigger $\\eta$) will help. This one plot diagnoses 90% of optimization problems." },
        { type: "heading", text: "Feature scaling: why the code standardized $x$" },
        { type: "p", text: "If one feature ranges over $[0, 1]$ and another over $[0, 100000]$, the loss surface is a long, thin ravine instead of a round bowl — its Hessian has a huge **condition number** $\\kappa = \\lambda_{\\max}/\\lambda_{\\min}$ (Linear Algebra page). Gradient descent then zig-zags across the ravine and crawls along it, needing far more steps. **Standardizing** every feature to zero mean and unit variance rounds the bowl out and can turn thousands of iterations into dozens." },
        { type: "code", lang: "py", code: "# Standardize on the TRAIN set, then apply the same transform to test/new data.\nmu, sigma = X_train.mean(axis=0), X_train.std(axis=0)\nX_train_z = (X_train - mu) / sigma\nX_test_z  = (X_test  - mu) / sigma      # <- reuse train stats, never refit!" },
        { type: "callout", variant: "gotcha", text: "**Never fit the scaler on your test data.** Compute mean/std on the training set and reuse them everywhere. Recomputing on the test set (or on the full dataset before splitting) leaks information and inflates your reported accuracy — a classic, silent data-leakage bug. In scikit-learn, `Pipeline` + `StandardScaler` enforces this automatically." },
        { type: "heading", text: "Convergence: when to stop" },
        { type: "p", text: "Because MSE is convex, gradient descent with a small-enough $\\eta$ is *guaranteed* to reach the global minimum — no local minima to trap it. In practice you stop when the loss stops improving: fix a number of iterations, or halt when $|J_{t} - J_{t-1}|$ (or the gradient norm) drops below a tolerance." },
      ]
    },

    {
      id: "sgd",
      title: "Stochastic & mini-batch gradient descent",
      level: "core",
      body: [
        { type: "p", text: "The gradient above sums over **all $m$ examples** every single step — that is **batch** gradient descent. Exact, but on a dataset of millions of rows one step means a full pass over everything before the weights move even once. That does not scale. The fix is one of the most consequential ideas in modern ML: **estimate the gradient from a small sample instead of the whole dataset.**" },
        { type: "heading", text: "The three flavours" },
        { type: "table",
          headers: ["Variant", "Examples per step", "Per-step gradient"],
          rows: [
            ["Batch GD", "all $m$", "exact, low variance"],
            ["Stochastic GD (SGD)", "1 (random)", "noisy, high variance"],
            ["Mini-batch GD", "a batch of $B$ (e.g. 32)", "the practical sweet spot"],
          ]
        },
        { type: "p", text: "**Stochastic** gradient descent uses a *single random example* per update. Its gradient is a wild, noisy estimate of the true one — but it is an **unbiased** estimate (correct on average), and you get $m$ updates per epoch instead of one. **Mini-batch** GD averages over a small batch of $B$ examples, keeping most of the speed while damping the noise. Mini-batch is what everyone actually uses — it is the workhorse that trains every neural network and LLM on Earth." },
        { type: "math", tex: String.raw`w \leftarrow w - \eta \cdot \frac{2}{B}\sum_{i \in \mathcal{B}}\big(\hat y^{(i)} - y^{(i)}\big)\,x^{(i)}, \qquad \mathcal{B} = \text{a random mini-batch}` },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef minibatch_sgd(X, y, eta=0.05, batch_size=32, epochs=50, seed=0):\n    rng = np.random.default_rng(seed)\n    m, n = X.shape\n    w = np.zeros(n)\n    for epoch in range(epochs):\n        idx = rng.permutation(m)          # reshuffle each epoch\n        for start in range(0, m, batch_size):\n            batch = idx[start:start + batch_size]\n            Xb, yb = X[batch], y[batch]\n            grad = (2 / len(batch)) * Xb.T @ (Xb @ w - yb)\n            w -= eta * grad\n    return w\n\n# (X, y from the previous section, features standardized)\nw = minibatch_sgd(X, y, eta=0.05, batch_size=32, epochs=50)\nprint(w)   # converges to the same minimum, having touched the full data 50x" },
        { type: "callout", variant: "note", text: "**One epoch = one full pass over the data.** With a batch size of 32 and 6400 examples, that is 200 weight updates per epoch, versus 1 update per epoch for batch GD. The gradient noise is not purely a cost, either: for non-convex problems (neural nets) it actively helps the optimizer *escape saddle points and sharp minima*, which is a large part of why SGD generalizes so well." },
        { type: "heading", text: "The variance / speed tradeoff" },
        { type: "p", text: "Batch size $B$ trades two things off. Small $B$: cheap, frequent, noisy updates — fast wall-clock progress but a jittery path that never quite settles (it bounces around the minimum). Large $B$: expensive, rare, accurate updates — a smooth path but slow to get going, and diminishing returns because the gradient's noise falls only as $1/\\sqrt{B}$. The standard resolution is a **decaying learning rate**: start large to make progress, shrink over time so the noise averages out and you settle into the minimum." },
        { type: "callout", variant: "tip", text: "**Batch sizes are powers of two (16, 32, 64, 256) for hardware reasons**, and 32–256 is a sane default. Because SGD's noise never fully vanishes with a fixed $\\eta$, pair it with a learning-rate schedule (e.g. `eta_t = eta_0 / (1 + decay*t)`). Every optimizer you meet later — Momentum, RMSProp, Adam — is a smarter way to use these same noisy mini-batch gradients." },
      ]
    },

    {
      id: "basis",
      title: "Polynomial & basis-function regression",
      level: "core",
      body: [
        { type: "p", text: "'Linear' regression sounds like it can only fit straight lines. It cannot — and the loophole is one of the most useful tricks in classical ML. The model must be **linear in the parameters $w$**, but nothing stops the *features* from being arbitrary non-linear transforms of the inputs. Feed it $x, x^2, x^3$ and it fits a cubic; the fit is a curve, yet the fitting problem is still plain linear least squares." },
        { type: "math", tex: String.raw`\hat y = w_0 + w_1 x + w_2 x^2 + \dots + w_d x^d = \sum_{j=0}^{d} w_j\,\phi_j(x), \quad \phi_j(x) = x^j` },
        { type: "p", text: "More generally, replace each input with a vector of **basis functions** $\\phi(x)$ — polynomials, but also sines/cosines (Fourier features), Gaussian bumps (radial basis functions), splines, or feature *interactions* like $x_1 x_2$. Build the transformed design matrix $\\Phi$ whose rows are $\\phi(x^{(i)})$, and every formula from this page applies verbatim with $\\Phi$ in place of $X$:" },
        { type: "math", tex: String.raw`w = (\Phi^\top \Phi)^{-1}\Phi^\top y, \qquad \Phi_{ij} = \phi_j(x^{(i)})` },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef polynomial_features(x, degree):\n    \"\"\"Map scalar x -> [1, x, x^2, ..., x^degree].\"\"\"\n    return np.vstack([x ** j for j in range(degree + 1)]).T\n\nrng = np.random.default_rng(1)\nx = np.sort(rng.uniform(-3, 3, size=40))\ny = np.sin(x) + rng.normal(0, 0.2, size=40)     # a genuinely non-linear target\n\nfor degree in (1, 3, 9):\n    Phi = polynomial_features(x, degree)\n    w, *_ = np.linalg.lstsq(Phi, y, rcond=None)\n    resid = Phi @ w - y\n    print(f\"degree {degree}: train MSE = {(resid @ resid) / len(y):.4f}\")\n# degree 1: underfits (too stiff).  degree 3: good.  degree 9: overfits." },
        { type: "heading", text: "The bias–variance knob" },
        { type: "p", text: "The polynomial degree $d$ is a dial for **model complexity**, and it exposes the central tradeoff of all supervised learning. Turn it and you trade one kind of error for another:" },
        { type: "table",
          headers: ["Degree", "Behaviour", "Bias", "Variance"],
          rows: [
            ["too low (e.g. 1)", "**underfits** — too rigid to follow the signal", "high", "low"],
            ["just right", "captures the true shape", "low", "low"],
            ["too high (e.g. 15)", "**overfits** — chases the noise, wiggles wildly", "low", "high"],
          ]
        },
        { type: "p", text: "**Bias** is error from wrong assumptions — a straight line trying to fit a curve is biased no matter how much data you give it. **Variance** is sensitivity to the particular training sample — a degree-15 polynomial contorts itself through every noisy point and swings drastically if you resample the data. Total expected error decomposes as $\\text{bias}^2 + \\text{variance} + \\text{irreducible noise}$, and the two terms pull in opposite directions." },
        { type: "math", tex: String.raw`\mathbb{E}\big[(y - \hat y)^2\big] = \underbrace{\text{Bias}[\hat y]^2}_{\text{too simple}} + \underbrace{\text{Var}[\hat y]}_{\text{too complex}} + \underbrace{\sigma^2}_{\text{irreducible}}` },
        { type: "callout", variant: "gotcha", text: "**A low *training* error is not success — it can be the symptom of failure.** A degree-9 polynomial can drive training MSE to nearly zero while predicting garbage between the points. The only honest measure of a model is its error on **held-out data it never saw during fitting**. Split your data, or use cross-validation, before you trust any number." },
        { type: "callout", variant: "tip", text: "The right complexity is chosen with **validation data**, not guessed. And you do not have to reduce the degree to fight overfitting — you can keep a flexible model and **regularize** it (section 9), penalizing large weights so the curve stays smooth. Complexity control and regularization are the same fight from two directions." },
      ]
    },

    {
      id: "diagnostics",
      title: "Assumptions & diagnostics",
      level: "core",
      body: [
        { type: "p", text: "Linear regression will happily fit *any* data and hand you weights — including when it is completely inappropriate. A responsible practitioner checks the model's **assumptions** and reads its **diagnostics** before trusting a single coefficient. These are the assumptions the least-squares / Gaussian-likelihood derivation quietly relied on:" },
        { type: "table",
          headers: ["Assumption", "Meaning", "How to check"],
          rows: [
            ["Linearity", "$\\mathbb{E}[y\\mid x]$ really is linear in the features", "residuals vs. fitted — look for curvature"],
            ["Independence", "examples' errors don't influence each other", "domain knowledge; watch time series"],
            ["Homoscedasticity", "noise variance $\\sigma^2$ is constant", "residual plot — look for a funnel"],
            ["Normality of errors", "residuals are roughly Gaussian", "Q–Q plot of residuals"],
            ["No multicollinearity", "features aren't redundant/collinear", "VIF, or a feature correlation matrix"],
          ]
        },
        { type: "heading", text: "The residual plot: your single most useful diagnostic" },
        { type: "p", text: "Plot the **residuals** $r^{(i)} = y^{(i)} - \\hat y^{(i)}$ against the fitted values $\\hat y^{(i)}$. If the model is appropriate, this should look like a **structureless, horizontal blur** centered on zero. Any *pattern* is the data telling you an assumption is violated:" },
        { type: "list", items: [
          "**A curve or U-shape** → the true relationship is non-linear; you need polynomial/basis features (section 6).",
          "**A funnel** (residuals fan out as $\\hat y$ grows) → heteroscedasticity; the constant-variance assumption is broken. Consider a log-transform of $y$, or weighted least squares.",
          "**Distinct clusters or a trend over row order** → a missing feature or correlated errors (common in time series).",
        ]},
        { type: "heading", text: "$R^2$ — the coefficient of determination" },
        { type: "p", text: "$R^2$ summarizes fit in one number: the fraction of the target's variance the model explains, relative to the dumbest possible baseline (always predicting the mean $\\bar y$)." },
        { type: "math", tex: String.raw`R^2 = 1 - \frac{\sum_i (y^{(i)} - \hat y^{(i)})^2}{\sum_i (y^{(i)} - \bar y)^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}` },
        { type: "p", text: "$R^2 = 1$ is a perfect fit; $R^2 = 0$ means you did no better than predicting the mean; **$R^2$ can go negative** on test data when the model is worse than the mean. It is intuitive but has a trap: adding *any* feature — even pure noise — never decreases training $R^2$, so it silently rewards overfitting. **Adjusted $R^2$** penalizes extra features to counter this." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef r2_score(y, y_hat):\n    ss_res = np.sum((y - y_hat) ** 2)\n    ss_tot = np.sum((y - y.mean()) ** 2)\n    return 1 - ss_res / ss_tot\n\ny_hat = X @ w\nprint(f\"R^2 = {r2_score(y, y_hat):.3f}\")   # e.g. 0.94\n\n# Residual diagnostic: this should look like structureless noise around 0.\nresiduals = y - y_hat\nprint(residuals.mean().round(6))            # ~0 by construction of least squares" },
        { type: "heading", text: "Multicollinearity" },
        { type: "p", text: "When features are strongly correlated (e.g. height in cm *and* in inches, or age *and* years-of-experience), the model cannot tell which one deserves credit. Mathematically $X^\\top X$ becomes near-singular; practically, the **coefficients become wildly unstable** — huge in magnitude, flipping sign on a tiny data change — even while predictions stay fine. The standard detector is the **Variance Inflation Factor (VIF)**; a VIF above ~5–10 flags a problem feature. Fixes: drop one of the pair, combine them, use PCA, or apply ridge regularization (which is *designed* to tame this)." },
        { type: "callout", variant: "gotcha", text: "**Predictions can be excellent while individual coefficients are meaningless.** Under multicollinearity, $\\hat y$ is fine but you must not interpret any single weight as 'the effect of that feature' — the model has arbitrarily split the shared credit between the correlated inputs. This is the #1 reason 'the sign of my coefficient is backwards!' bug reports happen." },
      ]
    },

    {
      id: "sklearn",
      title: "The full worked example in scikit-learn",
      level: "core",
      body: [
        { type: "p", text: "Now the professional workflow. We fit the California Housing dataset three ways — scikit-learn's exact `LinearRegression` (which uses `lstsq` internally), its iterative `SGDRegressor`, and our own from-scratch gradient descent — and confirm all three agree. Note the **`Pipeline`**: it bundles the scaler and the model so the scaler is fit on the training fold only, killing the data-leakage bug from section 4 by construction." },
        { type: "code", lang: "py", code: "import numpy as np\nfrom sklearn.datasets import fetch_california_housing\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LinearRegression, SGDRegressor\nfrom sklearn.metrics import r2_score, mean_squared_error\n\nX, y = fetch_california_housing(return_X_y=True)   # 20640 rows, 8 features\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)\n\n# --- Model 1: closed-form least squares (normal equation via SVD) ---\nols = make_pipeline(StandardScaler(), LinearRegression())\nols.fit(X_tr, y_tr)\npred = ols.predict(X_te)\nprint(f\"OLS   : R^2={r2_score(y_te, pred):.3f}  \"\n      f\"RMSE={mean_squared_error(y_te, pred) ** 0.5:.3f}\")\n\n# --- Model 2: iterative SGD (the same optimizer that scales to huge data) ---\nsgd = make_pipeline(\n    StandardScaler(),\n    SGDRegressor(loss=\"squared_error\", penalty=None,\n                 learning_rate=\"invscaling\", eta0=0.01,\n                 max_iter=1000, random_state=0),\n)\nsgd.fit(X_tr, y_tr)\npred_sgd = sgd.predict(X_te)\nprint(f\"SGD   : R^2={r2_score(y_te, pred_sgd):.3f}  \"\n      f\"RMSE={mean_squared_error(y_te, pred_sgd) ** 0.5:.3f}\")" },
        { type: "code", lang: "py", code: "# --- Model 3: our from-scratch batch gradient descent, same problem ---\nscaler_mu = X_tr.mean(axis=0)\nscaler_sd = X_tr.std(axis=0)\nXtr = (X_tr - scaler_mu) / scaler_sd\nXte = (X_te - scaler_mu) / scaler_sd\nXtr = np.column_stack([np.ones(len(Xtr)), Xtr])    # bias column\nXte = np.column_stack([np.ones(len(Xte)), Xte])\n\nw = np.zeros(Xtr.shape[1])\nfor _ in range(2000):\n    grad = (2 / len(Xtr)) * Xtr.T @ (Xtr @ w - y_tr)\n    w -= 0.1 * grad\n\npred_scratch = Xte @ w\nfrom sklearn.metrics import r2_score\nprint(f\"scratch: R^2={r2_score(y_te, pred_scratch):.3f}\")\n\n# All three land on ~R^2=0.58 — the from-scratch code is doing exactly what\n# LinearRegression does, just with our own loop instead of LAPACK.\nprint(\"sklearn coefs :\", ols.named_steps['linearregression'].coef_[:3].round(3))\nprint(\"our coefs     :\", w[1:4].round(3))   # matches (both in standardized space)" },
        { type: "callout", variant: "note", text: "**`LinearRegression` vs. `SGDRegressor` — when to reach for each.** `LinearRegression` computes the exact closed-form solution and is the default for small/medium data. `SGDRegressor` optimizes iteratively, supports mini-batches via `partial_fit`, and is what you use for datasets too large for memory or for online/streaming learning — the *same* algorithm from section 5, production-hardened. Their answers converge; the difference is *how* they get there." },
        { type: "callout", variant: "good", text: "The punchline of the whole page: **the from-scratch NumPy loop and scikit-learn's battle-tested code produce the same coefficients and the same $R^2$.** The library is faster, safer, and full of conveniences, but there is no hidden magic — you now understand every line of what it does. That understanding is what lets you debug it when it misbehaves." },
      ]
    },

    {
      id: "regularization",
      title: "Regularization teaser",
      level: "core",
      body: [
        { type: "p", text: "Two problems keep surfacing above: the normal equation dies when $X^\\top X$ is singular (collinear features, or $n > m$), and flexible models overfit by growing enormous, wildly-oscillating weights. **Regularization** solves both with one idea — add a penalty on the *size* of the weights to the loss, so the optimizer is discouraged from using large coefficients unless the data really demands them." },
        { type: "math", tex: String.raw`J_{\text{ridge}}(w) = \underbrace{\|Xw - y\|_2^2}_{\text{fit the data}} + \underbrace{\lambda\,\|w\|_2^2}_{\text{keep }w\text{ small}}` },
        { type: "p", text: "This is **ridge regression** ($L_2$ penalty), and it has a beautiful side effect: its closed form is $w = (X^\\top X + \\lambda I)^{-1}X^\\top y$. Adding $\\lambda I$ to $X^\\top X$ makes the matrix **invertible for any $\\lambda > 0$** — the singularity problem simply vanishes. The hyperparameter $\\lambda$ dials the bias–variance tradeoff directly: $\\lambda = 0$ is ordinary least squares; large $\\lambda$ shrinks weights toward zero, trading a little bias for a large drop in variance." },
        { type: "table",
          headers: ["Method", "Penalty", "Signature effect"],
          rows: [
            ["Ridge ($L_2$)", "$\\lambda\\|w\\|_2^2$", "shrinks weights smoothly; fixes collinearity"],
            ["Lasso ($L_1$)", "$\\lambda\\|w\\|_1$", "drives weights to *exactly* zero — feature selection"],
            ["Elastic Net", "mix of both", "grouped selection + shrinkage"],
          ]
        },
        { type: "callout", variant: "tip", text: "**This is a teaser, not the full story.** *Why* the $L_1$ penalty produces exactly-zero weights (its geometry), how to choose $\\lambda$ by cross-validation, the Bayesian reading of ridge as a Gaussian prior on $w$ (MAP estimation, echoing this page's MLE argument), and the full derivations get their own **Regularization** page. For now: whenever a linear model overfits or its coefficients look unstable, ridge is your first move." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Reading these derivations builds recognition; typing the code builds fluency. Do at least two of these end-to-end — the from-scratch ones especially, because matching your own NumPy against scikit-learn is the moment the algorithm stops being a black box." },
        { type: "list", ordered: true, items: [
          "**Least squares three ways.** Generate noisy points on a known line. Recover the weights via the normal equation, `np.linalg.lstsq`, and your own gradient descent, and confirm all three agree. Then add a duplicate (collinear) feature on purpose and watch the normal equation blow up while `lstsq` survives — you will *feel* section 3's failure mode.",
          "**Gradient descent visualizer.** Fit a simple model while logging the loss and the weight vector every iteration. Plot the loss curve, then overlay the descent path on a contour plot of the loss surface. Re-run with $\\eta$ too small, just right, and too large to see creeping, converging, and diverging with your own eyes.",
          "**Bias–variance sweep.** Fit polynomials of degree 1…15 to a noisy sine wave. Plot *training* and *validation* MSE vs. degree on the same axes and watch the classic U-shaped validation curve appear. Mark the degree that minimizes validation error — that is model selection in one picture.",
          "**Diagnose a real dataset.** Take California Housing (or any tabular set), fit `LinearRegression`, then generate the residual-vs-fitted plot, a Q–Q plot, and per-feature VIFs. Interpret what each says about the assumptions, and try a log-transform of the target if you see a funnel.",
          "**Mini-batch SGD from scratch.** Implement mini-batch SGD with a decaying learning rate and reshuffling each epoch. Sweep batch sizes (1, 32, 256, full) and plot the loss curves together to see the variance/speed tradeoff — the single-example run is jittery, the full-batch run is smooth but slow.",
          "**Ridge vs. OLS under collinearity.** Build a dataset with two strongly-correlated features. Fit ordinary least squares and ridge across a range of $\\lambda$, and plot each coefficient's path as $\\lambda$ grows. Watch ridge stabilize the wild OLS coefficients — a preview of the Regularization page.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "Linear regression is the best-documented algorithm in all of statistics and ML. These are the resources worth your time, in roughly the order I would read them:" },
        { type: "link", url: "https://www.statlearning.com/", text: "James, Witten, Hastie, Tibshirani — An Introduction to Statistical Learning (ISLR), Chapter 3. Free PDF; the clearest treatment of linear regression, assumptions, and diagnostics anywhere." },
        { type: "link", url: "https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/", text: "Aurélien Géron — Hands-On Machine Learning (3rd ed.), Chapter 4. The normal equation, gradient-descent variants, and regularization with runnable scikit-learn code." },
        { type: "link", url: "https://www.coursera.org/learn/machine-learning", text: "Andrew Ng — Machine Learning Specialization (Coursera). The canonical from-scratch derivation of MSE and gradient descent; where most practitioners first learn this." },
        { type: "link", url: "https://scikit-learn.org/stable/modules/linear_model.html", text: "scikit-learn — Linear Models user guide. The authoritative reference for LinearRegression, SGDRegressor, Ridge, Lasso, and their exact parameters." },
        { type: "link", url: "https://hastie.su.domains/ElemStatLearn/", text: "Hastie, Tibshirani, Friedman — The Elements of Statistical Learning, Chapter 3. The graduate-level version: bias–variance, subset selection, shrinkage, in full rigour. Free PDF." },
        { type: "link", url: "https://www.deeplearningbook.org/contents/ml.html", text: "Goodfellow, Bengio, Courville — Deep Learning, Chapter 5. Frames linear regression as maximum likelihood and as the entry point to all of ML — the exact two-derivations view of this page." },
      ]
    },
  ],

  packages: [
    { name: "numpy", why: "arrays, `@` matmul, and `np.linalg.lstsq` — build regression from scratch" },
    { name: "scikit-learn", why: "`LinearRegression`, `SGDRegressor`, `Ridge`, `Lasso` — the production API" },
    { name: "sklearn.preprocessing.StandardScaler", why: "feature scaling done right (fit on train only)" },
    { name: "sklearn.pipeline", why: "bundle scaler + model so no test-set leakage is possible" },
    { name: "sklearn.model_selection", why: "`train_test_split`, `cross_val_score` — honest evaluation" },
    { name: "statsmodels", why: "regression with p-values, confidence intervals, and full diagnostic tables" },
    { name: "matplotlib", why: "the loss curve and residual plot — your core debugging tools" },
  ],

  gotchas: [
    "MSE is **maximum likelihood under Gaussian noise**, not an arbitrary choice — change the noise model and you change the loss.",
    "Prefer `np.linalg.lstsq(X, y)` over the explicit normal equation: forming $X^\\top X$ squares the condition number and loses precision.",
    "Collinear features make $X^\\top X$ singular, so $(X^\\top X)^{-1}$ blows up. `lstsq`/SVD and ridge ($+\\lambda I$) survive it.",
    "**Always scale features** before gradient descent (and before ridge/lasso). Unscaled features make an ill-conditioned bowl and GD crawls.",
    "Fit the scaler on the **training set only**, then reuse those stats — recomputing on test data leaks information and inflates your score.",
    "A learning rate too large makes the loss **oscillate or diverge to NaN**; always plot the loss curve to diagnose it.",
    "Low *training* error can mean overfitting, not success. Judge every model on **held-out** data; $R^2$ can be negative on test data.",
    "Under multicollinearity, predictions stay fine but individual coefficients become unstable and un-interpretable — don't over-read a single weight.",
  ],

  flashcards: [
    { q: "Write the linear-regression model in vector form.", a: "$\\hat y = w^\\top x + b$; with a bias column of ones folded in, $\\hat y = Xw$ for the whole dataset at once." },
    { q: "Why is squared error the 'right' loss for regression?", a: "It is the **maximum-likelihood** estimator assuming i.i.d. Gaussian noise $\\varepsilon \\sim \\mathcal{N}(0,\\sigma^2)$: maximizing that likelihood equals minimizing $\\sum_i(y^{(i)}-w^\\top x^{(i)})^2$." },
    { q: "State the normal equation and when it fails.", a: "$w = (X^\\top X)^{-1}X^\\top y$. It fails when $X^\\top X$ is singular — collinear features or $n>m$ — where you need `lstsq`/SVD or ridge." },
    { q: "What is the gradient of MSE with respect to $w$?", a: "$\\nabla_w J = \\frac{2}{m}X^\\top(Xw-y) = \\frac{2}{m}\\sum_i(\\hat y^{(i)}-y^{(i)})x^{(i)}$ — each example's error times its features, averaged." },
    { q: "Batch vs. stochastic vs. mini-batch GD?", a: "Batch uses all $m$ examples per step (exact, slow); SGD uses 1 (noisy, fast); mini-batch uses ~32 (the practical sweet spot that trains every neural net)." },
    { q: "How does linear regression fit a curve?", a: "Transform the inputs with **basis functions** (e.g. $x, x^2, x^3$). The model stays linear in the parameters $w$, so all the least-squares math still applies to $\\Phi$." },
    { q: "What does model complexity trade off?", a: "**Bias vs. variance.** Too simple = high bias (underfit); too complex = high variance (overfit). Expected error $= \\text{bias}^2 + \\text{variance} + \\sigma^2$." },
    { q: "What does a residual-vs-fitted plot tell you?", a: "It should be a structureless horizontal blur. A curve means non-linearity; a funnel means heteroscedasticity (non-constant variance)." },
    { q: "Define $R^2$.", a: "$R^2 = 1 - \\text{SS}_{\\text{res}}/\\text{SS}_{\\text{tot}}$ — the fraction of target variance explained vs. predicting the mean. It can go negative on test data." },
    { q: "What is ridge regression and why does it fix singularity?", a: "It adds an $L_2$ penalty: $w=(X^\\top X + \\lambda I)^{-1}X^\\top y$. The $+\\lambda I$ makes the matrix invertible for any $\\lambda>0$ and shrinks weights, reducing variance." },
    { q: "Why standardize features before gradient descent?", a: "Unequal feature scales make the loss bowl long and thin (high condition number), so GD zig-zags. Standardizing rounds the bowl and speeds convergence dramatically." },
  ],

  cheatsheet: [
    { label: "Add bias column", code: "X = np.column_stack([np.ones(len(x)), x])" },
    { label: "Normal equation (solve)", code: "w = np.linalg.solve(X.T @ X, X.T @ y)" },
    { label: "Least squares (preferred)", code: "w, *_ = np.linalg.lstsq(X, y, rcond=None)" },
    { label: "MSE gradient", code: "grad = (2/m) * X.T @ (X @ w - y)" },
    { label: "GD update", code: "w -= eta * grad" },
    { label: "Standardize (train stats)", code: "Xz = (X - X_tr.mean(0)) / X_tr.std(0)" },
    { label: "sklearn OLS", code: "LinearRegression().fit(X, y)" },
    { label: "sklearn SGD", code: "SGDRegressor(max_iter=1000).fit(X, y)" },
    { label: "Ridge", code: "Ridge(alpha=1.0).fit(X, y)" },
    { label: "Lasso (feature selection)", code: "Lasso(alpha=0.1).fit(X, y)" },
    { label: "No-leak pipeline", code: "make_pipeline(StandardScaler(), LinearRegression())" },
    { label: "R^2 score", code: "from sklearn.metrics import r2_score" },
    { label: "Train/test split", code: "train_test_split(X, y, test_size=0.2)" },
    { label: "Polynomial features", code: "PolynomialFeatures(degree=3).fit_transform(X)" },
  ],
});
