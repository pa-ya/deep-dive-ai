(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "ml-regularization",
  name: "Regularization",
  language: "Classical ML",
  group: "Classical Machine Learning",
  navLabel: "Regularization",
  tagline: "How to fight overfitting — Ridge, Lasso, Elastic Net, and the unifying Bayesian idea behind every penalty, derived and coded from scratch.",
  color: "#14B8A6",
  readMinutes: 44,
  sections: [
    {
      id: "overfitting",
      title: "The problem: overfitting and the bias–variance tradeoff",
      level: "core",
      body: [
        { type: "p", text: "A model **overfits** when it memorizes the training data — including its noise — instead of learning the underlying pattern. It scores beautifully on data it has already seen and falls apart on anything new. Regularization is the collection of techniques for deliberately *handicapping* a model so it generalizes. It is, in practice, one of the highest-leverage skills in applied ML: the difference between a model that works in a notebook and one that works in production is very often a single well-tuned penalty." },
        { type: "p", text: "The classic symptom is a large gap between training and test error, especially when the model is flexible (many features, high-degree polynomial, deep tree). Fit a degree-15 polynomial to 12 noisy points and it will thread every point exactly — training error zero, test error catastrophic." },
        { type: "callout", variant: "note", text: "**Where this fits.** Overfitting and the bias–variance decomposition are introduced in **ML Foundations**; here we take them as given and focus on the *cure*. The optimization machinery (gradients, the normal equation) comes from **Linear Algebra** and **Calculus & Optimization**." },
        { type: "heading", text: "Bias–variance: the thing regularization controls" },
        { type: "p", text: "Expected test error of a model at a point decomposes into three pieces. For squared-error loss with true function $f$ and noise variance $\\sigma^2$:" },
        { type: "math", tex: String.raw`\mathbb{E}\big[(y - \hat f(x))^2\big] \;=\; \underbrace{\big(\mathbb{E}[\hat f(x)] - f(x)\big)^2}_{\text{bias}^2} \;+\; \underbrace{\operatorname{Var}\big(\hat f(x)\big)}_{\text{variance}} \;+\; \underbrace{\sigma^2}_{\text{irreducible}}` },
        { type: "p", text: "**Bias** is error from wrong assumptions — too simple a model systematically misses the pattern (underfitting). **Variance** is sensitivity to the particular training sample — too flexible a model swings wildly if you resample the data (overfitting). The irreducible term is noise you can never beat. You cannot drive both bias and variance to zero; you *trade* one for the other." },
        { type: "table",
          headers: ["Regime", "Bias", "Variance", "Symptom"],
          rows: [
            ["Underfit (too simple)", "high", "low", "train error high, test $\\approx$ train"],
            ["Sweet spot", "low-ish", "low-ish", "train and test both low, small gap"],
            ["Overfit (too flexible)", "low", "high", "train error tiny, test error large"],
          ]
        },
        { type: "p", text: "Regularization works by **adding a little bias to buy a large reduction in variance**. It nudges the model toward simpler, smoother solutions — smaller weights, fewer active features — so the fit stops chasing noise. The knob that sets *how much* is the regularization strength $\\lambda$, and choosing it well is the whole game." },
        { type: "callout", variant: "tip", text: "**The mental model for everything below:** we take a loss $\\mathcal{L}(w)$ we already wanted to minimize and bolt on a **penalty** $\\lambda\\, R(w)$ that grows with model complexity. We then minimize $\\mathcal{L}(w) + \\lambda R(w)$. Different choices of $R$ — the squared norm, the absolute norm, a mix — give Ridge, Lasso, and Elastic Net. Same recipe, different penalty." },
      ]
    },

    {
      id: "ridge",
      title: "Ridge regression (L2): shrink the weights",
      level: "core",
      body: [
        { type: "p", text: "**Ridge regression** adds the squared $L_2$ norm of the weights to the least-squares objective. Big weights are expensive, so the optimizer keeps them small unless the data really insists otherwise." },
        { type: "math", tex: String.raw`\min_{w}\; \underbrace{\|Xw - y\|_2^2}_{\text{fit}} \;+\; \underbrace{\lambda\,\|w\|_2^2}_{\text{penalty}}, \qquad \|w\|_2^2 = \sum_{j} w_j^2` },
        { type: "p", text: "$\\lambda \\ge 0$ controls the strength. At $\\lambda = 0$ you recover ordinary least squares; as $\\lambda \\to \\infty$ every weight is crushed toward zero. Everything interesting happens in between." },
        { type: "heading", text: "Deriving the modified normal equation" },
        { type: "p", text: "This has a clean closed form. Write the objective $J(w)$, expand, differentiate, and set the gradient to zero — exactly as for ordinary least squares, but with one extra term. (The gradient rules $\\nabla_w\\, w^\\top A w = 2Aw$ for symmetric $A$ and $\\nabla_w\\, \\lambda w^\\top w = 2\\lambda w$ come from **Calculus & Optimization**.)" },
        { type: "math", tex: String.raw`J(w) = (Xw - y)^\top (Xw - y) + \lambda\, w^\top w` },
        { type: "math", tex: String.raw`\nabla_w J = 2X^\top(Xw - y) + 2\lambda w \;\overset{!}{=}\; 0` },
        { type: "math", tex: String.raw`X^\top X\, w + \lambda w = X^\top y \;\;\Longrightarrow\;\; (X^\top X + \lambda I)\,w = X^\top y` },
        { type: "math", tex: String.raw`\boxed{\; w_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y \;}` },
        { type: "p", text: "Compare to the ordinary normal equation $w = (X^\\top X)^{-1}X^\\top y$: the *only* change is $+\\lambda I$ on the diagonal. That tiny change does two remarkable things." },
        { type: "heading", text: "Why it fixes singularity" },
        { type: "p", text: "$X^\\top X$ is positive *semi*-definite, so it can be singular — it *is* singular whenever features are collinear or you have more features than samples ($n > m$). Then plain least squares has no unique solution and the inverse blows up. Adding $\\lambda I$ lifts every eigenvalue by $\\lambda$: if $X^\\top X$ has eigenvalues $\\mu_i \\ge 0$, then $X^\\top X + \\lambda I$ has eigenvalues $\\mu_i + \\lambda \\ge \\lambda > 0$. The matrix becomes **positive definite and always invertible**. Ridge was in fact invented for exactly this — Hoerl and Kennard's 1970 paper introduced it as a fix for ill-conditioned regression, decades before 'overfitting' was ML vocabulary." },
        { type: "callout", variant: "note", text: "**This is why it is called 'ridge.'** You are adding a small ridge of value $\\lambda$ down the diagonal of $X^\\top X$. The name is literally geometric." },
        { type: "heading", text: "Why it shrinks weights (the SVD view)" },
        { type: "p", text: "Substitute the SVD $X = U\\Sigma V^\\top$ into the closed form. Ordinary least squares divides each singular direction's contribution by $\\sigma_i$; ridge divides by $\\sigma_i$ but then multiplies by a **shrinkage factor**:" },
        { type: "math", tex: String.raw`w_{\text{ridge}} = \sum_{i} \underbrace{\frac{\sigma_i^2}{\sigma_i^2 + \lambda}}_{\text{shrinkage} \,\in\, (0,1)} \cdot \frac{u_i^\top y}{\sigma_i}\, v_i` },
        { type: "p", text: "The factor $\\sigma_i^2/(\\sigma_i^2+\\lambda)$ is near $1$ for strong directions (large $\\sigma_i$) and near $0$ for weak, noise-dominated directions (small $\\sigma_i$). Ridge leaves the informative directions almost untouched and aggressively damps the flimsy ones — which is precisely the high-variance noise you wanted gone. It shrinks weights *smoothly* toward zero but, crucially, **never to exactly zero**. Every feature stays in the model, just quieter." },
        { type: "callout", variant: "gotcha", text: "**Never penalize the intercept.** The bias term $w_0$ sets the overall level of $y$; shrinking it toward zero just biases predictions toward the origin for no benefit. In the matrix form, leave the intercept's diagonal entry of $\\lambda I$ at $0$, or (simpler) center $y$ and $X$ first so the intercept vanishes. Every library does this for you — but if you hand-roll it, remember." },
      ]
    },

    {
      id: "lasso",
      title: "Lasso (L1): sparsity and automatic feature selection",
      level: "core",
      body: [
        { type: "p", text: "**Lasso** (Least Absolute Shrinkage and Selection Operator, Tibshirani 1996) swaps the squared norm for the *absolute* norm:" },
        { type: "math", tex: String.raw`\min_{w}\; \|Xw - y\|_2^2 \;+\; \lambda\,\|w\|_1, \qquad \|w\|_1 = \sum_{j} |w_j|` },
        { type: "p", text: "One character changed — $\\|w\\|_2^2 \\to \\|w\\|_1$ — but the behavior is qualitatively different. Lasso doesn't just shrink weights; it drives many of them to **exactly zero**, producing a *sparse* model that has automatically selected a subset of features. For high-dimensional problems where you suspect most features are irrelevant, this is gold: the model tells you which inputs matter." },
        { type: "heading", text: "Why L1 gives exact zeros — the geometry" },
        { type: "p", text: "Both penalties can be written as a **constrained** problem: minimize the fit error subject to keeping the weight vector inside a ball of some radius $t$ (there is a one-to-one correspondence between $\\lambda$ and $t$)." },
        { type: "math", tex: String.raw`\min_w \|Xw-y\|_2^2 \;\; \text{s.t.} \;\; \|w\|_1 \le t \quad\text{(Lasso)}, \qquad \|w\|_2 \le t \quad\text{(Ridge)}` },
        { type: "p", text: "The unregularized loss has elliptical contours centered on the least-squares solution. The optimum sits where the smallest ellipse first *touches* the constraint region. Now picture the two regions in 2-D:" },
        { type: "table",
          headers: ["Penalty", "Constraint region", "First contact point"],
          rows: [
            ["$L_2$ (Ridge)", "a round **disk** (no corners)", "a generic point — both $w_j \\neq 0$"],
            ["$L_1$ (Lasso)", "a **diamond** with corners on the axes", "usually a **corner** — some $w_j = 0$"],
          ]
        },
        { type: "p", text: "The $L_1$ diamond has sharp **corners sitting exactly on the axes**, where one or more coordinates are zero. An expanding ellipse is overwhelmingly likely to strike a pointy corner before it grazes a flat edge — and at a corner, some weights are exactly $0$. The round $L_2$ disk has no corners, so contact happens at a generic point with all coordinates nonzero. That is the whole story of sparsity, and it only gets *more* true in high dimensions, where the $L_1$ ball is mostly corners and edges." },
        { type: "callout", variant: "tip", text: "**One-line intuition:** the kink in $|w|$ at $w=0$ is what creates sparsity. A smooth penalty like $w^2$ has zero slope at the origin, so there's no force pinning a weight *at* zero; the $|w|$ penalty has a constant slope $\\lambda$ right up to zero, so a feature must 'earn' its way out of zero by reducing the loss faster than $\\lambda$." },
        { type: "heading", text: "The price: no closed form" },
        { type: "p", text: "Because $|w_j|$ is not differentiable at $w_j = 0$, there is **no normal-equation-style closed form** for Lasso. You cannot just invert a matrix. Instead you use the *subgradient* — a set-valued generalization of the derivative — where the subgradient of $|w_j|$ is $\\operatorname{sign}(w_j)$ for $w_j \\neq 0$ and any value in $[-1, 1]$ at $w_j = 0$:" },
        { type: "math", tex: String.raw`\partial\, |w_j| = \begin{cases} \{+1\} & w_j > 0 \\ [-1,\, 1] & w_j = 0 \\ \{-1\} & w_j < 0 \end{cases}` },
        { type: "p", text: "The flat $[-1,1]$ interval at zero is exactly the mathematical room that lets a weight *stay pinned* at zero across a whole range of data — that interval is sparsity, written as calculus. In practice Lasso is solved by **coordinate descent** (update one weight at a time via soft-thresholding) or LARS; we build coordinate descent from scratch in the code section." },
      ]
    },

    {
      id: "elastic-net",
      title: "Elastic Net & choosing λ by cross-validation",
      level: "core",
      body: [
        { type: "p", text: "Lasso has two well-known failure modes. First, when a group of features is highly correlated, Lasso tends to arbitrarily pick *one* and zero the rest — unstable across resamples. Second, when $n > m$ (more features than samples), Lasso can select at most $m$ features. **Elastic Net** (Zou & Hastie, 2005) fixes both by combining *both* penalties:" },
        { type: "math", tex: String.raw`\min_w\; \|Xw - y\|_2^2 \;+\; \lambda\Big(\alpha\,\|w\|_1 \;+\; (1-\alpha)\,\|w\|_2^2\Big)` },
        { type: "p", text: "The mixing parameter $\\alpha \\in [0,1]$ interpolates: $\\alpha = 1$ is pure Lasso, $\\alpha = 0$ is pure Ridge, and in between you get sparsity *plus* the grouping behavior of Ridge (correlated features tend to enter or leave together, and get shared credit). It is a strong default when you want feature selection but don't fully trust Lasso's stability." },
        { type: "callout", variant: "note", text: "scikit-learn parameterizes this slightly differently — it uses `alpha` for the overall strength (our $\\lambda$) and `l1_ratio` for the mix (our $\\alpha$). Read the docs for the exact objective; the *idea* is identical." },
        { type: "heading", text: "Choosing λ: cross-validation" },
        { type: "p", text: "$\\lambda$ is a **hyperparameter** — it is not learned by the fit, it governs the fit. You cannot pick it by minimizing training error (that always prefers $\\lambda = 0$). You pick it by estimating *out-of-sample* error with **k-fold cross-validation**:" },
        { type: "list", ordered: true, items: [
          "Choose a grid of candidate $\\lambda$ values, usually **log-spaced** (e.g. $10^{-4}$ to $10^{2}$) since the effect of $\\lambda$ is multiplicative.",
          "Split the training data into $k$ folds (typically $k = 5$ or $10$).",
          "For each $\\lambda$: train on $k-1$ folds, measure validation error on the held-out fold, and repeat so every fold is held out once. Average the $k$ errors.",
          "Pick the $\\lambda$ with the lowest mean validation error — or apply the **one-standard-error rule**: choose the *largest* $\\lambda$ (simplest model) whose mean error is within one standard error of the best. This deliberately errs toward more regularization.",
          "Refit on the full training set at the chosen $\\lambda$, then report on a truly held-out **test** set.",
        ]},
        { type: "callout", variant: "gotcha", text: "**Do the scaling inside the CV loop.** Standardization must be fit on the *training folds only* and applied to the validation fold — if you standardize using the whole dataset first, statistics from the validation fold leak into training and your CV score is optimistic. Use a scikit-learn `Pipeline` so the scaler is refit per fold automatically." },
        { type: "callout", variant: "tip", text: "Libraries provide efficient built-ins — `RidgeCV`, `LassoCV`, `ElasticNetCV` — that compute the entire **regularization path** (all $\\lambda$ on the grid) far faster than a naive loop, using warm starts. Prefer them over rolling your own grid search for these specific models." },
      ]
    },

    {
      id: "bayesian",
      title: "The Bayesian view: penalties are priors",
      level: "core",
      body: [
        { type: "p", text: "Regularization can feel like an arbitrary hack — why *this* penalty? The Bayesian interpretation makes it principled: **a regularization penalty is a prior belief about the weights**, and minimizing 'loss + penalty' is finding the **maximum a posteriori (MAP)** estimate. Ridge and Lasso fall out of two different priors. (Bayes' rule and these distributions are covered on the **Probability & Statistics** page.)" },
        { type: "heading", text: "The setup" },
        { type: "p", text: "Model the data as linear with Gaussian noise, $y_i = w^\\top x_i + \\varepsilon_i$ with $\\varepsilon_i \\sim \\mathcal{N}(0, \\sigma^2)$. The MAP estimate maximizes the posterior, which by Bayes' rule is proportional to likelihood times prior. Taking $-\\log$ turns the product into a sum we *minimize*:" },
        { type: "math", tex: String.raw`\hat w_{\text{MAP}} = \arg\max_w\; p(w \mid y) = \arg\min_w\; \Big[ -\log p(y \mid w) \;-\; \log p(w) \Big]` },
        { type: "p", text: "The negative log-likelihood of Gaussian noise is just the sum of squared errors (up to constants): $-\\log p(y\\mid w) = \\frac{1}{2\\sigma^2}\\|y - Xw\\|_2^2 + \\text{const}$. So the fit term is fixed — the *prior* is what determines the penalty." },
        { type: "heading", text: "Gaussian prior → Ridge (L2)" },
        { type: "p", text: "Put an independent zero-mean **Gaussian** prior on each weight, $w_j \\sim \\mathcal{N}(0, \\tau^2)$ — a belief that weights are probably small. Its negative log-density is quadratic:" },
        { type: "math", tex: String.raw`-\log p(w) = \frac{1}{2\tau^2}\sum_j w_j^2 + \text{const} = \frac{1}{2\tau^2}\|w\|_2^2 + \text{const}` },
        { type: "math", tex: String.raw`\hat w_{\text{MAP}} = \arg\min_w\; \frac{1}{2\sigma^2}\|y - Xw\|_2^2 + \frac{1}{2\tau^2}\|w\|_2^2 \;\;\equiv\;\; \arg\min_w \|y - Xw\|_2^2 + \lambda\|w\|_2^2` },
        { type: "p", text: "with $\\lambda = \\sigma^2/\\tau^2$. **Ridge is MAP under a Gaussian prior.** A tighter prior (small $\\tau^2$, strong belief that weights are near zero) means a larger $\\lambda$ — exactly the right behavior." },
        { type: "heading", text: "Laplace prior → Lasso (L1)" },
        { type: "p", text: "Now put a **Laplace** (double-exponential) prior on each weight, $p(w_j) \\propto \\exp(-|w_j|/b)$. It is sharply peaked at zero with heavier tails than a Gaussian — a belief that most weights are *exactly* zero but a few can be large. Its negative log-density is the absolute value:" },
        { type: "math", tex: String.raw`-\log p(w) = \frac{1}{b}\sum_j |w_j| + \text{const} = \frac{1}{b}\|w\|_1 + \text{const}` },
        { type: "math", tex: String.raw`\hat w_{\text{MAP}} = \arg\min_w\; \frac{1}{2\sigma^2}\|y - Xw\|_2^2 + \frac{1}{b}\|w\|_1 \;\;\equiv\;\; \arg\min_w \|y - Xw\|_2^2 + \lambda\|w\|_1` },
        { type: "p", text: "**Lasso is MAP under a Laplace prior.** The sharp peak of the Laplace density at zero is the probabilistic echo of the geometric corner we saw earlier — the prior itself *wants* weights to sit exactly at zero." },
        { type: "callout", variant: "tip", text: "**Why this matters beyond elegance:** it tells you regularization strength has a *meaning* — it encodes how strongly you believe weights are small before seeing data. It also explains Elastic Net (a prior that's a mix of Gaussian and Laplace) and connects to the full Bayesian treatment, where instead of a single MAP point you keep the whole posterior distribution over $w$ and get uncertainty for free." },
      ]
    },

    {
      id: "everywhere",
      title: "The same idea everywhere: dropout, early stopping, augmentation",
      level: "core",
      body: [
        { type: "p", text: "'Regularization' is not a linear-models thing — it is *any* technique that reduces variance by constraining the effective complexity of a model. Once you see the pattern, you find it in every corner of ML. Here are the big ones, and the single idea that unifies them." },
        { type: "table",
          headers: ["Technique", "Where", "What it constrains"],
          rows: [
            ["$L_2$ / weight decay", "linear models, neural nets", "magnitude of weights"],
            ["$L_1$ / Lasso", "linear models, sparse nets", "number of nonzero weights"],
            ["Early stopping", "any iterative training", "how far weights move from init"],
            ["Dropout", "neural networks", "co-adaptation of neurons"],
            ["Data augmentation", "vision, audio, text", "invariances the model must respect"],
            ["Batch/layer norm", "deep nets", "activation scale (a side effect)"],
          ]
        },
        { type: "heading", text: "Weight decay = L2, almost" },
        { type: "p", text: "**Weight decay** shrinks every weight by a small factor each optimizer step: $w \\leftarrow (1 - \\eta\\gamma)\\,w - \\eta\\,\\nabla \\mathcal{L}$. For plain SGD this is *exactly* equivalent to adding an $L_2$ penalty $\\frac{\\gamma}{2}\\|w\\|^2$ to the loss — the penalty's gradient is $\\gamma w$, which subtracts $\\eta\\gamma w$ each step. This is the most common regularizer in deep learning, and it is the ridge penalty you just derived, wearing a different name. (For Adam the equivalence breaks; use `AdamW`, which decouples decay from the adaptive step — you will meet this in **Neural Network training**.)" },
        { type: "heading", text: "Early stopping" },
        { type: "p", text: "Train an iterative model and watch validation error: it falls, bottoms out, then rises as the model begins to overfit. **Early stopping** simply halts at the bottom. It costs nothing and, remarkably, for linear models trained by gradient descent it is *provably* a form of $L_2$ regularization — limiting the number of steps limits how far weights can travel from their small initialization, which is a soft budget on their magnitude." },
        { type: "heading", text: "Dropout" },
        { type: "p", text: "**Dropout** (Srivastava et al., 2014) randomly zeros a fraction $p$ of a layer's activations on each training step, forcing the network to not rely on any single neuron. It behaves like training an ensemble of exponentially many sub-networks that share weights and averaging them — an ensembling-flavored regularizer unique to nets. Full treatment in the **Neural Networks** track." },
        { type: "heading", text: "Data augmentation" },
        { type: "p", text: "**Data augmentation** expands the training set with label-preserving transforms — flip/crop/rotate an image, paraphrase a sentence, add noise to audio. It teaches the model the *invariances* it should respect and makes memorization harder because it never sees the same example twice. It is regularization applied to the data instead of the loss." },
        { type: "callout", variant: "note", text: "**The unifying idea.** Every one of these injects a *preference for simpler solutions* — whether by penalizing weight size, limiting training, randomizing structure, or enriching data. Overfitting is the model having more capacity than the data can constrain; regularization is any way of quietly spending that excess capacity on robustness instead of memorization." },
      ]
    },

    {
      id: "from-scratch",
      title: "From scratch: Ridge, coordinate-descent Lasso, then scikit-learn",
      level: "core",
      body: [
        { type: "p", text: "Now build them. Ridge is a one-line closed form; Lasso needs an iterative solver because of that non-differentiable kink. We do both by hand in NumPy, then reproduce with scikit-learn." },
        { type: "heading", text: "Ridge from the normal equation" },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef ridge_fit(X, y, lam):\n    \"\"\"Closed-form ridge. Assumes X, y already standardized/centered.\n    Does NOT penalize an intercept (add it back separately if needed).\"\"\"\n    n_features = X.shape[1]\n    A = X.T @ X + lam * np.eye(n_features)   # the +lambda*I ridge\n    return np.linalg.solve(A, X.T @ y)       # solve, never invert\n\n# Synthetic data: only the first 3 of 20 features actually matter.\nrng = np.random.default_rng(0)\nm, n = 60, 20\nX = rng.standard_normal((m, n))\nw_true = np.zeros(n); w_true[:3] = [4.0, -2.0, 3.0]\ny = X @ w_true + 0.5 * rng.standard_normal(m)\n\nfor lam in [0.0, 1.0, 100.0]:\n    w = ridge_fit(X, y, lam)\n    print(f\"lambda={lam:6.1f}  ||w||_2={np.linalg.norm(w):6.3f}  \"\n          f\"w[:3]={np.round(w[:3], 2)}\")\n# As lambda grows, ||w|| shrinks and all weights move toward 0 (none hit 0)." },
        { type: "callout", variant: "gotcha", text: "Use `np.linalg.solve(A, X.T @ y)`, **not** `np.linalg.inv(A) @ X.T @ y`. Solving the system is faster and numerically stabler than forming the inverse — the same lesson as the normal equation in **Linear Algebra**." },
        { type: "heading", text: "Lasso by coordinate descent" },
        { type: "p", text: "Coordinate descent optimizes one weight at a time, holding the rest fixed. For Lasso each 1-D subproblem has a closed-form solution given by the **soft-thresholding operator**, which shrinks a value toward zero and clamps it to exactly zero once it's small enough:" },
        { type: "math", tex: String.raw`S(z, \gamma) = \operatorname{sign}(z)\,\max\big(|z| - \gamma,\; 0\big) = \begin{cases} z - \gamma & z > \gamma \\ 0 & |z| \le \gamma \\ z + \gamma & z < -\gamma \end{cases}` },
        { type: "p", text: "For the objective $\\tfrac{1}{2}\\|y - Xw\\|_2^2 + \\lambda\\|w\\|_1$, cycling coordinate $j$ uses the partial residual $\\rho_j = x_j^\\top(y - Xw + w_j x_j)$ and the update $w_j \\leftarrow S(\\rho_j,\\, \\lambda) / (x_j^\\top x_j)$. That flat interval $[-\\lambda, \\lambda]$ mapped to $0$ *is* the sparsity, in code." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef soft_threshold(z, gamma):\n    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)\n\ndef lasso_cd(X, y, lam, n_iters=500, tol=1e-6):\n    \"\"\"Lasso via cyclic coordinate descent. Standardize X first!\"\"\"\n    m, n = X.shape\n    w = np.zeros(n)\n    col_sq = (X ** 2).sum(axis=0)          # x_j^T x_j per column\n    for _ in range(n_iters):\n        w_old = w.copy()\n        for j in range(n):\n            # partial residual: everything except feature j's contribution\n            r_j = y - X @ w + w[j] * X[:, j]\n            rho_j = X[:, j] @ r_j\n            w[j] = soft_threshold(rho_j, lam) / col_sq[j]\n        if np.max(np.abs(w - w_old)) < tol:\n            break\n    return w\n\nrng = np.random.default_rng(0)\nm, n = 60, 20\nX = rng.standard_normal((m, n))\nX = (X - X.mean(0)) / X.std(0)             # standardize (essential for Lasso)\nw_true = np.zeros(n); w_true[:3] = [4.0, -2.0, 3.0]\ny = X @ w_true + 0.5 * rng.standard_normal(m)\n\nw = lasso_cd(X, y, lam=5.0)\nprint(\"nonzero weights:\", np.flatnonzero(np.abs(w) > 1e-8))\nprint(\"w[:5] =\", np.round(w[:5], 2))\n# Lasso zeros out the 17 junk features EXACTLY and keeps features 0,1,2." },
        { type: "callout", variant: "tip", text: "Notice the contrast: run the ridge and lasso snippets side by side. Ridge returns 20 small-but-nonzero weights; Lasso returns 3 nonzero weights and 17 that are *identically* zero. That is the L1 corner geometry showing up in your terminal." },
        { type: "heading", text: "The scikit-learn versions" },
        { type: "code", lang: "py", code: "import numpy as np\nfrom sklearn.linear_model import Ridge, Lasso, ElasticNetCV\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\n\nrng = np.random.default_rng(0)\nX = rng.standard_normal((60, 20))\nw_true = np.zeros(20); w_true[:3] = [4.0, -2.0, 3.0]\ny = X @ w_true + 0.5 * rng.standard_normal(60)\n\n# Ridge: 'alpha' is our lambda. Scale first via a Pipeline.\nridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)\n\n# Lasso: same API, gives a sparse coef_.\nlasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1)).fit(X, y)\nprint(\"lasso nonzeros:\", np.flatnonzero(lasso[-1].coef_))\n\n# ElasticNetCV: picks BOTH lambda and the L1/L2 mix by cross-validation.\nencv = make_pipeline(\n    StandardScaler(),\n    ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, n_alphas=100, random_state=0),\n).fit(X, y)\nnet = encv[-1]\nprint(f\"chosen alpha={net.alpha_:.4f}, l1_ratio={net.l1_ratio_}\")\nprint(\"nonzeros:\", np.flatnonzero(net.coef_))" },
        { type: "heading", text: "The regularization path" },
        { type: "p", text: "The single most illuminating plot in this topic is the **regularization path**: fit the model across the whole grid of $\\lambda$ and plot each coefficient as a function of $\\lambda$ (usually with a log x-axis). For **Ridge**, every curve slides *smoothly* toward zero as $\\lambda$ grows but none reaches it. For **Lasso**, curves hit *exactly* zero one by one and stay there — you literally watch features drop out of the model. Reading right-to-left, you see the order in which features 'earn' their way into the model as regularization loosens; the last few standing are your most important predictors." },
        { type: "code", lang: "py", code: "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.linear_model import lasso_path\n\nrng = np.random.default_rng(0)\nX = rng.standard_normal((60, 20))\nX = (X - X.mean(0)) / X.std(0)\nw_true = np.zeros(20); w_true[:3] = [4.0, -2.0, 3.0]\ny = X @ w_true + 0.5 * rng.standard_normal(60)\n\nalphas, coefs, _ = lasso_path(X, y, n_alphas=100)\nplt.plot(np.log10(alphas), coefs.T)   # one line per feature\nplt.xlabel(\"log10(lambda)\"); plt.ylabel(\"coefficient\")\nplt.title(\"Lasso regularization path\")\n# Left (small lambda): all 20 coefs active. Right (large lambda): all crushed\n# to 0. The 3 true features are the LAST to leave — read that off the plot." },
      ]
    },

    {
      id: "practical",
      title: "Practical guidance: what to actually do",
      level: "core",
      body: [
        { type: "heading", text: "Standardize first — always" },
        { type: "p", text: "A penalty on $\\|w\\|$ treats all weights equally, but a feature measured in millimeters gets a huge weight while the same feature in kilometers gets a tiny one — so an un-scaled penalty silently punishes small-scale features and ignores large-scale ones. **Standardize every feature to zero mean and unit variance before regularizing.** This is not optional for Ridge/Lasso/Elastic Net; skipping it is the single most common regularization mistake." },
        { type: "code", lang: "py", code: "from sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.linear_model import Ridge\n\n# Right way: scaler is refit on the training data of each CV fold.\nmodel = make_pipeline(StandardScaler(), Ridge(alpha=1.0))" },
        { type: "callout", variant: "gotcha", text: "**Fit the scaler on training data only.** If you standardize using statistics from the full dataset (including the test/validation rows), information leaks and your reported error is optimistic. A `Pipeline` inside cross-validation handles this correctly by construction — always regularize inside a pipeline." },
        { type: "heading", text: "Read the coefficient path, not just one λ" },
        { type: "p", text: "Don't fixate on a single $\\lambda$. Plot the path and the CV error curve together. The path shows you *which* features survive and in what order; the CV curve shows you the bias–variance tradeoff as a U-shape — error high on the left (overfit, too little penalty), high on the right (underfit, too much), lowest in the valley. Pick $\\lambda$ from the valley, or one standard error to its right for a simpler model." },
        { type: "heading", text: "When to use which" },
        { type: "table",
          headers: ["Situation", "Use", "Why"],
          rows: [
            ["Many features, all somewhat useful", "**Ridge**", "keeps everything, shrinks smoothly, best when signal is spread out"],
            ["Suspect most features are junk", "**Lasso**", "zeros the junk, gives an interpretable sparse model"],
            ["Correlated feature groups", "**Elastic Net**", "Lasso alone picks one arbitrarily; Elastic Net shares credit"],
            ["$n > m$ (more features than rows)", "**Ridge / Elastic Net**", "$X^\\top X$ singular; the penalty makes it solvable"],
            ["Neural network", "**weight decay + dropout**", "$L_2$ on weights plus stochastic regularization"],
            ["Need interpretability / feature selection", "**Lasso**", "the zeros *are* the feature-selection answer"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Default recipe:** standardize → `ElasticNetCV` (it spans Ridge at `l1_ratio=0` through Lasso at `l1_ratio=1`, so it auto-discovers which end of the spectrum your problem wants) → inspect the path and CV curve → refit at the chosen $\\lambda$ and report on a held-out test set. When in doubt, start there." },
        { type: "callout", variant: "warn", text: "Regularization reduces variance at the cost of bias — it is **not** a fix for a genuinely mis-specified model or for too little data. If train *and* validation error are both high (underfitting), more $\\lambda$ makes things worse; you need a richer model or better features, not a bigger penalty." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Regularization only clicks when you *watch* it work. Each of these makes the bias–variance tradeoff visible. Do at least two." },
        { type: "list", ordered: true, items: [
          "**Overfitting, then fixing it.** Fit polynomials of degree 1–15 to ~15 noisy points from a known curve. Plot train vs. test error against degree to see the U-shape, then add a Ridge penalty to the degree-15 fit and sweep $\\lambda$ — watch the wild curve calm down into the true shape.",
          "**Ridge vs. Lasso paths.** On a dataset with a few real features and many junk ones (build it like the code sections), fit both across a log-spaced $\\lambda$ grid and plot both regularization paths side by side. Confirm Ridge shrinks smoothly while Lasso zeros features one at a time.",
          "**Coordinate-descent Lasso from scratch.** Implement the soft-thresholding coordinate-descent solver, then verify your coefficients match scikit-learn's `Lasso` to a few decimals on the same data. Add a convergence plot of $\\|w\\|_1$ over iterations.",
          "**Cross-validate honestly.** Build a `Pipeline(StandardScaler, ElasticNetCV)`, run 5-fold CV, and plot mean validation error with error bars vs. $\\lambda$. Mark both the min and the one-standard-error choice. Then break it: standardize *before* the split and show the CV score gets optimistically better (leakage).",
          "**The Bayesian equivalence, empirically.** Implement MAP estimation with a Gaussian prior and, separately, closed-form Ridge with $\\lambda = \\sigma^2/\\tau^2$. Confirm they return identical weights. Repeat with a Laplace prior vs. your Lasso solver.",
          "**Weight decay in a tiny net.** Train a small MLP on a noisy dataset with and without weight decay (and with/without dropout). Plot the train/test gap for each to see regularization shrink it — connecting this page to the Neural Networks track.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The standard treatments, in recommended order — the first two are the canonical textbook chapters every ML practitioner should read on this topic:" },
        { type: "link", url: "https://hastie.su.domains/ElemStatLearn/", text: "Hastie, Tibshirani & Friedman — The Elements of Statistical Learning, Ch. 3 (Ridge, Lasso, and the geometry; free PDF from the authors)" },
        { type: "link", url: "https://www.statlearning.com/", text: "James, Witten, Hastie & Tibshirani — An Introduction to Statistical Learning, Ch. 6 (the gentler companion; free PDF, with labs)" },
        { type: "link", url: "https://www.jstor.org/stable/2346178", text: "Tibshirani (1996) — Regression Shrinkage and Selection via the Lasso (the original Lasso paper)" },
        { type: "link", url: "https://hastie.su.domains/Papers/elasticnet.pdf", text: "Zou & Hastie (2005) — Regularization and Variable Selection via the Elastic Net (the original Elastic Net paper)" },
        { type: "link", url: "https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification", text: "scikit-learn User Guide — Ridge, Lasso, Elastic Net and their CV variants (the practical API reference)" },
        { type: "link", url: "https://www.jmlr.org/papers/v15/srivastava14a.html", text: "Srivastava et al. (2014) — Dropout: A Simple Way to Prevent Neural Networks from Overfitting (JMLR)" },
        { type: "link", url: "https://www.deeplearningbook.org/contents/regularization.html", text: "Goodfellow, Bengio & Courville — Deep Learning, Ch. 7 (regularization for deep models: weight decay, dropout, early stopping, augmentation)" },
      ]
    },
  ],

  packages: [
    { name: "numpy", why: "closed-form ridge (`solve`) and a hand-rolled coordinate-descent Lasso — the from-scratch foundation" },
    { name: "sklearn.linear_model", why: "`Ridge`, `Lasso`, `ElasticNet` and their `*CV` variants — the production estimators" },
    { name: "RidgeCV / LassoCV / ElasticNetCV", why: "efficient path-based cross-validation to pick $\\lambda$ (and the L1/L2 mix)" },
    { name: "sklearn.preprocessing.StandardScaler", why: "standardize features before penalizing — mandatory, not optional" },
    { name: "sklearn.pipeline.Pipeline", why: "chain scaler + model so scaling is refit per CV fold and never leaks" },
    { name: "sklearn.linear_model.lasso_path", why: "compute the full regularization path for the classic coefficient-vs-$\\lambda$ plot" },
    { name: "matplotlib", why: "plot regularization paths and CV error U-curves — seeing the tradeoff is the point" },
  ],

  gotchas: [
    "**Standardize before regularizing.** A penalty on $\\|w\\|$ punishes features by their scale, not their importance; unscaled Ridge/Lasso is meaningless. Always use a `Pipeline` with `StandardScaler`.",
    "**Never penalize the intercept.** Shrinking $w_0$ just biases predictions toward the origin. Center the data or exclude the bias from the penalty (libraries do this automatically).",
    "Fit the scaler on training folds only. Standardizing on the full dataset before CV leaks validation statistics and makes your score optimistic.",
    "Ridge has a closed form; **Lasso does not** — $|w|$ is non-differentiable at $0$, so it needs coordinate descent or LARS, not a normal equation.",
    "Ridge shrinks weights toward zero but **never to exactly zero**; only $L_1$ (Lasso) produces exact zeros and thus feature selection.",
    "Larger $\\lambda$ = more bias, less variance. If both train and validation error are high you are *under*fitting — more $\\lambda$ makes it worse, not better.",
    "In scikit-learn, `alpha` is the regularization strength (our $\\lambda$) and `l1_ratio` is the L1/L2 mix (our $\\alpha$) — the opposite of some textbooks. Check the objective in the docs.",
    "Search $\\lambda$ on a **log scale** ($10^{-4}$ to $10^{2}$), not linearly — its effect is multiplicative.",
  ],

  flashcards: [
    { q: "What does regularization trade against what?", a: "It adds a little **bias** to buy a large reduction in **variance** — nudging the model toward simpler solutions so it stops fitting noise." },
    { q: "Write the ridge (L2) closed-form solution and say what changed vs. OLS.", a: "$w = (X^\\top X + \\lambda I)^{-1}X^\\top y$. The only change from OLS is $+\\lambda I$, which lifts every eigenvalue by $\\lambda$ — making the matrix invertible and shrinking weights." },
    { q: "Why does Lasso (L1) drive weights to exactly zero but Ridge doesn't?", a: "The $L_1$ constraint ball is a diamond with corners on the axes (where coordinates are $0$); the loss contours usually first touch a corner. The $L_2$ ball is round with no corners, so contact is at a generic nonzero point." },
    { q: "Why is there no closed form for Lasso?", a: "$|w_j|$ is non-differentiable at $w_j=0$. You use the subgradient and solve via coordinate descent (soft-thresholding) or LARS." },
    { q: "State the soft-thresholding operator used in Lasso coordinate descent.", a: "$S(z,\\gamma)=\\operatorname{sign}(z)\\max(|z|-\\gamma, 0)$: shrink toward zero and clamp to exactly $0$ once $|z|\\le\\gamma$." },
    { q: "What prior makes Ridge the MAP estimate? What prior makes Lasso?", a: "A zero-mean **Gaussian** prior on the weights gives Ridge ($L_2$); a **Laplace** (double-exponential) prior gives Lasso ($L_1$), with $\\lambda \\propto \\sigma^2$." },
    { q: "What is Elastic Net and when do you reach for it?", a: "A weighted mix of $L_1$ and $L_2$ penalties. Use it when features are correlated in groups — Lasso alone picks one arbitrarily; Elastic Net gives them shared credit while still being sparse." },
    { q: "How do you choose $\\lambda$?", a: "Cross-validation over a log-spaced grid: pick the $\\lambda$ minimizing mean validation error, or the largest within one standard error of it (simpler model). Never from training error." },
    { q: "How is weight decay related to L2 regularization?", a: "For plain SGD they are exactly equivalent: decaying $w \\leftarrow (1-\\eta\\gamma)w$ each step is the gradient of a $\\frac{\\gamma}{2}\\|w\\|^2$ penalty. (For Adam, use AdamW to keep them equivalent.)" },
    { q: "Name three non-penalty regularizers and the unifying idea.", a: "Early stopping, dropout, data augmentation. Unifying idea: any technique that constrains effective model complexity, spending excess capacity on robustness instead of memorization." },
    { q: "Why must you standardize features before Ridge/Lasso?", a: "The penalty treats all weights equally, so a feature's scale changes how hard it's penalized. Standardizing to unit variance makes the penalty fair across features." },
  ],

  cheatsheet: [
    { label: "Ridge objective", code: "min ||Xw - y||^2 + lam * ||w||_2^2" },
    { label: "Ridge closed form", code: "w = solve(X.T@X + lam*np.eye(n), X.T@y)" },
    { label: "Lasso objective", code: "min ||Xw - y||^2 + lam * ||w||_1" },
    { label: "Soft-threshold", code: "np.sign(z)*np.maximum(np.abs(z)-g, 0)" },
    { label: "sklearn Ridge", code: "Ridge(alpha=1.0).fit(X, y)" },
    { label: "sklearn Lasso", code: "Lasso(alpha=0.1).fit(X, y)" },
    { label: "Pick lambda (CV)", code: "RidgeCV(alphas=np.logspace(-4,2,50)).fit(X,y)" },
    { label: "Elastic Net + CV", code: "ElasticNetCV(l1_ratio=[.1,.5,.9,1], cv=5)" },
    { label: "Scale then regularize", code: "make_pipeline(StandardScaler(), Ridge())" },
    { label: "Regularization path", code: "alphas, coefs, _ = lasso_path(X, y)" },
    { label: "Selected features", code: "np.flatnonzero(model.coef_)" },
    { label: "Weight decay (torch)", code: "torch.optim.AdamW(params, weight_decay=1e-2)" },
    { label: "Lambda grid", code: "np.logspace(-4, 2, 50)  # log-spaced" },
  ],
});
