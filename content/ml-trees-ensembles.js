(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "ml-trees-ensembles",
  name: "Decision Trees & Ensembles",
  language: "Classical ML",
  group: "Classical Machine Learning",
  navLabel: "Trees & Ensembles",
  tagline: "The models that win most tabular-data competitions — decision trees, random forests, and gradient boosting, derived from scratch and in scikit-learn / XGBoost.",
  color: "#16A34A",
  readMinutes: 52,
  sections: [
    {
      id: "why",
      title: "Why trees dominate tabular data",
      level: "core",
      body: [
        { type: "p", text: "Neural networks own images, audio, and text. But on **tabular data** — the spreadsheets, logs, and database dumps that make up the overwhelming majority of real business ML — a completely different family wins almost every time: **gradient-boosted decision trees**. Scan the winning solutions of Kaggle's tabular competitions and you will see XGBoost, LightGBM, and CatBoost again and again, usually beating carefully tuned deep nets. This section is about *why*, and how to build every piece from a single decision tree up to a boosted ensemble." },
        { type: "p", text: "The whole family is built from one humble primitive — a tree of yes/no questions — combined in two ways:" },
        { type: "table",
          headers: ["Model", "How it's built", "One-line intuition"],
          rows: [
            ["Decision tree", "recursive greedy splits", "a flowchart of `if/else` questions"],
            ["Random forest", "**bagging** many deep trees", "average many overfit trees to cancel their variance"],
            ["AdaBoost", "**boosting** stumps by reweighting", "focus each new model on the previous mistakes"],
            ["Gradient boosting", "**boosting** by fitting residuals", "gradient descent in function space"],
            ["XGBoost / LightGBM", "regularized 2nd-order boosting", "the same idea, engineered to win"],
          ]
        },
        { type: "p", text: "Two of these — *bagging* (variance reduction by averaging) and *boosting* (bias reduction by sequential correction) — are general ideas that apply to any model. We meet them here because trees are the ideal base learner: fast, non-linear, and just unstable enough to benefit enormously from ensembling." },
        { type: "callout", variant: "note", text: "**How to read this deck.** Every idea is shown three ways: the **intuition** (what it *means*), the **math** (the exact criterion + a short derivation), and the **code** (NumPy from scratch first, then the scikit-learn / XGBoost one-liner). Build the from-scratch tree at least once — it is only ~40 lines and it demystifies the entire family." },
        { type: "callout", variant: "tip", text: "**Why trees, in one sentence:** they need no feature scaling, handle mixed numeric/categorical data, capture non-linear interactions automatically, are invariant to monotone transforms of each feature, and are robust to outliers in $x$. That is exactly the profile of messy real-world tabular data — and exactly where neural nets struggle without heavy preprocessing." },
      ]
    },

    {
      id: "the-tree",
      title: "Decision trees: axis-aligned splits & interpretability",
      level: "core",
      body: [
        { type: "p", text: "A **decision tree** predicts by asking a sequence of simple threshold questions about single features — 'is `age < 30`?', then 'is `income > 50k`?' — walking down from the root until it reaches a **leaf**, which holds the prediction. For classification the leaf stores a class (or class probabilities); for regression it stores a number (usually the mean of the training targets that landed there)." },
        { type: "p", text: "Each internal node tests **one feature against one threshold**, so every split is a hyperplane perpendicular to an axis. The tree therefore carves the input space into **axis-aligned rectangles** (hyper-rectangles in higher dimensions), and predicts a constant inside each box." },
        { type: "math", tex: String.raw`\hat{y}(x) = \sum_{r=1}^{R} c_r \,\mathbb{1}\!\left[x \in R_r\right], \qquad R_r \text{ = a leaf's rectangle}, \; c_r \text{ = its value}` },
        { type: "heading", text: "What the geometry buys you — and what it costs" },
        { type: "p", text: "Because splits are axis-aligned, a tree can model wild non-linearities and feature interactions for free, but it draws only 'staircase' boundaries. A simple diagonal line — trivial for a linear model — costs a tree a deep staircase of splits to approximate. This is the fundamental trade: trees are flexible but *blocky*." },
        { type: "code", lang: "text", code: "A tiny tree deciding \"will this user churn?\"\n\n            [ tenure_months < 12 ? ]\n              /                 \\\n           yes                   no\n            |                     |\n   [ monthly_charge > 70 ? ]   predict: STAY (0.08)\n      /            \\\n    yes             no\n     |               |\n  CHURN (0.71)    STAY (0.22)\n\nEach path root->leaf is one 'if AND if AND if' rule.\nEach leaf is one axis-aligned box in feature space." },
        { type: "heading", text: "Interpretability — the tree's superpower" },
        { type: "p", text: "A single shallow tree is one of the few genuinely **glass-box** models: every prediction is a short chain of human-readable rules you can print, audit, and explain to a regulator. That is why trees survive in credit scoring, medicine, and fraud, where 'the model said so' is not an acceptable answer. The catch: this transparency only holds for *small* trees. A depth-30 tree, or a forest of 500 of them, is as opaque as any neural net — which is why Section 8 (feature importance / SHAP) exists." },
        { type: "callout", variant: "note", text: "**A little history.** The two classic tree algorithms are **CART** (Breiman, Friedman, Olshen & Stone, 1984), which uses Gini impurity and builds binary trees, and the **ID3 / C4.5** lineage (Quinlan, 1986/1993), which uses information gain. scikit-learn implements an optimized CART. The core greedy idea has barely changed in 40 years — what changed is how we *combine* trees." },
      ]
    },

    {
      id: "splitting",
      title: "Splitting criteria: Gini, entropy & variance reduction",
      level: "core",
      body: [
        { type: "p", text: "A tree is only as good as the questions it asks. At every node the algorithm searches over **every feature and every candidate threshold** and picks the split that makes the resulting child nodes as **pure** as possible — as close as possible to containing a single class (classification) or a single value (regression). We need to define 'impurity' precisely." },
        { type: "heading", text: "Gini impurity" },
        { type: "p", text: "For a node with class proportions $p_1, \\dots, p_K$, the **Gini impurity** is the probability that you misclassify a randomly drawn sample if you label it by randomly drawing a class from the node's own distribution:" },
        { type: "math", tex: String.raw`G = \sum_{k=1}^{K} p_k\,(1 - p_k) = 1 - \sum_{k=1}^{K} p_k^{2}` },
        { type: "p", text: "It is $0$ when the node is pure (one $p_k = 1$) and maximal ($1 - 1/K$) when classes are perfectly balanced. For binary problems $G = 2p(1-p)$, a downward parabola peaking at $p = 0.5$." },
        { type: "heading", text: "Entropy & information gain" },
        { type: "p", text: "**Entropy** measures the same thing in bits — the average surprise of the node's label distribution, straight from information theory:" },
        { type: "math", tex: String.raw`H = -\sum_{k=1}^{K} p_k \log_2 p_k` },
        { type: "p", text: "Also $0$ at purity, maximal ($\\log_2 K$) at uniform. To score a *split*, we compare the parent's impurity to the size-weighted impurity of the children. For entropy this difference is called **information gain**:" },
        { type: "math", tex: String.raw`\mathrm{IG} = H(\text{parent}) - \sum_{c \in \{L,R\}} \frac{n_c}{n}\, H(c)` },
        { type: "p", text: "The identical formula with $G$ in place of $H$ gives the **Gini gain**. The tree chooses the split that maximizes this gain — equivalently, that minimizes the weighted child impurity $\\frac{n_L}{n}I_L + \\frac{n_R}{n}I_R$." },
        { type: "table",
          headers: ["Criterion", "Formula", "Notes"],
          rows: [
            ["Gini", "$1 - \\sum_k p_k^2$", "CART default; no logs, slightly faster"],
            ["Entropy", "$-\\sum_k p_k \\log_2 p_k$", "ID3/C4.5; info-theoretic, very similar splits"],
            ["Misclassification", "$1 - \\max_k p_k$", "used for *pruning*, too flat for *growing*"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Gini vs entropy — does it matter?** Almost never. They agree on the best split the vast majority of the time and produce near-identical trees; Gini is marginally cheaper (no logarithm). Spend your tuning budget on `max_depth` and the ensemble, not on this choice." },
        { type: "heading", text: "Regression trees: variance reduction" },
        { type: "p", text: "For regression there are no classes, so 'impurity' becomes **variance** (equivalently, the sum of squared errors around the node's mean). A leaf predicts the mean $\\bar{y}$ of the targets that reach it, and the node impurity is the mean squared error of that constant prediction:" },
        { type: "math", tex: String.raw`\mathrm{MSE}(node) = \frac{1}{n}\sum_{i \in node}(y_i - \bar{y})^2, \qquad \bar{y} = \frac{1}{n}\sum_{i \in node} y_i` },
        { type: "p", text: "The best split is the one that most reduces the total weighted variance of the children — the **variance reduction** criterion. Everything else (greedy search, recursion, stopping) is identical to classification; only the impurity function changes." },
        { type: "callout", variant: "gotcha", text: "Impurity is computed on the split's **children**, weighted by their sizes — never on the parent alone. A split that produces one huge impure child and one tiny pure child is usually a *bad* split; the size weighting $\\frac{n_c}{n}$ is what encodes that." },
      ]
    },

    {
      id: "greedy-cart",
      title: "How a split is chosen: greedy CART",
      level: "core",
      body: [
        { type: "p", text: "Finding the globally optimal decision tree is **NP-hard** — you cannot search all trees. CART instead grows the tree **greedily and recursively**: at each node it picks the single best split *right now* (myopically, ignoring future splits), then recurses on each child. It never backtracks. This greedy choice is provably suboptimal in general, but it is fast and works remarkably well in practice, especially once ensembled." },
        { type: "heading", text: "The best-split search" },
        { type: "p", text: "At a node with data $S$, for **every feature** $j$ and **every candidate threshold** $t$ (the midpoints between consecutive sorted values of feature $j$), split $S$ into $S_L = \\{x : x_j \\le t\\}$ and $S_R = \\{x : x_j > t\\}$ and score it. Keep the $(j, t)$ with the lowest weighted child impurity:" },
        { type: "math", tex: String.raw`(j^\star, t^\star) = \arg\min_{j,\,t}\; \frac{|S_L|}{|S|}\,I(S_L) + \frac{|S_R|}{|S|}\,I(S_R)` },
        { type: "p", text: "Naively this is $O(\\text{features} \\times \\text{samples}^2)$, but if you **sort each feature once** and sweep the threshold while incrementally updating the class counts, each feature costs $O(n\\log n)$ for the sort plus $O(n)$ for the sweep. This sort-and-sweep is exactly the trick histogram-based libraries (Section 7) push to its limit." },
        { type: "heading", text: "When does recursion stop?" },
        { type: "p", text: "A node becomes a **leaf** when any stopping rule fires: it is pure, it has too few samples to split (`min_samples_split`), splitting would create a child that is too small (`min_samples_leaf`), it hits the depth cap (`max_depth`), or no split improves impurity by at least `min_impurity_decrease`. These are the knobs that control the bias–variance trade-off — the subject of the next section." },
        { type: "callout", variant: "gotcha", text: "Greedy growth means a feature that is useless *on its own* but valuable *in combination* can be missed at the top of the tree (the classic XOR pattern). Ensembles paper over this: different bagged trees split in different orders and collectively recover the interaction. A lone tree can be fooled by it." },
      ]
    },

    {
      id: "from-scratch",
      title: "Building a tree from scratch, plus overfitting & pruning",
      level: "core",
      body: [
        { type: "p", text: "Here is a complete classification tree in NumPy — recursive best-split with Gini. Read it top-down; it is the whole algorithm in ~45 lines and everything later in this deck is built on top of it." },
        { type: "code", lang: "py", code: "import numpy as np\n\nclass Node:\n    def __init__(self, feature=None, thresh=None, left=None, right=None, value=None):\n        self.feature, self.thresh = feature, thresh   # split rule (internal node)\n        self.left, self.right = left, right           # child nodes\n        self.value = value                            # class label (leaf only)\n\ndef gini(y):\n    _, counts = np.unique(y, return_counts=True)\n    p = counts / counts.sum()\n    return 1.0 - np.sum(p ** 2)\n\ndef best_split(X, y):\n    n, d = X.shape\n    parent = gini(y)\n    best_gain, best = 0.0, None\n    for j in range(d):                              # every feature\n        thresholds = np.unique(X[:, j])\n        for t in thresholds:                        # every candidate threshold\n            left = X[:, j] <= t\n            if left.sum() == 0 or left.sum() == n:  # no real split\n                continue\n            nl, nr = left.sum(), n - left.sum()\n            child = (nl/n) * gini(y[left]) + (nr/n) * gini(y[~left])\n            gain = parent - child                   # weighted Gini gain\n            if gain > best_gain:\n                best_gain, best = gain, (j, t)\n    return best\n\ndef build(X, y, depth=0, max_depth=5, min_samples=2):\n    # Stopping rules -> make a leaf holding the majority class\n    if depth >= max_depth or len(y) < min_samples or len(np.unique(y)) == 1:\n        vals, counts = np.unique(y, return_counts=True)\n        return Node(value=vals[np.argmax(counts)])\n    split = best_split(X, y)\n    if split is None:\n        vals, counts = np.unique(y, return_counts=True)\n        return Node(value=vals[np.argmax(counts)])\n    j, t = split\n    mask = X[:, j] <= t\n    left  = build(X[mask],  y[mask],  depth+1, max_depth, min_samples)\n    right = build(X[~mask], y[~mask], depth+1, max_depth, min_samples)\n    return Node(feature=j, thresh=t, left=left, right=right)\n\ndef predict_one(node, x):\n    while node.value is None:\n        node = node.left if x[node.feature] <= node.thresh else node.right\n    return node.value" },
        { type: "code", lang: "py", code: "# Sanity check on a toy problem\nfrom sklearn.datasets import make_classification\nX, y = make_classification(n_samples=400, n_features=6, n_informative=4,\n                           random_state=0)\ntree = build(X, y, max_depth=4)\npred = np.array([predict_one(tree, x) for x in X])\nprint(\"train accuracy:\", (pred == y).mean())   # ~0.9+, and it will overfit if max_depth is large" },
        { type: "heading", text: "Overfitting: the disease of unpruned trees" },
        { type: "p", text: "A tree grown until every leaf is pure will **memorize the training set** — it can always keep splitting until each leaf holds one point, achieving 100% training accuracy and terrible test accuracy. Trees are **low-bias, high-variance**: tiny changes in the data can flip an early split and reshape the whole tree. This instability is a bug for a single tree and, remarkably, the *feature* that makes ensembles work." },
        { type: "heading", text: "Two cures: pre-pruning and post-pruning" },
        { type: "table",
          headers: ["Approach", "How", "scikit-learn knob"],
          rows: [
            ["Pre-pruning", "stop growing early", "`max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`"],
            ["Post-pruning", "grow full, then cut back weak branches", "`ccp_alpha` (cost-complexity pruning)"],
          ]
        },
        { type: "p", text: "**Cost-complexity (weakest-link) pruning** grows the full tree, then minimizes a penalized objective that trades training error against the number of leaves $|T|$, exactly like $L_1$/$L_2$ regularization trades fit against model size:" },
        { type: "math", tex: String.raw`R_\alpha(T) = R(T) + \alpha\,|T|` },
        { type: "p", text: "As $\\alpha$ rises from $0$, you get a nested sequence of ever-smaller trees; you pick $\\alpha$ by cross-validation. This is the classic ISLR recipe. In modern practice you more often just cap `max_depth` and lean on the ensemble." },
        { type: "callout", variant: "tip", text: "**Rule of thumb.** For a *single* interpretable tree, prune hard — depth 3–5 is often plenty and stays readable. For trees *inside a random forest*, do the opposite: grow them deep and unpruned; the averaging (next section) handles the variance for you." },
      ]
    },

    {
      id: "bagging-rf",
      title: "Bagging & random forests: killing variance by averaging",
      level: "core",
      body: [
        { type: "p", text: "A single deep tree is a high-variance mess. **Bagging** (bootstrap aggregating, Breiman 1996) fixes this with one idea from statistics: **averaging many noisy-but-unbiased estimators reduces variance without adding bias.** Train $B$ trees on $B$ different bootstrap samples of the data and average their predictions (or vote, for classification)." },
        { type: "heading", text: "Why averaging works — the variance-of-the-mean argument" },
        { type: "p", text: "Suppose you have $B$ estimators, each with variance $\\sigma^2$ and pairwise correlation $\\rho$. The variance of their average is:" },
        { type: "math", tex: String.raw`\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} T_b\right) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2` },
        { type: "p", text: "Read this formula slowly — it is the entire theory of ensembles. As $B \\to \\infty$ the second term vanishes, so variance falls to $\\rho\\sigma^2$. If the trees were **independent** ($\\rho = 0$), variance would shrink all the way to $\\sigma^2/B$. The bias is unchanged because the average of unbiased estimators is unbiased. So the game is clear: **average many low-bias trees, and drive down their correlation $\\rho$.**" },
        { type: "heading", text: "Random forests: decorrelating the trees" },
        { type: "p", text: "Plain bagged trees are still highly correlated — one dominant feature gets chosen as the top split in nearly every tree. **Random forests** (Breiman 2001) add a second dose of randomness that directly attacks the $\\rho$ term: at each split, consider only a **random subset of $m$ features** (typically $m = \\sqrt{d}$ for classification, $d/3$ for regression) instead of all $d$." },
        { type: "math", tex: String.raw`m = \lfloor\sqrt{d}\rfloor \;\text{(classification)}, \qquad m = \lfloor d/3 \rfloor \;\text{(regression)}` },
        { type: "p", text: "This forces different trees to rely on different features, lowering $\\rho$ and thus the ensemble variance — at the cost of slightly higher bias per tree (each split sees fewer options). The trade is overwhelmingly worth it. That is the whole recipe: **bagging + per-split feature subsampling = random forest.**" },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef bootstrap(X, y, rng):\n    n = len(y)\n    idx = rng.integers(0, n, size=n)   # sample n rows WITH replacement\n    oob = np.setdiff1d(np.arange(n), idx)   # rows left out -> out-of-bag\n    return X[idx], y[idx], oob\n\nclass RandomForest:\n    def __init__(self, n_trees=100, max_depth=8, seed=0):\n        self.n_trees, self.max_depth = n_trees, max_depth\n        self.rng = np.random.default_rng(seed)\n        self.trees = []\n\n    def fit(self, X, y):\n        # 'build' here would subsample sqrt(d) features per split internally.\n        for _ in range(self.n_trees):\n            Xb, yb, _ = bootstrap(X, y, self.rng)\n            self.trees.append(build(Xb, yb, max_depth=self.max_depth))\n        return self\n\n    def predict(self, X):\n        # majority vote across trees\n        votes = np.array([[predict_one(t, x) for t in self.trees] for x in X])\n        return np.array([np.bincount(v).argmax() for v in votes])" },
        { type: "heading", text: "Out-of-bag (OOB) error: free cross-validation" },
        { type: "p", text: "A bootstrap sample of size $n$ leaves out, on average, a fixed fraction of the data. The probability a given row is *never* picked in $n$ draws is:" },
        { type: "math", tex: String.raw`\left(1 - \frac{1}{n}\right)^{n} \xrightarrow{\;n\to\infty\;} e^{-1} \approx 0.368` },
        { type: "p", text: "So each tree never sees ~**37%** of the rows — its **out-of-bag** set. Predict each row using only the trees that did *not* train on it, and you get a validation estimate essentially for free, no separate holdout or k-fold loop required. Set `oob_score=True` in scikit-learn to get it." },
        { type: "callout", variant: "good", text: "**Random forests are the best 'no-thinking' baseline in ML.** They need almost no tuning, don't overfit as you add trees (more trees only helps, it just plateaus), handle mixed data, give you OOB error and feature importances for free, and parallelize trivially. Always run one first — it tells you what score is achievable before you invest in tuning a booster." },
        { type: "callout", variant: "gotcha", text: "More trees never *hurts* accuracy but has diminishing returns and linear cost — 100–500 is usually plenty. And a forest is **not** interpretable: you have traded the single tree's glass box for a black box you now need permutation importance or SHAP (Section 8) to understand." },
      ]
    },

    {
      id: "boosting",
      title: "Boosting: turn weak learners into a strong one",
      level: "core",
      body: [
        { type: "p", text: "Bagging builds trees **in parallel** and averages them to cut *variance*. **Boosting** builds them **sequentially**, each new model correcting the errors of the ensemble so far, to cut *bias*. It answers a startling theoretical question posed by Kearns & Valiant and settled by Schapire (1990): can a bunch of 'weak' learners — each only slightly better than a coin flip — be combined into an arbitrarily 'strong' one? **Yes.** Boosting is the constructive proof." },
        { type: "heading", text: "AdaBoost: boosting by reweighting" },
        { type: "p", text: "**AdaBoost** (Freund & Schapire, 1997 — Gödel Prize) was the first practical booster. It keeps a weight $w_i$ on every training example. Start uniform, then repeat: fit a weak learner (usually a depth-1 'stump'), **increase the weights of the misclassified points** so the next learner focuses on them, and give each learner a vote proportional to how well it did." },
        { type: "math", tex: String.raw`\varepsilon_m = \frac{\sum_i w_i\,\mathbb{1}[y_i \neq h_m(x_i)]}{\sum_i w_i}, \qquad \alpha_m = \tfrac{1}{2}\ln\!\frac{1 - \varepsilon_m}{\varepsilon_m}` },
        { type: "math", tex: String.raw`w_i \leftarrow w_i \, e^{-\alpha_m\, y_i\, h_m(x_i)} \quad(\text{then renormalize}), \qquad y_i \in \{-1, +1\}` },
        { type: "p", text: "The final prediction is a weighted vote, $H(x) = \\operatorname{sign}\\!\\big(\\sum_m \\alpha_m h_m(x)\\big)$. A learner better than chance has $\\varepsilon_m < 0.5$, so $\\alpha_m > 0$; a near-perfect learner gets a huge vote. Misclassified points get their weight multiplied by $e^{\\alpha_m} > 1$, correctly classified points by $e^{-\\alpha_m} < 1$ — attention flows to the hard cases." },
        { type: "callout", variant: "note", text: "**The deep insight (Friedman, Hastie & Tibshirani, 2000):** AdaBoost is *exactly* forward stagewise additive modeling that minimizes the **exponential loss** $\\sum_i e^{-y_i F(x_i)}$. Once you see boosting as greedily minimizing a loss one learner at a time, the door opens to minimizing *any* differentiable loss — which is gradient boosting, the next section." },
        { type: "callout", variant: "gotcha", text: "AdaBoost's exponential loss punishes misclassified points *exponentially*, so it is sensitive to **label noise and outliers** — a mislabeled point gets its weight blown up every round. Modern gradient boosting with a robust loss (e.g. Huber) largely superseded it, but AdaBoost remains the cleanest way to *first understand* boosting." },
      ]
    },

    {
      id: "gradient-boosting",
      title: "Gradient boosting: gradient descent in function space",
      level: "core",
      body: [
        { type: "p", text: "**Gradient boosting** (Friedman, 2001) is the idea that dominates tabular ML. The reframe is beautiful: instead of doing gradient descent on a vector of *parameters*, do gradient descent on the *function itself*, taking one small step per iteration — where each step is a regression tree." },
        { type: "heading", text: "The setup: additive model, one tree at a time" },
        { type: "p", text: "We build the predictor $F$ as a sum of trees, adding one per round and never revising the old ones (forward stagewise):" },
        { type: "math", tex: String.raw`F_M(x) = F_0(x) + \sum_{m=1}^{M} \nu\, h_m(x)` },
        { type: "p", text: "$F_0$ is a constant (the optimal single prediction, e.g. the mean for squared loss), $h_m$ is the tree added at round $m$, and $\\nu \\in (0,1]$ is the **learning rate** (shrinkage). Our goal is to minimize the total loss over all examples:" },
        { type: "math", tex: String.raw`\mathcal{L}(F) = \sum_{i=1}^{n} \ell\big(y_i,\, F(x_i)\big)` },
        { type: "heading", text: "The derivation: what should the next tree fit?" },
        { type: "p", text: "Think of $F$ as a giant vector of its values at the training points, $\\big(F(x_1),\\dots,F(x_n)\\big)$. Gradient descent says: to reduce $\\mathcal{L}$, step in the direction of the **negative gradient** with respect to those values. The gradient component at example $i$ is:" },
        { type: "math", tex: String.raw`g_i = \frac{\partial\, \ell\big(y_i, F(x_i)\big)}{\partial\, F(x_i)}\Bigg|_{F = F_{m-1}}` },
        { type: "p", text: "The ideal update would be $F_m(x_i) = F_{m-1}(x_i) - \\nu\\, g_i$. But we cannot store an arbitrary function — we can only add a *tree*. So we **fit a regression tree $h_m$ to the negative gradients** $-g_i$ (called the **pseudo-residuals**), giving us a tree-shaped approximation to the ideal descent direction:" },
        { type: "math", tex: String.raw`h_m = \arg\min_{h}\; \sum_{i=1}^{n} \big(-g_i - h(x_i)\big)^2, \qquad F_m = F_{m-1} + \nu\, h_m` },
        { type: "heading", text: "The special case that makes it click: squared loss" },
        { type: "p", text: "Take $\\ell(y, F) = \\tfrac{1}{2}(y - F)^2$. Then the gradient is $g_i = -(y_i - F_{m-1}(x_i))$, so the negative gradient is *literally the residual*:" },
        { type: "math", tex: String.raw`-g_i = y_i - F_{m-1}(x_i) = \text{current residual}` },
        { type: "p", text: "So for squared loss, **gradient boosting = repeatedly fit a tree to the leftover errors and add it in.** That is the whole algorithm in one line, and the 'gradient' machinery is just what generalizes 'fit the residual' to log-loss (classification), Huber loss (robust regression), Poisson loss (counts), and beyond — you only swap the formula for $g_i$." },
        { type: "code", lang: "py", code: "import numpy as np\nfrom sklearn.tree import DecisionTreeRegressor\n\ndef gradient_boost_mse(X, y, n_trees=100, lr=0.1, max_depth=3):\n    F0 = y.mean()                      # optimal constant for squared loss\n    F  = np.full_like(y, F0, dtype=float)\n    trees = []\n    for _ in range(n_trees):\n        residual = y - F               # -gradient of 1/2 (y-F)^2\n        t = DecisionTreeRegressor(max_depth=max_depth)\n        t.fit(X, residual)             # fit a tree to the residuals\n        F += lr * t.predict(X)         # take a shrunken step\n        trees.append(t)\n    return F0, trees\n\ndef gb_predict(F0, trees, X, lr=0.1):\n    return F0 + lr * sum(t.predict(X) for t in trees)\n\n# Each tree nudges the prediction a little closer; lr controls the step size." },
        { type: "heading", text: "The two knobs that matter most: learning rate & number of trees" },
        { type: "p", text: "Shrinkage $\\nu$ and the tree count $M$ trade off directly: a **smaller $\\nu$ needs more trees** but generalizes better (each tree contributes less, so the ensemble is smoother and less prone to overfitting). The standard recipe is a small learning rate (0.01–0.1) with as many trees as your **early-stopping** validation curve allows." },
        { type: "callout", variant: "gotcha", text: "Unlike random forests, **gradient boosting overfits if you add too many trees** — each tree chases the training residual, so past the optimum the validation loss turns back up. Always use a validation set with `early_stopping_rounds` / `n_iter_no_change` to pick $M$ automatically. This is the single most important difference from bagging: forests plateau, boosters overfit." },
        { type: "callout", variant: "tip", text: "**Depth of the base trees encodes interaction order.** Stumps (depth 1) model only additive main effects; depth-$k$ trees can capture $k$-way feature interactions. For boosting, shallow trees (depth 3–8) are the sweet spot — deep enough to catch interactions, shallow enough to stay weak learners." },
      ]
    },

    {
      id: "xgboost",
      title: "XGBoost, LightGBM & CatBoost: why they win Kaggle",
      level: "core",
      body: [
        { type: "p", text: "Plain gradient boosting is the idea; **XGBoost** (Chen & Guestrin, 2016), **LightGBM** (Microsoft, 2017), and **CatBoost** (Yandex, 2017) are the engineering that made it dominate. They share three upgrades over textbook GBM: a **regularized objective**, a **second-order (Newton) step**, and **histogram-based split finding**." },
        { type: "heading", text: "1. A regularized objective with a second-order step" },
        { type: "p", text: "XGBoost adds an explicit complexity penalty to the loss — $T$ is the number of leaves and $w_j$ their output values — so regularization is baked into training, not bolted on:" },
        { type: "math", tex: String.raw`\mathcal{L} = \sum_i \ell(y_i, \hat y_i) + \sum_m \Omega(f_m), \qquad \Omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2` },
        { type: "p", text: "Then, instead of a first-order (gradient-only) step, it takes a **second-order Taylor expansion** of the loss around the current prediction, using both the gradient $g_i$ and the Hessian $h_i = \\partial^2\\ell / \\partial \\hat y_i^2$:" },
        { type: "math", tex: String.raw`\mathcal{L}^{(m)} \approx \sum_i \Big[ g_i\, f_m(x_i) + \tfrac{1}{2} h_i\, f_m(x_i)^2 \Big] + \Omega(f_m)` },
        { type: "p", text: "This is Newton's method in function space. Because a tree is constant on each leaf, we can solve for the **optimal leaf weight** in closed form (sum the $g$'s and $h$'s of the points in leaf $j$ into $G_j, H_j$):" },
        { type: "math", tex: String.raw`w_j^\star = -\frac{G_j}{H_j + \lambda}, \qquad G_j = \!\!\sum_{i \in \text{leaf } j}\!\! g_i, \quad H_j = \!\!\sum_{i \in \text{leaf } j}\!\! h_i` },
        { type: "p", text: "Substituting back gives a **structure score** that rates how good a tree shape is, and from it the **gain** of a candidate split — the exact criterion XGBoost maximizes when growing a tree (the last $\\gamma$ term prunes splits that don't earn their keep):" },
        { type: "math", tex: String.raw`\text{Gain} = \tfrac{1}{2}\!\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma` },
        { type: "heading", text: "2. Histogram-based split finding (the speed win)" },
        { type: "p", text: "Textbook GBM sorts every feature at every node — the bottleneck. Instead, these libraries **bucket each feature into ~255 histogram bins once** and search only over bin edges. That turns split finding from $O(n \\log n)$ per feature into $O(\\text{bins})$, and the bin counts are cache-friendly and trivially parallel. This single change is most of the 10–100× speedup." },
        { type: "table",
          headers: ["Library", "Signature trick", "Best when"],
          rows: [
            ["XGBoost", "regularized 2nd-order objective; `hist` mode", "the robust default; great docs & tooling"],
            ["LightGBM", "leaf-wise growth + GOSS + feature bundling", "large data, fastest; watch overfitting on small sets"],
            ["CatBoost", "ordered boosting + native categorical encoding", "many categorical features; strong defaults, less tuning"],
          ]
        },
        { type: "p", text: "**LightGBM** grows trees **leaf-wise** (always split the leaf with the highest gain) rather than level-wise, plus GOSS (keep large-gradient rows, subsample the rest) and exclusive feature bundling for sparse data — fastest on big data, but likelier to overfit small data unless you cap `num_leaves`. **CatBoost** handles categorical features natively with target statistics and uses *ordered boosting* to prevent the target leakage that naive encodings cause — often the best out-of-the-box on categorical-heavy data." },
        { type: "callout", variant: "good", text: "**Why they dominate Kaggle:** state-of-the-art accuracy on tabular data, built-in regularization ($\\lambda$, $\\gamma$, subsampling, column sampling), native missing-value handling, second-order convergence in few iterations, GPU support, and early stopping. On structured data they routinely beat deep nets that cost 100× more to train. Start with XGBoost or LightGBM defaults; you are already near the top of the leaderboard." },
        { type: "callout", variant: "gotcha", text: "These models have *many* knobs (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `reg_alpha`, `num_leaves`). Tune the **learning rate + n_estimators (via early stopping)** and **tree complexity** first; the rest are second-order. Random or Bayesian search (Optuna) beats grid search here." },
      ]
    },

    {
      id: "importance",
      title: "Feature importance: Gini, permutation & SHAP",
      level: "core",
      body: [
        { type: "p", text: "You traded the single tree's interpretability for the ensemble's accuracy — so how do you know *why* a forest or booster predicted what it did? Three tools, from cheapest-but-flawed to gold-standard." },
        { type: "heading", text: "1. Impurity (Gini) importance — free but biased" },
        { type: "p", text: "The default `.feature_importances_`: for each feature, sum the impurity decrease it produced across all splits, weighted by how many samples pass through, averaged over trees. It is free (computed during training) but has a serious flaw: it is **biased toward high-cardinality features** (continuous variables and categoricals with many levels get more chances to split), and it is computed on *training* data, so it rewards overfitting." },
        { type: "heading", text: "2. Permutation importance — model-agnostic, honest" },
        { type: "p", text: "Measure a feature's importance by how much the model's *validation* score **drops when you shuffle that feature's values**, breaking its relationship with the target while preserving its marginal distribution:" },
        { type: "math", tex: String.raw`\text{Imp}(j) = s(\text{model}, X, y) \;-\; \frac{1}{K}\sum_{k=1}^{K} s\big(\text{model}, X^{(j\text{-shuffled})}_k, y\big)` },
        { type: "p", text: "Big drop = important feature. It works on *any* fitted model, uses held-out data, and avoids the cardinality bias. The catch: **correlated features** dilute each other's scores (shuffle one, its correlated twin still leaks the signal), so importances can look artificially low for redundant features." },
        { type: "heading", text: "3. SHAP — per-prediction attribution with theory behind it" },
        { type: "p", text: "**SHAP** (SHapley Additive exPlanations, Lundberg & Lee 2017) borrows **Shapley values** from cooperative game theory to fairly divide a single prediction among its features. It answers 'how much did each feature push *this specific prediction* above or below the baseline?', with a guarantee that contributions sum exactly to the prediction:" },
        { type: "math", tex: String.raw`f(x) = \phi_0 + \sum_{j=1}^{d} \phi_j, \qquad \phi_j = \!\!\sum_{S \subseteq F \setminus \{j\}}\!\! \frac{|S|!\,(d - |S| - 1)!}{d!}\big[f(S \cup \{j\}) - f(S)\big]` },
        { type: "p", text: "`shap.TreeExplainer` computes these exactly and *fast* for tree ensembles (polynomial time, not the exponential sum above). It gives both **local** explanations (why this one prediction?) and **global** ones (mean $|\\phi_j|$ across the dataset), and is now the default way to explain XGBoost/LightGBM in production." },
        { type: "table",
          headers: ["Method", "Cost", "Main weakness"],
          rows: [
            ["Impurity (Gini)", "free (during fit)", "biased to high-cardinality; uses train data"],
            ["Permutation", "moderate (refit-free, $K$ shuffles)", "correlated features dilute each other"],
            ["SHAP", "higher (fast for trees though)", "still fooled by strong correlation; more setup"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Practical recipe:** never trust default impurity importance for decisions. Use **permutation importance** for a quick, honest global ranking, and **SHAP** when you need per-prediction explanations (debugging, fairness audits, or telling a user why they were declined). Both are model-agnostic and worth the extra minutes." },
      ]
    },

    {
      id: "worked-example",
      title: "Worked example & when to reach for trees",
      level: "core",
      body: [
        { type: "p", text: "A realistic end-to-end comparison on tabular data: a random-forest baseline, then an XGBoost model with early stopping, then SHAP to explain it. This is the workflow you will run for most tabular problems." },
        { type: "code", lang: "py", code: "import numpy as np\nfrom sklearn.datasets import fetch_california_housing\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import mean_squared_error\n\nX, y = fetch_california_housing(return_X_y=True)\nXtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)\n\n# 1) Random-forest baseline: almost no tuning, OOB score for free\nrf = RandomForestRegressor(n_estimators=300, oob_score=True,\n                           n_jobs=-1, random_state=0).fit(Xtr, ytr)\nprint(\"RF   OOB R^2:\", round(rf.oob_score_, 3))\nprint(\"RF   test RMSE:\", round(mean_squared_error(yte, rf.predict(Xte)) ** 0.5, 3))" },
        { type: "code", lang: "py", code: "# 2) XGBoost with early stopping — usually the winner on tabular data\nimport xgboost as xgb\n\nmodel = xgb.XGBRegressor(\n    n_estimators=2000,        # an upper bound; early stopping picks the real count\n    learning_rate=0.05,       # small LR + many trees + early stopping\n    max_depth=6,\n    subsample=0.8,            # row subsampling (stochastic GB)\n    colsample_bytree=0.8,    # feature subsampling per tree\n    reg_lambda=1.0,          # L2 on leaf weights\n    early_stopping_rounds=50,\n    eval_metric=\"rmse\",\n)\nmodel.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)\nprint(\"XGB best #trees:\", model.best_iteration)\nprint(\"XGB test RMSE:\", round(mean_squared_error(yte, model.predict(Xte)) ** 0.5, 3))" },
        { type: "code", lang: "py", code: "# 3) Explain it with SHAP (exact + fast for trees)\nimport shap\nexplainer = shap.TreeExplainer(model)\nshap_values = explainer.shap_values(Xte)\n# shap.summary_plot(shap_values, Xte)   # global feature importance + direction\n# Contributions for one row sum exactly to that prediction minus the baseline:\nrow = 0\nprint(\"pred:\", model.predict(Xte[row:row+1])[0])\nprint(\"baseline + sum(shap):\", explainer.expected_value + shap_values[row].sum())" },
        { type: "heading", text: "When trees win — and when they lose" },
        { type: "p", text: "Choosing a model family is mostly about the *shape of your data*, not the size of your ambition." },
        { type: "table",
          headers: ["Data / situation", "Reach for", "Why"],
          rows: [
            ["Tabular (rows × columns), mixed types", "**gradient boosting**", "SOTA accuracy, minimal preprocessing, handles interactions"],
            ["Tabular, need a quick strong baseline", "**random forest**", "near-zero tuning, OOB score, hard to misuse"],
            ["Images / audio / video", "**CNNs / transformers**", "spatial structure & weight sharing; trees can't see pixels"],
            ["Text / sequences", "**transformers (LLMs)**", "need embeddings & long-range context, not axis splits"],
            ["Need a smooth extrapolation beyond training range", "**linear / GP / NN**", "trees predict a *constant* outside seen data — they can't extrapolate"],
            ["Tiny data, need a fully auditable rule", "**a single shallow tree** or linear model", "glass-box; an ensemble is overkill and opaque"],
          ]
        },
        { type: "callout", variant: "note", text: "**The honest summary from the research:** despite years of 'deep learning for tabular data' papers, careful 2022–2024 benchmarks (e.g. Grinsztajn et al.) keep finding that **gradient-boosted trees still match or beat deep nets on typical tabular datasets**, while training far faster and needing far less tuning. Trees' inductive bias — axis-aligned, non-smooth, feature-order-invariant — happens to fit tabular data. On perceptual data (pixels, waveforms, tokens) that bias is wrong, and neural nets win decisively." },
        { type: "callout", variant: "gotcha", text: "Trees **cannot extrapolate**: a regression tree/forest predicts the mean of the nearest training leaf, so outside the range of your training data it outputs a flat constant. If you need sensible predictions beyond the observed range (forecasting a rising trend, physics with known asymptotics), a linear or neural model is safer." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Build at least two of these end-to-end. The from-scratch ones cement the algorithm; the applied ones teach the tuning intuition that no formula gives you. Everything here runs on a laptop." },
        { type: "list", ordered: true, items: [
          "**Decision tree from scratch.** Extend the ~45-line tree in Section 5 to support both Gini and entropy, `min_samples_leaf`, and regression (variance reduction). Validate it against `sklearn.tree.DecisionTreeClassifier` on the Iris and Titanic datasets — you should match its accuracy closely. Then plot training vs. test accuracy as you increase `max_depth` to *see* overfitting appear.",
          "**Random forest + OOB.** Wrap your tree in a bagging loop with $\\sqrt{d}$ feature subsampling. Track the OOB error as you add trees and confirm it plateaus (never rises). Compare your OOB estimate to a real held-out test score — they should agree within noise.",
          "**Gradient boosting from scratch.** Implement squared-loss gradient boosting (fit a `DecisionTreeRegressor` to residuals, add with a learning rate) as in Section 6. Sweep `learning_rate` × `n_trees` and plot validation loss — watch it turn back *up* when there are too many trees, the signature of boosting overfitting. Then add early stopping.",
          "**XGBoost vs. everything on a real dataset.** On a Kaggle tabular set (e.g. House Prices or Titanic), benchmark logistic/linear regression, a random forest, and XGBoost with Optuna-tuned hyperparameters and early stopping. Report the accuracy *and* the training-time cost of each — the point is to feel the trade-off, not just chase the top score.",
          "**Explainability deep dive.** Take your best XGBoost model and compare all three importance methods (impurity, permutation, SHAP) on the same features. Deliberately add a duplicated (perfectly correlated) column and watch how each method splits or dilutes its importance. Produce a SHAP summary plot and explain one individual prediction in plain English.",
          "**Trees vs. neural net, head to head.** Pick one tabular dataset and one image dataset (e.g. MNIST as flattened vectors). Train a gradient-boosting model and a small MLP on each. Confirm the expected result: trees win on tabular, the net wins on pixels — and articulate *why* in terms of inductive bias.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The best places to go deeper, in recommended order — from visual intuition to the original papers:" },
        { type: "link", url: "https://www.statlearning.com/", text: "James, Witten, Hastie & Tibshirani — An Introduction to Statistical Learning (ISLR), Chapter 8 (Trees, Bagging, Random Forests, Boosting) — free PDF, the ideal first read" },
        { type: "link", url: "https://www.youtube.com/playlist?list=PLblh5JKOoLUKAtDViTvRGFpphEc24M-QH", text: "StatQuest (Josh Starmer) — Decision Trees, Random Forests, AdaBoost & Gradient Boost playlists — the clearest visual explanations anywhere; watch these first" },
        { type: "link", url: "https://arxiv.org/abs/1603.02754", text: "Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System (the paper; the regularized objective and split-finding derivations)" },
        { type: "link", url: "https://jerryfriedman.su.domains/ftp/trebst.pdf", text: "Friedman (2001) — Greedy Function Approximation: A Gradient Boosting Machine (the founding gradient-boosting paper)" },
        { type: "link", url: "https://scikit-learn.org/stable/modules/ensemble.html", text: "scikit-learn — Ensemble methods user guide (RandomForest, HistGradientBoosting, stacking) with practical parameter advice" },
        { type: "link", url: "https://xgboost.readthedocs.io/en/stable/tutorials/model.html", text: "XGBoost docs — 'Introduction to Boosted Trees' (the same second-order derivation, worked slowly with pictures)" },
        { type: "link", url: "https://christophm.github.io/interpretable-ml-book/", text: "Molnar — Interpretable Machine Learning (free book; the definitive treatment of permutation importance and SHAP)" },
        { type: "link", url: "https://arxiv.org/abs/2207.08815", text: "Grinsztajn, Oyallon & Varoquaux (2022) — Why do tree-based models still outperform deep learning on tabular data?" },
      ]
    },
  ],

  packages: [
    { name: "scikit-learn", why: "`DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `HistGradientBoosting*` — the reference implementations" },
    { name: "xgboost", why: "regularized 2nd-order gradient boosting; the tabular workhorse with early stopping & GPU support" },
    { name: "lightgbm", why: "leaf-wise histogram boosting — fastest on large datasets" },
    { name: "catboost", why: "native categorical handling + ordered boosting; strongest defaults on categorical-heavy data" },
    { name: "shap", why: "`TreeExplainer` for exact, fast per-prediction feature attributions" },
    { name: "numpy", why: "arrays and the from-scratch tree/forest/boosting implementations" },
    { name: "optuna", why: "Bayesian hyperparameter search — beats grid search for tuning boosters" },
    { name: "matplotlib", why: "plot learning curves, tree structure, SHAP summaries, importance rankings" },
  ],

  gotchas: [
    "A tree grown to pure leaves memorizes the training set. Trees are **low-bias, high-variance** — always cap `max_depth` / `min_samples_leaf` or ensemble them.",
    "Random forests **plateau** as you add trees (more never hurts); gradient boosting **overfits** past the optimal tree count. Use `early_stopping_rounds` for boosters.",
    "Default `.feature_importances_` (impurity) is **biased toward high-cardinality features** and computed on train data. Prefer permutation importance or SHAP for decisions.",
    "Trees **cannot extrapolate**: outside the training range they predict a flat constant $\\bar{y}$. Use linear/NN models when you need to extrapolate a trend.",
    "Boosting's tree depth sets the **interaction order** — stumps model only main effects. Depth 3–8 is the sweet spot; deeper base trees stop being 'weak' learners.",
    "In gradient boosting, a **smaller learning rate needs more trees**. Never tune `n_estimators` alone — tune it jointly with `learning_rate` via early stopping.",
    "Correlated features fool importance methods: permutation dilutes them and even SHAP splits credit between duplicates. Deduplicate before interpreting.",
    "Greedy CART is **myopic**: a feature useful only in combination (XOR-like) can be missed by a single tree. Ensembles recover it; a lone tree may not.",
  ],

  flashcards: [
    { q: "Write Gini impurity and entropy for class proportions $p_k$.", a: "Gini $G = 1 - \\sum_k p_k^2$; entropy $H = -\\sum_k p_k \\log_2 p_k$. Both are $0$ at purity and maximal at a uniform class mix." },
    { q: "How does a CART tree choose a split?", a: "Greedily: over every feature and threshold, pick the split minimizing the size-weighted child impurity $\\frac{n_L}{n}I_L + \\frac{n_R}{n}I_R$. It never backtracks (NP-hard to do optimally)." },
    { q: "What impurity does a regression tree minimize?", a: "Variance / MSE around the node mean: each leaf predicts $\\bar{y}$, and the best split maximizes variance reduction." },
    { q: "Why does averaging bagged trees reduce variance?", a: "$\\operatorname{Var}(\\frac1B\\sum T_b) = \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$. As $B\\to\\infty$ variance falls to $\\rho\\sigma^2$; bias is unchanged. So average low-bias trees and drive down their correlation $\\rho$." },
    { q: "What extra randomness makes a random forest better than plain bagging?", a: "Per-split **feature subsampling** ($m=\\sqrt{d}$ features considered at each split). It decorrelates the trees, lowering $\\rho$ and thus ensemble variance." },
    { q: "What is out-of-bag (OOB) error?", a: "Each bootstrap leaves out ~$e^{-1}\\approx37\\%$ of rows; scoring each row with only the trees that didn't train on it gives a free validation estimate." },
    { q: "Derive what the next tree fits in gradient boosting.", a: "Fit a regression tree to the negative gradient (pseudo-residuals) $-g_i = -\\partial\\ell/\\partial F(x_i)$ at $F_{m-1}$, then $F_m = F_{m-1} + \\nu h_m$. For squared loss, $-g_i$ is exactly the residual $y_i - F_{m-1}(x_i)$." },
    { q: "Why is gradient boosting called 'gradient descent in function space'?", a: "It treats $F$ as a vector of its values at the data points and steps along the negative functional gradient — but each step must be a tree, so it fits a tree to that gradient." },
    { q: "What does XGBoost add over textbook GBM?", a: "A regularized objective ($\\gamma T + \\frac12\\lambda\\|w\\|^2$), a **second-order (Newton) step** using gradient and Hessian, and **histogram-based** split finding for speed." },
    { q: "Give XGBoost's optimal leaf weight and split gain.", a: "$w_j^\\star = -G_j/(H_j+\\lambda)$ with $G_j,H_j$ the summed gradients/Hessians in the leaf; a split's gain is $\\frac12[\\frac{G_L^2}{H_L+\\lambda}+\\frac{G_R^2}{H_R+\\lambda}-\\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}]-\\gamma$." },
    { q: "When do trees beat neural nets, and when do they lose?", a: "Trees win on **tabular** data (mixed types, interactions, little preprocessing); nets win on **perceptual** data — images, audio, text — where the axis-aligned bias is wrong." },
    { q: "Impurity vs permutation vs SHAP importance — one line each.", a: "Impurity: free but biased to high-cardinality & train-fit. Permutation: honest, model-agnostic drop-in-score, but dilutes correlated features. SHAP: game-theoretic per-prediction attribution that sums to the prediction; best but costlier." },
  ],

  cheatsheet: [
    { label: "Decision tree", code: "from sklearn.tree import DecisionTreeClassifier; DecisionTreeClassifier(max_depth=4)" },
    { label: "Random forest", code: "from sklearn.ensemble import RandomForestClassifier; RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs=-1)" },
    { label: "Gradient boosting (sklearn)", code: "from sklearn.ensemble import HistGradientBoostingClassifier; HistGradientBoostingClassifier(learning_rate=0.05)" },
    { label: "XGBoost", code: "import xgboost as xgb; xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, early_stopping_rounds=50)" },
    { label: "LightGBM", code: "import lightgbm as lgb; lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05)" },
    { label: "CatBoost", code: "from catboost import CatBoostClassifier; CatBoostClassifier(cat_features=[...])" },
    { label: "Fit with early stopping", code: "model.fit(Xtr, ytr, eval_set=[(Xval, yval)])" },
    { label: "Impurity importance", code: "model.feature_importances_" },
    { label: "Permutation importance", code: "from sklearn.inspection import permutation_importance; permutation_importance(model, Xval, yval)" },
    { label: "SHAP (trees)", code: "import shap; shap.TreeExplainer(model).shap_values(X)" },
    { label: "OOB score", code: "RandomForestClassifier(oob_score=True).fit(X, y).oob_score_" },
    { label: "Visualize a tree", code: "from sklearn.tree import plot_tree; plot_tree(tree, filled=True)" },
    { label: "Key GB knobs", code: "learning_rate, n_estimators, max_depth, subsample, colsample_bytree, reg_lambda" },
  ],
});
