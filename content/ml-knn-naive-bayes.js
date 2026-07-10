(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "ml-knn-naive-bayes",
  name: "k-NN & Naive Bayes",
  language: "Classical ML",
  group: "Classical Machine Learning",
  navLabel: "k-NN & Naive Bayes",
  tagline: "Two dead-simple classifiers that teach two huge ideas — instance-based (lazy) learning and probabilistic learning — built from scratch and in scikit-learn.",
  color: "#22C55E",
  readMinutes: 42,
  sections: [
    {
      id: "why",
      title: "Why start with the two simplest classifiers",
      level: "core",
      body: [
        { type: "p", text: "Before gradient descent, before neural nets, there are two algorithms so simple you can implement each in a dozen lines — yet each is the cleanest possible illustration of a whole *family* of ML. **k-Nearest Neighbors (k-NN)** is the purest form of **instance-based learning**: it has no parameters and does no training; it just remembers the data and answers by analogy. **Naive Bayes** is the purest form of **probabilistic learning**: it turns Bayes' theorem into a classifier by making one gloriously wrong assumption that happens to work anyway." },
        { type: "p", text: "They sit at opposite corners of a useful map, and holding both in your head sharpens your instinct for every model that follows:" },
        { type: "table",
          headers: ["", "k-Nearest Neighbors", "Naive Bayes"],
          rows: [
            ["Learning style", "instance-based (lazy)", "model-based (eager)"],
            ["What it stores", "the entire training set", "per-class probabilities"],
            ["Training cost", "~zero (just copy data)", "one pass to count/estimate"],
            ["Prediction cost", "expensive — scan all points", "cheap — a few multiplies"],
            ["Assumption", "nearby points share a label", "features independent given class"],
            ["Decision boundary", "non-linear, local, flexible", "smooth (often quadratic/linear)"],
          ]
        },
        { type: "p", text: "Notice the mirror symmetry: k-NN pays nothing to train and everything to predict; Naive Bayes pays a little to train and almost nothing to predict. That trade-off — **when do you do the work?** — recurs through all of ML." },
        { type: "callout", variant: "note", text: "**How to read this page.** Everything is shown three ways: the **intuition** (what it means), the **math** (the exact rule plus a short derivation), and the **code** (NumPy from scratch, then the scikit-learn one-liner). The prerequisites are the Probability page (Bayes' theorem, distributions) and the Linear Algebra page (vectors, norms, the dot product)." },
      ]
    },

    {
      id: "knn-model",
      title: "k-Nearest Neighbors: the model with no model",
      level: "core",
      body: [
        { type: "p", text: "The k-NN rule fits on an index card. To classify a new point $x$: find the $k$ training points closest to it, and let them **vote**. The majority label wins. That is the entire algorithm — there are no weights to learn, no loss to minimize, no `.fit()` in any real sense." },
        { type: "math", tex: String.raw`\hat y(x) = \operatorname*{arg\,max}_{c}\; \sum_{i \in N_k(x)} \mathbb{1}[\,y_i = c\,]` },
        { type: "p", text: "Here $N_k(x)$ is the set of the $k$ nearest neighbors of $x$ in the training set, and $\\mathbb{1}[\\cdot]$ is $1$ when the neighbor's label equals class $c$ and $0$ otherwise. The prediction is just the most common label among those $k$ neighbors." },
        { type: "heading", text: "\"Training\" is copying — this is lazy learning" },
        { type: "p", text: "k-NN is the canonical **lazy learner**: `fit(X, y)` literally just stores `X` and `y`. All the computation is deferred to prediction time, when the query point finally shows up. The opposite — an **eager learner** like Naive Bayes or logistic regression — does the work up front to build a compact model and then throws the training data away. Lazy learning has one superpower: adding a new training point is free (append it), so k-NN adapts instantly to new data, which is why it underlies many recommendation and retrieval systems." },
        { type: "callout", variant: "note", text: "**k-NN is the ancestor of vector search.** Every embedding-based retrieval system — semantic search, RAG, face recognition, recommendation — is k-NN at scale: embed everything into $\\mathbb{R}^d$, then find nearest neighbors of a query vector. The whole 'vector database' industry is fast approximate k-NN. Learn it here and you have the mental model for all of it." },
        { type: "heading", text: "Distance metrics — 'nearest' needs a definition" },
        { type: "p", text: "'Closest' is meaningless until you pick a **distance metric**. The choice is not cosmetic — it changes which points count as neighbors and therefore the whole classifier. The three you must know:" },
        { type: "math", tex: String.raw`\underbrace{d_2(x,z) = \sqrt{\sum_{j=1}^{d}(x_j - z_j)^2}}_{\text{Euclidean } (L_2)} \qquad \underbrace{d_1(x,z) = \sum_{j=1}^{d}|x_j - z_j|}_{\text{Manhattan } (L_1)}` },
        { type: "p", text: "Euclidean ($L_2$) is straight-line distance and the default. Manhattan ($L_1$) sums per-axis distances (taxicab blocks) and is more robust to outliers in individual coordinates. Both are special cases of the **Minkowski** distance $\\left(\\sum_j |x_j - z_j|^p\\right)^{1/p}$ — $p=2$ gives Euclidean, $p=1$ gives Manhattan." },
        { type: "p", text: "**Cosine distance** measures the *angle* between vectors, ignoring their magnitude. It is the metric of choice for text and embeddings, where direction (which words, which concepts) matters more than length (document size):" },
        { type: "math", tex: String.raw`d_{\cos}(x,z) = 1 - \frac{x^\top z}{\|x\|\,\|z\|}` },
        { type: "callout", variant: "gotcha", text: "**k-NN needs feature scaling — always.** Distances add up per-feature squared differences, so a feature measured in the thousands (salary) drowns out a feature measured in single digits (years of experience). One big-scale feature and k-NN silently becomes a 1-NN on that feature alone. Standardize (`StandardScaler`) or min-max scale *before* fitting. This is the single most common k-NN mistake." },
        { type: "heading", text: "Choosing k — the bias–variance dial" },
        { type: "p", text: "$k$ is the one knob, and it is a direct bias–variance control. Small $k$ means each prediction is decided by a tiny local neighborhood: **low bias, high variance** — the boundary is jagged and hugs every point, including noise ($k=1$ can memorize outliers). Large $k$ averages over a big neighborhood: **high bias, low variance** — the boundary smooths out, but push $k$ toward the dataset size and every query just predicts the overall majority class." },
        { type: "table",
          headers: ["$k$", "Bias", "Variance", "Boundary", "Failure mode"],
          rows: [
            ["$k=1$", "very low", "very high", "jagged, memorizes", "overfits noise/outliers"],
            ["moderate $k$", "balanced", "balanced", "smooth but local", "the sweet spot"],
            ["$k \\approx m$", "very high", "very low", "nearly constant", "predicts majority class"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Pick $k$ with cross-validation, and prefer an odd $k$ for two-class problems** so a vote can't tie. A common starting heuristic is $k \\approx \\sqrt{m}$, but never trust a heuristic over a validation curve — sweep $k$ and plot accuracy." },
        { type: "heading", text: "Weighted k-NN — closer neighbors should count more" },
        { type: "p", text: "Plain voting treats the 1st and $k$-th neighbor equally, which is odd — the closest point is surely more relevant. **Distance-weighted k-NN** fixes this by weighting each neighbor's vote by $1/d$ (or a Gaussian kernel of the distance), so nearby points dominate:" },
        { type: "math", tex: String.raw`\hat y(x) = \operatorname*{arg\,max}_{c}\; \sum_{i \in N_k(x)} w_i\,\mathbb{1}[\,y_i = c\,], \qquad w_i = \frac{1}{d(x, x_i) + \varepsilon}` },
        { type: "p", text: "The small $\\varepsilon$ avoids division by zero when a query coincides with a training point. Weighting makes the result far less sensitive to the exact value of $k$, since distant, weakly-relevant neighbors barely register." },
      ]
    },

    {
      id: "knn-scratch",
      title: "k-NN from scratch, and the curse of dimensionality",
      level: "core",
      body: [
        { type: "p", text: "Let's build k-NN in NumPy. The whole thing is: store the data, then at query time compute distances to every training point, take the $k$ smallest, and vote. Vectorize the distance computation and it's genuinely fast for small-to-medium datasets." },
        { type: "code", lang: "py", code: "import numpy as np\nfrom collections import Counter\n\nclass KNN:\n    def __init__(self, k=5):\n        self.k = k\n\n    def fit(self, X, y):\n        # \"Training\" a lazy learner: just remember everything.\n        self.X = np.asarray(X, dtype=float)\n        self.y = np.asarray(y)\n        return self\n\n    def _distances(self, x):\n        # Euclidean distance from x to every stored point, vectorized.\n        # (X - x) broadcasts (m,d) - (d,) -> (m,d); sum squares along axis 1.\n        return np.sqrt(((self.X - x) ** 2).sum(axis=1))\n\n    def predict_one(self, x):\n        d = self._distances(x)\n        nn = np.argsort(d)[:self.k]          # indices of the k closest points\n        votes = Counter(self.y[nn])\n        return votes.most_common(1)[0][0]     # majority label\n\n    def predict(self, X):\n        return np.array([self.predict_one(x) for x in np.asarray(X, float)])\n\n# Toy check: two clusters, one per class.\nrng = np.random.default_rng(0)\nX0 = rng.normal([0, 0], 1, size=(50, 2))\nX1 = rng.normal([4, 4], 1, size=(50, 2))\nX = np.vstack([X0, X1]); y = np.array([0]*50 + [1]*50)\n\nmodel = KNN(k=5).fit(X, y)\nprint(model.predict([[0, 0], [4, 4], [2, 2]]))   # [0 1 ...] boundary point ~2,2" },
        { type: "callout", variant: "tip", text: "**One-line weighted variant.** Replace the vote with a weighted tally: `w = 1 / (d[nn] + 1e-9)`, then for each class sum the weights of neighbors in that class and take the arg-max. Distance-weighting usually buys a point or two of accuracy for free." },
        { type: "heading", text: "The curse of dimensionality — why k-NN breaks in high-D" },
        { type: "p", text: "k-NN's entire premise is that 'near' means 'similar.' In high dimensions that premise **collapses**, and this is one of the most important and counterintuitive facts in all of ML. As the number of features $d$ grows, *all pairwise distances converge to roughly the same value* — the nearest neighbor is barely closer than the farthest, so 'nearest' stops carrying information." },
        { type: "p", text: "Here is the derivation that makes it concrete. Take $m$ points spread uniformly in the unit cube $[0,1]^d$. Consider the ratio between the farthest and nearest distances from a query point. Beyer et al. (1999) showed that under broad conditions:" },
        { type: "math", tex: String.raw`\lim_{d \to \infty} \mathbb{E}\!\left[\frac{\operatorname{dist}_{\max} - \operatorname{dist}_{\min}}{\operatorname{dist}_{\min}}\right] \longrightarrow 0` },
        { type: "p", text: "The intuition behind it: each coordinate contributes an independent $(x_j - z_j)^2$ term to the squared distance. Summing $d$ of these, the **mean** distance grows like $\\sqrt{d}$ while the **spread** (standard deviation) grows much more slowly. So relative to the typical distance, every point is about equidistant — the distance distribution becomes a thin shell." },
        { type: "code", lang: "py", code: "import numpy as np\n\n# Watch neighbors become meaningless as dimension grows.\nrng = np.random.default_rng(0)\nfor d in [2, 10, 100, 1000]:\n    X = rng.random((1000, d))          # 1000 uniform points in [0,1]^d\n    q = rng.random(d)\n    dist = np.sqrt(((X - q) ** 2).sum(axis=1))\n    dmin, dmax = dist.min(), dist.max()\n    # As d grows, (dmax - dmin) / dmin -> 0: nearest ~ farthest.\n    print(f\"d={d:4d}  min={dmin:.2f}  max={dmax:.2f}  \"\n          f\"(max-min)/min={(dmax - dmin) / dmin:.3f}\")\n# d=2 -> ~10x contrast; d=1000 -> a few percent. Neighbors stop meaning anything." },
        { type: "callout", variant: "gotcha", text: "**The practical takeaway:** k-NN (and any distance-based method) degrades badly past a few dozen *informative* dimensions. Combat it by **reducing dimensionality first** (PCA, feature selection) or by using embeddings that concentrate signal into few axes. This is also *why* raw high-dimensional pixel/text data needs learned representations before nearest-neighbor search works — the whole point of a good embedding is to make distances meaningful again." },
        { type: "heading", text: "Making k-NN fast: kd-trees and ball-trees" },
        { type: "p", text: "Brute-force k-NN is $O(md)$ per query — a full scan of all $m$ points. For low-dimensional data you can do far better with a spatial **index** built once at fit time. A **kd-tree** recursively splits space with axis-aligned planes; a **ball-tree** partitions with nested hyperspheres. Both prune whole regions of points that can't contain a nearer neighbor, dropping queries toward $O(\\log m)$." },
        { type: "table",
          headers: ["Method", "Query cost", "Best regime"],
          rows: [
            ["Brute force", "$O(md)$", "small $m$, or very high $d$"],
            ["kd-tree", "$\\sim O(\\log m)$", "low $d$ (roughly $d \\lesssim 20$)"],
            ["ball-tree", "$\\sim O(\\log m)$", "moderate $d$, non-Euclidean metrics"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**Trees don't beat the curse.** kd-trees and ball-trees degrade back toward brute force as $d$ grows — above ~20 dimensions the pruning stops helping. scikit-learn's `algorithm='auto'` knows this and quietly falls back to brute force in high-D. For truly large, high-dimensional problems you move to **approximate** nearest neighbors (HNSW, FAISS, Annoy), which trade a little accuracy for enormous speed." },
      ]
    },

    {
      id: "knn-regression",
      title: "k-NN for regression, and when to use it",
      level: "core",
      body: [
        { type: "p", text: "k-NN isn't only a classifier. For **regression**, keep the same 'find the $k$ neighbors' machinery but replace the majority vote with an **average** of their target values:" },
        { type: "math", tex: String.raw`\hat y(x) = \frac{1}{k}\sum_{i \in N_k(x)} y_i \qquad\text{or, distance-weighted:}\qquad \hat y(x) = \frac{\sum_{i \in N_k(x)} w_i\, y_i}{\sum_{i \in N_k(x)} w_i}` },
        { type: "p", text: "This produces a piecewise-constant (or, when weighted, piecewise-smooth) fit that follows the local density of the data with zero assumptions about the functional form — no line, no curve, just 'what did nearby points do.' It's a fully **non-parametric** regressor." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef knn_regress(X_train, y_train, x, k=5, weighted=True):\n    d = np.sqrt(((X_train - x) ** 2).sum(axis=1))\n    nn = np.argsort(d)[:k]\n    if not weighted:\n        return y_train[nn].mean()\n    w = 1.0 / (d[nn] + 1e-9)\n    return np.sum(w * y_train[nn]) / np.sum(w)\n\n# Fit a noisy sine with purely local averaging — no model, no equation.\nrng = np.random.default_rng(0)\nX = np.sort(rng.uniform(0, 6, 80)).reshape(-1, 1)\ny = np.sin(X).ravel() + rng.normal(0, 0.1, 80)\npreds = [knn_regress(X, y, xi, k=7) for xi in X]\nprint(np.round(preds[:5], 3))   # tracks the sine wave locally" },
        { type: "heading", text: "Pros, cons, and when to reach for k-NN" },
        { type: "table",
          headers: ["Strengths", "Weaknesses"],
          rows: [
            ["No training; adapts instantly to new data", "Slow, memory-heavy at prediction (stores all data)"],
            ["No assumptions — any decision boundary shape", "Dies in high dimensions (curse of dimensionality)"],
            ["Naturally handles multi-class", "Demands feature scaling and clean data"],
            ["One hyperparameter ($k$), easy to reason about", "Sensitive to irrelevant features and noise"],
            ["Great for retrieval / recommendation", "No interpretable model or feature importances"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Reach for k-NN when:** the dataset is small-to-medium and low-dimensional, the decision boundary is irregular, you want a quick strong baseline, or the task *is* retrieval (find similar items). **Avoid it when:** you have many features, a huge dataset with tight latency budgets, or lots of irrelevant/unscaled features. It's the perfect baseline to beat — if a fancy model can't outperform tuned k-NN, question the fancy model." },
      ]
    },

    {
      id: "nb-derivation",
      title: "Naive Bayes: from Bayes' theorem to a classifier",
      level: "core",
      body: [
        { type: "p", text: "Now the opposite philosophy: instead of comparing to stored examples, build an explicit **probabilistic model** of each class and ask 'which class most probably generated this input?' The starting point is **Bayes' theorem** (derived on the Probability page), which inverts a conditional probability:" },
        { type: "math", tex: String.raw`P(y \mid x) = \frac{P(x \mid y)\,P(y)}{P(x)}` },
        { type: "p", text: "Read the pieces as a classifier. We want the **posterior** $P(y \\mid x)$ — the probability of class $y$ given the observed features $x$. It factors into the **prior** $P(y)$ (how common each class is), the **likelihood** $P(x \\mid y)$ (how probable these features are within class $y$), and the **evidence** $P(x)$ (a normalizer). To classify, we pick the class with the highest posterior — the **maximum a posteriori (MAP)** decision:" },
        { type: "math", tex: String.raw`\hat y = \operatorname*{arg\,max}_{y}\; P(y \mid x) = \operatorname*{arg\,max}_{y}\; \frac{P(x \mid y)\,P(y)}{P(x)}` },
        { type: "p", text: "The denominator $P(x)$ is the *same* for every class, so it can't change which class wins the arg-max. Drop it — we only need proportionality:" },
        { type: "math", tex: String.raw`\hat y = \operatorname*{arg\,max}_{y}\; P(x \mid y)\,P(y)` },
        { type: "heading", text: "The problem: the likelihood is unlearnable" },
        { type: "p", text: "For a $d$-feature input $x = (x_1, \\dots, x_d)$, the likelihood $P(x \\mid y) = P(x_1, x_2, \\dots, x_d \\mid y)$ is a full joint distribution over all features. With $d$ binary features that's $2^d - 1$ probabilities to estimate *per class* — hopeless for any real $d$. You'd never have enough data to estimate the interactions between all feature combinations." },
        { type: "heading", text: "The 'naive' assumption that rescues everything" },
        { type: "p", text: "Naive Bayes makes one bold, usually-false simplification: **given the class, all features are conditionally independent.** That means the joint likelihood factors into a product of one-feature likelihoods:" },
        { type: "math", tex: String.raw`P(x_1, \dots, x_d \mid y) \;\overset{\text{naive}}{=}\; \prod_{j=1}^{d} P(x_j \mid y)` },
        { type: "p", text: "This is the whole trick. Instead of one impossible $2^d$-way joint, we estimate $d$ tiny one-dimensional distributions per class — each just 'how does feature $j$ behave among class-$y$ examples?' Plug the factorization back in and you have the full **Naive Bayes classifier**:" },
        { type: "math", tex: String.raw`\boxed{\; \hat y = \operatorname*{arg\,max}_{y}\; P(y)\prod_{j=1}^{d} P(x_j \mid y) \;}` },
        { type: "callout", variant: "note", text: "**Why call it 'naive'?** Because conditional independence is almost always wrong. In spam detection, 'viagra' and 'cheap' co-occur; in medicine, symptoms correlate. The assumption ignores all of that. The astonishing part — covered two sections down — is that the classifier is *still* excellent, because it only needs to get the *ranking* of classes right, not the exact probabilities." },
      ]
    },

    {
      id: "nb-variants",
      title: "The three Naive Bayes variants & Laplace smoothing",
      level: "core",
      body: [
        { type: "p", text: "Naive Bayes needs a form for each per-feature likelihood $P(x_j \\mid y)$. The right choice depends on what the features *are* — continuous, counts, or binary — and gives the three standard variants." },
        { type: "heading", text: "Gaussian NB — for continuous features" },
        { type: "p", text: "When features are real-valued (height, temperature, sensor readings), model each feature within each class as a **Gaussian**. From the training data, estimate a mean $\\mu_{jy}$ and variance $\\sigma_{jy}^2$ for feature $j$ in class $y$, then evaluate the normal density:" },
        { type: "math", tex: String.raw`P(x_j \mid y) = \frac{1}{\sqrt{2\pi\sigma_{jy}^2}}\exp\!\left(-\frac{(x_j - \mu_{jy})^2}{2\sigma_{jy}^2}\right)` },
        { type: "heading", text: "Multinomial NB — for counts (the text workhorse)" },
        { type: "p", text: "When features are **counts** — how many times each word appears in a document (bag-of-words) — use the multinomial model. The likelihood of word $w$ in class $c$ is estimated as its share of all word occurrences in that class's documents:" },
        { type: "math", tex: String.raw`P(w \mid c) = \frac{\text{count}(w, c)}{\sum_{w' \in V}\text{count}(w', c)}` },
        { type: "p", text: "where $V$ is the vocabulary and $\\text{count}(w,c)$ is the total number of times word $w$ appears across all documents of class $c$. This is the default for text classification." },
        { type: "heading", text: "Bernoulli NB — for binary presence/absence" },
        { type: "p", text: "When features are **binary** — does the word appear at all, yes/no, ignoring how many times — use Bernoulli NB. It explicitly models *absence* too, so a word *not* appearing is itself evidence:" },
        { type: "math", tex: String.raw`P(x_j \mid y) = P(w_j \mid y)^{\,x_j}\,\bigl(1 - P(w_j \mid y)\bigr)^{\,1 - x_j}, \qquad x_j \in \{0,1\}` },
        { type: "table",
          headers: ["Variant", "Feature type", "Typical use"],
          rows: [
            ["Gaussian NB", "continuous real values", "sensor data, Iris, medical measurements"],
            ["Multinomial NB", "counts / frequencies", "text (bag-of-words, TF-IDF)"],
            ["Bernoulli NB", "binary 0/1 (presence)", "short text, spam with word-presence flags"],
          ]
        },
        { type: "heading", text: "Laplace smoothing — why zeros are lethal, and the fix" },
        { type: "p", text: "The product $\\prod_j P(x_j \\mid y)$ hides a landmine. Suppose the word 'blockchain' never appeared in any *ham* email in training. Then $P(\\text{blockchain} \\mid \\text{ham}) = 0$, and because we're multiplying, a **single** zero annihilates the entire product — the posterior for ham becomes exactly zero no matter how ham-like every other word is. One unseen word vetoes the whole decision. That's absurd: absence of evidence in a finite sample isn't proof of impossibility." },
        { type: "p", text: "The fix is **Laplace (add-one) smoothing**: pretend you saw every word one extra time. Add a pseudocount $\\alpha$ (usually $1$) to every numerator, and compensate in the denominator by adding $\\alpha |V|$ so the probabilities still sum to one:" },
        { type: "math", tex: String.raw`P(w \mid c) = \frac{\text{count}(w, c) + \alpha}{\sum_{w' \in V}\text{count}(w', c) + \alpha\,|V|}` },
        { type: "p", text: "Now no probability is ever exactly zero — an unseen word gets a small but nonzero probability instead of a veto. The $\\alpha |V|$ term in the denominator (one $\\alpha$ for each of the $|V|$ vocabulary words we bumped) keeps it a valid distribution. $\\alpha$ is a hyperparameter: $\\alpha = 1$ is classic Laplace, smaller values (Lidstone smoothing) smooth less." },
        { type: "callout", variant: "gotcha", text: "**Forgetting smoothing is the classic Naive Bayes bug.** Without it, your first test document containing any word unseen in a class gets that class's probability crushed to zero — often producing bizarre, confident-but-wrong predictions. scikit-learn smooths by default (`alpha=1.0`); if you build NB from scratch, add it from the start." },
      ]
    },

    {
      id: "nb-text-logspace",
      title: "Why Naive Bayes crushes text, and log-space math",
      level: "core",
      body: [
        { type: "p", text: "Naive Bayes has been a top text-classification baseline since the 1990s (spam filtering, sentiment, topic labeling) *despite* its independence assumption being flagrantly false in language. Why does a wrong model work so well?" },
        { type: "heading", text: "Wrong probabilities, right decisions" },
        { type: "p", text: "The key insight (Domingos & Pazzani, 1997): classification only needs the **arg-max** to be correct, not the probabilities themselves. Naive Bayes' independence assumption makes its estimated probabilities badly miscalibrated — often near $0$ or $1$ with false confidence — but the *ranking* of classes stays right across a huge region of parameter space. You can double-count correlated evidence ('viagra' and 'cheap' each pushing toward spam) and still land on the correct class; you just get there more emphatically than you should." },
        { type: "callout", variant: "good", text: "**Practical reasons NB is a great text baseline:** it trains in one pass over the data (fast even on millions of documents), needs little training data to work, handles tens of thousands of features (words) effortlessly, is trivial to update incrementally, and is genuinely interpretable — you can read off which words push toward each class. Always run it first; make the deep model justify its cost by beating it." },
        { type: "heading", text: "Log-space computation — or your probabilities underflow to zero" },
        { type: "p", text: "A document has hundreds of words, so $P(y)\\prod_j P(x_j \\mid y)$ multiplies hundreds of numbers each below $1$. The result underflows a 64-bit float to exactly $0.0$ — every class gets zero and the classifier is dead. The fix is universal in ML: **work in log-space.** Because $\\log$ is monotonic, the arg-max is unchanged, and the product becomes a numerically safe **sum**:" },
        { type: "math", tex: String.raw`\hat y = \operatorname*{arg\,max}_{y}\;\left[\, \log P(y) + \sum_{j=1}^{d} \log P(x_j \mid y) \,\right]` },
        { type: "p", text: "Adding a few hundred moderate negative numbers is perfectly stable, whereas multiplying a few hundred small positive numbers is not. This log-sum trick is the same one behind cross-entropy loss and log-likelihood everywhere in ML — get comfortable with it here." },
        { type: "callout", variant: "tip", text: "**If you need actual probabilities back** (not just the winning class), convert the log-scores with a numerically stable **softmax**: subtract the max log-score before exponentiating so nothing overflows. scikit-learn's `predict_proba` does exactly this internally." },
      ]
    },

    {
      id: "nb-scratch-sklearn",
      title: "Multinomial NB from scratch, then scikit-learn",
      level: "core",
      body: [
        { type: "p", text: "Let's build a Multinomial Naive Bayes text classifier end to end — with Laplace smoothing and log-space math — then show the scikit-learn one-liners for all the variants." },
        { type: "code", lang: "py", code: "import numpy as np\n\nclass MultinomialNB:\n    \"\"\"Naive Bayes for count features (bag-of-words). From scratch.\"\"\"\n    def __init__(self, alpha=1.0):\n        self.alpha = alpha           # Laplace smoothing pseudocount\n\n    def fit(self, X, y):\n        X = np.asarray(X, float)     # (n_docs, n_words) count matrix\n        self.classes = np.unique(y)\n        n_words = X.shape[1]\n        self.log_prior = {}          # log P(y)\n        self.log_lik = {}            # log P(word | y) for every word\n        for c in self.classes:\n            Xc = X[y == c]                                   # docs in class c\n            self.log_prior[c] = np.log(len(Xc) / len(X))\n            word_counts = Xc.sum(axis=0) + self.alpha        # +alpha smoothing\n            total = word_counts.sum()                        # includes alpha*|V|\n            self.log_lik[c] = np.log(word_counts / total)    # (n_words,)\n        return self\n\n    def predict(self, X):\n        X = np.asarray(X, float)\n        out = []\n        for x in X:\n            # log-space score per class: log P(y) + sum_j x_j * log P(w_j|y)\n            scores = {c: self.log_prior[c] + x @ self.log_lik[c]\n                      for c in self.classes}\n            out.append(max(scores, key=scores.get))\n        return np.array(out)\n\n# Tiny corpus already turned into word counts over a 5-word vocab.\n# vocab = [free, money, meeting, project, now]\nX = np.array([\n    [2, 2, 0, 0, 1],   # spam\n    [3, 1, 0, 0, 1],   # spam\n    [0, 0, 2, 1, 0],   # ham\n    [0, 0, 1, 2, 0],   # ham\n])\ny = np.array(['spam', 'spam', 'ham', 'ham'])\n\nnb = MultinomialNB(alpha=1.0).fit(X, y)\nprint(nb.predict([[1, 1, 0, 0, 1],    # 'free money now' -> spam\n                  [0, 0, 1, 1, 0]]))  # 'meeting project' -> ham" },
        { type: "callout", variant: "note", text: "**Note the dot product** `x @ self.log_lik[c]`: multiplying the word-count vector by the log-likelihood vector and summing *is* the term $\\sum_j x_j \\log P(w_j \\mid y)$ — each word contributes its count times its log-probability. This is the Linear Algebra page paying off: a whole probabilistic sum collapses into one dot product." },
        { type: "heading", text: "The scikit-learn way — text pipeline in a few lines" },
        { type: "p", text: "In practice you let scikit-learn vectorize the raw text and fit the model. `CountVectorizer` builds the bag-of-words count matrix; `MultinomialNB` does the rest, smoothing included by default." },
        { type: "code", lang: "py", code: "from sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.naive_bayes import MultinomialNB\nfrom sklearn.pipeline import make_pipeline\n\ndocs = [\"free money now\", \"win free cash\", \"cheap meds\",\n        \"project meeting tomorrow\", \"lunch schedule\", \"team sync notes\"]\nlabels = [\"spam\", \"spam\", \"spam\", \"ham\", \"ham\", \"ham\"]\n\n# CountVectorizer -> bag-of-words counts; MultinomialNB(alpha=1.0) smooths.\nclf = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))\nclf.fit(docs, labels)\n\nprint(clf.predict([\"free cash now\", \"meeting notes\"]))   # ['spam' 'ham']\nprint(clf.predict_proba([\"free cash now\"]).round(3))     # calibrated-ish probs" },
        { type: "heading", text: "The other variants, and k-NN, in one line each" },
        { type: "code", lang: "py", code: "from sklearn.naive_bayes import GaussianNB, BernoulliNB\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.datasets import load_iris\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\nX, y = load_iris(return_X_y=True)   # 4 continuous features -> GaussianNB\n\ngnb = GaussianNB()\nprint(\"GaussianNB :\", cross_val_score(gnb, X, y, cv=5).mean().round(3))\n\n# k-NN: ALWAYS scale first (distances are scale-sensitive). Use a pipeline.\nknn = make_pipeline(StandardScaler(),\n                    KNeighborsClassifier(n_neighbors=5, weights='distance'))\nprint(\"k-NN (k=5):\", cross_val_score(knn, X, y, cv=5).mean().round(3))\n\n# BernoulliNB expects binary features; shown here for API completeness.\n# bnb = BernoulliNB(alpha=1.0)  # for 0/1 word-presence matrices" },
        { type: "callout", variant: "gotcha", text: "**Wire scaling into a `Pipeline`, not by hand.** If you scale the whole dataset before splitting, statistics from the test set leak into training (data leakage) and your reported accuracy is optimistic. A `make_pipeline(StandardScaler(), KNeighborsClassifier())` refits the scaler on each CV fold's training data only — leak-free. This applies to any distance-based model." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Build at least two of these end to end. Reading the equations gives recognition; implementing them gives fluency. Everything below uses only NumPy and scikit-learn." },
        { type: "list", ordered: true, items: [
          "**Spam filter from scratch.** Grab the SMS Spam Collection or Enron email dataset, tokenize into a bag-of-words matrix by hand, and train your own Multinomial NB (with Laplace smoothing and log-space scoring). Report precision/recall/F1, inspect the highest-weight 'spammy' words, then confirm you match `sklearn.naive_bayes.MultinomialNB`.",
          "**Visualize the k-NN decision boundary.** On a 2-D toy dataset (`make_moons` or `make_blobs`), plot the classifier's decision regions for $k = 1, 5, 25, 101$. Watch the boundary go from jagged (overfit) to smooth to nearly constant — a live picture of the bias–variance trade-off.",
          "**Feel the curse of dimensionality.** Sample uniform points in $[0,1]^d$ for $d = 2, 10, 100, 1000$ and plot the ratio $(\\text{dist}_{\\max}-\\text{dist}_{\\min})/\\text{dist}_{\\min}$ against $d$. Then run k-NN classification on the same data as $d$ grows and watch accuracy decay toward chance as neighbors lose meaning.",
          "**Choose $k$ with cross-validation.** Sweep $k$ from 1 to 50 on a real dataset (Iris, Wine, or Breast Cancer), plot CV accuracy vs. $k$, and pick the peak. Compare uniform vs. distance weighting, and $L_1$ vs. $L_2$ vs. cosine metrics. Justify your final choice from the curves, not a heuristic.",
          "**Gaussian NB vs. logistic regression.** On a continuous-feature dataset, train both and compare accuracy *and* calibration (reliability curves). Confirm the folklore: Naive Bayes often wins with very little data, logistic regression overtakes it as data grows.",
          "**Ablate Laplace smoothing.** Turn smoothing off ($\\alpha \\to 0$) in your from-scratch NB and find a test document whose prediction gets destroyed by a single unseen word. Then sweep $\\alpha \\in \\{0.01, 0.1, 1, 10\\}$ and plot accuracy to see how much smoothing your data wants.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The two ideas here — instance-based and probabilistic learning — thread through the rest of ML. To go deeper, in recommended order:" },
        { type: "link", url: "https://www.statlearning.com/", text: "James, Witten, Hastie & Tibshirani — An Introduction to Statistical Learning (ISLR), free PDF: k-NN in ch. 2–4, the bias–variance trade-off explained cleanly" },
        { type: "link", url: "https://web.stanford.edu/~jurafsky/slp3/4.pdf", text: "Jurafsky & Martin — Speech and Language Processing, Ch. 4: Naive Bayes and sentiment classification (the definitive NB-for-text chapter, free PDF)" },
        { type: "link", url: "https://scikit-learn.org/stable/modules/naive_bayes.html", text: "scikit-learn User Guide — Naive Bayes (Gaussian/Multinomial/Bernoulli/Complement) with the exact formulas and API" },
        { type: "link", url: "https://scikit-learn.org/stable/modules/neighbors.html", text: "scikit-learn User Guide — Nearest Neighbors (kd-tree/ball-tree internals, metrics, and the algorithm='auto' logic)" },
        { type: "link", url: "https://link.springer.com/article/10.1023/A:1007413511361", text: "Domingos & Pazzani (1997) — On the Optimality of the Simple Bayesian Classifier: *why* the naive independence assumption still classifies well" },
        { type: "link", url: "https://www.cs.cornell.edu/johannes/papers/1999/icdt1999-onhighdim.pdf", text: "Beyer et al. (1999) — When Is 'Nearest Neighbor' Meaningful?: the formal treatment of the curse of dimensionality" },
      ]
    },
  ],

  packages: [
    { name: "numpy", why: "vectorized distances, count matrices, and log-space arithmetic — the from-scratch foundation" },
    { name: "sklearn.neighbors", why: "`KNeighborsClassifier` / `KNeighborsRegressor` with kd-tree/ball-tree indexing built in" },
    { name: "sklearn.naive_bayes", why: "`GaussianNB`, `MultinomialNB`, `BernoulliNB` — smoothing and log-space handled for you" },
    { name: "sklearn.feature_extraction.text", why: "`CountVectorizer` / `TfidfVectorizer` — turn raw text into the count matrices NB needs" },
    { name: "sklearn.preprocessing", why: "`StandardScaler` — mandatory before any distance-based model like k-NN" },
    { name: "sklearn.pipeline", why: "`make_pipeline` — chain scaler/vectorizer + model so cross-validation stays leak-free" },
  ],

  gotchas: [
    "k-NN **requires feature scaling**: an unscaled large-magnitude feature dominates the distance and silently becomes the only feature that matters.",
    "Scale/vectorize **inside a `Pipeline`**, not on the full dataset — otherwise test statistics leak into training and inflate your scores.",
    "k-NN dies in high dimensions: past a few dozen informative features, all distances converge and 'nearest' stops meaning anything. Reduce dimensionality first.",
    "kd-trees and ball-trees only help in low $d$; above ~20 dimensions they collapse back to brute force. Use approximate NN (FAISS/HNSW) for large high-D problems.",
    "Naive Bayes without **Laplace smoothing** lets a single unseen feature ($P=0$) zero out a whole class via the product. Always add $\\alpha \\ge 1$.",
    "Never multiply hundreds of probabilities directly — they underflow to $0.0$. Sum **logs** instead: $\\log P(y) + \\sum_j \\log P(x_j\\mid y)$.",
    "Match the NB variant to the data: Gaussian for continuous, Multinomial for counts, Bernoulli for binary presence. Wrong variant, worse results.",
    "Prefer an **odd $k$** for two-class k-NN so votes can't tie, and tune $k$ with cross-validation rather than a $\\sqrt{m}$ rule of thumb.",
  ],

  flashcards: [
    { q: "How does k-NN classify a new point, and why is it called a 'lazy' learner?", a: "Find the $k$ nearest training points and take a majority vote. It's lazy because `fit` just stores the data — all computation is deferred to prediction time." },
    { q: "What is the bias–variance effect of increasing $k$ in k-NN?", a: "Larger $k$ raises bias and lowers variance (smoother boundary); $k=1$ is low-bias/high-variance and memorizes noise; $k\\approx m$ predicts the majority class." },
    { q: "Why must you scale features before k-NN?", a: "Distance sums per-feature squared differences, so a large-magnitude feature dominates. Standardize or min-max scale first, or k-NN effectively ignores small-scale features." },
    { q: "State the curse of dimensionality for k-NN.", a: "As dimension $d$ grows, all pairwise distances converge — $(\\text{dist}_{\\max}-\\text{dist}_{\\min})/\\text{dist}_{\\min}\\to 0$ — so 'nearest' loses meaning and distance-based methods break down." },
    { q: "Write the Naive Bayes decision rule and name its key assumption.", a: "$\\hat y = \\arg\\max_y P(y)\\prod_j P(x_j\\mid y)$. The 'naive' assumption is conditional independence of features given the class, which factorizes the likelihood." },
    { q: "Why is Laplace smoothing necessary?", a: "A feature unseen in a class gives $P(x_j\\mid y)=0$, which zeros the whole product. Add-$\\alpha$ smoothing, $\\frac{\\text{count}+\\alpha}{\\text{total}+\\alpha|V|}$, keeps every probability nonzero." },
    { q: "Why compute Naive Bayes in log-space?", a: "Multiplying hundreds of sub-1 probabilities underflows to $0.0$. Since $\\log$ is monotonic, use $\\log P(y)+\\sum_j\\log P(x_j\\mid y)$ — a stable sum with the same arg-max." },
    { q: "When do you use Gaussian vs. Multinomial vs. Bernoulli NB?", a: "Gaussian for continuous features, Multinomial for count features (bag-of-words), Bernoulli for binary presence/absence features." },
    { q: "Why is Naive Bayes a strong text baseline despite the false independence assumption?", a: "Classification needs only the correct arg-max, not exact probabilities. The class ranking stays right even when probabilities are miscalibrated, and it trains in one fast pass." },
    { q: "How does k-NN do regression instead of classification?", a: "Replace the majority vote with the (optionally distance-weighted) average of the $k$ neighbors' target values — a non-parametric, piecewise fit." },
  ],

  cheatsheet: [
    { label: "k-NN classify", code: "KNeighborsClassifier(n_neighbors=5)" },
    { label: "Distance-weighted", code: "KNeighborsClassifier(weights='distance')" },
    { label: "k-NN regress", code: "KNeighborsRegressor(n_neighbors=5)" },
    { label: "Scale then k-NN", code: "make_pipeline(StandardScaler(), KNeighborsClassifier())" },
    { label: "Euclidean dist (vec)", code: "np.sqrt(((X - x)**2).sum(axis=1))" },
    { label: "k nearest indices", code: "np.argsort(d)[:k]" },
    { label: "Gaussian NB", code: "GaussianNB()" },
    { label: "Multinomial NB", code: "MultinomialNB(alpha=1.0)" },
    { label: "Bernoulli NB", code: "BernoulliNB(alpha=1.0)" },
    { label: "Text -> counts", code: "CountVectorizer().fit_transform(docs)" },
    { label: "NB text pipeline", code: "make_pipeline(CountVectorizer(), MultinomialNB())" },
    { label: "Log-space score", code: "log_prior[c] + x @ log_lik[c]" },
    { label: "Cross-validate", code: "cross_val_score(clf, X, y, cv=5).mean()" },
  ],
});
