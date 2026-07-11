(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "ml-dimensionality",
  name: "Dimensionality Reduction",
  language: "Classical ML",
  group: "Classical Machine Learning",
  navLabel: "Dimensionality Reduction",
  tagline: "Squeeze high-dimensional data down to a few honest axes — PCA derived from scratch, then t-SNE, UMAP, and autoencoders for the non-linear world.",
  color: "#065F46",
  readMinutes: 48,
  sections: [
    {
      id: "why",
      title: "Why reduce dimensions at all",
      level: "core",
      body: [
        { type: "p", text: "Real datasets are wide. A single face photo is $\\sim 10^4$ pixels; a gene-expression sample is $\\sim 2\\times10^4$ genes; a bag-of-words document lives in a vocabulary-sized space. **Dimensionality reduction** finds a small set of new coordinates that keep most of what matters and throw away the rest. You do it for five concrete reasons." },
        { type: "table",
          headers: ["Reason", "What it buys you", "Typical tool"],
          rows: [
            ["**Curse of dimensionality**", "distances stop being meaningful in high $d$; fewer dims restore signal", "PCA"],
            ["**Visualization**", "you can only plot 2–3 axes; project $d \\to 2$ to see structure", "t-SNE / UMAP"],
            ["**Compression**", "store $k \\ll d$ numbers per example instead of $d$", "PCA / autoencoder"],
            ["**Denoising**", "drop low-variance directions that are mostly noise", "PCA truncation"],
            ["**Speed**", "downstream models train faster on fewer features", "PCA before a classifier"],
          ]
        },
        { type: "heading", text: "The curse of dimensionality, concretely" },
        { type: "p", text: "As $d$ grows, volume explodes and data becomes *sparse*: every point drifts toward the surface of the space and roughly equidistant from every other point. The ratio of nearest to farthest distance approaches $1$, so $k$-NN, clustering, and anything distance-based quietly degrade." },
        { type: "math", tex: String.raw`\lim_{d\to\infty} \frac{\operatorname{dist}_{\max} - \operatorname{dist}_{\min}}{\operatorname{dist}_{\min}} \longrightarrow 0` },
        { type: "callout", variant: "note", text: "**The saving grace: the manifold hypothesis.** Real high-dimensional data (faces, digits, sentences) does not fill its space — it clusters on a low-dimensional curved surface (a *manifold*) sitting inside it. All of dimensionality reduction is a bet that this manifold exists and can be flattened. PCA flattens it *linearly*; t-SNE, UMAP, and autoencoders flatten it *non-linearly*." },
        { type: "callout", variant: "note", text: "**How to read this page.** We derive PCA fully — twice, from two objectives that turn out to be the same problem — then implement it from scratch two ways and check them against scikit-learn. Then we move to the non-linear methods (Kernel PCA, t-SNE, UMAP, autoencoders), which are for *seeing* data, not for feeding a model. The linear-algebra machinery (eigenvectors, SVD, covariance) lives in the **Linear Algebra** page; we lean on it constantly." },
      ]
    },

    {
      id: "pca-derivation",
      title: "PCA, fully derived",
      level: "core",
      body: [
        { type: "p", text: "**Principal Component Analysis** (Pearson 1901, Hotelling 1933) finds the orthogonal directions along which your data varies the most, and uses them as new axes. Project onto the top few and you keep the bulk of the variation in a handful of coordinates. It is the single most-used dimensionality reduction method, and it is *entirely* an eigenvalue problem." },
        { type: "heading", text: "Step 0 — center the data (this is not optional)" },
        { type: "p", text: "PCA measures *variance about the mean*, so the mean must be at the origin first. Subtract the column mean from every row. (Uncentered 'PCA' silently finds the direction toward the data cloud instead of its axis of spread — a classic bug.)" },
        { type: "math", tex: String.raw`\tilde{x}^{(i)} = x^{(i)} - \mu, \qquad \mu = \frac{1}{m}\sum_{i=1}^{m} x^{(i)}` },
        { type: "heading", text: "Objective A — maximize projected variance" },
        { type: "p", text: "We want a unit direction $w$ (with $\\|w\\|=1$) such that the projections $z_i = w^\\top \\tilde{x}^{(i)}$ of the centered data spread out as much as possible. Since the data is centered the projections have mean zero, so their variance is just their mean square:" },
        { type: "math", tex: String.raw`\operatorname{Var}(z) = \frac{1}{m}\sum_{i=1}^{m} \bigl(w^\top \tilde{x}^{(i)}\bigr)^2 = w^\top \!\left(\frac{1}{m}\sum_i \tilde{x}^{(i)} \tilde{x}^{(i)\top}\right)\! w = w^\top C\, w` },
        { type: "p", text: "where $C = \\frac{1}{m}\\tilde{X}^\\top \\tilde{X}$ is the **covariance matrix** (symmetric, positive semidefinite). We maximize $w^\\top C w$ subject to $w^\\top w = 1$. Form the Lagrangian and differentiate:" },
        { type: "math", tex: String.raw`\mathcal{L}(w,\lambda) = w^\top C w - \lambda\,(w^\top w - 1), \qquad \nabla_w \mathcal{L} = 2Cw - 2\lambda w \overset{!}{=} 0` },
        { type: "math", tex: String.raw`\boxed{\; C w = \lambda w \;}` },
        { type: "p", text: "The stationarity condition is *literally the eigenvector equation*. The maximizing $w$ is the eigenvector of $C$ with the **largest eigenvalue**, and the variance captured along it is that eigenvalue: substituting back, $w^\\top C w = w^\\top(\\lambda w) = \\lambda$. The second principal component is the next eigenvector (orthogonal to the first, by the spectral theorem for symmetric matrices), and so on down the spectrum." },
        { type: "callout", variant: "tip", text: "That $C$ is symmetric is exactly why this works cleanly: the **spectral theorem** (see the Linear Algebra page) guarantees real, non-negative eigenvalues and a full set of *orthogonal* eigenvectors — so your new axes are automatically perpendicular, i.e. an honest rotation of the original space." },
        { type: "heading", text: "Objective B — minimize reconstruction error" },
        { type: "p", text: "Here is the second story. Suppose instead we want the best $k$-dimensional *linear subspace* to approximate the data: an orthonormal basis $W_k = [w_1, \\dots, w_k]$ so that projecting each point in and back out loses as little as possible. Minimize squared reconstruction error:" },
        { type: "math", tex: String.raw`\min_{W_k^\top W_k = I}\; \frac{1}{m}\sum_{i=1}^{m} \bigl\| \tilde{x}^{(i)} - W_k W_k^\top \tilde{x}^{(i)} \bigr\|_2^2` },
        { type: "p", text: "Expand the norm using $W_k^\\top W_k = I$. The cross terms give $\\|\\tilde{x}\\|^2 - \\|W_k^\\top \\tilde{x}\\|^2$, and the first term is constant, so **minimizing reconstruction error is identical to maximizing the projected variance** $\\sum_i \\|W_k^\\top \\tilde{x}^{(i)}\\|^2$:" },
        { type: "math", tex: String.raw`\underbrace{\min\; \tfrac{1}{m}\textstyle\sum_i \|\tilde{x}^{(i)}\|^2}_{\text{const}} \; - \; \underbrace{\max\; \tfrac{1}{m}\textstyle\sum_i \|W_k^\top \tilde{x}^{(i)}\|^2}_{\text{Objective A}}` },
        { type: "callout", variant: "good", text: "**The two objectives are the same problem.** 'Keep the most variance' and 'lose the least when you compress and decompress' lead to the *same* eigenvectors of the *same* covariance matrix. This equivalence is why PCA feels canonical — it is optimal under two different, natural definitions of 'best'." },
        { type: "heading", text: "Explained-variance ratio — how much did you keep?" },
        { type: "p", text: "Because the eigenvalues *are* the variances along each component, the fraction of total variance captured by the top $k$ components is a simple ratio of eigenvalues. This is your single most important diagnostic." },
        { type: "math", tex: String.raw`\text{explained variance ratio}(k) = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{d} \lambda_j}` },
        { type: "callout", variant: "note", text: "$\\operatorname{tr}(C) = \\sum_j \\lambda_j$ is the **total variance** of the dataset — the trace equals the sum of eigenvalues (Linear Algebra page). PCA redistributes that fixed budget onto orthogonal axes and lets you keep the biggest slices." },
      ]
    },

    {
      id: "pca-svd",
      title: "PCA via SVD, and PCA from scratch",
      level: "core",
      body: [
        { type: "p", text: "You *can* form the covariance matrix $C = \\frac{1}{m}\\tilde{X}^\\top\\tilde{X}$ and eigendecompose it — and we will, to prove the concept. But in production nobody forms $C$. Squaring $\\tilde{X}$ into $\\tilde{X}^\\top\\tilde{X}$ **squares the condition number**, throwing away numerical precision. The numerically right way is to take the SVD of the centered data directly." },
        { type: "heading", text: "Why the SVD gives you PCA for free" },
        { type: "p", text: "Take the SVD of the centered data $\\tilde{X} = U\\Sigma V^\\top$. Then:" },
        { type: "math", tex: String.raw`\tilde{X}^\top \tilde{X} = V\Sigma^\top U^\top U \Sigma V^\top = V\,\Sigma^2\,V^\top` },
        { type: "p", text: "That is exactly the eigendecomposition of $m\\,C$. So the **right-singular vectors $V$ are the principal components**, and the eigenvalues of the covariance are $\\lambda_i = \\sigma_i^2 / m$ (or $/(m-1)$ for the unbiased estimate). The low-dimensional scores are $Z = \\tilde{X}V_k = U_k\\Sigma_k$. No covariance matrix ever gets built." },
        { type: "callout", variant: "gotcha", text: "This is the same $\\sigma_i^2/(m-1) = \\lambda_i$ identity noted on the Linear Algebra SVD page. If you ever see PCA and SVD described as 'related,' they are not merely related — running PCA *is* running an SVD of centered data." },
        { type: "heading", text: "From scratch, both ways, verified equal" },
        { type: "code", lang: "py", code: "import numpy as np\n\nrng = np.random.default_rng(0)\n# 200 points, correlated in 3D so the intrinsic spread is really ~2D\nA = np.array([[3, 1, 0.5],\n              [1, 2, 0.2],\n              [0.5, 0.2, 0.4]])\nX = rng.standard_normal((200, 3)) @ A          # shape (m=200, d=3)\n\n# ---- center (always) ----\nmu = X.mean(axis=0)\nXc = X - mu\nm = Xc.shape[0]\n\n# ---- Route 1: eigendecomposition of the covariance matrix ----\nC = (Xc.T @ Xc) / (m - 1)                       # (d, d) covariance\neigvals, eigvecs = np.linalg.eigh(C)            # ascending; eigh for symmetric\norder = np.argsort(eigvals)[::-1]               # sort descending\neigvals, eigvecs = eigvals[order], eigvecs[:, order]\ncomponents_eig = eigvecs.T                      # rows = principal directions\n\n# ---- Route 2: SVD of the centered data (the right way) ----\nU, s, Vt = np.linalg.svd(Xc, full_matrices=False)\ncomponents_svd = Vt                             # rows = principal directions\nsvd_var = s**2 / (m - 1)                        # eigenvalues of covariance\n\n# ---- they agree (up to a sign flip per component) ----\nfor a, b in zip(components_eig, components_svd):\n    if np.dot(a, b) < 0: b = -b                 # sign is arbitrary in PCA\n    assert np.allclose(a, b, atol=1e-6)\nprint('eigenvalues (eig):', np.round(eigvals, 4))\nprint('eigenvalues (svd):', np.round(svd_var, 4))\nprint('explained ratio :', np.round(eigvals / eigvals.sum(), 3))\n\n# ---- project to k=2 dimensions ----\nk = 2\nZ = Xc @ components_svd[:k].T                   # scores, shape (200, 2)\nX_recon = Z @ components_svd[:k] + mu           # reconstruct back to 3D\nprint('reconstruction MSE:', np.mean((X - X_recon)**2))" },
        { type: "callout", variant: "gotcha", text: "**Signs are arbitrary.** An eigenvector $w$ and $-w$ describe the same axis, so PCA outputs from two libraries (or `eig` vs `svd`) can differ by a per-component sign flip. That is *not* a bug — never compare PCA components without allowing for it, as the assertion above does." },
        { type: "heading", text: "The scikit-learn one-liner (which does exactly this)" },
        { type: "code", lang: "py", code: "from sklearn.decomposition import PCA\n\npca = PCA(n_components=2)         # uses SVD of centered data internally\nZ = pca.fit_transform(X)         # centers for you, returns scores (200, 2)\n\nprint(pca.components_)                     # the principal directions (rows)\nprint(pca.explained_variance_)             # the eigenvalues lambda_i\nprint(pca.explained_variance_ratio_)       # lambda_i / sum(lambda)\nprint(pca.explained_variance_ratio_.cumsum())\nX_back = pca.inverse_transform(Z)          # decompress back to 3D" },
        { type: "callout", variant: "tip", text: "`PCA` centers the data automatically (it stores `pca.mean_`), but it does **not** scale it. Whether to scale is a real decision — see the next section." },
      ]
    },

    {
      id: "choosing-whitening-pitfalls",
      title: "Choosing k, whitening & the pitfalls",
      level: "core",
      body: [
        { type: "heading", text: "How many components to keep" },
        { type: "p", text: "Three standard rules, in rough order of rigor:" },
        { type: "list", ordered: false, items: [
          "**Variance threshold.** Keep the smallest $k$ whose cumulative explained variance clears a target (0.90–0.99). One line: `np.searchsorted(np.cumsum(ratio), 0.95) + 1`.",
          "**The scree / elbow.** Plot eigenvalues in descending order and cut where the curve flattens into the 'noise floor.' Subjective but often obvious.",
          "**Downstream metric.** If PCA feeds a classifier, treat $k$ as a hyperparameter and cross-validate it against the actual task score — the only rule that optimizes what you care about.",
        ]},
        { type: "code", lang: "py", code: "# Let PCA pick k for you by naming a variance target instead of a count.\nfrom sklearn.decomposition import PCA\npca = PCA(n_components=0.95)     # keep just enough PCs for 95% variance\npca.fit(X)\nprint('kept', pca.n_components_, 'of', X.shape[1], 'dimensions')" },
        { type: "heading", text: "Whitening" },
        { type: "p", text: "**Whitening** rescales each principal component to unit variance after projecting — dividing the scores by $\\sqrt{\\lambda_i}$. The output has an identity covariance: decorrelated *and* equal-variance in every direction." },
        { type: "math", tex: String.raw`z_{\text{white}} = \Lambda_k^{-1/2} W_k^\top \tilde{x} = \sqrt{m-1}\,\Sigma_k^{-1} V_k^\top \tilde{x}` },
        { type: "callout", variant: "tip", text: "Whitening helps downstream models that assume isotropic features (some clustering, older CNN preprocessing pipelines) and is `PCA(whiten=True)` in scikit-learn. But it **amplifies the low-variance components' noise** by dividing by a small $\\sqrt{\\lambda_i}$ — only whiten when the next model actually benefits." },
        { type: "heading", text: "Three pitfalls that bite everyone" },
        { type: "callout", variant: "gotcha", text: "**1. You must scale first (usually).** PCA maximizes variance, and variance depends on units. A feature measured in millimeters will dominate one measured in meters purely because its numbers are bigger. Unless your features share a natural unit (pixels, standardized log-returns), run `StandardScaler` *before* PCA. This single mistake produces more garbage PCA plots than any other." },
        { type: "callout", variant: "gotcha", text: "**2. PCA is linear.** It can only rotate and project — it finds *flat* subspaces. If your manifold is curved (a spiral, a Swiss roll, the surface of digits), PCA will smear it. That is exactly what Kernel PCA, t-SNE, UMAP, and autoencoders exist to fix." },
        { type: "callout", variant: "gotcha", text: "**3. Components are hard to interpret.** A principal component is a linear blend of *all* original features (e.g. $0.4\\,\\text{age} - 0.7\\,\\text{income} + \\dots$). It is a direction of variance, not a meaningful concept. If you need human-interpretable factors, reach for **sparse PCA**, **NMF** (non-negative parts-based factors), or a domain model instead." },
        { type: "code", lang: "py", code: "# The correct PCA pipeline: scale, then reduce — as one object.\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\npipe = make_pipeline(StandardScaler(), PCA(n_components=0.95))\nZ = pipe.fit_transform(X)     # scaling is fit on train stats, not leaked" },
      ]
    },

    {
      id: "kernel-pca",
      title: "Kernel PCA — going non-linear with the kernel trick",
      level: "deep",
      body: [
        { type: "p", text: "The cheapest way past PCA's linearity is the **kernel trick** (the same one that powers SVMs — see the SVM page). Instead of running PCA in the input space, imagine mapping every point through a non-linear feature map $\\phi(x)$ into a huge (possibly infinite) space, then doing linear PCA *there*. Curved structure in the input becomes linear in feature space." },
        { type: "p", text: "The trick is that you never compute $\\phi(x)$ explicitly. PCA only needs inner products, and a **kernel** gives them directly: $k(x, x') = \\phi(x)^\\top \\phi(x')$. Build the $m\\times m$ kernel matrix $K$, center it *in feature space*, and eigendecompose it — that is the whole algorithm." },
        { type: "math", tex: String.raw`K_{ij} = k(x^{(i)}, x^{(j)}), \qquad \tilde{K} = K - \mathbf{1}_m K - K\mathbf{1}_m + \mathbf{1}_m K \mathbf{1}_m, \qquad \tilde{K}\,\alpha = m\lambda\,\alpha` },
        { type: "p", text: "The common kernel is the RBF (Gaussian) $k(x,x') = \\exp(-\\gamma\\|x-x'\\|^2)$, the same kernel you meet with SVMs." },
        { type: "code", lang: "py", code: "from sklearn.decomposition import KernelPCA\nfrom sklearn.datasets import make_moons\n\nX, y = make_moons(n_samples=300, noise=0.05, random_state=0)\n# Two interleaving crescents: NOT linearly separable, so linear PCA fails.\nkpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)\nZ = kpca.fit_transform(X)     # unfolds the moons into two clean blobs" },
        { type: "callout", variant: "gotcha", text: "Kernel PCA scales as $O(m^2)$ in memory and $O(m^3)$ for the eigendecomposition (the kernel matrix is $m\\times m$), so it does not fit large datasets. It is also acutely sensitive to $\\gamma$. For big-data non-linear reduction, people reach for t-SNE, UMAP, or autoencoders instead." },
      ]
    },

    {
      id: "tsne",
      title: "t-SNE — seeing local neighborhoods",
      level: "core",
      body: [
        { type: "p", text: "**t-SNE** (t-distributed Stochastic Neighbor Embedding, van der Maaten & Hinton, 2008) is the method behind those beautiful colored blobs of MNIST digits or word embeddings. Its one job: take high-dimensional points and lay them out in 2D so that **points that were neighbors stay neighbors**. It is a visualization tool, full stop." },
        { type: "heading", text: "The idea: match neighbor probability distributions" },
        { type: "p", text: "In the high-dimensional space, convert distances into a probability that $i$ would pick $j$ as a neighbor — a Gaussian around each point. In the 2D map, define an analogous probability $q_{ij}$, but using a **heavy-tailed Student-t** (Cauchy) distribution. Then move the 2D points until the two distributions match." },
        { type: "math", tex: String.raw`p_{j\mid i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k\neq i}\exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}, \qquad q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k\neq l}(1 + \|y_k - y_l\|^2)^{-1}}` },
        { type: "p", text: "The conditional $p_{j\\mid i}$ is first symmetrized into a joint distribution $p_{ij} = (p_{j\\mid i} + p_{i\\mid j}) / (2m)$ (so it matches the joint $q_{ij}$ and no point is ignored because of a tiny $\\sigma_i$). 'Match' then means minimize the **Kullback–Leibler divergence** between the high-D neighbor distribution $P$ and the low-D one $Q$, by gradient descent on the 2D coordinates $y_i$:" },
        { type: "math", tex: String.raw`\mathcal{L} = \mathrm{KL}(P \,\|\, Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}` },
        { type: "callout", variant: "note", text: "**Why the Student-t in 2D?** KL divergence punishes putting *near* points far apart far more than *far* points near — so t-SNE preserves local structure. The heavy tail gives distant points a little room in the cramped 2D map, which fixes the 'crowding problem' and is where the *t* comes from." },
        { type: "heading", text: "Perplexity — the one knob that matters" },
        { type: "p", text: "Each point's Gaussian width $\\sigma_i$ is set so the effective number of neighbors equals a target called **perplexity** (roughly, 'how many neighbors should I care about'). Typical range 5–50. Small perplexity sees tiny local clusters; large perplexity blends them into broader structure." },
        { type: "callout", variant: "gotcha", text: "**Do not read anything into t-SNE except *which points are near which*.** Cluster *sizes* are meaningless (t-SNE equalizes dense and sparse regions). Distances *between* clusters are meaningless. Global layout is meaningless. And it is stochastic — re-running gives a different picture. Above all: **never cluster or classify on t-SNE coordinates.** It is a lossy, non-parametric *picture*, not a feature space." },
        { type: "callout", variant: "gotcha", text: "t-SNE has no `transform` for new points — it is not a reusable mapping, it re-optimizes every time. And it is $O(m^2)$ naively (Barnes–Hut brings it to $O(m\\log m)$). For > ~10k points, reduce to ~50 dims with PCA first, then t-SNE — the standard pipeline." },
      ]
    },

    {
      id: "umap",
      title: "UMAP — faster, and keeps more of the big picture",
      level: "core",
      body: [
        { type: "p", text: "**UMAP** (Uniform Manifold Approximation and Projection, McInnes & Healy, 2018) is the modern default for embedding visualization. Same goal as t-SNE — a 2D map that preserves neighborhoods — but built on a graph/topology foundation instead of probability matching, and it is dramatically faster with better-preserved *global* structure." },
        { type: "p", text: "UMAP builds a weighted $k$-nearest-neighbor graph in high-D (a fuzzy simplicial set), then optimizes a low-D layout whose graph matches it, using a cross-entropy loss with attractive forces on edges and repulsive forces via negative sampling — the same trick word2vec uses, which is what makes it scale." },
        { type: "table",
          headers: ["", "t-SNE", "UMAP"],
          rows: [
            ["Speed", "slow; $O(m\\log m)$ Barnes–Hut", "much faster; scales to millions"],
            ["Global structure", "largely destroyed", "better preserved (cluster spacing more meaningful)"],
            ["Reusable mapping", "no `transform`", "**yes** — `transform` embeds new points"],
            ["Main knobs", "`perplexity`", "`n_neighbors`, `min_dist`"],
            ["Can embed to > 2D usefully", "rarely", "yes (usable as features, with care)"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Prefer UMAP when** you have many points (it is far faster), you want to embed new data with a saved model (`transform`), or you care about the relative arrangement of clusters. **Reach for t-SNE when** you want the tightest possible separation of tiny local clusters for a single static figure. In 2024–2026 practice UMAP is the more common default." },
        { type: "callout", variant: "gotcha", text: "UMAP preserves *more* global structure than t-SNE, but 'more' is not 'faithfully.' Inter-cluster distances are still only loosely meaningful, and `min_dist`/`n_neighbors` visibly reshape the plot. The same discipline applies: it is for *seeing*, and clustering on UMAP output is defensible only with great care and validation on the original space." },
        { type: "code", lang: "py", code: "# pip install umap-learn  (import name is 'umap')\nimport umap\n\nreducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,\n                    random_state=42)\nZ = reducer.fit_transform(X)         # (m, 2) embedding\nZ_new = reducer.transform(X_test)    # <-- t-SNE cannot do this" },
      ]
    },

    {
      id: "autoencoders",
      title: "Autoencoders — dimensionality reduction by neural network",
      level: "deep",
      body: [
        { type: "p", text: "An **autoencoder** is a neural network trained to copy its input to its output through a narrow middle layer — the **bottleneck** (or *code*, or *latent*). Because information must squeeze through $k \\ll d$ units, the network is forced to learn a compressed representation. It is dimensionality reduction where the encoder and decoder are learned, non-linear functions." },
        { type: "math", tex: String.raw`z = f_{\text{enc}}(x) \in \mathbb{R}^k, \qquad \hat{x} = g_{\text{dec}}(z), \qquad \min_{\theta}\; \frac{1}{m}\sum_i \|x^{(i)} - \hat{x}^{(i)}\|_2^2` },
        { type: "callout", variant: "note", text: "**PCA is a linear autoencoder.** If the encoder and decoder are single linear layers with no activation and you minimize squared reconstruction error, the optimal solution spans exactly the top-$k$ PCA subspace. Add non-linear activations and hidden layers and you get a *non-linear* PCA that can follow curved manifolds — the reconstruction-error objective from earlier, made deep." },
        { type: "p", text: "The full mechanics — layers, activations, backprop, training loops — live in the **Neural Networks** track. The point to carry there: the bottleneck $z$ is a learned, non-linear, low-dimensional coordinate system, and unlike t-SNE/UMAP it gives you a reusable encoder *and* a decoder (so you can generate/reconstruct). **Variational autoencoders (VAEs)** turn that latent space into a smooth probability distribution you can sample from — the bridge to generative models." },
        { type: "callout", variant: "tip", text: "**Which to use?** PCA for a fast linear baseline and honest variance accounting. t-SNE/UMAP for a 2D *picture*. Autoencoders when you have lots of data, need a non-linear *and reusable* encoder, or want to feed the latent code into a downstream model or generator." },
      ]
    },

    {
      id: "worked-examples",
      title: "Worked examples: MNIST & Iris end to end",
      level: "core",
      body: [
        { type: "heading", text: "PCA on MNIST with a variance-explained plot" },
        { type: "code", lang: "py", code: "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import fetch_openml\nfrom sklearn.decomposition import PCA\n\nX, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)\nX = X / 255.0                      # pixels already share a unit (0..1); no StandardScaler\n\npca = PCA().fit(X)                # all 784 components, to see the spectrum\ncum = np.cumsum(pca.explained_variance_ratio_)\nk95 = np.searchsorted(cum, 0.95) + 1\nprint(f'{k95} components explain 95% of variance (down from 784)')\n\n# The variance-explained curve: a steeply rising elbow that flattens.\nplt.plot(cum)\nplt.axhline(0.95, ls='--'); plt.axvline(k95, ls='--')\nplt.xlabel('number of components'); plt.ylabel('cumulative explained variance')\nplt.title('MNIST PCA scree — most signal in the first ~150 dims')\n# Reading the plot: the curve climbs fast then bends into a shallow tail;\n# the elbow near ~150 marks where extra components buy mostly noise." },
        { type: "callout", variant: "note", text: "On MNIST the cumulative-variance curve rises steeply and reaches ~0.95 by roughly 150 of the 784 dimensions — a >5x compression that keeps almost all the variance. That elbow shape is the visual signature of data living on a low-dimensional manifold." },
        { type: "heading", text: "t-SNE vs UMAP on the same digits" },
        { type: "code", lang: "py", code: "from sklearn.manifold import TSNE\nfrom sklearn.decomposition import PCA\nimport umap\n\n# Standard pipeline: PCA to 50 dims first (denoise + 15x speedup), then embed.\nsub = slice(0, 10000)                          # subsample for t-SNE's O(n log n)\nX50 = PCA(n_components=50).fit_transform(X[sub])\n\nZ_tsne = TSNE(n_components=2, perplexity=30, init='pca',\n              random_state=0).fit_transform(X50)\nZ_umap = umap.UMAP(n_neighbors=15, min_dist=0.1,\n                   random_state=0).fit_transform(X50)\n\n# Colour each point by its digit label y[sub]: both plots show ten islands,\n# one per digit. UMAP packs them tighter with more meaningful gaps between\n# related digits (4/9/7 sit near each other); t-SNE spreads them more evenly.\nfor Z, name in [(Z_tsne, 't-SNE'), (Z_umap, 'UMAP')]:\n    plt.figure(); plt.scatter(Z[:,0], Z[:,1], c=y[sub].astype(int),\n                              cmap='tab10', s=3); plt.title(name)" },
        { type: "heading", text: "Iris — the tiny sanity-check dataset" },
        { type: "code", lang: "py", code: "from sklearn.datasets import load_iris\nfrom sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\n\nX, y = load_iris(return_X_y=True)          # 150 flowers, 4 features\nXs = StandardScaler().fit_transform(X)\npca = PCA(n_components=2).fit(Xs)\nZ = pca.transform(Xs)\nprint(pca.explained_variance_ratio_)       # ~[0.73, 0.23] -> 2 PCs ~ 96%\n# Plotting Z coloured by species: setosa splits off cleanly on PC1;\n# versicolor & virginica overlap slightly. 4D -> 2D with ~96% variance kept." },
        { type: "callout", variant: "tip", text: "**A reliable habit:** always run PCA first — even before t-SNE/UMAP. It denoises, it speeds up the non-linear method 10–20x, and its explained-variance ratio instantly tells you how compressible your data even is. If PCA already separates your classes in 2D, you may not need anything fancier." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Reading these methods builds recognition; implementing them builds intuition for when each one lies to you. Do at least two, and in each case look at the *reconstructions* or the *original-space neighbors*, not just the pretty plot." },
        { type: "list", ordered: true, items: [
          "**Eigenfaces.** Load the LFW or Olivetti faces dataset, run PCA, and *look at the top components reshaped back into face images* — they are ghostly 'eigenfaces,' directions of facial variation. Reconstruct a face from $k = 10, 50, 200$ components and watch identity emerge. Then build a nearest-neighbor face classifier in the PCA space and compare accuracy vs. $k$.",
          "**PCA from scratch, verified two ways.** Implement PCA via `eigh` of the covariance *and* via `svd` of the centered data. Assert the components match up to sign and the eigenvalues equal $\\sigma_i^2/(m-1)$. Add a collinear feature and confirm the SVD route stays stable while the covariance route degrades.",
          "**Scree-plot decision tool.** Write a helper that takes any dataset, standardizes it, plots cumulative explained variance, and reports the $k$ needed for 90/95/99%. Run it on MNIST, Iris, and a gene-expression set and compare how compressible each is.",
          "**Visualize embeddings.** Take word2vec/GloVe vectors (or your own model's embeddings) and project them to 2D with both t-SNE and UMAP. Verify that semantically related words land together. Then deliberately mislead yourself: point to two far-apart clusters and confirm you *cannot* trust the distance between them.",
          "**t-SNE knob sweep.** Embed the same data at perplexity 5, 30, and 100, and side by side. Watch clusters fragment then merge. Internalize that the 'right' number of clusters in a t-SNE plot is partly an artifact of one hyperparameter.",
          "**Linear autoencoder = PCA.** In PyTorch, build a one-hidden-layer autoencoder with *no* activation and MSE loss. Train it, then show its bottleneck weights span the same subspace as `PCA(n_components=k)`. Add a ReLU and hidden layers and beat PCA's reconstruction error on a curved (Swiss-roll) dataset.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The best places to go deeper, in recommended order — from the linear foundation to the modern non-linear methods:" },
        { type: "link", url: "https://scikit-learn.org/stable/modules/decomposition.html", text: "scikit-learn — Decomposition user guide (PCA, KernelPCA, and friends, with the exact API you'll use)" },
        { type: "link", url: "https://distill.pub/2016/misread-tsne/", text: "How to Use t-SNE Effectively (Distill) — interactive, and the definitive warning against over-reading t-SNE plots" },
        { type: "link", url: "https://umap-learn.readthedocs.io/en/latest/", text: "UMAP documentation — how it works, parameters, and the theory (McInnes & Healy)" },
        { type: "link", url: "https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf", text: "van der Maaten & Hinton (2008) — the original t-SNE paper (JMLR)" },
        { type: "link", url: "https://arxiv.org/abs/1802.03426", text: "McInnes, Healy & Melville (2018) — UMAP: Uniform Manifold Approximation and Projection (arXiv)" },
        { type: "link", url: "https://www.deeplearningbook.org/contents/autoencoders.html", text: "Goodfellow et al., Deep Learning — Chapter 14 (autoencoders as learned dimensionality reduction)" },
        { type: "link", url: "https://mml-book.github.io/", text: "Mathematics for Machine Learning — Chapter 10 derives PCA cleanly from both objectives (free PDF)" },
      ]
    },
  ],

  packages: [
    { name: "scikit-learn", why: "`PCA`, `KernelPCA`, `TSNE`, `StandardScaler` — the whole classical toolkit" },
    { name: "sklearn.decomposition.PCA", why: "linear reduction via SVD; `explained_variance_ratio_` is your key diagnostic" },
    { name: "sklearn.manifold.TSNE", why: "local-structure visualization (Barnes–Hut); pair with PCA-to-50 first" },
    { name: "umap-learn", why: "faster, more global-aware embeddings, and a reusable `transform` — import as `umap`" },
    { name: "numpy", why: "`np.linalg.svd` / `eigh` — PCA from scratch, and the numerically right route" },
    { name: "sklearn.preprocessing.StandardScaler", why: "scale before PCA — variance is unit-dependent" },
    { name: "matplotlib", why: "scree plots, 2D embeddings, and eigenface grids" },
    { name: "torch", why: "autoencoders — non-linear, reusable, generative dimensionality reduction" },
  ],

  gotchas: [
    "**Center before PCA, scale before PCA.** PCA needs mean-zero data ($\\mathtt{PCA}$ centers for you); it does *not* scale, and variance depends on units — run `StandardScaler` unless features share a unit.",
    "Don't form the covariance matrix and `eig` it in production — squaring $\\tilde{X}^\\top\\tilde{X}$ squares the condition number. Use `svd` of the centered data (what `PCA` does).",
    "PCA components come with an **arbitrary sign**: $w$ and $-w$ are the same axis. Two libraries can disagree by a sign flip per component — that's not a bug.",
    "**Never cluster or classify on t-SNE or UMAP coordinates.** They are lossy pictures for the human eye; distances between clusters and cluster sizes are not meaningful.",
    "t-SNE is stochastic and has no `transform` — re-running changes the plot and you can't embed new points. UMAP *does* have `transform`.",
    "Kernel PCA and t-SNE are $O(m^2)$+ in memory/time — reduce to ~50 dims with PCA first, then run the non-linear method on the subset.",
    "PCA is **linear**: it can't unfold a curved manifold (a spiral, a Swiss roll). Use Kernel PCA / t-SNE / UMAP / autoencoders for that.",
    "Explained-variance ratio is $\\lambda_i / \\sum_j \\lambda_j$; a low cumulative value at your chosen $k$ means the data isn't as compressible as you hoped — don't force it.",
  ],

  flashcards: [
    { q: "State PCA's two equivalent objectives.", a: "Maximize the variance of the projected data, OR minimize squared reconstruction error of a $k$-dim linear subspace. Both give the top-$k$ eigenvectors of the covariance matrix." },
    { q: "What is the solution to 'maximize $w^\\top C w$ s.t. $\\|w\\|=1$'?", a: "The eigenvector of the covariance $C$ with the largest eigenvalue; the captured variance equals that eigenvalue $\\lambda$." },
    { q: "How does PCA relate to the SVD?", a: "For centered $\\tilde{X}=U\\Sigma V^\\top$, the principal components are the right-singular vectors $V$, and covariance eigenvalues are $\\lambda_i=\\sigma_i^2/(m-1)$. SVD is the numerically stable route." },
    { q: "What is the explained-variance ratio?", a: "$\\sum_{i\\le k}\\lambda_i / \\sum_j \\lambda_j$ — the fraction of total variance the top $k$ components keep. The trace $\\sum_j\\lambda_j$ is the total variance." },
    { q: "Why must you scale features before PCA?", a: "PCA maximizes variance, which depends on units; a large-unit feature dominates artificially. Standardize unless all features share a natural unit." },
    { q: "What does t-SNE optimize, and what's it for?", a: "It minimizes KL divergence between high-D and low-D neighbor probability distributions, using a Student-t in the map. Visualization of local structure only — never cluster on it." },
    { q: "What does perplexity control in t-SNE?", a: "The effective number of neighbors each point considers (sets each Gaussian's width $\\sigma_i$); typically 5–50. Small = tiny local clusters, large = broader structure." },
    { q: "When prefer UMAP over t-SNE?", a: "When you have many points (faster), need to embed new points (`transform`), or care about relative cluster spacing (better global structure). UMAP is the modern default." },
    { q: "In what sense is PCA a special case of an autoencoder?", a: "A linear autoencoder (no activations) with MSE loss learns exactly the top-$k$ PCA subspace. Non-linear layers generalize it to curved manifolds." },
    { q: "What is the kernel trick in Kernel PCA?", a: "Do linear PCA in an implicit high-D feature space using only inner products from a kernel $k(x,x')=\\phi(x)^\\top\\phi(x')$ — eigendecompose the centered kernel matrix instead of the covariance." },
    { q: "Why is a PCA component hard to interpret?", a: "It's a linear combination of all original features — a direction of variance, not a concept. Use sparse PCA or NMF if you need interpretable factors." },
  ],

  cheatsheet: [
    { label: "PCA (fit + reduce)", code: "Z = PCA(n_components=2).fit_transform(X)" },
    { label: "Keep 95% variance", code: "PCA(n_components=0.95)" },
    { label: "Explained variance", code: "pca.explained_variance_ratio_.cumsum()" },
    { label: "Reconstruct", code: "X_hat = pca.inverse_transform(Z)" },
    { label: "PCA via SVD (scratch)", code: "U,s,Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)" },
    { label: "Covariance eigen (scratch)", code: "np.linalg.eigh(np.cov(Xc, rowvar=False))" },
    { label: "Scale then PCA", code: "make_pipeline(StandardScaler(), PCA(0.95))" },
    { label: "Whiten", code: "PCA(n_components=k, whiten=True)" },
    { label: "Kernel PCA (RBF)", code: "KernelPCA(n_components=2, kernel='rbf', gamma=15)" },
    { label: "t-SNE", code: "TSNE(n_components=2, perplexity=30, init='pca')" },
    { label: "UMAP", code: "umap.UMAP(n_neighbors=15, min_dist=0.1)" },
    { label: "UMAP embed new pts", code: "reducer.transform(X_new)" },
    { label: "Choose k for 95%", code: "np.searchsorted(cum_ratio, 0.95) + 1" },
  ],
});
