(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "ml-clustering",
  name: "Clustering",
  language: "Classical ML",
  group: "Classical Machine Learning",
  navLabel: "Clustering",
  tagline: "Unsupervised learning from the ground up — k-means as coordinate descent, GMMs & EM, hierarchical & DBSCAN, and how to trust the result with no labels.",
  color: "#047857",
  readMinutes: 46,
  sections: [
    {
      id: "unsupervised-setup",
      title: "Unsupervised learning & what clustering is for",
      level: "core",
      body: [
        { type: "p", text: "Everything you have seen so far is **supervised**: each example $x^{(i)}$ arrives with a target $y^{(i)}$, and the model learns the map $x \\mapsto y$. **Unsupervised learning** removes the labels. You are handed only the features:" },
        { type: "math", tex: String.raw`\mathcal{D} = \{x^{(1)}, x^{(2)}, \dots, x^{(m)}\}, \qquad x^{(i)} \in \mathbb{R}^n, \qquad \text{no } y` },
        { type: "p", text: "The goal shifts from *predict a known answer* to *discover structure*. **Clustering** is the flagship unsupervised task: partition the data into groups so that points in the same group are similar and points in different groups are not. Nobody tells you the groups exist — you assert them from geometry alone." },
        { type: "heading", text: "Why you would ever want this" },
        { type: "table",
          headers: ["Use case", "What a cluster means", "Payoff"],
          rows: [
            ["Customer segmentation", "a behavioural persona", "target marketing per segment, not per person"],
            ["Anomaly / fraud detection", "the normal region", "anything far from every cluster is suspicious"],
            ["Image color quantization", "a representative color", "compress an image to $k$ colors"],
            ["Document / topic grouping", "a theme", "organize an unlabeled corpus"],
            ["Vector quantization", "a codebook entry", "lossy compression of embeddings/audio"],
            ["Semi-supervised labeling", "a candidate class", "label one point per cluster, propagate"],
          ]
        },
        { type: "p", text: "Two of these deserve emphasis. **Anomaly detection** is clustering read inside-out: instead of caring about the clusters, you care about the points that *refuse* to join one. **Compression** (vector quantization) is clustering used as a codebook — replace each point by the label of its cluster and store $k$ representatives instead of $m$ points." },
        { type: "heading", text: "The problem that haunts the entire field: no labels, no ground truth" },
        { type: "p", text: "In supervised learning you have an unambiguous scoreboard: held-out accuracy. Clustering has **no such scoreboard**, because there is no correct answer to check against. Is a shopper 'a bargain hunter' or 'a weekend browser'? Both, neither — the labels are a story you impose. This creates three hard problems that thread through the whole topic:" },
        { type: "list", ordered: true, items: [
          "**How many clusters?** $k$ is usually an input you must choose, and the data will not tell you cleanly.",
          "**What does 'similar' mean?** The answer depends entirely on your distance metric and feature scaling — change either and the clusters change.",
          "**Is the result any good?** With no labels you can only measure *internal* consistency (are clusters tight and separated?), which is a proxy, not truth.",
        ]},
        { type: "callout", variant: "note", text: "**How to read this section.** Clustering is a family, not one algorithm. Each member encodes a different *assumption* about what a cluster is: k-means says 'a round blob around a center', GMM says 'a Gaussian', DBSCAN says 'a dense region', hierarchical says 'whatever merges first'. Pick the algorithm whose assumption matches your data's shape — that single decision matters more than any hyperparameter." },
        { type: "callout", variant: "gotcha", text: "**Scale your features first, always.** Distance-based clustering treats a feature measured in dollars (range $0$–$100{,}000$) as thousands of times more important than one measured in years (range $0$–$100$). Standardize (`StandardScaler`) or normalize before clustering unless you have a deliberate reason not to. This is the single most common clustering mistake." },
      ]
    },

    {
      id: "kmeans-objective",
      title: "k-means: the objective & Lloyd's algorithm as coordinate descent",
      level: "core",
      body: [
        { type: "p", text: "**k-means** is the 'hello world' of clustering, published in this form by Stuart Lloyd at Bell Labs in 1957 (for pulse-code modulation) and independently by others through the 1960s. Fix a number of clusters $k$. Each cluster is summarized by a single point — its **centroid** $\\mu_c$. Every data point is assigned to its nearest centroid, and we want centroids placed so that points sit as close to their centroid as possible." },
        { type: "heading", text: "The objective: within-cluster sum of squares (inertia)" },
        { type: "p", text: "Let $r_{ic} \\in \\{0,1\\}$ be a hard assignment: $r_{ic}=1$ if point $i$ belongs to cluster $c$, else $0$, with exactly one $1$ per row. k-means minimizes the total squared distance from each point to its assigned centroid — the **within-cluster sum of squares**, also called **inertia** or **distortion**:" },
        { type: "math", tex: String.raw`J(\{r_{ic}\}, \{\mu_c\}) = \sum_{i=1}^{m} \sum_{c=1}^{k} r_{ic}\, \lVert x^{(i)} - \mu_c \rVert_2^2` },
        { type: "p", text: "This single function has **two kinds of variables**: the discrete assignments $r_{ic}$ and the continuous centroids $\\mu_c$. Minimizing over both at once is NP-hard. But minimizing over *one while holding the other fixed* is trivial in closed form — and that observation is the entire algorithm." },
        { type: "heading", text: "Lloyd's algorithm = coordinate descent on $J$" },
        { type: "p", text: "**Coordinate descent** minimizes a function by cyclically optimizing one block of variables at a time. Apply it to $J$ and you get exactly Lloyd's algorithm — two alternating steps." },
        { type: "p", text: "**Assignment step (optimize $r$, hold $\\mu$ fixed).** With centroids frozen, the sum decouples across points: each point's contribution $\\sum_c r_{ic}\\lVert x^{(i)}-\\mu_c\\rVert^2$ is minimized independently by putting its single $1$ on the nearest centroid." },
        { type: "math", tex: String.raw`r_{ic} = \begin{cases} 1 & \text{if } c = \arg\min_{c'} \lVert x^{(i)} - \mu_{c'} \rVert_2^2 \\ 0 & \text{otherwise} \end{cases}` },
        { type: "p", text: "**Update step (optimize $\\mu$, hold $r$ fixed).** With assignments frozen, $J$ is a smooth quadratic in each $\\mu_c$. Take the gradient with respect to $\\mu_c$ and set it to zero:" },
        { type: "math", tex: String.raw`\nabla_{\mu_c} J = \sum_{i=1}^{m} r_{ic}\, \cdot 2\,(\mu_c - x^{(i)}) \;\overset{!}{=}\; 0` },
        { type: "math", tex: String.raw`\Longrightarrow\quad \mu_c = \frac{\sum_{i=1}^{m} r_{ic}\, x^{(i)}}{\sum_{i=1}^{m} r_{ic}} = \text{mean of the points assigned to cluster } c` },
        { type: "p", text: "So the optimal centroid is literally the **average** of its members. That is where the name comes from: $k$ means (averages)." },
        { type: "heading", text: "Why it must converge: each step cannot increase $J$" },
        { type: "p", text: "This is the part worth internalizing. Both steps are *exact minimizations* of the same objective over their block of variables, so neither can ever raise $J$:" },
        { type: "list", ordered: false, items: [
          "The **assignment step** picks, for every point, the centroid that minimizes its term — any other assignment gives an equal or larger $J$.",
          "The **update step** sets each centroid to the provably optimal value (the mean) for the current assignments — any other centroid gives an equal or larger $J$.",
        ]},
        { type: "math", tex: String.raw`J_{t} \;\ge\; J_{t+\frac{1}{2}}\ (\text{after assign}) \;\ge\; J_{t+1}\ (\text{after update}) \;\ge\; 0` },
        { type: "p", text: "So $J$ is **monotonically non-increasing and bounded below by $0$**, hence it converges. Moreover there are only finitely many possible assignments ($k^m$ of them), and $J$ strictly decreases whenever the assignment changes, so the algorithm reaches a fixed point in a **finite** number of iterations. In practice that is a handful — usually 10–30." },
        { type: "callout", variant: "gotcha", text: "**Convergence is not the same as correctness.** Lloyd's algorithm converges to a **local** minimum of $J$, and $J$ is riddled with them. A bad start can trap you in a clearly-wrong clustering that is nonetheless a valid fixed point. The standard defense is to run it many times from different seeds and keep the lowest-inertia result (`n_init` in scikit-learn) — plus smarter initialization, which is the next section." },
      ]
    },

    {
      id: "kmeans-scratch",
      title: "k-means from scratch, k-means++, and choosing k",
      level: "core",
      body: [
        { type: "p", text: "The derivation is short enough to code directly — the whole algorithm is 'assign, average, repeat'. Here it is in NumPy with no loops over points, vectorized via broadcasting." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef kmeans(X, k, n_iters=100, seed=0):\n    rng = np.random.default_rng(seed)\n    # naive init: pick k distinct data points as starting centroids\n    mu = X[rng.choice(len(X), size=k, replace=False)]\n\n    for _ in range(n_iters):\n        # --- assignment step -------------------------------------------\n        # dist[i, c] = ||x_i - mu_c||^2  via broadcasting -> (m, k)\n        dist = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)\n        labels = dist.argmin(axis=1)          # nearest centroid per point\n\n        # --- update step -----------------------------------------------\n        new_mu = np.array([\n            X[labels == c].mean(axis=0) if np.any(labels == c) else mu[c]\n            for c in range(k)\n        ])\n        if np.allclose(new_mu, mu):           # fixed point reached\n            break\n        mu = new_mu\n\n    inertia = ((X - mu[labels]) ** 2).sum()   # the objective J\n    return labels, mu, inertia\n\nX = np.random.default_rng(1).normal(size=(300, 2))\nlabels, mu, inertia = kmeans(X, k=3)\nprint(mu.shape, inertia.round(2))" },
        { type: "callout", variant: "gotcha", text: "**Empty clusters happen.** If no point is assigned to a centroid, its mean is undefined (`nan`). The code above keeps the old centroid; scikit-learn instead relocates the empty centroid to the point farthest from its current centroid. Never ignore this case — a stray `nan` centroid poisons every subsequent assignment." },
        { type: "heading", text: "Why random initialization is bad, and what k-means++ fixes" },
        { type: "p", text: "Seeding centroids by picking $k$ random points frequently places two seeds inside the *same* true cluster, leaving another true cluster with no seed. Lloyd's algorithm then converges to a local minimum that splits one real cluster and merges two others — a bad clustering that is nonetheless stable. **k-means++** (Arthur & Vassilvitskii, 2007) fixes this by spreading the initial seeds out, choosing each new center with probability proportional to its squared distance from the nearest already-chosen center." },
        { type: "math", tex: String.raw`\Pr(\text{pick } x^{(i)} \text{ next}) = \frac{D(x^{(i)})^2}{\sum_{j} D(x^{(j)})^2}, \qquad D(x) = \min_{c \,\in\, \text{chosen}} \lVert x - \mu_c \rVert` },
        { type: "code", lang: "py", code: "def kmeans_pp_init(X, k, seed=0):\n    rng = np.random.default_rng(seed)\n    centers = [X[rng.integers(len(X))]]        # first center: uniform random\n    for _ in range(1, k):\n        D2 = np.min(                            # squared dist to nearest center\n            [((X - c) ** 2).sum(axis=1) for c in centers], axis=0)\n        probs = D2 / D2.sum()                   # farther points -> more likely\n        centers.append(X[rng.choice(len(X), p=probs)])\n    return np.array(centers)" },
        { type: "callout", variant: "good", text: "**k-means++ is the default for a reason.** It gives an expected $O(\\log k)$-competitive clustering *before Lloyd's even runs*, and it usually converges faster too. scikit-learn's `KMeans` uses it by default (`init='k-means++'`). You almost never want plain random init." },
        { type: "heading", text: "Choosing k: the elbow method" },
        { type: "p", text: "Inertia $J$ always decreases as $k$ grows (more centroids can only fit tighter), reaching $0$ when $k=m$. So you cannot just minimize it — you look for the **elbow**, the $k$ past which adding clusters buys little extra reduction. Plot $J$ versus $k$ and pick the kink where the curve flattens." },
        { type: "code", lang: "py", code: "inertias = [kmeans(X, k)[2] for k in range(1, 10)]\n# plot(range(1,10), inertias); the elbow is where the slope drops sharply." },
        { type: "callout", variant: "gotcha", text: "The elbow is **subjective** — often there is no crisp corner, just a gentle bend. Treat it as a hint, not a verdict. The silhouette score below is more principled because it actually measures cluster quality rather than raw compactness." },
        { type: "heading", text: "Choosing k: the silhouette score (derived)" },
        { type: "p", text: "The **silhouette** (Rousseeuw, 1987) scores each point on how well it fits its own cluster versus the nearest other cluster. For a point $i$, define two quantities:" },
        { type: "math", tex: String.raw`a(i) = \frac{1}{|C_i| - 1} \sum_{j \in C_i,\, j \neq i} d(i, j) \quad(\text{mean intra-cluster distance})` },
        { type: "math", tex: String.raw`b(i) = \min_{C \neq C_i} \; \frac{1}{|C|} \sum_{j \in C} d(i, j) \quad(\text{mean distance to the nearest other cluster})` },
        { type: "p", text: "$a(i)$ is how tight $i$ sits with its own kind; $b(i)$ is how far the nearest rival cluster is. A good point has $a(i)$ small and $b(i)$ large. The **silhouette coefficient** combines them into a bounded score:" },
        { type: "math", tex: String.raw`s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}} \;\in\; [-1, 1]` },
        { type: "p", text: "Read it off directly: $s(i) \\approx 1$ means $i$ is deep inside a well-separated cluster; $s(i) \\approx 0$ means $i$ sits on the boundary between two clusters; $s(i) < 0$ means $i$ is probably in the *wrong* cluster (it is closer to a neighboring cluster than its own). Average $s(i)$ over all points to score a whole clustering, and pick the $k$ that **maximizes** the mean silhouette." },
        { type: "callout", variant: "tip", text: "Silhouette works with **any** distance metric and **any** clustering algorithm, not just k-means — so it is your go-to internal metric across this whole section. Its cost is $O(m^2)$ distances, so on large datasets score a random subsample." },
      ]
    },

    {
      id: "kmeans-limits",
      title: "The assumptions k-means bakes in (and where they break)",
      level: "core",
      body: [
        { type: "p", text: "k-means is fast and simple, but that simplicity comes from strong hidden assumptions. Because it assigns each point to the *nearest centroid by Euclidean distance* and summarizes clusters by a *single mean*, it implicitly assumes clusters are:" },
        { type: "list", ordered: false, items: [
          "**Spherical (isotropic).** Equal spread in every direction. A long, thin, or diagonal cluster gets sliced in half because the mean sits in the middle and the Euclidean ball ignores the cluster's true shape.",
          "**Similar in size and density.** A big loose cluster next to a small tight one gets 'eaten': boundary points of the big cluster are stolen by the small one because k-means balances *distance*, not *membership counts*.",
          "**Linearly (convex-region) separable.** The decision boundary between two centroids is always a straight hyperplane (the perpendicular bisector). Concentric rings or crescent moons cannot be separated by straight lines, so k-means fails outright.",
          "**Known in number.** You must supply $k$; the algorithm will happily split a single blob into $k$ pie slices if you ask for too many.",
        ]},
        { type: "heading", text: "A failure case you can run in 10 lines" },
        { type: "p", text: "The canonical demonstration is two interlocking crescents (`make_moons`). Each moon is one obvious cluster to a human, but k-means draws a straight line through both because it can only carve space with hyperplanes:" },
        { type: "code", lang: "py", code: "from sklearn.datasets import make_moons\nfrom sklearn.cluster import KMeans\n\nX, y_true = make_moons(n_samples=400, noise=0.05, random_state=0)\nlabels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)\n\n# Plotting labels vs y_true shows k-means cutting BOTH moons in half\n# with a vertical line, instead of following each crescent.\n# DBSCAN (later) recovers the two moons perfectly.\nimport numpy as np\nprint(\"k-means groups points by side, not by moon:\",\n      np.mean(labels == y_true))   # ~0.5-0.75, essentially chance" },
        { type: "callout", variant: "note", text: "The moons dataset is the 'hello world' of *why other algorithms exist*. Every algorithm after this one — GMM, DBSCAN, spectral clustering — can be motivated by 'here is a shape k-means gets wrong.' When k-means disappoints, ask which assumption your data violates, then pick the algorithm that drops it." },
        { type: "callout", variant: "tip", text: "**Rescue k-means with features, not just algorithms.** If clusters are non-spherical but you love k-means' speed, transform first: run it in a PCA or (better) a *kernel/spectral* embedding where the clusters become round. Spectral clustering is literally k-means on the eigenvectors of a similarity graph — it solves the moons perfectly." },
      ]
    },

    {
      id: "gmm-em",
      title: "Gaussian Mixture Models & the EM algorithm",
      level: "core",
      body: [
        { type: "p", text: "k-means makes a **hard** decision: a point is 100% in one cluster. But a point halfway between two clusters is genuinely ambiguous, and forcing a hard label throws that information away. A **Gaussian Mixture Model (GMM)** does **soft** clustering — it gives each point a *probability* of belonging to each cluster, and it lets clusters be stretched, rotated ellipses rather than spheres." },
        { type: "heading", text: "The generative story" },
        { type: "p", text: "A GMM models the data as coming from a mixture of $k$ Gaussians. To generate a point: first pick a cluster $c$ with probability $\\pi_c$ (the mixing weights, $\\sum_c \\pi_c = 1$), then draw the point from that cluster's Gaussian $\\mathcal{N}(\\mu_c, \\Sigma_c)$. The overall density is a weighted sum:" },
        { type: "math", tex: String.raw`p(x) = \sum_{c=1}^{k} \pi_c \, \mathcal{N}(x \mid \mu_c, \Sigma_c)` },
        { type: "p", text: "Each cluster now has three parameters: a weight $\\pi_c$, a mean $\\mu_c$, and a full **covariance** $\\Sigma_c$ that encodes its shape and orientation. That covariance is what lets a GMM fit diagonal, elongated clusters that k-means cannot." },
        { type: "heading", text: "The chicken-and-egg problem EM solves" },
        { type: "p", text: "If we knew which cluster each point came from, fitting each Gaussian would be trivial (just estimate its mean and covariance from its members). If we knew the Gaussians, computing each point's cluster probability would be trivial (Bayes' rule). We know neither — a latent-variable deadlock. **Expectation–Maximization (EM)** (Dempster, Laird & Rubin, 1977) breaks it by alternating, exactly like Lloyd's algorithm but with soft assignments." },
        { type: "p", text: "**E-step (Expectation).** Fix the parameters; compute the **responsibility** $\\gamma_{ic}$ — the posterior probability that point $i$ came from cluster $c$, via Bayes' rule:" },
        { type: "math", tex: String.raw`\gamma_{ic} = \Pr(c \mid x^{(i)}) = \frac{\pi_c\, \mathcal{N}(x^{(i)} \mid \mu_c, \Sigma_c)}{\sum_{c'=1}^{k} \pi_{c'}\, \mathcal{N}(x^{(i)} \mid \mu_{c'}, \Sigma_{c'})}` },
        { type: "p", text: "These are the soft assignments: $\\gamma_{ic} \\in [0,1]$ and $\\sum_c \\gamma_{ic} = 1$. A point can be 70% cluster A, 30% cluster B." },
        { type: "p", text: "**M-step (Maximization).** Fix the responsibilities; re-estimate each Gaussian's parameters as *responsibility-weighted* averages. Let $N_c = \\sum_i \\gamma_{ic}$ be the effective number of points in cluster $c$:" },
        { type: "math", tex: String.raw`\mu_c = \frac{1}{N_c}\sum_{i} \gamma_{ic}\, x^{(i)}, \qquad \Sigma_c = \frac{1}{N_c}\sum_{i} \gamma_{ic}\,(x^{(i)} - \mu_c)(x^{(i)} - \mu_c)^\top, \qquad \pi_c = \frac{N_c}{m}` },
        { type: "p", text: "Notice these are exactly the ordinary maximum-likelihood estimates for a Gaussian, but each point is counted with weight $\\gamma_{ic}$ instead of a hard $0/1$. EM provably increases the data **log-likelihood** $\\sum_i \\log p(x^{(i)})$ every iteration (or leaves it unchanged), so like Lloyd's it converges monotonically to a local optimum." },
        { type: "callout", variant: "note", text: "**Why EM increases likelihood (the one-line reason).** EM does not maximize the log-likelihood directly; it maximizes a lower bound (the ELBO) that touches the true likelihood at the current parameters. The E-step makes the bound tight; the M-step pushes the bound up. Because the bound touches from below, raising it raises the true likelihood too. This is the same variational machinery behind VAEs — you will meet it again." },
        { type: "heading", text: "GMM vs k-means: k-means is the hard-assignment limit" },
        { type: "p", text: "These two algorithms are not cousins — they are the *same algorithm* at different temperatures. Take a GMM, force every covariance to be spherical with the same variance $\\Sigma_c = \\sigma^2 I$, and let $\\sigma^2 \\to 0$. The responsibility becomes a `softmax` over negative squared distances with temperature $\\sigma^2$:" },
        { type: "math", tex: String.raw`\gamma_{ic} \;\propto\; \exp\!\left(-\frac{\lVert x^{(i)} - \mu_c \rVert^2}{2\sigma^2}\right) \;\xrightarrow[\;\sigma^2 \to 0\;]{}\; \begin{cases} 1 & c = \arg\min_{c'} \lVert x^{(i)} - \mu_{c'}\rVert^2 \\ 0 & \text{otherwise} \end{cases}` },
        { type: "p", text: "As $\\sigma^2 \\to 0$ the softmax sharpens into an argmax: soft responsibilities collapse to hard nearest-centroid assignments, and the M-step mean update becomes the k-means centroid update. So **k-means is EM for a GMM with equal spherical covariances in the zero-variance limit** — hard clustering is just soft clustering with the temperature turned to zero." },
        { type: "table",
          headers: ["", "k-means", "GMM (EM)"],
          rows: [
            ["Assignment", "hard (0/1)", "soft (probabilities $\\gamma_{ic}$)"],
            ["Cluster shape", "sphere (equal size)", "any ellipse (per-cluster $\\Sigma_c$)"],
            ["Objective", "inertia (WCSS)", "log-likelihood"],
            ["Output per point", "one label", "$k$ probabilities + density $p(x)$"],
            ["Speed", "fast", "slower (covariances are expensive)"],
          ]
        },
        { type: "callout", variant: "tip", text: "Because a GMM gives you a density $p(x)$, it doubles as an **anomaly detector**: points with very low $p(x)$ are outliers. And `covariance_type='full'` fits rotated ellipses, `'diag'` axis-aligned ones, `'spherical'` circles (≈ soft k-means). Start with `'full'` unless you have too little data per cluster." },
      ]
    },

    {
      id: "hierarchical",
      title: "Hierarchical (agglomerative) clustering & dendrograms",
      level: "core",
      body: [
        { type: "p", text: "k-means and GMM demand you fix $k$ up front and give a *flat* partition. **Hierarchical clustering** instead builds a whole *tree* of clusterings at every granularity, and you slice it at whatever level you like afterwards. The bottom-up (**agglomerative**) variant is by far the most common:" },
        { type: "list", ordered: true, items: [
          "Start with every point as its own singleton cluster ($m$ clusters).",
          "Find the two **closest** clusters and merge them into one.",
          "Repeat until a single cluster contains everything.",
        ]},
        { type: "p", text: "Every merge is recorded with the distance at which it happened, producing a **dendrogram** — a tree whose leaves are points and whose branch heights are merge distances. To get a flat clustering, cut the dendrogram with a horizontal line: the number of vertical branches it crosses is your $k$. Cut low for many tight clusters, high for a few broad ones." },
        { type: "heading", text: "Linkage: the definition of 'distance between two clusters'" },
        { type: "p", text: "The one design choice is how to measure the distance between two *clusters* (not two points). This is the **linkage criterion**, and it dramatically changes the shape of the result:" },
        { type: "table",
          headers: ["Linkage", "Cluster distance $d(A,B)$", "Tendency"],
          rows: [
            ["Single", "$\\min_{a\\in A, b\\in B} d(a,b)$ (closest pair)", "long, chained, snaky clusters; sensitive to noise"],
            ["Complete", "$\\max_{a\\in A, b\\in B} d(a,b)$ (farthest pair)", "compact, roughly equal-diameter clusters"],
            ["Average", "mean over all cross pairs", "a balanced compromise between single & complete"],
            ["Ward", "increase in within-cluster variance from merging", "minimizes variance; spherical, k-means-like clusters"],
          ]
        },
        { type: "p", text: "**Ward's linkage** is the usual default: it merges the pair of clusters that increases the total within-cluster sum of squares the least, making it the hierarchical analogue of k-means' objective. **Single linkage** is the one to reach for when clusters are stringy or non-convex (it can trace a chain of nearby points), at the cost of being fragile to noisy bridges between clusters." },
        { type: "code", lang: "py", code: "from scipy.cluster.hierarchy import linkage, dendrogram, fcluster\nimport numpy as np\n\nX = np.random.default_rng(0).normal(size=(30, 2))\n\nZ = linkage(X, method='ward')       # the merge tree (an (m-1) x 4 matrix)\n# dendrogram(Z)                     # -> the tree plot\n\nlabels = fcluster(Z, t=3, criterion='maxclust')   # cut into exactly 3 clusters\nprint(np.unique(labels))            # [1 2 3]" },
        { type: "callout", variant: "gotcha", text: "**Agglomerative clustering is $O(m^2)$ memory and up to $O(m^3)$ time** because it reasons about all pairwise distances. It is wonderful for a few thousand points and a beautiful dendrogram, but it does not scale to millions — use k-means or DBSCAN there. It is also **greedy**: a merge, once made, is never undone, so an early mistake propagates up the tree." },
        { type: "callout", variant: "good", text: "The killer feature is that you **do not commit to $k$ in advance** and you get an interpretable *taxonomy*. This is why hierarchical clustering dominates in biology (gene-expression heatmaps, phylogenetic trees) — the tree itself is the deliverable, not just the flat labels." },
      ]
    },

    {
      id: "dbscan",
      title: "DBSCAN: density-based clustering that finds any shape",
      level: "core",
      body: [
        { type: "p", text: "**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise; Ester, Kriegel, Sander & Xu, 1996 — winner of a 'test of time' award) throws out the centroid idea entirely. A cluster is defined not by a center but as a **dense region of points**: keep absorbing neighbors as long as the neighborhood stays crowded, and stop at the sparse gaps. This buys three things k-means cannot give you." },
        { type: "list", ordered: false, items: [
          "**Arbitrary shapes.** Because clusters grow by local density, DBSCAN traces crescents, rings, and spirals — it solves the moons dataset instantly.",
          "**No $k$ required.** The number of clusters emerges from the data; you never specify it.",
          "**Built-in noise handling.** Points in no dense region are labeled **outliers** (label $-1$), not forced into a cluster. This makes DBSCAN a natural anomaly detector.",
        ]},
        { type: "heading", text: "The two knobs: eps and minPts" },
        { type: "p", text: "DBSCAN has exactly two hyperparameters that together define 'dense':" },
        { type: "table",
          headers: ["Parameter", "Meaning"],
          rows: [
            ["$\\varepsilon$ (eps)", "neighborhood radius — two points are neighbors if within distance $\\varepsilon$"],
            ["minPts", "how many neighbors (incl. itself) a point needs inside its $\\varepsilon$-ball to be 'core'"],
          ]
        },
        { type: "p", text: "From these two numbers, every point falls into one of three roles:" },
        { type: "list", ordered: false, items: [
          "**Core point** — has at least `minPts` points within distance $\\varepsilon$. These are the dense interior.",
          "**Border point** — within $\\varepsilon$ of a core point but not itself dense enough. The edges of a cluster.",
          "**Noise point** — neither core nor border. Reachable from nothing; labeled $-1$.",
        ]},
        { type: "p", text: "The algorithm then grows clusters by **density-reachability**: pick an unvisited core point, start a new cluster, and recursively pull in everything density-reachable (every point in a core point's $\\varepsilon$-ball, and their $\\varepsilon$-balls if they too are core). When the expansion stalls, start a new cluster from the next unvisited core point. Border points join whichever cluster first reaches them; leftover points are noise." },
        { type: "code", lang: "py", code: "from sklearn.cluster import DBSCAN\nfrom sklearn.datasets import make_moons\nimport numpy as np\n\nX, _ = make_moons(n_samples=400, noise=0.05, random_state=0)\ndb = DBSCAN(eps=0.2, min_samples=5).fit(X)\n\nlabels = db.labels_\nn_clusters = len(set(labels)) - (1 if -1 in labels else 0)\nn_noise = np.sum(labels == -1)\nprint(f\"{n_clusters} clusters, {n_noise} noise points\")   # -> 2 clusters\n# DBSCAN recovers BOTH crescents that k-means sliced in half." },
        { type: "callout", variant: "tip", text: "**Choosing $\\varepsilon$: the k-distance plot.** Compute each point's distance to its `minPts`-th nearest neighbor, sort ascending, and plot. The curve has a sharp 'knee' where you cross from dense (intra-cluster) to sparse (inter-cluster) distances — set $\\varepsilon$ at that knee. A common heuristic sets `minPts = 2 * n_features` as a starting point." },
        { type: "callout", variant: "gotcha", text: "**DBSCAN struggles with clusters of very different densities.** A single global $\\varepsilon$ cannot be simultaneously right for a tight cluster and a loose one — it either merges the loose one into noise or bridges the tight ones together. When densities vary, reach for **HDBSCAN**, which builds a hierarchy over $\\varepsilon$ and extracts the most stable clusters automatically. It is the modern default when you can install it." },
      ]
    },

    {
      id: "evaluation",
      title: "Evaluation: how to trust a clustering (with and without labels)",
      level: "core",
      body: [
        { type: "p", text: "Recall the core difficulty: no labels means no accuracy. Evaluation metrics split into two families depending on whether you have ground-truth labels to compare against." },
        { type: "heading", text: "Internal metrics — no labels needed" },
        { type: "p", text: "These score a clustering by its own geometry: are clusters tight inside and far apart from each other? Use them to compare clusterings or pick $k$." },
        { type: "table",
          headers: ["Metric", "Measures", "Better = "],
          rows: [
            ["Silhouette $\\in[-1,1]$", "(separation − cohesion) per point, averaged", "higher (near $+1$)"],
            ["Davies–Bouldin $\\ge 0$", "avg ratio of within-cluster spread to between-cluster distance", "lower (near $0$)"],
            ["Calinski–Harabasz", "ratio of between- to within-cluster variance", "higher"],
            ["Inertia (WCSS)", "total within-cluster squared distance", "lower (but decreases with $k$)"],
          ]
        },
        { type: "p", text: "The **Davies–Bouldin index** is worth knowing alongside silhouette. For each cluster it finds its *worst* rival — the other cluster with the highest (spread$_i$ + spread$_j$) / (distance between centers) — and averages that worst-case ratio over all clusters:" },
        { type: "math", tex: String.raw`\text{DB} = \frac{1}{k}\sum_{i=1}^{k} \max_{j \neq i} \frac{s_i + s_j}{d(\mu_i, \mu_j)}, \qquad s_i = \text{avg distance of cluster } i\text{'s points to } \mu_i` },
        { type: "p", text: "Small DB means clusters are compact ($s_i$ small) and well separated ($d(\\mu_i,\\mu_j)$ large) — exactly what you want. Unlike silhouette it is cheap ($O(mk)$), so it scales better." },
        { type: "heading", text: "External metrics — when you *do* have labels" },
        { type: "p", text: "Sometimes you have ground-truth classes (a labeled benchmark, or human annotations) and want to check whether your unsupervised clusters *recover* them. You cannot compare labels directly — the algorithm's 'cluster 3' has no reason to equal your 'class 3'. You need a metric invariant to label permutation. The **Rand index** counts agreements over all pairs of points:" },
        { type: "math", tex: String.raw`\text{RI} = \frac{a + b}{\binom{m}{2}}` },
        { type: "p", text: "where $a$ = pairs in the *same* cluster **and** same true class, $b$ = pairs in *different* clusters **and** different true classes. It asks 'do these two points agree about togetherness?' for every pair, ignoring what the labels are actually named. The **Adjusted Rand Index (ARI)** corrects for the fact that even random labelings score above zero by chance:" },
        { type: "math", tex: String.raw`\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]} \;\in\; [-1, 1]` },
        { type: "p", text: "ARI $= 1$ is a perfect match, $0$ is no better than random chance, and negative means *worse* than random. It is the standard when a ground truth exists." },
        { type: "code", lang: "py", code: "from sklearn.metrics import (silhouette_score, davies_bouldin_score,\n                             adjusted_rand_score, normalized_mutual_info_score)\nfrom sklearn.cluster import KMeans\nfrom sklearn.datasets import load_iris\n\nX, y_true = load_iris(return_X_y=True)\nlabels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)\n\nprint(\"silhouette   :\", round(silhouette_score(X, labels), 3))     # higher better\nprint(\"davies-bould :\", round(davies_bouldin_score(X, labels), 3)) # lower  better\nprint(\"ARI vs truth :\", round(adjusted_rand_score(y_true, labels), 3))\nprint(\"NMI vs truth :\", round(normalized_mutual_info_score(y_true, labels), 3))" },
        { type: "callout", variant: "tip", text: "**Adjusted Mutual Information (AMI)** and **Normalized Mutual Information (NMI)** are the information-theoretic cousins of ARI — they measure how much knowing the cluster tells you about the true class. Report ARI *and* AMI together; they can disagree, and the disagreement is informative (ARI is pair-counting, AMI is entropy-based)." },
        { type: "callout", variant: "warn", text: "**Never tune your model on an external metric and then report that same metric as 'validation'.** If you had the labels to compute ARI, you would usually just do supervised learning. External metrics are for *research benchmarking* ('does my clustering match a known structure?'), not for production model selection where labels genuinely do not exist." },
      ]
    },

    {
      id: "sklearn-showdown",
      title: "scikit-learn worked examples: which algorithm wins where",
      level: "core",
      body: [
        { type: "p", text: "The point of learning four algorithms is knowing which to reach for. Below, each algorithm meets a dataset engineered to be its home turf — and you can see the others stumble on it. All four share scikit-learn's `.fit_predict(X)` interface." },
        { type: "heading", text: "The unified API" },
        { type: "code", lang: "py", code: "from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering\nfrom sklearn.mixture import GaussianMixture\n\n# All return an integer label per point (DBSCAN uses -1 for noise).\nkm  = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)\ngmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)\nagg = AgglomerativeClustering(n_clusters=3, linkage='ward')\ndb  = DBSCAN(eps=0.3, min_samples=5)\n\n# labels = km.fit_predict(X)           # k-means / agglomerative / dbscan\n# labels = gmm.fit(X).predict(X)       # GMM: fit then predict (or fit_predict)\n# probs  = gmm.predict_proba(X)        # <-- soft responsibilities, GMM only" },
        { type: "heading", text: "Blobs — spherical & separated: k-means wins (and is fastest)" },
        { type: "code", lang: "py", code: "from sklearn.datasets import make_blobs\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import adjusted_rand_score\n\nX, y = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=0)\nlabels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(X)\nprint(\"blobs, k-means ARI:\", round(adjusted_rand_score(y, labels), 3))  # ~1.0" },
        { type: "heading", text: "Elongated / rotated blobs: GMM wins (k-means can't tilt)" },
        { type: "code", lang: "py", code: "import numpy as np\nfrom sklearn.mixture import GaussianMixture\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import adjusted_rand_score\n\nX, y = make_blobs(n_samples=500, centers=3, random_state=0)\nX = X @ np.array([[0.6, -0.6], [-0.4, 0.8]])   # stretch+rotate -> ellipses\n\ngmm_lab = GaussianMixture(3, covariance_type='full', random_state=0).fit_predict(X)\nkm_lab  = KMeans(3, n_init=10, random_state=0).fit_predict(X)\nprint(\"stretched, GMM ARI    :\", round(adjusted_rand_score(y, gmm_lab), 3))  # high\nprint(\"stretched, k-means ARI:\", round(adjusted_rand_score(y, km_lab),  3))  # lower" },
        { type: "heading", text: "Moons & noise: DBSCAN wins (arbitrary shape + outliers)" },
        { type: "code", lang: "py", code: "from sklearn.datasets import make_moons\nfrom sklearn.cluster import DBSCAN\nfrom sklearn.metrics import adjusted_rand_score\n\nX, y = make_moons(n_samples=400, noise=0.06, random_state=0)\nlabels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)\nprint(\"moons, DBSCAN ARI:\", round(adjusted_rand_score(y, labels), 3))   # ~1.0\n# k-means on this data scores ~0.25 -- it cuts both crescents in half." },
        { type: "heading", text: "The decision table" },
        { type: "table",
          headers: ["Situation", "Reach for", "Why"],
          rows: [
            ["Round, similar-size clusters; big data; need speed", "**KMeans / MiniBatchKMeans**", "fast, scalable, simple"],
            ["Overlapping / elliptical clusters; want probabilities", "**GaussianMixture**", "soft assignment + rotated covariances + density"],
            ["Arbitrary shapes, outliers, $k$ unknown", "**DBSCAN / HDBSCAN**", "density-based, noise-aware, no $k$"],
            ["Want a taxonomy / dendrogram; small data", "**AgglomerativeClustering**", "interpretable tree, no fixed $k$ up front"],
            ["Non-convex but you want $k$ clusters", "**SpectralClustering**", "k-means in a graph-embedded space"],
          ]
        },
        { type: "callout", variant: "tip", text: "For big data, `MiniBatchKMeans` fits k-means on random mini-batches and is often 10–100× faster than full `KMeans` for a tiny quality loss — the standard choice past ~$10^5$ points. And always wrap clustering in a `Pipeline` with `StandardScaler` so scaling is never forgotten." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Clustering only clicks when you *see* the clusters. Every project below should end in a scatter plot (or an image) colored by cluster label — the plot is the point. Do at least two, and for each, second-guess the result with a silhouette score." },
        { type: "list", ordered: true, items: [
          "**Image color quantization.** Load a photo, reshape it to an $(H\\cdot W, 3)$ array of RGB pixels, and run k-means with $k = 8, 16, 32$ colors. Replace each pixel by its centroid color and reshape back. Plot the reconstructions side by side and report the compression ratio — this is vector quantization, and it makes the 'cluster = codebook entry' idea concrete. (Use `MiniBatchKMeans`; a photo has millions of pixels.)",
          "**Customer segmentation.** Take a retail dataset (e.g. the classic RFM: Recency, Frequency, Monetary value per customer). Standardize, find $k$ with the elbow *and* silhouette (they may disagree — reconcile them), cluster, then *characterize* each segment by its feature means ('high-frequency low-spend', 'whales', 'churned'). The interpretation is the deliverable, not the labels.",
          "**k-means vs the world on hard shapes.** Run k-means, GMM, DBSCAN, and agglomerative on `make_moons`, `make_circles`, and anisotropic blobs. Build a grid of scatter plots (algorithms × datasets) and write one sentence per cell explaining the success or failure via the algorithm's assumption. This single figure teaches the whole section.",
          "**Anomaly detection with a GMM.** Fit a GMM to 'normal' data, then score new points by their density $p(x)$ and flag the lowest-density tail as anomalies. Compare against DBSCAN's $-1$ noise labels on the same data. Where do they agree and disagree?",
          "**Elbow & silhouette from scratch.** Implement inertia and the silhouette coefficient yourself (no `sklearn.metrics`), sweep $k$ from 2 to 10 on a real dataset, and plot both curves. Confirm your silhouette matches `silhouette_score` to the third decimal. Deriving it once beats reading it ten times.",
          "**Cluster embeddings.** Take sentence embeddings (or MNIST digits reduced with PCA/UMAP to 2-D), cluster them with k-means and DBSCAN, and color the 2-D projection by cluster. If you have the true labels, compute ARI to see how much unsupervised structure recovers the real classes.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The math above is the working subset. To go deeper — especially on EM's convergence proof and density-based methods — these are the best sources, in recommended order:" },
        { type: "link", url: "https://scikit-learn.org/stable/modules/clustering.html", text: "scikit-learn — Clustering user guide (the definitive practical comparison, with the famous algorithm-vs-dataset gallery)" },
        { type: "link", url: "https://www.cs.cmu.edu/~aarti/Class/10701/readings/ArthurVassilvitskii2007.pdf", text: "Arthur & Vassilvitskii (2007) — k-means++: The Advantages of Careful Seeding (the initialization everyone now uses)" },
        { type: "link", url: "https://www.cs.columbia.edu/~jebara/4771/tutorials/DempsterLairdRubin.pdf", text: "Dempster, Laird & Rubin (1977) — the original EM algorithm paper (dense but foundational)" },
        { type: "link", url: "https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf", text: "Ester, Kriegel, Sander & Xu (1996) — the original DBSCAN paper (short and readable)" },
        { type: "link", url: "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", text: "Bishop — Pattern Recognition and Machine Learning, Ch. 9 (mixtures & EM, the canonical derivation)" },
        { type: "link", url: "https://hdbscan.readthedocs.io/en/latest/", text: "HDBSCAN docs — the modern, density-varying successor to DBSCAN (and a great tour of density clustering)" },
        { type: "link", url: "https://www.jstatsoft.org/article/view/v053i09", text: "Rousseeuw (1987) & the silhouette method — how to read and use silhouette plots" },
      ]
    },
  ],

  packages: [
    { name: "sklearn.cluster.KMeans", why: "the workhorse — k-means++ init and `n_init` restarts built in" },
    { name: "sklearn.cluster.MiniBatchKMeans", why: "mini-batch k-means for large datasets (10–100× faster)" },
    { name: "sklearn.mixture.GaussianMixture", why: "soft clustering via EM; `predict_proba` gives responsibilities" },
    { name: "sklearn.cluster.DBSCAN", why: "density-based, arbitrary shapes, noise as label $-1$, no $k$" },
    { name: "sklearn.cluster.AgglomerativeClustering", why: "bottom-up hierarchical clustering with selectable linkage" },
    { name: "scipy.cluster.hierarchy", why: "`linkage` + `dendrogram` for the merge tree and its plot" },
    { name: "sklearn.metrics", why: "silhouette, Davies–Bouldin, adjusted Rand, NMI/AMI scores" },
    { name: "sklearn.preprocessing.StandardScaler", why: "scale features first — non-negotiable for distance-based clustering" },
  ],

  gotchas: [
    "**Always scale features first.** A feature in dollars will dominate one in years purely because of its range. `StandardScaler` before clustering, every time.",
    "k-means finds a **local** minimum of inertia — run it with `n_init >= 10` and keep the lowest-inertia result. Plain random init frequently traps a real cluster.",
    "Inertia always decreases as $k$ grows, so **never pick $k$ by minimizing inertia** — use the elbow (a kink) or, better, maximize the silhouette.",
    "k-means assumes **spherical, equal-size** clusters. On moons, rings, or elongated blobs it fails — that is a shape mismatch, not a tuning problem. Switch algorithms.",
    "DBSCAN uses a single global $\\varepsilon$, so it **breaks on clusters of very different densities**. Use the k-distance knee to set $\\varepsilon$, or switch to HDBSCAN.",
    "GMM can **collapse**: a Gaussian can shrink onto a single point, driving $\\Sigma_c \\to 0$ and likelihood $\\to \\infty$. scikit-learn adds `reg_covar` to prevent this — don't set it to $0$.",
    "Agglomerative clustering is $O(m^2)$ memory / up to $O(m^3)$ time and is **greedy** (merges are permanent) — great for thousands of points, hopeless for millions.",
    "External metrics (ARI/NMI) need ground-truth labels — if you have those in production you probably want *supervised* learning. They are for benchmarking, not label-free model selection.",
  ],

  flashcards: [
    { q: "What objective does k-means minimize, and via what algorithm?", a: "Within-cluster sum of squares (inertia) $J = \\sum_i \\sum_c r_{ic}\\lVert x^{(i)} - \\mu_c\\rVert^2$, via Lloyd's algorithm — coordinate descent alternating an assign step and a mean-update step." },
    { q: "Why is k-means guaranteed to converge?", a: "Each step exactly minimizes $J$ over its variable block, so $J$ is monotonically non-increasing and bounded below by $0$. With finitely many assignments, it reaches a fixed point in finite steps — but only a *local* minimum." },
    { q: "Why is random initialization bad, and what does k-means++ do?", a: "Random init often seeds two centroids in one true cluster. k-means++ picks each new seed with probability $\\propto D(x)^2$ (squared distance to the nearest chosen center), spreading seeds apart for an expected $O(\\log k)$-competitive start." },
    { q: "Define the silhouette coefficient of a point.", a: "$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))} \\in [-1,1]$, where $a(i)$ is mean intra-cluster distance and $b(i)$ is mean distance to the nearest other cluster. Near $+1$ = well clustered; $<0$ = probably misassigned." },
    { q: "What is the E-step of EM for a GMM?", a: "Compute responsibilities $\\gamma_{ic} = \\Pr(c \\mid x^{(i)})$ by Bayes' rule: $\\pi_c \\mathcal{N}(x \\mid \\mu_c,\\Sigma_c)$ normalized over all clusters. These are the soft cluster memberships." },
    { q: "How is k-means a special case of a GMM?", a: "Take a GMM with equal spherical covariances $\\Sigma_c = \\sigma^2 I$ and let $\\sigma^2 \\to 0$. The soft responsibilities (a softmax over $-\\lVert x-\\mu_c\\rVert^2 / 2\\sigma^2$) sharpen into hard nearest-centroid assignments — exactly k-means." },
    { q: "What do single vs complete vs Ward linkage tend to produce?", a: "Single (min pairwise distance) → chained/stringy clusters; complete (max) → compact equal-diameter clusters; Ward (min variance increase) → spherical, k-means-like clusters. Ward is the usual default." },
    { q: "What are DBSCAN's two parameters and three point types?", a: "Parameters: $\\varepsilon$ (neighborhood radius) and minPts (density threshold). Point types: core (≥ minPts neighbors within $\\varepsilon$), border (near a core but not dense), noise (label $-1$, in no dense region)." },
    { q: "Name two internal and one external clustering metric.", a: "Internal (no labels): silhouette (higher better) and Davies–Bouldin (lower better). External (needs labels): Adjusted Rand Index, which counts pairwise agreements and corrects for chance, giving $1$ = perfect, $0$ = random." },
    { q: "When should you pick DBSCAN over k-means?", a: "When clusters have arbitrary (non-convex) shapes, when there are outliers you want flagged rather than absorbed, or when you don't know $k$. k-means wins for round, similar-size clusters at scale." },
    { q: "Why must you standardize features before clustering?", a: "Distance-based clustering weights each feature by its numeric range, so an unscaled large-range feature dominates the distance and hence the clusters. StandardScaler (or normalization) equalizes their influence." },
  ],

  cheatsheet: [
    { label: "k-means", code: "KMeans(n_clusters=k, n_init=10).fit_predict(X)" },
    { label: "Mini-batch k-means", code: "MiniBatchKMeans(n_clusters=k).fit_predict(X)" },
    { label: "GMM (soft)", code: "GaussianMixture(n_components=k, covariance_type='full')" },
    { label: "GMM responsibilities", code: "gmm.predict_proba(X)" },
    { label: "DBSCAN", code: "DBSCAN(eps=0.3, min_samples=5).fit_predict(X)" },
    { label: "Agglomerative", code: "AgglomerativeClustering(n_clusters=k, linkage='ward')" },
    { label: "Dendrogram", code: "Z = linkage(X, 'ward'); dendrogram(Z)" },
    { label: "Cut tree into k", code: "fcluster(Z, t=k, criterion='maxclust')" },
    { label: "Inertia (elbow)", code: "KMeans(k).fit(X).inertia_" },
    { label: "Silhouette", code: "silhouette_score(X, labels)" },
    { label: "Davies–Bouldin", code: "davies_bouldin_score(X, labels)" },
    { label: "Adjusted Rand", code: "adjusted_rand_score(y_true, labels)" },
    { label: "Scale first", code: "StandardScaler().fit_transform(X)" },
    { label: "Noise mask (DBSCAN)", code: "labels == -1" },
  ],
});
