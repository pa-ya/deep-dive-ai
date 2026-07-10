(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "math-linear-algebra",
  name: "Linear Algebra",
  language: "Math Foundations",
  group: "Math Foundations",
  navLabel: "Linear Algebra",
  tagline: "The language ML is written in — vectors, matrices, eigenvalues & SVD, built up from scratch and in NumPy.",
  color: "#6366F1",
  readMinutes: 52,
  sections: [
    {
      id: "why",
      title: "Why linear algebra runs machine learning",
      level: "core",
      body: [
        { type: "p", text: "Every machine-learning system, from a 1958 perceptron to a modern LLM, is at its core a pile of **linear algebra with a few non-linearities sprinkled in**. A dataset is a matrix. A model's parameters are matrices. Training is repeated matrix multiplication and differentiation. If you understand linear algebra deeply, most of ML stops being magic and becomes *bookkeeping*." },
        { type: "p", text: "Concretely, here is how the objects show up:" },
        { type: "table",
          headers: ["ML thing", "Linear-algebra object", "Shape"],
          rows: [
            ["One example (e.g. a house)", "a **vector** $x \\in \\mathbb{R}^n$ of $n$ features", "$(n,)$"],
            ["A whole dataset of $m$ examples", "a **matrix** $X \\in \\mathbb{R}^{m \\times n}$", "$(m, n)$"],
            ["A linear model's weights", "a **vector** $w \\in \\mathbb{R}^n$", "$(n,)$"],
            ["All predictions at once", "a **matrix-vector product** $Xw$", "$(m,)$"],
            ["A neural-net layer", "$a = \\phi(Wx + b)$", "$W$ is $(\\text{out}, \\text{in})$"],
            ["An embedding table", "a matrix; each row is a token vector", "$(\\text{vocab}, d)$"],
          ]
        },
        { type: "p", text: "That fifth row — $Wx + b$ followed by a non-linear $\\phi$ — *is* a neural network layer. Stack a few hundred of them and you have GPT. So we start here." },
        { type: "callout", variant: "note", text: "**How to read this deck.** Math blocks are rendered with real notation. Every idea is shown three ways: the **intuition** (what it *means*), the **math** (the exact statement + a short derivation), and the **code** (NumPy from scratch, then the one-liner). Do the derivations by hand once — reading them is not the same as being able to reproduce them." },
      ]
    },

    {
      id: "vectors",
      title: "Vectors: points, directions, and the dot product",
      level: "core",
      body: [
        { type: "p", text: "A **vector** is an ordered list of numbers. Geometrically it is an arrow from the origin, or equivalently a point in $n$-dimensional space. In ML a vector is almost always *one example* or *one set of weights*." },
        { type: "math", tex: String.raw`x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n` },
        { type: "heading", text: "Addition and scaling — the two operations that define a vector space" },
        { type: "p", text: "A **vector space** is any set closed under two operations: adding two vectors, and multiplying a vector by a scalar. That closure is the whole definition, and it is why linear models compose so cleanly." },
        { type: "math", tex: String.raw`(x + y)_i = x_i + y_i, \qquad (\alpha x)_i = \alpha\, x_i` },
        { type: "heading", text: "The dot product — the single most useful operation in ML" },
        { type: "p", text: "The **dot product** (inner product) multiplies two vectors elementwise and sums the result, collapsing two vectors into one number:" },
        { type: "math", tex: String.raw`x \cdot y \;=\; x^\top y \;=\; \sum_{i=1}^{n} x_i y_i` },
        { type: "p", text: "This one scalar is doing a lot of work. A linear model's prediction $\\hat y = w^\\top x$ is *literally a dot product* between weights and features. It also has a geometric meaning:" },
        { type: "math", tex: String.raw`x^\top y = \|x\|\,\|y\|\cos\theta` },
        { type: "p", text: "where $\\theta$ is the angle between the vectors. So the dot product measures **alignment**: positive when the vectors point the same way, zero when they are perpendicular (orthogonal), negative when opposed. Cosine similarity — the backbone of search, embeddings, and RAG — is exactly this formula rearranged." },
        { type: "heading", text: "Norms — how long is  a vector?" },
        { type: "p", text: "A **norm** measures length. The Euclidean ($L_2$) norm is the familiar one; the $L_1$ norm sums absolute values and shows up in Lasso regularization:" },
        { type: "math", tex: String.raw`\|x\|_2 = \sqrt{\sum_i x_i^2} = \sqrt{x^\top x}, \qquad \|x\|_1 = \sum_i |x_i|` },
        { type: "callout", variant: "tip", text: "**Why $L_2$ vs $L_1$ matters later:** squaring in $L_2$ punishes large errors far more than small ones (this is why least-squares regression exists), while $L_1$ treats all errors proportionally and drives weights *exactly* to zero (this is why Lasso does feature selection). Same idea, two norms." },
        { type: "heading", text: "From scratch, then the one-liner" },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef dot(x, y):\n    \"\"\"Dot product from first principles.\"\"\"\n    return sum(xi * yi for xi, yi in zip(x, y))\n\ndef l2_norm(x):\n    return dot(x, x) ** 0.5\n\ndef cosine_similarity(x, y):\n    return dot(x, y) / (l2_norm(x) * l2_norm(y))\n\nx = np.array([1.0, 2.0, 3.0])\nw = np.array([0.5, -1.0, 2.0])\n\nprint(dot(x, w))            # 4.5   (hand-rolled)\nprint(x @ w)                # 4.5   (NumPy: @ is matmul/dot)\nprint(np.linalg.norm(x))    # 3.7416...\nprint(cosine_similarity(x, w))" },
        { type: "callout", variant: "gotcha", text: "In NumPy, `x @ y` is the dot product for 1-D arrays and matrix multiplication for 2-D. `x * y` is **elementwise** and is almost never what you want when you mean a dot product. Mixing these up is the #1 shape bug for beginners." },
      ]
    },

    {
      id: "matrices",
      title: "Matrices as linear maps & matrix multiplication",
      level: "core",
      body: [
        { type: "p", text: "A **matrix** is a rectangular grid of numbers, but the useful way to think of it is as a **function that transforms vectors linearly** — it rotates, scales, shears, and projects space. Multiplying by a matrix is applying that transformation." },
        { type: "math", tex: String.raw`A \in \mathbb{R}^{m \times n} \quad\text{maps}\quad x \in \mathbb{R}^n \;\longmapsto\; Ax \in \mathbb{R}^m` },
        { type: "heading", text: "Matrix–vector product: a weighted combination of columns" },
        { type: "p", text: "There are two equally correct readings of $Ax$. **Row view:** each output entry is a dot product of a row of $A$ with $x$. **Column view:** the output is a linear combination of the *columns* of $A$, weighted by the entries of $x$. The column view is the one that unlocks intuition later (it is why 'the column space' matters)." },
        { type: "math", tex: String.raw`(Ax)_i = \sum_{j=1}^{n} A_{ij} x_j \qquad\Longleftrightarrow\qquad Ax = \sum_{j=1}^{n} x_j\, a_{:,j}` },
        { type: "heading", text: "Matrix–matrix product" },
        { type: "p", text: "Multiplying $A \\in \\mathbb{R}^{m\\times n}$ by $B \\in \\mathbb{R}^{n\\times p}$ gives $C \\in \\mathbb{R}^{m\\times p}$, where each entry is a dot product of a row of $A$ with a column of $B$. The inner dimensions must match ($n = n$); the outer dimensions become the result's shape." },
        { type: "math", tex: String.raw`C_{ik} = \sum_{j=1}^{n} A_{ij} B_{jk}, \qquad (m\times n)(n\times p) = (m \times p)` },
        { type: "callout", variant: "gotcha", text: "**Matrix multiplication is not commutative:** $AB \\neq BA$ in general (often the shapes don't even allow both). It *is* associative — $A(BC) = (AB)C$ — and this associativity is quietly the reason backpropagation works: you can regroup a long chain of matrix multiplications." },
        { type: "heading", text: "Special matrices you must recognize on sight" },
        { type: "table",
          headers: ["Matrix", "Definition", "Why it matters"],
          rows: [
            ["Identity $I$", "$1$s on diagonal, else $0$", "$IA = AI = A$; the 'do nothing' map"],
            ["Diagonal", "non-zero only on diagonal", "scales each axis independently; cheap to invert"],
            ["Symmetric", "$A = A^\\top$", "covariance matrices; guarantees real eigenvalues"],
            ["Orthogonal $Q$", "$Q^\\top Q = I$", "pure rotation/reflection; preserves lengths & angles"],
            ["Positive definite", "$x^\\top A x > 0\\;\\forall x\\neq 0$", "bowl-shaped loss; unique minimum exists"],
          ]
        },
        { type: "heading", text: "Implement matmul from scratch to feel the cost" },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef matmul(A, B):\n    m, n = A.shape\n    n2, p = B.shape\n    assert n == n2, f\"inner dims must match: {n} != {n2}\"\n    C = np.zeros((m, p))\n    for i in range(m):\n        for k in range(p):\n            for j in range(n):          # the triple loop = O(m*n*p)\n                C[i, k] += A[i, j] * B[j, k]\n    return C\n\nA = np.random.randn(4, 3)\nB = np.random.randn(3, 5)\nprint(np.allclose(matmul(A, B), A @ B))   # True\n# A @ B calls into BLAS (optimized C/Fortran) — orders of magnitude faster." },
        { type: "callout", variant: "tip", text: "The triple loop is $O(mnp)$. This is why GPUs matter: they run thousands of these multiply-adds in parallel. Every 'the model is training' progress bar is really 'BLAS is grinding through matmuls.'" },
      ]
    },

    {
      id: "core-ops",
      title: "Transpose, inverse, rank, determinant & trace",
      level: "core",
      body: [
        { type: "heading", text: "Transpose" },
        { type: "p", text: "The **transpose** $A^\\top$ flips rows and columns: $(A^\\top)_{ij} = A_{ji}$. It appears everywhere — $w^\\top x$, the normal equations, covariance $X^\\top X$. Two rules to memorize: $(A^\\top)^\\top = A$ and, crucially, the product reverses order:" },
        { type: "math", tex: String.raw`(AB)^\top = B^\top A^\top` },
        { type: "heading", text: "Inverse — undoing a linear map" },
        { type: "p", text: "The **inverse** $A^{-1}$ is the matrix that undoes $A$: applying one then the other returns you to where you started. It only exists for square matrices that don't collapse any dimension (non-singular)." },
        { type: "math", tex: String.raw`A A^{-1} = A^{-1} A = I \qquad\Longrightarrow\qquad Ax = b \;\Rightarrow\; x = A^{-1}b` },
        { type: "callout", variant: "gotcha", text: "**Never actually compute an inverse to solve $Ax=b$ in code.** It is slower and numerically worse than solving the system directly. Use `np.linalg.solve(A, b)`, not `np.linalg.inv(A) @ b`. The inverse is a *conceptual* tool; `solve` is the *practical* one." },
        { type: "heading", text: "Rank — how much real information a matrix carries" },
        { type: "p", text: "The **rank** is the number of linearly independent columns (equivalently, rows) — the true dimensionality of the space the matrix can reach. A matrix is **full rank** if no column is a combination of the others. **Rank deficiency** is the mathematical name for redundant features, and it is exactly why perfectly correlated inputs break linear regression (the matrix $X^\\top X$ becomes non-invertible)." },
        { type: "heading", text: "Determinant & trace" },
        { type: "p", text: "The **determinant** $\\det(A)$ is the factor by which the map scales volume; $\\det(A) = 0$ means the map squashes space flat (singular, no inverse). The **trace** $\\operatorname{tr}(A)$ is the sum of the diagonal, equal to the sum of eigenvalues — a quantity that survives inside many derivations." },
        { type: "math", tex: String.raw`\det(A) = 0 \iff A \text{ is singular} \iff \operatorname{rank}(A) < n, \qquad \operatorname{tr}(A) = \sum_i A_{ii} = \sum_i \lambda_i` },
        { type: "code", lang: "py", code: "import numpy as np\n\nA = np.array([[4.0, 3.0],\n              [6.0, 3.0]])\nb = np.array([10.0, 12.0])\n\nx = np.linalg.solve(A, b)          # solve Ax=b directly (preferred)\nprint(x)                           # [1. 2.]\nprint(np.linalg.matrix_rank(A))    # 2  (full rank)\nprint(np.linalg.det(A))            # -6.0 (nonzero -> invertible)\nprint(np.trace(A))                 # 7.0" },
      ]
    },

    {
      id: "least-squares",
      title: "Linear systems & the least-squares normal equations",
      level: "core",
      body: [
        { type: "p", text: "This section is the payoff: we derive **linear regression** as pure linear algebra. You have $m$ data points and want the best line/hyperplane $\\hat y = Xw$. Usually there are more equations than unknowns ($m > n$), so no exact solution exists — the system is *overdetermined*. We settle for the $w$ that minimizes squared error." },
        { type: "math", tex: String.raw`\min_{w}\; \|Xw - y\|_2^2 \;=\; \min_w\; (Xw - y)^\top (Xw - y)` },
        { type: "heading", text: "Derivation of the normal equations" },
        { type: "p", text: "Expand the objective $J(w)$, take its gradient with respect to $w$, and set it to zero. (The gradient rules used here — $\\nabla_w\\, w^\\top A w = 2Aw$ for symmetric $A$, and $\\nabla_w\\, b^\\top w = b$ — are derived in the Calculus section.)" },
        { type: "math", tex: String.raw`J(w) = w^\top X^\top X w - 2\, y^\top X w + y^\top y` },
        { type: "math", tex: String.raw`\nabla_w J = 2 X^\top X w - 2 X^\top y \;\overset{!}{=}\; 0` },
        { type: "math", tex: String.raw`\boxed{\; X^\top X\, w = X^\top y \quad\Longrightarrow\quad w = (X^\top X)^{-1} X^\top y \;}` },
        { type: "p", text: "That boxed result is the **normal equation**. Every linear-regression library computes some numerically-stable version of it. The term $(X^\\top X)^{-1}X^\\top$ is the **Moore–Penrose pseudo-inverse** $X^{+}$ — the closest thing to an inverse for a non-square matrix." },
        { type: "callout", variant: "note", text: "**Geometric meaning:** $Xw$ can only ever land in the column space of $X$. The best $w$ is the one where $Xw$ is the **orthogonal projection** of $y$ onto that column space — the residual $Xw - y$ is perpendicular to every column of $X$, which is precisely what $X^\\top(Xw - y) = 0$ says. Least squares is projection." },
        { type: "code", lang: "py", code: "import numpy as np\n\n# Fit y ~= w0 + w1*x by least squares, three equivalent ways.\nx = np.array([1, 2, 3, 4, 5], dtype=float)\ny = np.array([2.1, 3.9, 6.2, 7.8, 10.1])\n\nX = np.column_stack([np.ones_like(x), x])   # add bias column -> shape (5, 2)\n\n# 1) normal equation, literally (educational, not for production)\nw_normal = np.linalg.inv(X.T @ X) @ X.T @ y\n\n# 2) pseudo-inverse\nw_pinv = np.linalg.pinv(X) @ y\n\n# 3) the right way: lstsq (stable, uses SVD internally)\nw_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)\n\nprint(w_normal)   # ~[0.06, 1.99]  ->  y ~= 0.06 + 1.99*x\nprint(np.allclose(w_normal, w_lstsq))   # True" },
        { type: "callout", variant: "gotcha", text: "If two features are collinear, $X^\\top X$ is singular and method (1) explodes. Methods (2) and (3) survive because the SVD handles rank deficiency gracefully. This is a preview of *why* SVD is the workhorse of numerical ML." },
      ]
    },

    {
      id: "eigen",
      title: "Eigenvalues & eigenvectors",
      level: "core",
      body: [
        { type: "p", text: "Most vectors, when hit by a matrix $A$, change both length and direction. A special few only get **scaled** — their direction is preserved. Those are the **eigenvectors**, and the scaling factor is the **eigenvalue**. They reveal the 'natural axes' of a transformation." },
        { type: "math", tex: String.raw`A v = \lambda v, \qquad v \neq 0` },
        { type: "p", text: "Rearranging gives $(A - \\lambda I)v = 0$, which has a non-zero solution only when the matrix $A - \\lambda I$ is singular — i.e. when its determinant vanishes. That yields the **characteristic polynomial**:" },
        { type: "math", tex: String.raw`\det(A - \lambda I) = 0` },
        { type: "heading", text: "Eigendecomposition" },
        { type: "p", text: "If an $n\\times n$ matrix has $n$ independent eigenvectors, stack them as columns of $V$ and put the eigenvalues on the diagonal of $\\Lambda$. Then $A$ factorizes — and this makes powers of $A$ trivial, which is what dynamical systems and PageRank exploit:" },
        { type: "math", tex: String.raw`A = V \Lambda V^{-1} \qquad\Longrightarrow\qquad A^k = V \Lambda^k V^{-1}` },
        { type: "callout", variant: "tip", text: "**Symmetric matrices are special.** If $A = A^\\top$ (like every covariance matrix), its eigenvalues are real and its eigenvectors are orthogonal, so $A = Q\\Lambda Q^\\top$ with $Q$ orthogonal. This is the **spectral theorem**, and it is the mathematical seed of PCA." },
        { type: "code", lang: "py", code: "import numpy as np\n\nA = np.array([[2.0, 0.0],\n              [0.0, 3.0]])\nvals, vecs = np.linalg.eig(A)\nprint(vals)          # [2. 3.]  -> eigenvalues\nprint(vecs)          # columns are eigenvectors\n\n# Verify A v = lambda v for the first eigenpair\nl, v = vals[0], vecs[:, 0]\nprint(np.allclose(A @ v, l * v))   # True\n\n# For symmetric matrices use eigh: faster and guarantees real output\nC = np.cov(np.random.randn(100, 3).T)   # a 3x3 covariance matrix\neigvals, eigvecs = np.linalg.eigh(C)     # ascending eigenvalues" },
      ]
    },

    {
      id: "svd",
      title: "Singular Value Decomposition — the crown jewel",
      level: "core",
      body: [
        { type: "p", text: "Eigendecomposition only works for square matrices. The **SVD** generalizes it to *any* matrix and is, without exaggeration, the most important factorization in applied ML: it powers PCA, recommender systems, latent-semantic analysis, low-rank model compression, and pseudo-inverses." },
        { type: "math", tex: String.raw`A = U \Sigma V^\top, \qquad A \in \mathbb{R}^{m\times n}` },
        { type: "p", text: "Here $U$ ($m\\times m$) and $V$ ($n\\times n$) are **orthogonal** (their columns are unit-length and mutually perpendicular), and $\\Sigma$ is diagonal with non-negative **singular values** $\\sigma_1 \\ge \\sigma_2 \\ge \\dots \\ge 0$. Geometrically, every linear map is a **rotation, then a scaling along axes, then another rotation** — that is all any matrix ever does." },
        { type: "heading", text: "Low-rank approximation — the reason SVD is everywhere" },
        { type: "p", text: "Keep only the top $k$ singular values and their vectors, and you get the **best possible rank-$k$ approximation** of $A$ (this optimality is the Eckart–Young theorem). That single fact is data compression, denoising, and dimensionality reduction all at once:" },
        { type: "math", tex: String.raw`A_k = \sum_{i=1}^{k} \sigma_i\, u_i v_i^\top \;\approx\; A, \qquad k \ll \min(m,n)` },
        { type: "callout", variant: "note", text: "**SVD ↔ PCA.** The principal components of centered data $X$ are exactly the right-singular vectors $V$ of $X$, and the variance explained by each is $\\sigma_i^2/(m-1)$. When you run PCA in the ML section, you are running an SVD. Same math, different story." },
        { type: "code", lang: "py", code: "import numpy as np\n\n# Compress a matrix (think: a grayscale image) with a rank-k SVD.\nA = np.random.randn(50, 30)\nU, s, Vt = np.linalg.svd(A, full_matrices=False)\nprint(U.shape, s.shape, Vt.shape)     # (50,30) (30,) (30,30)\n\nk = 5\nA_k = (U[:, :k] * s[:k]) @ Vt[:k, :]  # rank-5 reconstruction\n\nrel_error = np.linalg.norm(A - A_k) / np.linalg.norm(A)\nkept_energy = (s[:k]**2).sum() / (s**2).sum()\nprint(f\"rank-{k}: {rel_error:.1%} error, {kept_energy:.1%} variance kept\")\n# Storing U[:, :k], s[:k], Vt[:k] costs k*(m+n+1) numbers instead of m*n." },
        { type: "callout", variant: "tip", text: "**Modern relevance:** LoRA — the standard way to fine-tune LLMs cheaply — freezes the big weight matrix and learns a *low-rank* update $\\Delta W = BA$. That is the low-rank idea from this section applied to a 70-billion-parameter model. You will meet it again in the LLM track." },
      ]
    },

    {
      id: "quadratic-forms",
      title: "Quadratic forms & positive definiteness (bridge to optimization)",
      level: "deep",
      body: [
        { type: "p", text: "A **quadratic form** is a scalar-valued function $f(x) = x^\\top A x$. It generalizes 'a parabola' to many dimensions, and it is the shape of essentially every loss surface near its minimum. Understanding its curvature tells you whether gradient descent will behave." },
        { type: "math", tex: String.raw`f(x) = x^\top A x = \sum_{i}\sum_{j} A_{ij} x_i x_j` },
        { type: "p", text: "The matrix $A$ (its eigenvalues, specifically) decides the geometry:" },
        { type: "table",
          headers: ["Condition on $A$", "Name", "Loss-surface shape"],
          rows: [
            ["$x^\\top A x > 0\\;\\forall x\\neq 0$ (all $\\lambda_i > 0$)", "positive definite", "bowl — a unique minimum"],
            ["$x^\\top A x \\ge 0$ (all $\\lambda_i \\ge 0$)", "positive semidefinite", "bowl with a flat valley"],
            ["mixed-sign eigenvalues", "indefinite", "saddle — the villain of deep learning"],
          ]
        },
        { type: "p", text: "Second-order optimization uses the **Hessian** (the matrix of second derivatives). When it is positive definite you are at a minimum; when it is indefinite you are at a saddle point. High-dimensional loss landscapes are riddled with saddles, which is why momentum and Adam exist — you will derive them in the Neural Networks track." },
        { type: "callout", variant: "tip", text: "**The condition number** $\\kappa = \\lambda_{\\max}/\\lambda_{\\min}$ of the Hessian predicts how slowly plain gradient descent converges. A stretched, ill-conditioned bowl (large $\\kappa$) makes gradient descent zig-zag. This single number motivates feature scaling, batch norm, and adaptive optimizers." },
      ]
    },

    {
      id: "numpy-fluency",
      title: "NumPy fluency: broadcasting, axes & vectorization",
      level: "core",
      body: [
        { type: "p", text: "You will write ML in NumPy (and PyTorch, which copies NumPy's API). Two ideas separate slow, loop-ridden code from fast, idiomatic code: **vectorization** and **broadcasting**." },
        { type: "heading", text: "Broadcasting" },
        { type: "p", text: "Broadcasting lets NumPy combine arrays of different shapes by virtually stretching size-1 dimensions. It is how you add a bias vector to a whole batch, or standardize every column, without a single loop." },
        { type: "code", lang: "py", code: "import numpy as np\n\nX = np.random.randn(1000, 4)      # 1000 examples, 4 features\n\n# Standardize each column: subtract mean, divide by std — no loops.\nmu = X.mean(axis=0)               # shape (4,)\nsigma = X.std(axis=0)            # shape (4,)\nXz = (X - mu) / sigma             # (1000,4) - (4,) broadcasts down the rows\nprint(Xz.mean(axis=0).round(6))   # ~[0 0 0 0]\nprint(Xz.std(axis=0).round(6))    # ~[1 1 1 1]\n\n# Add a per-example bias with an explicit new axis\nb = np.arange(1000).reshape(-1, 1)   # (1000,1)\nprint((X + b).shape)                 # (1000,4)  -> b broadcasts across columns" },
        { type: "callout", variant: "gotcha", text: "**`axis` is the dimension you collapse, not the one you keep.** `X.mean(axis=0)` on a `(rows, cols)` array averages *down the rows* and returns one value per column (shape `(cols,)`). Beginners reliably get this backwards. When broadcasting fails, print `.shape` first — 90% of NumPy bugs are shape mismatches." },
        { type: "heading", text: "The rule, precisely" },
        { type: "p", text: "Align shapes from the **right**. Two dimensions are compatible when they are equal or one of them is $1$. A missing leading dimension is treated as $1$. That is the entire broadcasting algorithm." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Do at least two of these by hand before moving on. Reading linear algebra builds *recognition*; implementing it builds *fluency*. Everything here uses only NumPy." },
        { type: "list", ordered: true, items: [
          "**Least-squares from scratch.** Generate noisy points on a known line, then recover the slope/intercept three ways: the normal equation, `np.linalg.lstsq`, and gradient descent. Plot all three fits. Confirm they agree, then break the normal equation on purpose by adding a duplicate (collinear) feature and watch which methods survive.",
          "**Image compression with SVD.** Load a grayscale image as a matrix, compute its SVD, and reconstruct it at $k = 5, 20, 50$ singular values. Plot compression ratio vs. visual quality, and plot the singular-value spectrum on a log scale to *see* where the information lives.",
          "**PCA by hand.** Take a 3-feature dataset (or the Iris dataset), center it, compute the covariance matrix, take its eigendecomposition with `eigh`, and project onto the top 2 eigenvectors. Then verify you get the identical result via SVD of the centered data. This is the whole of PCA in ~15 lines.",
          "**A tiny linear-map visualizer.** Write a function that takes a 2×2 matrix and plots what it does to the unit square and unit circle. Feed it a rotation, a shear, a scaling, and a singular matrix (det = 0). Watch the singular matrix collapse the circle to a line segment.",
          "**Power iteration.** Implement the power method — repeatedly multiply a random vector by a symmetric matrix and normalize — to find the top eigenvector without calling `eig`. Confirm it converges to what `eigh` reports. (This is the ancestor of the algorithm behind PageRank.)",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "This section covers everything you need to proceed. If you want the material to *click* at a deeper level — especially the geometry — these are the best resources, in recommended order:" },
        { type: "link", url: "https://www.3blue1brown.com/topics/linear-algebra", text: "3Blue1Brown — Essence of Linear Algebra (the visual intuition; watch this first, it is the best on the internet)" },
        { type: "link", url: "https://math.mit.edu/~gs/linearalgebra/", text: "Gilbert Strang — Introduction to Linear Algebra + MIT 18.06 lectures (the canonical course)" },
        { type: "link", url: "https://mml-book.github.io/", text: "Mathematics for Machine Learning (Deisenroth, Faisal, Ong) — free PDF, the exact subset ML needs" },
        { type: "link", url: "https://numpy.org/doc/stable/user/basics.broadcasting.html", text: "NumPy broadcasting — the official rules, worth reading once carefully" },
        { type: "link", url: "https://www.deeplearningbook.org/contents/linear_algebra.html", text: "Goodfellow et al., Deep Learning — Chapter 2 (linear algebra for DL, concise)" },
      ]
    },
  ],

  packages: [
    { name: "numpy", why: "arrays, matmul (`@`), and `np.linalg` — the foundation of every ML library" },
    { name: "np.linalg.solve", why: "solve $Ax=b$ stably — never invert a matrix by hand" },
    { name: "np.linalg.lstsq", why: "least squares via SVD; survives rank deficiency" },
    { name: "np.linalg.svd", why: "the crown-jewel factorization — PCA, compression, pseudo-inverse" },
    { name: "np.linalg.eigh", why: "eigendecomposition for symmetric matrices (covariance) — fast & real" },
    { name: "scipy.linalg", why: "richer decompositions (QR, Cholesky, LU) when you outgrow NumPy" },
  ],

  gotchas: [
    "`x @ y` is dot/matmul; `x * y` is **elementwise**. Confusing them silently produces wrong-shaped results.",
    "Never `np.linalg.inv(A) @ b` — use `np.linalg.solve(A, b)`. Inversion is slower and numerically unstable.",
    "`axis=0` collapses **rows** (one result per column); `axis=1` collapses columns. Easy to get backwards.",
    "Matrix multiplication is **not commutative**: $AB \\neq BA$. Order matters, and half the time only one order has valid shapes.",
    "Collinear (linearly dependent) features make $X^\\top X$ singular — the normal equation blows up but `lstsq`/SVD survive.",
    "Broadcasting failures are shape bugs. Print `.shape` before assuming; align dimensions from the right.",
    "Floating-point means `A @ np.linalg.inv(A)` is only *approximately* $I$. Compare with `np.allclose`, never `==`.",
  ],

  flashcards: [
    { q: "What does the dot product $x^\\top y$ measure geometrically?", a: "Alignment: $x^\\top y = \\|x\\|\\|y\\|\\cos\\theta$. Zero means orthogonal; it is the basis of cosine similarity." },
    { q: "Write the least-squares normal equation for $\\min_w \\|Xw-y\\|^2$.", a: "$X^\\top X\\,w = X^\\top y$, so $w = (X^\\top X)^{-1}X^\\top y$. The residual is orthogonal to the column space of $X$." },
    { q: "What is an eigenvector?", a: "A non-zero vector whose direction is unchanged by $A$: $Av = \\lambda v$. Only its length scales, by the eigenvalue $\\lambda$." },
    { q: "What does the SVD $A = U\\Sigma V^\\top$ say every matrix does?", a: "A rotation ($V^\\top$), then an axis-aligned scaling ($\\Sigma$), then another rotation ($U$). It exists for any matrix." },
    { q: "Why is a low-rank SVD truncation useful?", a: "Keeping the top $k$ singular values gives the provably best rank-$k$ approximation (Eckart–Young): compression, denoising, PCA, and LoRA." },
    { q: "Why should you never compute a matrix inverse to solve $Ax=b$?", a: "It is slower and numerically unstable. Solve the system directly (`np.linalg.solve`)." },
    { q: "What makes a matrix positive definite, and why care?", a: "$x^\\top A x > 0$ for all $x\\neq0$ (all eigenvalues positive). It means a bowl-shaped surface with a unique minimum — a well-behaved loss." },
    { q: "How does PCA relate to SVD?", a: "The principal components are the right-singular vectors $V$ of the centered data; variance explained is $\\sigma_i^2/(m-1)$." },
  ],

  cheatsheet: [
    { label: "Dot / matmul", code: "x @ y" },
    { label: "Elementwise", code: "x * y" },
    { label: "Transpose", code: "A.T" },
    { label: "Solve Ax=b", code: "np.linalg.solve(A, b)" },
    { label: "Least squares", code: "np.linalg.lstsq(X, y, rcond=None)" },
    { label: "SVD", code: "U, s, Vt = np.linalg.svd(A)" },
    { label: "Eigen (symmetric)", code: "np.linalg.eigh(A)" },
    { label: "Norm", code: "np.linalg.norm(x)" },
    { label: "Rank", code: "np.linalg.matrix_rank(A)" },
    { label: "Standardize cols", code: "(X - X.mean(0)) / X.std(0)" },
    { label: "Add bias column", code: "np.column_stack([np.ones(len(x)), x])" },
    { label: "Pseudo-inverse", code: "np.linalg.pinv(X)" },
  ],
});
