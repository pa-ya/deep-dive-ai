(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "ml-svm",
  name: "Support Vector Machines",
  language: "Classical ML",
  group: "Classical Machine Learning",
  navLabel: "SVMs & Kernels",
  tagline: "The most mathematically elegant classical classifier: maximum-margin hyperplanes, Lagrangian duality, and the kernel trick that buys you infinite dimensions for free.",
  color: "#15803D",
  readMinutes: 50,
  sections: [
    {
      id: "intuition",
      title: "The intuition: find the widest street",
      level: "core",
      body: [
        { type: "p", text: "Imagine two clouds of points — positives and negatives — that a straight line can separate. There are *infinitely many* such lines. A perceptron stops at the first one that works; logistic regression finds one that maximizes likelihood. The **Support Vector Machine (SVM)** asks a sharper question: of all separating lines, which one leaves the **widest empty gap** between the classes?" },
        { type: "p", text: "Vladimir Vapnik and Alexey Chervonenkis framed this in the 1960s, and the modern soft-margin, kernelized SVM arrived with Cortes & Vapnik in 1995. For a decade — before deep learning — the SVM was the default high-accuracy classifier for everything from handwritten digits to text categorization. It is still the right tool for small-to-medium, high-dimensional problems, and it is the cleanest place in all of ML to learn convex optimization, duality, and kernels." },
        { type: "callout", variant: "note", text: "**The 'widest street' picture.** Don't think of the boundary as a line — think of it as a *street* with the line running down its middle. The SVM pushes the curbs apart until they touch the nearest points of each class. Those touching points are the only ones that matter; everything else could be deleted without changing the answer." },
        { type: "heading", text: "Why the widest margin generalizes best" },
        { type: "p", text: "A wide margin is a bet about *unseen* data. If the gap between classes is large, a new point drawn from the same distribution has room to land near its class without crossing the boundary — small perturbations don't flip the prediction. A boundary crammed right up against the training points is fragile: nudge a test point slightly and it misclassifies. Statistical learning theory makes this precise — the margin bounds the model's capacity (VC dimension) independently of the number of features, which is exactly why SVMs survive in thousand-dimensional spaces." },
        { type: "heading", text: "Support vectors: the points that hold up the boundary" },
        { type: "p", text: "The **support vectors** are the training points lying exactly on the edge of the street (and, later, the ones inside it). They 'support' the hyperplane the way tent poles support a tent. Remarkably, the final decision function is written *entirely* in terms of these few points — often a tiny fraction of the data. This sparsity is what makes the kernel trick affordable and gives the SVM its name." },
        { type: "table",
          headers: ["Classifier", "What it optimizes", "Boundary depends on"],
          rows: [
            ["Perceptron", "any separating line (first found)", "the mistake sequence"],
            ["Logistic regression", "log-likelihood over **all** points", "every point, weighted"],
            ["SVM", "the **maximum margin**", "only the **support vectors**"],
          ]
        },
        { type: "p", text: "The rest of this deck derives that maximum-margin idea from geometry, turns it into a convex program, passes to its dual (where dot products appear and the kernel trick becomes inevitable), and then builds one from scratch in NumPy." },
      ]
    },

    {
      id: "hard-margin",
      title: "The hard-margin problem: deriving the geometry",
      level: "core",
      body: [
        { type: "p", text: "Assume for now the data is **linearly separable** — a hyperplane can split the two classes with no errors. Label the classes $y_i \\in \\{-1, +1\\}$ (not $\\{0,1\\}$; the symmetry makes the algebra clean). A hyperplane is the set of points $x$ satisfying:" },
        { type: "math", tex: String.raw`w^\top x + b = 0` },
        { type: "p", text: "Here $w$ is the **normal vector** (perpendicular to the plane) and $b$ shifts it off the origin. The classifier predicts the **sign** of $f(x) = w^\\top x + b$: positive is one class, negative the other." },
        { type: "heading", text: "The distance from a point to the hyperplane" },
        { type: "p", text: "A standard fact from geometry: the signed distance from a point $x_i$ to the hyperplane $w^\\top x + b = 0$ is the value of $f$ there, normalized by the length of the normal vector." },
        { type: "math", tex: String.raw`\text{distance}(x_i) = \frac{w^\top x_i + b}{\|w\|}` },
        { type: "p", text: "If $x_i$ is correctly classified, $y_i(w^\\top x_i + b) > 0$, so multiplying by the label makes the distance positive regardless of side. This label-times-score quantity is the **functional margin**; dividing by $\\|w\\|$ gives the **geometric margin** — the actual perpendicular distance:" },
        { type: "math", tex: String.raw`\gamma_i = \frac{y_i\,(w^\top x_i + b)}{\|w\|}` },
        { type: "heading", text: "The scaling ambiguity — and how to kill it" },
        { type: "p", text: "Notice a redundancy: the hyperplane $(w, b)$ and $(2w, 2b)$ describe the *same* plane, but the functional margin doubles. To pin things down we **fix the scale** so that the closest points have functional margin exactly $1$:" },
        { type: "math", tex: String.raw`\min_i\; y_i\,(w^\top x_i + b) = 1` },
        { type: "p", text: "With this normalization, the closest points on each side satisfy $w^\\top x_i + b = \\pm 1$. These two parallel planes are the curbs of the street. The geometric margin — the *half-width* from center to curb — is then simply $1/\\|w\\|$, and the full street width is:" },
        { type: "math", tex: String.raw`\text{margin (full width)} = \frac{2}{\|w\|}` },
        { type: "callout", variant: "tip", text: "**Where $2/\\|w\\|$ comes from, concretely.** Take a point $x_+$ on the $+1$ curb and step along the unit normal $w/\\|w\\|$ until you hit the $-1$ curb at $x_-$. The gap satisfies $w^\\top(x_+ - x_-) = 1 - (-1) = 2$. Since $x_+ - x_-$ is parallel to $w$, that projection *is* $\\|w\\|$ times the distance, so the distance is $2/\\|w\\|$. Widening the street means **shrinking $\\|w\\|$**." },
        { type: "heading", text: "The optimization problem" },
        { type: "p", text: "Maximizing $2/\\|w\\|$ is the same as minimizing $\\|w\\|$, which is the same as minimizing $\\tfrac{1}{2}\\|w\\|^2$ (the square and the $\\tfrac12$ are cosmetic — they make the derivative clean and the objective convex/differentiable). Subject to every point being on the correct side of its curb, we get the **hard-margin SVM**:" },
        { type: "math", tex: String.raw`\min_{w,\,b}\;\; \tfrac{1}{2}\|w\|^2 \qquad \text{subject to}\qquad y_i\,(w^\top x_i + b) \ge 1 \quad \forall i` },
        { type: "callout", variant: "good", text: "**This is a convex quadratic program.** The objective $\\tfrac12\\|w\\|^2$ is a bowl (positive-definite quadratic) and the constraints are linear half-spaces. There is a **unique global minimum** — no local optima, no random restarts, no luck. That guarantee is a big part of why SVMs were trusted in production long before deep nets." },
        { type: "callout", variant: "gotcha", text: "Hard margin **requires perfect separability**. One outlier on the wrong side, and the feasible set is empty — there is no $w$ satisfying all constraints and the problem has *no solution*. Real data is noisy, so we almost never use hard margin directly. The next section fixes this." },
      ]
    },

    {
      id: "soft-margin",
      title: "Soft margin: slack, the C knob, and hinge loss",
      level: "core",
      body: [
        { type: "p", text: "Real data overlaps. We relax the hard constraints by letting each point violate its margin by a **slack** amount $\\xi_i \\ge 0$, then *pay a penalty* for the total violation. This is the **soft-margin SVM** (Cortes & Vapnik, 1995) — the one you actually use." },
        { type: "math", tex: String.raw`\min_{w,\,b,\,\xi}\;\; \tfrac{1}{2}\|w\|^2 + C\sum_{i=1}^{m}\xi_i \qquad \text{s.t.}\qquad y_i\,(w^\top x_i + b) \ge 1 - \xi_i,\;\; \xi_i \ge 0` },
        { type: "p", text: "Read the slack geometrically: $\\xi_i = 0$ means the point is on or beyond its curb (fine). $0 < \\xi_i < 1$ means it's inside the street but still correctly classified. $\\xi_i > 1$ means it's on the *wrong* side of the boundary — a genuine misclassification. The sum $\\sum \\xi_i$ is a soft count of margin violations." },
        { type: "heading", text: "The C hyperparameter: the whole trade-off in one number" },
        { type: "p", text: "$C \\ge 0$ balances the two competing goals — a wide margin (small $\\|w\\|$) versus few violations (small $\\sum\\xi_i$). It is the single most important knob on an SVM:" },
        { type: "table",
          headers: ["$C$", "Priority", "Behaviour", "Risk"],
          rows: [
            ["large", "obey constraints", "narrow margin, few violations", "**overfitting** (approaches hard margin)"],
            ["small", "wide margin", "big margin, tolerates errors", "**underfitting** (ignores some data)"],
          ]
        },
        { type: "callout", variant: "note", text: "**$C$ is an inverse regularization strength.** Big $C$ = weak regularization = trust the data = complex boundary. Small $C$ = strong regularization = smooth, simple boundary. It plays exactly the role that $1/\\lambda$ plays in ridge regression. Always tune it on a validation set (cross-validation), typically over a log grid like $\\{0.01, 0.1, 1, 10, 100\\}$." },
        { type: "heading", text: "The hinge-loss view — SVM is just regularized empirical risk" },
        { type: "p", text: "The slack formulation hides a beautiful equivalence. At the optimum, each $\\xi_i$ is pushed as small as its constraint allows: $\\xi_i = \\max(0,\\; 1 - y_i(w^\\top x_i + b))$. Substitute that back in and the constrained problem becomes an **unconstrained** minimization of a loss plus a penalty:" },
        { type: "math", tex: String.raw`\min_{w,\,b}\;\; \underbrace{\sum_{i=1}^{m}\max\!\big(0,\; 1 - y_i\,f(x_i)\big)}_{\text{hinge loss}} \;+\; \underbrace{\frac{1}{2C}\|w\|^2}_{\text{L2 regularization}}, \qquad f(x)=w^\top x + b` },
        { type: "p", text: "This is the same shape as every other model in the deck: **data-fit term + regularizer**. The SVM's distinctive choice is the **hinge loss** $\\ell(z) = \\max(0, 1 - z)$ where $z = y\\,f(x)$ is the margin. Compare the classification losses:" },
        { type: "table",
          headers: ["Loss", "Formula in margin $z=y f(x)$", "Behaviour"],
          rows: [
            ["0/1 (ideal)", "$\\mathbb{1}[z < 0]$", "non-convex, non-differentiable — unusable"],
            ["Hinge (SVM)", "$\\max(0, 1 - z)$", "convex; **zero loss once $z \\ge 1$**"],
            ["Logistic", "$\\log(1 + e^{-z})$", "convex; never exactly zero"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Why the SVM is sparse.** The hinge loss is *flat zero* for any point comfortably past its margin ($z \\ge 1$). Those points contribute nothing to the gradient — they are not support vectors. Logistic regression's loss is always slightly positive, so *every* point tugs on the boundary. This flat region is precisely why the SVM solution depends on only a handful of points." },
        { type: "callout", variant: "note", text: "The hinge loss is convex but **not differentiable** at the kink $z = 1$. That's fine — it is *subdifferentiable*, and subgradient descent handles it (we use exactly this to train one from scratch below). The kink is the mathematical fingerprint of 'support vector vs. not.'" },
      ]
    },

    {
      id: "duality",
      title: "Lagrangian duality: why only dot products survive",
      level: "core",
      body: [
        { type: "p", text: "We now derive the SVM's **dual** form. This is the conceptual heart of the whole method: the dual reveals that the solution depends on the data *only through dot products* $x_i^\\top x_j$ and *only through the support vectors*. Once you see that, the kernel trick is not a clever hack — it is inevitable." },
        { type: "heading", text: "Step 1 — the Lagrangian" },
        { type: "p", text: "Attach a multiplier $\\alpha_i \\ge 0$ to each margin constraint $y_i(w^\\top x_i + b) - 1 \\ge 0$ (hard margin for clarity; soft margin just adds an upper bound on $\\alpha_i$). The Lagrangian folds the constraints into the objective:" },
        { type: "math", tex: String.raw`\mathcal{L}(w, b, \alpha) = \tfrac{1}{2}\|w\|^2 - \sum_{i=1}^{m}\alpha_i\big[\,y_i(w^\top x_i + b) - 1\,\big]` },
        { type: "heading", text: "Step 2 — stationarity (set gradients to zero)" },
        { type: "p", text: "Minimize over the primal variables $w, b$ by setting the partial derivatives to zero:" },
        { type: "math", tex: String.raw`\frac{\partial \mathcal{L}}{\partial w} = w - \sum_i \alpha_i y_i x_i = 0 \;\;\Longrightarrow\;\; \boxed{\,w = \sum_{i=1}^{m}\alpha_i y_i x_i\,}` },
        { type: "math", tex: String.raw`\frac{\partial \mathcal{L}}{\partial b} = -\sum_i \alpha_i y_i = 0 \;\;\Longrightarrow\;\; \sum_{i=1}^{m}\alpha_i y_i = 0` },
        { type: "callout", variant: "good", text: "**Look at the boxed result.** The optimal weight vector is a *linear combination of the training points*, weighted by $\\alpha_i y_i$. The model doesn't store an abstract $w$ — it stores the data itself. And as we'll see, most $\\alpha_i$ are zero, so it stores only the support vectors." },
        { type: "heading", text: "Step 3 — substitute back to get the dual" },
        { type: "p", text: "Plug $w = \\sum_i \\alpha_i y_i x_i$ back into $\\mathcal{L}$. The $\\tfrac12\\|w\\|^2$ term becomes $\\tfrac12 \\sum_{i,j}\\alpha_i\\alpha_j y_i y_j\\, x_i^\\top x_j$, the cross term produces the same sum with a minus sign, and the $b$ term vanishes by the constraint $\\sum\\alpha_i y_i = 0$. What remains is a problem in the $\\alpha$'s alone:" },
        { type: "math", tex: String.raw`\max_{\alpha}\;\; \sum_{i=1}^{m}\alpha_i - \tfrac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i \alpha_j\, y_i y_j\, \big(x_i^\top x_j\big)` },
        { type: "math", tex: String.raw`\text{subject to}\quad \alpha_i \ge 0,\qquad \sum_{i=1}^{m}\alpha_i y_i = 0 \qquad(\text{soft margin adds } \alpha_i \le C)` },
        { type: "callout", variant: "tip", text: "**The punchline — read the dual carefully.** The training data appears in *exactly one place*: the inner product $x_i^\\top x_j$. Nowhere else. The dimension of $x$ never appears explicitly — only how points relate via dot products. Replace that dot product with any valid similarity function and you have generalized to a new feature space *without ever touching the objective's structure*. That is the kernel trick, and the dual is what makes it possible." },
        { type: "heading", text: "Step 4 — KKT conditions and support vectors" },
        { type: "p", text: "The Karush–Kuhn–Tucker (KKT) **complementary slackness** condition says $\\alpha_i\\big[y_i(w^\\top x_i + b) - 1\\big] = 0$ for every $i$. So for each point, either $\\alpha_i = 0$ **or** the constraint is tight ($y_i f(x_i) = 1$, the point sits on the margin). This dichotomy defines the support vectors:" },
        { type: "table",
          headers: ["Point type", "Multiplier $\\alpha_i$", "Location (soft margin)"],
          rows: [
            ["Non-support vector", "$\\alpha_i = 0$", "outside the margin, correctly classified"],
            ["Margin support vector", "$0 < \\alpha_i < C$", "exactly on the curb ($y_i f = 1$)"],
            ["Bound support vector", "$\\alpha_i = C$", "inside the margin or misclassified"],
          ]
        },
        { type: "p", text: "The decision function, written from $w = \\sum_i \\alpha_i y_i x_i$, becomes a sum over support vectors only (the rest drop out since $\\alpha_i=0$):" },
        { type: "math", tex: String.raw`f(x) = \sum_{i \in \text{SV}} \alpha_i y_i\, \big(x_i^\top x\big) + b` },
        { type: "p", text: "To predict, you take a dot product between the new point and each support vector. This is the exact form the kernel trick will supercharge. The bias $b$ is recovered from any margin support vector ($0<\\alpha_i<C$) via $b = y_i - \\sum_j \\alpha_j y_j (x_j^\\top x_i)$." },
      ]
    },

    {
      id: "kernel-trick",
      title: "The kernel trick: infinite dimensions for free",
      level: "core",
      body: [
        { type: "p", text: "A linear boundary is useless when the classes aren't linearly separable — think of one class forming a ring around the other. The classic fix is to **map the data into a higher-dimensional feature space** $\\phi(x)$ where it *does* separate linearly, then run a linear SVM there. A circle in 2-D becomes a plane in the right 3-D lift." },
        { type: "math", tex: String.raw`x \in \mathbb{R}^n \;\xrightarrow{\;\phi\;}\; \phi(x) \in \mathbb{R}^{D}, \qquad D \gg n \;\;(\text{possibly } \infty)` },
        { type: "p", text: "The problem: computing $\\phi(x)$ explicitly is expensive — or, for infinite-dimensional feature maps, literally impossible. Here is where the dual pays off spectacularly." },
        { type: "heading", text: "The trick" },
        { type: "p", text: "Recall the dual and decision function touch the data *only* through inner products $x_i^\\top x_j$. In the lifted space that becomes $\\phi(x_i)^\\top \\phi(x_j)$. A **kernel** is a function that computes this inner product *directly from the original inputs*, never forming $\\phi$ at all:" },
        { type: "math", tex: String.raw`K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)` },
        { type: "p", text: "Swap every dot product in the dual for $K$, and you are training a linear SVM in a (possibly infinite-dimensional) space while doing all arithmetic in the original low-dimensional one. The decision function becomes:" },
        { type: "math", tex: String.raw`f(x) = \sum_{i \in \text{SV}} \alpha_i y_i\, K(x_i, x) + b` },
        { type: "callout", variant: "good", text: "**Why this feels like magic.** You get the expressive power of an enormous feature space for the *computational cost of the original one*. You never write down $\\phi$. For the RBF kernel, $\\phi$ has infinitely many dimensions — you could never store it — yet $K$ is a one-line exponential. The dual formulation is the only reason this is possible." },
        { type: "heading", text: "The kernels you'll actually use" },
        { type: "table",
          headers: ["Kernel", "$K(x, x')$", "Feature space / use"],
          rows: [
            ["Linear", "$x^\\top x'$", "no lift; text, high-dim sparse data"],
            ["Polynomial", "$(\\gamma\\, x^\\top x' + r)^d$", "all monomials up to degree $d$"],
            ["RBF / Gaussian", "$\\exp(-\\gamma\\,\\|x - x'\\|^2)$", "**infinite**-dim; the default go-to"],
            ["Sigmoid", "$\\tanh(\\gamma\\, x^\\top x' + r)$", "loosely neural-net-like; rarely best"],
          ]
        },
        { type: "heading", text: "The RBF kernel — the workhorse" },
        { type: "p", text: "The Gaussian / **Radial Basis Function** kernel measures similarity as a bump that decays with distance. Two points close together give $K \\approx 1$; far apart, $K \\approx 0$." },
        { type: "math", tex: String.raw`K(x, x') = \exp\!\big(-\gamma\,\|x - x'\|^2\big), \qquad \gamma > 0` },
        { type: "callout", variant: "gotcha", text: "**$\\gamma$ sets the reach of each support vector.** Large $\\gamma$ = narrow bumps = each point influences only its immediate neighborhood = wiggly, overfitting boundary. Small $\\gamma$ = wide bumps = smooth, near-linear boundary that can underfit. $\\gamma$ and $C$ interact, so you must tune them **jointly** (a 2-D grid search). This pair is the entire art of using an RBF SVM." },
        { type: "heading", text: "Mercer's condition — which functions are valid kernels" },
        { type: "p", text: "Not every similarity function is a legal kernel. $K$ must correspond to *some* inner product in *some* feature space, otherwise the dual is no longer a convex problem and the whole guarantee collapses. **Mercer's theorem** gives the test: $K$ is a valid (positive semi-definite) kernel iff for every finite set of points the **Gram matrix** $G_{ij} = K(x_i, x_j)$ is symmetric positive semi-definite ($z^\\top G z \\ge 0$ for all $z$)." },
        { type: "math", tex: String.raw`G_{ij} = K(x_i, x_j) \;\succeq\; 0 \quad\text{(PSD for all point sets)} \;\Longleftrightarrow\; K \text{ is a valid Mercer kernel}` },
        { type: "callout", variant: "tip", text: "**You rarely check Mercer by hand.** Valid kernels are closed under addition, positive scaling, and multiplication, so you build new ones from the known-good primitives (linear, RBF, polynomial). The practical takeaway: the PSD requirement is what preserves convexity — it is why an SVM with a proper kernel still has a unique global optimum." },
      ]
    },

    {
      id: "multiclass-svr",
      title: "Beyond binary: multiclass SVM and regression (SVR)",
      level: "core",
      body: [
        { type: "p", text: "The SVM is fundamentally a **binary, classification** machine. Two standard extensions cover the rest." },
        { type: "heading", text: "Multiclass: combine binary classifiers" },
        { type: "p", text: "There is no single clean multiclass margin, so we decompose a $K$-class problem into many binary ones and vote:" },
        { type: "table",
          headers: ["Scheme", "How", "Classifiers", "Notes"],
          rows: [
            ["One-vs-Rest (OvR)", "each class vs. all others", "$K$", "fewer models; can suffer class imbalance"],
            ["One-vs-One (OvO)", "every pair of classes", "$K(K-1)/2$", "**scikit-learn's default for SVC**; each fit is small & fast"],
          ]
        },
        { type: "p", text: "OvO trains more classifiers but each sees only two classes' data, so on the small subsets SVMs love it is often faster overall. A test point is classified by majority vote across all pairwise duels." },
        { type: "heading", text: "Support Vector Regression (SVR)" },
        { type: "p", text: "For regression the margin idea flips: instead of a *widest empty street between classes*, SVR fits a tube of half-width $\\epsilon$ around the regression function and ignores any error that falls *inside* the tube. This is the **$\\epsilon$-insensitive loss** — you're only penalized for predictions more than $\\epsilon$ off." },
        { type: "math", tex: String.raw`\ell_\epsilon(y, f(x)) = \max\!\big(0,\; |y - f(x)| - \epsilon\big)` },
        { type: "p", text: "As with classification, SVR is kernelizable (RBF-SVR is a strong nonlinear regressor) and sparse — points inside the $\\epsilon$-tube have zero weight, so again only the support vectors define the fit. Two knobs: $\\epsilon$ (tube width, how much error to ignore) and the usual $C$." },
        { type: "callout", variant: "note", text: "**Mental model.** Classification SVM: *maximize the empty margin between two classes.* Regression SVR: *fit the flattest tube that contains most of the data.* Same convex-optimization machinery, same kernel trick, mirror-image geometry." },
      ]
    },

    {
      id: "from-scratch",
      title: "From scratch: linear SVM in NumPy, then scikit-learn RBF",
      level: "core",
      body: [
        { type: "p", text: "We'll train a **linear soft-margin SVM by subgradient descent on the hinge loss** — the unconstrained formulation from the soft-margin section. This is exactly the objective $\\sum_i \\max(0, 1 - y_i f(x_i)) + \\tfrac{1}{2C}\\|w\\|^2$, and it's the algorithm behind large-scale linear SVMs (Pegasos)." },
        { type: "heading", text: "The subgradient" },
        { type: "p", text: "For the regularized objective $J(w,b) = \\tfrac{\\lambda}{2}\\|w\\|^2 + \\tfrac1m\\sum_i \\max(0, 1 - y_i(w^\\top x_i + b))$, the (sub)gradient per point is piecewise: if the point satisfies its margin ($y_i f(x_i) \\ge 1$) only the regularizer contributes; otherwise the hinge term is active." },
        { type: "math", tex: String.raw`\nabla_w J = \lambda w - \frac{1}{m}\sum_{i:\, y_i f(x_i) < 1} y_i x_i, \qquad \nabla_b J = -\frac{1}{m}\sum_{i:\, y_i f(x_i) < 1} y_i` },
        { type: "code", lang: "py", code: "import numpy as np\n\nclass LinearSVM:\n    \"\"\"Soft-margin linear SVM via subgradient descent on the hinge loss.\"\"\"\n    def __init__(self, lambda_=0.01, lr=0.1, epochs=1000):\n        self.lambda_ = lambda_   # L2 strength (= 1/(C*m) up to scaling)\n        self.lr = lr\n        self.epochs = epochs\n\n    def fit(self, X, y):\n        m, n = X.shape\n        y = np.where(y <= 0, -1, 1).astype(float)   # ensure labels are -1/+1\n        self.w = np.zeros(n)\n        self.b = 0.0\n        for epoch in range(self.epochs):\n            scores = y * (X @ self.w + self.b)       # functional margins, shape (m,)\n            active = scores < 1                       # points violating the margin\n            # subgradients\n            grad_w = self.lambda_ * self.w - (X[active] * y[active, None]).sum(0) / m\n            grad_b = -y[active].sum() / m\n            self.w -= self.lr * grad_w\n            self.b -= self.lr * grad_b\n        return self\n\n    def decision_function(self, X):\n        return X @ self.w + self.b\n\n    def predict(self, X):\n        return np.sign(self.decision_function(X))\n\n# --- toy separable-ish data ---\nrng = np.random.default_rng(0)\nX_pos = rng.normal([2, 2], 1.0, size=(50, 2))\nX_neg = rng.normal([-2, -2], 1.0, size=(50, 2))\nX = np.vstack([X_pos, X_neg])\ny = np.hstack([np.ones(50), -np.ones(50)])\n\nsvm = LinearSVM(lambda_=0.01, lr=0.1, epochs=2000).fit(X, y)\nacc = (svm.predict(X) == y).mean()\nprint(f\"train accuracy: {acc:.2%}\")\nprint(\"w =\", svm.w.round(3), \" b =\", round(svm.b, 3))\nprint(\"margin width 2/||w|| =\", round(2 / np.linalg.norm(svm.w), 3))" },
        { type: "callout", variant: "gotcha", text: "This hinge-loss trainer learns $w$ *directly* (the primal). That works for **linear** SVMs but cannot use the kernel trick — kernels live in the **dual** ($\\alpha$'s and dot products). For nonlinear kernels you solve the dual QP, which is what `libsvm` (and therefore scikit-learn's `SVC`) does under the hood via the SMO algorithm. Don't write your own dual QP solver in production — use the library." },
        { type: "heading", text: "The real thing: scikit-learn SVC with an RBF kernel" },
        { type: "p", text: "Now a genuinely non-linear dataset — two interleaving moons — where a linear boundary is hopeless but an RBF SVM carves a curved boundary effortlessly." },
        { type: "code", lang: "py", code: "import numpy as np\nfrom sklearn.datasets import make_moons\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.svm import SVC\n\nX, y = make_moons(n_samples=400, noise=0.25, random_state=0)\nXtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)\n\n# Scaling is MANDATORY for RBF: it uses raw Euclidean distances.\n# The pipeline fits the scaler on train only, preventing leakage.\nclf = make_pipeline(\n    StandardScaler(),\n    SVC(kernel=\"rbf\", C=10.0, gamma=\"scale\"),   # gamma='scale' = 1/(n_features * X.var())\n)\nclf.fit(Xtr, ytr)\n\nprint(\"test accuracy:\", clf.score(Xte, yte))          # ~0.96\nsvc = clf.named_steps[\"svc\"]\nprint(\"n support vectors:\", svc.n_support_, \"of\", len(Xtr))\n# Only a fraction of points are support vectors -> the sparse solution in action." },
        { type: "heading", text: "Drawing the decision boundary" },
        { type: "p", text: "Visualizing the boundary is the fastest way to build intuition for $C$ and $\\gamma$ — change them and watch the curve tighten or smooth." },
        { type: "code", lang: "py", code: "import numpy as np\nimport matplotlib.pyplot as plt\n\n# assumes clf, X, y from the previous cell\nxx, yy = np.meshgrid(\n    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),\n    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300),\n)\ngrid = np.c_[xx.ravel(), yy.ravel()]\nZ = clf.decision_function(grid).reshape(xx.shape)\n\nplt.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], alpha=0.15, colors=[\"#2563eb\", \"#dc2626\"])\nplt.contour(xx, yy, Z, levels=[-1, 0, 1],           # margins at f=-1,0,+1\n            linestyles=[\"--\", \"-\", \"--\"], colors=\"k\", linewidths=[1, 2, 1])\nplt.scatter(X[:, 0], X[:, 1], c=y, cmap=\"coolwarm\", edgecolors=\"k\", s=25)\nplt.title(\"RBF SVM: solid = boundary, dashed = margins\")\nplt.tight_layout(); plt.show()\n# The dashed curves are the street's curbs; points on/inside them are support vectors." },
        { type: "callout", variant: "tip", text: "**Tune $C$ and $\\gamma$ properly** with `GridSearchCV` over a log grid, e.g. `param_grid={'svc__C':[0.1,1,10,100], 'svc__gamma':[0.01,0.1,1,'scale']}`. Never hand-pick them from a single train/test split — the two interact strongly and the optimum is data-specific." },
      ]
    },

    {
      id: "practical",
      title: "Practical notes: scaling, tuning, and when to reach for an SVM",
      level: "core",
      body: [
        { type: "heading", text: "Scaling is mandatory (not optional)" },
        { type: "p", text: "The RBF kernel is a function of $\\|x - x'\\|^2$ and the linear/poly kernels of $x^\\top x'$ — both are **distance/dot-product based**. If one feature ranges over $[0, 1]$ and another over $[0, 100000]$, the large-scale feature drowns out the rest and the kernel effectively ignores the small one. Always standardize (`StandardScaler`) or min-max scale first, fitting the scaler inside a pipeline on the training fold only." },
        { type: "callout", variant: "gotcha", text: "Forgetting to scale is the **single most common SVM mistake**. A perfectly good classifier will score near chance and you'll blame the model. If your SVM 'doesn't work,' check scaling before anything else." },
        { type: "heading", text: "Choosing C and gamma" },
        { type: "list", ordered: false, items: [
          "**$C$** (both kernels): inverse regularization. Low $C$ → wider margin, simpler boundary, more bias. High $C$ → fits training data harder, more variance. Grid over $\\{0.1, 1, 10, 100\\}$.",
          "**$\\gamma$** (RBF/poly): reach of each support vector. Low $\\gamma$ → smooth, near-linear. High $\\gamma$ → wiggly, memorizes points. `gamma='scale'` is a solid default starting point.",
          "Tune the two **jointly** with `GridSearchCV`; they trade off against each other (high $C$ + high $\\gamma$ is the classic overfit corner).",
          "For polynomial kernels also pick the degree $d$ (2–3 is usually plenty; higher degrees overfit and slow training).",
        ]},
        { type: "heading", text: "Cost and scaling with data size" },
        { type: "p", text: "Kernel SVM training is roughly $O(m^2)$ to $O(m^3)$ in the number of examples $m$ (you form and work with the $m\\times m$ Gram matrix). This is the SVM's Achilles' heel." },
        { type: "table",
          headers: ["Dataset size", "SVM verdict"],
          rows: [
            ["$\\lesssim$ 10k rows", "**ideal** — kernel SVM shines, especially high-dimensional"],
            ["10k–100k rows", "use `LinearSVC` / `SGDClassifier(loss='hinge')`, or approximate kernels (Nyström, RFF)"],
            ["$\\gtrsim$ 100k rows", "kernel SVM impractical — prefer gradient-boosted trees or a neural net"],
          ]
        },
        { type: "heading", text: "When SVMs shine — and when they don't" },
        { type: "table",
          headers: ["Situation", "Reach for", "Why"],
          rows: [
            ["Small/medium, high-dimensional (text, genomics)", "**SVM**", "margin theory ignores feature count; robust in $p \\gg m$"],
            ["Clear geometric margin, need max accuracy", "**SVM (RBF)**", "convex, unique optimum, strong small-data generalization"],
            ["Large tabular data, mixed feature types", "gradient-boosted trees", "handle scale/missing/categoricals natively, faster at size"],
            ["Need calibrated probabilities out of the box", "logistic regression / trees", "SVM outputs scores; probabilities need Platt scaling"],
            ["Huge data, images, sequences", "neural networks", "scale to millions of examples; learn features end-to-end"],
          ]
        },
        { type: "callout", variant: "note", text: "**Historical arc.** From ~1995 to ~2012 the RBF SVM was *the* benchmark-topping classifier — it beat neural nets on many tasks and defined the pre-deep-learning era. Deep learning overtook it on large perceptual data (images, audio, text) where representation learning matters. But on small, structured, high-dimensional problems, a well-tuned SVM is still frequently the best and most reliable choice. It never left the toolbox." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Implement at least two of these. The SVM only clicks once you've watched a margin widen and a boundary bend under your own $C$ and $\\gamma$." },
        { type: "list", ordered: true, items: [
          "**Hinge-loss SVM from scratch.** Extend the `LinearSVM` above: log the loss and margin width each epoch and plot them converging. Then vary $\\lambda$ (i.e. $C$) and plot how the margin width and number of violated points trade off — reproduce the $C$ story from the soft-margin section empirically.",
          "**Decision-boundary explorer.** Train `SVC` on `make_moons` and `make_circles`, then draw the boundary for a grid of $(C, \\gamma)$ values in a small-multiples plot. Watch high $\\gamma$ + high $C$ overfit into islands around individual points, and low values collapse to a near-line.",
          "**Kernel from scratch.** Implement the RBF and polynomial kernels as functions, build the Gram matrix, and confirm it is symmetric PSD (all eigenvalues $\\ge 0$ via `np.linalg.eigvalsh`) — verifying Mercer's condition numerically. Feed a precomputed kernel to `SVC(kernel='precomputed')` and match the built-in RBF result.",
          "**Text classification (where SVMs still win).** TF-IDF-vectorize the 20 Newsgroups dataset and classify with `LinearSVC`. Compare accuracy and training time against logistic regression and a small MLP — the linear SVM is typically at or near the top on this high-dimensional sparse data.",
          "**Support-vector inspection.** After fitting an RBF `SVC`, plot the support vectors (`svc.support_vectors_`) on top of the data. Confirm they lie on or inside the margins, and check how `n_support_` grows as you increase noise or $C$ — a direct look at the model's sparsity.",
          "**Grid-search tuning pipeline.** Build a `Pipeline(StandardScaler, SVC)` and run `GridSearchCV` over $C$ and $\\gamma$ with cross-validation on a real dataset (e.g. breast-cancer). Report the best params and the CV heatmap of accuracy over the $(C,\\gamma)$ grid.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The SVM is one of the best-documented algorithms in ML. For the geometry and the derivations, in recommended order:" },
        { type: "link", url: "https://www.statlearning.com/", text: "James, Witten, Hastie & Tibshirani — An Introduction to Statistical Learning (ISLR), Chapter 9 (the gentlest correct treatment; free PDF)" },
        { type: "link", url: "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", text: "Bishop — Pattern Recognition and Machine Learning, Chapter 7 (sparse kernel machines; the full dual + kernel derivation)" },
        { type: "link", url: "https://cs229.stanford.edu/notes2022fall/main_notes.pdf", text: "Andrew Ng — Stanford CS229 lecture notes (the SVM / kernels sections; the classic careful derivation of margins and duality)" },
        { type: "link", url: "https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf", text: "Hsu, Chang & Lin — A Practical Guide to Support Vector Classification (the definitive 'how to actually use one': scaling, RBF, C/gamma grid search)" },
        { type: "link", url: "https://scikit-learn.org/stable/modules/svm.html", text: "scikit-learn — Support Vector Machines user guide (SVC/SVR/LinearSVC APIs, kernel options, tips)" },
        { type: "link", url: "https://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf", text: "MIT 6.034 — SVM notes (a clean, self-contained walk from margins to the dual)" },
        { type: "link", url: "https://www.csie.ntu.edu.tw/~cjlin/libsvm/", text: "LibSVM (Chang & Lin) — the reference implementation scikit-learn wraps; the SMO solver behind kernel SVMs" },
      ]
    },
  ],

  packages: [
    { name: "sklearn.svm.SVC", why: "kernel SVM classifier (RBF/poly/linear); the one you reach for on small/medium data" },
    { name: "sklearn.svm.LinearSVC", why: "linear SVM scaled to larger data (liblinear); much faster than `SVC(kernel='linear')`" },
    { name: "sklearn.svm.SVR", why: "support vector **regression** with the $\\epsilon$-insensitive tube" },
    { name: "sklearn.preprocessing.StandardScaler", why: "mandatory feature scaling — SVMs are distance/dot-product based" },
    { name: "sklearn.pipeline.make_pipeline", why: "chain scaler + SVM so scaling is fit on train folds only (no leakage)" },
    { name: "sklearn.model_selection.GridSearchCV", why: "joint cross-validated tuning of $C$ and $\\gamma$" },
    { name: "sklearn.linear_model.SGDClassifier", why: "`loss='hinge'` = linear SVM by SGD; scales to 100k+ rows" },
    { name: "numpy", why: "implement the hinge-loss subgradient trainer and Gram matrices from scratch" },
  ],

  gotchas: [
    "**Always scale features.** RBF/linear kernels use raw distances/dot products; an unscaled large-range feature dominates and the SVM scores near chance. This is the #1 SVM bug.",
    "Use labels $\\{-1, +1\\}$ in the math (and when hand-rolling hinge loss), not $\\{0, 1\\}$ — the margin $y_i f(x_i)$ relies on the symmetric signs.",
    "$C$ is **inverse** regularization: large $C$ overfits (narrow margin), small $C$ underfits (wide margin). It is not a learning rate.",
    "$C$ and $\\gamma$ interact strongly — tune them **jointly** on a 2-D grid. High $C$ + high $\\gamma$ is the classic overfitting corner.",
    "The kernel trick lives in the **dual**. A primal hinge-loss trainer learns $w$ directly and can only do **linear** SVM; nonlinear kernels need the $\\alpha$/dot-product form.",
    "Kernel SVM training is $O(m^2)$–$O(m^3)$ in examples $m$. Past ~50–100k rows switch to `LinearSVC`/`SGDClassifier` or gradient-boosted trees.",
    "`SVC` outputs a signed **score**, not a probability. Set `probability=True` (Platt scaling, slower, refits) only if you truly need calibrated probabilities.",
    "`SVC(kernel='linear')` uses the slow kernel path; for genuinely linear problems use `LinearSVC` (liblinear) — often 10–100× faster.",
  ],

  flashcards: [
    { q: "What does an SVM maximize, and why does that generalize well?", a: "The **margin** — the width $2/\\|w\\|$ of the empty street between classes. A wide gap means test points have room to fall on the correct side, bounding capacity independently of feature count." },
    { q: "Write the hard-margin SVM optimization problem.", a: "$\\min_{w,b}\\tfrac12\\|w\\|^2$ subject to $y_i(w^\\top x_i + b) \\ge 1$ for all $i$. A convex QP with a unique global optimum." },
    { q: "What are slack variables and $C$ in the soft-margin SVM?", a: "$\\xi_i \\ge 0$ let points violate the margin; $C$ weights the total violation $\\sum\\xi_i$ against margin width. Large $C$ → overfit, small $C$ → underfit. $C$ is inverse regularization." },
    { q: "State the hinge loss and its key property.", a: "$\\ell(z) = \\max(0, 1 - z)$ with $z = y\\,f(x)$. It is **exactly zero** once $z \\ge 1$, so well-classified points contribute nothing — this is why the SVM solution is sparse (support vectors only)." },
    { q: "In the dual SVM, how does the training data enter?", a: "Only through inner products $x_i^\\top x_j$, and only via the support vectors ($\\alpha_i > 0$). $w = \\sum_i \\alpha_i y_i x_i$." },
    { q: "What is the kernel trick?", a: "Replace $x_i^\\top x_j$ with $K(x_i,x_j) = \\phi(x_i)^\\top\\phi(x_j)$ — computing inner products in a high/infinite-dimensional feature space directly, without ever forming $\\phi$." },
    { q: "Give the RBF kernel and describe $\\gamma$.", a: "$K(x,x') = \\exp(-\\gamma\\|x-x'\\|^2)$. $\\gamma$ sets each support vector's reach: large $\\gamma$ = narrow, wiggly, overfit; small $\\gamma$ = wide, smooth, underfit." },
    { q: "What is Mercer's condition?", a: "$K$ is a valid kernel iff its Gram matrix $G_{ij}=K(x_i,x_j)$ is symmetric positive semi-definite for every point set. This preserves convexity of the dual." },
    { q: "How does an SVM do multiclass classification?", a: "It doesn't natively — it combines binary SVMs: One-vs-Rest ($K$ classifiers) or One-vs-One ($K(K-1)/2$, scikit-learn's default for SVC) and votes." },
    { q: "How does SVR differ from classification SVM?", a: "It fits a tube of half-width $\\epsilon$ around the function using the $\\epsilon$-insensitive loss $\\max(0,|y-f(x)|-\\epsilon)$ — errors inside the tube cost nothing." },
    { q: "Why must you scale features before an SVM?", a: "Kernels depend on distances/dot products; an unscaled large-range feature dominates and the kernel ignores the others. Standardize inside a pipeline (train folds only)." },
    { q: "When is an SVM the right choice vs. trees or neural nets?", a: "SVM: small/medium, high-dimensional data (text, genomics), clear margins. Trees: large mixed tabular. Neural nets: huge perceptual data (images/audio/text at scale)." },
  ],

  cheatsheet: [
    { label: "RBF classifier", code: "SVC(kernel='rbf', C=10, gamma='scale')" },
    { label: "Linear (fast, big data)", code: "LinearSVC(C=1.0)" },
    { label: "Polynomial kernel", code: "SVC(kernel='poly', degree=3, C=1)" },
    { label: "Regression", code: "SVR(kernel='rbf', C=10, epsilon=0.1)" },
    { label: "Scale + SVM pipeline", code: "make_pipeline(StandardScaler(), SVC())" },
    { label: "Tune C and gamma", code: "GridSearchCV(pipe, {'svc__C':[.1,1,10], 'svc__gamma':[.01,.1,1]})" },
    { label: "Signed score", code: "clf.decision_function(X)" },
    { label: "Probabilities (Platt)", code: "SVC(probability=True)" },
    { label: "Support vectors", code: "svc.support_vectors_ , svc.n_support_" },
    { label: "SVM by SGD (huge data)", code: "SGDClassifier(loss='hinge', alpha=1e-4)" },
    { label: "Precomputed kernel", code: "SVC(kernel='precomputed').fit(gram, y)" },
    { label: "Margin width", code: "2 / np.linalg.norm(w)" },
    { label: "Hinge loss", code: "np.maximum(0, 1 - y * (X @ w + b))" },
    { label: "RBF kernel value", code: "np.exp(-gamma * np.sum((x - xp)**2))" },
  ],
});
