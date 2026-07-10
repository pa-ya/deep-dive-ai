(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "nn-cnn",
  name: "Convolutional Neural Networks",
  language: "Neural Networks",
  group: "Neural Networks",
  navLabel: "CNNs (Vision)",
  tagline: "How machines see — convolution derived from scratch, the architectures that won ImageNet (LeNet → ResNet → EfficientNet), and transfer learning that actually ships.",
  color: "#D97706",
  readMinutes: 52,
  sections: [
    {
      id: "why-not-mlp",
      title: "Why a plain MLP fails at images",
      level: "core",
      body: [
        { type: "p", text: "You already know how to build a fully-connected network (an **MLP**): flatten the input into one long vector, multiply by weight matrices, apply non-linearities, backprop. So why not just flatten an image and feed it in? Two reasons, and they are the entire justification for this whole topic." },
        { type: "heading", text: "Reason 1: parameter explosion" },
        { type: "p", text: "A dense layer connects *every* input to *every* output. A modest $224\\times224$ RGB image is a vector of $224\\times224\\times3 = 150{,}528$ numbers. Connect that to a first hidden layer of just $1000$ units and you already have:" },
        { type: "math", tex: String.raw`150{,}528 \times 1000 \;=\; 1.5 \times 10^{8} \text{ weights in one layer.}` },
        { type: "p", text: "That is 150 million parameters *before you have done anything*, for a single small layer. It will overfit instantly, it will not fit in memory at any real depth, and almost all of those weights are wasted — because of reason 2." },
        { type: "heading", text: "Reason 2: no translation invariance" },
        { type: "p", text: "In an MLP, the weight connecting pixel $(10, 10)$ to a hidden unit is *completely independent* of the weight connecting pixel $(11, 10)$. So a cat in the top-left corner and the identical cat shifted 20 pixels right are, to the network, unrelated inputs — it must re-learn 'cat' separately at every location. That is absurd. A feature detector for 'whisker' or 'edge' should work **the same way everywhere in the image**." },
        { type: "table",
          headers: ["Property we want", "Dense layer", "Convolution"],
          rows: [
            ["A detector reused at every location", "no — every position has its own weights", "**yes** — one small kernel slid everywhere"],
            ["Parameters independent of image size", "no — grows with $H\\times W$", "**yes** — fixed by kernel size"],
            ["Locality (nearby pixels relate)", "ignored — input is flattened", "**built in** — kernels are local"],
            ["Translation equivariance", "no", "**yes** — shift input, output shifts too"],
          ]
        },
        { type: "callout", variant: "note", text: "**The core idea in one sentence.** Replace the dense 'every-pixel-to-every-unit' connection with a small, *shared* filter that slides across the image. This bakes in the two priors that images obey — **locality** and **translation invariance** — and collapses 150 million weights down to a few hundred. That shared, sliding filter is a **convolution**." },
        { type: "callout", variant: "tip", text: "This is a running theme in deep learning: the right architecture is one whose *inductive biases* match the structure of the data. CNNs assume locality and translation invariance (great for pixels). RNNs and Transformers assume sequence structure. You will see the same reasoning when you reach the NLP and Transformer tracks." },
      ]
    },

    {
      id: "convolution",
      title: "The convolution operation, derived",
      level: "core",
      body: [
        { type: "p", text: "The operation at the heart of a CNN is, strictly speaking, **2-D cross-correlation** (deep-learning libraries call it 'convolution' anyway — the distinction is a kernel flip that a learned kernel absorbs, so we will not fuss over it). You take a small grid of weights called a **kernel** or **filter**, slide it over the image, and at each position compute a dot product between the kernel and the patch of image underneath it." },
        { type: "heading", text: "The discrete 2-D formula" },
        { type: "p", text: "Let $I$ be the input image and $K$ be an $F\\times F$ kernel. The output feature map $S$ at position $(i,j)$ is the sum, over the kernel window, of each kernel weight times the pixel beneath it:" },
        { type: "math", tex: String.raw`S(i,j) \;=\; (I * K)(i,j) \;=\; \sum_{m=0}^{F-1}\sum_{n=0}^{F-1} I(i+m,\; j+n)\, K(m,n)` },
        { type: "p", text: "Read it slowly: for one output pixel we place the kernel over an $F\\times F$ patch of the image, multiply elementwise, and sum to one number. Then we slide the kernel one step and repeat. Every output pixel uses the **same** kernel weights $K(m,n)$ — that is the weight sharing that kills the parameter explosion." },
        { type: "heading", text: "Kernels are learned feature detectors" },
        { type: "p", text: "The profound part: in a CNN we do **not** hand-design the kernel. We initialize $K$ randomly and let backpropagation learn its values. The network discovers, on its own, that early kernels should detect edges and blobs, that middle kernels should detect textures and parts, and that late kernels should detect whole objects. Convolution provides the *mechanism*; gradient descent provides the *feature engineering*, for free." },
        { type: "heading", text: "A hand-built edge detector" },
        { type: "p", text: "To build intuition for what a kernel *does*, here is one we can design by hand: the vertical **Sobel** filter, a classic edge detector. It computes a horizontal brightness gradient — large where the image goes dark-to-light left-to-right (a vertical edge), near zero on flat regions." },
        { type: "math", tex: String.raw`K_{\text{sobel-}x} = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}` },
        { type: "p", text: "The left column is negative, the right column positive. On a flat patch the positives and negatives cancel to $\\approx 0$. On a vertical edge — dark on the left, bright on the right — the right column lands on bright pixels and the left on dark ones, so the sum is large. The kernel *responds* to the feature it is shaped like." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef conv2d_valid(I, K):\n    \"\"\"2-D cross-correlation, 'valid' (no padding). The literal formula.\"\"\"\n    H, W = I.shape\n    F, _ = K.shape\n    out_h, out_w = H - F + 1, W - F + 1\n    S = np.zeros((out_h, out_w))\n    for i in range(out_h):\n        for j in range(out_w):\n            patch = I[i:i+F, j:j+F]        # the F x F window\n            S[i, j] = np.sum(patch * K)    # elementwise mult, then sum\n    return S\n\n# A tiny image: dark left half, bright right half -> one vertical edge.\nI = np.array([[0, 0, 0, 9, 9, 9]] * 6, dtype=float)\nsobel_x = np.array([[-1, 0, 1],\n                    [-2, 0, 2],\n                    [-1, 0, 1]], dtype=float)\n\nedges = conv2d_valid(I, sobel_x)\nprint(edges.round(1))\n# The response spikes exactly at the dark->bright boundary column,\n# and is ~0 over the flat dark and flat bright regions." },
        { type: "callout", variant: "good", text: "Run this and stare at the output. The big numbers appear precisely where the edge is. A CNN's first layer learns dozens of kernels like this — edges at every orientation, color-opponent blobs — but it *discovers* them from data instead of you writing them by hand. Everything downstream is built from these primitives." },
        { type: "callout", variant: "gotcha", text: "True mathematical convolution flips the kernel before sliding: $S(i,j)=\\sum I(i-m,j-n)K(m,n)$. Frameworks (PyTorch, TF) actually implement cross-correlation (no flip). Because the kernel is *learned*, a flip is just a relabeling of weights, so it makes zero difference to the trained model. Do not lose sleep over it — but know it when you read a signal-processing textbook." },
      ]
    },

    {
      id: "hyperparameters",
      title: "Stride, padding, dilation & the output-size formula",
      level: "core",
      body: [
        { type: "p", text: "A convolution has a few knobs beyond the kernel weights. They control the geometry of the sliding — how far the kernel jumps, whether the edges are handled, and how spread-out the kernel's reach is. Get these wrong and your tensor shapes silently break three layers later." },
        { type: "heading", text: "Stride — how far the kernel jumps" },
        { type: "p", text: "**Stride** $S$ is the step size of the slide. Stride 1 moves one pixel at a time (dense overlap). Stride 2 skips every other position, which **downsamples** the output by roughly $2\\times$ in each dimension — a common way to shrink spatial size and increase the receptive field without a pooling layer." },
        { type: "heading", text: "Padding — what to do at the borders" },
        { type: "p", text: "Without help, convolution shrinks the image (a $3\\times3$ kernel loses a 1-pixel border), and corner pixels get used far less than center pixels. **Padding** adds a border of zeros around the input to fix both. Two conventions:" },
        { type: "list", ordered: false, items: [
          "**'valid'** padding = no padding ($P=0$). The output shrinks; only fully-covered positions are computed.",
          "**'same'** padding = add just enough zeros so the output has the **same** spatial size as the input (for stride 1). For an odd kernel $F$, that means $P = (F-1)/2$ on each side.",
        ]},
        { type: "heading", text: "Dilation — reach without cost" },
        { type: "p", text: "**Dilation** $d$ inserts gaps of $d-1$ pixels *between* kernel elements, so a $3\\times3$ kernel with dilation 2 covers a $5\\times5$ region while still using only 9 weights. It enlarges the receptive field cheaply and is heavily used in segmentation (dilated/atrous convolutions) where you need global context at full resolution." },
        { type: "heading", text: "Deriving the output-size formula" },
        { type: "p", text: "This is the one formula you must be able to reproduce. Consider one dimension of size $W$, a kernel of size $F$, padding $P$ on each side, stride $S$. After padding, the usable width is $W + 2P$. The kernel's left edge can sit at position $0$ and must fit its full width $F$, so its last valid start is $W + 2P - F$. Stepping by $S$, the number of start positions is that span divided by $S$, plus one for the position at $0$:" },
        { type: "math", tex: String.raw`W_{\text{out}} \;=\; \left\lfloor \frac{W - F + 2P}{S} \right\rfloor + 1` },
        { type: "p", text: "With dilation $d$ the kernel's *effective* size grows to $F_{\\text{eff}} = d(F-1)+1$; substitute that for $F$:" },
        { type: "math", tex: String.raw`W_{\text{out}} \;=\; \left\lfloor \frac{W - d(F-1) - 1 + 2P}{S} \right\rfloor + 1` },
        { type: "table",
          headers: ["Config", "$F$", "$P$", "$S$", "$W=32 \\to W_{\\text{out}}$"],
          rows: [
            ["3×3, same", "3", "1", "1", "$32$ (size preserved)"],
            ["3×3, valid", "3", "0", "1", "$30$ (loses border)"],
            ["3×3, stride 2", "3", "1", "2", "$16$ (halved)"],
            ["5×5, valid", "5", "0", "1", "$28$"],
            ["1×1 (pointwise)", "1", "0", "1", "$32$ (channel mixing only)"],
          ]
        },
        { type: "callout", variant: "tip", text: "The $3\\times3$-kernel, stride-1, padding-1 combo is the workhorse of modern CNNs precisely because it preserves spatial size — you can stack dozens of them and only downsample deliberately (with stride-2 convs or pooling). VGG made this pattern famous." },
        { type: "callout", variant: "gotcha", text: "The floor $\\lfloor\\cdot\\rfloor$ matters. If $(W - F + 2P)$ is not divisible by $S$, the kernel can't tile the input evenly and the rightmost pixels are silently dropped. This is a classic source of off-by-one shape mismatches; when a network won't compile, compute this formula by hand for every layer." },
      ]
    },

    {
      id: "channels",
      title: "Channels, feature maps & the full conv layer",
      level: "core",
      body: [
        { type: "p", text: "Real convolutions are 3-D in a subtle way. A color image is not $H\\times W$ — it is $H\\times W\\times C_{\\text{in}}$ with $C_{\\text{in}}=3$ channels (R, G, B). Deeper in the network, 'channels' become the stack of feature maps from the previous layer, often hundreds of them." },
        { type: "heading", text: "A kernel spans all input channels" },
        { type: "p", text: "A single convolutional filter is therefore not $F\\times F$ but $F\\times F\\times C_{\\text{in}}$ — it has a slice for every input channel. At each position it computes a dot product over the *entire depth*, summing across channels into **one** output number. So one filter, sweeping the image, produces **one** 2-D output feature map." },
        { type: "math", tex: String.raw`S(i,j) \;=\; b \;+\; \sum_{c=0}^{C_{\text{in}}-1}\sum_{m=0}^{F-1}\sum_{n=0}^{F-1} I(i+m,\,j+n,\,c)\; K(m,n,c)` },
        { type: "p", text: "To detect many features we use many filters. A layer with $C_{\\text{out}}$ filters produces $C_{\\text{out}}$ feature maps, stacked into an output tensor of shape $H_{\\text{out}}\\times W_{\\text{out}}\\times C_{\\text{out}}$. Each filter has its own **bias** $b$, added after the sum." },
        { type: "heading", text: "The weight tensor & parameter count" },
        { type: "p", text: "The full weight tensor of a conv layer has shape $(C_{\\text{out}},\\, C_{\\text{in}},\\, F,\\, F)$ plus $C_{\\text{out}}$ biases. So the parameter count is:" },
        { type: "math", tex: String.raw`\#\text{params} \;=\; \underbrace{C_{\text{out}} \cdot C_{\text{in}} \cdot F \cdot F}_{\text{weights}} \;+\; \underbrace{C_{\text{out}}}_{\text{biases}}` },
        { type: "p", text: "Crucially, this **does not depend on the image height or width**. Whether the input is $32\\times32$ or $2000\\times2000$, a $3\\times3$ conv from 64 to 128 channels has the same number of weights. Compare that to a dense layer on the same tensors:" },
        { type: "table",
          headers: ["Layer on a $32\\times32\\times64$ input → 128 features", "Parameter count"],
          rows: [
            ["Dense (flatten → 128 units)", "$32\\cdot32\\cdot64\\cdot128 \\approx 8.4$ million"],
            ["$3\\times3$ conv, $64\\to128$ channels", "$128\\cdot64\\cdot9 + 128 \\approx 73.9$ thousand"],
            ["Ratio (the weight-sharing win)", "$\\approx 114\\times$ fewer parameters"],
          ]
        },
        { type: "callout", variant: "note", text: "**Weight sharing, stated precisely.** The same $F\\times F\\times C_{\\text{in}}$ filter is applied at *every* spatial location. So the layer's parameters scale with kernel size and channel counts — never with resolution. This is what makes CNNs trainable on images at all, and it is why a CNN trained on $224\\times224$ can often run on larger images unchanged." },
        { type: "callout", variant: "tip", text: "The **$1\\times1$ convolution** looks trivial but is a workhorse: with $F=1$ it does no spatial mixing, only a learned linear combination *across channels* at each pixel. It's how networks cheaply change channel depth (Inception, ResNet bottlenecks, MobileNet all lean on it)." },
      ]
    },

    {
      id: "pooling",
      title: "Pooling, receptive fields & the modern shift away from pooling",
      level: "core",
      body: [
        { type: "p", text: "Convolutions detect features; **pooling** summarizes them and shrinks the spatial size. A pooling layer slides a window (typically $2\\times2$, stride 2) and replaces each window with a single number — its max or its average." },
        { type: "math", tex: String.raw`\text{maxpool}(i,j) = \max_{0\le m,n < F} I(2i+m,\; 2j+n), \qquad \text{avgpool}(i,j) = \frac{1}{F^2}\sum_{m,n} I(2i+m,\; 2j+n)` },
        { type: "heading", text: "Why pool at all?" },
        { type: "list", ordered: false, items: [
          "**Downsampling** — a $2\\times2$ stride-2 pool quarters the number of activations, cutting compute for deeper layers.",
          "**Local translation invariance** — max-pooling reports 'this feature is present in this region' without caring exactly *where*, so small shifts of the input don't change the output.",
          "**Growing the receptive field** — after pooling, each downstream neuron sees a larger patch of the original image.",
        ]},
        { type: "callout", variant: "gotcha", text: "Pooling has **no learnable parameters**. Max-pooling's backward pass simply routes the incoming gradient to whichever input was the max (the 'argmax'), and sends zero to the rest. Average-pooling spreads the gradient equally. This is worth implementing once so it isn't magic." },
        { type: "heading", text: "Receptive fields" },
        { type: "p", text: "A neuron's **receptive field** is the region of the *original input* that can influence it. It grows as you go deeper: stacking two $3\\times3$ convs gives each output neuron a $5\\times5$ view; three gives $7\\times7$. This is exactly why depth matters — deep layers see large, semantic chunks of the image while early layers see tiny local patches. For stride-1 convs the receptive field grows by $(F-1)$ per layer; strides and pooling multiply it." },
        { type: "callout", variant: "tip", text: "Two stacked $3\\times3$ convs cover the same $5\\times5$ receptive field as one $5\\times5$ conv, but with fewer parameters ($2\\cdot9=18$ vs $25$ per channel-pair) **and** an extra non-linearity in between. This 'small kernels, more depth' insight is the entire thesis of **VGG** and is why $3\\times3$ became the default kernel." },
        { type: "heading", text: "The modern move away from pooling" },
        { type: "p", text: "Newer architectures increasingly *drop* max-pooling in favor of **strided convolutions** — a conv with stride 2 downsamples and learns *how* to downsample, instead of a fixed max. ResNet uses strided convs for most of its downsampling; the 'All-Convolutional Net' (Springenberg et al., 2014) showed you can throw pooling out entirely. What survives almost universally is **global average pooling** at the very end: average each final feature map to one number, replacing the huge dense classifier head that AlexNet/VGG used." },
        { type: "callout", variant: "note", text: "**Global average pooling (GAP)** turns a $7\\times7\\times512$ tensor into a $512$-vector by averaging each map. It has no parameters, resists overfitting far better than a giant dense layer, and makes the network accept variable input sizes. Introduced in 'Network in Network' (2013), it is now standard in ResNet, Inception, and EfficientNet." },
      ]
    },

    {
      id: "from-scratch",
      title: "A conv layer from scratch: the im2col trick, then nn.Conv2d",
      level: "core",
      body: [
        { type: "p", text: "The naive four-nested-loop convolution is correct but painfully slow. The trick every framework uses is **im2col**: unfold every image patch into a column of a big matrix, so that the whole convolution becomes a *single matrix multiplication* — which BLAS/cuDNN execute at hardware speed. Understanding this demystifies how CNNs actually run on a GPU." },
        { type: "heading", text: "The im2col idea" },
        { type: "p", text: "For a kernel of size $F\\times F\\times C_{\\text{in}}$ and $C_{\\text{out}}$ filters: reshape each filter into a row, stacking to a matrix $W$ of shape $(C_{\\text{out}},\\; C_{\\text{in}}FF)$. Extract every input patch, flatten it, and stack the patches as columns of a matrix $X_{\\text{col}}$ of shape $(C_{\\text{in}}FF,\\; H_{\\text{out}}W_{\\text{out}})$. Then the entire convolution is:" },
        { type: "math", tex: String.raw`\text{out} \;=\; W \, X_{\text{col}} \;+\; b, \qquad (C_{\text{out}} \times C_{\text{in}}FF)\,(C_{\text{in}}FF \times H_{\text{out}}W_{\text{out}}) = (C_{\text{out}} \times H_{\text{out}}W_{\text{out}})` },
        { type: "p", text: "Convolution *is* matrix multiplication once you lay the patches out correctly. That single realization is why the linear-algebra track was worth it." },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef im2col(X, F, stride=1):\n    \"\"\"X: (C_in, H, W) -> columns of shape (C_in*F*F, out_h*out_w).\"\"\"\n    C, H, W = X.shape\n    out_h = (H - F) // stride + 1\n    out_w = (W - F) // stride + 1\n    cols = np.zeros((C * F * F, out_h * out_w))\n    idx = 0\n    for i in range(out_h):\n        for j in range(out_w):\n            patch = X[:, i*stride:i*stride+F, j*stride:j*stride+F]\n            cols[:, idx] = patch.reshape(-1)   # flatten (C,F,F) -> vector\n            idx += 1\n    return cols, out_h, out_w\n\ndef conv2d(X, W, b, stride=1):\n    \"\"\"X:(C_in,H,W)  W:(C_out,C_in,F,F)  b:(C_out,) -> (C_out,out_h,out_w).\"\"\"\n    C_out, C_in, F, _ = W.shape\n    cols, out_h, out_w = im2col(X, F, stride)      # (C_in*F*F, L)\n    W_row = W.reshape(C_out, -1)                    # (C_out, C_in*F*F)\n    out = W_row @ cols + b[:, None]                 # one matmul!\n    return out.reshape(C_out, out_h, out_w)\n\nX = np.random.randn(3, 8, 8)         # 3-channel 8x8 input\nW = np.random.randn(4, 3, 3, 3)      # 4 filters, 3x3, over 3 channels\nb = np.zeros(4)\nout = conv2d(X, W, b)\nprint(out.shape)                      # (4, 6, 6)  -> matches (8-3)/1+1 = 6" },
        { type: "heading", text: "The exact same thing in PyTorch" },
        { type: "code", lang: "py", code: "import torch\nimport torch.nn as nn\n\n# nn.Conv2d expects a batch: (N, C_in, H, W)\nx = torch.randn(1, 3, 8, 8)\n\nconv = nn.Conv2d(in_channels=3, out_channels=4,\n                 kernel_size=3, stride=1, padding=0, bias=True)\n\nout = conv(x)\nprint(out.shape)                 # torch.Size([1, 4, 6, 6])\nprint(conv.weight.shape)         # torch.Size([4, 3, 3, 3])  <- (C_out,C_in,F,F)\nprint(conv.bias.shape)           # torch.Size([4])\n\n# param count matches our formula: 4*3*3*3 + 4 = 112\nprint(sum(p.numel() for p in conv.parameters()))   # 112" },
        { type: "callout", variant: "good", text: "The from-scratch `conv2d` and `nn.Conv2d` compute the identical thing — nn.Conv2d just calls a fused cuDNN kernel instead of your Python loop, and it wires up autograd so the backward pass is automatic. Build the scratch version once; use `nn.Conv2d` forever after." },
        { type: "callout", variant: "note", text: "im2col trades memory for speed: it materializes every overlapping patch, so $X_{\\text{col}}$ is $F^2$ times larger than the input. Production libraries (cuDNN) also offer FFT-based and Winograd convolutions that avoid this blow-up for certain kernel sizes — but im2col is the one to understand first." },
      ]
    },

    {
      id: "architectures",
      title: "The classic architectures, as a story",
      level: "core",
      body: [
        { type: "p", text: "You do not need to memorize every architecture, but the *narrative* of how they evolved teaches you what actually matters in CNN design. Each one contributed a single durable idea." },
        { type: "heading", text: "LeNet-5 (1998) — the blueprint" },
        { type: "p", text: "Yann LeCun's **LeNet-5** read handwritten digits (ZIP codes, bank cheques). It established the template still used today: **[conv → non-linearity → pool] blocks, then dense layers**. Tiny by modern standards (~60k params), it proved convolution + backprop could learn vision. Then the field went quiet for over a decade — data and compute weren't there yet." },
        { type: "heading", text: "AlexNet (2012) — the earthquake" },
        { type: "p", text: "**AlexNet** (Krizhevsky, Sutskever, Hinton) won ImageNet 2012 with a top-5 error of 15.3% vs the runner-up's 26.2% — a margin so large it ended the debate and launched the deep-learning era. It scaled LeNet's recipe with three key enablers: training on **GPUs**, the **ReLU** activation (fast, no vanishing saturation), and **dropout** for regularization. This is *the* moment modern AI started." },
        { type: "heading", text: "VGG (2014) — depth via small kernels" },
        { type: "p", text: "**VGG** (Oxford) asked: what if we only ever use $3\\times3$ convs, but go *deep* (16–19 layers)? Its uniform, boring design — stacks of $3\\times3$ convs, double the channels after each pool — made it the go-to feature extractor for years. The lesson: **many small kernels beat few big ones** (same receptive field, fewer params, more non-linearities). Its downside: a huge dense head made it parameter-heavy (~138M)." },
        { type: "heading", text: "ResNet (2015) — the vanishing-gradient fix" },
        { type: "p", text: "Here is the crisis VGG ran into: stack too many layers and accuracy got *worse*, not from overfitting but because gradients **vanish** propagating back through dozens of layers (the exact problem derived in the backprop track's [vanishing/exploding gradients](#nn-backprop) section). **ResNet's** fix is beautifully simple — the **residual connection**. Instead of asking a block to learn a mapping $H(x)$, ask it to learn the *residual* $F(x) = H(x) - x$, and add the input back:" },
        { type: "math", tex: String.raw`y \;=\; F(x, \{W_i\}) \;+\; x` },
        { type: "p", text: "Why this defeats vanishing gradients: the $+x$ term gives the gradient a **direct highway** back through every block. Differentiating, $\\partial y/\\partial x = \\partial F/\\partial x + 1$ — that $+1$ guarantees the gradient never fully dies, no matter how deep. ResNet trained networks of **152 layers** (and later 1000+), won ImageNet 2015, and residual connections are now in *everything*, including Transformers." },
        { type: "callout", variant: "good", text: "The residual connection is arguably the single most important architectural idea in modern deep learning. That humble $+x$ is why we can train networks hundreds of layers deep, and it reappears verbatim inside every Transformer block you will meet in the LLM track. If you remember one thing from this section, remember the $+1$ in the gradient." },
        { type: "heading", text: "Inception / GoogLeNet (2014) — width and efficiency" },
        { type: "p", text: "**Inception** asked a different question: why pick one kernel size? Its **Inception module** runs $1\\times1$, $3\\times3$, and $5\\times5$ convs *in parallel* and concatenates them, letting the network choose the scale per location. It leaned hard on $1\\times1$ convs as cheap 'bottlenecks' to cut channel counts before expensive convs — winning ImageNet 2014 with 12× fewer parameters than VGG." },
        { type: "heading", text: "EfficientNet (2019) — principled scaling" },
        { type: "p", text: "**EfficientNet** (Tan & Le) asked *how* to scale a CNN. Instead of arbitrarily making it deeper or wider, they found a **compound scaling** rule that grows depth, width, and input resolution *together* by fixed ratios. EfficientNet-B7 matched the best accuracy of its day with ~8× fewer parameters and ~6× less compute — it remains a strong default when you care about the accuracy-per-FLOP trade-off." },
        { type: "table",
          headers: ["Architecture", "Year", "The one idea it contributed"],
          rows: [
            ["LeNet-5", "1998", "the conv→pool→dense blueprint"],
            ["AlexNet", "2012", "GPUs + ReLU + dropout at scale (the ImageNet moment)"],
            ["VGG", "2014", "depth via stacked $3\\times3$ kernels"],
            ["Inception", "2014", "multi-scale parallel convs + $1\\times1$ bottlenecks"],
            ["ResNet", "2015", "**residual connections** — trains 100+ layers"],
            ["EfficientNet", "2019", "compound depth/width/resolution scaling"],
          ]
        },
      ]
    },

    {
      id: "transfer-learning",
      title: "Transfer learning & fine-tuning (the practical way to do vision)",
      level: "core",
      body: [
        { type: "p", text: "Here is the most important practical fact in this entire topic: **you will almost never train a CNN from scratch.** Someone has already trained a ResNet or EfficientNet on ImageNet's 1.2 million images, and those learned features — edges, textures, object parts — transfer astonishingly well to *your* problem, even a completely different one. This is **transfer learning**, and it is how real vision projects are built." },
        { type: "heading", text: "Why it works" },
        { type: "p", text: "Early conv layers learn *generic* features (edges, colors, textures) that are useful for basically any image task. Only the last layers are task-specific ('this particular arrangement of parts is a golden retriever'). So we keep the pretrained early layers and only re-learn the final classifier for our classes." },
        { type: "heading", text: "The two modes" },
        { type: "table",
          headers: ["Mode", "What you do", "When"],
          rows: [
            ["**Feature extraction**", "freeze the whole backbone, train only a new classifier head", "small dataset, or task similar to ImageNet"],
            ["**Fine-tuning**", "unfreeze some/all layers, train them at a *small* learning rate", "more data, or a task further from ImageNet"],
          ]
        },
        { type: "heading", text: "Worked example with torchvision" },
        { type: "code", lang: "py", code: "import torch\nimport torch.nn as nn\nfrom torchvision import models\n\n# 1) Load a ResNet-18 pretrained on ImageNet (weights downloaded once).\nmodel = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)\n\n# 2) FEATURE EXTRACTION: freeze every pretrained parameter.\nfor param in model.parameters():\n    param.requires_grad = False\n\n# 3) Replace the final fully-connected layer for OUR number of classes.\n#    resnet18.fc maps 512 features -> 1000 ImageNet classes; we want, say, 10.\nnum_classes = 10\nmodel.fc = nn.Linear(model.fc.in_features, num_classes)\n#    The new fc has requires_grad=True by default, so ONLY it will train.\n\n# 4) Optimizer sees only the params that still require grad (the new head).\noptimizer = torch.optim.Adam(\n    [p for p in model.parameters() if p.requires_grad], lr=1e-3)\n\ntrainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\ntotal = sum(p.numel() for p in model.parameters())\nprint(f\"training {trainable:,} of {total:,} params\")  # ~5k of ~11.2M\n\n# --- To FINE-TUNE instead: unfreeze the last block and use a tiny LR. ---\n# for param in model.layer4.parameters():\n#     param.requires_grad = True\n# optimizer = torch.optim.Adam(\n#     [p for p in model.parameters() if p.requires_grad], lr=1e-4)" },
        { type: "callout", variant: "tip", text: "**Match the preprocessing to the pretrained weights.** A model trained on ImageNet expects inputs normalized with ImageNet's mean/std and resized to what it saw in training. torchvision bundles this: `weights.transforms()` returns the exact preprocessing pipeline. Skip it and your accuracy will mysteriously crater — the network sees inputs from a distribution it never trained on." },
        { type: "callout", variant: "gotcha", text: "When fine-tuning, use a **much smaller learning rate** (e.g. $10^{-4}$ or $10^{-5}$) than for a fresh head. The pretrained weights are already good; a large LR will 'catastrophically forget' the useful features by taking huge gradient steps. A common recipe: train the new head first (frozen backbone), *then* unfreeze and fine-tune everything gently." },
        { type: "callout", variant: "good", text: "Transfer learning routinely reaches strong accuracy on a few hundred labeled images in minutes — a task that would need millions of images and days of GPU time from scratch. For any real vision problem, **start here**, always. Training from scratch is for research and for learning, not for shipping." },
      ]
    },

    {
      id: "beyond-classification",
      title: "Beyond classification: detection, segmentation & a note on ViT",
      level: "core",
      body: [
        { type: "p", text: "Image classification ('what is in this image?') is the entry point, but vision is bigger. The same convolutional backbones power richer tasks by changing what sits on top of them." },
        { type: "heading", text: "Object detection — what, and where" },
        { type: "p", text: "Detection outputs **bounding boxes + labels** for every object. Two lineages:" },
        { type: "list", ordered: false, items: [
          "**Two-stage (R-CNN → Fast R-CNN → Faster R-CNN):** first propose candidate regions, then classify each. Accurate, historically slower.",
          "**One-stage (YOLO, SSD):** 'You Only Look Once' predicts all boxes and classes in a **single forward pass** over a grid. Fast enough for real-time video, and the modern default for most applications.",
        ]},
        { type: "heading", text: "Segmentation — a label for every pixel" },
        { type: "p", text: "Segmentation classifies *each pixel* (road vs car vs pedestrian). The classic architecture is **U-Net** (2015, born in biomedical imaging): an encoder that downsamples to capture context, a symmetric decoder that upsamples back to full resolution, and **skip connections** that copy high-resolution detail from encoder to decoder so fine edges survive. Its U-shape gives it the name — and, notably, the same U-Net encoder-decoder-with-skips design is the backbone of modern **diffusion image generators**." },
        { type: "heading", text: "A note on Vision Transformers" },
        { type: "p", text: "In 2020, the **Vision Transformer (ViT)** showed you can skip convolution entirely: chop the image into patches, treat each patch as a token, and feed them to a plain **Transformer**. Given enough data (or good pretraining), ViTs match or beat CNNs — and they dominate at the largest scales. CNNs remain excellent, more data-efficient on small datasets, and ubiquitous in production; the two are increasingly blended (ConvNeXt, hybrid models). You will meet the attention mechanism that powers ViT in the **Transformers** section of the NLP/LLM track — this is the forward reference to keep in mind." },
        { type: "callout", variant: "note", text: "The through-line: a CNN **backbone** produces feature maps; different **heads** turn those maps into classifications, boxes, or per-pixel masks. Learn the backbone deeply (this topic) and the heads are comparatively small additions. Transfer learning applies to all of them — most detectors and segmenters start from an ImageNet-pretrained backbone." },
      ]
    },

    {
      id: "full-example",
      title: "Full PyTorch example: train a CNN, then transfer-learn",
      level: "core",
      body: [
        { type: "p", text: "Time to put it together. First a small CNN trained from scratch on **FashionMNIST** (28×28 grayscale, 10 classes — like MNIST but harder), then the transfer-learning path with a pretrained ResNet on the same data. This is a complete, runnable training loop." },
        { type: "heading", text: "A small CNN from scratch" },
        { type: "code", lang: "py", code: "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader\nfrom torchvision import datasets, transforms\n\ndevice = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n\ntfm = transforms.Compose([\n    transforms.ToTensor(),\n    transforms.Normalize((0.2860,), (0.3530,)),   # FashionMNIST mean/std\n])\ntrain_ds = datasets.FashionMNIST(\"./data\", train=True,  download=True, transform=tfm)\ntest_ds  = datasets.FashionMNIST(\"./data\", train=False, download=True, transform=tfm)\ntrain_dl = DataLoader(train_ds, batch_size=128, shuffle=True)\ntest_dl  = DataLoader(test_ds,  batch_size=256)\n\nclass SmallCNN(nn.Module):\n    def __init__(self, num_classes=10):\n        super().__init__()\n        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 28x28 -> 28x28\n        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # keeps size (same padding)\n        self.pool  = nn.MaxPool2d(2)                   # halves H,W each call\n        self.fc1   = nn.Linear(64 * 7 * 7, 128)        # 28 ->14 ->7 after 2 pools\n        self.fc2   = nn.Linear(128, num_classes)\n        self.drop  = nn.Dropout(0.25)\n\n    def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))   # -> (32,14,14)\n        x = self.pool(F.relu(self.conv2(x)))   # -> (64, 7, 7)\n        x = torch.flatten(x, 1)                # -> (N, 64*7*7)\n        x = self.drop(F.relu(self.fc1(x)))\n        return self.fc2(x)                     # logits (no softmax; CE does it)\n\nmodel = SmallCNN().to(device)\nopt = torch.optim.Adam(model.parameters(), lr=1e-3)\nloss_fn = nn.CrossEntropyLoss()\n\ndef evaluate():\n    model.eval()\n    correct = total = 0\n    with torch.no_grad():\n        for xb, yb in test_dl:\n            xb, yb = xb.to(device), yb.to(device)\n            preds = model(xb).argmax(1)\n            correct += (preds == yb).sum().item()\n            total += yb.size(0)\n    return correct / total\n\nfor epoch in range(5):\n    model.train()\n    for xb, yb in train_dl:\n        xb, yb = xb.to(device), yb.to(device)\n        opt.zero_grad()\n        loss = loss_fn(model(xb), yb)   # forward\n        loss.backward()                 # autograd fills every .grad\n        opt.step()                      # update weights\n    print(f\"epoch {epoch+1}: test acc = {evaluate():.3f}\")\n# ~5 epochs reaches ~0.90 test accuracy on FashionMNIST." },
        { type: "callout", variant: "gotcha", text: "`nn.CrossEntropyLoss` expects **raw logits**, not softmax probabilities — it fuses `log_softmax` + negative-log-likelihood internally for numerical stability. A very common beginner bug is to apply `softmax` in the model *and* use CrossEntropyLoss, which double-applies it and quietly wrecks training. Output logits; let the loss handle the rest." },
        { type: "heading", text: "The transfer-learning path on the same data" },
        { type: "p", text: "Now the pretrained route. ResNet expects 3-channel $224\\times224$ inputs, so we adapt the grayscale FashionMNIST to fit — in practice you'd use this pattern on your own realistic image dataset." },
        { type: "code", lang: "py", code: "import torch, torch.nn as nn\nfrom torchvision import datasets, transforms, models\nfrom torch.utils.data import DataLoader\n\n# ResNet wants 3x224x224, ImageNet-normalized. Adapt FashionMNIST to match.\ntfm = transforms.Compose([\n    transforms.Resize(224),\n    transforms.Grayscale(num_output_channels=3),   # 1 -> 3 channels\n    transforms.ToTensor(),\n    transforms.Normalize([0.485, 0.456, 0.406],     # ImageNet stats\n                         [0.229, 0.224, 0.225]),\n])\ntrain_dl = DataLoader(datasets.FashionMNIST(\"./data\", train=True, download=True,\n                      transform=tfm), batch_size=64, shuffle=True)\n\ndevice = \"cuda\" if torch.cuda.is_available() else \"cpu\"\nmodel = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)\nfor p in model.parameters():\n    p.requires_grad = False                          # freeze backbone\nmodel.fc = nn.Linear(model.fc.in_features, 10)       # new 10-class head\nmodel = model.to(device)\n\nopt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)   # only the head\nloss_fn = nn.CrossEntropyLoss()\n\nmodel.train()\nfor xb, yb in train_dl:                              # one epoch shown\n    xb, yb = xb.to(device), yb.to(device)\n    opt.zero_grad()\n    loss = loss_fn(model(xb), yb)\n    loss.backward()\n    opt.step()\n# Training only the ~5k-param head, this hits high accuracy in a single epoch\n# because ResNet's frozen features already 'see' shapes and textures." },
        { type: "callout", variant: "good", text: "Run both and compare. The from-scratch CNN takes several epochs to reach ~90%; the frozen-backbone ResNet gets there faster and with almost no trainable parameters, because it stands on ImageNet's shoulders. That contrast is the whole practical argument for transfer learning in one experiment." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Vision is a domain where you learn by *seeing*. Every project below should end with you visualizing something — feature maps, misclassifications, learned kernels. Do at least three; the from-scratch ones build real intuition, the applied ones build real skill." },
        { type: "list", ordered: true, items: [
          "**Convolution from scratch.** Implement `conv2d` with im2col in NumPy (padding + stride), then verify it matches `torch.nn.functional.conv2d` to floating-point tolerance. Then visualize a Sobel and a Gaussian-blur kernel applied to a real photo — *see* what convolution does.",
          "**Beat FashionMNIST.** Take the `SmallCNN` above and push test accuracy past 92%: add batch-norm, data augmentation (`RandomHorizontalFlip`, `RandomCrop`), a learning-rate schedule, and more epochs. Plot train vs test accuracy to watch for (and fix) overfitting.",
          "**Train on CIFAR-10 with a ResNet.** Build a small ResNet (with residual blocks) from scratch for CIFAR-10 (32×32 color, 10 classes). Then compare against a pretrained ResNet fine-tuned on the same data. Which wins, and after how much training?",
          "**Transfer learning on your own images.** Collect ~100 photos each of 3–5 categories you care about (your pets, plants, tools). Fine-tune a pretrained EfficientNet or ResNet. Confirm you get strong accuracy from a tiny dataset — the everyday magic of transfer learning.",
          "**Visualize what the network learned.** For a trained CNN, plot the first-layer kernels as images (they'll look like edge/color detectors) and plot intermediate feature-map activations for a sample image. Then implement **Grad-CAM** to see *which pixels* drove a prediction.",
          "**A tiny object detector or segmenter.** Fine-tune a pretrained YOLO (via `ultralytics`) on a small labeled dataset, or train a U-Net on a segmentation set. Evaluate with the right metric (mAP for detection, IoU for segmentation) — not accuracy.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "In recommended order — the course first for a structured foundation, then the papers to read the primary sources, then fast.ai for the practitioner's top-down path:" },
        { type: "link", url: "https://cs231n.github.io/", text: "Stanford CS231n — Convolutional Neural Networks for Visual Recognition (the canonical course; the notes on convolution and architectures are the best free explanation anywhere)" },
        { type: "link", url: "https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html", text: "Krizhevsky, Sutskever & Hinton (2012) — 'ImageNet Classification with Deep CNNs' (the AlexNet paper; the moment deep learning took over)" },
        { type: "link", url: "https://arxiv.org/abs/1512.03385", text: "He, Zhang, Ren & Sun (2015) — 'Deep Residual Learning for Image Recognition' (the ResNet paper; read it for the residual connection)" },
        { type: "link", url: "https://arxiv.org/abs/1409.1556", text: "Simonyan & Zisserman (2014) — 'Very Deep Convolutional Networks' (the VGG paper; the case for small kernels and depth)" },
        { type: "link", url: "https://arxiv.org/abs/1905.11946", text: "Tan & Le (2019) — 'EfficientNet: Rethinking Model Scaling' (compound scaling, the accuracy-per-FLOP frontier)" },
        { type: "link", url: "https://course.fast.ai/", text: "fast.ai — Practical Deep Learning for Coders (top-down, code-first; get a transfer-learning model working in lesson 1)" },
        { type: "link", url: "https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html", text: "PyTorch — Transfer Learning tutorial (the official, copy-pasteable feature-extraction & fine-tuning recipe)" },
      ]
    },
  ],

  packages: [
    { name: "torch", why: "the tensors, autograd, and `nn.Conv2d` — the core deep-learning engine" },
    { name: "torchvision", why: "pretrained models (`models.resnet18`), datasets (CIFAR/FashionMNIST), and image transforms" },
    { name: "torchvision.transforms", why: "resize/normalize/augment pipelines — and `weights.transforms()` for correct preprocessing" },
    { name: "numpy", why: "implement convolution and im2col from scratch before trusting the framework" },
    { name: "Pillow (PIL)", why: "load and manipulate images; the format torchvision datasets return" },
    { name: "matplotlib", why: "visualize kernels, feature maps, and training curves — essential for vision debugging" },
    { name: "ultralytics", why: "state-of-the-art YOLO object detection in a few lines, for the detection project" },
    { name: "timm", why: "hundreds of pretrained SOTA image models (EfficientNet, ConvNeXt, ViT) beyond torchvision" },
  ],

  gotchas: [
    "`nn.Conv2d` expects a **batched 4-D** tensor $(N, C, H, W)$. Feeding $(C,H,W)$ throws a shape error — add a batch dim with `x.unsqueeze(0)`.",
    "`nn.CrossEntropyLoss` wants **raw logits**, not softmax outputs — it applies `log_softmax` internally. Never softmax before it.",
    "Output size is $\\lfloor (W-F+2P)/S \\rfloor + 1$. If it doesn't divide evenly, the rightmost pixels are silently dropped — compute it by hand when shapes break.",
    "With a pretrained model you **must** match its training preprocessing (ImageNet mean/std, expected resolution) or accuracy collapses. Use `weights.transforms()`.",
    "When fine-tuning, use a tiny learning rate ($10^{-4}$ or less) or you'll catastrophically forget the pretrained features.",
    "Forgetting `model.eval()` at inference leaves dropout and batch-norm in training mode, giving wrong, unstable predictions. Pair it with `torch.no_grad()`.",
    "Max-pool has no parameters; its backward pass routes gradient only to the argmax input. Don't expect it to 'learn' anything.",
    "Convolution is really cross-correlation in every framework (no kernel flip). Harmless because kernels are learned — but it surprises people from signal processing.",
  ],

  flashcards: [
    { q: "Why can't you just flatten an image into an MLP?", a: "Parameter explosion (a small dense layer is ~$10^8$ weights) and no translation invariance (a feature must be re-learned at every location). Convolution fixes both via local, shared filters." },
    { q: "Write the discrete 2-D convolution (cross-correlation) formula.", a: "$S(i,j)=\\sum_m\\sum_n I(i+m, j+n)\\,K(m,n)$ — slide the kernel, elementwise-multiply the patch, and sum to one output pixel." },
    { q: "What is the conv output-size formula, and derive the $+1$.", a: "$W_{\\text{out}}=\\lfloor (W-F+2P)/S \\rfloor + 1$. The last valid kernel start is $W+2P-F$; dividing the span by stride $S$ counts the steps, and $+1$ includes the start position at 0." },
    { q: "How many parameters in a conv layer, and why is it a win?", a: "$C_{\\text{out}}\\cdot C_{\\text{in}}\\cdot F\\cdot F + C_{\\text{out}}$. It's independent of image resolution — weight sharing gives ~100× fewer params than a dense layer on the same tensor." },
    { q: "What does a single conv filter operate over, and what does it output?", a: "It spans all $C_{\\text{in}}$ input channels ($F\\times F\\times C_{\\text{in}}$) and produces one 2-D feature map. $C_{\\text{out}}$ filters give $C_{\\text{out}}$ maps." },
    { q: "What is the im2col trick?", a: "Unfold every input patch into columns of a matrix, reshape filters into rows, and compute the whole convolution as one matrix multiply — so BLAS/cuDNN can run it fast." },
    { q: "What is a residual connection and why does it matter?", a: "$y=F(x)+x$. The $+x$ makes $\\partial y/\\partial x = \\partial F/\\partial x + 1$, giving gradients a highway that never vanishes — it lets ResNet train 100+ layers deep." },
    { q: "What was the significance of AlexNet in 2012?", a: "It won ImageNet by a huge margin (15.3% vs 26.2% top-5 error) using GPUs, ReLU, and dropout — launching the modern deep-learning era." },
    { q: "Feature extraction vs fine-tuning in transfer learning?", a: "Feature extraction freezes the backbone and trains only a new head (small/similar data). Fine-tuning unfreezes layers and trains them at a small LR (more/different data)." },
    { q: "What does global average pooling do and why use it?", a: "Averages each final feature map to one number, replacing a huge dense head — no parameters, less overfitting, and it accepts variable input sizes." },
    { q: "What's the receptive field, and how do two 3×3 convs compare to one 5×5?", a: "The input region influencing a neuron. Two stacked 3×3 convs cover a 5×5 receptive field with fewer parameters and an extra non-linearity — VGG's core insight." },
    { q: "How do detection and segmentation differ from classification?", a: "Detection outputs bounding boxes + labels (YOLO, Faster R-CNN); segmentation labels every pixel (U-Net). Both reuse a CNN backbone with a different head." },
  ],

  cheatsheet: [
    { label: "Conv layer", code: "nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)" },
    { label: "Output size", code: "out = (W - F + 2*P) // S + 1" },
    { label: "Same padding (odd F)", code: "padding = (F - 1) // 2" },
    { label: "Max pool", code: "nn.MaxPool2d(kernel_size=2, stride=2)" },
    { label: "Global avg pool", code: "nn.AdaptiveAvgPool2d((1, 1))" },
    { label: "Batch norm", code: "nn.BatchNorm2d(num_features)" },
    { label: "Flatten for FC head", code: "x = torch.flatten(x, 1)" },
    { label: "Load pretrained", code: "models.resnet18(weights=models.ResNet18_Weights.DEFAULT)" },
    { label: "Freeze backbone", code: "for p in model.parameters(): p.requires_grad = False" },
    { label: "Swap classifier head", code: "model.fc = nn.Linear(model.fc.in_features, num_classes)" },
    { label: "Correct preprocessing", code: "tfm = weights.transforms()" },
    { label: "Classification loss", code: "nn.CrossEntropyLoss()  # feed raw logits" },
    { label: "Add batch dim", code: "x = x.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)" },
    { label: "Inference mode", code: "model.eval(); with torch.no_grad(): ..." },
  ],
});
