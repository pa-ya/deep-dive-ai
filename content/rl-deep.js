(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "rl-deep",
  name: "Deep RL & RLHF",
  language: "Reinforcement Learning",
  group: "Reinforcement Learning",
  navLabel: "Deep RL & RLHF",
  tagline: "From tabular Q-learning to neural policies — DQN, policy gradients, PPO, and the exact recipe that turns a base LLM into ChatGPT.",
  color: "#2563EB",
  readMinutes: 52,
  sections: [
    {
      id: "why-deep",
      title: "Why go deep: function approximation & the deadly triad",
      level: "core",
      body: [
        { type: "p", text: "In **RL Foundations** we solved control by filling in a table: one cell per state (or per state–action pair) holding a value $V(s)$ or $Q(s,a)$. Tabular methods are exact and provably convergent, but they die the instant the state space gets large. A chessboard has $\\sim 10^{43}$ positions; an Atari screen is $210 \\times 160$ pixels with 128 colors, i.e. $128^{33600}$ possible frames; a robot's joint angles are *continuous*. You cannot store a table with a row per state, and even if you could, you would never visit each state enough times to estimate its value." },
        { type: "p", text: "The fix is the same idea that powers all of supervised learning: **function approximation**. Instead of a lookup table, use a parameterized function — a neural network — that *generalizes* across states. Feed it a state and it outputs values or action probabilities. Learning from one state now improves the estimate at all similar states, because they share parameters." },
        { type: "table",
          headers: ["Tabular RL", "Deep RL", "What changed"],
          rows: [
            ["$V(s)$ is a stored number", "$V_\\theta(s)$ is a network output", "storage → generalization"],
            ["$Q(s,a)$ is a table cell", "$Q_\\theta(s,a)$ predicted from raw state", "no feature engineering"],
            ["update one cell", "gradient step on shared $\\theta$", "one update touches many states"],
            ["exact, small problems", "approximate, huge/continuous problems", "scale at the cost of guarantees"],
          ]
        },
        { type: "callout", variant: "note", text: "**This deck continues [RL Foundations](#/rl-foundations).** You should already know MDPs, returns $G_t = \\sum_k \\gamma^k r_{t+k}$, the Bellman equations, and tabular Q-learning / TD learning. Here we swap the table for a network and follow the thread all the way to how modern LLMs are aligned." },
        { type: "heading", text: "The deadly triad — why deep RL is hard" },
        { type: "p", text: "Supervised learning is stable because its targets are *fixed* labels. RL is not: the target you regress toward is computed from your own current estimate (bootstrapping), and it moves as you learn. Sutton & Barto name three ingredients that, when combined, can make value-based learning **diverge** — the so-called **deadly triad**:" },
        { type: "list", ordered: true, items: [
          "**Bootstrapping** — updating an estimate from other estimates (TD targets like $r + \\gamma \\max_a Q(s',a)$), rather than from true returns. Efficient, but the target chases a moving prediction.",
          "**Off-policy learning** — learning about the greedy policy while collecting data from a different (exploratory) policy. The data distribution no longer matches what the policy would generate.",
          "**Function approximation** — a network ties states together; nudging $Q$ for one state perturbs it for others, and errors can amplify instead of averaging out.",
        ]},
        { type: "p", text: "Any one or two of these is fine. All three together can send values to infinity. Every algorithm in this deck is, in part, a set of **tricks to tame the triad** — target networks freeze the bootstrap target, replay buffers de-correlate data, trust regions and clipping bound how far a policy can move per step. Keep the triad in mind; it explains *why* the machinery looks the way it does." },
        { type: "callout", variant: "tip", text: "**Two families, one goal.** Deep RL splits into **value-based** methods (learn $Q_\\theta$, act greedily — DQN and friends) and **policy-based** methods (learn a policy $\\pi_\\theta$ directly — REINFORCE, PPO). **Actor–critic** fuses them. RLHF, the payoff at the end, is policy-based (PPO) with a learned reward. We build up in exactly that order." },
      ]
    },

    {
      id: "dqn",
      title: "DQN: deep Q-networks that beat Atari",
      level: "core",
      body: [
        { type: "p", text: "**Deep Q-Networks (DQN)** were the 2013–2015 breakthrough from DeepMind that made 'deep RL' a phrase. A single architecture learned to play 49 Atari 2600 games from raw pixels, reaching human level on many, with *no game-specific tuning* — same network, same hyperparameters, reward = the game score. It was the proof that neural function approximation could carry classic RL to hard, high-dimensional problems." },
        { type: "heading", text: "The Q-network" },
        { type: "p", text: "Recall tabular Q-learning updates $Q(s,a) \\leftarrow Q(s,a) + \\alpha[\\,r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)\\,]$. DQN replaces the table with a network $Q_\\theta(s,a)$. For Atari the input is the last 4 grayscale frames (to encode motion), passed through a small CNN; the output layer has one unit *per action*, giving all $Q(s,\\cdot)$ in a single forward pass." },
        { type: "math", tex: String.raw`Q_\theta:\; s \;\longmapsto\; \big(Q_\theta(s,a_1),\, Q_\theta(s,a_2),\, \dots,\, Q_\theta(s,a_{|\mathcal{A}|})\big)` },
        { type: "heading", text: "The loss — a regression toward the Bellman target" },
        { type: "p", text: "We want $Q_\\theta$ to satisfy the Bellman optimality equation, so we regress it toward the one-step target $y = r + \\gamma \\max_{a'} Q(s',a')$. The loss is squared TD error over sampled transitions $(s,a,r,s')$:" },
        { type: "math", tex: String.raw`\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')}\Big[\big(\underbrace{r + \gamma \max_{a'} Q_{\theta^-}(s',a')}_{\text{target } y} - Q_\theta(s,a)\big)^2\Big]` },
        { type: "p", text: "Two problems make naive training diverge — and DQN's two famous tricks fix exactly them." },
        { type: "heading", text: "Trick 1 — Experience replay" },
        { type: "p", text: "Consecutive RL transitions are **highly correlated** (frame $t$ and $t{+}1$ look almost identical), which violates the i.i.d. assumption SGD relies on and lets the network overfit to whatever it is doing right now. **Experience replay** stores transitions in a large buffer (e.g. 1M) and trains on *random minibatches* drawn from it. This de-correlates updates and reuses each experience many times — far more sample-efficient." },
        { type: "heading", text: "Trick 2 — Target network" },
        { type: "p", text: "Notice the target $y$ depends on the network's own weights. If the target moves every gradient step, you are chasing a moving goalpost — a direct hit from the deadly triad. DQN keeps a **separate target network** $Q_{\\theta^-}$ whose weights $\\theta^-$ are a *frozen copy* of $\\theta$, synced only every $C$ steps (e.g. every 10k). The target is stationary between syncs, which stabilizes the bootstrap." },
        { type: "callout", variant: "gotcha", text: "The $\\theta^-$ subscript on the target is the whole point. If you write the target using the live $\\theta$ (a common beginner bug), training oscillates or blows up. The target network is a *stop-gradient*: you never backprop through $y$." },
        { type: "code", lang: "py", code: "import torch, torch.nn as nn, random\nfrom collections import deque\n\nclass QNet(nn.Module):\n    def __init__(self, n_obs, n_act):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(n_obs, 128), nn.ReLU(),\n            nn.Linear(128, 128), nn.ReLU(),\n            nn.Linear(128, n_act))       # one Q-value per action\n    def forward(self, s):\n        return self.net(s)\n\nq, q_target = QNet(4, 2), QNet(4, 2)\nq_target.load_state_dict(q.state_dict())     # start identical\nbuffer = deque(maxlen=100_000)               # experience replay\nopt = torch.optim.Adam(q.parameters(), lr=1e-3)\ngamma = 0.99\n\ndef learn(batch_size=64):\n    if len(buffer) < batch_size: return\n    s, a, r, s2, done = zip(*random.sample(buffer, batch_size))\n    s  = torch.tensor(s,  dtype=torch.float32)\n    s2 = torch.tensor(s2, dtype=torch.float32)\n    a  = torch.tensor(a).unsqueeze(1)\n    r  = torch.tensor(r,  dtype=torch.float32)\n    done = torch.tensor(done, dtype=torch.float32)\n\n    q_sa = q(s).gather(1, a).squeeze(1)              # Q_theta(s,a)\n    with torch.no_grad():                            # stop-gradient on target\n        y = r + gamma * q_target(s2).max(1).values * (1 - done)\n    loss = nn.functional.mse_loss(q_sa, y)\n    opt.zero_grad(); loss.backward(); opt.step()" },
        { type: "heading", text: "The improvements you should know by name" },
        { type: "table",
          headers: ["Variant", "The fix", "Why"],
          rows: [
            ["**Double DQN**", "pick $a'$ with $Q_\\theta$, evaluate it with $Q_{\\theta^-}$", "the $\\max$ over-estimates values; decoupling selection from evaluation removes the bias"],
            ["**Dueling DQN**", "split the head into $V(s)$ + advantage $A(s,a)$", "learns state value even when the action barely matters"],
            ["**Prioritized replay**", "sample high-TD-error transitions more often", "learn fastest from the most surprising experiences"],
            ["**Rainbow**", "combine all of the above + more", "the 2017 'kitchen sink' that set the Atari SOTA"],
          ]
        },
        { type: "callout", variant: "tip", text: "**DQN's ceiling.** DQN needs a $\\max_a$ over actions, so it only handles **discrete** action sets. A robot arm with continuous torques has no finite $\\max$ to take. That limitation is exactly why we now turn to **policy gradient** methods, which parameterize the policy directly and handle continuous actions natively." },
      ]
    },

    {
      id: "policy-gradient",
      title: "Policy gradients: the REINFORCE theorem",
      level: "core",
      body: [
        { type: "p", text: "Value-based methods learn *how good* each action is, then act greedily. **Policy-gradient** methods skip the middleman and learn the policy itself: a network $\\pi_\\theta(a\\mid s)$ that outputs a probability distribution over actions. We then do gradient *ascent* to make good trajectories more likely. This handles continuous actions, learns stochastic policies (sometimes optimal — think bluffing), and is the foundation of PPO and RLHF." },
        { type: "heading", text: "The objective" },
        { type: "p", text: "We want to maximize expected return under trajectories $\\tau = (s_0,a_0,s_1,a_1,\\dots)$ sampled by running the policy:" },
        { type: "math", tex: String.raw`J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\big], \qquad R(\tau) = \sum_{t=0}^{T} \gamma^t r_t` },
        { type: "p", text: "The difficulty: the thing we sample from ($\\pi_\\theta$) is the thing we differentiate. We cannot push the gradient inside an expectation whose *distribution* depends on $\\theta$. The **log-derivative trick** (a.k.a. the REINFORCE trick) solves this cleanly." },
        { type: "heading", text: "Deriving the policy gradient theorem" },
        { type: "p", text: "Write the expectation as an integral over trajectory probability $p_\\theta(\\tau)$ and differentiate. The key identity is $\\nabla_\\theta p_\\theta(\\tau) = p_\\theta(\\tau)\\,\\nabla_\\theta \\log p_\\theta(\\tau)$ (since $\\nabla \\log f = \\nabla f / f$):" },
        { type: "math", tex: String.raw`\nabla_\theta J = \nabla_\theta \!\int p_\theta(\tau) R(\tau)\, d\tau = \int p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, R(\tau)\, d\tau = \mathbb{E}_{\tau}\big[\nabla_\theta \log p_\theta(\tau)\, R(\tau)\big]` },
        { type: "p", text: "Now expand $\\log p_\\theta(\\tau)$. The trajectory probability factorizes into the (unknown) environment dynamics $p(s_{t+1}\\mid s_t,a_t)$ and the policy $\\pi_\\theta(a_t\\mid s_t)$. Crucially the dynamics terms **do not depend on $\\theta$**, so their gradient is zero and the environment model drops out entirely:" },
        { type: "math", tex: String.raw`\nabla_\theta \log p_\theta(\tau) = \nabla_\theta\!\sum_{t}\Big[\log \pi_\theta(a_t\mid s_t) + \underbrace{\log p(s_{t+1}\mid s_t,a_t)}_{\text{no } \theta}\Big] = \sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)` },
        { type: "p", text: "That is the miracle of policy gradients: **you never need a model of the world.** Substituting back, and replacing the full-trajectory return with the reward-to-go $G_t = \\sum_{k\\ge t}\\gamma^{k-t} r_k$ (only future rewards can be caused by action $a_t$), gives the **policy gradient theorem** / REINFORCE estimator:" },
        { type: "math", tex: String.raw`\boxed{\;\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t\mid s_t)\; G_t\right]\;}` },
        { type: "callout", variant: "note", text: "**Read the formula as blame assignment.** $\\nabla_\\theta \\log \\pi_\\theta(a_t\\mid s_t)$ is the direction in parameter space that makes action $a_t$ *more likely*. Multiplying by $G_t$ scales that push by how good the outcome was: high return → push hard to repeat, negative return → push the other way. It is trial-and-error made differentiable." },
        { type: "heading", text: "REINFORCE, the algorithm" },
        { type: "p", text: "The 1992 REINFORCE algorithm (Williams) is the Monte-Carlo version: run a full episode, compute each $G_t$, then take one gradient step. In an autodiff framework you do not code the gradient directly — you build a **surrogate loss** whose gradient equals the estimator above, and let backprop handle it:" },
        { type: "math", tex: String.raw`\mathcal{L}(\theta) = -\,\frac{1}{N}\sum_{\text{episodes}}\sum_{t} \log \pi_\theta(a_t\mid s_t)\; G_t \qquad (\text{minimize} = \text{ascend } J)` },
        { type: "callout", variant: "gotcha", text: "**Policy gradients are notoriously high-variance.** $G_t$ is a Monte-Carlo return summed over a whole noisy episode, so the gradient estimate is noisy and learning is slow and unstable. Every subsequent method — baselines, advantages, actor–critic, PPO — is fundamentally an attempt to **reduce this variance** while keeping the estimate unbiased. That single problem organizes the rest of this deck." },
      ]
    },

    {
      id: "baselines",
      title: "Baselines & the advantage function",
      level: "core",
      body: [
        { type: "p", text: "The cheapest, most important variance reduction: **subtract a baseline** $b(s)$ from the return before weighting the log-probability. Astonishingly, this changes the variance *without changing the expected gradient* — it stays unbiased." },
        { type: "math", tex: String.raw`\nabla_\theta J = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\,(G_t - b(s_t))\big]` },
        { type: "heading", text: "Why subtracting a baseline is free (unbiased)" },
        { type: "p", text: "The extra term it introduces has expectation zero, because probabilities always sum to one. For any baseline $b(s_t)$ that does not depend on the action:" },
        { type: "math", tex: String.raw`\mathbb{E}_{a}\big[\nabla_\theta \log \pi_\theta(a\mid s)\,b(s)\big] = b(s)\sum_a \pi_\theta(a\mid s)\,\nabla_\theta \log \pi_\theta(a\mid s) = b(s)\,\nabla_\theta \!\sum_a \pi_\theta(a\mid s) = b(s)\,\nabla_\theta 1 = 0` },
        { type: "p", text: "So the mean is untouched, but the *variance* can drop a lot. Intuition: without a baseline, an environment where every reward is positive pushes *every* action's probability up — only their relative magnitudes matter. Subtracting a baseline (a running estimate of typical return) recenters the signal so it says 'better or worse **than expected**,' which is far more informative." },
        { type: "heading", text: "The best baseline: the value function → the advantage" },
        { type: "p", text: "The optimal choice of baseline is (close to) the state-value function $V(s)$ — the average return from $s$. Plugging $b(s_t) = V(s_t)$ turns the weight into the **advantage function**:" },
        { type: "math", tex: String.raw`A(s,a) = Q(s,a) - V(s)` },
        { type: "p", text: "The advantage answers the exact question policy gradients need: *how much better is taking action $a$ in state $s$ than the policy's average behavior there?* Positive advantage → reinforce; negative → suppress. This is the single most important quantity in modern policy optimization." },
        { type: "math", tex: String.raw`\nabla_\theta J = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\; A(s_t,a_t)\big]` },
        { type: "callout", variant: "tip", text: "**Estimating $A$ cheaply.** We rarely know $Q$ and $V$ exactly. A one-step estimate uses the TD error $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$, which is an unbiased estimate of $A(s_t,a_t)$ — low variance but biased if $V$ is wrong. The full Monte-Carlo return $G_t - V(s_t)$ is unbiased but high variance. You want a knob between them." },
        { type: "heading", text: "GAE — the bias/variance knob" },
        { type: "p", text: "**Generalized Advantage Estimation** (Schulman et al., 2015) is that knob. It exponentially averages $n$-step TD errors with a factor $\\lambda \\in [0,1]$, interpolating smoothly between low-variance one-step ($\\lambda{=}0$) and high-variance Monte-Carlo ($\\lambda{=}1$):" },
        { type: "math", tex: String.raw`\hat{A}^{\text{GAE}}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)` },
        { type: "callout", variant: "note", text: "GAE with $\\lambda \\approx 0.95$ is the default advantage estimator inside PPO — including the PPO that trains ChatGPT. When you read a modern RL config and see `gae_lambda: 0.95`, this is the equation it refers to. We now need a *critic* to supply $V(s)$ — which is exactly actor–critic." },
      ]
    },

    {
      id: "actor-critic",
      title: "Actor–critic: a policy and a value network together",
      level: "core",
      body: [
        { type: "p", text: "To use the advantage we need $V(s)$, but $V$ is unknown. **Actor–critic** learns it alongside the policy with a second network. Now two networks train together, each fixing the other's weakness:" },
        { type: "list", ordered: false, items: [
          "**Actor** $\\pi_\\theta(a\\mid s)$ — the policy. Chooses actions and is updated by the policy gradient, weighted by the advantage the critic reports.",
          "**Critic** $V_\\phi(s)$ — a value network. Estimates how good states are, supplying the baseline/advantage so the actor's updates are low-variance.",
        ]},
        { type: "p", text: "The critic is trained by regression toward its own TD target (exactly like the value half of DQN), and the actor is trained by the policy gradient using the critic's advantage estimate. Two losses, optimized jointly:" },
        { type: "math", tex: String.raw`\mathcal{L}_{\text{critic}}(\phi) = \big(r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\big)^2, \qquad \mathcal{L}_{\text{actor}}(\theta) = -\log \pi_\theta(a_t\mid s_t)\,\hat{A}_t` },
        { type: "callout", variant: "note", text: "**Actor–critic vs. the two families.** It is the bridge: the actor keeps the policy-gradient ability to handle continuous/stochastic actions, while the critic brings the value-based ability to bootstrap and cut variance. It gets the best of both — at the cost of two networks and the extra bias a wrong critic introduces." },
        { type: "heading", text: "A2C and A3C" },
        { type: "p", text: "**A3C** (Asynchronous Advantage Actor–Critic, 2016) ran many actor–critic workers in parallel, each on its own copy of the environment, asynchronously pushing gradients to shared parameters. Running many uncorrelated environments at once decorrelates the data — the *same* problem experience replay solves for DQN, solved a different way — so A3C needs no replay buffer. **A2C** is the synchronous version: gather a batch of steps from $N$ parallel environments, average the gradient, update once. A2C is simpler, uses the GPU better, and performs just as well — so it became the standard, and its structure is the direct ancestor of PPO." },
        { type: "callout", variant: "tip", text: "**Add an entropy bonus.** Actor–critic methods almost always add $+\\beta\\, \\mathcal{H}(\\pi_\\theta(\\cdot\\mid s))$ to the actor loss — a reward for keeping the action distribution uncertain. It prevents the policy from collapsing to a single action too early, i.e. it keeps *exploration* alive. You will see the identical entropy term in PPO." },
      ]
    },

    {
      id: "ppo",
      title: "PPO: the clipped surrogate that runs everything",
      level: "core",
      body: [
        { type: "p", text: "**Proximal Policy Optimization** (Schulman et al., 2017) is the workhorse of modern RL — robotics, game-playing, and the RL step of RLHF all use it. Its job: take the actor–critic idea and make the policy update *stable* so you can safely reuse each batch of data for several gradient steps without the policy exploding." },
        { type: "heading", text: "The problem PPO solves" },
        { type: "p", text: "Vanilla policy gradient takes one gradient step per batch of experience — throwing away expensive data. If you instead take *many* steps on the same batch, the policy can move so far that the data (collected under the *old* policy) no longer describes it, and performance collapses. We want to reuse data but forbid the policy from moving too far per update. That 'don't move too far' constraint is a **trust region**." },
        { type: "heading", text: "TRPO ancestry" },
        { type: "p", text: "PPO's predecessor **TRPO** (Trust Region Policy Optimization, 2015) enforced this literally, constraining each update to keep the KL divergence between the new and old policy below a threshold $\\delta$. It works but requires second-order optimization (conjugate gradients, Fisher matrices) — powerful but a nightmare to implement. PPO's insight: get ~90% of the benefit with a *first-order* method by baking the constraint into the objective as a simple clip." },
        { type: "heading", text: "The probability ratio and the clipped surrogate objective" },
        { type: "p", text: "Define the **probability ratio** between the new and old policy for the action actually taken:" },
        { type: "math", tex: String.raw`r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}, \qquad r_t(\theta_{\text{old}}) = 1` },
        { type: "p", text: "A plain surrogate objective $r_t(\\theta)\\,\\hat{A}_t$ would keep increasing $r_t$ whenever $\\hat{A}_t > 0$ — driving the policy arbitrarily far. PPO's **clipped surrogate** caps how much the ratio can help by taking the *minimum* of the unclipped and a clipped version:" },
        { type: "math", tex: String.raw`\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\,\hat{A}_t,\;\; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\big)\Big]` },
        { type: "heading", text: "Reading the clip, case by case" },
        { type: "p", text: "The clip range is typically $\\epsilon = 0.2$, i.e. the ratio is pinned to $[0.8, 1.2]$. Work through the two signs of the advantage:" },
        { type: "table",
          headers: ["Case", "What happens", "Effect"],
          rows: [
            ["$\\hat{A}_t > 0$ (good action)", "objective rises with $r_t$ but is **clipped at $1+\\epsilon$**", "reinforce, but stop once the action is $1.2\\times$ more likely — no runaway"],
            ["$\\hat{A}_t < 0$ (bad action)", "objective is **clipped at $1-\\epsilon$**", "suppress, but not below $0.8\\times$ — no over-correction"],
            ["ratio already inside $[1{-}\\epsilon, 1{+}\\epsilon]$", "clip is inactive, gradient flows normally", "small updates behave like ordinary policy gradient"],
          ]
        },
        { type: "p", text: "The $\\min$ is what makes it a *pessimistic* bound: when moving the policy would help the objective a lot, the clip removes the incentive; but when the ratio moves in a way that *hurts* (the unclipped term is smaller), the $\\min$ keeps that penalty so the policy is still pushed back. The result is a soft trust region with nothing but a clamp and a min — a few lines of code." },
        { type: "math", tex: String.raw`\mathcal{L}^{\text{PPO}} = \underbrace{\mathcal{L}^{\text{CLIP}}}_{\text{actor}} - c_1\underbrace{(V_\phi(s_t) - G_t)^2}_{\text{critic}} + c_2\underbrace{\mathcal{H}[\pi_\theta]}_{\text{entropy bonus}}` },
        { type: "callout", variant: "tip", text: "**Why PPO won.** It is stable, sample-efficient (reuses each batch for ~4–10 epochs), needs no second-order math, and has few hyperparameters. That robustness is exactly why OpenAI reached for PPO when they needed an RL algorithm reliable enough to fine-tune a language model without babysitting — which is the next section." },
        { type: "callout", variant: "gotcha", text: "PPO is **on-policy**: the ratio $r_t$ is only valid for data collected under $\\pi_{\\theta_{\\text{old}}}$. After a handful of epochs on a batch, $\\theta$ has drifted enough that you must *throw the batch away* and collect fresh rollouts. Trying to reuse old data like an off-policy replay buffer breaks the ratio's assumptions." },
      ]
    },

    {
      id: "rlhf",
      title: "RLHF: how ChatGPT is actually trained",
      level: "core",
      body: [
        { type: "p", text: "This is the payoff. A base LLM, pretrained on the internet (see [LLM Pretraining](#/llm-pretraining)), is a brilliant next-token predictor but not a helpful assistant — it will happily continue a prompt with more questions, or produce fluent nonsense. **RLHF (Reinforcement Learning from Human Feedback)** is the technique that turned GPT-3 into InstructGPT and then ChatGPT (2022): it aligns the model's behavior to human preferences using the exact RL machinery we just built. The [LLM Fine-tuning](#/llm-finetuning) deck covers the supervised side; here we do the RL side in depth." },
        { type: "heading", text: "The LLM as an RL policy" },
        { type: "p", text: "The mapping from language generation to an MDP is the crux — once you see it, RLHF is just PPO:" },
        { type: "table",
          headers: ["RL concept", "In an LLM", ""],
          rows: [
            ["Policy $\\pi_\\theta$", "the language model itself", "outputs a distribution over the next token"],
            ["State $s_t$", "the prompt + tokens generated so far", "the context"],
            ["Action $a_t$", "the next token to emit", "chosen from the vocabulary $|\\mathcal{A}|\\approx 50{,}000$"],
            ["Episode", "generating a full response", "ends at the stop token"],
            ["Reward $r$", "how much a human likes the response", "given (mostly) at the end"],
          ]
        },
        { type: "p", text: "So generating an answer *is* rolling out a policy for a few hundred timesteps. If only we had a reward signal, we could run PPO. But we cannot ask a human to score every one of millions of samples during training — so we **learn** the reward." },
        { type: "heading", text: "Step 1 — the reward model from preferences (Bradley–Terry)" },
        { type: "p", text: "Humans are bad at assigning absolute scores but good at **comparisons**. So we collect data of the form: for a prompt, here are two model responses $y_w$ (winner) and $y_l$ (loser); a labeler picked $y_w$. We fit a **reward model** $r_\\psi(x,y)$ — usually the LLM with its output head replaced by a single scalar — using the **Bradley–Terry** model of pairwise preference, which says the probability a human prefers $y_w$ is a sigmoid of the reward gap:" },
        { type: "math", tex: String.raw`P(y_w \succ y_l \mid x) = \sigma\big(r_\psi(x,y_w) - r_\psi(x,y_l)\big), \qquad \mathcal{L}(\psi) = -\,\mathbb{E}\big[\log \sigma\big(r_\psi(x,y_w) - r_\psi(x,y_l)\big)\big]` },
        { type: "p", text: "Minimizing this loss teaches $r_\\psi$ to assign higher scores to responses humans prefer. Now we have a cheap, automatic judge that scores any response — the reward function for RL." },
        { type: "heading", text: "Step 2 — PPO on the LLM with a KL leash" },
        { type: "p", text: "Now run PPO. The policy is the LLM ($\\pi_\\theta$, initialized from the supervised-fine-tuned model), the reward for a completed response $y$ is $r_\\psi(x,y)$, and we optimize the LLM to produce high-reward responses. But there is a catch unique to this setting: the reward model is only accurate *near* the responses it was trained on. Left unchecked, PPO will find adversarial gibberish that the reward model loves — **reward hacking**. The fix is a **KL penalty** that leashes the policy to the original reference model $\\pi_{\\text{ref}}$:" },
        { type: "math", tex: String.raw`r_{\text{total}}(x,y) = \underbrace{r_\psi(x,y)}_{\text{learned reward}} - \;\beta\,\underbrace{\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}}_{\text{per-token KL to the frozen reference}}` },
        { type: "p", text: "The KL term is a per-token penalty for drifting away from the sensible base model: it lets the policy improve toward human preference while forbidding it from wandering into the degenerate regions where the reward model is wrong. This is the *same* trust-region instinct as PPO's clip, applied against a fixed anchor rather than the previous step." },
        { type: "callout", variant: "gotcha", text: "**Reward hacking is the central failure mode of RLHF.** The policy optimizes the *proxy* (the reward model), not true human preference (Goodhart's law). Symptoms: the model becomes sycophantic, pads answers with hedging, exploits formatting the reward model liked, or produces confident-sounding wrongness. Tuning $\\beta$, refreshing the reward model on new samples, and capping KL are the practical defenses." },
        { type: "heading", text: "DPO — RLHF without the RL" },
        { type: "p", text: "The full pipeline (train a reward model, then run PPO with a value network, a reference model, and rollouts) is heavy: four models in memory and a finicky RL loop. **Direct Preference Optimization** (Rafailov et al., 2023) asks: can we skip the reward model and PPO entirely and optimize the policy *directly* from the preference pairs? Yes. DPO derives — from the same KL-constrained RLHF objective — that the optimal policy and the reward are linked in closed form, so the reward model can be substituted away, leaving a simple *supervised* classification loss on preference pairs:" },
        { type: "math", tex: String.raw`\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}\Big[\log \sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]` },
        { type: "p", text: "The policy *is* its own implicit reward model. No reward network, no sampling, no PPO — just a loss over $(x, y_w, y_l)$ triples. DPO is more stable and far cheaper, which is why many open models (Zephyr, Llama-3 variants) are aligned with it. The trade-off: DPO learns only from the fixed preference dataset, whereas online PPO can explore and get reward on *fresh* generations." },
        { type: "heading", text: "GRPO — the reasoning-model workhorse" },
        { type: "p", text: "**Group Relative Policy Optimization** (introduced with DeepSeekMath, and central to DeepSeek-R1's reasoning training, 2024–2025) is a PPO variant tuned for the era of verifiable rewards (math/code, where correctness is checkable). Its trick: **drop the critic**. Instead of a value network to compute the baseline, sample a *group* of $G$ responses to the same prompt and use the group's *mean reward* as the baseline; the advantage of each response is its reward standardized within the group:" },
        { type: "math", tex: String.raw`\hat{A}_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}` },
        { type: "callout", variant: "tip", text: "GRPO halves the memory (no value network) and pairs naturally with rule-based rewards (did the code pass tests? is the final answer correct?), which sidesteps reward-model hacking. This is a big part of how recent reasoning models are trained to 'think' with long chains of thought — RL, all the way down, still descended from the policy gradient theorem in section 3." },
      ]
    },

    {
      id: "continuous",
      title: "Continuous control: DDPG & SAC (one look)",
      level: "deep",
      body: [
        { type: "p", text: "DQN cannot handle continuous actions (no $\\max_a$), and vanilla policy gradients can be sample-hungry. For robotics and control — steering angles, joint torques — the go-to methods are **off-policy actor–critics**. **DDPG** (Deep Deterministic Policy Gradient) learns a *deterministic* actor $\\mu_\\theta(s)$ plus a $Q_\\phi(s,a)$ critic, and trains the actor by pushing actions in the direction that increases $Q$ (backpropagating the critic's gradient into the actor) — think 'DQN for continuous actions,' complete with a replay buffer and target networks. **SAC** (Soft Actor–Critic, 2018) is the modern default: it adds **maximum-entropy** RL, augmenting the reward with the policy's entropy so the agent is rewarded for staying as random as possible while still succeeding. That makes SAC remarkably stable and sample-efficient, and it is off-policy so it reuses data heavily — the reason it dominates continuous-control benchmarks and real-robot learning." },
        { type: "callout", variant: "note", text: "**Rule of thumb.** Discrete actions, care about simplicity → DQN/Rainbow. On-policy, need robustness (or you are doing RLHF) → PPO. Continuous control, want sample efficiency → SAC. All are the same three ingredients — a policy, a value estimate, and a way to bound updates — recombined." },
      ]
    },

    {
      id: "scratch",
      title: "From scratch: REINFORCE on CartPole",
      level: "core",
      body: [
        { type: "p", text: "Time to make it concrete. Below is a **complete, runnable** REINFORCE agent for `CartPole-v1` — the 'hello world' of RL, where you balance a pole by pushing a cart left or right. It is only the policy-gradient core from section 3: sample an episode, compute discounted returns-to-go, normalize them as a cheap baseline, and take one gradient ascent step. In ~40 lines it reliably solves CartPole." },
        { type: "code", lang: "py", code: "import gymnasium as gym\nimport torch, torch.nn as nn\n\nenv = gym.make(\"CartPole-v1\")\npolicy = nn.Sequential(               # actor: state -> action logits\n    nn.Linear(4, 128), nn.ReLU(),\n    nn.Linear(128, 2))\nopt = torch.optim.Adam(policy.parameters(), lr=1e-2)\ngamma = 0.99\n\ndef returns_to_go(rewards):\n    G, out = 0.0, []\n    for r in reversed(rewards):        # G_t = r_t + gamma * G_{t+1}\n        G = r + gamma * G\n        out.insert(0, G)\n    g = torch.tensor(out)\n    return (g - g.mean()) / (g.std() + 1e-8)   # normalize = cheap baseline\n\nfor episode in range(1000):\n    s, _ = env.reset()\n    log_probs, rewards, done = [], [], False\n    while not done:\n        logits = policy(torch.tensor(s, dtype=torch.float32))\n        dist = torch.distributions.Categorical(logits=logits)\n        a = dist.sample()                      # sample from pi_theta(.|s)\n        log_probs.append(dist.log_prob(a))     # store log pi(a|s)\n        s, r, term, trunc, _ = env.step(a.item())\n        rewards.append(r); done = term or trunc\n\n    G = returns_to_go(rewards)\n    # surrogate loss: -sum log pi(a_t|s_t) * G_t   (ascend J => minimize -J)\n    loss = -(torch.stack(log_probs) * G).sum()\n    opt.zero_grad(); loss.backward(); opt.step()\n\n    if episode % 50 == 0:\n        print(f\"ep {episode}  return {sum(rewards):.0f}\")\n# Returns climb from ~20 toward 500 (the CartPole cap) within a few hundred eps." },
        { type: "callout", variant: "gotcha", text: "**Reproducing the theory in code.** `dist.log_prob(a)` is the $\\log \\pi_\\theta(a_t\\mid s_t)$ term; multiplying by the normalized return `G` is the $G_t - b(s_t)$ advantage estimate; the leading minus turns gradient *ascent* on $J$ into gradient *descent* on a loss. Every symbol in the boxed policy-gradient theorem has a line here. If returns don't rise, lower the learning rate or check you normalized the returns — un-normalized returns make this diverge." },
        { type: "heading", text: "Don't hand-roll PPO — use Stable-Baselines3" },
        { type: "p", text: "REINFORCE is perfect for *learning*, but for real work you want a battle-tested PPO. **Stable-Baselines3** gives you the whole thing — GAE, clipping, value network, entropy bonus, parallel envs — behind two lines. Reach for a from-scratch implementation only to understand it; reach for SB3 to get results." },
        { type: "code", lang: "py", code: "from stable_baselines3 import PPO\nfrom stable_baselines3.common.env_util import make_vec_env\n\n# 8 parallel CartPole envs (the A2C-style batching PPO relies on)\nenv = make_vec_env(\"CartPole-v1\", n_envs=8)\n\nmodel = PPO(\"MlpPolicy\", env, verbose=1,\n            n_steps=2048, batch_size=256, n_epochs=10,\n            gamma=0.99, gae_lambda=0.95, clip_range=0.2)  # <- the clip epsilon\nmodel.learn(total_timesteps=100_000)\nmodel.save(\"ppo_cartpole\")\n\n# For RLHF-style LLM training, the analogous library is HuggingFace TRL\n# (PPOTrainer / DPOTrainer / GRPOTrainer) — same algorithms, LLM policies." },
        { type: "callout", variant: "tip", text: "The SB3 kwargs *are* this deck: `gae_lambda=0.95` (section 4), `clip_range=0.2` (the $\\epsilon$ of section 6), `n_epochs=10` (reuse each on-policy batch 10 times), `gamma=0.99`. You now know what every one of them does and why it is there." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "RL is unusually unforgiving — small bugs silently prevent learning rather than crashing. Build these in order; each isolates one concept from the deck. Always plot the episode-return curve; it is your only window into whether anything is working." },
        { type: "list", ordered: true, items: [
          "**REINFORCE, then add a baseline.** Start from the CartPole code above. First run it *without* return normalization and watch the variance wreck training; then add the baseline and watch it stabilize. You will *feel* section 4 rather than just read it.",
          "**DQN on CartPole (or LunarLander).** Implement the replay buffer, target network, and $\\epsilon$-greedy exploration from section 2. Ablate each trick: remove the target network (watch it diverge), then remove replay (watch it overfit). This teaches the deadly triad viscerally.",
          "**Actor–critic from scratch.** Add a value head to your REINFORCE policy and replace the Monte-Carlo return with a TD advantage $\\delta_t = r + \\gamma V(s') - V(s)$. Compare sample efficiency against plain REINFORCE.",
          "**PPO, then check it against SB3.** Implement the clipped surrogate and GAE yourself, then run Stable-Baselines3 PPO on the same env and match learning curves. If yours is much worse, you have a bug in the ratio or the advantage — the usual suspects.",
          "**A tiny RLHF loop.** Take a small model (e.g. GPT-2), write a toy reward (e.g. 'reward responses that are positive/short/contain a keyword'), and use HuggingFace TRL's `PPOTrainer` to optimize it. Watch it reward-hack your toy reward — the best possible lesson in why the KL penalty exists.",
          "**DPO vs. PPO.** Build a small preference dataset, align a model with TRL's `DPOTrainer`, and compare to the PPO run: fewer moving parts, more stable, and no reward model. See section 7's trade-offs first-hand.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The canonical path from the theory here to the frontier. Sutton & Barto for the foundations, Spinning Up to *implement*, then the primary papers." },
        { type: "link", url: "http://incompleteideas.net/book/the-book-2nd.html", text: "Sutton & Barto — Reinforcement Learning: An Introduction (2nd ed., free PDF): the bible. Chapters 9–13 cover function approximation and policy gradients." },
        { type: "link", url: "https://spinningup.openai.com/", text: "OpenAI Spinning Up in Deep RL: the best hands-on intro to policy gradients, VPG → TRPO → PPO, with clean reference code and derivations." },
        { type: "link", url: "https://www.nature.com/articles/nature14236", text: "Mnih et al. 2015 — Human-level control through deep RL (the Nature DQN paper; the Atari milestone)." },
        { type: "link", url: "https://arxiv.org/abs/1707.06347", text: "Schulman et al. 2017 — Proximal Policy Optimization Algorithms (the PPO paper; short and readable)." },
        { type: "link", url: "https://arxiv.org/abs/1506.02438", text: "Schulman et al. 2015 — High-Dimensional Continuous Control Using GAE (the advantage estimator PPO uses)." },
        { type: "link", url: "https://arxiv.org/abs/2203.02155", text: "Ouyang et al. 2022 — Training language models to follow instructions with human feedback (InstructGPT; the RLHF recipe behind ChatGPT)." },
        { type: "link", url: "https://arxiv.org/abs/2305.18290", text: "Rafailov et al. 2023 — Direct Preference Optimization (DPO: RLHF without the RL)." },
        { type: "link", url: "https://arxiv.org/abs/2402.03300", text: "Shao et al. 2024 — DeepSeekMath / GRPO (the critic-free variant powering recent reasoning models)." },
      ]
    },
  ],

  packages: [
    { name: "gymnasium", why: "the standard RL environment API (successor to OpenAI Gym) — CartPole, Atari, MuJoCo" },
    { name: "stable-baselines3", why: "reliable PyTorch implementations of PPO, A2C, DQN, SAC, DDPG — the practical default" },
    { name: "torch", why: "define the policy/value networks and get gradients via autograd" },
    { name: "trl", why: "HuggingFace Transformer RL — PPOTrainer / DPOTrainer / GRPOTrainer for RLHF on LLMs" },
    { name: "trlx / OpenRLHF", why: "scalable RLHF training frameworks for larger models and multi-GPU" },
    { name: "cleanrl", why: "single-file, readable reference implementations — the best code to learn each algorithm from" },
  ],

  gotchas: [
    "The **deadly triad** (bootstrapping + off-policy + function approximation) can make value learning diverge. Target networks and replay buffers exist to tame it.",
    "In DQN the target uses the **frozen** network $Q_{\\theta^-}$ and is a stop-gradient — never backprop through the TD target or training oscillates.",
    "Policy gradients are **high variance**: always subtract a baseline (normalize returns, or use a value critic / GAE). Un-normalized returns often diverge.",
    "PPO is **on-policy**: the ratio $r_t(\\theta)$ is only valid for freshly collected data. Reuse a batch for a few epochs, then discard it — do not treat it like a replay buffer.",
    "**Reward hacking** is RLHF's central failure: the policy games the reward model, not real preferences (Goodhart). The KL penalty to $\\pi_{\\text{ref}}$ is the leash — tune $\\beta$.",
    "DQN needs $\\max_a$, so it only does **discrete** actions. Continuous control needs policy gradients or off-policy actor–critics (DDPG/SAC).",
    "RL is silently fragile — bugs stop learning without crashing. Fix seeds, plot the return curve every run, and check network output shapes before blaming the algorithm.",
    "The advantage sign convention matters: $A>0$ means *better than average*, so reinforce; a flipped sign trains the agent to fail. Sanity-check on CartPole first.",
  ],

  flashcards: [
    { q: "Why does deep RL replace tabular methods?", a: "Function approximation: a network $Q_\\theta$ or $\\pi_\\theta$ **generalizes** across huge/continuous state spaces you could never tabulate or visit exhaustively." },
    { q: "What is the deadly triad?", a: "Bootstrapping + off-policy learning + function approximation. Any two are safe; all three together can make value estimates diverge." },
    { q: "What are DQN's two stabilizing tricks?", a: "**Experience replay** (train on random minibatches from a buffer to de-correlate data) and a **target network** (a frozen copy $Q_{\\theta^-}$ for a stationary TD target)." },
    { q: "State the policy gradient theorem.", a: "$\\nabla_\\theta J = \\mathbb{E}[\\nabla_\\theta \\log \\pi_\\theta(a_t\\mid s_t)\\, G_t]$. The log-derivative trick makes it an expectation, and the environment dynamics cancel out — no model needed." },
    { q: "Why subtract a baseline, and what is the best one?", a: "It reduces variance without adding bias (its expected contribution is zero). The best baseline is $V(s)$, which turns the weight into the advantage $A = Q - V$." },
    { q: "What is the advantage function?", a: "$A(s,a) = Q(s,a) - V(s)$: how much better action $a$ is than the policy's average behavior in state $s$. Positive → reinforce, negative → suppress." },
    { q: "What does an actor–critic combine?", a: "An **actor** (policy $\\pi_\\theta$, updated by policy gradient) and a **critic** (value net $V_\\phi$ supplying the low-variance advantage). It bridges policy- and value-based RL." },
    { q: "What is PPO's clipped surrogate objective?", a: "$\\min(r_t \\hat{A}_t,\\ \\text{clip}(r_t, 1{-}\\epsilon, 1{+}\\epsilon)\\hat{A}_t)$ where $r_t = \\pi_\\theta/\\pi_{\\text{old}}$. The clip is a cheap trust region that stops the policy moving too far." },
    { q: "How is an LLM cast as an RL policy in RLHF?", a: "State = prompt + tokens so far, action = next token, episode = the full response, reward = a learned reward model's score. Generating an answer is a policy rollout." },
    { q: "How is the RLHF reward model trained?", a: "From pairwise human preferences via **Bradley–Terry**: $P(y_w \\succ y_l) = \\sigma(r_\\psi(x,y_w) - r_\\psi(x,y_l))$, minimizing the negative log-likelihood of the chosen response." },
    { q: "Why the KL penalty in RLHF PPO?", a: "It leashes the policy to the reference model $\\pi_{\\text{ref}}$, preventing **reward hacking** — the policy exploiting regions where the reward model is inaccurate." },
    { q: "What is DPO and how does it differ from PPO?", a: "Direct Preference Optimization: derives that the optimal RLHF policy is its own implicit reward model, giving a simple supervised loss on preference pairs — no reward model, no RL loop, but only learns from the fixed dataset." },
  ],

  cheatsheet: [
    { label: "Policy gradient", code: "grad J = E[ grad log pi(a|s) * A_t ]" },
    { label: "REINFORCE loss", code: "loss = -(log_probs * returns_norm).sum()" },
    { label: "Advantage", code: "A = Q - V   # or delta = r + gamma*V(s') - V(s)" },
    { label: "GAE", code: "A_gae = sum_l (gamma*lam)**l * delta_{t+l}" },
    { label: "DQN target", code: "y = r + gamma * Q_target(s2).max(1).values * (1-done)" },
    { label: "PPO ratio", code: "r = torch.exp(logp_new - logp_old)" },
    { label: "PPO clip", code: "min(r*A, clamp(r, 1-eps, 1+eps)*A)" },
    { label: "Sample action", code: "a = Categorical(logits=policy(s)).sample()" },
    { label: "RLHF reward", code: "r_total = r_model(x,y) - beta*log(pi/pi_ref)" },
    { label: "Bradley-Terry loss", code: "-log(sigmoid(r_w - r_l))" },
    { label: "PPO (SB3)", code: "PPO('MlpPolicy', env, gae_lambda=0.95, clip_range=0.2)" },
    { label: "RLHF (TRL)", code: "from trl import PPOTrainer, DPOTrainer, GRPOTrainer" },
  ],
});
