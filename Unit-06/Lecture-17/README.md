# Unit VI — Lecture 17: Neural Networks and Backpropagation

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The fifth of the Unit VI lecture packages. Budgeted at a minimum of 2 contact
periods.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| A unit is `φ(wᵀx + b)`: the same object as linear and logistic regression, with a different activation | 41 |
| The perceptron rule, traced update by update on a truth table | 41 |
| The convergence theorem, and what it does not promise | 41 |
| XOR: the geometric proof, the four-inequality proof, and the way through | 41–42 |
| The hidden layer as a change of coordinates | 42 |
| Feedforward networks, fully connected layers, the MLP, and counting parameters | 42 |
| Activation functions and their derivatives — and why depth without one buys nothing | 41–42 |
| The vanishing gradient, measured layer by layer | 42 |
| Backpropagation derived in full on a 2–2–1 network, then gradient-checked twice | 42 |
| An MLP written from scratch in NumPy, then reproduced by two libraries | 42 |
| Weight initialisation: why all-equal weights can never train | 42 |
| Recurrent (feedback) architecture and backpropagation through time | 43 |

Course outcome: **CO2** — develop machine learning models using popular
libraries and frameworks.

References: HKP §9.2; Bishop §4.1.7, §5.1 and §5.3; Zhang et al., *Dive into
Deep Learning* Ch. 5 and Ch. 9.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 60 frames, 66 overlay pages — for the room |
| `notes.pdf` | 41 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | ten vector figures, every one produced by running code rather than drawn |

## The one idea

**One equation, applied over and over: the chain rule.** A model with ten
thousand parameters raises one question — which way should each of them move?
Answer it the obvious way, by nudging each parameter and re-running the model,
and you need ten thousand passes. Read the chain rule from the loss backwards
instead, and **one forward pass and one backward pass give you all ten
thousand derivatives**. That asymmetry is the difference between possible and
impossible, and it is the whole of backpropagation.

The second idea, which arrives before the first: a stack of linear layers is a
single linear layer, exactly and provably. Depth is worth nothing without a
non-linearity in between. The activation function is not a detail bolted on
for flavour — it is the thing that makes the extra layers mean anything, and
its *derivative* is what the backward pass multiplies by at every level.

## What the lecture turns on

Every number in the notes was produced by running the code shown beside it, on
scikit-learn's bundled datasets and fixed-seed synthetic problems. **Nothing
downloads anything.**

- **The perceptron rule, traced.** On AND it converges in **6 epochs and 11
  weight updates** to `2x₁ + x₂ − 3 ≥ 0`, and the notes print every one of the
  twenty-four row visits. Almost everyone guesses two or three epochs; it takes
  six because the very first row `(0,0)` is misclassified and fixing it breaks
  what had already been learned. On separable iris the same eight lines
  converge in six epochs to accuracy **1.0000**.
- **XOR, and the two separate failures.** The rule ends at **0.50**, cycling
  between two weight states forever — and an exhaustive search over
  **531 441** candidate lines shows the best any line can do is **0.75**. So
  the rule does not even find the best line, and the best line is not good
  enough. The four-inequality contradiction is written out; it is three lines
  of algebra and it is examinable.
- **Two hidden units fix it**, and then the network finds its own units: a
  trained 2–2–1 sigmoid network reaches a final loss of **4.42 × 10⁻⁴** —
  and it learns neither OR nor AND, but the same pair of detectors with one of
  them inverted. With **one** hidden unit the loss stalls at **0.1186**,
  barely below its starting **0.1257**.
- **Depth without non-linearity is free of charge and worth nothing.** Three
  linear layers with **77** parameters equal one with **10** to
  **7.1 × 10⁻¹⁵**, and the product matrix has rank 2, so a narrow middle layer
  caps the whole stack. On moons, **1058** identity-activated parameters beat
  3 by **0.0025** — one point out of four hundred — while the same
  architecture with ReLU gains **+0.0875**.
- **The vanishing gradient, measured rather than asserted.** `σ′ ≤ 0.25`,
  `tanh′ ≤ 1`, `ReLU′ ∈ {0, 1}`. Ten sigmoid layers deliver
  **6.9609 × 10⁻⁷** to the first layer against ReLU's **1.9017 × 10⁻²** — a
  factor of **27 320** — and `0.25¹⁰ = 9.54 × 10⁻⁷` predicted it in advance.
  The consequence is not theoretical: on digits, **three sigmoid hidden layers
  score 0.1000**, which is chance on ten balanced classes, while three ReLU
  layers score 0.9733.
- **Backpropagation in full, on nine parameters.** Forward to
  `L = 0.1429003882`, then `δ₃ = −0.1330107147`, then the recursion down to
  `δ₁` and `δ₂`, then all nine gradients — every number to ten decimal places
  and all of it doable with a pen. **Then checked twice**: central finite
  differences agree to **4.107 × 10⁻¹¹**, and the course's own autodiff library
  agrees to **0.000e+00**, bit for bit. The whole 64–32–10 network's gradients
  check to 2.796 × 10⁻¹¹.
- **The library is not doing anything else.** Fourteen lines of NumPy —
  forward, backward, update — reach **0.9711** on `load_digits`;
  `MLPClassifier` gets 0.9756 and `gradkit` 0.9667. A spread of **0.0089** on
  450 test rows is four rows. Read the loss curves in the figure: two
  independent implementations tracking each other for sixty epochs.
- **Overfitting, in the same table.** Training accuracy reaches 1.0000 while
  test accuracy *falls* from its peak of 0.9800 at epoch 20 to 0.9711 at
  epoch 60. That is the argument for early stopping, made with numbers.
- **All-zero initialisation gives 0.1000 — chance — and it is not slow
  learning you could fix with more epochs.** It is structurally stuck: all 32
  hidden units compute the same thing, receive the same gradient, and take the
  same step, forever. All-`0.1` fails the same way at **0.2356**, which is the
  point: the problem is *equality*, not zero.
- **Recurrent networks and BPTT.** Unroll the loop and backpropagation applies
  unchanged, with each shared weight's gradient summed over the steps. Three
  tanh steps by hand give `∂h₃/∂h₀ = 0.150353` — a product of T factors, which
  can only vanish or explode. Measured on a task rather than a diagram: on
  *remember the first symbol*, accuracy is **1.0000** at T = 10 and **0.4925**
  at T = 40, where the gradient arriving at t = 1 has fallen to 10⁻¹².

## Before the next lecture

- Read `notes.pdf`. There are ten practice questions with full worked
  solutions, and **three of them are forward-and-backward passes by hand** —
  those three are the examinable core.
- **Reproduce the 2–2–1 derivation with a pen and no notes**: forward to
  `L = 0.1429003882`, then all nine gradients. Then change the target from
  `1.0` to `0.0` and do it again. If you can do that, you can answer any
  backpropagation question that can be asked.
- **Gradient-check every backward pass you write.** Central difference,
  `float64`, `h = 1e-6`, four lines of code. It saves the week you would
  otherwise spend blaming the learning rate.
- The library written for this course is small enough to read:
  `https://github.com/tali7c/gradkit`. Open `functions/activation.py` and you
  will find `σ(1 − σ)` exactly as the lecture wrote it.
- Next comes support vector machines, where a single boundary is chosen not by
  descending a loss over many parameters but by asking which line is furthest
  from the data.

## About the figures

Every figure here is computed, not drawn. The generating script lives with the
course's internal tooling and prints each measured number that the notes quote,
so the prose and the pictures cannot drift apart. If a number in the notes looks
wrong, run the code beside it — that is the check that matters.

## Build

```bash
cd latex
for i in 1 2 3; do pdflatex -interaction=nonstopmode slides.tex; done
for i in 1 2 3; do pdflatex -interaction=nonstopmode notes.tex;  done

grep -c '^!'                slides.log notes.log   # must be 0
grep -c 'Overfull \\vbox'   slides.log notes.log   # must be 0
```

Compile from inside `latex/` — the `../../../_shared/` paths depend on it.
