# Unit VI — Lecture 18: Support Vector Machines

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The sixth of the Unit VI lecture packages. Budgeted at a minimum of 1 contact
period.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Why training accuracy cannot choose between two perfect classifiers | 44 |
| Distance from a point to a hyperplane; functional against geometric margin | 44 |
| The canonical hyperplane, and the derivation of the width `2/‖w‖` | 44 |
| The hard-margin primal as a convex quadratic program | 44 |
| The Lagrangian, the dual, and why the data appears only in inner products | 44 |
| KKT conditions, complementary slackness, and what a support vector *is* | 44 |
| Six points solved completely by hand, then by `SVC` | 44 |
| The deletion experiment: which rows the model actually depends on | 44 |
| The soft margin, slack variables, and `C` as *inverse* regularisation | 44 |
| The hinge loss, and why the flat part makes the solution sparse | 44 |
| The kernel trick, with the degree-two identity verified rather than asserted | 44 |
| The RBF kernel, `γ` as a reach, scaling, cost, and more than two classes | 44 |

Course outcome: **CO2** — develop machine learning models using popular
libraries and frameworks.

References: Bishop §7.1 (with §6.1–§6.3 on kernels); ISLP Ch. 9; Müller and
Guido §2.3.7.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 51 frames, 56 overlay pages — for the room |
| `notes.pdf` | 38 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | thirteen vector figures, every one produced by running code rather than drawn |

## The one idea

**Of all the lines that separate the data, one is furthest from it.** When a
hundred boundaries all classify the training set perfectly, accuracy cannot
rank them — every one scores the same. So pick a criterion accuracy cannot
see: the width of the empty band around the boundary. Maximise that, and the
answer is unique, it is the solution of a convex quadratic program with no
local minima, and it is fixed by a handful of points.

The second idea is the consequence, and it is the one worth remembering: the
solution depends on the data **only through inner products**. Once that is
true, you may change what "inner product" means without changing a line of the
algorithm — and you are fitting a linear model in a space you never build, one
that may have ninety-six million dimensions or infinitely many.

## What the lecture turns on

Every number in the notes was produced by running the code shown beside it, on
scikit-learn's bundled datasets and fixed-seed synthetic problems. **Nothing
downloads anything.**

- **Accuracy is blind to safety.** Five candidate lines on the same six points,
  all with zero training errors, with nearest-point distances from **0.0707**
  to **0.7071** — a factor of ten in clearance for an identical training
  score. That gap is what the rest of the lecture turns into an objective a
  solver can optimise.
- **Six points, solved completely with a pen.** Guess the support set, use
  `Σ αᵢyᵢ = 0` and the two margin equations, and out come **α_A = α_D = 1**,
  **w = (1,1)**, **b = −3**, width `2/‖w‖` = **1.414214**. Then
  `SVC(kernel='linear', C=1e6)` returns the *identical* numbers to 10⁻⁹, and
  the dual objective equals the primal objective at **1.0000000000**. Four of
  the six multipliers are exactly zero.
- **The deletion experiment, which is the memorable result.** Delete B, which
  is not a support vector: `w` and `b` move by **0.00e+00** — not "a small
  change", nothing. Delete support vector A and the margin *widens*, 1.4142 →
  **2.1215**, with a hand check that agrees to four decimals. On an 80-point
  problem, refitting on the **11** support vectors alone gives **identical
  predictions on all 80 rows**: 86% of the training set could have been thrown
  away afterwards. The support vectors *are* the training set.
- **The soft margin, and the honest half of it.** Across `C` = 0.01 → 1000 the
  band narrows **5.2135 → 1.3483** and the points inside it fall **32 → 5** —
  while **training accuracy is 0.9625 in every row**, with the same three
  points misclassified throughout. `C` changes the geometry enormously and the
  training score not at all, which is exactly why you cannot tune it on the
  training set.
- **`C` is *inverse* regularisation**, and most of the room gets this
  backwards. `‖w‖` climbs from **0.3555** to **20.8998** as `C` goes 0.001 →
  100; the best cross-validated score, **0.9772**, is in the middle. Divide the
  objective by `C` and the penalty on `‖w‖²` is `1/(2C)`, so `λ = 1/C`.
- **The SVM is the model whose loss is the hinge.** Eliminate the slack
  variables and the constrained problem becomes an unconstrained one. The hinge
  is exactly zero beyond `m = 1`, so a comfortable point contributes no loss
  *and no gradient* — **11 of 80 points have `m ≤ 1`, and `n_support_` is 11.
  The same eleven.** Logistic regression on the same data uses all 569 rows,
  points almost the same way (cosine **0.9132**), and scores within 0.0035.
- **The kernel identity, checked rather than believed.** At `x = (1,2)`,
  `z = (3,1)`: `(1 + xᵀz)² = 36` and `⟨φ(x), φ(z)⟩ = 1 + 6 + 4 + 9 + 4 + 12 =
  36`, difference **0.00e+00** — both sides doable in your head. Over a
  200 × 200 Gram matrix the worst disagreement is **2.842 × 10⁻¹⁴**, which is
  floating point, not mathematics. And the kernel machine and the
  explicit-map machine agree to **1.24 × 10⁻¹⁴** with the same **38** support
  vectors: one model, computed two ways.
- **Why anyone bothers.** At `d = 100` and degree 5 the explicit map has
  **96 560 646** columns; the kernel evaluates all of them with `d + 1`
  multiplications. On concentric circles a linear SVM scores **0.5682** and the
  degree-two kernel scores **1.0000**. It is not an optimisation — it is the
  difference between possible and impossible.
- **More flexibility is not automatically better.** On moons, degree **two** is
  *worse* than linear (0.8417 against 0.8583), because an even-degree
  polynomial is symmetric in a way that data is not.
- **`γ` is a reach, and a large one overfits spectacularly.** On
  `breast_cancer`, `γ = 10` gives training **1.0000** and cross-validated
  **0.6274** — *exactly* the majority-class rate — with **569 of 569** rows as
  support vectors. The model has memorised every row and generalises to
  nothing.
- **Scaling is not optional, with one honest exception.** The widest column has
  s.d. 568.86 and the narrowest 0.002644, a ratio of **215 171**; an RBF SVM
  loses **0.0561** of cross-validated accuracy on raw columns. The degree-3
  polynomial got *worse* when scaled, by 0.0142 — reported rather than hidden,
  because scaling is a reliable default and not a theorem.
- **What it costs, counted rather than timed.** Over a sweep from 500 to
  32 000 rows the kernel matrix grows **4096×** while the support-vector set
  grows only **18.5×** (`nSV ~ n^0.69`), and the fraction of rows kept falls
  0.50 → 0.14. Those counts are exact on every machine, and they are why the
  measured *time* exponent lands below the documented O(n²)–O(n³). Make the
  problem harder and the count exponent rises to `n^0.76`, with the time
  behind it. Prediction is counted the same way: scoring 8000 rows costs
  exactly `8000 × nSV` kernel evaluations, **4.4×** more at `C = 0.01` than at
  `C = 100`, because an RBF SVM is a learned, weighted, sparse
  nearest-neighbour rule. The seconds in the notes are labelled as a
  measurement on one machine and should not be quoted as a constant.

## Before the next lecture

- Read `notes.pdf`. There are **ten practice questions with full worked
  solutions**, including two complete by-hand SVM solutions and a kernel
  identity to expand.
- **Reproduce the six-point solution with a pen**: guess the support set, use
  `Σ αᵢyᵢ = 0` and the two margin equations, get `α = 1`, `w = (1,1)`,
  `b = −3`, width 1.414214. Then verify
  `(1 + xᵀz)² = ⟨φ(x), φ(z)⟩` for `x = (1,2)`, `z = (3,1)` by expanding both
  sides.
- The habit: `make_pipeline(StandardScaler(), SVC())` with a grid over `C` and
  `γ`. Never an `SVC` on raw columns; never `C` or `γ` chosen on the training
  score; and report the held-out number, not the best cross-validated one.
- The habit to unlearn: *"large `C` regularises more."* It is the opposite.
- Next comes the question this lecture kept deferring: once you have several
  classifiers, how do you compare them honestly? Accuracy has failed twice
  already in this lecture — once because it could not rank five perfect lines,
  and once because it did not move across a whole sweep of `C`.

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
