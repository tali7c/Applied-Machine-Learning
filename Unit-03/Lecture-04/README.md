# Unit III — Lecture 4: Adaptive Optimisers and Numerical Practice

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) Semester III · Autumn 2026**

The second of the two Unit III lecture packages, and the one that closes the
unit. Budgeted at a minimum of 2 contact periods; the expected schedule
allocates 3.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Why a single learning rate fails: curvature, and feature frequency | 9 |
| Adagrad; the per-coordinate accumulator; sparse features | 10 |
| RMSprop; the exponentially weighted average of squared gradients | 10 |
| Adadelta; dimensional consistency; no learning rate at all | 11 |
| Adam; momentum + RMSprop; bias correction | 11 |
| Numerical practice: comparison, learning-rate sensitivity, ε, decay | 12 |

Course outcome: **CO2** — apply appropriate machine learning techniques to
solve a given problem.

References: D2L §12.7–12.10 (and §12.11 for the schedules).

## Reproducibility

Two datasets, one seed. The datasets are scikit-learn's bundled
`load_diabetes` (standardised) and a synthetic sparse-feature problem built in
the notes; nothing is fetched over the network.

The seed is **1**. Every mini-batch shuffle and every gradient-noise draw is
`np.random.default_rng(1)`, and every listing that shuffles or samples shows
that line rather than hiding it in a helper.

The sparse-feature problem is built with a different fixed number, and the
distinction matters: that number *defines the dataset* rather than introducing
randomness into an experiment. Change the shuffle seed and you get another run
of the same experiment; change the dataset seed and you get a different
experiment.

Every number in `notes.pdf` and `slides.pdf` therefore reproduces exactly if
the code is run as written. There are no wall-clock timings in this lecture, so
nothing here is machine-dependent.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 47 frames, 77 overlay pages — for the room |
| `notes.pdf` | 31 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | ten vector figures, every one produced by running code rather than drawn |

## The one idea

**One learning rate cannot serve every parameter.** Every optimiser here is the
same line —

```
w_i  <-  w_i  -  (eta / sqrt(s_i)) * g_i
```

— and they differ only in what `s_i` accumulates. Adagrad sums the squared
gradients; RMSprop averages them; Adadelta replaces the numerator so that no
learning rate is needed at all; Adam averages the gradient too and corrects
both averages for their zero start.

The previous lecture answered *how* to descend. This one answers *how fast, per
coordinate* — and then spends its last third on what that actually buys you,
which is not what most people think.

## The measurements the lecture turns on

Every number below was produced by running code on scikit-learn's bundled
`load_diabetes` (442 patients, 10 features) or on fixed-seed synthetic
problems. Nothing downloads anything.

- On a 40-feature problem where every true weight is `1.0`, plain SGD recovers
  the dense weights at **1.0004** and the five rarest at **0.1562**. Adagrad,
  at its own best learning rate, gets the rare ones to **0.8631** and cuts the
  loss by a factor of eight.
- Adagrad turns one learning rate into forty, with a **39×** spread, giving the
  rarest feature a step **32×** larger than the dense ones.
- Adagrad's accumulator never decreases: the effective step falls **9.4×** over
  2800 updates and keeps falling like `1/sqrt(t)`. Chasing a moving minimum it
  needed **5,092** updates to reach what RMSprop reached in **100**.
- RMSprop changes exactly one line and its first step is `eta/sqrt(1-gamma)` =
  **3.16×** Adagrad's. On the same little bowl at the same `eta`, Adagrad
  reaches 1.064 in three steps and RMSprop reaches 2.650.
- Adadelta has no learning rate. It worked out a step size of **0.0124** where
  a human found **0.005** by grid search — and multiplying the loss by 100
  changed its answer by **0.02%** while SGD diverged.
- Adam's corrected first step has size **exactly `eta`**, whatever the
  gradient. Without bias correction the same run took steps of **0.534**
  instead of 0.100 and shot to **4.41** past a minimum at 3.
- With a tuned learning rate, all six optimisers land within **two per cent**
  of each other on the real fit — and plain momentum wins. On the sparse
  problem the same comparison spans a factor of **eight**.
- What adaptive methods really buy is a wider target: Adagrad worked over
  **2.24 decades** of learning rate, plain SGD over **1.31**.
- In a ravine, **momentum still wins** (135 iterations against Adam's 262),
  because a ravine is a curvature problem and adaptive methods measure
  gradients.

## Before the next lecture

- Read `notes.pdf`, especially the Adadelta units argument and the full
  bias-correction algebra.
- Attempt practice questions 3, 5 and 9. Question 5 is Adam by hand from a
  fresh start; question 9 is a debugging story you will meet for real.
- Make sure you can write all five updates — SGD, momentum, Adagrad, RMSprop,
  Adam — from memory. Everything after Unit III assumes it.
- Next up is **Unit IV — data preprocessing**. Carry one warning across: no
  optimiser in this lecture can rescue an unstandardised feature matrix.

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
