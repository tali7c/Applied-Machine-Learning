# Unit III — Lecture 3: Gradient Descent (Stochastic, Mini-Batch and Momentum)

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) Semester III · Autumn 2026**

The first of the two Unit III lecture packages. Budgeted at a minimum of
2 contact periods; the expected schedule allocates 3.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Gradient descent, the update rule, geometric convergence | Unit III intro |
| The learning rate: four regimes and the stability threshold `η < 2/L` | 7 |
| Batch gradient descent, epochs vs updates | 7 |
| Stochastic gradient descent; unbiased gradients; the noise floor | 7 |
| Mini-batch gradient descent; the `1/√B` law; why it wins in practice | 8 |
| Momentum; the exponentially weighted average; ravines | 9 |

Course outcome: **CO2** — apply appropriate machine learning techniques to
solve a given problem.

References: D2L §12.4–12.6; MML Ch. 7 (§7.1).

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 40 frames, 58 overlay pages — for the room |
| `notes.pdf` | 30 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | seven vector figures, every one produced by running code rather than drawn |

## The one idea

**Step downhill, again and again.** Everything interesting is in three
questions: *how big* a step, computed from *how much data*, and with *how much
memory* of the last step. Those three questions are the learning rate, the
batch size, and momentum.

Unit II answered *what* quantity a model minimises. This lecture answers *how*
the minimising is actually done — and the answer does not depend on which loss
you picked, which is the whole payoff of having separated the two questions.

## The measurements the lecture turns on

Every number below was produced by running one script on scikit-learn's
bundled `load_diabetes` (442 patients, 10 features, features and target both
standardised) or on a deterministic synthetic quadratic. Every random number
generator in that script is `np.random.default_rng(0)` — one dataset, one
seed, so the deck, the notes and the figures all agree and anything you run
yourself reproduces them. Nothing downloads anything.

- The stability cliff: at `η = 1.00` the iterates bounce `0 → 6 → 0` forever;
  at `η = 1.05` they reach **−17.18** in twenty steps.
- SGD reaches MSE **0.5355** after one epoch. Full-batch gradient descent needs
  **71 epochs** for the same result, with identical arithmetic per epoch.
- A single-example gradient is typically wrong by **6.32** when the true
  gradient has length **2.42**. A single SGD step often points uphill.
- `B=32` reached a slightly better MSE than `B=1` (**0.4860** against
  **0.4969** after 30 epochs) with **31.6× fewer updates** per epoch — 442
  against 14 — and about an order of magnitude less wall-clock time. The
  update count is exact on any machine; the milliseconds are not, and are the
  one measurement in this lecture a fixed seed cannot pin down.
- Momentum with `β = 0.9` overshoots a minimum at 3 all the way to **5.13**,
  and on that well-conditioned problem it is **slower** than plain gradient
  descent (92 iterations against 36).
- In a ravine it is transformative: **954 → 135** iterations at `κ = 200`, and
  **1605 → 128** on the real data at `κ = 470`.

## Before the next lecture

- Read `notes.pdf`, especially the derivation of the stability threshold and
  the exponentially weighted average.
- Attempt practice questions 1, 6 and 10. Question 6 is momentum by hand;
  question 10 ties feature scaling, conditioning and the learning rate
  together.
- Next up is **L4 — adaptive optimisers**: Adagrad, RMSprop, Adadelta and Adam.
  Momentum gives every coordinate the same step size; L4 asks what happens when
  each coordinate gets its own.

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
