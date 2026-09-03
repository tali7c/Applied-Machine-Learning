# Unit II — Lecture 2: Loss Functions

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) Semester III · Autumn 2026**

One lecture package covering the whole of Unit II. Budgeted at a minimum of
2 contact periods; the expected schedule allocates 3.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| What a loss function is; loss vs cost vs metric | Unit II intro |
| Mean Squared Error, Mean Absolute Error, RMSE | 4 |
| Huber loss | 4 |
| Binary, Categorical and Sparse Categorical Cross-Entropy | 5 |
| Hinge loss (SVM loss), squared hinge | 6 |
| Triplet loss | 6 |

Course outcome: **CO3** — evaluate the performance of machine learning models
using appropriate metrics.

## Reproducibility

No dataset and no seed — and for once that is the correct answer rather than an
omission. A loss function is arithmetic on numbers you already have, so nothing
here samples, shuffles or initialises. Every example runs on a handful of
values written out in full in the notes: four house prices, three emails, six
sensor readings. There is no `random_state` and no `default_rng` anywhere.

Every number in `notes.pdf` and `slides.pdf` is therefore not merely
reproducible but inevitable. If you run a listing and get something different,
it is a typing mistake. Randomness enters the course in Lecture 3, and with it
the convention of one seed per lecture, always shown in the listing.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 40 frames, 54 overlay pages — for the room |
| `notes.pdf` | 23 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | six vector figures, every one produced by running code rather than drawn |

## The one idea

**Choosing a loss is not choosing how to score your model. It is choosing what
your model will become.**

The lecture earns that claim with a concrete demonstration: six readings, one
of them corrupted. The constant that minimises MSE moves from 12.33 to 25.67 —
past every clean observation — while the constant that minimises MAE does not
move at all. MSE fits the mean; MAE fits the median.

## Before the next lecture

- Read `notes.pdf`, especially the derivations skipped in class.
- Attempt practice questions 2, 4 and 10. Question 10 ties the whole lecture
  together: squared error gives the mean, absolute error the median, 0–1 error
  the mode.
- Next up is **Unit III — Optimizer Functions**. We now know *what* to
  minimise; next is *how*.

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
