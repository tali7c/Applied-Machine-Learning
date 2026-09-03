# Unit I — Lecture 1: Introduction to Machine Learning and the Python ML Stack

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) Semester III · Autumn 2026**

The single Unit I lecture package. Budgeted at a minimum of 2 contact periods.

Course outcomes: **CO1** (core concepts of ML and AI) and **CO2** (develop
models using popular libraries and frameworks).

| File | Purpose |
|---|---|
| `slides.pdf` | Beamer deck, UPES theme, 113 overlay pages |
| `notes.pdf` | 25-page student handout — same content in prose, with practice questions and worked answers |
| `latex/slides.tex`, `latex/notes.tex` | Sources; shared preambles in `public/_shared/` |

## Reproducibility

One dataset, one seed. The dataset is scikit-learn's bundled `load_iris`;
nothing is fetched over the network. Every arbitrary split uses
`random_state=0`, and every listing that shuffles or samples shows its
`np.random.default_rng(0)` rather than hiding it in a helper.

Four small synthetic datasets are generated for demonstrations that iris
cannot supply — the over-fitting curve, the two-scale KNN comparison, the
regression-versus-classification illustration and the pure-noise leakage
demonstration. Each keeps its own fixed seed, because there the seed *defines
the data* rather than introducing randomness into an experiment.

Every number in `notes.pdf` and `slides.pdf` therefore reproduces exactly if
the code is run as written. There are no wall-clock timings in this lecture.

## What this lecture covers

- Overview of machine learning and its applications
- Types of machine learning: supervised, unsupervised, reinforcement
- Python and its libraries for ML: NumPy, Pandas, scikit-learn

## Before the next session

1. Install Python 3.11+ and `numpy pandas scikit-learn matplotlib seaborn jupyter`.
2. Run the twelve-line Iris script from the notes and print each library's version.
3. Attempt the practice questions in `notes.pdf`; the answers follow them.

## Rebuilding

```bash
cd latex
pdflatex slides.tex && pdflatex slides.tex   # twice, for the navigation bar and TOC
pdflatex notes.tex  && pdflatex notes.tex
```

Compile from inside `latex/` — the `../../../_shared/` paths depend on it.
