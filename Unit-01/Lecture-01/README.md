# L1 — Introduction to Machine Learning and the Python ML Stack

Unit I · Lecture L1 · Sessions 1–2 · CO1, CO2

| File | Purpose |
|---|---|
| `slides.pdf` | Beamer deck, UPES theme |
| `notes.pdf` | Student handout — same content in prose, with practice questions and worked answers |
| `latex/slides.tex`, `latex/notes.tex` | Sources; shared preambles in `public/_shared/` |

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
