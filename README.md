# Applied Machine Learning (CSAI2017P)

Lecture slides (Beamer), teaching notes, lab assignments and starter notebooks for **Applied Machine Learning**, B.Tech CSE (AI & ML), Semester IV.

GitHub: `https://github.com/tali7c/Applied-Machine-Learning`

**Students, start here → [`Course-Handbook/Course-Handbook.pdf`](Course-Handbook/Course-Handbook.pdf)** — syllabus, the full marks structure, every deadline, grade bands and the reading list in one 12-page document.

| | |
|---|---|
| Course code | CSAI2017 (theory) · CSAI2017P (theory + lab) |
| L-T-P-C | 4-0-1-5 |
| Units | 7 |
| Delivery | 24 lecture packages across 40 theory sessions |
| Laboratory | 15 experiments; experiments 1–14 assessed as 10 assignments, experiment 15 as the project |
| Session | Autumn 2026 · AY 2026-27 |
| Faculty | Dr. Tofik Ali |

## Structure

- `Unit-xx/Lecture-yy/latex/`
  - `slides.tex` — Beamer slides (UPES theme)
  - `notes.tex` — detailed lecture notes (article class)
- `Unit-xx/Lecture-yy/`
  - `slides.pdf` and `notes.pdf` — compiled outputs
  - `README.md` — what the lecture covers and what to do before the next one
- `lab-assignments/LAxx/`
  - `assignment.pdf` — the brief you are marked against
  - `AML_LAxx_starter.ipynb` — starter notebook with the questions inlined
  - `latex/assignment.tex` — source
- `lab-assignments/Anchor-Datasets.pdf` — the three dataset families, with sources and loading snippets
- `Course-Handbook/Course-Handbook.pdf` — the student handbook (syllabus, marks, calendar, resources); source in `Course-Handbook/latex/`
- `_shared/` — common LaTeX preambles and the UPES beamer theme

Notes:

- Large datasets are not stored here. Download each anchor dataset once into a local `data/` folder; `Anchor-Datasets.pdf` has the links and the loading code.
- Internal planning documents, marking schemes and question papers live under `../admin/` and `../private/`, which are outside this repository.

## Course outcomes

- **CO1** — Understand the core concepts and techniques of machine learning and artificial intelligence.
- **CO2** — Develop machine learning models using popular libraries and frameworks.
- **CO3** — Evaluate the performance of machine learning models using appropriate metrics.
- **CO4** — Apply machine learning to various real-world problems and domains.

## Assessment

| Component | Marks |
|---|---|
| Theory assignments (2, each with an assessment test) | 5 |
| Theory quizzes (10 short tests) | 10 |
| Project — individual quiz | 10 |
| Project — group submission state | 5 |
| Lab assignments (10) | 10 |
| Lab quizzes (10) | 10 |
| Mid-Semester examination | 20 |
| End-Semester examination | 30 |
| **Total** | **100** |

The project carries 15 marks, of which **10 are individual** — a strong contributor in a weak group
can still score well.

> **Indicative.** The component split is subject to departmental approval; the confirmed scheme will
> be posted on the LMS and this table updated to match.

## Release schedule

Material is published as it is delivered, not all at once. Currently released:
**Unit I Lecture 1** and **Lab Assignment 1**. Later lectures and assignments appear here as the
semester progresses.

## Unit order

Read the theory lectures in unit order. The sequence is deliberate:

1. **Unit I — Introduction.** What machine learning is, its three families, and the Python stack.
2. **Unit II — Loss functions.** How we put a number on "wrong": MSE, MAE, Huber, cross-entropy, hinge, triplet.
3. **Unit III — Optimizer functions.** How that number is made smaller: SGD, mini-batch, momentum, Adagrad, RMSprop, Adadelta, Adam.
4. **Unit IV — Data preprocessing.** Cleaning, features, splitting, cross-validation, imbalance.
5. **Unit V — Regression.** Linear, logistic, multiple, regularization.
6. **Unit VI — Classification.** KNN, trees, Bayes, ensembles, neural networks, SVM, evaluation.
7. **Unit VII — Clustering.** Hierarchical, K-means, K-medoids, CLARA, DBSCAN.

Units II and III come early on purpose. Once you know **what quantity we are minimising** and **how the minimisation is carried out**, every method in Units V–VII is a variation on one theme rather than a new thing to memorise.

## Laboratory

Lab experiments 1–14 are grouped into ten assignments. Experiment 15 (the mini-project) is assessed
under the Project component instead. Each assignment is marked out of 10: Section A (attempt 2 of 3,
2 marks each), Section B (attempt 1 of 2, 4 marks each), Section C (consolidation, compulsory, 2 marks).

| Assignment | Experiments | Topic |
|---|---|---|
| LA1 | 1, 2 | Environment setup; preprocessing and EDA |
| LA2 | 3 | Regression: housing price prediction |
| LA3 | 4 | Time-series regression: stock price prediction |
| LA4 | 5 | Customer churn prediction |
| LA5 | 6, 7 | Digit recognition; Iris classification |
| LA6 | 8 | Spam email detection |
| LA7 | 9 | Credit risk assessment |
| LA8 | 10, 11 | Anomaly detection; physiological signal classification |
| LA9 | 12 | Breast cancer diagnosis from imaging features |
| LA10 | 13, 14 | Clustering medical images; crop disease classification |

Three rules apply to every assignment:

- Fit every transformer inside a `Pipeline`. Never fit on the test set.
- Set `random_state` wherever a question asks for reproducibility.
- Every question wants a short written observation, not just a number. Code without commentary caps you at half marks.

The lab quiz that follows each assignment asks you to read, trace, debug and modify **your own** submitted code.

## Building the PDFs

Compiled PDFs are committed, so you do not need LaTeX to read anything. To rebuild:

```bash
cd Unit-01/Lecture-01/latex
pdflatex slides.tex && pdflatex slides.tex   # twice, for the navigation bar and TOC
pdflatex notes.tex  && pdflatex notes.tex
```

Compile from inside the `latex/` folder — the `../../../_shared/` paths depend on it.

## Environment

```bash
python -m venv aml-env && source aml-env/bin/activate    # or: conda create -n aml python=3.11
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## Licence

Teaching materials (slides, notes, assignment briefs) are released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Code and notebooks are released under the MIT licence.

The UPES name, logo and beamer theme assets in `_shared/UPES-Beamer-Theme/` are property of the University of Petroleum and Energy Studies and are not covered by the above.
