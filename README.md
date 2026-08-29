# Applied Machine Learning (CSAI2017P)

Lecture slides (Beamer), teaching notes, lab assignments and starter notebooks for **Applied Machine Learning**, B.Tech CSE (AI & ML), Semester III.

GitHub: `https://github.com/tali7c/Applied-Machine-Learning`

**Students, start here → [`Course-Handbook/Course-Handbook.pdf`](Course-Handbook/Course-Handbook.pdf)** — syllabus, the full marks structure, every deadline and the reading list in one document.

**New to VS Code? → [`student-guides/`](student-guides/)** — a 5-minute video and
a step list covering the whole submission workflow: clone the repository, open it
as a folder, install the Jupyter and PDF extensions, and work each section of an
assignment.

| | |
|---|---|
| Course code | CSAI2017 (theory) · CSAI2017P (theory + lab) |
| L-T-P-C | 4-0-1-5 |
| Units | 7 |
| Delivery | 24 lecture packages (L1–L24) + 2 assessment sessions |
| Laboratory | 15 experiments; 1–14 assessed as 10 assignments, 15 as the project |
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
- `theory-quizzes/Txx/`
  - `Txx-QP.pdf` — the question paper, all sets
  - `Txx-Solutions.pdf` — worked solutions for every set
  - `latex/` — the LaTeX sources of those two documents
  - Published only after every group has sat that quiz
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
| Theory quizzes (10 tests, 40 min each) | 10 |
| Project — individual quiz | 10 |
| Project — group submission state | 5 |
| Lab assignments (10) | 10 |
| Lab quizzes (10, 60 min each) | 10 |
| Mid-Semester examination | 20 |
| End-Semester examination | 30 |
| **Total** | **100** |

The project carries 15 marks, of which **10 are individual** — a strong contributor in a weak group
can still score well.

> **Indicative.** The component split is subject to departmental approval; the confirmed scheme will
> be posted on the LMS and this table updated to match.

## Release schedule

Material is published as it is delivered, not all at once. Currently released:

| Item | Covers |
|---|---|
| **Unit I — Lecture 1** | Introduction to ML and the Python ML stack |
| **Unit II — Lecture 2** | Loss functions: MSE, MAE, Huber, cross-entropy, hinge, triplet |
| **Unit III — Lecture 3** | Gradient descent: batch, stochastic, mini-batch and momentum |
| **Lab Assignments 1–10** | All ten briefs and starter notebooks, covering experiments 1–14 |
| **Anchor-Datasets.pdf** | The three dataset families, with sources and loading code |
| **student-guides/** | VS Code tutorial video + step list for the submission workflow |
| **Theory Quiz 1 (T01)** | Paper and worked solutions, sets A–H — prerequisites, L1, L2 |

Each theory quiz paper and its worked solutions are published here **after every
group has sat it** — so you can check your own answers against a full solution
for your set. Later lectures appear here as the semester progresses. **All ten lab assignments
are published up front** so you can see where the lab is going and get the data
in advance — but work them in order, and check the release schedule for when
each one is due.

> **Get your data early.** Several datasets download over the internet on first
> use and the campus network blocks that download. See *Getting the data* in
> `Anchor-Datasets.pdf` before your first lab — it costs five minutes at home and
> saves you a whole session.

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

Lab experiments 1–14 are grouped into **ten assignments** — the assignment is the fixed unit, not the
lab session, so this grouping holds however many labs the timetable delivers. Each assignment is
worked once the lectures it depends on are done, with enough time to complete the experiments it
covers. All ten briefs are published now so you can plan and fetch data ahead. Experiment 15 (the mini-project) is assessed under the Project component instead.
Each assignment is marked out of 10: Section A (attempt 2 of 3,
2 marks each), Section B (attempt 1 of 2, 4 marks each), Section C (consolidation, compulsory, 2 marks).

| Assignment | Experiments | Topic |
|---|---|---|
| LA1 | 1, 2 | Environment setup; preprocessing and EDA |
| LA2 | 3 | Regression: housing price prediction |
| LA3 | 4 | Time-series regression: stock price prediction |
| LA4 | 5 | Customer churn prediction |
| LA5 | 6, 7 | Image and multiclass classification: digits and Iris |
| LA6 | 8 | Spam email detection |
| LA7 | 9 | Credit risk assessment |
| LA8 | 10, 11 | Anomaly detection; physiological signal classification |
| LA9 | 12 | Breast cancer diagnosis from imaging features |
| LA10 | 13, 14 | Clustering medical images; crop disease classification |

Three rules apply to every assignment:

- Fit every transformer inside a `Pipeline`. Never fit on the test set.
- Set `random_state` wherever a question asks for reproducibility.
- Every question wants a short written observation, not just a number. Code without commentary caps you at half marks.

The 60-minute lab quiz that follows each submission asks you to read, trace, debug and modify **your own** submitted code.

## Building the PDFs

Compiled PDFs are committed, so you do not need LaTeX to read anything. To rebuild:

```bash
cd Unit-02/Lecture-02/latex          # or any Unit-xx/Lecture-yy
for i in 1 2 3; do pdflatex -interaction=nonstopmode slides.tex; done
for i in 1 2 3; do pdflatex -interaction=nonstopmode notes.tex;  done
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
