# Unit V — Lecture 10: Logistic Regression

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The third of the five Unit V lecture packages, and a single-topic one: it goes
deep on one derivation rather than broad across several.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Why linear regression fails on a 0/1 label | 28 |
| Odds, log-odds, the logit link and its inverse the sigmoid | 28 |
| Logistic regression and its mathematical proof: the Bernoulli likelihood, the log-likelihood, and the gradient X'(p − y) | 28 |
| No closed form; concavity of the log-likelihood; gradient descent and IRLS | 28 |
| Interpreting coefficients as odds ratios | 28 |
| The decision threshold as a choice | 28 |
| Multiclass: softmax and one-vs-rest | 28 |
| Perfect separation and why regularisation fixes it | 28 |

Course outcome: **CO2** — develop machine learning models using popular
libraries and frameworks.

References: ISLP §4.2–4.3; Bishop §4.3.2–4.3.4; Murphy (PML1) Ch. 10;
Müller and Guido §2.3.3.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 40 frames, 63 overlay pages — for the room |
| `notes.pdf` | 29 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | nine vector figures, every one produced by running code rather than drawn |

## The one idea

**A linear model for the log-odds, not for the label.** A probability lives in
(0, 1); a straight line does not. Odds move the ceiling; a logarithm moves the
floor; and the resulting quantity — the log-odds — is exactly what a linear
model can produce. Inverting that statement gives the sigmoid, the Bernoulli
likelihood gives the objective, and one chain rule gives the gradient

```
dL/dw = X'(p - y)      prediction minus truth
```

Everything else in the lecture is a consequence of those four lines.

## The measurements the lecture turns on

Every number below was produced by running code on scikit-learn's bundled
`load_breast_cancer` (569 patients) or `load_iris` (150 flowers), or on tables
small enough to check by hand. Nothing downloads anything.

- Least squares on eight 0/1-labelled tumours predicted **−0.1667** and
  **+1.1667**. Neither is a probability.
- Adding one *correct* malignant tumour at 30 mm flattened the least-squares
  slope from 0.1905 to 0.0312 and moved its decision boundary from **4.50 to
  5.55 mm** — misclassifying a malignant tumour at 5 mm. The logistic boundary
  moved by **0.000000**.
- The sigmoid's derivative peaks at exactly **0.25**, checked against a finite
  difference to 6.3 × 10⁻¹¹.
- Three points at w = 0 give p = (0.5, 0.5, 0.5), gradient **X'(p − y) =
  (0.5, −2.0)** and, after one step at η = 0.5, **w = (−0.25, 1.0)**. The
  log-likelihood rises from **−2.0794 to −0.8364**. The same numbers come out
  of six lines of NumPy, and out of a numerical gradient that knows none of
  the algebra.
- There is **no closed form**: σ cannot be inverted out of X'(σ(Xw) − y) = 0.
  But the Hessian −X'SX has eigenvalues (−7.0, −1.5), (−2.24, −0.61),
  (−0.48, −0.02) at three different parameter vectors — **negative definite
  everywhere**, so the log-likelihood is strictly concave and the optimum is
  unique.
- Hand-written gradient descent agreed with `sklearn` to **eleven decimal
  places** (w₁ = 0.732488). Newton/IRLS needed **6 steps** to gradient
  descent's ~100.
- A coefficient of ln 2 multiplies the odds by 2 everywhere, but raises the
  probability by **+0.1667** from p = 0.50 and only **+0.0244** from p = 0.95
  — a factor of **6.8** from one and the same number.
- On breast_cancer, mean radius gives exp(β) = **2.8124 per mm**, and one
  extra millimetre is worth anywhere between **+0.0028** and **+0.2490** of
  probability. Divide-by-four bound: β/4 = **0.2585**.
- The threshold is a choice. At 0.10: **1 missed cancer, 14 false alarms**. At
  0.90: **8 missed, 0 false alarms**. Best accuracy on that split is **0.9708
  at a threshold of 0.62**, not at 0.50. ROC AUC is **0.9917** throughout,
  because it ignores the threshold entirely.
- Softmax beat one-vs-rest on iris **0.9600 to 0.9267** in accuracy, and
  **0.1295 to 0.2804** in cross-entropy.
- On perfectly separable data the MLE **does not exist**: unpenalised gradient
  descent took w₁ past **3.39, 4.80, 6.93, 9.21, 11.51** at 10, 100, 1000,
  10 000 and 100 000 steps and kept going, adding ln 10 per tenfold. On real
  iris data (setosa vs versicolor, petal length) the unpenalised coefficient
  is **+51.83** against **+2.90** with the default L2 penalty — and **both
  score 100% accuracy**, so no score would ever tell you.

## Before the next lecture

- Read `notes.pdf`. Section 4 is the derivation, written out in full, and it
  is examinable line by line.
- Attempt practice questions 2, 4 and 5. Question 4 is one gradient step by
  hand on two points; question 5 is an odds-ratio sentence that most people
  get wrong the first time.
- Start one habit now: whenever you report a logistic coefficient, report
  exp(β) beside it and say the sentence out loud — *"one more unit multiplies
  the odds by …"*. If the sentence sounds wrong, the number is being misread.
- Next up is **multiple linear regression**: many predictors at once, partial
  and standardised coefficients, and model validation. The gradient
  X'(ŷ − y) you met here reappears there with ŷ = Xw.

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
