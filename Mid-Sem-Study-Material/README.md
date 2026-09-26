# Mid-Semester Study Material

The examinable topics of Lectures 1–9, worked by hand, with the course's
practice questions, exercises and full answers. The booklet is 38 pages.

| | |
|---|---|
| Syllabus | Lectures 1–9: Units I–IV and Unit V up to Lecture 9 (least squares; maximum likelihood, R², model adequacy, over-fitting) |
| Not examined | The classification and margin losses of Lecture 2 (binary, categorical and sparse cross-entropy, hinge and triplet loss), and everything from Lecture 10 on (logistic regression, multiple regression, regularisation) |
| Format | For each topic: the method as a numbered recipe, worked examples, the mistakes that cost marks, numerical exercises, and short-answer questions with model answers |
| Practice | 84 numerical exercises, answered at the back; 71 short-answer questions with model answers; two complete practice papers with solutions (Class Test Set B and a new Set C) |

## Files

- `Mid-Sem-Study-Material.pdf`: the booklet
- `latex/`: its LaTeX source

## The twenty-one topics

| | | | |
|---|---|---|---|
| 1 | What ML is; types of learning (L1) | 10 | Outliers and transformations (L5) |
| 2 | Regression losses: SSE, MSE, RMSE, MAE, Huber (L2) | 11 | Feature scaling (L6) |
| 3 | Gradient descent by hand (L3) | 12 | Encoding and feature engineering (L6) |
| 4 | Batch, stochastic and mini-batch GD (L3) | 13 | Data reduction and PCA (L6) |
| 5 | Momentum (L3) | 14 | Splitting and cross-validation (L7) |
| 6 | Adagrad and RMSprop (L4) | 15 | Imbalanced data (L7) |
| 7 | Adam (L4) | 16 | Data leakage and pipeline order (L5–L7) |
| 8 | Adadelta; comparing optimisers (L4) | 17 | Regression foundations (L8) |
| 9 | Missing data (L5) | 18 | Least squares by hand (L8) |
| | | 19 | Direct regression; maximum likelihood (L9) |
| | | 20 | R², adjusted R² and the error summary (L9) |
| | | 21 | Model adequacy, over-fitting and cross-validation (L9) |

Unit IV uses the twelve-house Dehradun dataset from the course's chapter
notes all the way through. The optimiser sections use the same toy problem
as those notes: targets 1 and 3, a constant model, and half-squared loss.

## How to use it

- **Do the exercises before you check the answers.** Work every numerical
  exercise on paper first; all the answers are together at the back.
- **Compare your method as well as your answer.** A correct final number with
  no working earns little, while a correct method with one slip earns most of
  the marks.
- **Check every result.** Residuals should sum to zero, Adam's first step
  should equal exactly η, and the fitted line should pass through (x̄, ȳ).
- **Sit the two practice papers under exam conditions** at the end, taking 60
  minutes each and using no notes.

## Where this sits alongside the other material

- **This booklet** covers the procedures and the most likely questions for
  Lectures 1–9.
- **`theory-assignments/A1/Mid-Sem-Question-Bank.pdf`** gives more breadth.
  It was written for a wider L1–L10 scope, so skip its cross-entropy, hinge
  and triplet questions and its Lecture 10 (logistic regression) material.
- **The lecture notes** in each `Unit-0*/Lecture-*/` folder cover the theory in
  full.

## Rebuilding

```bash
cd latex
for i in 1 2 3; do pdflatex -interaction=nonstopmode Mid-Sem-Study-Material.tex; done
```
