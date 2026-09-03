# Unit VII — Lecture 24: Course Revision and Consolidation

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The fifth and last of the Unit VII lecture packages, and **the final package of
the course**. Budgeted at a minimum of 1 contact period.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Consolidation across Units I–VII: the four questions the course answers | 60 |
| One recipe, several fillings — hypothesis class, loss, and how it is minimised | 60 |
| What each unit actually established, in one measured result apiece | 60 |
| What changes in Unit VII when the target column is removed | 60 |
| Seven models on one dataset under one honest protocol, twenty folds | 60 |
| When a gap is smaller than the noise, and why the ranking depends on the metric | 60 |
| Choosing a method from a property of the problem, with the measurement behind it | 60 |
| The five mistakes that cost the most, each with its price | 60 |
| One complete run: raw arrays with holes in them to a result you could defend | 60 |
| The by-hand arithmetic that recurs across the units | 60 |
| The revision map: the one idea and the examinable calculation for each unit | 60 |
| The remaining internal assessment components | 60 |

Course outcomes: **CO1** — explain the concepts and types of machine learning
and the workflow of a learning problem; **CO2** — apply appropriate
representations, similarity measures and pre-processing to a data set;
**CO3** — evaluate models and interpret their results; **CO4** — apply and
evaluate learning techniques and select an appropriate model for a given
problem. This is the only package that serves all four, which is why it is
last.

References: ISLP Ch. 2 (the four-question frame), Ch. 3 and 6 (Unit V), Ch. 4
and 8–9 (Unit VI), Ch. 5 (the cross-validation discipline), Ch. 12 (Unit VII);
HKP Ch. 3, Ch. 8–9 and §8.5, Ch. 10; Müller and Guido Ch. 2, 3, 5 and 6;
ESL §7.10.2; Bishop §1.5, §3.1, §4.3, Ch. 5 and §7.1.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 48 frames, 55 overlay pages — for the room |
| `notes.pdf` | 30 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | ten vector figures, every one produced by running code rather than drawn |

## The one idea

**Seven units are one argument, not seven subjects.** The whole course answers
four questions in a fixed order, and every method on the syllabus is those four
answers with different contents:

1. **What quantity are we minimising?** — the loss, and choosing it chooses what
   the model becomes (Unit II).
2. **How is the minimisation carried out?** — descent, and the how does not
   depend on the what (Unit III).
3. **Over what class of functions?** — linear, boxes, sums of trees, composed
   layers, a feature map; the class is a statement about the shape you believe
   the answer has (Units V and VI).
4. **How would I know it worked?** — held-out data, with the spread (Units IV
   and VI).

And before all four: **what do you feed it?** — because the columns you supply
decide what the optimiser can possibly find (Unit IV).

Unit VII is the same picture with a hole in it. Delete the target column and
questions 1 and 4 both lose their meaning: there is no loss to minimise and
nothing to score against. What replaces the loss is a **distance**, and what
replaces the score is **your own stated assumption**.

So least squares, ridge, lasso, logistic regression, trees, forests, boosting,
neural networks and SVMs are **not nine inventions**. They are one recipe with
the hypothesis class and the loss swapped out. Read the summary table down the
columns, not across the rows.

## What the lecture turns on

Everything quoted here was **re-measured for this package** under a single
protocol — one dataset, one pipeline, twenty folds — rather than copied from the
earlier lectures. Every number was produced by running the code shown beside it,
on scikit-learn's bundled datasets and fixed-seed synthetic problems. **Nothing
downloads anything.**

- **Seven models, one dataset, one honest pipeline.** 569 patients, 30 columns,
  repeated stratified cross-validation over twenty folds with the scaler inside
  the `Pipeline`. Accuracy is won by **logistic regression at 0.9767**; AUC is
  won by **the RBF SVM at 0.9956**. The AUC gap to second place is **0.0013**
  and the fold-to-fold standard deviation is **0.0046** — the gap is about a
  quarter of the noise.
- **And the winner lost six of the twenty folds.** A paired *t*-test on AUC
  gives *p* = 0.0232, so the difference is "significant" and immaterial at the
  same time. On accuracy the same two models differ by 0.0022 in the *other*
  direction, *p* = 0.3965, with 7 wins, 4 losses and **9 ties**. A difference
  smaller than the fold-to-fold spread is not a difference.
- **Five of seven models change rank when you change the metric.** There is no
  metric-free notion of "the best model", so name the metric before you name
  the winner — and choose the metric from the problem, not from the
  leaderboard. Gradient boosting's 100 trees and 1406 nodes buy nothing
  measurable over logistic regression's 31 coefficients.
- **Choosing is not folklore; every row is a number.** Sparse truth with
  *n* = 50 and *p* = 200: OLS *R*² +0.2926, ridge +0.2492, lasso **+0.9543**,
  and the lasso recovered 5 of the 5 real columns. Curved boundary on two
  moons: logistic 0.8550 against the SVM's **0.9517**. Interpretability: thirty
  readable coefficients score AUC 0.9947 against gradient boosting's 0.9919 —
  the readable model cost nothing. No labels: *k*-means on `wine` scores ARI
  0.3711 raw and **0.8975** standardised, because one column carried 99.77% of
  the raw variance.
- **The five mistakes, priced.** A feature selector fitted outside the pipeline
  turns 0.5100 into 0.8000 — **+0.2900 of pure fiction**, and over twenty fresh
  noise problems the leaky protocol scored higher on **20 of 20**. Tuning and
  scoring on the same folds costs **+0.0515** on noise (and nothing on an easy
  problem — which is exactly why you cannot tell without the nested loop).
  Accuracy under 1% imbalance ranks always-say-no at **0.9900**, above the only
  model that actually finds 19 of the 24 positives. Forgetting the scaler in
  front of an RBF SVM **costs 0.0633**, while the tree and the forest are
  changed by exactly **+0.0000**. And a coefficient read as a cause: total
  serum cholesterol has slope **+0.4723** alone and **−1.0900** with the other
  nine present.
- **A confounder built so the truth is known.** *Z* → *X*, *Z* → *Y*, with *X*
  appearing nowhere in the equation for *Y*. The regression of *Y* on *X* alone
  reports slope **+2.3662** and *R*² 0.7790. Given *Z*, the slope is −0.0119.
  The true causal effect is exactly **+0.0000**. No amount of data fixes this,
  because it is not a statistical error.
- **One complete run, end to end.** 569 × 30 with 506 missing cells (2.96%) —
  listwise deletion would have thrown away **340 of 569** patients. Split,
  baseline (accuracy 0.6257, AUC 0.5000), one pipeline, nested CV on the
  training rows (**AUC 0.9937 ± 0.0086**), `best_score_` 0.9941, and the test
  set touched exactly once: **AUC 0.9966**, accuracy 0.9766, TN 106, FP 1,
  FN 3, TP 61.
- **The threshold is a separate decision.** Moving it from 0.5 to 0.2525 for a
  10:1 miss-to-false-alarm cost takes missed malignancies from **3 to 1** and
  false alarms from 1 to 6. Nothing about the model changed — only the price
  list. And with 171 test rows, one patient is worth **0.0058** of accuracy.
- **Six hand calculations that recur across the whole paper**, all worked in
  full: least squares on five integer points (*b*₁ = 0.6, *b*₀ = 2.2,
  *R*² = 0.6000); ridge as the same calculation with one extra term
  (β̂ = 18/(9+λ): 2.0, 1.8, 1.0, 0.2); entropy and information gain on the
  fourteen-row table (*H* = 0.940286 bits, Gain(Outlook) = 0.246750); one
  forward and one backward pass through nine parameters, agreeing with central
  differences to 2.095 × 10⁻¹¹; Bayes at 1% prevalence (*P*(*D*|+) = 0.1667 —
  99 true positives arriving beside 495 false ones); one Lloyd iteration
  (*W* falls 24.7500 → 10.6667); and a confusion matrix with its ROC point
  ((0.2000, 0.6000), AUC 0.8400 by trapezoids and 21/25 concordant pairs — the
  same number, because that is what AUC *is*).
- **The honest finding.** On this dataset seven very different models are
  separated by less than the fold-to-fold noise. **The method is rarely what
  decides the answer.** The data, the columns and the protocol are — and that
  is where the work is.

## Before the assessments and the examinations

This is the last lecture, so there is no next one to prepare for. What is left
is the assessment, described here only at the level already set out in the
course handbook.

- **Read `notes.pdf` first, then decide what to reread.** The notes carry a
  revision map for all seven units — the one idea and the genuinely examinable
  calculations for each — plus fifteen practice questions weighted towards the
  recurring hand computations, with full worked solutions.
- **If you have one evening:** the four boxes of the spine figure, then the six
  by-hand calculations above. Those six recur across the whole paper.
- **If you have a week:** add the derivations — why least squares is the
  Gaussian MLE, why *Xᵀ X* + λ*I* is always invertible, why the logistic
  log-likelihood is concave, why a bootstrap sample holds 63.2% of the unique
  rows, and why single linkage chains.
- **What is not worth memorising:** library argument names, coefficient tables
  you can look up, and any number you can recompute in ten seconds. Spend the
  time on arithmetic you can be asked to *produce* rather than recognise.
- **The remaining internal assessment.** The last of the ten 40-minute theory
  quizzes covers Unit VII's algorithms — hierarchical, partitioning and
  density-based clustering. The last lab assignment covers the final two
  experiments, and its 60-minute lab quiz follows it, at the machine, on your
  own submitted code. Then the project: the final submission — code, report and
  reproducibility notes — carries the 5 group marks for the state of what was
  handed in, and the **individual project quiz** carries 10, asked of every
  member separately.
- **Then the two examinations**, mid-semester and end-semester, on the
  university's schedule. Every remaining date is announced on the LMS and in
  class. Do not infer one from the syllabus.
- **The one preparation that helps most for the project quiz:** be able to run
  your own pipeline end to end and explain each line of it. Every question it
  asks — what problem, what data, what you cleaned and why, what you compared,
  on what protocol, with what spread, and what you would do differently — is a
  question this course has spent a semester answering.

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
