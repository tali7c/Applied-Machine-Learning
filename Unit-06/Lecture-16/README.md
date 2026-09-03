# Unit VI — Lecture 16: Ensembles — Bagging, Boosting and Random Forests

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The fourth of the Unit VI lecture packages. Budgeted at a minimum of 2 contact
periods.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Why an ensemble can beat its members: Condorcet's jury theorem, worked | 38 |
| Ensembles of classifiers, and three ways to manufacture disagreement | 38 |
| The bootstrap sample and the 63.2% result, derived | 39 |
| Bagging, and out-of-bag error as free validation | 39 |
| Bagging reduces variance, not bias — measured on both halves of the claim | 39 |
| Random forests: bagging plus a random feature subset per split | 40 |
| Feature importance, and where impurity-based importance misleads | 40 |
| AdaBoost: the algorithm, three rounds by hand, then the library | 39 |
| Gradient boosting as fitting the residuals | 39 |
| Bagging against boosting, and how many estimators to use | 39–40 |

Course outcome: **CO4** — apply and evaluate classification techniques.

References: HKP §8.6.1; ISLP §8.2 and §8.2.2; Müller and Guido §2.3.6.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 55 frames, 61 overlay pages — for the room |
| `notes.pdf` | 36 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | ten vector figures, every one produced by running code rather than drawn |

## The one idea

**Diversity, not accuracy, is what you are buying.** A committee of experts who
all make the same mistake is no better than one expert. A committee that fails
on *different rows* can be substantially better than any of its members,
because the majority vote survives any single member being wrong.

That single sentence organises the whole lecture. Every ensemble method is a
different answer to one question: *how do I force these models to disagree?*
Bagging perturbs the training rows. Random forests additionally perturb the
columns available at each split. Boosting changes the problem itself after each
round, so the next model is forced to work on what the last one failed at.

The corollary is the part worth remembering: as the members become more
correlated, the benefit collapses. Independence is the whole game.

## What the lecture turns on

Every number in the notes was produced by running the code shown beside it, on
scikit-learn's bundled datasets and fixed-seed synthetic problems. **Nothing
downloads anything.**

- **Condorcet's jury theorem, worked.** Independent members each slightly
  better than chance, combined by majority vote, become far better than any of
  them. The notes compute the exact binomial number rather than quoting it —
  and then show the benefit collapsing as the members' correlation rises. The
  vote *amplifies whatever the member already is*: below 0.5, the ensemble is
  worse than a member, which is why "weak learner" means better than chance and
  not merely bad.
- **Honest counter-example, kept in.** In one of the ensemble-of-classifiers
  experiments **the ensemble did not beat its best member.** The notes say so
  plainly and use it to make the diversity point sharper rather than hiding it.
- **The 63.2% result derived** — a bootstrap sample contains about 1 − 1/e of
  the unique rows — then verified empirically. This is what makes out-of-bag
  error possible: every row is unused by roughly a third of the members, so
  those members can score it for free, with no held-out split.
- **Bagging reduces variance and not bias, shown on both halves.** The variance
  across resamples falls sharply while the bias stays in the same place. The
  second half of the claim is the one usually skipped: bag a *stump* and you
  get very little, because there was no variance to remove. Put deep unpruned
  trees in the bag.
- **Why random forests add feature randomness.** With trees already grown on
  bootstrap samples, the only remaining lever is decorrelation. The notes
  measure the average pairwise correlation of tree predictions for bagging
  against a forest, and show individual trees getting *worse* while the
  ensemble gets *better* — the clearest demonstration in the lecture that
  member quality is not the objective.
- **Feature importance, honestly.** Impurity-based importance is biased towards
  high-cardinality columns. The notes run two demonstrations: one where three
  planted useless columns are handled correctly — reported as it happened — and
  one deliberately built to mislead it. Permutation importance, measured on
  held-out rows, behaves better.
- **AdaBoost by hand for three rounds**: the weighted error, the alpha, the
  reweighting, all with a pen, then the combined classifier, then the identical
  result from `AdaBoostClassifier`. Gradient boosting follows as the same idea
  with residuals in place of weights.
- **Bagging against boosting**, demonstrated rather than tabulated: flipping
  training labels degrades boosting sharply while bagging absorbs it, because
  boosting's whole mechanism is to concentrate on the rows it is getting wrong
  — and a mislabelled row is one it will never get right. Forests plateau as
  trees are added; boosting can overfit.

## Before the next lecture

- Read `notes.pdf`. The practice questions have full worked solutions.
- **Derive the 63.2% result yourself** and check it against a short simulation.
  It is three lines of algebra and it explains out-of-bag error completely.
- **Work the three AdaBoost rounds with a pen.** Weighted error, alpha,
  reweight, repeat. This is the examinable calculation of the section.
- Before you reach for a forest in the lab, ask the diversity question first:
  will these members fail on different rows? If they will not, an ensemble is
  just a slower single model.
- Next up is neural networks, where a single model with many parameters
  replaces many models with few.

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
