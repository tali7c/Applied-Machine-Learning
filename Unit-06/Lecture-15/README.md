# Unit VI — Lecture 15: Bayesian Algorithms

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The third of the Unit VI lecture packages. Budgeted at a minimum of 1 contact
period.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Bayes' theorem: derivation from the product rule, and the four names on it | 37 |
| The base-rate problem, worked three ways | 37 |
| Sequential updating — yesterday's posterior is today's prior | 37 |
| ML and MAP decision rules | 37 |
| From Bayes' theorem to a classifier, and the parameter count that forces the naive assumption | 37 |
| Naive Bayes worked entirely by hand, then reproduced in scikit-learn | 37 |
| The zero-frequency problem and Laplace smoothing | 37 |
| Text classification, log-probabilities, underflow and log-sum-exp | 37 |
| Gaussian naive Bayes for continuous features | 37 |
| Why it classifies well and calibrates badly | 37 |

Course outcome: **CO4** — apply and evaluate classification techniques.

References: HKP §8.3; Bishop Ch. 8.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 53 frames, 94 overlay pages — for the room |
| `notes.pdf` | 36 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | ten vector figures, every one produced by running code rather than drawn |

## The one idea

**Turn the question around.** You cannot directly measure "given these
symptoms, what is the disease?" — but you can measure "given the disease, how
often do these symptoms appear?", because that is what a study of known
patients gives you. Bayes' theorem is the machine for reversing a conditional
probability, and the price of the reversal is that you must supply a prior. The
whole lecture is that one manoeuvre, applied first to a single medical test and
then to every feature of a dataset at once.

The second idea, which arrives when the first is scaled up: the exact
calculation needs a joint probability table exponential in the number of
features, so it is unusable. Assume the features are independent given the
class — an assumption that is almost always false — and the table collapses to
a handful of counts. Training becomes counting. The remarkable part is that the
false assumption does far less damage than it should.

## What the lecture turns on

Every number in the notes was produced by running the code shown beside it,
on scikit-learn's bundled datasets or on a twelve-message corpus written out
inline in the source. **Nothing downloads anything.**

**One seed.** Every random choice in this lecture — the two-million-person
base-rate simulation, the synthetic document corpora, the correlated-feature
draws, every split and shuffle — uses `0`: `np.random.default_rng(0)` and
`random_state=0`. Where the point of a demonstration is variation across
draws (twenty document corpora, thirty repeated splits) the seeds run
`0, 1, 2, …` from that same one. Run the code as written and you get these
numbers, not similar ones. **Fit timings are the exception**, so the claim
about naive Bayes being cheap to fit is stated as a pass count — one pass
against 15 to 32 L-BFGS iterations — rather than in milliseconds.

- **The base-rate result.** A disease with 1% prevalence, a test with 99%
  sensitivity and 95% specificity: a positive result means about a **17%**
  chance of having the disease, not 99%. Of 1000 people, **99** true positives
  arrive alongside **495** false ones, because the 99% of healthy people is a
  much larger pool than the 1% who are ill. Worked three ways — algebraically,
  as natural frequencies, and in odds form — because the frequency version is
  the one that makes it feel obvious.
- **The parameter count.** The exact joint distribution needs a table
  exponential in the number of features; the naive assumption reduces it to a
  linear number of counts. The section gives the arithmetic for a concrete
  case, which is the honest justification for an assumption that is otherwise
  hard to defend.
- **Naive Bayes by hand**, every conditional probability computed with a pen on
  a small categorical table, one new instance classified, then the identical
  numbers out of `CategoricalNB` — with the caveat that the library smooths by
  default, so `alpha=0` is required to reproduce textbook arithmetic.
- **Zero frequency.** One empty cell sends the whole product to exactly zero,
  and no amount of evidence from the other features can recover it. Laplace
  smoothing fixes it. Reported honestly: **on this particular table the
  predicted label never actually flips** — the notes say so and explain why the
  correction still matters.
- **Underflow is real, not theoretical.** A product of just **148** per-token
  probabilities is exactly `0.0` in double precision — about one short
  paragraph. At 240 tokens all 180 test documents score `0.0` for both
  classes, `0.0 > 0.0` is `False`, every document is silently assigned to
  class 0 and accuracy falls **0.9722 → 0.5056** with no error message.
  Summing logs fixes it, and log-sum-exp recovers a usable probability at the
  end.
- **Smoothing, measured over twenty corpora rather than one.** Averaged over
  twenty synthetic corpora, no smoothing scores **0.9387**, the best α (which
  is 1) scores **0.9596**, and α = 100 falls to **0.8454**. On any single
  corpus it can go the other way — and on the one printed in the notes it
  does. That is shown rather than hidden.
- **Where independence hurts.** Thirty-one near-copies of one informative
  column took `GaussianNB` from **0.9475 to 0.8200** on no new information,
  while its mean confidence *rose* from 0.9457 to **0.9916**. Logistic
  regression did not move. On XOR — where the four cells are balanced by
  construction — naive Bayes and logistic regression both score **0.4600**
  and a random forest **0.9833**.
- **Good classifier, bad probability estimator.** Naive Bayes claims a mean
  confidence of **0.9936** and is right **0.9342** of the time. Logistic
  regression on the same data claims **0.9626** and is right **0.9649** of the
  time. One of these models knows what it does not know. The reliability table
  makes the gap visible.

## Before the next lecture

- Read `notes.pdf`. The Bayes' theorem and naive-Bayes sections are one
  continuous argument; the practice questions have full worked solutions.
- **Redo the medical-test calculation with a pen, both ways** — algebraically
  and as natural frequencies out of 1000 people. Being able to produce the 17%
  from nothing is worth more than remembering it.
- Work the by-hand classification again on the training table, including the
  smoothed version, and check your arithmetic against the library with
  `alpha=0`.
- Carry one question into the next lecture: naive Bayes is a single model with
  a strong assumption. What if instead of assuming, you built many models and
  let them vote?

## About the figures

Every figure here is computed, not drawn. The generating script lives with the
course's internal tooling and prints each measured number that the notes quote,
so the prose and the pictures cannot drift apart. If a number in the notes looks
wrong, run the code beside it — that is the check that matters. The only
numbers that will not reproduce on your machine are the two `fit ms` columns
of the Gaussian naive Bayes comparison; the pass counts printed beside them
will.

## Build

```bash
cd latex
for i in 1 2 3; do pdflatex -interaction=nonstopmode slides.tex; done
for i in 1 2 3; do pdflatex -interaction=nonstopmode notes.tex;  done

grep -c '^!'                slides.log notes.log   # must be 0
grep -c 'Overfull \\vbox'   slides.log notes.log   # must be 0
```

Compile from inside `latex/` — the `../../../_shared/` paths depend on it.
