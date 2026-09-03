# Unit VI — Lecture 14: Decision Trees and Attribute Selection

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The second of the Unit VI lecture packages. Budgeted at a minimum of 2 contact
periods.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Decision trees: what they are, how a prediction is made, the space partition | 35 |
| Top-down induction, and why it must be greedy | 35 |
| Attribute selection measures: entropy and the Gini index | 35–36 |
| Information gain: the definition, and the full worked calculation | 35–36 |
| The ID3 algorithm, run to completion on a categorical table | 36 |
| Gain ratio, split information, and the bias towards many-valued attributes | 36 |
| Converting a tree to IF–THEN rules, and simplifying the rule set | 36 |
| Overfitting: pre-pruning and cost-complexity post-pruning | 36 |

Course outcome: **CO4** — apply and evaluate classification techniques.

References: HKP §8.2 and §8.2.2; ISLP §8.1; Mitchell Ch. 3; Müller and Guido
§2.3.5.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 56 frames, 90 overlay pages — for the room |
| `notes.pdf` | 32 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | nine vector figures, every one produced by running code rather than drawn |

## The one idea

**Choosing the best question is arithmetic.** The previous lecture's model
stored the data and answered a query by looking at who was nearby. A decision
tree stores no rows at all — it stores a sequence of questions, and the path
you take through them is the explanation for the prediction. The whole lecture
reduces to one quantity: given a set of labelled rows, *how mixed up is it?*
Once that is a number, picking the root of the tree is a matter of trying every
attribute and keeping the one that reduces the mixed-up-ness the most. That
measure is Shannon's entropy, the reduction is the information gain, and the
algorithm that repeats it top-down is ID3.

## The measurements the lecture turns on

Every number below was produced by running code on Quinlan's fourteen-row
play-tennis table (built by hand in the code, not downloaded) or on
scikit-learn's bundled `load_iris`, `load_wine` and `load_breast_cancer`, plus
one synthetic problem for the pruning section, fixed by
`make_classification(..., random_state=0)`. **Nothing downloads anything.**

**One seed.** Everywhere randomness is merely arbitrary — splits, the
cross-validation shuffle, the tie-breaking inside `DecisionTreeClassifier` —
the seed is `0`. Three demonstrations deliberately vary it, because the
variation *is* the result: the 2000 entropy-versus-Gini contests, the 200
bootstrap root splits, and the 30 repeated 70/30 splits behind every pruning
curve. Those run over seeds `0, 1, 2, …` derived from the same one, so the
spread is real and the whole sweep still reproduces. Run the code as written
and you get these numbers, not similar ones.

- Entropy is **0** for a pure node, **1.0000** for an evenly split binary node,
  and **log₂ m** for m equal classes — 1.5850 at three classes, 3.3219 at ten.
  A thousand labels that are 90% "Yes" carry **0.4562 bits each**, so they fit
  in 456 bits rather than 1000.
- On the fourteen rows, **H(S) = 0.940286 bits** and **Gini(S) = 0.459184**.
- The four information gains: **Outlook 0.246750**, Humidity 0.151836,
  Wind 0.048127, Temperature 0.029223. Outlook wins because it is the **only
  attribute with a pure child** — four Overcast days, four "Yes", entropy 0.
  The runner-up is behind by 0.094914.
- Recursing: the five Sunny days are split **perfectly** by Humidity and the
  five Rain days **perfectly** by Wind, both at gain 0.970951 — the entire
  entropy of the node. Temperature and Humidity **tie** at 0.019973 in the Rain
  branch, so ties are real and an implementation must break them somehow.
- The finished tree has depth 2 and **five leaves**, and **never uses
  Temperature** — not because temperature is irrelevant, but because a greedy
  algorithm answers "what is the best question next", not "which variables
  matter".
- Five leaves means **five IF–THEN rules and nine conditions**, reproducing
  **14 of 14** training rows. On an iris tree, two rules predicting the same
  class let a condition be dropped: five rules became **four**, still correct
  on **146 of 150** rows.
- The Gini index gives the **identical ranking** of all four attributes
  (0.116327 for Outlook), and entropy agreed with Gini on **99.55%** of two
  thousand random two-attribute contests. But on `breast_cancer` the two chose
  **different root splits on 25% of bootstrap resamples** — far more than the
  folklore admits — and those stumps still labelled **98.40%** of rows the
  same. Choose on cost, not on principle.
- CART splits in two, so it must pick a subset: the best binary cut on Outlook
  scores **0.102041** against the multiway split's **0.116327**. Given the same
  fourteen rows one-hot encoded, scikit-learn builds **depth 4 with seven
  leaves** where ID3 builds **depth 2 with five** — and reports a root impurity
  of **0.940286**, exactly the H(S) computed with a pen.
- **Information gain is biased towards many-valued attributes.** The `Day`
  identifier scores **0.940286**, the largest gain possible, **3.81×** the best
  real attribute — and Gini has exactly the same bias (0.459184). ID3 given it
  builds a depth-1, fourteen-leaf tree that is perfect on the training set and
  **matches none of its own rules on a new day**.
- Gain ratio divides by SplitInfo = log₂14 = **3.807355**, a **73.7% cut** —
  but at fourteen rows **0.246966 still beats Outlook's 0.156428**, so the
  identifier would still be chosen. Textbooks that say gain ratio fixes the
  bias are being sloppy. Replicate the table to **seventy rows** and gain ratio
  recovers the correct five-leaf tree while plain gain builds a seventy-leaf
  one.
- A tree is **invariant to any monotone transform**: raw, standardised, log1p
  and cubed all gave 16 leaves, depth 6, test **0.906433** and **identical
  predictions on every test row**. The same four transforms moved 5-NN over a
  range of **0.0351**, and scaling one iris feature by 1000 took 5-NN from
  0.9600 to **0.7933** while the tree stayed at 0.9533.
- An unrestricted tree reaches training accuracy **1.0000** every time. On
  `breast_cancer` that costs **0.0936** of test accuracy, with five of sixteen
  leaves holding a single patient; with 20% label noise it costs **0.3750**.
- Pre-pruning, averaged over 30 splits: the best `max_depth` was **5**
  (0.9312 against 0.9271 unrestricted) and `min_samples_leaf = 5` gave
  **0.9339**. On the noisy problem depth 4 gave 0.7380 against 0.6920.
- Post-pruning by cost complexity with α chosen by cross-validation cut
  `breast_cancer` from **16 leaves to 4** and *cost* **0.0058** of test
  accuracy — reported as it happened, not as we would have liked. On the noisy
  problem the same procedure cut **50 leaves to 8** and bought **+0.0750**.

## Before the next lecture

- Read `notes.pdf`. Sections on entropy, information gain and ID3 are one
  continuous argument; the practice questions at the end have full worked
  solutions, not hints.
- **Reproduce the whole fourteen-row calculation with a pen**: H(S), all four
  gains, both recursions, the five rules. This is the most examinable piece of
  arithmetic in Unit VI, and it is worth being able to do from memory.
- Then do it again for the `Day` column and write one sentence saying why the
  answer is a disaster.
- Start one habit now: **before you fit anything, look at the columns and
  delete the identifiers.** Row numbers, ticket numbers, patient IDs,
  timestamps used as labels. Information gain will happily pick every one of
  them.
- Next up is more classification. Carry one contrast across from the previous
  lecture: k-NN weights every column equally; a tree chooses one column at a
  time and ignores the rest.

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
