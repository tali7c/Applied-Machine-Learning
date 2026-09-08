# P06 · Clothing-item recognition with classical machine learning

> **Question.** How far can non-deep methods — PCA with KNN, SVM, random forests,
> a small MLP — get on 28×28 clothing images, and which classes get confused with
> which?

Read `../Project-Handbook.md` first — repository rules, commit history, milestones,
group rubric, written quiz. This file defines only the work.

**Units:** IV (dimensionality reduction), VI (+ VII optional) · **COs:** CO1–CO4  
**Difficulty:** middle. No cleaning at all; the cost is compute time and
experimental discipline.

---

## 1. The situation

Fashion-MNIST was published as a drop-in replacement for handwritten digits,
because digits had become too easy to be informative. Every classifier in this
course can be pointed at it. The question is not whether they work — it is what
you can learn by comparing them *properly*: the same features, the same split,
accuracy **and** the time each one costs.

That last column is the one students skip and practitioners care about most. A
KNN that scores 0.86 in eight minutes of prediction time is a different object
from an SVM that scores 0.89 in one second of prediction time, and your report
should treat it that way.

PCA is the spine of the project. 784 raw pixels is both slow and redundant;
choosing how many components to keep — and defending the number — is Unit IV
applied to something where the consequences are visible.

---

## 2. The data

**Fashion-MNIST** (Xiao, Rasul & Vollgraf, Zalando Research). 70 000 grayscale
28×28 images, 10 balanced classes, with an **official** 60 000 / 10 000
train/test split.

- Page: https://github.com/zalandoresearch/fashion-mnist (MIT licence)
- Download the **four gzip files** linked in that README, in a browser:
  `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`,
  `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz` (~30 MB total).
  A CSV version exists on Kaggle if you prefer; say which you used.
- Commit the four gzip files to `data/raw/` — they are under the limit.

**Classes:** 0 T-shirt/top · 1 Trouser · 2 Pullover · 3 Dress · 4 Coat ·
5 Sandal · 6 Shirt · 7 Sneaker · 8 Bag · 9 Ankle boot.

**Use the official test split as your test set.** Do not reshuffle all 70 000 —
results then are not comparable with any published number, and comparability is
half the value of this dataset.

**The IDX format.** You must write the loader yourself (magic number, big-endian
header, then the bytes). It is about fifteen lines with `numpy.frombuffer`, and
writing it is part of the project — do not paste a black-box loader you cannot
explain.

---

## 3. What you must do

### Phase A — set up (Unit IV, before Checkpoint 1)

1. Repository per handbook §3; commit the four gzip files.
2. `src/data.py`: the IDX loader. Verify shapes `(60000, 28, 28)` and
   `(10000, 28, 28)`, and label counts of 6 000 per class.
3. `01-eda.ipynb`: a grid of sample images with labels; mean image per class;
   pixel-intensity distribution. The mean images already suggest which classes
   will be confused — say which, and check your prediction later.
4. Scale pixels to [0, 1] (or standardise — decide and justify; it matters for
   SVM, MLP and PCA, and not at all for trees).

### Phase B — Checkpoint 1 (end of Unit V)

5. Fit **PCA on the training set only**, and plot the cumulative explained-variance
   curve. Choose a component count, and state the rule you used (e.g. 90 % or
   95 % variance, or the elbow). Report how many components that is.
6. One classifier — KNN or logistic regression — on the PCA features, evaluated on
   the official test set. That is your baseline number.
7. Tag `checkpoint-1`, push.

### Phase C — the model comparison (Unit VI, to Checkpoint 2)

8. `03-models.ipynb`. **At least four** classifiers, all on the same PCA
   representation, all on the same split:
   - K-nearest neighbours (tune *k*),
   - SVM (RBF; tune `C` and `gamma` — use a subsample for the search if the full
     set is too slow, and say that you did),
   - random forest (and/or gradient boosting),
   - a small MLP (`MLPClassifier` — one or two hidden layers; report the
     architecture, the activation and the number of epochs).
   Tune on a validation split carved out of the training set; the official test
   set is touched once.
9. **Record fit time and predict time for every model** (`time.perf_counter`),
   in the same table as the scores. Also record the number of PCA components.
10. Tag `checkpoint-2`, push.

### Phase D — evaluation and writing (Unit VI–VII)

11. `04-evaluation.ipynb`:
    - one table: model × {accuracy, macro-F1, fit seconds, predict seconds} →
      `results/metrics.csv`;
    - **per-class precision and recall** for the best model, and the full 10×10
      **confusion matrix** as a heatmap;
    - **the required written analysis**: the model will confuse T-shirt/top (0),
      Pullover (2), Coat (4) and above all **Shirt (6)** with one another, while
      getting Trouser, Bag and the three shoe classes nearly perfect. Explain
      why, in terms of what the features actually are — 784 raw intensities that
      know nothing about texture, drape or sleeve length. Show a few misclassified
      images to support it;
    - a PCA sweep: accuracy against number of components (say 10, 25, 50, 100,
      200) for one model, plotted. Where does adding components stop paying?
12. Report, README, REPRODUCIBILITY.md, final push.

---

## 4. Definition of done

- [ ] Your own IDX loader, in `src/`, with shapes verified.
- [ ] Official 60 000 / 10 000 split used, not a reshuffle.
- [ ] Scaling decided and justified.
- [ ] PCA fitted **on training data only**, explained-variance curve plotted,
      component count chosen by a stated rule.
- [ ] ≥4 classifiers under one identical protocol, hyper-parameters tuned on a
      validation split.
- [ ] **Fit and predict times** recorded alongside the scores.
- [ ] Per-class precision/recall and a 10×10 confusion-matrix heatmap.
- [ ] A written explanation of the shirt/T-shirt/pullover/coat confusion, with
      example misclassified images.
- [ ] Accuracy-vs-components sweep, plotted.
- [ ] Report, README, REPRODUCIBILITY.md; runs from a clean checkout.

## 5. Going further

- **Write the MLP yourself**: forward pass, backpropagation and mini-batch
  gradient descent in NumPy (Units II, III, VI), and compare its accuracy and
  runtime against `MLPClassifier`. The strongest extension on this theme.
- Simple augmentation: horizontal flips, ±2-pixel shifts. Does more data from the
  same data help, and for which classes?
- **Clustering (Unit VII):** k-means with k=10 on the PCA embeddings, ignoring
  labels. Match clusters to classes and report the adjusted Rand index. Which
  classes does unsupervised structure fail to separate — the same ones the
  classifier confuses?
- A binary "shirt vs T-shirt" specialist model. Can a dedicated model fix the
  hardest pair, and would a two-stage system be worth it?

## 6. Pitfalls specific to this theme

| Pitfall | Consequence |
|---|---|
| PCA fitted on all 70 000 before splitting | leakage; the test set shaped the features |
| reshuffling instead of the official split | results not comparable, and I will notice |
| KNN on all 784 raw pixels with the full test set | runs for a very long time; use PCA |
| an unbounded grid search on 60 000 rows with an RBF SVM | will not finish; subsample and say so |
| no timing column | the most useful comparison in the project is missing |
| accuracy only, no confusion matrix | the actual finding — *which* classes collide — is missing |

## 7. What the written quiz will probe

Territory: what your IDX loader does with the first sixteen bytes; how many
components you kept and the rule that chose them; what a principal component of a
clothing image *is*; why the SVM needed scaling and the random forest did not;
which two classes collide most and your explanation; what your timing table means
for someone deploying on a phone.

## 8. Deliverables

Per handbook §3.4–§3.6. You submit the repository URL only.
