# Lab Assignment 5 — worked walkthrough

**▶ Watch:
[`AML-LA05-Walkthrough.mp4`](AML-LA05-Walkthrough.mp4)**
— 11 min, 1080p. Click the file, then Download or play it in the browser.

Multiclass classification: an image is just a row of numbers. Ten classes instead
of two, and a confusion matrix you have to actually read.

> Iris and digits ship inside scikit-learn — nothing to download, works offline.
> Only C1 needs MNIST.

---

## Loading

```python
iris = load_iris()      # 150 flowers, 4 measurements, 3 species
dig  = load_digits()    # 1,797 images, 8x8 greyscale, 10 classes
# (150, 4) (1797, 64) (1797, 8, 8)
```

The campus filter blocks the MNIST download with `CERTIFICATE_VERIFY_FAILED`.
Fetch it at home once and it caches; Sections A and B need nothing.

---

## A1. Iris in ten lines

```python
X_tr, X_te, y_tr, y_te = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=0, stratify=iris.target)

knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(X_tr, y_tr)
accuracy_score(y_te, knn.predict(X_te))      # 0.9737
```

With 38 test flowers, one error moves accuracy by 2.6 points — quote the count
alongside the percentage.

| measurement | standardised gap, versicolor vs virginica |
|---|---|
| **petal width** | **2.96** |
| petal length | 2.55 |
| sepal length | 1.14 |
| sepal width | 0.65 |

**Model observation.** Test accuracy 0.974 — **37 of 38** flowers. The confusion
matrix is diagonal except for one cell: a single virginica classified as
versicolor.

Those are the two species that get confused, and it is structural rather than bad
luck. setosa is linearly separable from both others on petal measurements alone,
so no classifier misses it; versicolor and virginica genuinely overlap at their
boundary. **Petal width** separates them best — the gap between species means is
2.96 pooled standard deviations, against 0.65 for sepal width. Sepal measurements
contribute almost nothing, which is why scaling matters: without it the larger
sepal numbers would dominate the distance.

## A2. Digits as vectors

```python
print(dig.images[0].shape)   # (8, 8)     — what imshow needs
print(dig.data.shape)        # (1797, 64) — what every estimator needs
print(dig.data.min(), dig.data.max())   # 0.0 16.0
```

Pixels run 0–16, not 0–255: these are 32×32 bitmaps downsampled into 4×4 blocks.

**Model observation.** One image is 8×8. The feature matrix is 1,797 × 64 — one
row per image, one column per pixel.

Flattening throws away **adjacency**. In the 8×8 grid, pixel (3,4) is next to
(3,5) and above (4,4); in the flat row they become columns 28, 29 and 36, and
nothing in the data says 29 neighbours 28 while 36 sits directly below it. The
classifier sees 64 unrelated numbers and must rediscover the geometry from
examples. It also loses **translation tolerance**: shift the same digit one pixel
right and every column changes. This is exactly the structure a convolutional
network is built to keep.

## A3. Choosing k

| k | 1 | 3 | 5 | 7 | 9 | 11 | 15 |
|---|---|---|---|---|---|---|---|
| accuracy | .969 | **.986** | .981 | .975 | .964 | .958 | .942 |

**Model observation.** Best **k = 3**, at 98.6%. The curve rises from 96.9% at
k = 1, peaks at 3, then declines steadily to 94.2% at k = 15.

At k = 1 the prediction is whatever single nearest image happens to be, so one
mislabelled or badly written training digit becomes a wrong answer for everything
near it — no averaging, lowest bias, highest variance. At large k the
neighbourhood grows past the class itself: with roughly 144 examples per digit,
k = 15 still fits inside one class, but the neighbours are merely the closest of a
widening circle, so the vote blurs across styles and eventually across classes.

> Choosing k on the test set is mild cheating. A validation split or
> cross-validation is the honest version — worth a line.

---

## B1. Three classifiers, one dataset

| model | accuracy | macro-F1 | fit time | pickled size |
|---|---|---|---|---|
| KNN (k=3) | 0.986 | 0.986 | 0.003 s | 732 KB |
| **SVM (RBF)** | 0.986 | 0.986 | 0.048 s | **412 KB** |
| random forest | 0.975 | 0.975 | 0.876 s | 14,604 KB |

```python
size_kb = len(pickle.dumps(pipe)) / 1024      # what you actually deploy
```

**Model observation.** For a device with tight memory I would ship the **SVM**. It
matches the best accuracy while storing 412 KB — 710 support vectors — against
KNN's 732 KB and the forest's 14.6 MB.

The deeper reason is scaling. KNN's size *is* the training set, so it grows with
every example you collect, and prediction cost grows with it because each query is
compared against all of them. The SVM's size is set by how complicated the
decision boundary is, not by how much data described it, so doubling the training
set typically adds far less than double the support vectors. The forest's size is
a modelling choice — 300 trees — and could be cut, but at 100 trees it is still
megabytes for accuracy it has already lost.

KNN's near-zero fit time is misleading: it has no training phase at all and defers
the entire cost to prediction, the worst place for it on a small device.

## B2. Which digits get confused

**First, a problem with (b).** The best model makes **5 errors** on the 360-image
test set. Five errors cannot rank three pairs — everything ties at one. Ranking
from 5 events is noise, not a finding. The fix is more predictions, not a worse
model:

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
pred = cross_val_predict(best, dig.data, dig.target, cv=cv)
# accuracy 0.976, 43 errors in 1,797 images
```

`shuffle=True` matters: the digits file is ordered by writer, and unshuffled folds
split along that order and drop accuracy to 0.945.

| rank | true digit | called it | count |
|---|---|---|---|
| 1 | **8** | **1** | 7 |
| 2 | **8** | **3** | 4 |
| 3 | 9 / 7 / 5 | 5 / 4 / 9 | 2 each |

**Model observation.** The digit **8** is the problem, and the direction is
one-way: 8 was called 1 seven times while 1 was never called 8. At 8×8 resolution
an 8 written thinly loses the pinch at its waist, and both loops fall below the
width of a pixel block — what remains is a vertical stroke, which is what a 1
looks like. The reverse cannot happen, because a 1 has no ink anywhere near where
the loops of an 8 would be.

8 against 3 is the same failure on the other side: erase the left half of an 8's
loops and a 3 is what is left. The shared feature of every hard case is **a closed
loop the grid is too coarse to keep closed** — a limit of the representation, not
of the classifier, which is what C1 goes on to test.

---

## C1. Does more resolution help? — compulsory

```python
model = make_pipeline(MinMaxScaler(), SVC(kernel='rbf', random_state=0))
#                     ^^^^^^^^^^^^^ not StandardScaler. Reason below.
```

| dataset | features | train rows | accuracy | fit time |
|---|---|---|---|---|
| digits 8×8 | 64 | 1,437 | **0.983** | 0.03 s |
| MNIST 28×28 | 784 | 8,000 | 0.957 | 5.57 s |

**Why `StandardScaler` is wrong for raw pixels.** It divides each pixel by its own
standard deviation. A border pixel in MNIST is black in almost every image, so its
standard deviation is tiny and dividing by it amplifies a rare speck of noise into
a large feature value. `MinMaxScaler` divides every pixel by the same 255,
preserving the image. StandardScaler: **93.0%** on MNIST. MinMaxScaler: **95.7%**.
On the 8×8 digits, where there is no empty border, the two agree within half a
point.

**Model observation.** Twelve times the pixels and five times the data bought a
2.6-point **loss** in accuracy and a fit 185 times slower.

The extra resolution did not buy accuracy because resolution was never the binding
constraint. The two tasks are not equally hard: the 8×8 digits come from a small,
controlled collection in which every image is centred and size-normalised, while
MNIST carries the handwriting of 250 writers with far more variation in slant,
stroke weight and shape. A harder task at higher resolution can easily score worse
than an easy one at low resolution.

Sample size matters too: 8,000 training rows is a tenth of MNIST, and an RBF SVM
given all 60,000 reaches about 98%. So the honest comparison is between two
different problems, and the fair conclusion is narrower than the question implies
— more pixels cost quadratically more compute and are only worth it when detail is
what the classifier is short of. What *did* move the number was preprocessing.

---

## Five common mistakes

- Quoting 97.4% on Iris without saying it is 37 of 38 flowers
- Fitting on `dig.images` instead of `dig.data`, and getting a shape error
- Ranking confused pairs from five errors, where everything ties at one
- `StandardScaler` on raw pixels, which quietly costs you points on MNIST
- Training on all 70,000 MNIST rows, then waiting out the lab session

## Before you submit

- [ ] Restart Kernel → **Run All**, top to bottom, no errors
- [ ] Every accuracy quoted with the count behind it
- [ ] Confusion matrices labelled: actual down, predicted across
- [ ] Filename `AML_LA05_<yourSAPID>.ipynb`, `README.md` beside it
- [ ] No datasets uploaded — Iris and digits ship with scikit-learn

Look at the errors. The confusion matrix tells you what the model does not know,
and the accuracy never will.
