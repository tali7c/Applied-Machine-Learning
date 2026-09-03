# Unit VII — Lecture 22: Hierarchical Clustering

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The third of the Unit VII lecture packages. Budgeted at a minimum of 2 contact
periods.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Hierarchical clustering: nested partitions, the dendrogram, how to read one | 54 |
| Agglomerative (AGNES) against divisive (DIANA), and why divisive is rare | 54, 57 |
| The proximity matrix and the HAC algorithm in nine lines | 54–55 |
| A complete HAC run by hand — single linkage, matrix rewritten after every merge | 55–56 |
| The same six points under complete linkage, and why the trees differ | 55–56 |
| Cluster distance measures: single, complete, average, centroid, Ward | 55–56 |
| Ward's criterion derived from the within-cluster sum of squares | 56 |
| The Lance–Williams recurrence, which unifies all five | 56 |
| What each linkage does to cluster shape, including the chaining failure | 56 |
| Centroid linkage and non-monotonic dendrograms (inversions) | 56 |
| Cutting the dendrogram to a flat clustering, and choosing the cut | 56 |
| Time and space requirements, measured | 57 |

Course outcomes: **CO2** — apply appropriate similarity and distance measures;
**CO4** — apply and evaluate learning techniques and select an appropriate
model.

References: HKP §10.3 and §10.3.1; ISLP §12.4.2; IIR Ch. 17.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 57 frames, 64 overlay pages — for the room |
| `notes.pdf` | 33 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | eleven vector figures, every one produced by running code rather than drawn |

## The one idea

**The algorithm is three words — merge the closest pair — and all the content
is in the word *closest*.** A cluster is a set, and the distance between two
sets has to be defined. Take the minimum over cross pairs and you get single
linkage, which treats a cluster as anything you can walk across in small steps.
Take the maximum and you get complete linkage, which treats a cluster as a set
of small diameter. Take the increase in within-cluster variance and you get
Ward. On the same six points these build three different trees, and on the same
300 points they score anywhere from ARI 0.00 to ARI 1.00.

So the linkage is not a tuning knob to be swept in a grid search. It is a
statement about what shape you believe a cluster has, and you should be able to
defend the one you chose.

## What the lecture turns on

Every number in the notes was produced by running the code shown beside it, on
scikit-learn's bundled datasets and fixed-seed synthetic problems. **Nothing
downloads anything.**

- **A complete HAC run worked entirely by hand, twice.** Six points, fifteen
  distinct pairwise distances, the proximity matrix written out again after
  every one of the five merges — once under single linkage and once under
  complete linkage. The two runs agree at step 1 and then diverge completely:
  heights 1, 2, 2.2361, 3.1623, 4 against 1, 2.2361, 4, 4.4721, 8.6023, and at
  k = 3 the partitions {ABCD}{F}{E} against {AB}{CD}{EF}. Both hand runs are
  then checked against `scipy.cluster.hierarchy.linkage` and match to 1e-12.
  This is the examinable calculation of the lecture.
- **All five linkage distances computed by hand for the same pair of
  clusters** — 3.1623, 8.6023, 5.6240, 5.2202 and 8.5245 — with Ward's height
  derived from the within-cluster sum-of-squares increase Δ = 36.3333 and
  √(2Δ) = 8.5245.
- **The Lance–Williams recurrence**, with the coefficient table, the two-line
  proof that it reduces to min and max, and a from-scratch reimplementation
  that reproduces SciPy exactly for all five methods.
- **The chaining failure, measured rather than asserted.** Two well-separated
  clusters fuse at height 5.571. Add twelve noise points in a line across the
  gap — ten per cent of the data, none of it near either cluster — and single
  linkage fuses them at 0.565, turning a [60, 60] answer into [131, 1] and an
  ARI of 1.000 into 0.000. Complete and Ward do not move at all. The notes also
  show *why*: the fusion height is exactly the bottleneck step along the
  bridge.
- **The other side of chaining, kept in.** Single linkage is the only one of
  the five that gets two interleaved moons right (ARI 1.000 against 0.241 for
  Ward) and the only one that recovers two long thin bands. It is a genuine
  capability, not a defect, and the lecture says so before it says the rest.
- **Centroid linkage can produce an inversion.** Three points, merges at 0.9675
  and then 0.8683 — the dendrogram goes down, and a non-monotone dendrogram
  cannot be cut. On 50 uniform random points, 75.5% of centroid trees contain
  at least one inversion.
- **Cutting the dendrogram, honestly.** The height-gap heuristic on iris points
  at k = 2, silhouette also points at k = 2, and the species say k = 3. The
  notes report all three and make the point that the number of clusters in the
  data and the number of classes in the label column are different quantities.
- **Two decisions that matter more than the linkage.** On iris, standardising
  made the answer *worse* (ARI 0.7312 raw against 0.6153 standardised). On
  wine, the same average linkage scores ARI −0.0054 with Euclidean distance and
  0.8224 with correlation distance.
- **Time and space, measured.** Fitted growth exponents of 2.1–2.3 for SciPy's
  implementations and close to 3 for the naive textbook loop — timings are
  machine-dependent, so the notes also count the work exactly: the naive loop
  makes C(n+1,3) pair comparisons, which is (n+1)/3 times the size of the
  matrix, **267×** at n = 800. And, the number that ends the discussion,
  **37.3 GiB** for the proximity matrix alone at n = 100,000.

## Before the next lecture

- Read `notes.pdf`. The practice questions have full worked solutions, and
  three of them are hand HAC runs with proximity-matrix updates.
- **Work a full HAC run with a pen.** Five or six objects, both single and
  complete linkage, matrix rewritten after every merge, both dendrograms drawn.
  Then check your heights against `scipy.cluster.hierarchy.linkage`. Twenty
  minutes now is worth a lot of marks later.
- Before you use hierarchical clustering on anything, compute n(n−1)/2 × 8
  bytes and check it fits in your memory. That one arithmetic step will save
  you a crashed session.
- Next up is partitioning and density-based clustering — methods that scale to
  the 200,000 rows this one cannot touch, at the price of having to choose k in
  advance and getting a different answer every run.

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
