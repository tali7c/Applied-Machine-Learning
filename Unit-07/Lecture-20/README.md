# Unit VII — Lecture 20: Clustering Foundations

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The first of the Unit VII lecture packages, and the course's first
unsupervised topic. Budgeted at a minimum of 1 contact period.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Introduction to clustering: what changes when there is no target column | 48 |
| Why one dataset admits two defensible clusterings that disagree | 48 |
| Clustering algorithms: partitioning, hierarchical, density, grid, model-based | 48–49 |
| What each family assumes about cluster shape, measured on four problems | 49 |
| Statistics of cluster analysis: centroid, WCSS, BCSS, and *T = W + B* | 49 |
| Why *W* cannot choose *k*; Calinski–Harabasz as the decomposition penalised | 49 |
| Internal validation: silhouette, Calinski–Harabasz, Davies–Bouldin | 49–50 |
| External validation: adjusted Rand index, NMI, and chance correction | 50 |
| Clustering as a preprocessing tool: cluster features and cluster-then-label | 50 |
| General applications of clustering | 48 |

Course outcome: **CO1** — understand the foundations of machine learning, the
types of learning problem, and the vocabulary used to describe them.

References: HKP §10.1; Müller and Guido §3.5 and §3.5.4.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 49 frames, 54 overlay pages — for the room |
| `notes.pdf` | 35 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | twelve vector figures, every one produced by running code rather than drawn |

## The one idea

**Every algorithm so far had a column to be scored against. This one does
not.** Remove the target and two things you have never had to think about
have to be rebuilt from scratch: *what counts as a good answer*, and *how you
would know*.

That is not a gap to be patched. It is the defining property of the subject.
Clustering is **exploratory**: it proposes a structure rather than
discovering a fact, and the proposal is a joint consequence of the data, the
distance you chose, the algorithm you chose and the settings you chose. As
Estivill-Castro put it in 2002, *clustering is in the eye of the beholder*.

The corollary is the part worth remembering: because there is no answer key,
the burden of stating your assumptions falls entirely on you.

## What the lecture turns on

Every number in the notes was produced by running the code shown beside it,
on scikit-learn's bundled datasets and fixed-seed synthetic problems.
**Nothing downloads anything.**

- **Two defensible clusterings that disagree, demonstrated not asserted.** On
  `load_iris`, the silhouette prefers *k* = 2 (0.6810 against 0.5528) and the
  species prefer *k* = 3 (ARI 0.7302 against 0.5399). The two clusterings
  agree with each other at ARI only **0.5516**. Neither is a mistake, and the
  notes give three defensible readings rather than picking one.
- **Scaling is the definition of "similar".** On `load_wine`, k-means on the
  raw table agrees with the cultivar at ARI 0.3711; after standardising,
  **0.8975**. The two clusterings agree with each other at 0.3543 — because
  `proline` carries **99.77%** of the raw variance and Euclidean distance is
  therefore almost entirely a comparison of proline.
- **The taxonomy, with teeth.** Four problems × five algorithms. k-means
  scores **0.2445** on two interleaved moons and DBSCAN **1.0000**; on
  elongated blobs k-means scores 0.5554 and a Gaussian mixture 1.0000; on
  clusters of unequal spread single linkage scores **0.0002**. No algorithm
  wins all four rows, and no column is free of a failure.
- **A grid clusterer built from scratch** — a two-dimensional histogram plus a
  flood fill, printed as a table you can read with your eyes: 43 dense cells
  of 100, two connected regions, ARI 0.9643. Loosen the threshold by one and
  the two arms bridge and it collapses to **0.0000**.
- **The decomposition verified numerically.** *T = W + B* worked by hand on
  six integer points, where 91 = 16 + 75 for one clustering and 91 = 79 + 12
  for another — same total, different split. On iris, *T* = 681.370600 at
  every *k* with residual at most 1.14 × 10⁻¹³. Since *T* is fixed,
  minimising *W* and maximising *B* are one problem.
- **Why *W* cannot choose *k*,** proved rather than stated: splitting a
  cluster can never increase *W*, so "minimise *W*" returns *k* = *n*. The
  elbow is a visual heuristic on a curve with no optimum, and the notes say
  so.
- **Silhouette by hand, agreeing with the library to six decimals.**
  *s*(P₁) = **0.691137** from a(i) = 2.581139 and b(i) = 8.356908; the rival
  clustering scores **−0.071164**, and a negative silhouette is the clearest
  signal a point has been misassigned. Davies–Bouldin is worked by hand on the
  same points to 0.455228.
- **Honest disagreement, kept in.** On iris the silhouette prefers a
  clustering the labels do not, and two of the three internal indices
  disagree with the species. The notes state this plainly and explain why
  the silhouette is answering its own question correctly.
- **Chance correction, demonstrated.** Two hundred completely random 3-way
  labellings of iris score Rand **0.5567** but adjusted Rand **−0.0010**.
  Renaming clusters changes neither ARI nor NMI, which is why accuracy
  against cluster labels is undefined.
- **Preprocessing, measured both ways.** Cluster indicators take logistic
  regression on the moons from 0.9056 to **0.9944**; on `breast_cancer` the
  same recipe gains 0.0058, which is **one patient**, and the notes report
  that null result as a null result. Cluster-then-label reaches **0.9389**
  on a budget of 50 labels against 0.8104 spent at random and 0.9722 for all
  1257 — at the cost of 4.06% wrong propagated labels.
- **Applications kept concrete**: colour quantisation from 96,615 colours to
  16 at MSE 0.0018; k-means on unlabelled digits reaching 79% purity; DBSCAN
  as an outlier detector with precision 1.0000 and recall **0.6667**,
  reported as it happened.

## Before the next lecture

- Read `notes.pdf`. The practice questions have full worked solutions, and
  three of them are pure hand arithmetic.
- **Invent a third clustering of the six points and compute *W*, *B* and *T*
  with a pen.** If *T* does not come to 91, one of your centroids is wrong.
  This is the most examinable calculation in the unit.
- **Work one silhouette by hand**, including one that comes out negative.
  Then say in a sentence what the negative sign means.
- Before you cluster anything in the lab, write down what *similar* means and
  what you will do with the groups. Both answers change the algorithm you
  should pick, and neither is in the data file.
- Next up is what "similar" means for binary, nominal and ordinal columns —
  the distances this lecture kept assuming.

## About the figures

Every figure here is computed, not drawn. The generating script lives with
the course's internal tooling and prints each measured number that the notes
quote, so the prose and the pictures cannot drift apart. If a number in the
notes looks wrong, run the code beside it — that is the check that matters.

## Build

```bash
cd latex
for i in 1 2 3; do pdflatex -interaction=nonstopmode slides.tex; done
for i in 1 2 3; do pdflatex -interaction=nonstopmode notes.tex;  done

grep -c '^!'                slides.log notes.log   # must be 0
grep -c 'Overfull \\vbox'   slides.log notes.log   # must be 0
```

Compile from inside `latex/` — the `../../../_shared/` paths depend on it.
