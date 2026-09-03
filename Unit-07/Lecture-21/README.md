# Unit VII — Lecture 21: Similarity Measures and Data Types in Clustering

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The second of the Unit VII lecture packages. Budgeted at a minimum of 1 contact
period; it is dense and will comfortably fill two.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Similarity, dissimilarity, and the dissimilarity matrix as the real input | 51 |
| The four metric axioms, and three named measures that violate them | 51 |
| Numeric attributes: Manhattan, Euclidean, Minkowski *p*, Chebyshev, unit balls | 51 |
| Cosine similarity against Euclidean distance, and the identity that links them | 51 |
| Binary attributes: the 2×2 table, simple matching, Jaccard | 52 |
| Nominal attributes: the mismatch ratio, and when one-hot reproduces it | 52 |
| Ordinal attributes: the (r−1)/(M−1) map and the assumption inside it | 52 |
| Mixed types: the Gower weighted combination, worked by hand | 52 |
| Standardisation changes which object is nearest; z-score against MAD scaling | 51 |
| Cluster centroid, medoid, and the single/complete/average/centroid distances | 53 |

Course outcomes: **CO1** — understand the fundamental concepts, terminology and
mathematical foundations; **CO2** — apply appropriate techniques and demonstrate
the underlying computation.

References: HKP §2.4 and §2.4.4–2.4.6, §10.2; ISLP §12.4; Tan et al. §2.4.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 51 frames, 62 overlay pages — for the room |
| `notes.pdf` | 36 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | eleven vector figures, every one produced by running code rather than drawn |

## The one idea

**Every clustering algorithm in this unit takes a distance as its input, and is
only as good as that distance.** Not the data — the *distances*. The
dissimilarity matrix is the entire interface between your table and the
algorithm, and everything you decide about how to measure similarity is decided
before the algorithm runs. Nothing you do to the algorithm afterwards will
recover a badly chosen distance.

The previous lecture said clustering groups *similar* objects and left the word
undefined. This lecture defines it, column type by column type, and then shows
the same algorithm on the same dataset scoring anywhere between 0.05 and 0.90
depending on nothing but which distance you handed it.

## What the lecture turns on

Every number in the notes was produced by running the code shown beside it, on
scikit-learn's bundled datasets and small tables you can check with a pen.
**Nothing downloads anything.**

- **The metric axioms, with named violators.** Non-negativity, identity,
  symmetry, the triangle inequality — and then *squared Euclidean distance*
  breaking the triangle inequality on three points of a line (1 + 1 < 4, and it
  is exactly what k-means minimises), *cosine distance* breaking it on three
  unit vectors (two 45° hops cost 0.585786, the one 90° hop costs 1), and *KL
  divergence* breaking symmetry. The axioms are not decoration: the triangle
  inequality is what licenses every neighbour-search shortcut in the library.
- **One formula, one dial.** Minkowski with p = 1, 2, 3, ∞ on the pair (1,2),
  (4,6): **7, 5, 4.497941, 4**. Then the unit balls, which show what p actually
  controls — how heavily one badly disagreeing attribute is punished relative to
  several mildly disagreeing ones. On the same query, p = 1 picks one neighbour
  and every p ≥ 2 picks another.
- **Cosine against Euclidean, on a pair where they disagree completely.**
  A = (1,2), B = 10A, C = (2,1). Euclidean says C is A's nearest neighbour by a
  factor of fourteen; cosine says B *is* A. Then the same contrast on three term
  vectors, where Euclidean pairs two unrelated *short* documents and puts a
  document 42.21 away from a longer copy of itself. Closed with the identity
  ‖â − b̂‖² = 2(1 − cos), so "use cosine" and "L2-normalise then use Euclidean"
  are the same procedure.
- **Binary attributes, and the one question that decides everything.** Is a
  shared *absence* evidence of similarity? Simple matching says yes, Jaccard
  says no. Worked on three patients and six tests (q = 2, r = 0, s = 1, t = 3),
  then pushed until the two coefficients **rank the same two pairs in opposite
  orders**: 0.8000 vs 0.6000 one way, 0.3333 vs 0.6000 the other. Padding with
  attributes nobody has drives simple matching to 0.9980 and leaves Jaccard
  exactly where it was.
- **Nominal, and whether one-hot is the same thing.** With L1 it reproduces
  (p − m)/p to the last digit; with L2 the ranking survives but the gaps
  compress by a square root; with *standardised* dummies the ranking can
  reverse, because a level occurring once in twenty acquires a weight nobody
  granted it.
- **Ordinal**, with the equal-spacing assumption stated rather than smuggled in.
- **Mixed types, worked by hand.** Four applicants with numeric, ordinal,
  nominal and asymmetric binary columns. d(A,B) = **0.340686** over six
  attributes; d(B,D) = **0.372794** over *five*, because neither owns a car and
  neither holds a patent, so those two attributes leave the numerator and the
  denominator both.
- **Standardisation changes the answer.** A ten-row table where the nearest
  neighbour changes identity when the columns are scaled — and then the same
  effect on real data: **94.4% of `wine` rows get a different nearest
  neighbour** after z-scoring, because one column carries 0.997738 of the raw
  squared distance. Includes z-score against mean-absolute-deviation scaling,
  with the honest finding that on `wine` and `iris` the two barely differ and
  the gap only opens on heavy-tailed columns.
- **Centroid, medoid and the four inter-cluster distances**, computed
  arithmetically on two three-point clusters: single 5.000000, centroid
  5.656854, average 5.715413, complete 6.403124 — in that order, and scipy
  reproduces all four exactly. This is the arithmetic the hierarchical
  clustering lecture starts from.
- **Honest counter-results, kept in.** Standardising *hurt* one row of the
  closing experiment (average-linkage Euclidean fell from 0.2926 to −0.0054),
  and the notes show the cluster sizes that explain why rather than quietly
  dropping the row. Cosine on raw `wine` scored 0.0535, and the notes say why
  that is the correct behaviour rather than a failure.

## Before the next lecture

- Read `notes.pdf`. The ten practice questions have full worked solutions, and
  four of them are pure hand computation.
- **Rebuild the three-patient binary table with a pen**: all three q, r, s, t
  counts, both coefficients, both dissimilarity matrices. This is the single
  most examinable calculation in the lecture.
- **Finish the mixed-type matrix**: compute d(A,C) and d(C,D) showing every
  δ. You should get 0.857143 and 0.974265.
- Before you cluster anything in the lab, write down for each column: its
  **type**, whether it is **symmetric**, its **units**, and what you are doing
  about the units. If you cannot write that table, you are not ready to run the
  algorithm.
- Next up is hierarchical clustering, which takes the dissimilarity matrix and
  the four linkage rules computed here and turns them into a dendrogram.

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
