# Unit VII — Lecture 23: Partitioning and Density-Based Clustering

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The fourth of the Unit VII lecture packages. Budgeted at a minimum of 2 contact
periods.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| The partitioning problem stated precisely, and the WCSS objective | 58 |
| Why exhaustive search is not an option: Stirling numbers, and NP-hardness | 58 |
| Lloyd's algorithm in eight lines — assign, update, repeat | 58 |
| A complete *k*-means run by hand: seven points, *k* = 2, every half step | 58 |
| Why it converges, and why only to a local optimum | 58 |
| Initialisation, *k*-means`++` and `n_init` | 58 |
| Choosing *k*: the elbow, the silhouette, and what neither of them is | 58 |
| The three assumptions inside the objective, and four data sets that break them | 58 |
| *k*-medoids: an actual object, any dissimilarity, and no squares | 58 |
| PAM — BUILD and SWAP — its per-iteration cost and its memory wall | 58 |
| CLARA: sample, run PAM, assign everything, keep the best | 58 |
| Density-based clustering: core, border, noise, and density-reachability | 59 |
| DBSCAN worked by hand on eleven points, arithmetically | 59 |
| Choosing ε from the *k*-distance graph, and DBSCAN's one global ε | 59 |
| Which of the four methods, when — one table, every row measured | 58–59 |

Course outcome: **CO4** — apply and evaluate learning techniques, and select an
appropriate model for a given problem.

References: HKP §10.2, §10.2.2 and §10.4; ISLP §12.4.1.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 65 frames, 73 overlay pages — for the room |
| `notes.pdf` | 38 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | eleven vector figures, every one produced by running code rather than drawn |

## The one idea

**Every method here is an answer to one question: what is a cluster
represented by?** *k*-means says a cluster is represented by its **mean**, so
the boundary between two clusters is a perpendicular bisector and every cluster
is convex. *k*-medoids says the representative must be an **actual object**, so
any dissimilarity will do and no single far-away point can drag the
representative anywhere. CLARA says the representative may be chosen from a
**sample**, so the method becomes linear in *n*. DBSCAN refuses the question
entirely: a cluster is **a connected dense region**, it has no representative,
its shape is arbitrary, *k* is discovered rather than supplied, and a point is
allowed to belong to nothing.

Once you can state which of those four sentences a method is committed to, the
rest follows — the shape it can find, the outliers it can survive, the size of
data it can touch, and whether you have to know *k* before you start.

## What the lecture turns on

Every number in the notes was produced by running the code shown beside it, on
scikit-learn's bundled datasets and fixed-seed synthetic problems. **Nothing
downloads anything.**

- **A complete *k*-means run worked entirely by hand.** Seven points, *k* = 2,
  starting from two centroids deliberately placed in the same clump. The
  distance table at every assignment step, the means at every update step, and
  the objective after **every half step**: 123.0000 → 32.2000 → 16.7400 →
  11.4167, converged in three iterations with exactly one point (*P*₃) changing
  cluster. The hand answer then matches `KMeans`'s `inertia_` of 11.416667 to
  1e-12 — and all 63 two-way partitions of the seven points were scored
  exhaustively, so we know it is the global optimum here. This is the
  examinable calculation of the first half.
- **Eleven points labelled core, border or noise arithmetically.** With
  ε = 1.5 and minPts = 3, the full 11 × 11 distance matrix gives **7 core
  points, 2 border points, 2 noise points and 2 clusters** — and *k* was never
  supplied. `DBSCAN`'s `labels_` and `core_sample_indices_` agree exactly. This
  is the examinable calculation of the second half.
- **Convergence is proved, and then bounded honestly.** Both steps are shown to
  be non-increasing in WCSS, and the set of partitions is finite, so the
  algorithm halts. But twenty runs on six well-separated blobs, changing only
  the seed, give **twelve distinct final WCSS values**, the worst 5010.58
  against the best 431.54 — **11.6×**. Every one of them is a genuine local
  optimum.
- **The fix, priced.** Over 200 single starts, *k*-means`++` reaches the best
  solution **99.5%** of the time against **32.0%** for uniform random seeding.
  Ten random restarts do exactly ten times the work of one. Never report a
  *k*-means result from a single random start, and always record the seed.
- **Choosing *k*, honestly.** On standardised iris the chord elbow says 3, the
  drop-ratio elbow says 2, the silhouette says 2 (0.5818) and the best ARI is
  at 3 (0.6201) — two automations of the *same* heuristic disagree with each
  other before the silhouette is even consulted. And 500 points drawn
  uniformly at random, with no cluster structure at all, still have a
  silhouette maximum, at *k* = 4.
- **The three assumptions, and what they cost.** ARI 0.274 on two moons,
  −0.002 on concentric circles, −0.001 on two long thin bands; and on a
  400/40/40 truth *k*-means returns 208/176/96 — it cut the big cluster into
  three rather than leave the small ones alone.
- **The mean is not robust, and the medoid is.** One extra point at (12, 9)
  moves the right-hand centroid by **2.1506** *and changes the partition* from
  3/4 to 4/4; the medoids stay at *P*₁ and *P*₅. Two outliers among 302 points
  take *k*-means from ARI 1.000 to 0.000 while *k*-medoids does not move at
  all — and does not move until there are twenty. Robust, not immune.
- **PAM, all twenty-one medoid pairs scored by hand**, best {*P*₁, *P*₅} at
  *E* = 7.8929, and a SWAP run from a bad start that improves by 18.6855 and
  then stops because the best remaining swap makes *E* **worse** by 0.5858.
  PAM also clusters eight words by Levenshtein distance, where *k*-means is
  simply not defined.
- **What PAM costs, and why CLARA exists.** Two orders of magnitude slower
  than *k*-means at *n* = 4000 — exactly, 7,998,000 distances before its first
  swap against 16,000 per Lloyd iteration, a factor of 500 — and growing like
  *n*^2.3. At *n* = 100,000 with *k* = 5 one
  SWAP iteration considers **5.0 × 10¹⁰** candidate evaluations against a
  dissimilarity matrix of **74.5 GiB**. CLARA does **1576× less swap work**
  than PAM at *n* = 4000 for **4.4%** more cost, and gives a different answer
  on every seed — a 5.3% spread over ten.
- **One result that is the objective's fault, not the sampling's.** On 6040
  points split 3000/3000/40, PAM *itself* puts two medoids in one big cluster
  and none on the small one, because the "correct" solution genuinely costs
  more: 7502.069 against 7239.719. Minimising total dissimilarity does not
  reward small clusters.
- **What DBSCAN buys.** ARI **1.000** on moons, circles and bands, where
  *k*-means scores 0.274, −0.002 and −0.001; the right number of clusters found
  at true *k* = 2, 3, 5 and 8 with ε and minPts held **fixed**; and 57 of 60
  planted noise points flagged (95.0%) at a cost of 2 genuine points in 300.
- **And what it costs.** One global ε cannot serve two densities: across a full
  sweep, either the loose cluster is entirely noise or the two tight ones have
  already been welded together — **there is no window**. The automatic
  *k*-distance knee on two moons gives 0.0875, which yields 7 clusters and ARI
  0.320, while the working range 0.12–0.25 gives ARI ≈ 1 and 0.27 collapses to
  a single cluster. In high dimensions the idea stops meaning anything: the
  ratio of largest to smallest nearest-neighbour distance falls from 300.4036
  at *d* = 2 to 1.3444 at *d* = 100, and on wine, breast cancer and digits
  DBSCAN finds **one** cluster.
- **The comparison table, measured.** At *n* = 200,000 with *k* = 5: *k*-means
  0.270 s, CLARA 0.065 s, DBSCAN 5.745 s (machine-dependent) — and PAM's
  distance matrix alone
  would be **298.0 GiB**.

## Before the next lecture

- Read `notes.pdf`. The practice questions have full worked solutions, and four
  of them are hand runs — two *k*-means, two DBSCAN.
- **Work a full *k*-means run with a pen.** Six to eight points, *k* = 2, two
  of the points chosen *badly* as initial centroids. Write the distance table
  at every assignment step, the means at every update step, and the WCSS after
  every half step. Check it against `KMeans(init=..., n_init=1)`.
- **Then label ten to twelve points core, border or noise** for a chosen ε and
  minPts, straight from the distance matrix, chain the core points into
  clusters, and check against `DBSCAN`. Remember that `min_samples` counts the
  point itself, and that `labels_` uses `-1` for noise.
- A test on Unit VII follows this lecture, covering hierarchical, partitioning
  and density-based clustering together. The two hand calculations above are
  its numerically examinable core.
- Next up is **consolidation across the whole course** — the last lecture, and
  the argument that ties all seven units together.

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
