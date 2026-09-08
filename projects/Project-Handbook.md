# AML Project Handbook — rules that apply to every theme

Applied Machine Learning (CSAI2017P) · Autumn 2026 · Dr. Tofik Ali

Read this once. It applies to all eight themes P01–P08. The theme folder tells you
*what* to build; this handbook tells you *how it is submitted and marked*.

---

## 1. The component in one table

| Sub-component | Marks | Granularity | Basis |
|---|---|---|---|
| **Project quiz** — a written paper, one set per theme | **10** | **per student** | your own understanding of your group's work |
| **Repository submission** — the state and the history of the repo | **5** | **per group** | identical mark for every member |

Total 15 of the 100-mark course scale.

**The 5 group marks are identical for every member of the group. They are not
split by contribution.** All individual differentiation happens in the 10-mark
written quiz, which every member sits separately. A member who did nothing still
receives the group's 5 marks — and then faces a paper about work they did not do.
That is the design. Choose your group knowing it.

**Both parts are compulsory.** A student who does not sit the quiz scores 0 of 10
for it; a group that does not submit a repository link scores 0 of 5.

---

## 2. Groups

- **4 to 8 students**, self-formed, from the same section.
- **At most two groups per theme.** Sign-up is first come, first served on the
  sheet; once a theme has two groups it is closed.
- One member is the **repo owner** — they create the repository and add everyone
  else as a collaborator. This is an administrative role, not a leadership one,
  and it earns no extra marks.
- Group membership is frozen at the end of Unit IV. Tell me before that if it
  changes; after that it does not change.

---

## 3. The GitHub repository — the only thing you submit

At the end, you send me **one line: the repository URL.** Nothing else. No zip,
no email attachment, no printed report.

### 3.1 Creating it

- One **public** repository per group, on GitHub, named
  `aml-2026-<theme-code>-<group-number>` — for example `aml-2026-p03-g2`.
- The repo owner adds **every member as a collaborator**, and adds me as a
  collaborator too if you choose to keep it private (public is simpler; either
  is accepted).
- Create it **by the end of Unit IV** — at the start of the work, not at the end
  of the semester. The creation date is visible and it is part of what I look at.

### 3.2 What I check when you send the link

I open the repository and read **the commit history**, not just the final files.
Specifically:

| I look at | What I am checking |
|---|---|
| `git log` over time | that the work was spread across the semester |
| who authored each commit | that more than one person actually pushed |
| commit messages | that they describe changes, not "update", "final", "final2" |
| the two checkpoint tags | that the checkpoints were real, not backdated |
| diffs at each checkpoint | that the state then matches what you claimed then |

### 3.3 The commit-history floor

A repository that meets all of these is safe:

1. **Created by the end of Unit IV** and first commit within that week.
2. **At least 20 commits** by the final submission.
3. Commits fall on **at least 6 distinct calendar weeks**.
4. **No more than 40 % of the commits in the final week.**
5. **Every member has commits under their own GitHub account.** Use your own
   account, set `user.email` correctly, and do not let one person push everyone's
   work. This is a condition on the *group's* repository, so a repository where
   one person pushed everything loses process marks for the whole group — and the
   members who never committed will also find the quiz very hard.
6. Two annotated tags, `checkpoint-1` and `checkpoint-2`, pushed **at the time of
   each checkpoint**.

**What fails.** A repository whose entire history is 1–3 commits in the last few
days before submission — the whole project uploaded at once, files unorganised, one commit called
"project" — scores **at most 1.5 of 5** for the group component, however good the
code inside it is. This is stated in advance and applied without exception.

**Honest exceptions exist.** If a genuine problem broke your history — a repo
recreated after an accident, a member's account issue — tell me *at the time*,
not at submission. A note in the README written on the day it happened is
evidence; an explanation invented at the end is not.

### 3.4 Repository layout

Every group uses this structure. It is the same for all eight themes; only the
number and names of the notebooks vary, and your theme brief tells you which.

```
aml-2026-pNN-gM/
├── README.md            # what the project is, how to run it, who is in the group
├── requirements.txt     # pinned versions
├── .gitignore
├── data/
│   ├── raw/             # the downloaded file(s), unmodified  (see §4)
│   └── processed/       # generated — usually gitignored
├── notebooks/          # numbered in run order; your theme brief names them
│   ├── 01-eda.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-models.ipynb
│   └── 04-evaluation.ipynb
├── src/                 # reusable functions imported by the notebooks
├── results/
│   ├── figures/
│   └── metrics.csv      # every number you report, in one file
├── report/
│   └── report.pdf       # 4–6 pages
└── REPRODUCIBILITY.md   # §3.6
```

### 3.5 The README must contain

- The theme code and title, and the group number.
- **Every member: name, enrolment number, GitHub username.**
- One paragraph on the question the project answers.
- The data source URL and the exact download steps.
- How to run it: environment, then the commands, in order.
- The headline result — the metric, the number, and on which split.
- A short **"who did what"** section. It does not change any mark. It exists so
  that the group agrees, in writing and in advance, on what each person will be
  able to answer for in the quiz.

### 3.6 REPRODUCIBILITY.md must contain

- Python version and how to recreate the environment.
- Every random seed used, and where it is set.
- The exact sequence of notebooks/scripts to run.
- Expected runtime and hardware.
- Any manual step (a browser download, a file placed by hand) spelled out.
- Anything that does **not** reproduce exactly, and why.

I will attempt a clean-checkout run on at least a sample of repositories.
A project that cannot be re-run is a claim, not a result.

---

## 4. Data on the UPES network

The campus TLS appliance breaks Python downloads — `fetch_openml`, `ucimlrepo`,
`yfinance`, `requests`, `pip` from some hosts. Details in
`../lab-assignments/Anchor-Datasets.pdf`.

So, for every theme:

- **Download the file once in a browser** (at home, on mobile data, or on campus —
  browsers work), and commit it to `data/raw/` if it is under 50 MB.
- If it is larger than 50 MB, do **not** commit it. Put it in `.gitignore`,
  and write the download instructions in the README precisely enough that I can
  reproduce your `data/raw/` myself in under two minutes.
- Mirrors of the raw files are placed on the LMS where licensing permits; check
  there first.
- **Never disable certificate verification.** `verify=False` or an unverified SSL
  context anywhere in the repository costs marks. It is the wrong lesson.

---

## 5. Milestones

Anchored to units, not dates. Exact dates announced in class.

| # | Milestone | Anchored to | Evidence |
|---|---|---|---|
| 1 | Catalogue released | during Unit IV | — |
| 2 | Groups formed, theme selected, **repo created** | end of Unit IV | repo URL on the sign-up sheet |
| 3 | **Checkpoint 1** — data loaded, one baseline number | end of Unit V | tag `checkpoint-1` |
| 4 | **Checkpoint 2** — full pipeline, initial results | end of Unit VI | tag `checkpoint-2` |
| 5 | **Final submission** — repo link sent | end of Unit VII | the URL |
| 6 | **Project quiz** — written, individual | final lecture / lab-15 slot | — |

**Checkpoint 1 is real, not a plan.** You show the data actually loaded — shape,
head, a chart — and one honest baseline number. Groups whose dataset turns out not
to exist are caught here, when there is still time to move. A checkpoint tag
pushed after the fact is worse than a missed checkpoint honestly declared.

---

## 6. The 5 group marks — rubric

| Criterion | Marks | Full marks looks like |
|---|---|---|
| **Process & commit history** | **1.0** | meets all six floor conditions in §3.3 — including commits from every member's own account; messages describe changes; both checkpoint tags pushed on time |
| **Completeness** | **1.5** | every item in the theme's "Done" list is present and demonstrated |
| **Method correctness** | **1.0** | no leakage; split appropriate to the data (stratified / chronological); metric fits the problem and the choice is argued |
| **Reproducibility** | **1.0** | runs from a clean checkout following REPRODUCIBILITY.md; seeds set; reported numbers reappear |
| **Documentation** | **0.5** | README and report are complete, honest about limitations, and readable |
| | **5.0** | |

The "Further" items in a theme brief are not needed for full marks. They are where
a strong group makes the quiz easy for itself.

---

## 7. The project quiz — 10 marks, written, individual

- **A written paper. One set per theme**, so groups on different themes sit
  different papers. Both groups on the same theme sit the same set.
- Conducted in the final lecture / lab slot. Closed-book unless announced
  otherwise; your own repository may be open on screen.
- Every member sits it **separately**. Answers are your own.
- The questions are **not pre-released**. What is pre-released is the *kind* of
  question:

  1. Trace the path from the raw file to one number in your report.
  2. You reported metric M. Why M and not the obvious alternative?
  3. Here is a block of code from your repository. What does it do? What breaks
     if line *k* is removed?
  4. Your result would change if X changed. Up or down, and why?
  5. What is the weakest part of your project, and what would you do with two
     more weeks?

- **Marking is on understanding, not fluency.** "I don't know, but I would find
  out by running Y and looking at Z" earns more than a memorised paragraph that
  does not fit the question.
- A contributor answers these comfortably. A passenger cannot, however polished
  the repository is.

---

## 8. Academic integrity

- Using AI tools, tutorials, Kaggle notebooks or Stack Overflow is **allowed and
  expected** — provided you (a) say so in the README, citing what you used and
  where, and (b) can explain every line you kept. The quiz tests (b).
- Copying another group's repository, or submitting a lab assignment as a project,
  is plagiarism and is reported.
- The eight themes deliberately avoid every lab anchor dataset, so a re-submitted
  lab is immediately visible.

---

## 9. Getting help

Bring a specific question and your repository link. "It doesn't work" is not a
question I can answer; "the chronological split gives an R² of −0.3 and I expected
it to be positive, here is the cell" is.

Ask early. The gap between Checkpoint 1 and Checkpoint 2 is where projects are won.
