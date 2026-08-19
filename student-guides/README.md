# VS Code for lab submission

**▶ Watch:
[`VS-Code-Lab-Submission-Tutorial.mp4`](VS-Code-Lab-Submission-Tutorial.mp4)**
— 4 min 55 s, 1080p. Click the file, then Download or play it in the browser.

This page is the same thing in text, so you can follow along without scrubbing.
Worked on **Lab Assignment 1**; the same steps apply to LA01–LA10.

**What you submit, every time:** `AML_LA01_<yourSAPID>.ipynb` + `README.md`, both
inside the assignment folder. No datasets.

---

## Part 1 — Get the repository (once)

**Option A — terminal**

```bash
git clone https://github.com/tali7c/Applied-Machine-Learning.git
cd Applied-Machine-Learning
git pull        # later, to get new lectures and briefs
```

**Option B — VS Code.** Welcome screen → **Clone Git Repository…** → paste the
same URL → choose a location. VS Code offers to open it when the clone finishes.

## Part 2 — Open it as a *folder*

1. **File → Open Folder** → select `Applied-Machine-Learning` itself, not a file
   inside it. Opening a single file breaks the kernel and relative paths.
2. The Explorer now shows `Unit-01…`, `lab-assignments/`, `Course-Handbook/`.
   After the first time it appears under **Recent** — one click to reopen.

## Part 3 — Two extensions

| Extension | Publisher | Why |
|---|---|---|
| **Jupyter** | Microsoft | Runs `.ipynb`; gives you Run All and Select Kernel |
| **vscode-pdf** | tomoki1207 | Displays `assignment.pdf` inside VS Code |

3. Extensions panel: four-squares icon in the left bar, or **Ctrl+Shift+X**.
4. Jupyter is usually already under **INSTALLED**. If not, search and install it.
5. Under **RECOMMENDED**, click **vscode-pdf** → **Install**.
6. Answer the trust prompt with **Trust Workspace & Install**. This is your own
   cloned repository. Until you do, the yellow *Restricted Mode* banner stays and
   notebooks will not run.

> Without vscode-pdf a brief opens as pages of `%PDF-1.5` and red control
> characters. Your file is fine — either install the extension or open the PDF in
> your normal reader.

## Part 4 — Working Lab Assignment 1

7. `lab-assignments/LA01/` contains four things: `assignment.pdf` (the brief you
   are marked against), `AML_LA01_starter.ipynb`, `latex/`, and `README.md`.
8. **Copy** the starter and rename the copy `AML_LA01_<yourSAPID>.ipynb`. Never
   work in the starter. Fill in the Name / SAP ID / Batch table in the first cell.
9. Click **Select Kernel** (top right) and pick your Python 3 environment.
   Nothing runs until you do — "my code does nothing" is almost always this.
10. Open **OUTLINE** at the bottom of the Explorer. **Double-click** a heading to
    jump to that question. (Single-click does not navigate.)

### Every question is three cells

```
markdown   the question, with its marks
code       # A1 — code          ← your code goes here
markdown   **Observation (A1):**  write 2–6 lines here
```

Fill in **both**. Code with no written observation is capped at half marks.

### The sections

| Section | Questions | Rule | Marks |
|---|---|---|---|
| A | A1, A2, A3 | attempt any **2 of 3** | 2 each |
| B | B1, B2 | attempt any **1 of 2** | 4 each |
| C | C1 | **compulsory** | 2 |

- **A1** needs no dataset — print your library versions. **A2** is a first look at
  the data, **A3** missing values and outliers. Answering all three is not
  penalised; the best two count.
- **B1** compares KNN with and without `StandardScaler`. **B2** is correlation and
  leakage. Pick one and do it properly.
- **C1** is the reproducibility check: restart the kernel, run top to bottom, and
  state whether every number reproduced. If one did not, say which and why —
  that is a valid, markable answer.

### The rule that costs the most marks

- Fit every transformer **inside a `Pipeline`**.
- Never fit a scaler or imputer on the test set.
- Set `random_state` wherever the question asks for reproducibility.

Fitting before the split leaks test information and inflates your score.

11. **Write `README.md`** in the same folder: name and SAP ID, a table of which
    questions you attempted, your results, and your observations in plain
    sentences. It is a report, not a log.

## Before you submit

- [ ] Restart Kernel → **Run All**, top to bottom, no errors
- [ ] Every output visible in the saved notebook
- [ ] Filename is `AML_LA01_<yourSAPID>.ipynb` with your real SAP ID
- [ ] `README.md` beside it, same folder
- [ ] Every attempted question has a written observation

## Get your data before the lab

Some datasets download on first use and the campus network blocks that download
(`CERTIFICATE_VERIFY_FAILED`). Run this **at home**, on a normal connection:

```bash
python -c "from sklearn.datasets import fetch_california_housing as f; f()"
```

Browsers work where Python does not, so a manual download also works on campus.
Full details: *Getting the data* in `Anchor-Datasets.pdf`.
