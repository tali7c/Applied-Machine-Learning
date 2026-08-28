# Theory Quiz 1 (T01)

Released after every group had sat the quiz and all answer sheets were collected.

| | |
|---|---|
| Covers | Prerequisite material, Unit I (Lecture 1), Unit II (Lecture 2) |
| Duration | 40 minutes |
| Marks | 20, scaled into the 10-mark theory-quiz component |
| Structure | 15 items — 5 MCQ, 4 fill-in-the-blank, 3 true/false, 2 short answer, 1 reasoning |
| Sets | Eight (A–H), four pages each |

## Files

- `T01-QP.pdf` — the question paper, all eight sets (32 pages)
- `T01-solutions.pdf` — answer key and marking scheme for every set
- `T01-continuation-sheet.pdf` — the blank continuation sheet used in the sitting
- `latex/` — sources. `content.py` holds the items, `build.py` generates
  `T01-QP.tex` and `key.tex` from them, so the eight sets cannot drift apart.

## About the eight sets

Every set is generated from one blueprint: the same item count, the same mark
allocation and the same difficulty. What changes is the order of the options,
the numbers in the computed questions and whether a question is presented
symbolically or as a table. Your set decides the *style* of question you met,
never how hard it was.

## How to use this

Work your own set first, unaided and against the clock, before you open the
solutions. Then read the marking scheme — not only the answers but the reasoning
credit in Parts C, D and E. Most marks lost on T01 were lost on the written
justification, not on the arithmetic.

## Rebuilding

```bash
cd latex
python3 build.py                                       # regenerates T01-QP.tex and key.tex
for i in 1 2 3; do pdflatex -interaction=nonstopmode T01-QP.tex;        done
for i in 1 2 3; do pdflatex -interaction=nonstopmode T01-solutions.tex; done
```
