# Project X — the tutor

An AI tutor for the Ontario curriculum, grades 1-12 plus first-year UWaterloo. Runs entirely on your laptop.

- **Grade-first navigation** — pick your grade, then your subject, then your strand within the subject.
- **Alloprof-inspired UI** for elementary — bright colored pills, playful, kid-friendly.
- **Three practice modes** — classic (with hint escalation), flashcards (flip and rate), blitz (60-second timed).
- **Full explanations always** — the "why" is shown after every answer, correct or wrong.
- **Image generator** — draws diagrams for multiplication questions (5 × 3 = 5 groups of 3), fractions, and addition, right in the browser.
- **Teach the AI** — upload PDFs / DOCX / TXT; the backend chunks, embeds via Ollama, and cites the passages back in chat.

---

## Requirements

- **Python 3.11+**
- **Chrome, Safari, or Firefox**
- **Ollama** *(optional, for real AI chat and embeddings)* — [ollama.com/download](https://ollama.com/download)
  - After install:  `ollama pull llama3.2:3b`  and  `ollama pull nomic-embed-text`
  - Without Ollama the app still runs; chat falls back to scripted answers and content search falls back to keyword matching.

---

## Run it — one line

```bash
cd project-x
python3 -m pip install -r requirements.txt && python3 tutor.py
```

Chrome opens automatically at http://localhost:8000. Ctrl+C in the terminal to stop.

If port 8000 is busy or you want a clean database:

```bash
lsof -ti:8000 | xargs kill -9 2>/dev/null; rm -f tutor.db && python3 tutor.py
```

---

## Push it to GitHub

From inside `project-x/`:

```bash
git init
git add .
git commit -m "Initial commit: Project X — Ontario K-12 + UW AI tutor"
```

Then create an empty repo on github.com (call it `project-x`, don't add any files from the web UI), then:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/project-x.git
git push -u origin main
```

Anyone can then `git clone` your repo and run it with the two commands above.

---

## Grade + subject coverage

The seed content covers the shape of the Ontario curriculum — a starter kit, not every expectation. Each outcome ships with a concept card, a worked example, common-mistake warning, and 3 practice questions.

| Level | Grades | Subjects |
|---|---|---|
| Elementary | 1-8 | Math · English · French · Science |
| Secondary  | 9-12 | Math · English · French · Science |
| University | UWaterloo 1A/1B | MATH 106 · MATH 127 · MATH 128 · CS 115 · CHEM 120 · PHYS 111 |

Within each grade + subject, outcomes are grouped by **strand** — for Grade 9 Math that's Algebra, Analytic Geometry, Data, and so on. The strand chips at the top of the panel filter which outcomes are visible.

---

## Extending the curriculum

The seed data lives at the top of `tutor.py` in the `SEED` dict. Adding a new outcome is copy-paste:

```python
"E9.2": _o("Solving multi-step linear equations",
    "One-line summary shown in the sidebar.",
    grade=9, strand="Algebra",
    concepts=[_c("id", "Name", "sub-title",
        "one-liner explanation",
        "full body paragraph",
        "worked example (multi-line ok)",
        "common misconception")],
    questions=[
        _q("q1", "mc", diff=2, prompt="Question?", answer="0",
           hints=["hint 1", "hint 2"], solution="worked answer",
           choices=["A", "B", "C", "D"]),
        _q("q2", "numeric", diff=3, prompt="What is 2+2?", answer="4",
           hints=["hint"], solution="2+2=4", tol=0.001),
    ]),
```

Delete `tutor.db` and restart — the new content seeds automatically.

---

## Layout

```
project-x/
├── README.md          # this file
├── LICENSE            # MIT
├── .gitignore
├── requirements.txt   # Python deps
├── tutor.py           # the whole app — one file
├── uploads/           # where uploaded PDFs / DOCX land
└── tutor.db           # SQLite (created on first run)
```

---

## License

MIT. Do what you want with the code. The seed curriculum content is provided as educational fair-use; the Ontario curriculum documents themselves are © Queen's Printer for Ontario.
