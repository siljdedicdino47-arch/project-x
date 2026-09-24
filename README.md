# Project X — the tutor

An AI tutor for the Ontario K-12 curriculum. Runs entirely on your laptop — no account, no cloud, nothing billed.

- **Grade-first navigation** — pick a grade, ask the tutor on the landing page, or pick a subject to open a full unit path.
- **Units, Duolingo-style** — each subject breaks into real Ontario strands (units); each unit is a path of lesson icons you can tap in any order. Finished lessons get a checkmark and stay open for more practice.
- **A small AI chat dock** rides alongside every unit so you can ask a question without losing your place.
- **Diagnostic + Summary + unit-final quizzes** — the unit-final quiz puts questions you've gotten wrong before at the front of the list.
- **"Explain it simpler"** — one click asks the AI to re-explain the current concept in plainer language, inline.
- **Image generator** — draws diagrams for multiplication, fractions, ratios, number lines, and labeled rectangles/triangles, right in the browser, with no external service.
- **Training Studio** — upload PDFs/notes, teach the tutor direct Q&A examples, and test the result live, side by side, before you trust it.
- **My AI** — an optional, small, from-scratch language model (plain PyTorch, no Anthropic, no Ollama, no network calls) that trains on your own content and saves its weights to disk. Realistic expectations: this is a genuinely local, genuinely yours model, not a claim it matches a lab-scale LLM — quality scales with how much you teach it.
- **Achievement Chart tags** — every question is labeled with the Ontario category it tests (Knowledge & Understanding, Thinking, Communication, Application).

---

## Requirements

- **Python 3.11+**
- **Chrome, Safari, or Firefox**
- **Ollama** *(optional, default AI engine)* — [ollama.com/download](https://ollama.com/download)
  - After install:  `ollama pull llama3.2:3b`  and  `ollama pull nomic-embed-text`
  - Without Ollama the app still runs; chat falls back to scripted answers and content search falls back to keyword matching.
- **PyTorch** *(optional, only needed for "My AI")* — `pip install torch`. Everything else works without it; My AI just won't be available until it's installed.

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

The seed content covers the shape of the Ontario curriculum — a broad starter kit, not a full transcription of every ministry expectation (that runs thousands of pages across dozens of courses). Each outcome ships with a concept card, a worked example, a common-mistake warning, and 3 practice questions tagged with an Achievement Chart category.

| Level | Grades | Subjects |
|---|---|---|
| Elementary | 1-8 | Math · English (Language) · French · Science |
| Secondary  | 9-10 | Math · English · French · Science (combined) · Geography · History · Business · Civics · Computer Science |
| Secondary  | 11-12 | Math (MCR3U/MHF4U/MCV4U) · English · **Biology, Chemistry, and Physics as three separate courses** (SBI/SCH/SPH — they split from the combined Science course after Grade 10, matching the real Ontario course codes) · French · Business · Computer Science |

Within each grade + subject, outcomes are grouped by **strand** (shown in the UI as **Units**) — for Grade 9 Math that's Number, Algebra, Data, Geometry and Measurement, and Financial Literacy, matching the real MTH1W (2021 destreamed) curriculum. Course codes throughout (MTH1W, MPM2D, MCR3U, SBI3U, SCH4U, …) were checked against the live Ministry course listing at [dcp.edu.gov.on.ca](https://www.dcp.edu.gov.on.ca/en/curriculum).

Each unit renders as a **lesson path**: tap any lesson icon (no locking — every lesson is open from the start), read the concept, answer its mini quiz. A completed lesson gets a checkmark and stays open — tap it again anytime to redo the practice questions. Once every lesson in a unit is done, the trophy node unlocks a unit-final quiz that puts previously-missed questions first.

---

## Extending the curriculum

There are two ways content gets into the app — both are merged by `seed_if_empty()` at first launch (SEED takes priority if the same outcome code appears in both):

**1. `SEED`** — the original hand-authored dict near the top of `tutor.py`. Copy-paste a new outcome into any course block:

```python
"M9.X1": _o("Solving multi-step linear equations",
    "One-line summary shown in the sidebar.",
    strand="Algebra",
    concepts=[_c("id", "Name", "sub-title",
        "one-liner explanation",
        "full body paragraph — keep sentences short, one idea each",
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

**2. `ONTARIO_*` row lists** (`ONTARIO_MATH_ELEM`, `ONTARIO_MATH_SECONDARY`, `ONTARIO_SECONDARY_BROAD`, etc., found via the `SEED` banner) — a more compact tuple format used for the bulk of the curriculum expansion, good for adding many outcomes quickly:

```python
("course-id", "CODE1", "Outcome name", "One-line blurb", "Strand name",
 "Concept body — short sentences, one idea per sentence.",
 "Worked example, multi-line ok.",
 [("Question prompt?", "answer", None, "numeric", 2, "hint", "solution text"),
  ("MC prompt?", "0", ["choice A","choice B","choice C","choice D"], "mc", 1, "hint", "solution text")]),
```

Question tuples are `(prompt, answer, choices_or_None, kind, diff, hint, solution)`. `kind` is `"mc"` (answer is the 0-based correct index as a string, `choices` required) or `"numeric"` (answer is the value, `choices` is `None`). Append your new list to `ALL_UNIT_ROWS` near the bottom of the SEED section.

New strand names should get an entry in `STRAND_CATEGORY` (maps to K/T/C/A — see `ACHIEVEMENT_CATEGORIES`) and, optionally, a pattern in `STRAND_ICON_MAP` (client-side JS) so its unit cards get a sensible icon instead of the subject's generic fallback.

In every case: **delete `tutor.db` and restart** — new content seeds automatically. (This also wipes personal progress/mastery data, so don't do it mid-use.)

---

## How it's built

Single file, no build step, no framework. Rough shape (search `tutor.py` for these banners):

- **`DB —`** SQLAlchemy models. SQLite in WAL mode. `Course → Outcome → Concept`/`Question`, plus `Attempt`/`Mastery` (progress), `Source`/`Chunk` (uploaded content + embeddings), `LogEvent` (history), `TrainingExample` (hand-taught Q&A pairs).
- **`SEED —`** the curriculum content — see "Extending the curriculum" above.
- **`MY AI —`** a small transformer language model written from scratch right here: a regex tokenizer built from your own corpus, a tiny multi-head-attention decoder (`MyAIModel`), a training loop (`myai_train_sync`), and greedy/top-k sampling generation (`myai_reply`). No Anthropic, no Ollama, no network call of any kind — pure local PyTorch. Weights + vocab save to `~/Library/Application Support/project-x/my_ai/` (or the project folder) so training survives a restart.
- **`INGEST + LLM`** Ollama client helpers, the RAG retrieval function (`retrieve()` — cosine similarity over uploaded chunks, falls back to keyword overlap if no embeddings), and `build_system_prompt()` — the tutor's full voice + behaviour rules for one chat turn.
- **`FASTAPI`** every HTTP route: `/api/courses`, `/api/chat` (takes an `engine` of `auto`/`ollama`/`my_ai`), `/api/attempts`, `/api/mastery`, `/api/wrong-attempts`, `/api/training/*`, `/api/my-ai/*`, `/api/sources/*`, `/api/log`.
- **`HTML_PAGE`** the entire frontend as one big Python string: CSS custom properties for theming (including per-grade-band sizing), then vanilla JS. `h(tag, props, children)` is a tiny hyperscript helper; `state` is one plain object; `render()` tears down and rebuilds `#app` from `state` on every change — no virtual DOM, no diffing, no framework. Simple enough to read top to bottom; fast enough for this app's size that it's never been worth adding one.

---

## Layout

```
project-x/
├── README.md          # this file
├── LICENSE            # MIT
├── .gitignore
├── requirements.txt   # Python deps (torch is optional, only for My AI)
├── tutor.py           # the whole app — one file
├── uploads/           # where uploaded PDFs / DOCX land
├── my_ai/             # My AI's saved weights + vocab (created once trained)
└── tutor.db           # SQLite (created on first run)
```

---

## License

MIT. Do what you want with the code. The seed curriculum content is provided as educational fair-use; the Ontario curriculum documents themselves are © King's Printer for Ontario.
