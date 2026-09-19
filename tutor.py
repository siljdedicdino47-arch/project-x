#!/usr/bin/env python3
"""
Project X — one-file AI tutor for Ontario K-12 + UWaterloo 1A/1B.

Grade → Subject → Strand → Outcome. Elementary (1-8), Secondary (9-12),
University (UWaterloo). Alloprof-style landing screen. Three practice modes:
classic (hint escalation), flashcards (Quizlet-style flip + rate),
blitz (60-second timed). Full explanations after every attempt.
SVG diagram generator for common question shapes.

REQUIREMENTS
    Python 3.11+
    pip install -r requirements.txt

    Optional real AI:
        ollama pull llama3.2:3b
        ollama pull nomic-embed-text

RUN
    python3 tutor.py
"""

from __future__ import annotations
import os, re, shutil, sys, threading, time, webbrowser
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx, numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine)
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            relationship, sessionmaker)


HERE = Path(__file__).parent.resolve()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _pick_writable_paths() -> tuple[Path, Path]:
    """Return (db_path, upload_dir), preferring the project folder but falling
    back to ~/Library/Application Support/project-x if the project folder isn't
    reliably writable (macOS Full Disk Access edge cases, paths with special
    characters, read-only mounts, etc.). SQLite silently opens files read-only
    when write permission is unclear, so we do a real write test up front."""
    for base in (HERE, Path.home() / "Library" / "Application Support" / "project-x"):
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            db = base / "tutor.db"
            up = base / "uploads"
            up.mkdir(exist_ok=True)
            # If a legacy DB exists here, make sure it's writable.
            if db.exists():
                try: db.chmod(0o644)
                except Exception: pass
            if base != HERE:
                print(f"⚠  project folder not writable — using {base}", file=sys.stderr)
            return db, up
        except (PermissionError, OSError):
            continue
    raise RuntimeError("no writable directory available for tutor.db")


DB_PATH, UPLOAD_DIR = _pick_writable_paths()


# ═════════════════════════════════════════════════════════════════════════════
#  DB — WAL mode + safe pragmas so concurrent request handlers don't collide
# ═════════════════════════════════════════════════════════════════════════════
from sqlalchemy import event as _sa_event
from sqlalchemy.engine.url import URL as _URL

# Build the URL through SQLAlchemy's URL helper — handles paths with colons,
# spaces, and other characters that the raw sqlite:/// string can mangle.
_db_url = _URL.create("sqlite", database=str(DB_PATH))
engine = create_engine(_db_url, connect_args={"check_same_thread": False, "timeout": 30})

@_sa_event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, connection_record):
    cur = dbapi_connection.cursor()
    try:
        cur.execute("PRAGMA journal_mode=WAL")     # concurrent readers + writers
        cur.execute("PRAGMA synchronous=NORMAL")   # good balance
        cur.execute("PRAGMA busy_timeout=30000")   # wait 30s on lock
    finally:
        cur.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase): pass

class Course(Base):
    __tablename__ = "courses"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(String(255))     # display name (e.g. "Math" or "MATH 106")
    level: Mapped[str] = mapped_column(String(16))      # elementary | secondary | uni
    grade: Mapped[int] = mapped_column(Integer)         # 1..12 for K-12; 13 for UW-1A, 14 for UW-1B
    subject: Mapped[str] = mapped_column(String(32))    # math | english | french | science | cs | physics | chemistry
    outcomes: Mapped[list["Outcome"]] = relationship(back_populates="course", cascade="all, delete-orphan")

class Outcome(Base):
    __tablename__ = "outcomes"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"))
    code: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(255))
    blurb: Mapped[str] = mapped_column(Text)
    strand: Mapped[str] = mapped_column(String(64))     # curriculum strand
    course: Mapped[Course] = relationship(back_populates="outcomes")
    concepts: Mapped[list["Concept"]] = relationship(back_populates="outcome", cascade="all, delete-orphan")
    questions: Mapped[list["Question"]] = relationship(back_populates="outcome", cascade="all, delete-orphan")

class Concept(Base):
    __tablename__ = "concepts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    outcome_id: Mapped[str] = mapped_column(ForeignKey("outcomes.id"))
    name: Mapped[str] = mapped_column(String(255))
    sub: Mapped[str] = mapped_column(String(255))
    one_liner: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    worked: Mapped[str] = mapped_column(Text)
    misconception: Mapped[str] = mapped_column(Text)
    outcome: Mapped[Outcome] = relationship(back_populates="concepts")

class Question(Base):
    __tablename__ = "questions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    outcome_id: Mapped[str] = mapped_column(ForeignKey("outcomes.id"))
    kind: Mapped[str] = mapped_column(String(16))         # mc | numeric | flashcard
    diff: Mapped[int] = mapped_column(Integer)
    prompt: Mapped[str] = mapped_column(Text)
    choices: Mapped[list | None] = mapped_column(JSON, nullable=True)
    answer: Mapped[str] = mapped_column(Text)
    tolerance: Mapped[float | None] = mapped_column(Float, nullable=True)
    hints: Mapped[list] = mapped_column(JSON, default=list)
    solution: Mapped[str] = mapped_column(Text)           # always shown — the "why"
    outcome: Mapped[Outcome] = relationship(back_populates="questions")

class Attempt(Base):
    __tablename__ = "attempts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    outcome_id: Mapped[str] = mapped_column(ForeignKey("outcomes.id"))
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"))
    mode: Mapped[str] = mapped_column(String(16), default="practice")  # practice | flashcard | blitz
    correct: Mapped[bool] = mapped_column(Boolean)
    hints_used: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Mastery(Base):
    __tablename__ = "mastery"
    outcome_id: Mapped[str] = mapped_column(ForeignKey("outcomes.id"), primary_key=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Source(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    outcome_id: Mapped[str | None] = mapped_column(ForeignKey("outcomes.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(32))
    bytes_size: Mapped[int] = mapped_column(Integer, default=0)
    chunks_n: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="source", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    outcome_id: Mapped[str | None] = mapped_column(ForeignKey("outcomes.id"), nullable=True)
    ord: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    source: Mapped[Source] = relationship(back_populates="chunks")

class LogEvent(Base):
    __tablename__ = "log_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32))
    detail: Mapped[str] = mapped_column(Text)
    outcome_id: Mapped[str | None] = mapped_column(ForeignKey("outcomes.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ═════════════════════════════════════════════════════════════════════════════
#  SEED — Ontario curriculum shape, grade × subject × strand
#  Compact helpers to keep the file readable.
# ═════════════════════════════════════════════════════════════════════════════

def _q(qid, kind, diff, prompt, answer, hints, solution, choices=None, tol=None):
    return {"id": qid, "kind": kind, "diff": diff, "prompt": prompt, "answer": answer,
            "hints": hints, "solution": solution, "choices": choices, "tolerance": tol}

def _c(cid, name, sub, one_liner, body, worked, misconception):
    return {"id": cid, "name": name, "sub": sub, "one_liner": one_liner,
            "body": body, "worked": worked, "misconception": misconception}

def _o(name, blurb, strand, concepts, questions):
    return {"name": name, "blurb": blurb, "strand": strand,
            "concepts": concepts, "questions": questions}


SEED = {

# ═══════════════════ ELEMENTARY · Grades 1-8 · Math ═══════════════════
"e-g1-math": {"label":"Math", "level":"elementary", "grade":1, "subject":"math", "outcomes":{
"M1.N1": _o("Counting to 100", "Say and write numbers to 100 in order.", "Number Sense",
    [_c("m1-n1","How to count","one at a time",
        "Numbers go in order: 1, 2, 3… and never skip.",
        "When you count, you say each number once. After 10 comes 11, after 20 comes 21. The pattern repeats.",
        "1, 2, 3, 4, 5, 6, 7, 8, 9, 10,\n11, 12, 13, ... 19, 20,\n21, 22, ...",
        "Some skip numbers or say them out of order. Slow down — say each one.")],
    [_q("m1n1-q1","numeric",1,"What number comes right after 27?","28",["Add 1."],"27 + 1 = 28.",tol=0.001),
     _q("m1n1-q2","mc",1,"Which is between 50 and 52?","1",["Just after 50."],"51.",choices=["49","51","55","60"]),
     _q("m1n1-q3","numeric",2,"What's 10 more than 43?","53",["Same ones, tens goes up 1."],"43 → 53.",tol=0.001)]),
"M1.O1": _o("Adding within 20", "Adding two small numbers.", "Operations",
    [_c("m1-o1","Count-on strategy","start big, add on",
        "Start with the bigger number, then count on.",
        "For 3 + 5, start at 5 and count 3 more: 6, 7, 8. Faster than counting from 1.",
        "3 + 5\nStart at 5. Count: 6, 7, 8.\nAnswer: 8",
        "Starting from the smaller number takes more steps and more mistakes.")],
    [_q("m1o1-q1","numeric",1,"6 + 3 = ?","9",["Start at 6, count on 3."],"6, 7, 8, 9 → 9.",tol=0.001),
     _q("m1o1-q2","numeric",2,"7 + 8 = ?","15",["Start at 8."],"8, 9, 10, 11, 12, 13, 14, 15.",tol=0.001),
     _q("m1o1-q3","mc",1,"Sum of 4 + 5?","2",["Count on."],"4+5 = 9.",choices=["8","10","9","11"])]),
}},

"e-g2-math": {"label":"Math", "level":"elementary", "grade":2, "subject":"math", "outcomes":{
"M2.N1": _o("Place value to 100", "Reading a two-digit number as tens and ones.", "Number Sense",
    [_c("m2-n1","Tens digit + ones digit","two positions",
        "The left digit tells you tens; the right digit tells you ones.",
        "In the number 47, the 4 is 4 tens (40) and the 7 is 7 ones. 40 + 7 = 47.",
        "47 = 4 tens + 7 ones\n   = 40 + 7\n   = 47",
        "Some read '47' as 'four, seven' instead of 'forty-seven'. Read the value, not the digits.")],
    [_q("m2n1-q1","numeric",1,"How many tens in 63?","6",["Left digit."],"6 tens = 60.",tol=0.001),
     _q("m2n1-q2","numeric",2,"58 = 50 + ?","8",["Ones digit."],"58 = 50 + 8.",tol=0.001),
     _q("m2n1-q3","mc",2,"Which number has 3 tens and 4 ones?","0",["30 + 4."],"34.",choices=["34","43","304","3.4"])]),
"M2.O1": _o("Adding two-digit numbers", "Add up to 100 by splitting into tens and ones.", "Operations",
    [_c("m2-o1","Split, add, combine","tens with tens, ones with ones",
        "Break each number into tens and ones. Add each part. Put them back.",
        "34 + 25: tens are 30 + 20 = 50; ones are 4 + 5 = 9; together 59.",
        "34 + 25\n= (30+20) + (4+5)\n= 50 + 9\n= 59",
        "Adding the whole thing at once causes carrying errors. Split first.")],
    [_q("m2o1-q1","numeric",1,"12 + 15 = ?","27",["Tens: 10+10=20."],"12+15 = 20+7 = 27.",tol=0.001),
     _q("m2o1-q2","numeric",2,"23 + 34 = ?","57",["Split each."],"50 + 7 = 57.",tol=0.001),
     _q("m2o1-q3","mc",1,"45 + 10 = ?","1",["Tens up by 1."],"55.",choices=["45","55","46","145"])]),
}},

"e-g3-math": {"label":"Math", "level":"elementary", "grade":3, "subject":"math", "outcomes":{
"M3.N1": _o("Times tables to 5×5", "Multiplication by skip-counting.", "Number Sense",
    [_c("m3-n1","Skip-count to multiply","jump by the same amount",
        "3 × 4 means 'jump by 3 four times' — or 'jump by 4 three times'.",
        "For 3×4, count 3, 6, 9, 12 (four jumps of 3). Or 4, 8, 12 (three jumps of 4). Same answer.",
        "3 × 4:  3, 6, 9, 12",
        "Don't count the start as jump 0. Four jumps means four numbers said.")],
    [_q("m3n1-q1","numeric",1,"3 × 2 = ?","6",["Count by 3 two times."],"3+3 = 6.",tol=0.001),
     _q("m3n1-q2","numeric",2,"4 × 5 = ?","20",["5, 10, 15, 20."],"4×5 = 20.",tol=0.001),
     _q("m3n1-q3","mc",1,"2 × 3 = ?","2",["Two 3s."],"6.",choices=["4","5","6","8"])]),
"M3.D1": _o("Reading pictographs", "Interpret data from a picture chart.", "Data",
    [_c("m3-d1","Each picture = a value","read the key",
        "Every picture stands for a number told by the key. Multiply pictures by the key value.",
        "If 🍎 = 2 apples and there are 5 apples in the row, that's 5 × 2 = 10 apples.",
        "Key: 🐟 = 3\nRow: 🐟🐟🐟🐟\nCount: 4 × 3 = 12 fish",
        "Some just count the pictures without checking the key.")],
    [_q("m3d1-q1","numeric",2,"🍎 = 5 apples. Row has 4 🍎. How many apples?","20",["4 × 5."],"4 × 5 = 20.",tol=0.001),
     _q("m3d1-q2","numeric",2,"🐟 = 2. Row has 6 🐟. Total fish?","12",["6 × 2."],"6 × 2 = 12.",tol=0.001),
     _q("m3d1-q3","mc",2,"Why do we use a key on a pictograph?","2",["What does the key tell you?"],"Each picture stands for more than 1.",choices=["Decoration","To make it colourful","Each picture stands for more than 1","Because it's required"])]),
}},

"e-g4-math": {"label":"Math", "level":"elementary", "grade":4, "subject":"math", "outcomes":{
"M4.N1": _o("Multiplication as repeated adding", "× is a shortcut for adding the same number.", "Number Sense",
    [_c("m4-n1","Groups of the same size","× = repeated add",
        "3 × 4 means 'add three 4s': 4+4+4 = 12.",
        "Picture 3 bags with 4 apples each. Count them all: 12. Same as adding 4 three times.",
        "3 × 4 = 4 + 4 + 4 = 12",
        "'3 × 4' is not the number 34.")],
    [_q("m4n1-q1","numeric",1,"5 × 3 = ?","15",["Add 5+5+5."],"5×3 = 15.",tol=0.001),
     _q("m4n1-q2","numeric",2,"4 bags × 3 apples each?","12",["4 threes."],"12 apples.",tol=0.001),
     _q("m4n1-q3","mc",2,"Same as 6 × 2?","0",["Two 6s."],"6 + 6 = 12.",choices=["6 + 6","6 + 2","62","6 - 2"])]),
"M4.N2": _o("Division as sharing", "Splitting a whole into equal groups.", "Number Sense",
    [_c("m4-n2","Sharing equally","× and ÷ undo each other",
        "12 ÷ 3 = 4 because 3 × 4 = 12.",
        "12 cookies shared among 3 friends → each gets 4. Division and multiplication are inverses.",
        "12 ÷ 3 = 4  (since 3 × 4 = 12)\n20 ÷ 5 = 4  (since 5 × 4 = 20)",
        "Some confuse ÷ with subtraction. 12 ÷ 3 shares evenly; 12 − 3 takes away 3 just once.")],
    [_q("m4n2-q1","numeric",1,"12 ÷ 3 = ?","4",["3 × ? = 12."],"12÷3 = 4.",tol=0.001),
     _q("m4n2-q2","numeric",2,"20 cookies ÷ 4 friends = ? each","5",["4 × ? = 20."],"5 each.",tol=0.001),
     _q("m4n2-q3","mc",2,"15 ÷ 3 = ?","2",["3 × ? = 15."],"5.",choices=["3","4","5","6"])]),
"M4.G1": _o("Perimeter of a rectangle", "Adding up all four sides.", "Geometry",
    [_c("m4-g1","Two sides twice","2·(L + W)",
        "Perimeter = 2 × (length + width). All four sides added up.",
        "A rectangle has two lengths and two widths. Add all four, or use the formula 2(L+W).",
        "L = 5, W = 3\nP = 2(5 + 3) = 2(8) = 16",
        "Some multiply L × W (that's area, not perimeter).")],
    [_q("m4g1-q1","numeric",1,"Rectangle L=4, W=2. Perimeter?","12",["2(4+2)."],"2·6 = 12.",tol=0.001),
     _q("m4g1-q2","numeric",2,"Rectangle L=10, W=5. Perimeter?","30",["2(10+5)."],"2·15 = 30.",tol=0.001),
     _q("m4g1-q3","mc",2,"Formula for perimeter of a rectangle?","1",["Two of each side."],"2(L+W).",choices=["L × W","2(L + W)","L + W","L² + W²"])]),
}},

"e-g5-math": {"label":"Math", "level":"elementary", "grade":5, "subject":"math", "outcomes":{
"M5.N1": _o("Adding fractions (same denominator)", "When bottoms match, add tops.", "Number Sense",
    [_c("m5-n1","Keep the bottom","add just the tops",
        "Same bottom → add tops, keep the bottom.",
        "Pizza in 8 slices. 3/8 + 2/8: eaten 5 slices out of 8 → 5/8. Not 5/16 — the pizza is still 8 slices.",
        "3/8 + 2/8 = 5/8",
        "Don't add the bottoms. The bottom is the SIZE of one piece — it doesn't change.")],
    [_q("m5n1-q1","mc",1,"1/5 + 2/5 = ?","0",["Add tops."],"3/5.",choices=["3/5","3/10","2/5","1/2"]),
     _q("m5n1-q2","mc",2,"3/8 + 4/8 = ?","2",["Tops: 3+4=7."],"7/8.",choices=["7/16","12/8","7/8","1/8"]),
     _q("m5n1-q3","mc",3,"2/6 + 3/6 = ?","0",["Tops: 2+3."],"5/6.",choices=["5/6","5/12","6/12","2/6"])]),
"M5.A1": _o("Growing patterns", "Continuing a pattern by finding the rule.", "Algebra",
    [_c("m5-a1","Look at the differences","same jump = linear",
        "Find how the pattern changes each step. Apply that change to keep going.",
        "For 3, 7, 11, 15, ...: differences are 4, 4, 4 — add 4 each time. Next is 19.",
        "3, 7, 11, 15, ?\nDiffs: 4, 4, 4\nNext: 15 + 4 = 19",
        "Some just repeat the last two numbers. Look at the RULE, not the numbers.")],
    [_q("m5a1-q1","numeric",1,"Next: 2, 5, 8, 11, ?","14",["Diff = 3."],"11 + 3 = 14.",tol=0.001),
     _q("m5a1-q2","numeric",2,"Next: 10, 15, 20, 25, ?","30",["Diff = 5."],"25 + 5 = 30.",tol=0.001),
     _q("m5a1-q3","mc",2,"Rule for 4, 8, 12, 16?","1",["Same jump?"],"Add 4.",choices=["Add 2","Add 4","Multiply by 2","Subtract 4"])]),
"M5.D1": _o("Finding the mean (average)", "Add all values, divide by how many.", "Data",
    [_c("m5-d1","Mean = sum ÷ count","two-step recipe",
        "Add every value, then divide by how many there are.",
        "Scores 8, 6, 10: sum is 24, count is 3, mean is 24 ÷ 3 = 8.",
        "Scores: 8, 6, 10\nSum: 24\nCount: 3\nMean: 24 ÷ 3 = 8",
        "Some divide by the biggest value instead of the count. Divide by how many values there are.")],
    [_q("m5d1-q1","numeric",1,"Mean of 4, 6, 8?","6",["Sum then divide."],"18 ÷ 3 = 6.",tol=0.001),
     _q("m5d1-q2","numeric",2,"Mean of 10, 20, 30, 40?","25",["Sum: 100."],"100 ÷ 4 = 25.",tol=0.001),
     _q("m5d1-q3","mc",2,"Mean of 5, 5, 5?","2",["All same."],"5.",choices=["3","4","5","15"])]),
}},

"e-g6-math": {"label":"Math", "level":"elementary", "grade":6, "subject":"math", "outcomes":{
"M6.N1": _o("Multiplying decimals by 10, 100, 1000", "Slide the decimal right.", "Number Sense",
    [_c("m6-n1","Move the point","one zero = one place",
        "×10 → 1 right. ×100 → 2 right. ×1000 → 3 right.",
        "Multiplying by 10 makes a number 10× bigger, which is the same as sliding the decimal point one place right.",
        "3.4 × 10   = 34\n3.4 × 100  = 340\n3.4 × 1000 = 3400",
        "Some slide LEFT. Multiplying grows the number, so RIGHT.")],
    [_q("m6n1-q1","numeric",1,"3.4 × 10 = ?","34",["One right."],"34.",tol=0.001),
     _q("m6n1-q2","numeric",2,"0.56 × 100 = ?","56",["Two right."],"56.",tol=0.001),
     _q("m6n1-q3","mc",2,"2.7 × 1000 = ?","2",["Three right."],"2700.",choices=["27","270","2700","27000"])]),
"M6.A1": _o("Order of operations", "BEDMAS — the fixed order.", "Algebra",
    [_c("m6-a1","BEDMAS","B·E·DM·AS",
        "Brackets, Exponents, Division/Multiplication (left→right), Addition/Subtraction (left→right).",
        "Do things in strict order. Division and multiplication are equal — go left to right. Same for + and −.",
        "3 + 4 × 2\n= 3 + 8\n= 11  (not 14)",
        "Don't just work left to right ignoring the order. Multiplication first!")],
    [_q("m6a1-q1","numeric",1,"3 + 4 × 2 = ?","11",["× first."],"3 + 8 = 11.",tol=0.001),
     _q("m6a1-q2","numeric",2,"(3 + 4) × 2 = ?","14",["Brackets first."],"7 × 2 = 14.",tol=0.001),
     _q("m6a1-q3","mc",3,"20 - 6 ÷ 2 = ?","1",["÷ first."],"20 - 3 = 17.",choices=["7","17","10","13"])]),
"M6.G1": _o("Angles in a straight line", "They add to 180°.", "Geometry",
    [_c("m6-g1","Straight = 180°","angles on a line sum to 180",
        "Two angles that make a straight line add to 180°.",
        "If one angle is 70°, the other is 180 − 70 = 110°.",
        "     ∠a   ∠b\n───────────────\n∠a + ∠b = 180°",
        "Some confuse straight (180°) with a right angle (90°). Straight line is HALF a circle.")],
    [_q("m6g1-q1","numeric",1,"One angle on a line is 40°. The other?","140",["180 - 40."],"140°.",tol=0.001),
     _q("m6g1-q2","numeric",2,"One angle is 95°. The other?","85",["180 - 95."],"85°.",tol=0.001),
     _q("m6g1-q3","mc",1,"Straight-line angles sum to…","2",["Half a circle."],"180°.",choices=["90°","120°","180°","360°"])]),
}},

"e-g7-math": {"label":"Math", "level":"elementary", "grade":7, "subject":"math", "outcomes":{
"M7.N1": _o("Percent of a number", "% means 'out of 100'. Multiply.", "Number Sense",
    [_c("m7-n1","% as a decimal","divide by 100",
        "25% = 25/100 = 0.25. Multiply to apply.",
        "25% of 80 = 0.25 × 80 = 20. Percent, decimal, and fraction are three ways to write the same idea.",
        "20% of 50\n= 0.20 × 50\n= 10",
        "'25% of 80' does not equal 25. The 25 is the percent — you still apply it.")],
    [_q("m7n1-q1","numeric",1,"10% of 50 = ?","5",["0.10 × 50."],"5.",tol=0.001),
     _q("m7n1-q2","numeric",2,"20% of 60 = ?","12",["0.20 × 60."],"12.",tol=0.001),
     _q("m7n1-q3","mc",3,"25% of 80 = ?","1",["A quarter."],"20.",choices=["25","20","40","10"])]),
"M7.A1": _o("Solving one-step equations", "Do the opposite to both sides.", "Algebra",
    [_c("m7-a1","Undo the operation","balance the equation",
        "To solve x + 5 = 12, subtract 5 from BOTH sides.",
        "The equal sign is a balance. Whatever you do to one side, do to the other. Undo the operation attached to the variable.",
        "x + 5 = 12\nx + 5 − 5 = 12 − 5\nx = 7",
        "Only do the operation to one side and the equation breaks. Always do to BOTH.")],
    [_q("m7a1-q1","numeric",1,"Solve x + 4 = 10","6",["Subtract 4 both sides."],"x = 6.",tol=0.001),
     _q("m7a1-q2","numeric",2,"Solve 3x = 21","7",["Divide by 3."],"x = 7.",tol=0.001),
     _q("m7a1-q3","numeric",2,"Solve x - 8 = 3","11",["Add 8."],"x = 11.",tol=0.001)]),
"M7.G1": _o("Area of a triangle", "½ × base × height.", "Geometry",
    [_c("m7-g1","Half of the box","triangle = half rectangle",
        "Area = ½ × base × height. The height is perpendicular to the base.",
        "A triangle fills half the rectangle you could draw around it. So its area is base × height, then halved.",
        "b = 6, h = 4\nA = ½ · 6 · 4 = 12",
        "The 'height' is NOT a slanted side — it's the perpendicular distance from base to the opposite vertex.")],
    [_q("m7g1-q1","numeric",1,"Triangle b=4, h=3. Area?","6",["½ · 4 · 3."],"6.",tol=0.001),
     _q("m7g1-q2","numeric",2,"Triangle b=10, h=8. Area?","40",["½ · 10 · 8."],"40.",tol=0.001),
     _q("m7g1-q3","mc",1,"Formula for triangle area?","2",["Half of…"],"½ · b · h.",choices=["b · h","b + h","½ · b · h","2 · b · h"])]),
}},

"e-g8-math": {"label":"Math", "level":"elementary", "grade":8, "subject":"math", "outcomes":{
"M8.N1": _o("Integers — adding and subtracting", "Positive and negative whole numbers.", "Number Sense",
    [_c("m8-n1","Signs matter","same sign add · different sign subtract",
        "Same signs → add sizes, keep the sign. Different signs → subtract smaller from bigger, keep the bigger's sign.",
        "For 3 + (−7): different signs, subtract 3 from 7 = 4, keep the negative → −4. For −3 + (−7): same sign, add 3+7=10, keep negative → −10.",
        "3 + (−7) = −4\n−3 + (−7) = −10\n5 − (−2) = 5 + 2 = 7",
        "'Subtract a negative' = add. −(−) = +.")],
    [_q("m8n1-q1","numeric",1,"3 + (−5) = ?","-2",["Different signs, subtract."],"−2.",tol=0.001),
     _q("m8n1-q2","numeric",2,"−7 − (−3) = ?","-4",["Two negatives make a plus."],"−7 + 3 = −4.",tol=0.001),
     _q("m8n1-q3","mc",2,"−4 + 9 = ?","2",["Different signs."],"5.",choices=["−13","−5","5","13"])]),
"M8.A1": _o("Linear patterns in tables", "Same jump = linear.", "Algebra",
    [_c("m8-a1","Constant difference","check ALL differences",
        "Linear = same difference between neighbours.",
        "For 3, 5, 7, 9: differences 2, 2, 2 → linear. For 1, 4, 9, 16: differences 3, 5, 7 → not linear (squared).",
        "3, 5, 7, 9 → diffs 2, 2, 2 → LINEAR\n1, 4, 9, 16 → diffs 3, 5, 7 → NOT",
        "Some check just the first two. Check them all.")],
    [_q("m8a1-q1","mc",2,"Which is linear?","0",["Same jump?"],"2,4,6,8 → jumps of 2.",choices=["2, 4, 6, 8","1, 4, 9, 16","2, 3, 5, 8","5, 10, 20, 40"]),
     _q("m8a1-q2","mc",3,"Is 4, 7, 10, 13 linear?","0",["Diffs 3,3,3."],"Yes.",choices=["Yes (jumps of 3)","No","Only first two","Just getting bigger"]),
     _q("m8a1-q3","mc",3,"Which is NOT linear?","2",["Check diffs."],"1,3,6,10 → 2,3,4.",choices=["3,6,9,12","10,20,30,40","1,3,6,10","-1,0,1,2"])]),
"M8.G1": _o("Pythagorean theorem", "In a right triangle, a² + b² = c².", "Geometry",
    [_c("m8-g1","a² + b² = c²","for right triangles only",
        "The square of the hypotenuse (c) equals the sum of the squares of the two legs (a, b).",
        "Only works in right triangles. c is always the LONGEST side (opposite the right angle). To find c: square both legs, add, take the square root.",
        "Legs 3 and 4:\n3² + 4² = 9 + 16 = 25\nc = √25 = 5",
        "The hypotenuse c is ONLY the side opposite the right angle. Don't mix up which side is c.")],
    [_q("m8g1-q1","numeric",2,"Legs 3 and 4. Hypotenuse?","5",["3² + 4²."],"√(9+16) = 5.",tol=0.01),
     _q("m8g1-q2","numeric",3,"Legs 5 and 12. Hypotenuse?","13",["5² + 12² = 169."],"√169 = 13.",tol=0.01),
     _q("m8g1-q3","mc",2,"Pythagorean theorem works for…","0",["Only certain triangles."],"Right triangles only.",choices=["Right triangles only","Any triangle","Equilateral only","Squares"])]),
}},

# ═══════════════════ ELEMENTARY · Grades 3-8 · English ═══════════════════
"e-g3-english": {"label":"English", "level":"elementary", "grade":3, "subject":"english", "outcomes":{
"E3.R1": _o("Main idea of a paragraph", "The one thing every sentence supports.", "Reading",
    [_c("e3-r1","One big idea","details support it",
        "The main idea is the one message every other sentence backs up.",
        "Ask: 'What is the writer trying to say overall?' The answer is the main idea. Everything else is a detail.",
        "PARA: 'Dogs make great pets. They protect the house. They love to play. They comfort you.'\nMAIN: Dogs make great pets.",
        "The main idea isn't always the first sentence. It's the one all others support.")],
    [_q("e3r1-q1","mc",2,"'Cats love to nap. They sleep 16 hours a day. They nap in the sun.' Main idea?","0",["What are all sentences about?"],"Cats love to nap.",choices=["Cats love to nap","Cats like sun","Cats are lazy","Cats have 4 legs"]),
     _q("e3r1-q2","mc",2,"'Recycling saves trees. It reduces trash. It uses less energy.' Main idea?","1",["What's the overall message?"],"Recycling helps.",choices=["Trees matter","Recycling helps","Trash smells","Energy costs"]),
     _q("e3r1-q3","mc",1,"A main idea is…","2",["Big vs small."],"The big message.",choices=["Any detail","The last sentence","The big message","The title"])]),
}},

"e-g5-english": {"label":"English", "level":"elementary", "grade":5, "subject":"english", "outcomes":{
"E5.W1": _o("Writing a topic sentence", "Open a paragraph with a clear statement.", "Writing",
    [_c("e5-w1","Preview the paragraph","promise, then deliver",
        "A topic sentence tells the reader what the paragraph is about in one clear sentence.",
        "It's a promise. It usually opens the paragraph. Specific enough to preview; not so specific it gives away everything.",
        "GOOD: 'Playing sports teaches life skills.'\nWEAK: 'Sports are cool.' (vague)\nWEAK: 'I scored 3 goals.' (a detail, not a topic)",
        "A topic sentence is not a title or a question. Make it a statement.")],
    [_q("e5w1-q1","mc",2,"Best topic sentence for a paragraph about reading?","1",["Preview, don't detail."],"'Reading opens worlds and builds vocab.'",choices=["I love books.","Reading opens worlds and builds vocab.","I finished a book.","Books are on shelves."]),
     _q("e5w1-q2","mc",2,"Too specific to be a topic sentence?","2",["A single detail."],"'Chapter 3 was 42 pages.'",choices=["Trains changed history.","Winter sports need practice.","Chapter 3 was 42 pages.","Volcanoes shape land."]),
     _q("e5w1-q3","mc",1,"A topic sentence should be…","0",["Statement or question?"],"A statement at the start.",choices=["A statement at the start","A question","The last sentence","Missing"])]),
}},

"e-g8-english": {"label":"English", "level":"elementary", "grade":8, "subject":"english", "outcomes":{
"E8.W1": _o("Persuasive paragraph structure", "Point · Reason · Evidence · Link.", "Writing",
    [_c("e8-w1","PREL","the four parts of an argument",
        "Every persuasive paragraph: state your Point, give your Reason, provide Evidence, then Link back.",
        "Point = the claim. Reason = why. Evidence = proof (fact, quote, study). Link = why the reader should care.",
        "POINT: School should start later.\nREASON: Teens need more sleep.\nEVIDENCE: Studies show teens need 8-10 hrs; most get 6-7.\nLINK: Later start = better learning.",
        "Skipping evidence leaves you with just an opinion. Skipping the link leaves the reader wondering 'so what?'.")],
    [_q("e8w1-q1","mc",2,"What does 'P' in PREL mean?","0",["First step."],"Point (your claim).",choices=["Point","Practice","Preview","Poem"]),
     _q("e8w1-q2","mc",3,"Which is EVIDENCE?","2",["Look for fact or study."],"'A 2023 study found phones lower scores by 15%.'",choices=["'Phones are bad.'","'I think phones distract.'","'A 2023 study found phones lower scores by 15%.'","'Everyone knows this.'"]),
     _q("e8w1-q3","mc",2,"Why write a LINK?","1",["What does it do?"],"Remind the reader why the point matters.",choices=["To make paragraph longer","Remind reader why it matters","Introduce new topic","Add opinion"])]),
}},

# ═══════════════════ ELEMENTARY · French ═══════════════════
"e-g4-french": {"label":"French", "level":"elementary", "grade":4, "subject":"french", "outcomes":{
"F4.O1": _o("Basic greetings", "Bonjour, salut, ça va, merci.", "Oral Communication",
    [_c("f4-o1","Greeting essentials","the everyday four",
        "Bonjour = hello (formal). Salut = hi/bye (casual). Ça va? = how's it going. Merci = thanks.",
        "'Bonjour' works any time. 'Salut' is only with people you know. 'Ça va?' rising tone = question, flat tone = statement ('doing fine'). 'Merci' — always polite.",
        "— Bonjour!\n— Salut! Ça va?\n— Ça va bien, merci.",
        "Ça va with rising voice = question. Ça va with flat voice = answer.")],
    [_q("f4o1-q1","mc",1,"'Bonjour' means…","0",["Common greeting."],"Hello.",choices=["Hello","Goodbye","Please","Sorry"]),
     _q("f4o1-q2","mc",2,"Answer to 'Ça va?' if you're OK?","1",["Positive."],"'Ça va bien.'",choices=["Ça va mal.","Ça va bien.","Non merci.","Bonjour."]),
     _q("f4o1-q3","mc",1,"'Merci' means…","2",["Politeness."],"Thank you.",choices=["Please","Sorry","Thank you","Excuse me"])]),
}},

"e-g6-french": {"label":"French", "level":"elementary", "grade":6, "subject":"french", "outcomes":{
"F6.G1": _o("Regular -er verbs (present tense)", "Drop the -er, add the ending.", "Grammar",
    [_c("f6-g1","-er endings","je/tu/il/elle",
        "Drop -er, add -e (je), -es (tu), -e (il/elle).",
        "Parler → je parle, tu parles, il parle. Same for manger, danser, chanter. Endings sound alike; spelling differs.",
        "parler → je parle · tu parles · il parle\ndanser → je danse · tu danses · il danse",
        "Silent 's' on tu forms is still written. 'Tu parles' not 'tu parle'.")],
    [_q("f6g1-q1","mc",2,"'Je _____ français.' (parler)","1",["Je ends in -e."],"parle.",choices=["parles","parle","parlent","parlez"]),
     _q("f6g1-q2","mc",2,"'Tu _____ bien.' (danser)","0",["Tu ends in -es."],"danses.",choices=["danses","danse","dansent","dansez"]),
     _q("f6g1-q3","mc",3,"'Elle _____ une chanson.' (chanter)","0",["Il/elle ends in -e."],"chante.",choices=["chante","chantes","chantent","chanter"])]),
}},

"e-g8-french": {"label":"French", "level":"elementary", "grade":8, "subject":"french", "outcomes":{
"F8.R1": _o("Reading comprehension strategies", "Scan for keywords, don't translate everything.", "Reading",
    [_c("f8-r1","Scan and skim","find the answer word",
        "Find the key word from the question in the passage. The answer will be right beside it.",
        "Don't translate every unknown word. If the question asks 'Où habite Marie?', scan for 'Marie' + 'habite' in the passage.",
        "PASSAGE: 'Marie habite à Toronto.'\nQ: Où habite Marie?\nSCAN: Marie + habite → 'à Toronto'\nANSWER: À Toronto.",
        "Trying to translate every word burns time. Focus on the words that answer the question.")],
    [_q("f8r1-q1","mc",2,"Passage: 'Paul mange une pomme.' Q: Que mange Paul?","0",["Que = what."],"Une pomme.",choices=["Une pomme","Paul","Le matin","Une banane"]),
     _q("f8r1-q2","mc",2,"Passage: 'J'ai deux chats.' Q: Combien de chats?","1",["Combien = how many."],"Deux.",choices=["Un","Deux","Trois","Zéro"]),
     _q("f8r1-q3","mc",3,"Passage: 'Le concert commence à sept heures.' Q: À quelle heure?","2",["Heure = hour."],"Sept heures.",choices=["À la maison","Le concert","Sept heures","Aujourd'hui"])]),
}},

# ═══════════════════ ELEMENTARY · Science ═══════════════════
"e-g4-science": {"label":"Science", "level":"elementary", "grade":4, "subject":"science", "outcomes":{
"S4.L1": _o("Habitats and needs", "Food, water, shelter, space.", "Life Systems",
    [_c("s4-l1","Habitat = home + needs","the four essentials",
        "A habitat supplies food, water, shelter, and space.",
        "Polar bears need Arctic ice — seals to eat, snow for dens, cold weather. In a desert, none of that exists.",
        "Polar bear: Arctic → seals, ice dens, cold\nCactus: Desert → sun, sand, little water",
        "'Habitat' isn't just 'where an animal lives' — it's what supplies its needs.")],
    [_q("s4l1-q1","mc",1,"A habitat provides…","3",["Basic needs."],"All of these.",choices=["Only food","Only water","Only shelter","All of these"]),
     _q("s4l1-q2","mc",2,"Fish removed from water will…","1",["Fish breathe in water."],"Die.",choices=["Grow","Die","Change colour","Walk"]),
     _q("s4l1-q3","mc",2,"NOT a basic need of an animal?","3",["Want vs need."],"Toys.",choices=["Food","Water","Shelter","Toys"])]),
}},

"e-g6-science": {"label":"Science", "level":"elementary", "grade":6, "subject":"science", "outcomes":{
"S6.E1": _o("Simple electric circuits", "Battery → wire → bulb → back.", "Matter & Energy",
    [_c("s6-e1","Complete loop","break it, no light",
        "A bulb lights only when the circuit is a closed loop from battery to bulb and back.",
        "Break the loop anywhere and current stops. A switch is a controlled break.",
        "Battery + → wire → bulb → wire → Battery -",
        "The bulb doesn't 'use up' the electricity. It converts some of the flow to light and heat.")],
    [_q("s6e1-q1","mc",1,"For a bulb to light, circuit must be…","0",["Loop unbroken."],"Closed.",choices=["Closed","Open","Cut","Empty"]),
     _q("s6e1-q2","mc",2,"Opening a switch does what?","1",["Breaks loop."],"Stops current.",choices=["Speeds up","Stops","Reverses","Doubles"]),
     _q("s6e1-q3","mc",2,"Power source in a basic circuit?","2",["Where does energy start?"],"Battery.",choices=["Wire","Bulb","Battery","Switch"])]),
}},

"e-g8-science": {"label":"Science", "level":"elementary", "grade":8, "subject":"science", "outcomes":{
"S8.C1": _o("Cell structure", "The main organelles and what they do.", "Life Systems",
    [_c("s8-c1","Cell parts","membrane · nucleus · mitochondria",
        "Every cell has a membrane, cytoplasm, nucleus (except bacteria). Mitochondria make energy.",
        "Nucleus holds DNA. Mitochondria = powerhouse. Plant cells add a cell wall and chloroplasts (photosynthesis); animal cells have neither.",
        "SHARED: membrane, cytoplasm, nucleus, mitochondria\nPLANT ONLY: cell wall, chloroplasts, big vacuole",
        "'Cell wall' and 'cell membrane' are different. All cells have a membrane; only plant/fungal/bacterial cells have a wall.")],
    [_q("s8c1-q1","mc",1,"Which organelle contains DNA?","1",["Command centre."],"Nucleus.",choices=["Mitochondrion","Nucleus","Ribosome","Vacuole"]),
     _q("s8c1-q2","mc",2,"Found in plant cells but NOT animal?","0",["Green energy factory."],"Chloroplast.",choices=["Chloroplast","Nucleus","Mitochondrion","Membrane"]),
     _q("s8c1-q3","mc",2,"Mitochondria are the cell's ___.","2",["Energy."],"Powerhouse.",choices=["Brain","Skeleton","Powerhouse","Wall"])]),
}},

# ═══════════════════ SECONDARY · Grades 9-12 ═══════════════════
"s-g9-math": {"label":"Math (MPM1D)", "level":"secondary", "grade":9, "subject":"math", "outcomes":{
"M9.A1": _o("Solving linear equations", "Isolate the variable.", "Algebra",
    [_c("m9-a1","Balance the equation","inverse operations",
        "Do the SAME operation to both sides. Undo whatever is done to x.",
        "Work backwards from the outside in. Undo +/− first, then ×/÷.",
        "3x + 5 = 20\n3x = 15   (subtract 5)\nx = 5     (divide by 3)",
        "Some do just one side of the equation. Both sides must always match.")],
    [_q("m9a1-q1","numeric",2,"Solve: 2x + 3 = 11","4",["Undo +3 first."],"2x = 8; x = 4.",tol=0.001),
     _q("m9a1-q2","numeric",3,"Solve: 5x - 7 = 18","5",["Undo -7."],"5x = 25; x = 5.",tol=0.001),
     _q("m9a1-q3","numeric",3,"Solve: 3(x + 2) = 15","3",["Distribute or divide first."],"x + 2 = 5; x = 3.",tol=0.001)]),
"M9.G1": _o("Slope from two points", "m = (y₂-y₁)/(x₂-x₁).", "Analytic Geometry",
    [_c("m9-g1","Rise over run","a rate of change",
        "Slope = change in y over change in x.",
        "Pick any two points. Subtract y-values on top, x-values on bottom. The straight line means slope is the same everywhere.",
        "(2, 3) and (5, 9):\nm = (9-3)/(5-2) = 6/3 = 2",
        "Slope is a RATE; intercept is a LOCATION. Don't confuse them.")],
    [_q("m9g1-q1","numeric",2,"Slope through (1,4) and (5,12)?","2",["(y2-y1)/(x2-x1)."],"8/4 = 2.",tol=0.001),
     _q("m9g1-q2","mc",1,"Slope of y = -3x + 7?","2",["m from y=mx+b."],"-3.",choices=["7","3","-3","-7"]),
     _q("m9g1-q3","numeric",3,"Slope -2 through (2,5). y-intercept?","9",["5 = -2(2) + b."],"b = 9.",tol=0.001)]),
"M9.D1": _o("Sampling — bias in surveys", "How you pick affects what you learn.", "Data",
    [_c("m9-d1","Random samples","representative or biased",
        "A good sample represents the whole population. Bad sampling (only friends, only online) skews the result.",
        "If you survey only sports fans about school policy, you don't know what non-fans think. Random sampling reduces bias.",
        "Bias example: 'Do you play video games?' asked only at a gaming store.\nBetter: random students from every grade.",
        "'Big sample size' does NOT fix bias. A biased-but-huge sample is still biased.")],
    [_q("m9d1-q1","mc",2,"Best sampling method for a school survey?","1",["Which is random?"],"Random students from every grade.",choices=["Ask friends","Random students from every grade","Only Grade 12s","Volunteers only"]),
     _q("m9d1-q2","mc",2,"Surveying only sports fans about gym class introduces…","1",["What kind of error?"],"Bias.",choices=["Randomness","Bias","Precision","Accuracy"]),
     _q("m9d1-q3","mc",3,"A large but biased sample is…","2",["Size fixes bias?"],"Still biased.",choices=["Accurate","Random","Still biased","Unbiased"])]),
}},

"s-g10-math": {"label":"Math (MPM2D)", "level":"secondary", "grade":10, "subject":"math", "outcomes":{
"M10.A1": _o("Factoring x² + bx + c", "Find p and q where p·q = c, p+q = b.", "Algebra",
    [_c("m10-a1","Sum-product","two conditions, both must hold",
        "x² + bx + c = (x+p)(x+q) where p·q=c AND p+q=b.",
        "List factor pairs of c. Find the one that also sums to b. Sign matters: c positive → same signs; c negative → opposite.",
        "x² + 7x + 12\nPairs of 12: (1,12), (2,6), (3,4)\n3 + 4 = 7 ✓\n= (x+3)(x+4)",
        "Some pick a pair with the right product but forget the sum. Both must hold.")],
    [_q("m10a1-q1","mc",2,"Factor x² + 7x + 12","1",["p·q=12, p+q=7."],"(x+3)(x+4).",choices=["(x+2)(x+6)","(x+3)(x+4)","(x+1)(x+12)","(x-3)(x-4)"]),
     _q("m10a1-q2","mc",3,"Factor x² − 5x + 6","0",["Both negative."],"(x-2)(x-3).",choices=["(x-2)(x-3)","(x+2)(x+3)","(x-1)(x-6)","(x+1)(x-6)"]),
     _q("m10a1-q3","mc",3,"Factor x² + x − 12","1",["Opposite signs."],"(x+4)(x-3).",choices=["(x+3)(x-4)","(x+4)(x-3)","(x-2)(x+6)","(x+2)(x-6)"])]),
"M10.T1": _o("Trigonometry — sine ratio", "sin(θ) = opposite / hypotenuse.", "Trigonometry",
    [_c("m10-t1","SOH","sine = opp / hyp",
        "In a right triangle, sin(θ) = length of opposite side ÷ length of hypotenuse.",
        "Label the sides relative to the angle: opposite (across from), adjacent (next to, not hypotenuse), hypotenuse (across from right angle, longest).",
        "θ = 30°, hyp = 10.\nOpp = 10 · sin 30° = 10 · 0.5 = 5",
        "Opposite and adjacent switch depending on WHICH angle you're using. Always look at the angle first.")],
    [_q("m10t1-q1","numeric",2,"θ = 30°, hyp = 10. Opposite side?","5",["sin 30 = 0.5."],"10 · 0.5 = 5.",tol=0.05),
     _q("m10t1-q2","mc",1,"SOH stands for…","1",["Which ratio?"],"Sine = Opposite / Hypotenuse.",choices=["Sine = Opp/Adj","Sine = Opp/Hyp","Sine = Adj/Hyp","Sine = Hyp/Opp"]),
     _q("m10t1-q3","numeric",3,"θ = 45°, hyp = 8. Opposite (round to 2 dp)?","5.66",["sin 45 ≈ 0.707."],"8 · 0.707 ≈ 5.66.",tol=0.05)]),
}},

"s-g11-math": {"label":"Math (MCR3U)", "level":"secondary", "grade":11, "subject":"math", "outcomes":{
"M11.F1": _o("Domain and range", "Allowed inputs and possible outputs.", "Functions",
    [_c("m11-f1","Restrict inputs, observe outputs","denominators, radicands, logs",
        "Domain = valid inputs. Range = outputs those inputs can produce.",
        "No zero denominators. No negative radicands (real sqrt). No non-positive logs. Then see which y-values the surviving inputs produce.",
        "f(x) = 1/(x-2)\nDomain: x ≠ 2\nRange:  y ≠ 0",
        "Domain = INPUTS (x). Range = OUTPUTS (y). Don't confuse.")],
    [_q("m11f1-q1","mc",2,"Domain of f(x) = 1/(x-3)?","1",["Denominator ≠ 0."],"All reals except 3.",choices=["all reals","all reals except 3","x > 3","x = 3"]),
     _q("m11f1-q2","mc",3,"Domain of √(x-4)?","2",["Radicand ≥ 0."],"x ≥ 4.",choices=["all reals","x > 4","x ≥ 4","x ≤ 4"]),
     _q("m11f1-q3","mc",3,"Range of x² + 1?","1",["x² ≥ 0."],"y ≥ 1.",choices=["all reals","y ≥ 1","y ≥ 0","y > 1"])]),
"M11.S1": _o("Arithmetic sequences", "Same jump every term: t_n = a + (n-1)d.", "Sequences",
    [_c("m11-s1","Arithmetic formula","first + (n-1)·common difference",
        "For an arithmetic sequence, term n = first term + (n − 1) × common difference.",
        "Given a and d, plug in n. Common difference d = t₂ − t₁ = t₃ − t₂ = …",
        "3, 7, 11, 15, ...\na=3, d=4\nt_n = 3 + (n-1)·4\nt_10 = 3 + 9·4 = 39",
        "Some use n instead of (n-1). The FIRST term has n=1, so the multiplier is 0.")],
    [_q("m11s1-q1","numeric",2,"Sequence 5, 8, 11, 14, ... Find t_10.","32",["a=5, d=3."],"5 + 9·3 = 32.",tol=0.01),
     _q("m11s1-q2","numeric",2,"Sequence 2, 7, 12, ... Find t_20.","97",["a=2, d=5."],"2 + 19·5 = 97.",tol=0.01),
     _q("m11s1-q3","mc",2,"Common difference of 4, 10, 16, 22?","2",["Any two neighbours."],"6.",choices=["2","4","6","10"])]),
}},

"s-g12-math": {"label":"Math (MCV4U)", "level":"secondary", "grade":12, "subject":"math", "outcomes":{
"M12.C1": _o("Intro to limits", "Behaviour of f(x) as x approaches a.", "Calculus",
    [_c("m12-c1","Approaching, not arriving","limit exists even if f(a) doesn't",
        "lim f(x) as x→a describes what f is heading toward — not necessarily f(a).",
        "If direct substitution gives 0/0, factor and cancel. The limit is what x+something heads to at x=a.",
        "lim (x²-4)/(x-2) as x→2\n= lim (x-2)(x+2)/(x-2)\n= lim (x+2), x≠2\n= 4",
        "0/0 is INDETERMINATE, not undefined. Simplify first.")],
    [_q("m12c1-q1","numeric",1,"lim (x+2) as x→3","5",["Substitute."],"5.",tol=0.001),
     _q("m12c1-q2","numeric",3,"lim (x²-9)/(x-3) as x→3","6",["Factor: (x-3)(x+3)."],"6.",tol=0.001),
     _q("m12c1-q3","mc",3,"When direct substitution gives 0/0…","2",["Meaning?"],"Simplify — the limit may still exist.",choices=["Limit is 0","Limit is ∞","Simplify — limit may still exist","Limit is undefined"])]),
"M12.V1": _o("Vectors — magnitude", "|v| = √(x² + y²).", "Vectors",
    [_c("m12-v1","Length of a vector","Pythagorean theorem in disguise",
        "For 2D vector v = (x, y), |v| = √(x² + y²).",
        "The magnitude is the length. Treat the components as legs of a right triangle; the vector itself is the hypotenuse.",
        "v = (3, 4)\n|v| = √(9 + 16) = √25 = 5",
        "Magnitude is always POSITIVE. It's a length.")],
    [_q("m12v1-q1","numeric",2,"|(3, 4)|?","5",["√(9+16)."],"5.",tol=0.01),
     _q("m12v1-q2","numeric",3,"|(5, 12)|?","13",["√(25+144)."],"13.",tol=0.01),
     _q("m12v1-q3","numeric",2,"|(0, 7)|?","7",["Just the length."],"7.",tol=0.01)]),
}},

"s-g9-english": {"label":"English (ENG1D)", "level":"secondary", "grade":9, "subject":"english", "outcomes":{
"E9.W1": _o("Writing a thesis statement", "One arguable, specific sentence.", "Writing",
    [_c("e9-w1","Arguable + specific","not a topic, not a fact",
        "A thesis states an arguable position in one clear sentence.",
        "It's not the topic ('Social media'), a fact ('Instagram launched 2010'), or too broad ('Media is important'). It's a claim.",
        "TOPIC: Social media\nWEAK: Social media affects teens\nSTRONG: Instagram's algorithm increases teen anxiety by rewarding comparison.",
        "Not a summary of your essay ('This essay will discuss X') — a defended claim.")],
    [_q("e9w1-q1","mc",2,"Strongest thesis?","2",["Arguable + specific."],"'Streaming hurt indie musicians by shifting revenue to top-1%.'",choices=["'This is about music.'","'Music matters.'","'Streaming hurt indie musicians by shifting revenue to top-1%.'","'Spotify launched 2008.'"]),
     _q("e9w1-q2","mc",2,"NOT a good thesis?","0",["Fact vs. claim."],"'Shakespeare wrote 37 plays.' (fact)",choices=["'Shakespeare wrote 37 plays.'","'His tragedies critique power.'","'Hamlet's inaction is a choice.'","'Iago drives Othello's fate.'"]),
     _q("e9w1-q3","mc",1,"A thesis is…","0",["What makes it work?"],"Arguable and specific.",choices=["Arguable and specific","A question","A list of topics","A common fact"])]),
}},

"s-g11-english": {"label":"English (ENG3U)", "level":"secondary", "grade":11, "subject":"english", "outcomes":{
"E11.L1": _o("Literary devices", "Metaphor · simile · personification · symbol.", "Literature",
    [_c("e11-l1","Four common devices","spot the mechanism",
        "Metaphor = X is Y. Simile = X is LIKE Y. Personification = giving human qualities to non-human things. Symbol = concrete stands for abstract.",
        "'Her voice is music' (metaphor). 'Her voice is like music' (simile). 'The wind whispered' (personification). Dove = peace (symbol).",
        "Metaphor:        Life is a highway.\nSimile:          Life is LIKE a highway.\nPersonification: The clock stared.\nSymbol:          Rose = love.",
        "'Like' or 'as' between two things = simile, not metaphor.")],
    [_q("e11l1-q1","mc",2,"'The sun is a golden coin' is a…","0",["X is Y."],"Metaphor.",choices=["Metaphor","Simile","Personification","Symbol"]),
     _q("e11l1-q2","mc",2,"'Her smile was like sunshine' is a…","1",["'Like' present."],"Simile.",choices=["Metaphor","Simile","Personification","Symbol"]),
     _q("e11l1-q3","mc",3,"'The wind whispered' uses…","2",["Wind ≠ speaker."],"Personification.",choices=["Metaphor","Simile","Personification","Symbol"])]),
}},

"s-g10-science": {"label":"Science (SNC2D)", "level":"secondary", "grade":10, "subject":"science", "outcomes":{
"S10.C1": _o("Stoichiometry — mole ratios", "Coefficients ARE the ratio.", "Chemistry",
    [_c("s10-c1","Read the balanced equation","coefficients not atoms",
        "The coefficients in a balanced equation give the mole ratio between species.",
        "For 2H₂ + O₂ → 2H₂O, ratio is 2:1:2. If 4 mol H₂ reacts, needs 2 mol O₂, makes 4 mol H₂O.",
        "2H₂ + O₂ → 2H₂O\n6 mol H₂:\n  O₂: 6·(1/2) = 3 mol\n  H₂O: 6·(2/2) = 6 mol",
        "Use coefficients from BALANCED equation, not the raw atom count.")],
    [_q("s10c1-q1","numeric",2,"2H₂ + O₂ → 2H₂O. 4 mol H₂ needs? mol O₂","2",["Ratio 2:1."],"2 mol O₂.",tol=0.01),
     _q("s10c1-q2","numeric",2,"N₂ + 3H₂ → 2NH₃. 6 mol H₂ → ? mol NH₃","4",["Ratio 3:2."],"6·(2/3) = 4 mol.",tol=0.01),
     _q("s10c1-q3","mc",3,"2A + B → 3C. 3 mol B → ? mol C","2",["Ratio 1:3."],"9 mol C.",choices=["3","6","9","1"])]),
"S10.P1": _o("Reflection and refraction", "Light bounces or bends.", "Physics",
    [_c("s10-p1","Two behaviours of light","bouncing vs bending",
        "Reflection: light bounces off a surface at the same angle. Refraction: light bends when it enters a different medium.",
        "Reflection follows angle-in = angle-out. Refraction happens because light changes speed between media (air, water, glass).",
        "Reflection: mirror\nAngle in = angle out\n\nRefraction: straw in water looks bent — light bends at the surface.",
        "Refraction only happens at the BOUNDARY between two media. Inside one medium, light goes straight.")],
    [_q("s10p1-q1","mc",1,"Light bounces off a mirror. This is…","0",["Bounce vs bend."],"Reflection.",choices=["Reflection","Refraction","Absorption","Diffraction"]),
     _q("s10p1-q2","mc",2,"Why does a straw look bent in water?","1",["Bend at surface."],"Refraction.",choices=["Reflection","Refraction","Colour change","Optical illusion"]),
     _q("s10p1-q3","mc",2,"In reflection, angle in equals…","0",["Symmetry."],"Angle out.",choices=["Angle out","Twice angle out","Half angle out","Zero"])]),
}},

# ═══════════════════ UNIVERSITY · UWaterloo 1A ═══════════════════
"u-math106": {"label":"MATH 106 · Linear Algebra 1", "level":"uni", "grade":13, "subject":"math", "outcomes":{
"U106.V1": _o("Vector operations", "Add, subtract, scalar multiply.", "Vectors",
    [_c("u106-v1","Component-wise","(x,y) + (a,b) = (x+a, y+b)",
        "Vector addition, subtraction, and scalar multiplication all work component by component.",
        "u + v adds components. cu scales each component by c. u - v = u + (-v). Order matters for subtraction but not addition.",
        "u = (2, 3), v = (1, 4)\nu + v = (3, 7)\n2u = (4, 6)\nu - v = (1, -1)",
        "Vectors need the same dimension to add. (2,3) + (1,4,5) is undefined.")],
    [_q("u106v1-q1","mc",2,"(2,3) + (4,1) = ?","0",["Add each component."],"(6, 4).",choices=["(6, 4)","(8, 3)","(2, 4)","(6, 3)"]),
     _q("u106v1-q2","mc",2,"3 · (2, -1) = ?","1",["Scale each."],"(6, -3).",choices=["(6, -1)","(6, -3)","(5, 2)","(3, -4)"]),
     _q("u106v1-q3","mc",3,"(5, 2) - (1, 6) = ?","1",["Component subtract."],"(4, -4).",choices=["(4, 8)","(4, -4)","(-4, 4)","(6, 8)"])]),
"U106.M1": _o("Matrix multiplication", "Row · column, sum.", "Matrices",
    [_c("u106-m1","AB entry (i,j)","row i of A · column j of B",
        "Entry (i,j) of AB = sum of a_{ik}·b_{kj} over k. AB defined only when cols(A) = rows(B).",
        "For A (m×n) and B (n×p), AB is (m×p). Compute each entry as the dot product of A's i-th row and B's j-th column.",
        "A = [[1,2],[3,4]], B = [[5,6],[7,8]]\n(AB)₁₁ = 1·5 + 2·7 = 19\n(AB)₁₂ = 1·6 + 2·8 = 22\nAB = [[19,22],[43,50]]",
        "AB ≠ BA in general. Matrix multiplication is NOT commutative.")],
    [_q("u106m1-q1","numeric",2,"A=[[1,2],[3,4]], B=[[5,6],[7,8]]. (AB)₁₁ = ?","19",["Row 1 · col 1."],"1·5 + 2·7 = 19.",tol=0.01),
     _q("u106m1-q2","numeric",3,"Same. (AB)₂₂ = ?","50",["Row 2 · col 2."],"3·6 + 4·8 = 50.",tol=0.01),
     _q("u106m1-q3","mc",2,"Matrix multiplication is…","1",["Order matters?"],"Associative, not commutative.",choices=["Commutative","Associative, not commutative","Neither","Both"])]),
}},

"u-math127": {"label":"MATH 127 · Calculus 1 (Science)", "level":"uni", "grade":13, "subject":"math", "outcomes":{
"U127.L1": _o("Evaluating limits algebraically", "Handle indeterminate 0/0.", "Limits",
    [_c("u127-l1","0/0 signals: simplify","factor, cancel, rationalize",
        "When substitution gives 0/0, the limit may still exist — simplify first.",
        "For rational functions, factor and cancel the (x-a) term. For radicals, multiply by the conjugate. The simplified expression equals the original everywhere except x=a — and that's where the limit lives.",
        "lim (x²-4)/(x-2) as x→2\n= lim (x-2)(x+2)/(x-2)\n= lim (x+2), x≠2\n= 4",
        "0/0 is indeterminate, not undefined. Do more algebra.")],
    [_q("u127l1-q1","numeric",1,"lim (x+2) as x→3","5",["Substitute."],"5.",tol=0.001),
     _q("u127l1-q2","numeric",3,"lim (x²-4)/(x-2) as x→2","4",["Factor top, cancel."],"4.",tol=0.001),
     _q("u127l1-q3","mc",3,"Which is TRUE about lim f(x) as x→a?","2",["Behaviour vs value."],"Describes behaviour near a, not at a.",choices=["Always = f(a)","Requires f defined at a","Describes behaviour near a, not at a","Equals ∞ at asymptotes"])]),
"U127.D1": _o("Chain rule", "d/dx f(g(x)) = f'(g(x))·g'(x).", "Derivatives",
    [_c("u127-d1","Outer × inner","differentiate outside, keep inside, times inside's derivative",
        "For y = f(g(x)), dy/dx = f'(g(x)) · g'(x).",
        "Identify outer and inner. Differentiate outer (leaving inner alone). Multiply by derivative of inner.",
        "d/dx sin(x²) = cos(x²)·2x\nd/dx (3x+1)⁵ = 5(3x+1)⁴·3",
        "Forgetting to multiply by g'(x) is the #1 error.")],
    [_q("u127d1-q1","mc",2,"d/dx sin(x²) = ?","0",["Chain rule."],"cos(x²)·2x.",choices=["cos(x²)·2x","2 sin x cos x","cos(x)·2x","2x sin(2x)"]),
     _q("u127d1-q2","mc",3,"d/dx (2x+3)⁴ = ?","1",["Power × chain."],"4(2x+3)³·2 = 8(2x+3)³.",choices=["4(2x+3)³","8(2x+3)³","(2x+3)³","2(2x+3)³"]),
     _q("u127d1-q3","mc",3,"d/dx e^(3x) = ?","2",["Chain: e^u · u'."],"3e^(3x).",choices=["e^(3x)","e^(3x)·x","3e^(3x)","e^3"])]),
}},

"u-math128": {"label":"MATH 128 · Calculus 2 (Science)", "level":"uni", "grade":14, "subject":"math", "outcomes":{
"U128.I1": _o("Basic integration", "Power rule for antiderivatives.", "Integration",
    [_c("u128-i1","Reverse the power rule","∫xⁿ dx = xⁿ⁺¹/(n+1) + C",
        "Antiderivative of xⁿ is xⁿ⁺¹/(n+1), for n ≠ -1. Always add + C.",
        "Bump the power up by 1, divide by the new power. The + C represents any constant, since d/dx of a constant is 0.",
        "∫ x² dx = x³/3 + C\n∫ x⁵ dx = x⁶/6 + C",
        "Forgetting + C loses generality. Also, ∫ 1/x dx is NOT x⁰/0 — it's ln|x| + C.")],
    [_q("u128i1-q1","mc",1,"∫ x² dx = ?","0",["Power rule."],"x³/3 + C.",choices=["x³/3 + C","x³ + C","2x + C","x²/3"]),
     _q("u128i1-q2","mc",2,"∫ x⁴ dx = ?","1",["Bump up, divide."],"x⁵/5 + C.",choices=["4x³","x⁵/5 + C","x⁵ + C","x³/3"]),
     _q("u128i1-q3","mc",3,"∫ 1/x dx = ?","2",["Not power rule."],"ln|x| + C.",choices=["x⁰/0 + C","1/(2x²)","ln|x| + C","x + C"])]),
}},

"u-cs115": {"label":"CS 115 · Intro Programming", "level":"uni", "grade":13, "subject":"cs", "outcomes":{
"U115.R1": _o("Structural recursion on lists", "Empty case + inductive case.", "Recursion",
    [_c("u115-r1","Two cases always","empty + cons",
        "Every list recursion: what to return for empty, and how to combine (first lst) with the recursive call on (rest lst).",
        "Return the identity for the empty case (0 for sum, empty for map, false for ormap). Inductive step combines head with recursive tail result.",
        "(define (my-sum lst)\n  (cond [(empty? lst) 0]\n        [else (+ (first lst) (my-sum (rest lst)))]))",
        "Forgetting the base case = infinite recursion.")],
    [_q("u115r1-q1","mc",2,"Empty case for a length function?","1",["Identity for length."],"0.",choices=["1","0","empty","undefined"]),
     _q("u115r1-q2","mc",3,"(my-length lst) = (+ 1 (my-length (rest lst))) — problem?","1",["Trace on empty."],"No base case — infinite loop.",choices=["Wrong operand","No base case","Should use first","Nothing"]),
     _q("u115r1-q3","mc",3,"Identity for ormap on booleans?","1",["Vacuous case."],"false.",choices=["true","false","empty","undefined"])]),
}},

"u-chem120": {"label":"CHEM 120 · General Chemistry 1", "level":"uni", "grade":13, "subject":"chemistry", "outcomes":{
"U120.S1": _o("Limiting reagent", "Which reactant runs out first?", "Stoichiometry",
    [_c("u120-s1","Divide moles by coefficient","smallest wins",
        "Compute moles of each reactant divided by its coefficient. Smallest quotient = limiting reagent.",
        "The limiting reagent caps the yield. Everything else is in excess. Max product = moles-of-limiter × (product coeff / limiter coeff).",
        "2H₂ + O₂ → 2H₂O.\nStart: 4 mol H₂, 3 mol O₂.\nH₂: 4/2 = 2\nO₂: 3/1 = 3\nH₂ smaller → limiting. Max H₂O = 4·(2/2) = 4 mol.",
        "Some pick the reactant with fewer moles without dividing by the coefficient. Always divide first.")],
    [_q("u120s1-q1","mc",3,"2A + B → C. 4 mol A, 3 mol B. Limiting?","0",["Divide by coefficient."],"A/2=2, B/1=3. A limits.",choices=["A","B","Neither","Both"]),
     _q("u120s1-q2","mc",3,"2H₂ + O₂ → 2H₂O. 4 mol H₂, 3 mol O₂. Max mol H₂O?","2",["H₂ limits."],"4 mol H₂O.",choices=["2","3","4","6"]),
     _q("u120s1-q3","mc",2,"'In excess' means…","1",["Opposite of limiting."],"More than needed.",choices=["Used first","More than needed","Doesn't react","Is the product"])]),
}},

"u-phys111": {"label":"PHYS 111 · Physics 1", "level":"uni", "grade":13, "subject":"physics", "outcomes":{
"U111.K1": _o("Kinematics — constant acceleration", "The 'big five' equations.", "Kinematics",
    [_c("u111-k1","Pick the equation missing the unknown","list knowns first",
        "Core three: v = v₀+at (no d), d = v₀t + ½at² (no v), v² = v₀² + 2ad (no t).",
        "List v₀, v, a, d, t. Choose the equation missing exactly the variable you neither have nor need.",
        "v₀=0, a=2, t=3. Find d.\nd = v₀t + ½at² = 0 + ½(2)(9) = 9 m.",
        "Reaching for the first familiar equation without listing knowns wastes time. List first.")],
    [_q("u111k1-q1","numeric",2,"Car from rest, a=3 m/s², 4s. Final v?","12",["v = v₀+at."],"12 m/s.",tol=0.01),
     _q("u111k1-q2","numeric",3,"Same. Distance (m)?","24",["d = ½at²."],"24 m.",tol=0.01),
     _q("u111k1-q3","mc",3,"Know v₀, v, a but NOT t. Equation for d?","2",["No-t."],"v² = v₀² + 2ad.",choices=["v=v₀+at","d=v₀t+½at²","v²=v₀²+2ad","d=vt"])]),
}},

# ═══════════════════ ELEMENTARY · missing subjects per grade ═══════════════════
"e-g1-english": {"label":"Language","level":"elementary","grade":1,"subject":"english","outcomes":{
"L1.R1": _o("Sight words","Words to recognize instantly.","Reading",
    [_c("l1r1","Recognize on sight","no sounding out",
        "Some words come up so often you should just know them.",
        "Words like 'the', 'and', 'is', 'you' — practice till you name them in a second, not sound them out.",
        "the · and · is · you · to · was · for · are",
        "Sounding out every word slows reading down. Sight words free your brain to think about meaning.")],
    [_q("l1r1-q1","mc",1,"Which is a sight word?","0",["Very common short word."],"'the' — one of the most common words.",choices=["the","elephant","umbrella","microscope"]),
     _q("l1r1-q2","mc",1,"Which is NOT usually a sight word?","2",["Longer, rarer words."],"'photosynthesis' is too long and rare.",choices=["and","is","photosynthesis","was"])])
}},
"e-g1-science": {"label":"Science","level":"elementary","grade":1,"subject":"science","outcomes":{
"S1.L1": _o("Living vs non-living","What makes something 'alive'?","Life Systems",
    [_c("s1l1","Signs of life","grow · eat · reproduce",
        "Living things grow, eat, breathe, and can make more of themselves.",
        "A rock doesn't grow, eat, or make more rocks. A puppy does all of these — so puppy is living, rock is not.",
        "Living: puppy, tree, ant\nNot living: rock, chair, cloud",
        "Some kids think anything that moves is alive. A car moves but doesn't eat or grow.")],
    [_q("s1l1-q1","mc",1,"Which is a living thing?","0",["Can it grow?"],"A tree grows and needs water.",choices=["Tree","Rock","Cloud","Chair"]),
     _q("s1l1-q2","mc",1,"Which is NOT living?","2",["Does it eat, grow, or reproduce?"],"A pencil doesn't eat or grow.",choices=["Dog","Baby","Pencil","Fish"])])
}},
"e-g1-social": {"label":"Social Studies","level":"elementary","grade":1,"subject":"social","outcomes":{
"SS1.C1": _o("Our roles and responsibilities","How we help at home and school.","Communities",
    [_c("ss1c1","Roles in a group","each person helps",
        "In every group — family, class, school — each person has jobs that help everyone.",
        "At home a parent might cook and a child might tidy toys. At school a teacher teaches and students listen and take turns.",
        "Home: cook, clean, care for pets\nSchool: listen, share, take turns",
        "'Responsibility' doesn't only mean big adult jobs. Cleaning up after yourself is a real responsibility.")],
    [_q("ss1c1-q1","mc",1,"Which is a student's responsibility at school?","2",["Something a kid does."],"Listening and taking turns is a student's job.",choices=["Cooking dinner","Driving the bus","Listening and taking turns","Paying bills"]),
     _q("ss1c1-q2","mc",1,"What helps a class run well?","1",["Group work."],"Everyone doing their part.",choices=["One person doing everything","Everyone doing their part","No one doing anything","Only the teacher"])])
}},
"e-g1-hpe": {"label":"Health & PE","level":"elementary","grade":1,"subject":"hpe","outcomes":{
"H1.N1": _o("Healthy eating basics","The main food groups.","Health",
    [_c("h1n1","Four food groups","Canada's simple version",
        "Vegetables & fruit, grains, protein, dairy — a balanced meal has bits from most of these.",
        "A balanced lunch might have carrots (veg), bread (grain), cheese (dairy), and turkey (protein).",
        "Veg/Fruit: apple, carrot\nGrain: bread, rice\nProtein: egg, beans\nDairy: milk, yogurt",
        "Candy isn't a food group. Sweets are OK sometimes, but they don't count as one of the healthy groups.")],
    [_q("h1n1-q1","mc",1,"Which is a fruit?","1",["Grows on trees or plants, sweet."],"An apple.",choices=["Carrot","Apple","Bread","Milk"]),
     _q("h1n1-q2","mc",1,"Which is a protein?","2",["Builds muscles."],"An egg — protein.",choices=["Cookie","Chips","Egg","Candy"])])
}},
"e-g1-arts": {"label":"The Arts","level":"elementary","grade":1,"subject":"arts","outcomes":{
"A1.V1": _o("Primary colors","Red, yellow, blue — the starter set.","Visual Arts",
    [_c("a1v1","Primary = can't be mixed","every other colour comes from these",
        "Red, yellow, and blue are the three primary colours. You can't make them by mixing others — but you can mix THEM to make other colours.",
        "Red + yellow = orange. Yellow + blue = green. Blue + red = purple. These are 'secondary' colours.",
        "Primary: red · yellow · blue\nSecondary (made by mixing): orange · green · purple",
        "'Primary' doesn't mean 'best'. It means 'starting point' — every other colour is built from these three.")],
    [_q("a1v1-q1","mc",1,"Which is a primary colour?","2",["Can't be mixed from others."],"Red — a primary colour.",choices=["Orange","Green","Red","Purple"]),
     _q("a1v1-q2","mc",1,"Red + yellow makes…","0",["Warm colour."],"Orange.",choices=["Orange","Green","Purple","Brown"])])
}},

"e-g2-english": {"label":"Language","level":"elementary","grade":2,"subject":"english","outcomes":{
"L2.R1": _o("Reading for meaning","Understanding what a short passage says.","Reading",
    [_c("l2r1","Read + think","don't just say the words",
        "Reading isn't just saying words out loud — it's building a picture in your head of what happened.",
        "Read a short passage. Pause. Ask yourself: WHO is in it? WHERE? WHAT happened first, then next?",
        "Passage: 'Sam went to the park. He played on the swings, then ate an apple.'\nWho? Sam. Where? The park. What? Swings, then apple.",
        "Some kids finish a passage and can't say what happened. Pause and picture it.")],
    [_q("l2r1-q1","mc",2,"'Ana walked her dog in the rain.' Where was Ana?","1",["Where the story happens."],"Outside — she walked in the rain.",choices=["Inside","Outside","At school","In bed"]),
     _q("l2r1-q2","mc",1,"'Max painted a red apple.' What colour is the apple?","0",["Look at the sentence."],"Red.",choices=["Red","Green","Blue","Yellow"])])
}},
"e-g2-science": {"label":"Science","level":"elementary","grade":2,"subject":"science","outcomes":{
"S2.M1": _o("Air and water","Two things all living things need.","Matter & Energy",
    [_c("s2m1","We need both","daily",
        "People, plants, and animals need both air (to breathe) and water (to drink or absorb) to live.",
        "You can hold your breath for about a minute. Without water, you'd get thirsty in hours. Living things constantly cycle both.",
        "Air: breathe in oxygen, breathe out CO₂.\nWater: drink, sweat, pee — cycle back to lakes and clouds.",
        "Air is invisible but very real — it has weight and takes up space.")],
    [_q("s2m1-q1","mc",1,"What do fish breathe?","1",["From water."],"Oxygen dissolved in water.",choices=["Water itself","Oxygen from water","Sand","Nothing"]),
     _q("s2m1-q2","mc",1,"Without water for many days a person will…","1",["Body needs water."],"Get very sick.",choices=["Grow taller","Get very sick","Sleep","Be fine"])])
}},
"e-g2-social": {"label":"Social Studies","level":"elementary","grade":2,"subject":"social","outcomes":{
"SS2.T1": _o("Traditions and celebrations","How different families celebrate.","Communities",
    [_c("ss2t1","Traditions vary","many kinds, all valid",
        "Different families and cultures celebrate different holidays and in different ways.",
        "One family might celebrate Diwali with lights, another Hanukkah with candles, another Eid with feasts. All are important to those families.",
        "Diwali: festival of lights\nHanukkah: menorah for 8 nights\nEid: feast after Ramadan\nLunar New Year: red envelopes",
        "'Different' doesn't mean 'wrong'. Learning about other traditions teaches respect.")],
    [_q("ss2t1-q1","mc",1,"What do many cultures use to celebrate special days?","0",["Common celebration element."],"Special foods and rituals.",choices=["Special foods","Homework","Silence","Nothing"]),
     _q("ss2t1-q2","mc",2,"Why learn about other cultures' traditions?","1",["Understanding others."],"To respect and understand others.",choices=["To copy them","To respect and understand","To make fun","Because you have to"])])
}},
"e-g2-hpe": {"label":"Health & PE","level":"elementary","grade":2,"subject":"hpe","outcomes":{
"H2.M1": _o("Basic movement skills","Running, jumping, throwing, catching.","Movement",
    [_c("h2m1","Fundamentals","building blocks of sports",
        "Simple movements — running, jumping, throwing, catching — are the base of almost every sport and game.",
        "A basketball player throws (chest pass) and catches (bounce). A soccer player runs and kicks. All rest on fundamentals learned young.",
        "Run · Jump · Throw · Catch · Kick · Balance",
        "You get better with practice. It's not that some kids are 'naturally' good — they've just tried more times.")],
    [_q("h2m1-q1","mc",1,"Which is a fundamental movement?","0",["Basic action."],"Jumping.",choices=["Jumping","Sleeping","Reading","Eating"]),
     _q("h2m1-q2","mc",1,"How do you get better at throwing?","2",["Repetition."],"By practicing.",choices=["Watching TV","Being born good","By practicing","Skipping it"])])
}},
"e-g2-arts": {"label":"The Arts","level":"elementary","grade":2,"subject":"arts","outcomes":{
"A2.M1": _o("Beat and rhythm","The pulse behind music.","Music",
    [_c("a2m1","Beat = steady pulse","rhythm = patterns of long/short",
        "The BEAT is the steady heartbeat of the music — clap your hands to it. RHYTHM is the pattern of long and short sounds on top.",
        "In 'Twinkle Twinkle Little Star', the beat is a steady tap. The rhythm is the way the words fall on those beats (some notes long, some short).",
        "Beat:   ♩ ♩ ♩ ♩\nRhythm: ♪♪ ♩ ♪♪ ♩",
        "Beat is steady even when the song gets busy — it doesn't speed up just because more notes are happening.")],
    [_q("a2m1-q1","mc",1,"The steady pulse of music is the…","0",["What you clap to."],"Beat.",choices=["Beat","Melody","Words","Volume"]),
     _q("a2m1-q2","mc",2,"What is rhythm?","1",["Pattern of long/short."],"Pattern of long and short sounds.",choices=["The loudness","Pattern of long and short sounds","The colour of the notes","The words"])])
}},

"e-g3-science": {"label":"Science","level":"elementary","grade":3,"subject":"science","outcomes":{
"S3.F1": _o("Push and pull forces","How pushes and pulls make things move.","Structures",
    [_c("s3f1","Force = push or pull","bigger force = bigger change",
        "A force is any push or pull that changes how something moves.",
        "Kicking a ball is a push. Pulling a wagon is a pull. If nothing pushes or pulls, a thing stays put.",
        "Push: kick, throw, tap\nPull: drag a sled, open a door",
        "'Force' doesn't only mean strong. A tiny tap is still a push force — just a small one.")],
    [_q("s3f1-q1","mc",1,"A push or pull that moves things is called…","2",["Science word."],"A force.",choices=["Speed","Weight","Force","Shape"]),
     _q("s3f1-q2","mc",2,"Kicking a ball is a…","0",["Toward or away?"],"Push.",choices=["Push","Pull","Neither","Sound"])])
}},
"e-g3-social": {"label":"Social Studies","level":"elementary","grade":3,"subject":"social","outcomes":{
"SS3.G1": _o("Communities around the world","How places are different and similar.","People & Environments",
    [_c("ss3g1","Different climates → different lives","weather shapes daily life",
        "How people live depends a lot on where they live. Hot places, cold places, and rainy places lead to different clothing, food, and shelter.",
        "In the Arctic, houses insulate against cold and people wear thick clothing. In a desert, houses stay cool and people wear loose light clothing to avoid heat.",
        "Arctic: parkas, warm homes, hunt/fish\nDesert: loose robes, shade, wells\nRainforest: raised houses, farming",
        "'Different' doesn't mean 'less advanced'. Every community adapts smartly to its environment.")],
    [_q("ss3g1-q1","mc",1,"People in hot places often wear…","1",["Reason for the clothing."],"Loose, light clothing.",choices=["Thick coats","Loose, light clothes","Wool sweaters","Boots"]),
     _q("ss3g1-q2","mc",2,"Why do different places have different types of houses?","2",["What shapes them?"],"Because the weather and land are different.",choices=["Because people are lazy","No reason","Because weather and land differ","Because they're forced to"])])
}},
"e-g3-hpe": {"label":"Health & PE","level":"elementary","grade":3,"subject":"hpe","outcomes":{
"H3.A1": _o("Active living every day","Why moving your body matters.","Active Living",
    [_c("h3a1","Aim for 60 minutes daily","any active movement counts",
        "Health experts recommend about 60 minutes of active movement every day for kids.",
        "'Active' doesn't only mean sports. Walking to school, biking, playing tag, or dancing all count.",
        "Ways to be active:\n  Walk or bike to school\n  Play at recess\n  Sports practice\n  Dance in your room",
        "You don't need to do all 60 minutes at once. Ten minutes here, twenty there — it adds up.")],
    [_q("h3a1-q1","numeric",1,"How many minutes of activity per day is recommended for kids?","60",["An hour."],"About 60 minutes.",tol=5),
     _q("h3a1-q2","mc",2,"Which counts as active movement?","3",["Anything that gets you moving."],"All of these count.",choices=["Walking","Biking","Playing tag","All of these"])])
}},
"e-g3-arts": {"label":"The Arts","level":"elementary","grade":3,"subject":"arts","outcomes":{
"A3.V1": _o("Shape and form","2D shapes vs 3D forms.","Visual Arts",
    [_c("a3v1","Flat vs solid","shape is 2D, form is 3D",
        "A SHAPE is flat — like a circle drawn on paper. A FORM is solid — like a ball you can hold.",
        "A square is a shape (2D, on paper). A cube is a form (3D, has depth). Every 3D form is made of 2D shapes on its surfaces.",
        "Shape (2D): circle, square, triangle\nForm (3D): sphere, cube, pyramid",
        "'Shape' and 'form' are often confused. Flat = shape. Solid = form.")],
    [_q("a3v1-q1","mc",1,"A cube is a…","1",["3D solid."],"Form.",choices=["Shape","Form","Line","Colour"]),
     _q("a3v1-q2","mc",1,"A triangle drawn on paper is a…","0",["Flat, 2D."],"Shape.",choices=["Shape","Form","Line","Sphere"])])
}},

"e-g4-english": {"label":"Language","level":"elementary","grade":4,"subject":"english","outcomes":{
"L4.R1": _o("Making inferences","Reading between the lines.","Reading",
    [_c("l4r1","Clue + what you know = inference","the author doesn't spell it out",
        "An inference is a smart guess based on what's in the text plus what you already know.",
        "If the text says 'Ana grabbed her umbrella and rushed outside', it doesn't SAY it's raining — but you can INFER it is. Umbrellas are for rain (what you know) + Ana grabbed one and rushed out (the clue).",
        "'Sam frowned when he opened his lunch.'\n Clue: he frowned\n Know: people frown when unhappy\n Infer: Sam didn't like his lunch",
        "An inference isn't a wild guess. It has to be supported by clues in the text AND real knowledge.")],
    [_q("l4r1-q1","mc",2,"'Kai zipped his coat and pulled on gloves.' Infer:","0",["Cold clue."],"It was cold outside.",choices=["It was cold","It was hot","It was raining","Kai was tired"]),
     _q("l4r1-q2","mc",3,"'Mia yawned during class.' Infer:","1",["Yawn suggests…"],"Mia was tired.",choices=["Mia was excited","Mia was tired","Mia was angry","Mia was full"])])
}},
"e-g4-social": {"label":"Social Studies","level":"elementary","grade":4,"subject":"social","outcomes":{
"SS4.H1": _o("Early Indigenous societies","Life before Europeans arrived in Canada.","People & Environments",
    [_c("ss4h1","Rich diverse cultures","many nations, unique to each land",
        "Long before Europeans arrived, hundreds of Indigenous nations lived across what is now Canada — with unique languages, governments, and ways of life.",
        "Haudenosaunee in the eastern woodlands built longhouses and grew corn/beans/squash. The Anishinaabe lived around the Great Lakes and moved with the seasons. Coastal nations on the Pacific built cedar houses and totem poles.",
        "Haudenosaunee: longhouses, farming, confederacy\nAnishinaabe: seasonal camps, hunting, fishing\nCoast Salish: cedar homes, salmon, art",
        "Not one 'culture' but many. Each nation has its own language, traditions, and history.")],
    [_q("ss4h1-q1","mc",2,"Before Europeans arrived, how many Indigenous nations lived in what is now Canada?","2",["A lot."],"Hundreds — each unique.",choices=["Just one","A dozen","Hundreds","None"]),
     _q("ss4h1-q2","mc",2,"What did coastal Pacific nations use to build houses?","1",["Local tree."],"Cedar.",choices=["Adobe","Cedar","Marble","Ice"])])
}},
"e-g4-hpe": {"label":"Health & PE","level":"elementary","grade":4,"subject":"hpe","outcomes":{
"H4.C1": _o("Cooperation in games","Working with a team.","Movement",
    [_c("h4c1","Team > individual","games work better when you help",
        "In team games, passing to a teammate who has a better shot usually beats trying to do everything yourself.",
        "In basketball, an open teammate with a clear shot is more likely to score than a covered player. Passing is smart, not weak.",
        "Team play:\n  Pass to open players\n  Talk to your team\n  Cheer for others\n  Play your role",
        "Ball-hogging feels heroic but often loses the game. Trust your team.")],
    [_q("h4c1-q1","mc",2,"When should you pass in a team game?","1",["Best chance to score."],"When a teammate has a better shot.",choices=["Never","When a teammate has a better shot","Always","When you're tired"]),
     _q("h4c1-q2","mc",1,"Cooperation means…","2",["Working together."],"Working together toward a shared goal.",choices=["Playing alone","Winning at any cost","Working together toward a goal","Ignoring others"])])
}},
"e-g4-arts": {"label":"The Arts","level":"elementary","grade":4,"subject":"arts","outcomes":{
"A4.M1": _o("Musical notation basics","Reading simple notes on a staff.","Music",
    [_c("a4m1","5-line staff","notes sit on lines or in spaces",
        "Music is written on a staff — five horizontal lines and four spaces. Notes sit ON lines or IN spaces.",
        "In treble clef (used for most singing and higher instruments), the lines from bottom to top are E-G-B-D-F ('Every Good Boy Deserves Fudge'). The spaces from bottom to top spell F-A-C-E.",
        "Treble clef:\nLines (bottom→top): E G B D F\nSpaces (bottom→top): F A C E",
        "Notes on lines and notes in spaces are different pitches. Look at whether the notehead touches a line or fits between lines.")],
    [_q("a4m1-q1","mc",2,"How many lines are in a musical staff?","1",["Standard staff."],"5.",choices=["4","5","6","7"]),
     _q("a4m1-q2","mc",2,"The spaces on a treble clef spell…","0",["F-A-C-E."],"FACE.",choices=["FACE","EGBDF","LINES","STAR"])])
}},

"e-g5-social": {"label":"Social Studies","level":"elementary","grade":5,"subject":"social","outcomes":{
"SS5.G1": _o("Levels of government","Federal, provincial, municipal — who does what.","People & Environments",
    [_c("ss5g1","Three levels","different jobs, different scale",
        "Canada has three levels of government: federal (whole country), provincial (one province), and municipal (a city or town).",
        "Federal handles things that affect the whole country: military, currency, international trade. Provincial handles education, health care, highways. Municipal handles local things: garbage, water, local roads, parks.",
        "Federal: PM, defence, currency\nProvincial: Premier, schools, hospitals\nMunicipal: Mayor, garbage, local roads",
        "The three levels don't fight — they handle different things. You need all three for the country to work.")],
    [_q("ss5g1-q1","mc",2,"Which level of government manages schools in Ontario?","1",["Ontario is a…"],"Provincial (Ontario government).",choices=["Federal","Provincial","Municipal","None"]),
     _q("ss5g1-q2","mc",2,"Who picks up the garbage in your town?","2",["Local service."],"Municipal government.",choices=["Federal","Provincial","Municipal","Nobody"])])
}},
"e-g5-hpe": {"label":"Health & PE","level":"elementary","grade":5,"subject":"hpe","outcomes":{
"H5.G1": _o("Puberty basics","How your body changes as you grow.","Health",
    [_c("h5g1","Everyone changes","different times, all normal",
        "Puberty is a set of body changes that happen between about age 8 and 14. Every person goes through it, but at their own pace.",
        "Some kids start earlier, some later — both are normal. The changes are gradual (over years, not overnight).",
        "Common changes: growth spurts, voice changes, hair growth, skin changes, mood shifts.",
        "Everyone develops at their own rate. There's no 'behind' or 'ahead' — just different timing.")],
    [_q("h5g1-q1","mc",1,"Puberty usually starts between ages…","1",["Range."],"About 8 to 14.",choices=["3-5","8-14","18-25","40-50"]),
     _q("h5g1-q2","mc",1,"Everyone goes through puberty at…","1",["Not the same time."],"Their own pace.",choices=["The same age","Their own pace","Age 12 exactly","Only some people"])])
}},
"e-g5-french": {"label":"French","level":"elementary","grade":5,"subject":"french","outcomes":{
"F5.G1": _o("Numbers and days","0-30 and the seven days of the week.","Grammar",
    [_c("f5g1","Numbers 1-30","the base you'll reuse everywhere",
        "Numbers un (1), deux (2), trois (3)... vingt (20), vingt-et-un (21), ... trente (30). After 30 the pattern keeps going.",
        "Days: lundi (Mon), mardi (Tue), mercredi (Wed), jeudi (Thu), vendredi (Fri), samedi (Sat), dimanche (Sun). No capital letters in French!",
        "1 un · 2 deux · 3 trois · 4 quatre · 5 cinq\n6 six · 7 sept · 8 huit · 9 neuf · 10 dix\n\nlundi · mardi · mercredi · jeudi · vendredi · samedi · dimanche",
        "Days of the week are NOT capitalized in French, unlike in English.")],
    [_q("f5g1-q1","mc",1,"'sept' means…","2",["Number."],"7.",choices=["6","5","7","10"]),
     _q("f5g1-q2","mc",2,"Which is Wednesday in French?","1",["Middle of the week."],"mercredi.",choices=["mardi","mercredi","vendredi","jeudi"])])
}},
"e-g5-arts": {"label":"The Arts","level":"elementary","grade":5,"subject":"arts","outcomes":{
"A5.M1": _o("Melody","A sequence of notes that make a tune.","Music",
    [_c("a5m1","Melody = notes in order","the part you sing or hum",
        "A melody is a sequence of pitches — the tune you'd sing along to. It goes up, down, or stays the same across time.",
        "'Twinkle Twinkle Little Star' has a very simple melody: two notes the same, then up, up, then back down. The rhythm is separate from the melody.",
        "'Twinkle Twinkle': C-C G-G A-A G  F-F E-E D-D C",
        "Melody and rhythm are different. Two songs can share a melody but have different rhythms.")],
    [_q("a5m1-q1","mc",1,"A melody is…","2",["What you hum."],"A sequence of notes that make a tune.",choices=["The drum part","Loud playing","A sequence of notes","The words"]),
     _q("a5m1-q2","mc",2,"Melody moves through…","1",["Up, down, same."],"Up, down, or same pitch across time.",choices=["Only up","Up, down, or same","Only down","In a circle"])])
}},

"e-g6-social": {"label":"Social Studies","level":"elementary","grade":6,"subject":"social","outcomes":{
"SS6.C1": _o("Canada and the world","How Canada connects to other countries.","People & Environments",
    [_c("ss6c1","Global connections","trade · diplomacy · migration",
        "Canada connects to the world through trade (what we sell and buy), diplomacy (how we work with other countries), and migration (people moving in and out).",
        "The US is Canada's biggest trading partner. Canada is a member of the UN and NATO. Many Canadians were born in another country — bringing languages and cultures with them.",
        "Trade: US · China · UK · Japan\nOrganizations: UN · NATO · G7\nImmigration: about 22% of Canadians born abroad",
        "Countries don't operate alone. Nearly everything you eat, wear, or use has been touched by international trade.")],
    [_q("ss6c1-q1","mc",1,"Canada's biggest trading partner is…","0",["Right next door."],"The United States.",choices=["The United States","France","Australia","Egypt"]),
     _q("ss6c1-q2","mc",2,"Which organization is Canada a member of?","2",["Global body."],"The UN.",choices=["Only the EU","Just NATO","The UN","None"])])
}},
"e-g6-hpe": {"label":"Health & PE","level":"elementary","grade":6,"subject":"hpe","outcomes":{
"H6.L1": _o("Living skills","Making healthy choices under pressure.","Health",
    [_c("h6l1","Pause before you decide","especially under peer pressure",
        "When friends push you to do something you're not sure about, taking a moment to think — 'is this what I actually want?' — is the single most useful habit.",
        "Peer pressure happens even without anyone saying anything mean. It's often subtle. Practising a quick pause ('Actually, no thanks') is a skill you build with reps.",
        "Steps:\n1. Pause — notice what you're feeling\n2. Ask: is this safe? is this me?\n3. Decide — say yes or no clearly\n4. Move on without drama",
        "'I said no' is a complete sentence. You don't owe anyone a big explanation.")],
    [_q("h6l1-q1","mc",1,"When friends pressure you, first…","1",["Take a moment."],"Pause and think.",choices=["Say yes","Pause and think","Run away","Ignore them forever"]),
     _q("h6l1-q2","mc",2,"Real friends will…","2",["How they react to 'no'."],"Respect your 'no' without drama.",choices=["Get angry","Threaten you","Respect your 'no'","Stop being friends"])])
}},
"e-g6-arts": {"label":"The Arts","level":"elementary","grade":6,"subject":"arts","outcomes":{
"A6.M1": _o("Harmony","Two or more notes played together.","Music",
    [_c("a6m1","Harmony ≠ melody","supports the melody underneath",
        "HARMONY is notes played at the same time as the melody, supporting it. A single voice singing = melody. Two voices in thirds = harmony.",
        "Chords are the building block. Simple songs use three chords (I, IV, V in a key). Try playing a C chord (C+E+G) with your right hand and a low C with your left — that's harmony.",
        "Melody: single note line (sing the tune)\nHarmony: chords or extra voices under/beside the melody\n\nChord example: C major = C + E + G",
        "Harmony is not just 'louder melody' — it's ADDITIONAL notes at the same time.")],
    [_q("a6m1-q1","mc",1,"Harmony is…","1",["Multiple notes."],"Notes played together.",choices=["Only one note","Notes played together","Silence","Loud noise"]),
     _q("a6m1-q2","mc",2,"A C major chord uses which notes?","0",["Root + third + fifth."],"C, E, G.",choices=["C, E, G","C, D, E","A, B, C","F, G, A"])])
}},

"e-g7-english": {"label":"Language","level":"elementary","grade":7,"subject":"english","outcomes":{
"L7.W1": _o("Structuring an essay","Intro, body paragraphs, conclusion.","Writing",
    [_c("l7w1","Three parts, always","intro · body · conclusion",
        "A basic essay has three parts: an INTRO (with a thesis), BODY paragraphs (each defending one point), and a CONCLUSION (restating and reflecting).",
        "Each body paragraph should be about ONE idea. Start with a topic sentence, back it up with 2-3 sentences of evidence or reasoning, then link back to your thesis.",
        "Intro:  hook → context → THESIS\nBody 1: topic sentence + evidence + link\nBody 2: topic sentence + evidence + link\nConclusion: restate thesis + broader thought",
        "Some students cram every idea into one paragraph. Give each idea its own paragraph — it makes your reasoning clearer.")],
    [_q("l7w1-q1","mc",1,"An essay body paragraph should focus on…","1",["How many ideas per paragraph?"],"One main idea.",choices=["Everything","One main idea","Only questions","The intro only"]),
     _q("l7w1-q2","mc",2,"Where does the thesis go?","0",["It sets up the essay."],"End of the intro paragraph.",choices=["End of the intro","Middle of body","In the title","In the conclusion only"])])
}},
"e-g7-history": {"label":"History","level":"elementary","grade":7,"subject":"history","outcomes":{
"H7.NF1": _o("New France","French colonization in what is now Canada.","History",
    [_c("h7nf1","1608-1763","from Champlain to the Conquest",
        "New France was the French colony in North America, founded when Samuel de Champlain established Québec City in 1608. It grew along the St. Lawrence and beyond.",
        "New France depended on the fur trade with Indigenous nations, especially the Wendat and later the Anishinaabe. It ended in 1763 when the British took it after the Seven Years' War.",
        "1608: Champlain founds Québec\n1670s: Fur trade expands\n1759: Battle of the Plains of Abraham\n1763: Treaty of Paris — New France becomes British",
        "'New France' didn't mean only what's now Québec. At its peak it stretched to the Great Lakes and down the Mississippi.")],
    [_q("h7nf1-q1","numeric",2,"When was Québec City founded?","1608",["Champlain."],"1608 by Champlain.",tol=2),
     _q("h7nf1-q2","mc",2,"New France's economy depended most on…","1",["Traded with Indigenous nations."],"The fur trade.",choices=["Sugar","Fur trade","Silver","Cotton"])])
}},
"e-g7-geography": {"label":"Geography","level":"elementary","grade":7,"subject":"geography","outcomes":{
"G7.P1": _o("Physical patterns","How landforms and climate shape where people live.","Geography",
    [_c("g7p1","Where and why","physical geography drives settlement",
        "People tend to live where the land supports them: near fresh water, flat farmable land, moderate climate, and natural resources.",
        "Deserts and high mountains have low population density because it's hard to grow food, get water, or move around. River valleys and coastlines have dense populations for the opposite reasons.",
        "High-density: coastlines, river valleys, moderate climates\nLow-density: deserts, high mountains, extreme cold\nExamples: Nile delta (crowded), Sahara (empty)",
        "It's not just about beauty. Physical geography constrains what humans can practically do in a place.")],
    [_q("g7p1-q1","mc",2,"Why are river valleys densely populated?","1",["What do rivers give?"],"Fresh water and flat, fertile land.",choices=["They're colder","Fresh water + flat farmland","Fewer animals","No reason"]),
     _q("g7p1-q2","mc",2,"Why do few people live in the Sahara?","2",["Climate & resources."],"Very hot, very dry, hard to grow food.",choices=["Too rainy","Too cold","Hot, dry, hard to farm","Too crowded"])])
}},
"e-g7-hpe": {"label":"Health & PE","level":"elementary","grade":7,"subject":"hpe","outcomes":{
"H7.F1": _o("Fitness principles","FITT: Frequency, Intensity, Time, Type.","Active Living",
    [_c("h7f1","FITT","the four dials of any workout",
        "Any exercise plan can be described by four dials: FREQUENCY (how often), INTENSITY (how hard), TIME (how long), TYPE (what kind).",
        "To improve, adjust one dial at a time. Adding running 3× a week is a frequency change. Sprinting between blocks is an intensity change. Same for time and type.",
        "F: 3-5x per week is typical for improvement\nI: light (walk) vs moderate (jog) vs vigorous (sprint)\nT: 20-60 min per session\nT: cardio · strength · flexibility",
        "Doing 'more of everything' at once is a fast way to burn out or get injured. Change one dial at a time.")],
    [_q("h7f1-q1","mc",1,"The 'F' in FITT stands for…","0",["How often."],"Frequency.",choices=["Frequency","Fitness","Fun","Fatigue"]),
     _q("h7f1-q2","mc",2,"'Intensity' means…","1",["How hard."],"How hard you're working.",choices=["How long","How hard you're working","What day","Where"])])
}},
"e-g7-arts": {"label":"The Arts","level":"elementary","grade":7,"subject":"arts","outcomes":{
"A7.V1": _o("Visual composition","How to arrange elements in a picture.","Visual Arts",
    [_c("a7v1","Rule of thirds","subject off-centre, not dead-middle",
        "Imagine a tic-tac-toe grid over your picture. Placing important subjects on those lines or intersections — instead of the exact centre — usually makes a stronger image.",
        "Professional photos, paintings, and film shots use this often. The eye finds off-centre compositions more interesting than perfectly centred ones.",
        "Grid: three vertical + three horizontal lines\nStrongest positions: where lines cross\nEye-catching: subject placed on a third, not dead centre",
        "Not every picture NEEDS off-centre. Portraits and symmetrical subjects can look great centred. Rule of thirds is a starting point, not a law.")],
    [_q("a7v1-q1","mc",1,"The rule of thirds divides a picture into…","1",["Grid."],"A 3×3 grid.",choices=["Two halves","A 3×3 grid","Four quarters","Nine squares diagonally"]),
     _q("a7v1-q2","mc",2,"According to rule of thirds, put your subject…","1",["Off-centre."],"On one of the third-lines.",choices=["Exactly in the centre","On one of the third-lines","At the very edge","Anywhere random"])])
}},
"e-g7-french": {"label":"French","level":"elementary","grade":7,"subject":"french","outcomes":{
"F7.G1": _o("Adjective agreement","Adjectives match nouns in gender and number.","Grammar",
    [_c("f7g1","Match gender and number","masc/fem, singular/plural",
        "French adjectives change form to match the noun's gender (masc/fem) and number (sing/plural).",
        "'petit' (small, masc sing) → petite (fem sing) → petits (masc pl) → petites (fem pl). The adjective USUALLY comes after the noun in French.",
        "un petit garçon  (masc sing)\nune petite fille  (fem sing)\ndes petits garçons (masc pl)\ndes petites filles (fem pl)",
        "In English adjectives never change. In French they always do — that's a common English-speaker mistake.")],
    [_q("f7g1-q1","mc",2,"'Une ____ fille.' (petit) — which form?","1",["Fem sing."],"petite.",choices=["petit","petite","petits","petites"]),
     _q("f7g1-q2","mc",3,"'Des ____ chats.' (noir) — masc pl","2",["Masc plural."],"noirs.",choices=["noir","noire","noirs","noires"])])
}},

"e-g8-history": {"label":"History","level":"elementary","grade":8,"subject":"history","outcomes":{
"H8.C1": _o("Confederation, 1867","How Canada became a country.","History",
    [_c("h8c1","1867 · four provinces","John A. Macdonald and the Fathers of Confederation",
        "Canada became a country on July 1, 1867, when Ontario, Quebec, New Brunswick, and Nova Scotia united under the British North America Act.",
        "Reasons: fear of American expansion, need for a shared railway, desire for economic self-sufficiency, British support for the union.",
        "1864: Charlottetown & Québec conferences\n1866: London conference\nJuly 1, 1867: BNA Act — Canada is born (4 provinces)\nLater: MB(1870), BC(1871), PEI(1873), AB & SK(1905), NL(1949)",
        "1867 wasn't full independence — Canada was a self-governing colony of Britain. Full sovereignty came gradually, finalizing in 1982.")],
    [_q("h8c1-q1","numeric",2,"What year did Confederation happen?","1867",["Canada Day originates."],"1867.",tol=2),
     _q("h8c1-q2","mc",2,"Which was NOT one of the original four provinces?","3",["Later joined."],"British Columbia (joined 1871).",choices=["Ontario","Quebec","Nova Scotia","British Columbia"])])
}},
"e-g8-geography": {"label":"Geography","level":"elementary","grade":8,"subject":"geography","outcomes":{
"G8.S1": _o("Global settlement patterns","Where people live and why.","Geography",
    [_c("g8s1","Uneven distribution","most people live in a few places",
        "Human population is very unevenly distributed. About half the world's people live in just a few countries (China, India, USA, Indonesia, Pakistan) and mostly along coasts, in river valleys, or in temperate zones.",
        "Population density (people per km²) is a good measure. Bangladesh has ~1,300/km²; Canada has ~4/km² overall, but our people cluster near the US border.",
        "Highly populated: East Asia, South Asia, Europe, US east coast\nSparsely populated: Sahara, Amazon, Siberia, Arctic\nCanada: 90% of us live within 200 km of the US border",
        "Big country ≠ big population. Canada is huge but most of it is empty land.")],
    [_q("g8s1-q1","mc",2,"Most Canadians live…","1",["Where's it warm/near markets?"],"Within 200 km of the US border.",choices=["In the Arctic","Within 200 km of the US border","In the Rocky Mountains","On Hudson Bay"]),
     _q("g8s1-q2","mc",2,"Population density means…","0",["People per area."],"People per unit area.",choices=["People per unit area","Total people","Land area","Rainfall"])])
}},
"e-g8-hpe": {"label":"Health & PE","level":"elementary","grade":8,"subject":"hpe","outcomes":{
"H8.R1": _o("Healthy relationships","What respect looks like in friendships and beyond.","Health",
    [_c("h8r1","Respect + honesty + boundaries","the three legs of any healthy relationship",
        "Healthy relationships — friendships, family, dating — share three things: respect for each other's feelings, honesty (no lying or manipulating), and clear boundaries.",
        "A friend who gets upset when you spend time with others isn't respecting you. Someone who threatens or guilt-trips isn't being honest. A person who ignores your 'no' isn't honouring your boundaries.",
        "Green flags: listens, respects 'no', apologizes and means it\nRed flags: controls, isolates, lies, ignores boundaries",
        "Being 'nice' isn't the same as being healthy. Someone who's charming in public but controlling in private is not a safe person.")],
    [_q("h8r1-q1","mc",1,"A healthy relationship is built on…","2",["Big three."],"Respect, honesty, boundaries.",choices=["Money","Fear","Respect + honesty + boundaries","Silence"]),
     _q("h8r1-q2","mc",2,"A friend who ignores when you say 'no' is showing…","1",["Which pillar violated?"],"Disrespect for your boundaries.",choices=["Politeness","Disrespect for boundaries","Friendship","Independence"])])
}},
"e-g8-arts": {"label":"The Arts","level":"elementary","grade":8,"subject":"arts","outcomes":{
"A8.ML1": _o("Media literacy basics","Reading advertising and media critically.","Media",
    [_c("a8ml1","Who made this, why, and what are they trying to make you feel","the four questions",
        "For any ad, video, or news post, ask: WHO made it? WHY (what do they want you to do)? WHAT is the message? What is it trying to make you FEEL?",
        "An ad for a car might use fast music, sweeping landscapes, and a happy family — making you FEEL freedom and belonging. The message is 'buying this car gives you those things', which is not literally true.",
        "Questions:\n1. Who made it?\n2. Who is it for?\n3. What technique is used (music, colours, celebrities)?\n4. What are they trying to make me DO?",
        "Ads work best on people who don't notice they're ads. Noticing the techniques weakens their power over you.")],
    [_q("a8ml1-q1","mc",2,"When you see an ad, first ask…","0",["Source matters."],"Who made it and why?",choices=["Who made it and why?","How pretty it is","How long it is","What colour it is"]),
     _q("a8ml1-q2","mc",2,"An ad using sweeping music and happy families is trying to make you feel…","1",["Emotional appeal."],"Happy and connected — even if the product doesn't deliver.",choices=["Bored","Emotionally connected","Angry","Sleepy"])])
}},

# ═══════════════════ SECONDARY · missing subjects per grade ═══════════════════
"s-g9-science": {"label":"Science (SNC1D)","level":"secondary","grade":9,"subject":"science","outcomes":{
"S9.C1": _o("Atomic structure","Protons, neutrons, electrons.","Chemistry",
    [_c("s9c1","Atom = nucleus + electrons","p+, n0, e-",
        "An atom has a tiny dense nucleus (protons + neutrons) with electrons orbiting far outside. Almost all the mass is in the nucleus; almost all the volume is empty space.",
        "Atomic number = number of protons (defines the element). Mass number = protons + neutrons. Electrons balance protons in a neutral atom.",
        "Carbon-12: 6 p, 6 n, 6 e\nOxygen-16: 8 p, 8 n, 8 e\nAtomic number is what makes an element what it is.",
        "'Neutrons' and 'electrons' get confused. Neutrons are in the nucleus (heavy, no charge). Electrons orbit outside (light, negative).")],
    [_q("s9c1-q1","numeric",2,"Carbon has atomic number 6. How many protons?","6",["Atomic number = protons."],"6.",tol=0.5),
     _q("s9c1-q2","mc",2,"Neutrons are found…","1",["Nucleus."],"In the nucleus.",choices=["Orbiting outside","In the nucleus","Everywhere","Nowhere"])])
}},
"s-g9-french": {"label":"French (FSF1D)","level":"secondary","grade":9,"subject":"french","outcomes":{
"F9.G1": _o("Present tense — regular -ir verbs","Second family of French verbs.","Grammar",
    [_c("f9g1","-ir endings","je -is, tu -is, il/elle -it",
        "-ir verbs (finir, choisir, réussir): drop -ir, add -is (je), -is (tu), -it (il/elle), -issons, -issez, -issent.",
        "Different from -er verbs. Notice the -iss- that appears in the plural forms.",
        "finir → je finis · tu finis · il finit\n            nous finissons · vous finissez · ils finissent",
        "The 'iss' in plural forms is what makes -ir verbs feel different from -er. Practice a few until it clicks.")],
    [_q("f9g1-q1","mc",2,"'Je _____ mon travail.' (finir)","0",["Je form."],"finis.",choices=["finis","finit","finissons","finissez"]),
     _q("f9g1-q2","mc",3,"'Nous _____ le film.' (choisir)","2",["Nous form."],"choisissons.",choices=["choisis","choisit","choisissons","choisissent"])])
}},
"s-g9-geography": {"label":"Canadian Geography (CGC1D)","level":"secondary","grade":9,"subject":"geography","outcomes":{
"G9.C1": _o("Canada's physical regions","Seven physical regions of Canada.","Geography",
    [_c("g9c1","Seven regions","from Cordillera to the Shield to the Arctic",
        "Canada is divided into seven physical (physiographic) regions: Western Cordillera, Interior Plains, Canadian Shield, Great Lakes-St. Lawrence Lowlands, Appalachian, Hudson Bay Lowlands, Innuitian (Arctic).",
        "Each region has distinct landforms, soils, and climate — which shapes where people settle and what industries grow there.",
        "West → East:\n  Cordillera (mountains) → Plains (farming) → Shield (mining) → Lowlands (people) → Appalachian (fishing) → Hudson Bay & Innuitian (sparse)",
        "The Canadian Shield covers about half of Canada by land area but supports very few people — thin soil, cold climate.")],
    [_q("g9c1-q1","mc",2,"How many physical regions does Canada have?","2",["Standard division."],"Seven.",choices=["Three","Five","Seven","Ten"]),
     _q("g9c1-q2","mc",2,"Which region covers about half of Canada?","1",["Rocky, ancient, thin soil."],"Canadian Shield.",choices=["Cordillera","Canadian Shield","Appalachian","Innuitian"])])
}},
"s-g9-hpe": {"label":"Health & PE (PPL1O)","level":"secondary","grade":9,"subject":"hpe","outcomes":{
"H9.F1": _o("Cardio and strength — the difference","Two main pillars of fitness.","Active Living",
    [_c("h9f1","Two pillars","different systems, different benefits",
        "CARDIO (aerobic) trains your heart and lungs. STRENGTH trains your muscles. Both matter — neither replaces the other.",
        "Running, biking, swimming = cardio. Lifting weights, pushups, resistance bands = strength. A balanced week has some of each — 3 cardio sessions + 2 strength sessions is a common recipe.",
        "Cardio: 20-60 min, elevated heart rate\nStrength: 8-12 reps, 2-4 sets per muscle group\nRecovery: 48h between strength sessions for same muscle",
        "Doing only cardio without strength = weaker muscles as you age. Only strength without cardio = weaker heart. You need both.")],
    [_q("h9f1-q1","mc",2,"Running trains mainly…","0",["Heart & lungs."],"Cardio (aerobic) system.",choices=["Cardio system","Muscle strength","Flexibility","Balance"]),
     _q("h9f1-q2","mc",2,"For strength training, how much rest between sessions of the same muscle?","1",["Muscles need time to rebuild."],"About 48 hours.",choices=["No rest","About 48 hours","A week","Never train it twice"])])
}},
"s-g9-arts": {"label":"The Arts","level":"secondary","grade":9,"subject":"arts","outcomes":{
"A9.V1": _o("Elements of design","Line, shape, form, colour, texture, value, space.","Visual Arts",
    [_c("a9v1","Seven elements","the vocabulary of visual design",
        "Every visual artwork is built from seven design elements: LINE, SHAPE, FORM, COLOUR, TEXTURE, VALUE (light/dark), and SPACE.",
        "These are just names for the building blocks — an artist arranges them to create the whole piece. Being able to name them lets you critique and improve your own work.",
        "Line: straight or curved marks\nShape: 2D area\nForm: 3D volume\nColour: hue\nTexture: how it feels or looks like it feels\nValue: darkness/lightness\nSpace: positive (subject) and negative (background)",
        "'Value' is not the same as 'colour'. A photo can be all-black-and-white and still have huge value contrast (bright whites, deep blacks).")],
    [_q("a9v1-q1","mc",1,"Which is NOT one of the seven elements of design?","3",["Not in the standard list."],"Symmetry (that's a principle, not an element).",choices=["Line","Shape","Colour","Symmetry"]),
     _q("a9v1-q2","mc",2,"Value refers to…","2",["Not hue."],"Lightness vs darkness.",choices=["The price","The colour","Lightness vs darkness","The size"])])
}},
"s-g9-business": {"label":"Business (BBI1O)","level":"secondary","grade":9,"subject":"business","outcomes":{
"B9.E1": _o("Needs vs wants","The starting point of money decisions.","Economics",
    [_c("b9e1","Needs are essential","wants are optional",
        "NEEDS are things you can't live without (food, water, shelter, basic clothing, healthcare). WANTS are things that make life better but aren't essential (a new phone, entertainment, brand-name clothes).",
        "Every dollar you spend is a needs-vs-wants call. A budget that meets your needs first, then chooses which wants matter most to you, is the foundation of personal finance.",
        "Needs: rent, groceries, transit to work, medicine\nWants: new phone, concert tickets, name-brand shoes, streaming services",
        "The line moves as you get older. A phone is closer to a 'need' for a working adult than for a 10-year-old. Be honest about which side each purchase is on.")],
    [_q("b9e1-q1","mc",1,"Which is a NEED?","2",["Essential for life."],"Water.",choices=["Video game","New sneakers","Water","Concert ticket"]),
     _q("b9e1-q2","mc",2,"When budgeting, cover ___ first.","0",["Priority."],"Needs first, then wants.",choices=["Needs first","Wants first","The most fun items","Whatever's cheapest"])])
}},

"s-g10-english": {"label":"English (ENG2D)","level":"secondary","grade":10,"subject":"english","outcomes":{
"E10.W1": _o("Paragraph structure — PIE","Point · Illustration · Explanation.","Writing",
    [_c("e10w1","PIE structure","the workhorse paragraph",
        "A strong body paragraph has three parts: POINT (topic sentence — the argument), ILLUSTRATION (evidence — quote, example, data), and EXPLANATION (why the evidence supports the point).",
        "The most common weakness is skipping the EXPLANATION — students paste a quote and expect it to speak for itself. It never does. Always spell out what the quote proves and how.",
        "POINT: Hamlet's inaction is a moral choice.\nILLUSTRATION: 'Thus conscience does make cowards of us all' (3.1).\nEXPLANATION: He literally names conscience as what stops him — deliberate deliberation, not weakness.",
        "Don't confuse PIE with 'quote sandwich'. PIE is structural — argument, then evidence, then interpretation.")],
    [_q("e10w1-q1","mc",1,"The 'E' in PIE stands for…","2",["Interpretation step."],"Explanation.",choices=["Evidence","Example","Explanation","Enunciation"]),
     _q("e10w1-q2","mc",3,"The most common paragraph weakness is skipping…","2",["Which step is often missing?"],"Explanation — leaving quotes to speak for themselves.",choices=["The point","The illustration","The explanation","The topic sentence"])])
}},
"s-g10-civics": {"label":"Civics (CHV2O)","level":"secondary","grade":10,"subject":"civics","outcomes":{
"C10.G1": _o("How Canada is governed","Parliament, PM, Cabinet, and the courts.","Government",
    [_c("c10g1","Three branches","legislative · executive · judicial",
        "Canada's federal government has three branches: LEGISLATIVE (Parliament — MPs make laws), EXECUTIVE (PM and Cabinet — run the country), JUDICIAL (courts — interpret the laws).",
        "The PM comes from the party with the most MPs and picks the Cabinet from among them. Elections choose MPs; MPs (in effect) choose the PM. The Supreme Court is separate — appointed, not elected.",
        "Legislative: House of Commons (elected MPs) + Senate (appointed)\nExecutive: Prime Minister + Cabinet\nJudicial: Supreme Court + lower courts",
        "You don't vote for the PM directly in Canada. You vote for your local MP; the party with the most MPs forms government, and its leader is PM.")],
    [_q("c10g1-q1","mc",2,"You vote directly for…","1",["What's on your ballot?"],"Your local MP.",choices=["The Prime Minister","Your local MP","The Chief Justice","The Governor General"]),
     _q("c10g1-q2","mc",2,"Which branch interprets the laws?","2",["Courts."],"The judicial branch.",choices=["Legislative","Executive","Judicial","None of these"])])
}},
"s-g10-french": {"label":"French (FSF2D)","level":"secondary","grade":10,"subject":"french","outcomes":{
"F10.G1": _o("Passé composé with avoir","Talking about the past.","Grammar",
    [_c("f10g1","avoir + past participle","present of avoir + -é ending",
        "Passé composé = present of avoir + past participle. Most verbs use avoir; a specific list uses être.",
        "For -er verbs, the past participle ends in -é. Parler → parlé. J'ai parlé (I spoke). Tu as parlé (you spoke). Il a parlé (he spoke).",
        "parler → parlé\n  j'ai parlé · tu as parlé · il a parlé\n  nous avons parlé · vous avez parlé · ils ont parlé",
        "Some students conjugate the main verb ('je parlai'). No — you conjugate AVOIR and keep the participle in one fixed form.")],
    [_q("f10g1-q1","mc",2,"'J'____ mangé une pomme.' Auxiliary?","0",["avoir, je form."],"ai.",choices=["ai","suis","est","as"]),
     _q("f10g1-q2","mc",3,"Passé composé of 'parler' with 'tu'?","2",["Auxiliary + participle."],"Tu as parlé.",choices=["Tu es parlé","Tu ai parlé","Tu as parlé","Tu a parlé"])])
}},
"s-g10-history": {"label":"Canadian History (CHC2D)","level":"secondary","grade":10,"subject":"history","outcomes":{
"H10.W1": _o("Canada in WW1","How WW1 shaped modern Canada.","History",
    [_c("h10w1","1914-1918","Vimy · conscription · postwar identity",
        "Canada entered WW1 as part of the British Empire in 1914. About 620,000 Canadians served; 66,000 died. Vimy Ridge (1917) — where all four Canadian divisions fought together for the first time — became a defining moment of national identity.",
        "The war also fractured Canada. The Conscription Crisis of 1917 pitted English Canada (mostly for) against French Canada (mostly against), a wound that lasted decades.",
        "1914: war begins, Canada joins with Britain\n1915: Second Ypres (first gas attack)\n1917: Vimy Ridge (April) · Conscription Crisis\n1918: war ends Nov 11\nAfter: Canada signs the Treaty of Versailles independently → step toward sovereignty",
        "Vimy is often called 'the birth of Canada as a nation' — a nice narrative, but it also erases how divided the country was over the war.")],
    [_q("h10w1-q1","numeric",2,"What year did the Battle of Vimy Ridge happen?","1917",["During WW1."],"1917.",tol=1),
     _q("h10w1-q2","mc",2,"The Conscription Crisis divided Canada mostly along which line?","1",["Which two groups disagreed?"],"English vs French Canada.",choices=["East vs West","English vs French Canada","North vs South","Rich vs poor"])])
}},
"s-g10-hpe": {"label":"Health & PE (PPL2O)","level":"secondary","grade":10,"subject":"hpe","outcomes":{
"H10.M1": _o("Mental health basics","Stress, anxiety, when to seek help.","Health",
    [_c("h10m1","Mental health = a spectrum","not just 'sick or well'",
        "Everyone has mental health — it's a spectrum from thriving to struggling to unwell. It goes up and down over your life, just like physical health.",
        "Warning signs to take seriously: persistent sadness for weeks, panic that stops daily life, thoughts of self-harm, sudden withdrawal from friends. Kids-Help-Phone (1-800-668-6868) is a free 24/7 resource in Canada.",
        "Everyday care: sleep, exercise, sunlight, social contact, talking to someone you trust.\nRed flags: weeks of low mood, panic that prevents daily life, self-harm thoughts.\nHelp: family doctor, school counsellor, Kids Help Phone 1-800-668-6868.",
        "Asking for help is a sign of strength, not weakness. Most people need it at some point.")],
    [_q("h10m1-q1","mc",1,"Mental health is…","1",["Not binary."],"A spectrum everyone is on.",choices=["Only for sick people","A spectrum everyone is on","Fake","Only mental illness"]),
     _q("h10m1-q2","mc",2,"If a friend has weeks of persistent low mood, you should…","2",["Take it seriously."],"Encourage them to talk to a trusted adult or professional.",choices=["Ignore it","Wait a year","Encourage them to talk to a trusted adult","Laugh at them"])])
}},
"s-g10-cs": {"label":"Intro to CS (ICS2O)","level":"secondary","grade":10,"subject":"cs","outcomes":{
"C10.P1": _o("Variables and types","Storing data in a program.","Programming",
    [_c("c10p1","Variables hold values","typed by what they hold",
        "A variable is a named container that holds a value. Types tell the computer what kind of value: integer, float, string, boolean.",
        "In Python: x = 5 (int), name = 'Ada' (string), pi = 3.14 (float), done = True (bool). Assignment (=) is not the same as equality (==).",
        "x = 5           # int\nname = 'Ada'   # string\npi = 3.14      # float\ndone = True    # bool",
        "'=' assigns. '==' compares. Confusing them causes half the bugs new programmers write.")],
    [_q("c10p1-q1","mc",1,"'name = \"Ada\"' — what type is name?","1",["Text."],"String.",choices=["Integer","String","Float","Boolean"]),
     _q("c10p1-q2","mc",2,"'x == 5' vs 'x = 5' — which is comparison?","1",["Two equals."],"x == 5.",choices=["x = 5","x == 5","Both","Neither"])])
}},

"s-g11-biology": {"label":"Biology (SBI3U)","level":"secondary","grade":11,"subject":"science","outcomes":{
"B11.C1": _o("Cell membrane transport","Diffusion, osmosis, active transport.","Biology",
    [_c("b11c1","Passive vs active","gradient direction and energy",
        "PASSIVE transport (diffusion, osmosis) moves molecules DOWN the concentration gradient — no cell energy needed. ACTIVE transport moves things UP the gradient — requires ATP.",
        "Osmosis is a special case: water diffusing across a membrane toward higher solute concentration. That's why a cell in salty water shrinks (water leaves) and in fresh water swells (water enters).",
        "Passive: down gradient, no ATP\n  Diffusion (any molecule)\n  Osmosis (water specifically)\nActive: up gradient, uses ATP\n  Sodium-potassium pump",
        "Osmosis moves WATER, not solute. The solute stays put; water crosses to even out concentration.")],
    [_q("b11c1-q1","mc",2,"Osmosis specifically moves…","1",["Which molecule?"],"Water.",choices=["Salt","Water","Sugar","Protein"]),
     _q("b11c1-q2","mc",3,"Active transport requires…","2",["Uphill needs energy."],"Cellular energy (ATP).",choices=["Nothing","Water","ATP","Sunlight"])])
}},
"s-g11-chemistry": {"label":"Chemistry (SCH3U)","level":"secondary","grade":11,"subject":"science","outcomes":{
"C11.M1": _o("Mole concept","Avogadro's number and molar mass.","Chemistry",
    [_c("c11m1","1 mole = 6.022 × 10²³","the chemist's counting unit",
        "A mole is just a very big number: 6.022 × 10²³ (Avogadro's number). Chemists use it because atoms are so small that meaningful reactions involve enormous counts.",
        "Molar mass = mass of one mole of an element or compound, in grams. Water (H₂O) has molar mass ≈ 18 g/mol. So 18 g of water = 1 mole = 6.022 × 10²³ water molecules.",
        "1 mol = 6.022 × 10²³ particles\nMolar mass of H₂O:\n  H: 1 × 2 = 2\n  O: 16 × 1 = 16\n  Total: 18 g/mol",
        "'Molar mass' and 'molecular mass' both exist — they're numerically the same but molar mass has units g/mol.")],
    [_q("c11m1-q1","numeric",2,"How many grams in 1 mole of water?","18",["H₂O molar mass."],"18 g/mol.",tol=0.5),
     _q("c11m1-q2","mc",3,"Avogadro's number is approximately…","2",["Very big."],"6.022 × 10²³.",choices=["6.022 × 10¹⁰","6.022 × 10¹⁵","6.022 × 10²³","6.022 × 10³⁰"])])
}},
"s-g11-physics": {"label":"Physics (SPH3U)","level":"secondary","grade":11,"subject":"science","outcomes":{
"P11.F1": _o("Newton's laws — three of them","Inertia, F=ma, action-reaction.","Physics",
    [_c("p11f1","Three laws","the whole foundation of mechanics",
        "1st (Inertia): an object at rest stays at rest; in motion stays in motion, unless a force acts. 2nd: F = ma. 3rd: for every action there's an equal and opposite reaction.",
        "1st law explains why you lurch forward when a bus stops. 2nd law lets you compute how much force to apply for a given acceleration. 3rd law is why a rocket works — pushing gas down pushes the rocket up.",
        "1st: no force → no change in motion\n2nd: F = ma\n3rd: F(A on B) = -F(B on A)",
        "'For every action there's a reaction' doesn't mean they cancel. They act on DIFFERENT objects.")],
    [_q("p11f1-q1","mc",1,"Newton's second law is…","1",["Force equation."],"F = ma.",choices=["F = mv","F = ma","F = m/a","F = mg"]),
     _q("p11f1-q2","mc",2,"When you push a wall, the wall pushes back on you with…","2",["Third law."],"Equal and opposite force.",choices=["No force","Half the force","Equal and opposite force","Twice the force"])])
}},
"s-g11-french": {"label":"French (FSF3U)","level":"secondary","grade":11,"subject":"french","outcomes":{
"F11.G1": _o("Imparfait","Past tense for descriptions and habits.","Grammar",
    [_c("f11g1","Imparfait vs passé composé","ongoing vs completed",
        "IMPARFAIT describes ongoing or repeated actions in the past ('I used to', 'was doing'). PASSÉ COMPOSÉ describes a single completed action ('I did', 'I have done').",
        "Formation: take the nous form of the present, drop -ons, add: -ais, -ais, -ait, -ions, -iez, -aient.",
        "parler (nous parlons):\n  je parlais · tu parlais · il parlait\n  nous parlions · vous parliez · ils parlaient",
        "The tenses have different jobs. 'Il pleuvait quand je suis arrivé' = 'It was raining when I arrived' — imparfait for ongoing rain, passé composé for the moment of arrival.")],
    [_q("f11g1-q1","mc",3,"Imparfait describes…","1",["Ongoing past."],"Ongoing or repeated past actions.",choices=["Future","Ongoing past","One completed past action","Present"]),
     _q("f11g1-q2","mc",3,"'Je _____ toujours à sept heures.' (finir, imparfait)","2",["-issais for je."],"finissais.",choices=["fini","finis","finissais","finirait"])])
}},
"s-g11-business": {"label":"Marketing (BMI3C)","level":"secondary","grade":11,"subject":"business","outcomes":{
"M11.4P1": _o("The 4 Ps of marketing","Product · Price · Place · Promotion.","Marketing",
    [_c("m11p1","Marketing mix","four levers a company controls",
        "Every marketing decision falls under one of four Ps: PRODUCT (what you sell), PRICE (what you charge), PLACE (where you sell it), PROMOTION (how you tell people).",
        "The four Ps are the levers a marketing team can pull. A great product at the wrong price, wrong place, or with no promotion still fails.",
        "Product: features, quality, brand\nPrice: cost, discounts, tiers\nPlace: online, retail, direct\nPromotion: ads, PR, social",
        "A marketing plan isn't just 'run ads'. The other 3 Ps decide whether the ads work.")],
    [_q("m11p1-q1","mc",1,"Which is one of the 4 Ps?","2",["Standard four."],"Price.",choices=["Physics","Psychology","Price","Politics"]),
     _q("m11p1-q2","mc",2,"'Where you sell it' is which P?","2",["Location."],"Place.",choices=["Product","Price","Place","Promotion"])])
}},

"s-g12-english": {"label":"English (ENG4U)","level":"secondary","grade":12,"subject":"english","outcomes":{
"E12.C1": _o("Close reading","Reading a passage for what it does, not just what it says.","Literature",
    [_c("e12c1","Beyond plot","how the author achieves the effect",
        "Close reading is the practice of paying attention to HOW a passage is written — word choice, syntax, imagery, rhythm — not just what it 'says'.",
        "Instead of 'this passage is sad', a close reading names the specific devices creating sadness: monosyllabic diction, short fragmented sentences, cold imagery, absence of colour words. That specificity is the entire game in senior English.",
        "Sample steps:\n1. Read once for meaning.\n2. Read again marking word choices, unusual syntax, imagery patterns.\n3. Ask: what EFFECT do these choices produce? How?\n4. Write about the HOW, not just the WHAT.",
        "The biggest mistake: quoting evidence and saying 'this shows the character is sad'. Say WHY those specific words show it — that's the analysis.")],
    [_q("e12c1-q1","mc",2,"Close reading focuses on…","1",["Craft, not just plot."],"How the passage is written, not just what it says.",choices=["Plot summary","How the passage is written","The cover","The author's biography"]),
     _q("e12c1-q2","mc",3,"'This passage is sad because the sentences are short and the diction is monosyllabic' is…","2",["Naming devices."],"A close-reading observation.",choices=["A plot summary","A biography","A close-reading observation","Off-topic"])])
}},
"s-g12-mhf": {"label":"Advanced Functions (MHF4U)","level":"secondary","grade":12,"subject":"math","outcomes":{
"M12.E1": _o("Exponential functions","Growth and decay patterns.","Functions",
    [_c("m12e1","f(x) = a·bˣ","exponential is UNLIKE linear",
        "An exponential function has form f(x) = a·bˣ. It grows (or shrinks) by a constant PERCENTAGE per step — not a constant amount.",
        "Compare: linear adds 3 per step (3, 6, 9, 12). Exponential multiplies by 3 per step (3, 9, 27, 81). Exponentials outpace ANY linear or polynomial function eventually.",
        "f(x) = 2·3ˣ\nf(0) = 2\nf(1) = 6\nf(2) = 18\nf(3) = 54\nGrowth factor per step: 3× ",
        "Some students mix up 2x (linear, add 2 each time) with 2ˣ (exponential, double each time). Read the position of x carefully.")],
    [_q("m12e1-q1","numeric",2,"f(x) = 2ˣ. f(4) = ?","16",["2^4."],"16.",tol=0.5),
     _q("m12e1-q2","numeric",3,"f(x) = 3·2ˣ. f(3) = ?","24",["3·(2^3)."],"3·8 = 24.",tol=0.5)])
}},
"s-g12-biology": {"label":"Biology (SBI4U)","level":"secondary","grade":12,"subject":"science","outcomes":{
"B12.G1": _o("Genetics — dominant/recessive","Mendelian inheritance.","Biology",
    [_c("b12g1","Alleles pair up","dominant hides recessive",
        "For a gene with two versions (alleles), if one is DOMINANT it 'shows' whenever it's present. RECESSIVE only shows when both copies are recessive.",
        "For a trait like brown vs blue eyes: B (brown) is dominant, b (blue) is recessive. A person with genotype BB or Bb has brown eyes. Only bb gives blue eyes.",
        "BB — brown eyes\nBb — brown eyes (carries blue)\nbb — blue eyes\n\nTwo Bb parents: 25% BB, 50% Bb, 25% bb → 75% brown, 25% blue",
        "'Dominant' does NOT mean 'more common' in the population. It means 'shows when present alongside recessive'.")],
    [_q("b12g1-q1","mc",2,"A recessive trait shows when the genotype is…","1",["Both copies must be recessive."],"Both alleles recessive (e.g., bb).",choices=["One dominant one recessive","Both recessive","Both dominant","Any combination"]),
     _q("b12g1-q2","mc",3,"Two Bb parents. Chance of a bb child?","2",["Punnett square."],"25%.",choices=["0%","10%","25%","50%"])])
}},
"s-g12-chemistry": {"label":"Chemistry (SCH4U)","level":"secondary","grade":12,"subject":"science","outcomes":{
"C12.E1": _o("Equilibrium — Le Chatelier","How a system responds to change.","Chemistry",
    [_c("c12e1","System shifts to counter","stress → response",
        "When a reaction at equilibrium is disturbed (temperature, concentration, or pressure change), it shifts in the direction that counteracts the change. That's Le Chatelier's principle.",
        "Add more reactant → equilibrium shifts to make more product. Cool an exothermic reaction → shift toward products (produces heat to compensate). Increase pressure → shift toward the side with fewer gas moles.",
        "N₂ + 3H₂ ⇌ 2NH₃ (exothermic)\n  Add N₂ → shift right\n  Remove NH₃ → shift right\n  Increase pressure → shift right (fewer moles gas)\n  Increase temperature → shift left (endothermic direction absorbs heat)",
        "Le Chatelier doesn't say the disturbance is undone. It says the system shifts to REDUCE the disturbance — but the new equilibrium is different from the old one.")],
    [_q("c12e1-q1","mc",2,"Adding more reactant shifts equilibrium…","0",["Toward more product."],"Toward products.",choices=["Toward products","Toward reactants","Doesn't change","Reverses direction"]),
     _q("c12e1-q2","mc",3,"For N₂ + 3H₂ ⇌ 2NH₃, increasing pressure favours…","0",["Fewer moles side."],"Products (2 mol < 4 mol).",choices=["Products","Reactants","No change","Reversal"])])
}},
"s-g12-physics": {"label":"Physics (SPH4U)","level":"secondary","grade":12,"subject":"science","outcomes":{
"P12.E1": _o("Conservation of energy","Total energy stays constant.","Physics",
    [_c("p12e1","Energy transforms","never created, never destroyed",
        "Energy can change form (kinetic ↔ potential ↔ thermal) but the total in a closed system stays the same.",
        "A ball dropped from height h has gravitational PE = mgh at the top, converting entirely to KE = ½mv² at the bottom (ignoring air resistance). Setting them equal: v = √(2gh).",
        "PE at top = mgh\nKE at bottom = ½mv²\nmgh = ½mv²\nv = √(2gh)",
        "'Ignoring air resistance' is a big deal. Real drops lose some energy to air. Conservation still holds — the 'lost' energy went into heating the air and the ball.")],
    [_q("p12e1-q1","numeric",2,"Ball dropped from 5 m. Speed at bottom (m/s, g=9.8)? Round to 1 dp","9.9",["v = √(2gh)."],"√(2·9.8·5) ≈ 9.9 m/s.",tol=0.2),
     _q("p12e1-q2","mc",2,"Energy is…","2",["Conservation law."],"Neither created nor destroyed, only transformed.",choices=["Made by moving","Destroyed by friction","Neither created nor destroyed","Different for each object"])])
}},
"s-g12-cs": {"label":"CS (ICS4U)","level":"secondary","grade":12,"subject":"cs","outcomes":{
"C12.A1": _o("Time complexity — Big O","How fast an algorithm grows.","Programming",
    [_c("c12a1","Big O = growth rate","not exact time — how it scales",
        "Big O notation describes how an algorithm's runtime grows as the input gets bigger. O(1) = constant, O(n) = linear, O(n²) = quadratic, O(log n) = logarithmic.",
        "A linear search through a list is O(n) — doubles the list, doubles the time. Binary search is O(log n) — doubles the list, adds one step. That's why sorted data + binary search beats unsorted + linear search at any real scale.",
        "O(1)     : lookup by index\nO(log n) : binary search\nO(n)     : linear search\nO(n log n): efficient sorts (mergesort)\nO(n²)    : nested loops over data",
        "Big O ignores constants. O(2n) and O(n) are both O(n) — the growth shape is what matters, not the specific multiplier.")],
    [_q("c12a1-q1","mc",2,"Binary search is…","1",["Doubles → one more step."],"O(log n).",choices=["O(1)","O(log n)","O(n)","O(n²)"]),
     _q("c12a1-q2","mc",3,"Which is fastest for large n?","0",["Lowest growth."],"O(log n).",choices=["O(log n)","O(n)","O(n²)","O(n³)"])])
}},

}


# Scripted Socratic conversation trees keyed by outcome id
CONVERSATIONS = {
    "M2.O1": [
        {"role":"user","content":"How do I add 34 + 25?"},
        {"role":"assistant","content":"Split each into tens and ones. What's the tens part of 34?"},
        {"role":"user","content":"30."},
        {"role":"assistant","content":"And 25 is 20 + 5. Add the tens: 30 + 20 = ?"},
        {"role":"user","content":"50."},
        {"role":"assistant","content":"Now the ones: 4 + 5 = 9. Together: 50 + 9 = 59!"},
    ],
    "M4.N1": [
        {"role":"user","content":"I don't get multiplication."},
        {"role":"assistant","content":"Do you know 4 + 4 + 4?"},
        {"role":"user","content":"12."},
        {"role":"assistant","content":"That's actually 3 × 4! You already knew — you just called it adding."},
    ],
    "M5.N1": [
        {"role":"user","content":"How do I add 1/4 + 2/4?"},
        {"role":"assistant","content":"Same bottom, so just add the tops. What's 1 + 2?"},
        {"role":"user","content":"3."},
        {"role":"assistant","content":"Right — keep the 4. Answer: 3/4."},
    ],
    "M7.A1": [
        {"role":"user","content":"How do I solve x + 5 = 12?"},
        {"role":"assistant","content":"Whatever's done to x, undo it. 5 is ADDED — so subtract 5 from BOTH sides. What's left?"},
        {"role":"user","content":"x = 7."},
        {"role":"assistant","content":"Nice. Always do the same thing to both sides — that keeps the balance."},
    ],
    "M8.G1": [
        {"role":"user","content":"Legs are 3 and 4. What's the hypotenuse?"},
        {"role":"assistant","content":"Use a² + b² = c². What's 3² + 4²?"},
        {"role":"user","content":"9 + 16 = 25."},
        {"role":"assistant","content":"So c² = 25, c = √25 = 5. That's the famous 3-4-5 triangle."},
    ],
    "M9.A1": [
        {"role":"user","content":"Solve 2x + 3 = 11."},
        {"role":"assistant","content":"Undo the +3 first. Subtract 3 from both sides — what's left?"},
        {"role":"user","content":"2x = 8."},
        {"role":"assistant","content":"Divide both sides by 2. x = 4."},
    ],
    "M10.A1": [
        {"role":"user","content":"Factor x² + 7x + 12."},
        {"role":"assistant","content":"Need two numbers with product 12 AND sum 7. Factor pairs of 12?"},
        {"role":"user","content":"1×12, 2×6, 3×4."},
        {"role":"assistant","content":"Which pair sums to 7?"},
        {"role":"user","content":"3 + 4."},
        {"role":"assistant","content":"So x² + 7x + 12 = (x+3)(x+4)."},
    ],
    "U127.L1": [
        {"role":"user","content":"lim (x²-4)/(x-2) as x→2 — I get 0/0."},
        {"role":"assistant","content":"0/0 is indeterminate, not undefined. Try factoring the numerator."},
        {"role":"user","content":"(x-2)(x+2)."},
        {"role":"assistant","content":"Cancel (x-2). You're left with x+2. Limit as x→2 is 4."},
    ],
    "U115.R1": [
        {"role":"user","content":"My length function loops forever."},
        {"role":"assistant","content":"What does it do when lst is empty?"},
        {"role":"user","content":"Recurses on (rest empty)…"},
        {"role":"assistant","content":"No base case. Add [(empty? lst) 0] as the first cond branch."},
    ],
}


def seed_if_empty():
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        if db.query(Course).count() > 0:
            return
        for cid, c in SEED.items():
            course = Course(id=cid, label=c["label"], level=c["level"],
                            grade=c["grade"], subject=c["subject"])
            db.add(course)
            for oid, o in c["outcomes"].items():
                outcome = Outcome(id=oid, course_id=cid, code=oid,
                                  name=o["name"], blurb=o["blurb"], strand=o["strand"])
                db.add(outcome)
                for concept in o["concepts"]:
                    db.add(Concept(outcome_id=oid, **concept))
                for q in o["questions"]:
                    db.add(Question(outcome_id=oid, **q))
        db.commit()


# ═════════════════════════════════════════════════════════════════════════════
#  INGEST + LLM
# ═════════════════════════════════════════════════════════════════════════════
def extract_text(path: Path, ext: str) -> str:
    ext = ext.lower().lstrip(".")
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        parts = []
        for i, page in enumerate(reader.pages):
            try: t = page.extract_text() or ""
            except Exception: t = ""
            if t.strip(): parts.append(f"[p.{i+1}]\n{t}")
        return "\n\n".join(parts)
    if ext == "docx":
        from docx import Document
        doc = Document(str(path))
        return "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    if ext in {"txt", "md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"unsupported .{ext}")


def chunk_text(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text.replace("\r", "")) if p.strip()]
    if not paras: return []
    TGT, OVL = 500, 60
    tok = lambda s: max(1, len(s) // 4)
    out, buf, bt = [], [], 0
    def flush():
        nonlocal buf, bt
        if not buf: return
        out.append("\n\n".join(buf))
        tail, tt = [], 0
        for p in reversed(buf):
            pt = tok(p)
            if tt + pt > OVL: break
            tail.insert(0, p); tt += pt
        buf = tail; bt = tt
    for p in paras:
        pt = tok(p)
        if bt + pt > TGT and buf: flush()
        buf.append(p); bt += pt
    if buf and (bt > OVL or not out): flush()
    return out


async def ollama_alive() -> tuple[bool, bool, bool]:
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{OLLAMA_HOST}/api/tags")
            if r.status_code != 200: return False, False, False
            names_full = {m.get("name", "") for m in r.json().get("models", [])}
            names_base = {n.split(":")[0] for n in names_full}
            chat_ok = OLLAMA_CHAT_MODEL in names_full or OLLAMA_CHAT_MODEL.split(":")[0] in names_base
            emb_ok = OLLAMA_EMBED_MODEL in names_full or OLLAMA_EMBED_MODEL.split(":")[0] in names_base
            return True, chat_ok, emb_ok
    except Exception:
        return False, False, False


async def ollama_chat(system: str, messages: list[dict]) -> str:
    body = {"model": OLLAMA_CHAT_MODEL,
            "messages": [{"role":"system","content":system}, *messages],
            "stream": False, "options":{"temperature":0.5}}
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(f"{OLLAMA_HOST}/api/chat", json=body)
        r.raise_for_status()
        return r.json()["message"]["content"]


async def ollama_embed(texts: list[str]) -> list[list[float]]:
    out = []
    async with httpx.AsyncClient(timeout=60.0) as c:
        for t in texts:
            r = await c.post(f"{OLLAMA_HOST}/api/embeddings",
                             json={"model":OLLAMA_EMBED_MODEL, "prompt":t})
            r.raise_for_status()
            out.append(r.json()["embedding"])
    return out


def build_system_prompt(level: str, mode: str, outcome_code: str, outcome_name: str,
                        course_label: str, grade: int, retrieved: list[dict]) -> str:
    # Voice per level — vocabulary and analogies calibrated, never patronising.
    if level == "elementary":
        voice = (f"You are a warm, patient tutor for a Grade {grade} student (about age {grade+5}). "
                 "Use short simple sentences. Reach for concrete analogies they can picture — pizza slices, "
                 "bags of apples, jumping. Never talk down. Encourage effort, not intelligence.")
    elif level == "secondary":
        voice = (f"You are a clear, respectful tutor for a Grade {grade} high-school student. "
                 "Use standard vocabulary and define new terms as you introduce them. Show reasoning steps "
                 "explicitly. Treat the student as capable.")
    else:
        voice = ("You are a precise, formal tutor for an undergraduate. Use standard technical vocabulary "
                 "and formal notation. Prefer definitions and worked examples over folksy analogies. "
                 "Be terse where terseness is clearer.")

    # Core principles embedded into every response — the 10 qualities of a good tutor.
    principles = (
        "Follow these principles on every turn:\n"
        "• DIAGNOSE FIRST. Before you explain, briefly probe what the student already knows or where they got "
        "  stuck. Ask ONE targeted question first, then respond. Never assume the gap.\n"
        "• EXPLAIN THE WHY, not just the how. If you give a step, say why that step is the right move.\n"
        "• SWITCH REGISTERS IF NEEDED. If one explanation doesn't land, offer a different angle — an analogy, "
        "  a worked example, a picture-in-words, or a first-principles derivation.\n"
        "• BE PATIENT WITH REPETITION. If the student asks something you already covered, re-explain it "
        "  without any hint of annoyance and try a different framing this time.\n"
        "• PLAN, BUT FLEX. Have a rough shape for the answer, but drop it instantly if the student surfaces "
        "  a more fundamental confusion — chase THAT down first.\n"
        "• ADMIT UNCERTAINTY. If you're not sure, say so and reason it out loud. Never bluff.\n"
        "• PRACTICE OUTSIDE THE TURN. When it fits, suggest a specific practice question or exercise the "
        "  student can try between messages — the best learning compounds outside the session.\n"
        "• SHOW MULTIPLICATION AS REPEATED ADDING. Any time you multiply, write it out: '5 × 3 = 3 + 3 + 3 + 3 + 3 = 15'. "
        "  This is non-optional for elementary students.\n"
        "• YOU CAN INCLUDE PICTURES. The interface AUTOMATICALLY generates a diagram whenever your message "
        "  contains an expression like '5 × 3', '3 groups of 4 apples', '3/4', or '12 + 15'. Use these forms "
        "  naturally in your explanations and a picture will appear beside your text — never say 'I can't show "
        "  pictures'. You can.\n"
    )

    if mode == "practice":
        rules = ("Practice mode: NEVER just hand over the final answer. Guide with the smallest possible "
                 "hint first, then a bigger one only if asked, then a walk-through only when the student "
                 "confirms they've tried. The point is the process, not the answer.")
    else:
        rules = ("Homework help mode: the student is under time pressure. Show the reasoning steps clearly "
                 "and quickly, but STILL show HOW the answer follows step-by-step. Never a bare answer. "
                 "Flag the one common mistake for this kind of problem as you go.")

    ctx = ""
    if retrieved:
        ctx = "\n\nUploaded material (cite as [title] when you use it):\n" + \
              "\n\n".join(f"[{r['title']}]  {r['text']}" for r in retrieved)
    return (f"{voice}\n\n{principles}\n{rules}\n\n"
            f"Current outcome: {outcome_code} — {outcome_name}\nCourse: {course_label}{ctx}")


FALLBACK_REPLY = {
    "practice": ("I'm in scripted mode — no AI is running. Try the hints on any practice question, "
                 "or read the Concept Explainer. To turn me on: install Ollama and run "
                 "'ollama pull llama3.2:3b'."),
    "homework": ("Homework mode needs the AI. Install Ollama and pull llama3.2:3b, then I can "
                 "walk through reasoning directly."),
}


async def chat_reply(system: str, messages: list[dict], mode: str) -> tuple[str, str]:
    alive, chat_ok, _ = await ollama_alive()
    if alive and chat_ok:
        try:
            return await ollama_chat(system, messages), "ollama"
        except Exception:
            pass
    return FALLBACK_REPLY[mode], "fallback"


def cosine_sim(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return Bn @ an


async def retrieve(db: Session, query: str, outcome_id: str | None, top_k: int = 3) -> list[dict]:
    rows = db.query(Chunk).filter((Chunk.outcome_id == outcome_id) | (Chunk.outcome_id.is_(None))).all()
    if not rows: return []
    if any(c.embedding for c in rows):
        try:
            q_vec = (await ollama_embed([query]))[0]
            q_arr = np.array(q_vec, dtype=np.float32)
            usable = [c for c in rows if c.embedding]
            M = np.array([c.embedding for c in usable], dtype=np.float32)
            sims = cosine_sim(q_arr, M)
            order = np.argsort(-sims)[:top_k]
            hits = []
            for idx in order:
                s = float(sims[idx])
                if s < 0.30: continue
                c = usable[idx]
                hits.append({"chunk_id":c.id, "source_id":c.source_id,
                             "title":c.source.title, "text":c.text, "similarity":s})
            if hits: return hits
        except Exception:
            pass
    terms = {t.lower() for t in re.findall(r"[a-zA-Z0-9]{3,}", query)}
    if not terms: return []
    scored = []
    for c in rows:
        toks = {t.lower() for t in re.findall(r"[a-zA-Z0-9]{3,}", c.text)}
        if not toks: continue
        ov = len(terms & toks) / max(1, len(terms))
        if ov > 0: scored.append((ov, c))
    scored.sort(key=lambda x: -x[0])
    return [{"chunk_id":c.id,"source_id":c.source_id,"title":c.source.title,
             "text":c.text,"similarity":float(s)} for s, c in scored[:top_k]]


# ═════════════════════════════════════════════════════════════════════════════
#  FASTAPI
# ═════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="Project X")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class MsgIn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatIn(BaseModel):
    outcome_id: str
    mode: Literal["practice", "homework"] = "practice"
    level: Literal["elementary", "secondary", "uni"] = "elementary"
    messages: list[MsgIn]

class SubmitIn(BaseModel):
    outcome_id: str
    question_id: str
    answer: str
    hints_used: int = 0
    mode: Literal["practice", "flashcard", "rapid", "blitz"] = "practice"

class PasteIn(BaseModel):
    title: str
    text: str
    outcome_id: str | None = None


def db_dep():
    d = SessionLocal()
    try: yield d
    finally: d.close()


@app.on_event("startup")
def _startup(): seed_if_empty()

@app.get("/")
def index(): return HTMLResponse(HTML_PAGE)

@app.get("/api/health")
async def health(db: Session = Depends(db_dep)):
    alive, chat_ok, emb_ok = await ollama_alive()
    return {"status":"ok","ollama_reachable":alive,"chat_model_ready":chat_ok,
            "embed_model_ready":emb_ok,"sources_indexed":db.query(Source).count(),
            "chunks_indexed":db.query(Chunk).count()}

@app.get("/api/courses")
def courses(db: Session = Depends(db_dep)):
    out = []
    for c in db.query(Course).order_by(Course.grade, Course.subject).all():
        outs = []
        for o in c.outcomes:
            outs.append({
                "id":o.id, "code":o.code, "name":o.name, "blurb":o.blurb, "strand":o.strand,
                "concepts":[{"id":co.id,"name":co.name,"sub":co.sub,"one_liner":co.one_liner,
                             "body":co.body,"worked":co.worked,"misconception":co.misconception} for co in o.concepts],
                "questions":[{"id":q.id,"kind":q.kind,"diff":q.diff,"prompt":q.prompt,
                              "choices":q.choices,"hints":q.hints,"answer":q.answer,
                              "solution":q.solution} for q in o.questions],
            })
        out.append({"id":c.id, "label":c.label, "level":c.level, "grade":c.grade,
                    "subject":c.subject, "outcomes":outs})
    return out

@app.get("/api/conversations/{outcome_id}")
def get_conv(outcome_id: str): return CONVERSATIONS.get(outcome_id, [])

@app.post("/api/chat")
async def chat(payload: ChatIn, db: Session = Depends(db_dep)):
    outcome = db.get(Outcome, payload.outcome_id)
    if not outcome: raise HTTPException(404, "outcome not found")
    course = db.get(Course, outcome.course_id)
    last = next((m.content for m in reversed(payload.messages) if m.role == "user"), "")
    retrieved = await retrieve(db, last, outcome_id=payload.outcome_id, top_k=3) if last else []
    system = build_system_prompt(payload.level, payload.mode, outcome.code, outcome.name,
                                 course.label, grade=course.grade, retrieved=retrieved)
    reply, provider = await chat_reply(system, [m.model_dump() for m in payload.messages], payload.mode)
    db.add(LogEvent(kind="chat", detail=last[:120] if last else "", outcome_id=payload.outcome_id))
    db.commit()
    return {"reply":reply,"provider":provider,
            "citations":[{"source_id":r["source_id"],"title":r["title"],"chunk_id":r["chunk_id"],
                          "excerpt":r["text"][:280],"similarity":r["similarity"]} for r in retrieved]}


def update_mastery(db: Session, outcome_id: str, correct: bool, diff: int) -> float:
    m = db.get(Mastery, outcome_id)
    if not m:
        m = Mastery(outcome_id=outcome_id, score=0.0, attempts=0, correct=0); db.add(m); db.flush()
    if m.score is None: m.score = 0.0
    if m.attempts is None: m.attempts = 0
    if m.correct is None: m.correct = 0
    target = 1.0 if correct else 0.0
    w = (0.5 + diff / 5.0) if correct else (1.5 - diff / 5.0)
    m.score = max(0.0, min(1.0, m.score + 0.20 * w * (target - m.score)))
    m.attempts += 1
    if correct: m.correct += 1
    m.updated_at = datetime.utcnow()
    db.commit()
    return m.score


@app.post("/api/attempts")
def submit_attempt(payload: SubmitIn, db: Session = Depends(db_dep)):
    q = db.get(Question, payload.question_id)
    if not q or q.outcome_id != payload.outcome_id:
        raise HTTPException(404, "question not found")
    correct = False
    if q.kind == "mc":
        try: correct = str(int(payload.answer)) == q.answer
        except ValueError: correct = False
    else:
        try:
            v = float(payload.answer.strip()); tgt = float(q.answer); tol = q.tolerance or 0.001
            correct = abs(v - tgt) <= tol
        except Exception: correct = False
    db.add(Attempt(outcome_id=payload.outcome_id, question_id=payload.question_id,
                   mode=payload.mode, correct=correct, hints_used=payload.hints_used))
    new_score = update_mastery(db, payload.outcome_id, correct, q.diff)
    db.add(LogEvent(kind=payload.mode,
                    detail=f"{q.id} · {'correct' if correct else 'wrong'} · {payload.hints_used} hints",
                    outcome_id=payload.outcome_id))
    db.commit()
    return {"correct":correct, "solution":q.solution, "mastery":new_score, "answer":q.answer}


@app.get("/api/mastery")
def get_mastery(db: Session = Depends(db_dep)):
    return [{"outcome_id":m.outcome_id,"score":m.score,"attempts":m.attempts,"correct":m.correct}
            for m in db.query(Mastery).all()]


@app.get("/api/sources")
def list_sources(db: Session = Depends(db_dep)):
    return [{"id":s.id,"title":s.title,"kind":s.kind,"bytes_size":s.bytes_size,
             "chunks_n":s.chunks_n,"outcome_id":s.outcome_id,"created_at":s.created_at.isoformat()}
            for s in db.query(Source).order_by(Source.created_at.desc()).all()]


async def _persist_and_embed(db: Session, title: str, kind: str, text: str,
                             outcome_id: str | None, bytes_size: int) -> Source:
    if not text.strip(): raise HTTPException(400, "no readable text")
    src = Source(title=title, kind=kind, outcome_id=outcome_id, bytes_size=bytes_size)
    db.add(src); db.flush()
    pieces = chunk_text(text)
    chunks = [Chunk(source_id=src.id, outcome_id=outcome_id, ord=i, text=p) for i, p in enumerate(pieces)]
    db.add_all(chunks)
    src.chunks_n = len(chunks); db.commit()
    try:
        vecs = await ollama_embed([c.text for c in chunks])
        for c, v in zip(chunks, vecs): c.embedding = v
        db.commit()
    except Exception:
        pass
    return src


@app.post("/api/sources/paste")
async def paste_source(payload: PasteIn, db: Session = Depends(db_dep)):
    src = await _persist_and_embed(db, title=payload.title or "Pasted note", kind="paste",
                                   text=payload.text, outcome_id=payload.outcome_id,
                                   bytes_size=len(payload.text.encode("utf-8")))
    db.add(LogEvent(kind="teach", detail=f"pasted · {src.title} · {src.chunks_n} chunks",
                    outcome_id=payload.outcome_id))
    db.commit()
    return {"id":src.id,"title":src.title,"kind":src.kind,"bytes_size":src.bytes_size,
            "chunks_n":src.chunks_n,"outcome_id":src.outcome_id,"created_at":src.created_at.isoformat()}


@app.post("/api/sources/upload")
async def upload_source(file: UploadFile = File(...), outcome_id: str | None = Form(None),
                        db: Session = Depends(db_dep)):
    if not file.filename: raise HTTPException(400, "no file")
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in {"pdf", "docx", "txt", "md"}:
        raise HTTPException(400, f"can't handle .{ext}")
    dest = UPLOAD_DIR / f"{int(datetime.utcnow().timestamp())}_{file.filename}"
    with dest.open("wb") as f: shutil.copyfileobj(file.file, f)
    try: text = extract_text(dest, ext)
    except Exception as e: raise HTTPException(400, f"couldn't read: {e}") from e
    src = await _persist_and_embed(db, title=file.filename, kind=ext, text=text,
                                   outcome_id=outcome_id, bytes_size=dest.stat().st_size)
    db.add(LogEvent(kind="teach", detail=f"uploaded · {file.filename} · {src.chunks_n} chunks",
                    outcome_id=outcome_id))
    db.commit()
    return {"id":src.id,"title":src.title,"kind":src.kind,"bytes_size":src.bytes_size,
            "chunks_n":src.chunks_n,"outcome_id":src.outcome_id,"created_at":src.created_at.isoformat()}


@app.delete("/api/sources/{sid}")
def delete_source(sid: int, db: Session = Depends(db_dep)):
    s = db.get(Source, sid)
    if not s: raise HTTPException(404, "not found")
    db.delete(s); db.commit()
    return {"ok": True}


@app.get("/api/log")
def get_log(limit: int = 200, db: Session = Depends(db_dep)):
    rows = db.query(LogEvent).order_by(LogEvent.created_at.desc()).limit(limit).all()
    return [{"id":r.id,"kind":r.kind,"detail":r.detail,"outcome_id":r.outcome_id,
             "created_at":r.created_at.isoformat()} for r in rows]


@app.delete("/api/log")
def clear_log(db: Session = Depends(db_dep)):
    db.query(LogEvent).delete(); db.commit(); return {"ok": True}


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Project X · Tutor</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Fredoka:wght@400;500;600;700&family=Nunito:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
  /* Level = elementary | secondary | uni. Toggled via data-level on <html>. */
  :root {
    --bg:#F7F4EC; --surface:#FDFBF5; --surface-2:#F1ECDD; --inset:#EDE8DA;
    --ink:#14171F; --ink-strong:#050609; --ink-muted:#55564F; --ink-faint:#8B8578;
    --rule:#D8D1BE; --rule-strong:#B8B0A0;
    --accent:#B84A1C; --accent-ink:#7A2E10; --accent-strong:#963A14;
    --accent-soft:rgba(184,74,28,0.10); --accent-line:rgba(184,74,28,0.32);
    --sky:#3F79A6; --sky-soft:rgba(63,121,166,0.10);
    --good:#3E8A4A; --warn:#C08A22; --risk:#B94434; --star:#E7A73B;
    --serif:'Fraunces','Iowan Old Style',Georgia,serif;
    --sans:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
    --mono:'JetBrains Mono','SF Mono',ui-monospace,Menlo,Consolas,monospace;
    --base:15.5px; --btn-r:8px; --card-r:10px; --btn-py:10px; --btn-px:18px;
    /* subject colours (Alloprof-flavoured) */
    --sub-math:#F4D03F; --sub-french:#5FB3B3; --sub-english:#4A90E2;
    --sub-science:#6DBE45; --sub-chemistry:#52B788; --sub-physics:#A3E048;
    --sub-cs:#9B59B6;
    color-scheme:light;
  }
  /* All modes share the same age-neutral palette. Only sizing varies. */
  :root[data-level="elementary"] {
    --base:16px; --btn-r:10px; --card-r:12px; --btn-py:12px; --btn-px:20px;
  }
  :root[data-level="uni"] {
    --base:14px; --btn-r:6px; --card-r:8px; --btn-py:8px; --btn-px:14px;
  }
  :root[data-theme="dark"] {
    --bg:#12100A; --surface:#1B1811; --surface-2:#221E15; --inset:#221E15;
    --ink:#EFE7D0; --ink-strong:#FBF5DF; --ink-muted:#A79E85; --ink-faint:#6A6350;
    --rule:#302818; --rule-strong:#45392A;
    --accent:#E68150; --accent-ink:#F4A87D;
    --accent-soft:rgba(230,129,80,0.13); --accent-line:rgba(230,129,80,0.35);
    --sky:#7CA9CE; --sky-soft:rgba(124,169,206,0.13);
    --good:#7BC48B; --warn:#E1AF57; --risk:#E68383;
  }

  *{box-sizing:border-box}
  html,body{height:100%;margin:0}
  body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:var(--base);line-height:1.55;-webkit-font-smoothing:antialiased;overflow:hidden}
  ::selection{background:var(--accent-soft);color:var(--accent-ink)}
  button{font:inherit;color:inherit}
  input,select,textarea{font:inherit;color:inherit}

  /* ═══════════════ LANDING (grade + subject picker) — sleek, lives inside stage ═══════════════ */
  .landing-inner{max-width:820px}
  .landing-hero{padding:0 0 30px;border-bottom:1px solid var(--rule);margin-bottom:32px}
  .landing-hero h1{font-family:var(--serif);font-weight:500;font-size:clamp(38px,5.4vw,60px);line-height:1;color:var(--ink-strong);margin:0 0 8px;letter-spacing:-0.028em}
  .landing-hero h1 em{font-style:italic;color:var(--accent);font-weight:400}
  .landing-hero p{font-family:var(--serif);font-style:italic;color:var(--ink-muted);margin:0;font-size:16px}

  .landing-section{margin-bottom:32px}
  .landing-section h2{font-family:var(--serif);font-weight:500;font-size:clamp(20px,2.4vw,26px);margin:0 0 16px;color:var(--ink-strong);letter-spacing:-0.015em}
  .landing-section h2 .num{font-family:var(--mono);font-size:11px;color:var(--accent);letter-spacing:0.14em;text-transform:uppercase;font-weight:600;margin-right:12px;vertical-align:5px}

  .grade-grid{display:flex;flex-wrap:wrap;gap:8px}
  .grade-pill{background:var(--surface);border:1.5px solid var(--rule);border-radius:999px;padding:9px 18px;font-family:var(--sans);font-weight:600;font-size:14px;color:var(--ink);cursor:pointer;transition:all 120ms ease}
  .grade-pill:hover{border-color:var(--accent-line);color:var(--accent-ink);background:var(--accent-soft)}
  .grade-pill.picked{background:var(--accent);color:#fff;border-color:var(--accent)}
  .grade-group{margin-bottom:14px}
  .grade-group-label{font-family:var(--mono);font-size:10.5px;letter-spacing:0.14em;text-transform:uppercase;color:var(--ink-faint);margin-bottom:10px;font-weight:600}

  .subject-grid{display:flex;flex-wrap:wrap;gap:8px}
  .subject-pill{background:var(--surface);border:1.5px solid var(--rule);border-radius:999px;padding:9px 18px 9px 14px;font-family:var(--sans);font-weight:600;font-size:14px;color:var(--ink);cursor:pointer;display:inline-flex;align-items:center;gap:10px;transition:all 120ms}
  .subject-pill::before{content:"";width:9px;height:9px;border-radius:50%;background:var(--sub-color,var(--accent))}
  .subject-pill:hover{border-color:var(--accent-line);color:var(--accent-ink)}
  .subject-pill.picked{background:var(--accent-soft);border-color:var(--accent);color:var(--accent-ink)}
  .subject-pill[data-sub="math"]{--sub-color:#D4A017}
  .subject-pill[data-sub="french"]{--sub-color:#3F8A88}
  .subject-pill[data-sub="english"]{--sub-color:#3A6EA5}
  .subject-pill[data-sub="science"]{--sub-color:#5CA146}
  .subject-pill[data-sub="chemistry"]{--sub-color:#3E9670}
  .subject-pill[data-sub="physics"]{--sub-color:#7DA82E}
  .subject-pill[data-sub="cs"]{--sub-color:#7B4A9E}
  .subject-pill[data-sub="hpe"]{--sub-color:#D97D6E}
  .subject-pill[data-sub="social"]{--sub-color:#B87D3E}
  .subject-pill[data-sub="history"]{--sub-color:#8E5A3B}
  .subject-pill[data-sub="geography"]{--sub-color:#4C8FA3}
  .subject-pill[data-sub="business"]{--sub-color:#5E7A4E}
  .subject-pill[data-sub="arts"]{--sub-color:#C1568E}
  .subject-pill[data-sub="civics"]{--sub-color:#6C5C9E}

  .landing-cta{margin-top:20px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
  .btn-big{background:var(--accent);color:#fff;border:none;border-radius:8px;padding:14px 28px;font-family:var(--sans);font-weight:700;font-size:15px;cursor:pointer;letter-spacing:0.01em;transition:background 120ms}
  .btn-big:hover{background:var(--accent-ink)}
  .btn-big:disabled{opacity:0.35;cursor:not-allowed}
  .landing-cta .picked-summary{font-family:var(--mono);font-size:12px;color:var(--ink-muted);letter-spacing:0.04em}

  /* Tutor philosophy list */
  .philosophy{padding-top:36px;margin-top:44px;border-top:1px solid var(--rule)}
  .philosophy .lead{font-family:var(--serif);font-style:italic;font-size:17px;color:var(--ink-muted);max-width:640px;margin:0 0 26px;line-height:1.55}
  .tutor-checklist{counter-reset:tc;list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(2,1fr);gap:26px 40px}
  @media(max-width:720px){.tutor-checklist{grid-template-columns:1fr}}
  .tutor-checklist li{position:relative;padding-left:44px;counter-increment:tc}
  .tutor-checklist li::before{content:counter(tc,decimal-leading-zero);position:absolute;left:0;top:2px;font-family:var(--mono);font-size:12px;color:var(--accent);letter-spacing:0.02em;font-weight:600;width:30px;text-align:right}
  .tutor-checklist h3{font-family:var(--serif);font-weight:600;font-size:16.5px;color:var(--ink-strong);margin:0 0 5px;letter-spacing:-0.005em}
  .tutor-checklist p{font-size:13.5px;color:var(--ink-muted);margin:0;line-height:1.55}
  .tutor-checklist em{font-style:italic;color:var(--accent-ink);font-weight:500;font-style:italic}

  /* ═══════════════ MAIN APP ═══════════════ */
  .app{display:flex;flex-direction:column;height:100vh}
  .topbar{display:flex;align-items:center;justify-content:space-between;padding:0 22px;height:60px;border-bottom:1px solid var(--rule);background:var(--surface);flex-shrink:0;gap:16px;flex-wrap:wrap}
  .brand{display:flex;align-items:center;gap:12px;flex-shrink:0}
  .brand-mark{width:34px;height:34px;border-radius:var(--btn-r);background:var(--accent);color:#fff;display:grid;place-items:center;font-family:var(--serif);font-weight:700;font-size:18px}
  .brand-name{font-family:var(--serif);font-weight:700;font-size:20px;letter-spacing:-0.01em;color:var(--ink-strong)}
  .crumb{font-family:var(--mono);font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:var(--ink-faint);padding:6px 10px;background:var(--inset);border-radius:var(--btn-r)}
  .crumb b{color:var(--accent-ink);font-weight:600}
  .top-actions{display:flex;align-items:center;gap:10px}
  .status{display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border-radius:999px;font-family:var(--sans);font-size:11.5px;font-weight:600}
  .status.ok{color:var(--good);background:rgba(62,138,74,0.10)}
  .status.warn{color:var(--warn);background:rgba(192,138,34,0.10)}
  .status .dot{width:6px;height:6px;border-radius:50%;background:currentColor}
  .status.ok .dot{box-shadow:0 0 6px currentColor}
  .icon-btn{background:transparent;border:1px solid var(--rule);color:var(--ink-muted);width:34px;height:34px;border-radius:var(--btn-r);cursor:pointer;display:grid;place-items:center;padding:0}
  .icon-btn:hover{color:var(--accent);border-color:var(--accent-line)}
  .icon-btn svg{width:14px;height:14px}

  .workspace{display:flex;flex:1;overflow:hidden}
  .rail{width:224px;flex-shrink:0;background:var(--surface);border-right:1px solid var(--rule);display:flex;flex-direction:column;overflow:hidden}
  .rail-nav{flex:1;overflow-y:auto;padding:18px 12px}
  .rail-group{margin-bottom:20px}
  .rail-group-label{font-family:var(--mono);font-size:10px;letter-spacing:0.16em;text-transform:uppercase;color:var(--ink-faint);padding:0 12px 8px;font-weight:600}
  .rail-item{display:block;width:100%;text-align:left;background:transparent;border:none;cursor:pointer;padding:10px 12px;border-radius:var(--btn-r);color:var(--ink);font-size:14px;font-weight:600;margin-bottom:2px}
  .rail-item:hover{background:var(--surface-2)}
  .rail-item.active{color:var(--accent-ink);background:var(--accent-soft);border-left:3px solid var(--accent);padding-left:9px}
  .rail-item .sub{display:block;font-size:11px;color:var(--ink-faint);font-family:var(--mono);letter-spacing:0.02em;margin-top:2px;font-weight:400}
  .rail-item.active .sub{color:var(--accent);opacity:0.75}

  .stage{flex:1;overflow-y:auto}
  .stage-inner{max-width:960px;padding:32px 40px 80px;margin:0 auto}

  /* Strand chips */
  .strand-chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:22px}
  .strand-chip{padding:6px 14px;border-radius:999px;background:var(--surface);border:1.5px solid var(--rule);color:var(--ink-muted);font-family:var(--sans);font-size:12.5px;font-weight:600;cursor:pointer}
  .strand-chip.on{background:var(--accent-soft);color:var(--accent-ink);border-color:var(--accent-line)}
  .strand-chip:hover{border-color:var(--accent-line)}

  /* Outcome picker (in-panel) */
  .outcome-select-row{display:flex;align-items:center;gap:8px;margin-bottom:18px;font-family:var(--mono);font-size:11px;color:var(--ink-faint);letter-spacing:0.08em;text-transform:uppercase}
  .outcome-select-row select{background:var(--inset);border:1px solid var(--rule);color:var(--ink);border-radius:var(--btn-r);padding:6px 12px;font-family:var(--sans);font-size:13px;font-weight:500;cursor:pointer;flex:1;max-width:520px}

  .panel-head{margin-bottom:26px}
  .panel-eye{font-family:var(--mono);font-size:11px;letter-spacing:0.16em;text-transform:uppercase;color:var(--accent);margin-bottom:10px;font-weight:600}
  .panel-title{font-family:var(--serif);font-weight:700;font-size:clamp(26px,3.2vw,38px);line-height:1.05;letter-spacing:-0.02em;color:var(--ink-strong);margin:0 0 10px}
  .panel-title em{font-style:italic;color:var(--accent);font-weight:500}
  .panel-purpose{font-family:var(--serif);font-weight:400;font-size:16px;line-height:1.5;color:var(--ink-muted);max-width:620px;margin:0}

  .card{background:var(--surface);border:1px solid var(--rule);border-radius:var(--card-r);padding:20px 22px}
  .card + .card{margin-top:12px}
  .card-title{font-family:var(--serif);font-weight:600;font-size:16px;color:var(--ink-strong);margin:0 0 6px}
  .card-sub{font-family:var(--mono);font-size:10.5px;letter-spacing:0.10em;text-transform:uppercase;color:var(--ink-faint);margin-bottom:12px;font-weight:600}

  .btn{background:var(--accent);color:#fff;border:none;border-radius:var(--btn-r);padding:var(--btn-py) var(--btn-px);font-family:var(--sans);font-size:13.5px;font-weight:700;cursor:pointer}
  .btn:hover{background:var(--accent-strong)}
  .btn:disabled{opacity:0.35;cursor:not-allowed}
  .btn.ghost{background:transparent;color:var(--ink-muted);border:1.5px solid var(--rule)}
  .btn.ghost:hover{color:var(--accent);border-color:var(--accent)}
  .btn.big{padding:calc(var(--btn-py) + 2px) calc(var(--btn-px) + 4px);font-size:14.5px}

  .field{display:flex;flex-direction:column;gap:6px}
  .field label{font-family:var(--mono);font-size:10.5px;letter-spacing:0.10em;text-transform:uppercase;color:var(--ink-faint);font-weight:600}
  .field input,.field textarea{background:var(--surface);border:1.5px solid var(--rule);color:var(--ink);border-radius:var(--btn-r);padding:10px 12px;font-family:var(--sans);font-size:14px}
  .field textarea{min-height:100px;resize:vertical}
  .field input:focus,.field textarea:focus{outline:none;border-color:var(--accent)}

  /* Chat */
  .chat-wrap{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:16px}
  @media(max-width:900px){.chat-wrap{grid-template-columns:1fr}}
  .chat-panel{background:var(--surface);border:1px solid var(--rule);border-radius:var(--card-r);display:flex;flex-direction:column;min-height:500px}
  .chat-mode{padding:12px 18px;border-bottom:1px solid var(--rule);display:flex;align-items:center;justify-content:space-between;gap:12px;background:var(--surface-2);border-radius:var(--card-r) var(--card-r) 0 0}
  .pill{padding:6px 14px;border-radius:var(--btn-r);font-family:var(--sans);font-size:12px;font-weight:600;background:transparent;border:1.5px solid transparent;color:var(--ink-muted);cursor:pointer}
  .pill.on{background:var(--accent-soft);color:var(--accent-ink);border-color:var(--accent-line)}
  .chat-log{flex:1;overflow-y:auto;padding:18px 20px;display:flex;flex-direction:column;gap:14px}
  .msg{display:flex;gap:12px;align-items:flex-start}
  .msg .avatar{width:32px;height:32px;border-radius:var(--btn-r);flex-shrink:0;display:grid;place-items:center;font-family:var(--sans);font-size:12px;font-weight:700}
  .msg.assistant .avatar{background:var(--accent);color:#fff}
  .msg.user .avatar{background:var(--sky);color:#fff}
  .msg .bubble{padding:12px 16px;border-radius:var(--btn-r);max-width:560px;line-height:1.6;font-size:14px;white-space:pre-wrap}
  .msg.assistant .bubble{background:var(--accent-soft);color:var(--ink);border:1px solid var(--accent-line)}
  .msg.user .bubble{background:var(--sky-soft);color:var(--ink);border:1px solid rgba(63,121,166,0.28)}
  .cites{margin-top:8px;display:flex;flex-wrap:wrap;gap:5px}
  .cite{font-family:var(--mono);font-size:10.5px;padding:3px 8px;border-radius:4px;background:var(--sky-soft);color:var(--sky);font-weight:500}
  .chat-in{padding:14px 16px;border-top:1px solid var(--rule);background:var(--surface);border-radius:0 0 var(--card-r) var(--card-r)}
  .suggs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
  .sugg{padding:7px 12px;background:var(--inset);border:1.5px solid var(--rule);border-radius:999px;font-size:12.5px;color:var(--ink);font-weight:600;cursor:pointer}
  .sugg:hover{color:var(--accent);border-color:var(--accent);background:var(--accent-soft)}
  .chat-row{display:flex;gap:8px}
  .chat-row input{flex:1;background:var(--inset);border:1.5px solid var(--rule);color:var(--ink);border-radius:var(--btn-r);padding:10px 14px;font-family:var(--sans);font-size:14px}
  .chat-side{display:flex;flex-direction:column;gap:12px}

  /* Explainer */
  .explain-wrap{display:grid;grid-template-columns:240px 1fr;gap:16px}
  @media(max-width:780px){.explain-wrap{grid-template-columns:1fr}}
  .concept-list{background:var(--surface);border:1px solid var(--rule);border-radius:var(--card-r);padding:8px;align-self:start}
  .concept-item{display:block;width:100%;text-align:left;padding:12px 14px;background:transparent;border:none;cursor:pointer;border-radius:var(--btn-r);color:var(--ink);font-size:14px;line-height:1.35;font-family:var(--serif);font-weight:600}
  .concept-item .sub{display:block;font-family:var(--mono);font-size:10.5px;letter-spacing:0.06em;color:var(--ink-faint);margin-top:3px;font-weight:400}
  .concept-item:hover{background:var(--surface-2)}
  .concept-item.on{background:var(--accent-soft);color:var(--accent-ink)}
  .concept-body{display:flex;flex-direction:column;gap:12px}
  .concept-body h3{font-family:var(--serif);font-weight:700;font-size:24px;line-height:1.15;color:var(--ink-strong);margin:0;letter-spacing:-0.01em}
  .cs{background:var(--surface);border:1px solid var(--rule);border-radius:var(--card-r);padding:20px 22px}
  .cs .lbl{font-family:var(--mono);font-size:10.5px;letter-spacing:0.14em;text-transform:uppercase;color:var(--accent);margin-bottom:8px;font-weight:600}
  .cs p{margin:0 0 10px;line-height:1.6;font-size:14.5px}
  .cs .worked{background:var(--inset);border-left:4px solid var(--accent);padding:12px 14px;margin:12px 0 0;border-radius:4px;font-family:var(--mono);font-size:12.5px;line-height:1.7;white-space:pre-wrap}
  .cs.one{background:var(--accent-soft);border-color:var(--accent-line)}
  .cs.one p{font-family:var(--serif);font-weight:500;font-size:19px;line-height:1.45;color:var(--ink-strong)}
  .cs.mis{border-left:4px solid var(--warn)}
  .cs.mis .lbl{color:var(--warn)}

  /* Question card (shared by practice + blitz) */
  .qcard{background:var(--surface);border:1px solid var(--rule);border-radius:var(--card-r);padding:24px 28px}
  .qmeta{display:flex;justify-content:space-between;align-items:center;font-family:var(--mono);font-size:11px;letter-spacing:0.08em;color:var(--ink-faint);margin-bottom:14px;text-transform:uppercase;font-weight:600}
  .diff{display:inline-flex;align-items:center;gap:5px}
  .dd{width:7px;height:7px;border-radius:50%;background:var(--rule-strong);display:inline-block}
  .dd.on{background:var(--accent)}
  .prompt{font-family:var(--serif);font-weight:500;font-size:20px;line-height:1.4;color:var(--ink-strong);margin:0 0 18px}
  .img-box{margin:0 0 20px;display:flex;justify-content:center;background:var(--inset);border-radius:var(--btn-r);padding:14px;overflow-x:auto}
  .img-box svg{max-width:100%;height:auto}
  .choices{display:flex;flex-direction:column;gap:9px}
  .choice{display:flex;align-items:center;gap:12px;padding:12px 14px;background:var(--surface);border:2px solid var(--rule);border-radius:var(--btn-r);cursor:pointer;font-size:14.5px;text-align:left;width:100%;font-weight:500}
  .choice:hover:not(:disabled){border-color:var(--accent-line);background:var(--accent-soft)}
  .choice.picked{border-color:var(--accent);background:var(--accent-soft)}
  .choice.right{border-color:var(--good);background:rgba(62,138,74,0.10)}
  .choice.wrong{border-color:var(--risk);background:rgba(185,68,52,0.08)}
  .choice .letter{width:26px;height:26px;border-radius:6px;background:var(--inset);border:1.5px solid var(--rule);display:grid;place-items:center;font-family:var(--mono);font-size:12px;font-weight:600}
  .choice.picked .letter{background:var(--accent);color:#fff;border-color:var(--accent)}
  .choice.right .letter{background:var(--good);color:#fff;border-color:var(--good)}
  .choice.wrong .letter{background:var(--risk);color:#fff;border-color:var(--risk)}
  .short input{width:100%;background:var(--surface);border:2px solid var(--rule);color:var(--ink);border-radius:var(--btn-r);padding:12px 14px;font-family:var(--mono);font-size:16px;font-weight:500}
  .short input:focus{outline:none;border-color:var(--accent)}
  .short input.right{border-color:var(--good)}
  .short input.wrong{border-color:var(--risk)}
  .qact{display:flex;gap:8px;margin-top:20px;padding-top:16px;border-top:1px solid var(--rule);justify-content:space-between;flex-wrap:wrap}
  .qact .g{display:flex;gap:8px}
  .hints{margin-top:16px;display:flex;flex-direction:column;gap:8px}
  .hint{padding:12px 16px;background:var(--accent-soft);border-left:4px solid var(--accent);border-radius:4px;font-size:14px;line-height:1.55}
  .hint .k{font-family:var(--mono);font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:var(--accent-ink);margin-bottom:4px;display:block;font-weight:600}
  .explain-box{margin-top:16px;padding:16px 18px;background:var(--sky-soft);border-left:4px solid var(--sky);border-radius:4px}
  .explain-box .k{font-family:var(--mono);font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;color:var(--sky);margin-bottom:8px;display:block;font-weight:600}
  .explain-box .title{font-family:var(--serif);font-weight:600;font-size:16px;color:var(--ink-strong);margin:0 0 6px}
  .explain-box .body{font-family:var(--mono);font-size:13px;line-height:1.7;white-space:pre-wrap;color:var(--ink)}
  .verdict{padding:12px 16px;border-radius:var(--btn-r);margin-top:12px;font-size:14.5px;font-weight:600}
  .verdict.right{background:rgba(62,138,74,0.12);color:var(--good);border-left:4px solid var(--good)}
  .verdict.wrong{background:rgba(185,68,52,0.10);color:var(--risk);border-left:4px solid var(--risk)}

  /* Flashcards */
  .flash-scene{perspective:1200px;margin:0 auto 20px;max-width:640px}
  .flash{position:relative;height:340px;transform-style:preserve-3d;transition:transform 500ms cubic-bezier(.3,.7,.4,1);cursor:pointer}
  .flash.flipped{transform:rotateY(180deg)}
  .flash-face{position:absolute;inset:0;-webkit-backface-visibility:hidden;backface-visibility:hidden;background:var(--surface);border:2px solid var(--rule);border-radius:20px;padding:30px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;box-shadow:0 6px 24px rgba(0,0,0,0.06)}
  .flash-face.back{transform:rotateY(180deg);background:var(--accent-soft);border-color:var(--accent-line)}
  .flash-face .face-label{font-family:var(--mono);font-size:11px;letter-spacing:0.16em;text-transform:uppercase;color:var(--ink-faint);margin-bottom:16px;font-weight:600}
  .flash-face.back .face-label{color:var(--accent)}
  .flash-face .face-text{font-family:var(--serif);font-weight:500;font-size:22px;line-height:1.4;color:var(--ink-strong);letter-spacing:-0.01em}
  .flash-face .face-explain{margin-top:20px;font-family:var(--sans);font-size:14px;color:var(--ink-muted);line-height:1.55;font-weight:400;max-width:520px}
  .flash-actions{display:flex;gap:12px;justify-content:center;margin-top:8px;flex-wrap:wrap}
  .flash-btn{border:2px solid var(--rule);background:var(--surface);padding:12px 22px;border-radius:var(--btn-r);font-family:var(--sans);font-size:14px;font-weight:700;cursor:pointer;color:var(--ink)}
  .flash-btn.learning{border-color:var(--warn);color:var(--warn)}
  .flash-btn.learning:hover{background:rgba(192,138,34,0.10)}
  .flash-btn.known{border-color:var(--good);color:var(--good)}
  .flash-btn.known:hover{background:rgba(62,138,74,0.10)}
  .flash-btn.flip{border-color:var(--accent);color:var(--accent)}
  .flash-btn.flip:hover{background:var(--accent-soft)}
  .flash-progress{text-align:center;font-family:var(--mono);font-size:11px;letter-spacing:0.08em;color:var(--ink-faint);margin-bottom:14px;text-transform:uppercase;font-weight:600}
  .flash-done{background:var(--surface);border:1px solid var(--rule);border-radius:var(--card-r);padding:32px;text-align:center;font-family:var(--serif);font-size:18px;color:var(--ink-strong)}

  /* Blitz */
  .blitz-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;padding:14px 20px;background:var(--surface);border:1px solid var(--rule);border-radius:var(--card-r)}
  .blitz-timer{font-family:var(--serif);font-weight:700;font-size:34px;color:var(--accent);letter-spacing:-0.02em;font-variant-numeric:tabular-nums}
  .blitz-timer.low{color:var(--risk)}
  .blitz-score{font-family:var(--serif);font-weight:700;font-size:22px;color:var(--ink-strong)}
  .blitz-score .u{font-family:var(--mono);font-size:11px;color:var(--ink-faint);letter-spacing:0.06em;text-transform:uppercase;margin-left:5px}
  .blitz-summary{background:var(--surface);border:1px solid var(--rule);border-radius:var(--card-r);padding:32px;text-align:center}
  .blitz-summary .big{font-family:var(--serif);font-weight:700;font-size:44px;color:var(--accent);margin-bottom:8px;letter-spacing:-0.02em}
  .blitz-summary .lbl{font-family:var(--mono);font-size:12px;letter-spacing:0.12em;text-transform:uppercase;color:var(--ink-faint);font-weight:600}

  /* Mastery */
  .msum{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
  @media(max-width:700px){.msum{grid-template-columns:repeat(2,1fr)}}
  .stat{background:var(--surface);border:1px solid var(--rule);border-radius:var(--card-r);padding:16px 18px}
  .stat .lbl{font-family:var(--mono);font-size:10.5px;letter-spacing:0.14em;text-transform:uppercase;color:var(--ink-faint);margin-bottom:6px;font-weight:600}
  .stat .val{font-family:var(--serif);font-weight:700;font-size:26px;color:var(--ink-strong);line-height:1;letter-spacing:-0.02em}
  .stat .val em{font-style:italic;color:var(--accent);font-weight:600}
  .stat .val .u{font-family:var(--mono);font-size:11px;color:var(--ink-faint);margin-left:4px;vertical-align:3px;letter-spacing:0.06em;text-transform:uppercase;font-weight:600}
  .mlist{background:var(--surface);border:1px solid var(--rule);border-radius:var(--card-r);overflow:hidden}
  .mrow{padding:16px 20px;border-bottom:1px solid var(--rule);display:grid;grid-template-columns:1fr 130px 80px;gap:18px;align-items:center}
  .mrow:last-child{border-bottom:none}
  .mrow .out-name{font-family:var(--serif);font-weight:600;font-size:15px;color:var(--ink-strong)}
  .mrow .out-code{font-family:var(--mono);font-size:11px;color:var(--accent);letter-spacing:0.06em;margin-bottom:3px;font-weight:600}
  .mrow .out-sub{font-family:var(--mono);font-size:10.5px;color:var(--ink-faint);margin-top:3px}
  .mstars{display:flex;gap:2px}
  .mstars svg{width:20px;height:20px;color:var(--rule-strong)}
  .mstars svg.on{color:var(--star)}
  .mrow .n{text-align:right;font-family:var(--serif);font-weight:700;font-size:20px;color:var(--ink-strong)}
  .mrow .n .u{font-family:var(--mono);font-size:11px;color:var(--ink-faint);letter-spacing:0.06em;text-transform:uppercase;margin-left:3px;vertical-align:3px;font-weight:600}

  /* Teach */
  .teach-wrap{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:900px){.teach-wrap{grid-template-columns:1fr}}
  .drop{border:3px dashed var(--rule-strong);border-radius:var(--card-r);padding:36px 24px;text-align:center;background:var(--surface);cursor:pointer}
  .drop:hover,.drop.hot{border-color:var(--accent);background:var(--accent-soft)}
  .drop .big{font-family:var(--serif);font-weight:600;font-size:18px;color:var(--ink-strong);margin:0 0 6px}
  .drop .small{font-family:var(--mono);font-size:11.5px;color:var(--ink-muted)}
  .slist{display:flex;flex-direction:column;gap:10px;margin-top:16px}
  .sitem{display:grid;grid-template-columns:1fr auto auto;gap:14px;padding:14px 16px;background:var(--surface);border:1px solid var(--rule);border-radius:var(--btn-r);align-items:center}
  .sitem .title{font-family:var(--serif);font-weight:600;color:var(--ink-strong);font-size:14px}
  .sitem .meta{font-family:var(--mono);font-size:10.5px;color:var(--ink-faint);margin-top:2px}
  .sitem .stats{font-family:var(--mono);font-size:11px;color:var(--ink-muted);text-align:right}
  .sitem .del{background:transparent;border:none;color:var(--ink-faint);cursor:pointer;padding:5px 8px;font-family:var(--sans);font-size:11.5px;font-weight:600}
  .sitem .del:hover{color:var(--risk)}

  /* Log */
  .llist{display:flex;flex-direction:column;gap:8px;margin-top:14px}
  .lrow{display:grid;grid-template-columns:88px 1fr auto;gap:14px;padding:12px 16px;background:var(--surface);border:1px solid var(--rule);border-radius:var(--btn-r);align-items:center}
  .lrow .when{font-family:var(--mono);font-size:10.5px;color:var(--accent);line-height:1.4;font-weight:600}
  .lrow .what{font-size:13.5px}
  .lrow .what .kind{font-family:var(--serif);font-weight:600;color:var(--ink-strong);font-size:14px;margin-right:8px}
  .lrow .what .det{color:var(--ink-muted)}
  .lrow .badge{font-family:var(--mono);font-size:10.5px;padding:3px 9px;border-radius:5px;background:var(--inset);color:var(--ink-muted);font-weight:600}

  .empty{color:var(--ink-faint);font-size:13.5px;padding:28px 0;text-align:center;font-family:var(--sans)}
  .loading{color:var(--ink-faint);font-family:var(--mono);font-size:12px;padding:18px 0}
  .error{padding:12px 14px;background:rgba(185,68,52,0.10);border-left:4px solid var(--risk);border-radius:5px;color:var(--risk);font-size:13px;margin:12px 0}

  @media(max-width:780px){
    .workspace{flex-direction:column}
    .rail{width:100%;height:auto;border-right:none;border-bottom:1px solid var(--rule)}
    .rail-nav{display:flex;overflow-x:auto;padding:10px 12px}
    .rail-group{margin:0 18px 0 0;display:flex;align-items:center;gap:4px}
    .rail-group-label{padding:0 8px 0 0;white-space:nowrap}
    .rail-item{white-space:nowrap;padding:6px 10px}
    .rail-item .sub{display:none}
    .stage-inner{padding:24px 20px 60px}
    .msum{grid-template-columns:repeat(2,1fr)}
    .mrow{grid-template-columns:1fr}
  }
</style>
</head>
<body>
<div id="app">Loading…</div>

<script>
// ═══════════════════ API ═══════════════════
const api = {
  health: () => fetch('/api/health').then(r => r.json()),
  courses: () => fetch('/api/courses').then(r => r.json()),
  conv: (oid) => fetch('/api/conversations/' + encodeURIComponent(oid)).then(r => r.json()),
  chat: (body) => fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}).then(async r => { if(!r.ok) throw new Error((await r.text()).slice(0,200)); return r.json(); }),
  submit: (body) => fetch('/api/attempts', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}).then(async r => { if(!r.ok) throw new Error((await r.text()).slice(0,200)); return r.json(); }),
  mastery: () => fetch('/api/mastery').then(r => r.json()),
  sources: () => fetch('/api/sources').then(r => r.json()),
  paste: (body) => fetch('/api/sources/paste', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}).then(async r => { if(!r.ok) throw new Error(await r.text()); return r.json(); }),
  upload: async (file, outcome_id) => { const fd = new FormData(); fd.append('file', file); if(outcome_id) fd.append('outcome_id', outcome_id); const r = await fetch('/api/sources/upload', {method:'POST', body:fd}); if(!r.ok) throw new Error(await r.text()); return r.json(); },
  del: (id) => fetch('/api/sources/'+id, {method:'DELETE'}).then(r => r.json()),
  log: () => fetch('/api/log').then(r => r.json()),
  clearLog: () => fetch('/api/log', {method:'DELETE'}).then(r => r.json()),
};

// ═══════════════════ STATE ═══════════════════
const state = {
  view: 'landing',           // 'landing' | 'app'
  courses: [], mastery: [], health: null,
  courseId: '', outcomeId: '', strand: null,
  active: 'tutor',
  // Landing picker state
  pickGrade: null, pickSubject: null,
  chat: {mode:'practice', messages:[], citations:[], provider:'', sending:false, input:'', conv:[], convCursor:0},
  explain: {conceptId:''},
  practice: {idx:0, picked:null, typed:'', hints:0, verdict:null, explainShown:false, correctAnswer:null},
  flash: {idx:0, side:'front', queue:[], known:[], learning:[]},
  rapid: {running:false, score:0, questionsAsked:0, current:null, picked:null, typed:'', verdict:null, correctAnswer:null},
  challenge: {running:false, timeLeft:60, score:0, questionsAsked:0, current:null, picked:null, typed:'', tid:null, verdict:null, correctAnswer:null},
  teach: {sources:[], attach:true, busy:false, err:null, msg:null, pt:'', px:'', hot:false},
  log: [],
};

// Restore
try {
  const s = localStorage.getItem('px:pick');
  if (s) { const p = JSON.parse(s); state.pickGrade = p.grade; state.pickSubject = p.subject; }
  if (localStorage.getItem('px:theme') === 'dark') document.documentElement.dataset.theme = 'dark';
  const v = localStorage.getItem('px:view');
  if (v === 'app' && state.pickGrade) state.view = 'app';
} catch(e){}

function currentCourse() {
  return state.courses.find(c => c.grade === state.pickGrade && c.subject === state.pickSubject) ||
         state.courses.find(c => c.id === state.courseId);
}
function currentOutcome() {
  const c = currentCourse(); if (!c) return null;
  return c.outcomes.find(o => o.id === state.outcomeId) || c.outcomes[0];
}
function outcomeMastery(oid) { const m = state.mastery.find(x => x.outcome_id === oid); return m ? m.score : 0; }
function levelOfGrade(g) { return g <= 8 ? 'elementary' : g <= 12 ? 'secondary' : 'uni'; }
function esc(s) { return (s||'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function starSvg(on) { return `<svg viewBox="0 0 24 24" fill="${on?'currentColor':'none'}" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="${on?'on':''}"><path d="M12 2l3 6 7 1-5 5 1 7-6-3-6 3 1-7-5-5 7-1z"/></svg>`; }
function stars(score) { const n = Math.round(score * 5); return [0,1,2,3,4].map(i => starSvg(i < n)).join(''); }
async function refreshMastery() { state.mastery = await api.mastery(); }

// ═══════════════════ IMAGE GENERATOR ═══════════════════
// Inspects a chunk of text and returns an SVG string when it recognises a
// visual pattern (multiplication, fractions, addition, etc.). Empty string = no image.
function drawFor(prompt) {
  if (!prompt) return '';
  const p = String(prompt);

  // Multiplication like "5 × 3" or "5 x 3" or "5*3"
  let m = p.match(/(\d+)\s*[×xX*]\s*(\d+)/);
  if (m) return drawGroups(+m[1], +m[2]);

  // "N groups of M" or "N bags of M" etc.
  m = p.match(/(\d+)\s+(?:bags?|groups?|sets?|boxes?|piles?|rows?|baskets?)\s+(?:of\s+|with\s+)?(?:.*?)(\d+)/i);
  if (m) return drawGroups(+m[1], +m[2]);

  // Simple fraction like "3/4" (proper fraction only, both under 20)
  m = p.match(/\b(\d+)\s*\/\s*(\d+)\b/);
  if (m && +m[2] > 0 && +m[2] <= 20 && +m[1] <= +m[2]) return drawFraction(+m[1], +m[2]);

  // Addition of small numbers "N + M" (both under 50)
  m = p.match(/(\d+)\s*\+\s*(\d+)/);
  if (m && +m[1] <= 30 && +m[2] <= 30) return drawAddition(+m[1], +m[2]);

  return '';
}

// Repeated-addition breakdown for multiplication.
// "5 × 3" → "5 × 3 = 3 + 3 + 3 + 3 + 3 = 15"
// Also catches word forms like "4 bags with 3 apples each".
function repeatedAddFor(prompt) {
  if (!prompt) return '';
  const p = String(prompt);

  let a, b, m;
  m = p.match(/(\d+)\s*[×xX*]\s*(\d+)/);
  if (m) { a = +m[1]; b = +m[2]; }
  else {
    m = p.match(/(\d+)\s+(?:bags?|groups?|sets?|boxes?|piles?|rows?|baskets?)\s+(?:of\s+|with\s+)?(?:.*?)(\d+)/i);
    if (m) { a = +m[1]; b = +m[2]; }
  }
  if (a == null || b == null) return '';
  if (a < 1 || b < 1 || a > 12 || b > 12) return '';

  // Show both directions: "5 threes" and "3 fives" — whichever has fewer terms first
  const aCopiesOfB = Array(a).fill(b).join(' + ');
  const bCopiesOfA = Array(b).fill(a).join(' + ');
  const total = a * b;
  return `${a} × ${b}  means  '${a} groups of ${b}':\n  ${a} × ${b}  =  ${aCopiesOfB}  =  ${total}\n\nOr the switch rule — '${b} groups of ${a}':\n  ${a} × ${b}  =  ${bCopiesOfA}  =  ${total}`;
}

// Compose the full "why" text: repeated-addition breakdown first (for
// multiplication questions), then the stored solution.
function fullExplanation(question) {
  const rep = repeatedAddFor(question.prompt);
  const sol = question.solution || '';
  return rep ? (rep + '\n\n' + sol) : sol;
}

function drawGroups(n, m) {
  // n groups of m circles. Cap for legibility.
  n = Math.min(n, 8); m = Math.min(m, 10);
  const dotR = 12, dotGap = 30, groupPadX = 20, groupPadY = 14;
  const cols = Math.min(m, 5);
  const rows = Math.ceil(m / cols);
  const groupW = cols * dotGap + groupPadX;
  const groupH = rows * dotGap + groupPadY + 12;
  const groupsPerRow = Math.min(n, 4);
  const groupRows = Math.ceil(n / groupsPerRow);
  const svgW = groupsPerRow * (groupW + 14) + 20;
  const svgH = groupRows * (groupH + 18) + 20;
  const colors = ['#EC6323','#38BB5C','#4A90E2','#F4D03F','#9B59B6','#E74C3C','#5FB3B3','#F39C12'];
  let out = `<svg viewBox="0 0 ${svgW} ${svgH}" xmlns="http://www.w3.org/2000/svg" style="max-width:${svgW}px">`;
  for (let g = 0; g < n; g++) {
    const gr = Math.floor(g / groupsPerRow), gc = g % groupsPerRow;
    const gx = 10 + gc * (groupW + 14), gy = 10 + gr * (groupH + 18);
    out += `<rect x="${gx}" y="${gy}" width="${groupW}" height="${groupH}" rx="12" fill="rgba(0,0,0,0.03)" stroke="rgba(0,0,0,0.10)" />`;
    const col = colors[g % colors.length];
    for (let i = 0; i < m; i++) {
      const r = Math.floor(i / cols), c = i % cols;
      const cx = gx + groupPadX/2 + dotGap/2 + c * dotGap;
      const cy = gy + groupPadY/2 + dotGap/2 + r * dotGap;
      out += `<circle cx="${cx}" cy="${cy}" r="${dotR}" fill="${col}" stroke="rgba(0,0,0,0.15)" />`;
    }
    out += `<text x="${gx + groupW/2}" y="${gy + groupH - 4}" text-anchor="middle" font-family="Nunito, sans-serif" font-size="11" font-weight="700" fill="rgba(0,0,0,0.55)">${m}</text>`;
  }
  return out + '</svg>';
}

function drawFraction(num, den) {
  // Draw a bar divided into `den` parts, `num` filled.
  const w = 480, h = 90, pad = 10;
  const bar = w - 2*pad;
  const seg = bar / den;
  let out = `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">`;
  out += `<rect x="${pad}" y="${pad}" width="${bar}" height="${h - 2*pad - 24}" fill="none" stroke="rgba(0,0,0,0.35)" stroke-width="2" rx="4"/>`;
  for (let i = 0; i < den; i++) {
    const x = pad + i * seg;
    const fill = i < num ? '#EC6323' : 'rgba(0,0,0,0.05)';
    out += `<rect x="${x + 1}" y="${pad + 1}" width="${seg - 2}" height="${h - 2*pad - 24 - 2}" fill="${fill}"/>`;
    if (i > 0) out += `<line x1="${x}" y1="${pad}" x2="${x}" y2="${h - pad - 24}" stroke="rgba(0,0,0,0.35)" stroke-width="1.5"/>`;
  }
  out += `<text x="${w/2}" y="${h - 8}" text-anchor="middle" font-family="Fraunces, serif" font-size="16" font-weight="600" fill="rgba(0,0,0,0.70)">${num}/${den}</text>`;
  return out + '</svg>';
}

function drawAddition(a, b) {
  // Two rows of dots — a orange, b blue. Cap at 30 each.
  a = Math.min(a, 30); b = Math.min(b, 30);
  const dotR = 10, gap = 24, cols = 10;
  const rowsA = Math.ceil(a / cols), rowsB = Math.ceil(b / cols);
  const w = cols * gap + 20;
  const h = (rowsA + rowsB) * gap + 44;
  let out = `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" style="max-width:${w}px">`;
  out += `<text x="10" y="16" font-family="Nunito, sans-serif" font-size="12" font-weight="700" fill="rgba(0,0,0,0.60)">${a}</text>`;
  for (let i = 0; i < a; i++) {
    const r = Math.floor(i / cols), c = i % cols;
    out += `<circle cx="${10 + gap/2 + c*gap}" cy="${28 + r*gap}" r="${dotR}" fill="#EC6323"/>`;
  }
  const yGap = rowsA * gap + 24;
  out += `<text x="10" y="${yGap + 4}" font-family="Nunito, sans-serif" font-size="12" font-weight="700" fill="rgba(0,0,0,0.60)">+ ${b}</text>`;
  for (let i = 0; i < b; i++) {
    const r = Math.floor(i / cols), c = i % cols;
    out += `<circle cx="${10 + gap/2 + c*gap}" cy="${yGap + 20 + r*gap}" r="${dotR}" fill="#4A90E2"/>`;
  }
  return out + '</svg>';
}

// ═══════════════════ RENDER ═══════════════════
function h(tag, attrs={}, children=[]) {
  const e = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'onClick') e.addEventListener('click', v);
    else if (k === 'onInput') e.addEventListener('input', v);
    else if (k === 'onKeyDown') e.addEventListener('keydown', v);
    else if (k === 'onChange') e.addEventListener('change', v);
    else if (k === 'onDragOver') e.addEventListener('dragover', v);
    else if (k === 'onDrop') e.addEventListener('drop', v);
    else if (k === 'onDragLeave') e.addEventListener('dragleave', v);
    else if (k === 'className') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else if (v === true) e.setAttribute(k, '');
    else e.setAttribute(k, v);
  }
  if (typeof children === 'string') e.innerHTML = children;
  else for (const c of children) if (c != null) e.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}

function render() {
  document.documentElement.dataset.level = state.pickGrade ? levelOfGrade(state.pickGrade) : 'elementary';
  renderApp();
}

// ═══════════════════ LANDING content (rendered inside the shell's stage) ═══════════════════
function renderLandingContent(host) {
  const gradesAvail = [...new Set(state.courses.map(c => c.grade))].sort((a,b)=>a-b);
  const elems = gradesAvail.filter(g => g <= 8);
  const secs  = gradesAvail.filter(g => g >= 9 && g <= 12);
  const uni   = gradesAvail.filter(g => g >= 13);
  const subsForGrade = state.pickGrade
    ? [...new Set(state.courses.filter(c => c.grade === state.pickGrade).map(c => c.subject))]
    : [];

  const inner = h('div', {className:'landing-inner'});
  inner.appendChild(h('div', {className:'landing-hero', html:`<h1>Project <em>X</em></h1><p>Pick where to start.</p>`}));

  // Grade
  const gSection = h('div', {className:'landing-section'});
  gSection.appendChild(h('h2', {html:`<span class="num">01</span>Grade`}));
  const mkPill = (g, label) => h('button',
    {className:'grade-pill' + (state.pickGrade===g?' picked':''),
     onClick:()=>{state.pickGrade = g; state.pickSubject = null; render();}}, label);
  if (elems.length) { const gp = h('div', {className:'grade-group'}, [h('div',{className:'grade-group-label'},'Elementary')]); const row = h('div',{className:'grade-grid'}); elems.forEach(g => row.appendChild(mkPill(g, 'Grade ' + g))); gp.appendChild(row); gSection.appendChild(gp); }
  if (secs.length) { const gp = h('div', {className:'grade-group'}, [h('div',{className:'grade-group-label'},'Secondary')]); const row = h('div',{className:'grade-grid'}); secs.forEach(g => row.appendChild(mkPill(g, 'Grade ' + g))); gp.appendChild(row); gSection.appendChild(gp); }
  if (uni.length) { const gp = h('div', {className:'grade-group'}, [h('div',{className:'grade-group-label'},'University')]); const row = h('div',{className:'grade-grid'}); uni.forEach(g => row.appendChild(mkPill(g, g === 13 ? 'UW 1A' : 'UW 1B'))); gp.appendChild(row); gSection.appendChild(gp); }
  inner.appendChild(gSection);

  // Subject
  if (state.pickGrade) {
    const sSection = h('div', {className:'landing-section'});
    sSection.appendChild(h('h2', {html:`<span class="num">02</span>Subject`}));
    const sGrid = h('div', {className:'subject-grid'});
    subsForGrade.forEach(sub => {
      const courses = state.courses.filter(c => c.grade === state.pickGrade && c.subject === sub);
      const label = state.pickGrade >= 13 && courses[0] ? courses[0].label : (SUB_LABEL[sub] || sub);
      sGrid.appendChild(h('button', {className:'subject-pill' + (state.pickSubject===sub?' picked':''),
        'data-sub':sub, onClick:()=>{state.pickSubject = sub; render();}}, label));
    });
    if (subsForGrade.length === 0) sSection.appendChild(h('div', {className:'empty'}, 'No subjects for this grade yet.'));
    sSection.appendChild(sGrid);

    if (state.pickSubject) {
      const cta = h('div', {className:'landing-cta'});
      cta.appendChild(h('button', {className:'btn-big', onClick:()=>{
        const c = state.courses.find(cc => cc.grade === state.pickGrade && cc.subject === state.pickSubject);
        state.courseId = c.id;
        state.outcomeId = c.outcomes[0]?.id || '';
        state.strand = null; state.active = 'tutor';
        resetPanels();
        state.view = 'app';
        try { localStorage.setItem('px:pick', JSON.stringify({grade:state.pickGrade, subject:state.pickSubject})); localStorage.setItem('px:view','app'); } catch(e){}
        render();
      }}, 'Continue →'));
      cta.appendChild(h('span', {className:'picked-summary'}, `Grade ${state.pickGrade >= 13 ? (state.pickGrade === 13 ? 'UW 1A' : 'UW 1B') : state.pickGrade} · ${SUB_LABEL[state.pickSubject] || state.pickSubject}`));
      sSection.appendChild(cta);
    }
    inner.appendChild(sSection);
  }

  host.appendChild(inner);
}

// ═══════════════════ APP SHELL ═══════════════════
const MODULES = [
  {group:'Learn', items:[{id:'tutor', label:'Tutor', sub:'ask & get help'},
                          {id:'explain', label:'Explain', sub:'concepts & examples'}]},
  {group:'Practice', items:[{id:'practice', label:'Practice', sub:'hint escalation'},
                             {id:'flashcards', label:'Flashcards', sub:'flip & rate'},
                             {id:'rapid', label:'Rapid fire', sub:'one after another'},
                             {id:'challenge', label:'Challenge', sub:'60-second timer'}]},
  {group:'Review', items:[{id:'mastery', label:'Progress', sub:'per outcome'}]},
  {group:'Teach the AI', items:[{id:'teach', label:'Content library', sub:'upload · embed · cite'}]},
  {group:'Session', items:[{id:'log', label:'History', sub:'events'}]},
];

const SUB_LABEL = {math:'Math', french:'French', english:'English', science:'Science',
                   chemistry:'Chemistry', physics:'Physics', cs:'Computer Science',
                   hpe:'Health & PE', social:'Social Studies', history:'History',
                   geography:'Geography', business:'Business', arts:'The Arts',
                   civics:'Civics'};

function renderApp() {
  const app = document.getElementById('app');
  app.innerHTML = '';

  const wrap = h('div', {className:'app'});
  const c = currentCourse();
  const o = currentOutcome();
  const onLanding = state.view === 'landing' || !c;

  // Top bar
  const brand = h('div', {className:'brand'}, [
    h('div', {className:'brand-mark'}, 'X'),
    h('div', {className:'brand-name'}, 'Project X'),
    h('div', {className:'crumb', html: c && !onLanding ? `Grade <b>${c.grade >= 13 ? c.label : c.grade}</b> · <b>${SUB_LABEL[c.subject] || c.subject}</b>` : `<span style="color:var(--ink-faint)">pick a grade →</span>`}),
  ]);

  const pill = state.health && state.health.ollama_reachable && state.health.chat_model_ready
    ? {c:'status ok', t:'AI ready'}
    : {c:'status warn', t:'AI off · scripted'};
  const statusPill = h('div', {className:pill.c}, [h('span',{className:'dot'}),' '+pill.t]);
  const homeBtn = h('button', {className:'icon-btn', title:'Change grade / subject', onClick:()=>{
    state.view = 'landing'; try{ localStorage.setItem('px:view','landing'); }catch(e){}; render();
  }}, `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12l9-9 9 9M5 10v10h14V10"/></svg>`);
  const themeBtn = h('button', {className:'icon-btn', title:'Toggle theme', onClick:()=>{
    const cur = document.documentElement.dataset.theme;
    if (cur === 'dark') { delete document.documentElement.dataset.theme; try{localStorage.setItem('px:theme','')}catch(e){} }
    else { document.documentElement.dataset.theme = 'dark'; try{localStorage.setItem('px:theme','dark')}catch(e){} }
    render();
  }}, `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 3v1M12 20v1M4.2 4.2l.7.7M19.1 19.1l.7.7M3 12h1M20 12h1M4.2 19.8l.7-.7M19.1 4.9l.7-.7"/></svg>`);
  wrap.appendChild(h('div', {className:'topbar'}, [brand, h('div',{className:'top-actions'},[statusPill, homeBtn, themeBtn])]));

  // Rail — always visible. Modules disabled on landing.
  const railNav = h('nav', {className:'rail-nav'}, MODULES.map(g =>
    h('div', {className:'rail-group'}, [
      h('div', {className:'rail-group-label'}, g.group),
      ...g.items.map(m => {
        const disabled = onLanding;
        const btn = h('button', {className:'rail-item' + (state.active === m.id && !onLanding ? ' active' : ''),
                                  disabled: disabled ? '' : null,
                                  style: disabled ? {opacity:'0.35', cursor:'not-allowed'} : {},
                                  onClick:()=>{
                                    if (disabled) return;
                                    state.active = m.id;
                                    if (m.id === 'challenge') challengeReset();
                                    if (m.id === 'rapid') rapidReset();
                                    render();
                                  }});
        btn.innerHTML = `<span>${m.label}</span><span class="sub">${m.sub}</span>`;
        return btn;
      }),
    ])));
  const rail = h('aside', {className:'rail'}, [railNav]);

  const stageInner = h('div', {className:'stage-inner'});
  const stage = h('main', {className:'stage'}, [stageInner]);
  wrap.appendChild(h('div', {className:'workspace'}, [rail, stage]));
  app.appendChild(wrap);

  // Landing view — render picker in the stage
  if (onLanding) { renderLandingContent(stageInner); return; }

  if (!o) { stageInner.appendChild(h('div', {className:'loading'}, 'No outcomes.')); return; }

  const needsSelectors = ['tutor','explain','practice','flashcards','rapid','challenge','teach'].includes(state.active);
  if (needsSelectors) renderStrandAndOutcome(stageInner, c);

  ({tutor:renderTutor, explain:renderExplain, practice:renderPractice, flashcards:renderFlashcards,
    rapid:renderRapid, challenge:renderChallenge, mastery:renderMastery, teach:renderTeach, log:renderLog})[state.active](stageInner);
}

function renderStrandAndOutcome(host, c) {
  const strands = [...new Set(c.outcomes.map(o => o.strand))];
  if (strands.length > 1) {
    const chips = h('div', {className:'strand-chips'});
    chips.appendChild(h('button', {className:'strand-chip' + (state.strand === null ? ' on' : ''),
      onClick:()=>{state.strand = null; render();}}, 'All strands'));
    strands.forEach(s => chips.appendChild(h('button', {className:'strand-chip' + (state.strand === s ? ' on' : ''),
      onClick:()=>{state.strand = s; const outs = c.outcomes.filter(o => o.strand === s); if(outs.length) state.outcomeId = outs[0].id; resetPanels(); render();}}, s)));
    host.appendChild(chips);
  }
  const filtered = state.strand ? c.outcomes.filter(o => o.strand === state.strand) : c.outcomes;
  const row = h('div', {className:'outcome-select-row'});
  row.appendChild(h('span', {}, 'Topic'));
  const sel = h('select', {onChange:e=>{state.outcomeId = e.target.value; resetPanels(); render();}},
    filtered.map(o => h('option', {value:o.id, selected:o.id===state.outcomeId||undefined}, o.code + ' · ' + o.name)));
  row.appendChild(sel);
  host.appendChild(row);
}

function resetPanels() {
  state.chat = {mode:state.chat.mode, messages:[], citations:[], provider:'', sending:false, input:'', conv:[], convCursor:0};
  state.explain.conceptId = '';
  state.practice = {idx:0, picked:null, typed:'', hints:0, verdict:null, explainShown:false, correctAnswer:null};
  state.flash = {idx:0, side:'front', queue:[], known:[], learning:[]};
  state.rapid = {running:false, score:0, questionsAsked:0, current:null, picked:null, typed:'', verdict:null, correctAnswer:null};
  if (state.challenge.tid) { clearInterval(state.challenge.tid); state.challenge.tid = null; }
  state.challenge = {running:false, timeLeft:60, score:0, questionsAsked:0, current:null, picked:null, typed:'', tid:null, verdict:null, correctAnswer:null};
}

function panelHead(host, eye, title, purpose) {
  const el = h('div', {className:'panel-head'});
  el.innerHTML = `<div class="panel-eye">${eye}</div><h1 class="panel-title">${title}</h1><p class="panel-purpose">${purpose}</p>`;
  host.appendChild(el);
}

// ═══════════════════ Panel: TUTOR ═══════════════════
async function renderTutor(host) {
  const outcome = currentOutcome();
  if (state.chat.conv.length === 0 && state.chat.convCursor === 0) {
    try { state.chat.conv = await api.conv(outcome.id); } catch(e){ state.chat.conv = []; }
    if (state.chat.messages.length === 0) {
      state.chat.messages = [{role:'assistant', content: `Let's work on ${outcome.name}. Pick a starter below or type a question — in Practice I'll guide with hints; in Homework I'll walk you through.`}];
    }
  }
  panelHead(host, 'Tutor', `Ask the <em>tutor</em>`,
    state.chat.mode === 'practice' ? "I nudge before I tell you." : "Homework mode: direct answers with reasoning.");
  const wrap = h('div', {className:'chat-wrap'});
  const panel = h('div', {className:'chat-panel'});
  panel.appendChild(h('div', {className:'chat-mode'}, [
    h('div', {style:{display:'flex', gap:'4px'}}, ['practice','homework'].map(m =>
      h('button', {className:'pill' + (state.chat.mode===m?' on':''),
                   onClick:()=>{state.chat.mode = m; state.chat.messages=[]; state.chat.convCursor=0; render();}},
        m === 'practice' ? 'Practice' : 'Homework help'))),
    h('span', {style:{fontFamily:'var(--serif)', fontStyle:'italic', fontSize:'13px', color:'var(--ink-muted)'}}, state.chat.mode === 'practice' ? "nudge, then answer" : "straight to it"),
  ]));

  const logEl = h('div', {className:'chat-log'});
  state.chat.messages.forEach((m, i) => {
    const last = i === state.chat.messages.length - 1;
    const cites = (m.role === 'assistant' && last && state.chat.citations.length) ? state.chat.citations : [];
    const bubble = h('div', {className:'bubble'}, m.content);

    // Auto-generate an image for assistant messages if the text has a math
    // pattern (5 × 3, 3 groups of 4, 3/4, etc.). Same drawFor used elsewhere.
    if (m.role === 'assistant') {
      const svg = drawFor(m.content);
      if (svg) {
        bubble.appendChild(h('div', {className:'img-box', style:{marginTop:'10px', background:'rgba(0,0,0,0.04)', padding:'10px', borderRadius:'6px'}, html:svg}));
      }
    }

    if (cites.length) {
      const cd = h('div', {className:'cites'});
      cites.forEach(c => cd.appendChild(h('span', {className:'cite', title:c.excerpt}, `[${c.title}] · ${Math.round(c.similarity*100)}%`)));
      bubble.appendChild(cd);
    }
    logEl.appendChild(h('div', {className:'msg '+m.role}, [h('div',{className:'avatar'}, m.role==='assistant'?'AI':'Me'), bubble]));
  });
  if (state.chat.sending) logEl.appendChild(h('div', {className:'msg assistant'}, [h('div',{className:'avatar'},'AI'), h('div',{className:'bubble', style:{opacity:'0.6'}}, 'thinking…')]));
  panel.appendChild(logEl);
  setTimeout(() => { logEl.scrollTop = logEl.scrollHeight; }, 0);

  const inp = h('input', {placeholder:'Type a question…', value: state.chat.input,
    onInput:e=>{state.chat.input=e.target.value;},
    onKeyDown:e=>{ if(e.key==='Enter' && !state.chat.sending){ const t=state.chat.input; state.chat.input=''; send(t); }}});
  const suggs = h('div', {className:'suggs'});
  const starters = state.chat.mode === 'practice'
    ? ["I don't get this","Give me a hint","Show an example","Am I on the right track?"]
    : ["Show the fastest steps","Walk me through it","What's the key idea?"];
  starters.forEach(s => suggs.appendChild(h('button', {className:'sugg', onClick:()=>send(s)}, s)));
  if (state.chat.convCursor > 0 && state.chat.convCursor < state.chat.conv.length)
    suggs.appendChild(h('button', {className:'sugg', style:{borderColor:'var(--accent)', color:'var(--accent-ink)', background:'var(--accent-soft)'},
      onClick:()=>advance()}, 'Continue →'));
  panel.appendChild(h('div', {className:'chat-in'}, [
    suggs,
    h('div', {className:'chat-row'}, [inp, h('button', {className:'btn', disabled:state.chat.sending||!state.chat.input.trim(),
      onClick:()=>{ const t=state.chat.input; state.chat.input=''; send(t); }}, 'Send')]),
  ]));
  wrap.appendChild(panel);

  const side = h('aside', {className:'chat-side'});
  const c1 = h('div', {className:'card'});
  c1.innerHTML = `<div class="card-title">Topic</div><div class="card-sub">${outcome.code}</div><p style="margin:0;color:var(--ink-muted);line-height:1.5;">${esc(outcome.blurb)}</p>`;
  side.appendChild(c1);
  const c2 = h('div', {className:'card'});
  c2.innerHTML = `<div class="card-title">Grounding</div><div class="card-sub">retrieved</div>` + (state.chat.citations.length
    ? '<ul style="list-style:none;padding:0;margin:0;font-size:12.5px;color:var(--ink-muted);">' + state.chat.citations.map(c => `<li>[${esc(c.title)}] · ${Math.round(c.similarity*100)}%</li>`).join('') + '</ul>'
    : '<p style="margin:0;color:var(--ink-muted);font-size:13px;">Nothing matched. Upload material in the Content library.</p>');
  side.appendChild(c2);
  wrap.appendChild(side);
  host.appendChild(wrap);

  async function send(text) {
    if (!text.trim() || state.chat.sending) return;
    state.chat.messages.push({role:'user', content:text});
    state.chat.sending = true; render();
    try {
      const r = await api.chat({outcome_id:outcome.id, mode:state.chat.mode, level:levelOfGrade(currentCourse().grade), messages:state.chat.messages});
      state.chat.messages.push({role:'assistant', content:r.reply});
      state.chat.citations = r.citations; state.chat.provider = r.provider;
    } catch(e) { state.chat.messages.push({role:'assistant', content:`(error: ${e.message})`}); }
    finally { state.chat.sending = false; render(); }
  }
  function advance() {
    while (state.chat.convCursor < state.chat.conv.length) {
      state.chat.messages.push(state.chat.conv[state.chat.convCursor]);
      const wasA = state.chat.conv[state.chat.convCursor].role === 'assistant';
      state.chat.convCursor++;
      if (wasA) break;
    }
    render();
  }
}

// ═══════════════════ Panel: EXPLAIN ═══════════════════
function renderExplain(host) {
  const outcome = currentOutcome();
  if (!state.explain.conceptId) state.explain.conceptId = outcome.concepts[0]?.id;
  const sel = outcome.concepts.find(c => c.id === state.explain.conceptId) || outcome.concepts[0];
  panelHead(host, 'Explain', `Concept <em>explainer</em>`, "Read this before you practice.");
  const wrap = h('div', {className:'explain-wrap'});
  const list = h('div', {className:'concept-list'});
  outcome.concepts.forEach(c => {
    const b = h('button', {className:'concept-item' + (c.id===sel?.id?' on':''), onClick:()=>{state.explain.conceptId=c.id; render();}});
    b.innerHTML = `${esc(c.name)}<span class="sub">${esc(c.sub)}</span>`;
    list.appendChild(b);
  });
  wrap.appendChild(list);
  const body = h('div', {className:'concept-body'});
  if (sel) body.innerHTML = `
    <h3>${esc(sel.name)}</h3>
    <div class="cs one"><div class="lbl">The one-liner</div><p>${esc(sel.one_liner)}</p></div>
    <div class="cs"><div class="lbl">Full explanation</div><p>${esc(sel.body)}</p><div class="worked">${esc(sel.worked)}</div></div>
    <div class="cs mis"><div class="lbl">Common mistake</div><p>${esc(sel.misconception)}</p></div>`;
  wrap.appendChild(body);
  host.appendChild(wrap);
}

// ═══════════════════ Panel: PRACTICE (classic with always-explain) ═══════════════════
function renderPractice(host) {
  const outcome = currentOutcome();
  const qs = outcome.questions;
  if (!qs.length) { host.appendChild(h('div', {className:'loading'}, 'No questions here yet.')); return; }
  const idx = Math.min(state.practice.idx, qs.length-1); const q = qs[idx]; const a = state.practice;
  panelHead(host, 'Practice', `Practice <em>questions</em>`, "The full explanation is always shown after you answer — even when you get it right.");
  const card = h('div', {className:'qcard'});
  const meta = h('div', {className:'qmeta'});
  meta.innerHTML = `<span>Question ${idx+1} of ${qs.length}</span><span class="diff">Difficulty ${[1,2,3,4,5].map(n => `<span class="dd ${n<=q.diff?'on':''}"></span>`).join('')}</span>`;
  card.appendChild(meta);
  card.appendChild(h('div', {className:'prompt'}, q.prompt));
  const svg = drawFor(q.prompt);
  if (svg) card.appendChild(h('div', {className:'img-box', html:svg}));

  if (q.kind === 'mc') {
    const choices = h('div', {className:'choices'});
    q.choices.forEach((c, i) => {
      let cls = 'choice';
      if (a.picked===i && !a.verdict) cls += ' picked';
      if (a.verdict && a.picked===i) cls += a.verdict==='right' ? ' right' : ' wrong';
      if (a.verdict && a.correctAnswer !== null && i === parseInt(a.correctAnswer)) cls += ' right';
      const b = h('button', {className:cls, disabled:!!a.verdict, onClick:()=>{ if(!a.verdict){ a.picked=i; render(); }}});
      b.innerHTML = `<span class="letter">${String.fromCharCode(65+i)}</span><span>${esc(c)}</span>`;
      choices.appendChild(b);
    });
    card.appendChild(choices);
  } else {
    const wr = h('div', {className:'short'});
    wr.appendChild(h('input', {value:a.typed, placeholder:'Type your answer',
      disabled:!!a.verdict, className:a.verdict||'',
      onInput:e=>{a.typed=e.target.value;},
      onKeyDown:e=>{if(e.key==='Enter') submit();}}));
    card.appendChild(wr);
  }

  if (a.verdict) card.appendChild(h('div', {className:'verdict '+a.verdict}, a.verdict==='right' ? '✓ Correct! See the full explanation below.' : '✗ Not quite. Read the full explanation below.'));

  // Always show explanation after answering — includes the repeated-addition
  // breakdown for multiplication questions.
  if (a.verdict) {
    const eb = h('div', {className:'explain-box'});
    eb.innerHTML = `<span class="k">Why</span><div class="title">Full explanation</div><div class="body">${esc(fullExplanation(q))}</div>`;
    card.appendChild(eb);
  }

  if (a.hints > 0) {
    const hh = h('div', {className:'hints'});
    q.hints.slice(0, a.hints).forEach((t, i) => {
      const d = h('div', {className:'hint'}); d.innerHTML = `<span class="k">Hint ${i+1}</span>${esc(t)}`; hh.appendChild(d);
    });
    card.appendChild(hh);
  }

  const act = h('div', {className:'qact'});
  const g1 = h('div', {className:'g'});
  if (!a.verdict) g1.appendChild(h('button', {className:'btn big', onClick:submit}, 'Check my answer'));
  if (!a.verdict && a.hints < q.hints.length) g1.appendChild(h('button', {className:'btn ghost', onClick:()=>{a.hints++; render();}}, `Hint (${q.hints.length - a.hints} left)`));
  act.appendChild(g1);
  act.appendChild(h('div', {className:'g'}, [h('button', {className:'btn ghost',
    onClick:()=>{state.practice={idx: idx<qs.length-1 ? idx+1 : 0, picked:null, typed:'', hints:0, verdict:null, explainShown:false, correctAnswer:null}; render();}},
    idx < qs.length-1 ? 'Next question →' : 'Start over ↻')]));
  card.appendChild(act);
  host.appendChild(card);

  async function submit() {
    if (a.verdict) return;
    let ans = '';
    if (q.kind==='mc') { if (a.picked===null) return; ans = String(a.picked); }
    else { if (!a.typed.trim()) return; ans = a.typed.trim(); }
    try {
      const r = await api.submit({outcome_id:outcome.id, question_id:q.id, answer:ans, hints_used:a.hints, mode:'practice'});
      a.verdict = r.correct ? 'right' : 'wrong'; a.correctAnswer = r.answer;
      await refreshMastery();
    } catch(e) { alert('Something broke: ' + e.message); }
    render();
  }
}

// ═══════════════════ Panel: FLASHCARDS ═══════════════════
function renderFlashcards(host) {
  const outcome = currentOutcome();
  const qs = outcome.questions;
  if (!qs.length) { host.appendChild(h('div', {className:'loading'}, 'No cards for this outcome.')); return; }

  // Build queue on first entry or after outcome change
  if (state.flash.queue.length === 0 && state.flash.known.length === 0 && state.flash.learning.length === 0) {
    state.flash.queue = qs.map((_, i) => i);
    state.flash.idx = 0;
    state.flash.side = 'front';
  }

  panelHead(host, 'Flashcards', `<em>Flashcards</em> · flip and rate`, "Quizlet-style. Tap the card to flip. Rate yourself honestly — you'll see 'still learning' cards again.");

  // Progress
  const total = qs.length;
  const done = state.flash.known.length + state.flash.learning.length;
  host.appendChild(h('div', {className:'flash-progress'}, `${done} of ${total} rated · ${state.flash.known.length} known · ${state.flash.learning.length} still learning`));

  if (state.flash.queue.length === 0) {
    // Round done — offer to redo the "still learning" ones
    const done = h('div', {className:'flash-done'});
    done.innerHTML = `
      <div style="font-size:36px;margin-bottom:8px">✨</div>
      <div style="font-weight:700;font-size:22px;margin-bottom:8px">All cards reviewed!</div>
      <div style="color:var(--ink-muted);margin-bottom:20px">${state.flash.known.length} known · ${state.flash.learning.length} still learning</div>`;
    if (state.flash.learning.length > 0) {
      done.appendChild(h('button', {className:'btn big', onClick:()=>{
        state.flash.queue = [...state.flash.learning]; state.flash.learning = []; state.flash.idx = 0; state.flash.side = 'front'; render();
      }}, `Review ${state.flash.learning.length} still-learning cards`));
    } else {
      done.appendChild(h('button', {className:'btn big', onClick:()=>{
        state.flash.queue = qs.map((_, i) => i); state.flash.known = []; state.flash.learning = []; state.flash.idx = 0; state.flash.side = 'front'; render();
      }}, 'Start over'));
    }
    host.appendChild(done);
    return;
  }

  const qIdx = state.flash.queue[0];
  const q = qs[qIdx];
  const svg = drawFor(q.prompt);

  const scene = h('div', {className:'flash-scene'});
  const card = h('div', {className:'flash' + (state.flash.side === 'back' ? ' flipped' : ''),
    onClick:()=>{state.flash.side = state.flash.side === 'front' ? 'back' : 'front'; render();}});
  // Front — question
  const front = h('div', {className:'flash-face front'});
  front.innerHTML = `<div class="face-label">Question · click to flip</div><div class="face-text">${esc(q.prompt)}</div>` + (svg ? `<div style="margin-top:14px">${svg}</div>` : '');
  card.appendChild(front);
  // Back — answer + explanation (with repeated-addition for multiplication)
  const back = h('div', {className:'flash-face back'});
  const answerText = q.kind === 'mc' ? (q.choices ? q.choices[parseInt(q.answer)] : q.answer) : q.answer;
  back.innerHTML = `<div class="face-label">Answer</div><div class="face-text">${esc(answerText)}</div><div class="face-explain">${esc(fullExplanation(q))}</div>`;
  card.appendChild(back);
  scene.appendChild(card);
  host.appendChild(scene);

  // Actions
  const actions = h('div', {className:'flash-actions'});
  if (state.flash.side === 'front') {
    actions.appendChild(h('button', {className:'flash-btn flip', onClick:()=>{state.flash.side='back'; render();}}, 'Flip card'));
  } else {
    actions.appendChild(h('button', {className:'flash-btn learning', onClick:()=>rate('learning')}, 'Still learning'));
    actions.appendChild(h('button', {className:'flash-btn known', onClick:()=>rate('known')}, 'I know it'));
  }
  host.appendChild(actions);

  async function rate(how) {
    if (how === 'known') state.flash.known.push(qIdx);
    else state.flash.learning.push(qIdx);
    // record attempt (correct = known)
    try {
      await api.submit({outcome_id:outcome.id, question_id:q.id, answer: (q.kind === 'mc' ? q.answer : q.answer), hints_used:0, mode:'flashcard'});
      await refreshMastery();
    } catch(e){}
    state.flash.queue.shift();
    state.flash.side = 'front';
    render();
  }
}

// ═══════════════════ Panels: RAPID FIRE + CHALLENGE (shared core) ═══════════════════
function pickRandomQuestion() {
  const outcome = currentOutcome();
  const qs = outcome.questions;
  if (!qs.length) return null;
  return qs[Math.floor(Math.random() * qs.length)];
}

function renderQuickQuestion(host, s, mode, onSubmit, onNextClick) {
  // Shared: renders the current question card for rapid/challenge modes.
  if (!s.current) return;
  const q = s.current;
  const svg = drawFor(q.prompt);
  const card = h('div', {className:'qcard'});
  card.appendChild(h('div', {className:'prompt'}, q.prompt));
  if (svg) card.appendChild(h('div', {className:'img-box', html:svg}));

  if (q.kind === 'mc') {
    const choices = h('div', {className:'choices'});
    q.choices.forEach((c, i) => {
      let cls = 'choice';
      if (s.verdict && s.picked === i) cls += s.verdict === 'right' ? ' right' : ' wrong';
      if (s.verdict && s.correctAnswer !== null && i === parseInt(s.correctAnswer)) cls += ' right';
      const btn = h('button', {className:cls, disabled:!!s.verdict, onClick:()=>{ if(!s.verdict){ s.picked = i; onSubmit(); }}});
      btn.innerHTML = `<span class="letter">${String.fromCharCode(65+i)}</span><span>${esc(c)}</span>`;
      choices.appendChild(btn);
    });
    card.appendChild(choices);
  } else {
    const wr = h('div', {className:'short'});
    wr.appendChild(h('input', {value:s.typed, placeholder:'Type answer, Enter to submit',
      disabled:!!s.verdict, autofocus:true,
      onInput:e=>{s.typed=e.target.value;},
      onKeyDown:e=>{if(e.key==='Enter') onSubmit();}}));
    card.appendChild(wr);
  }

  if (s.verdict) {
    card.appendChild(h('div', {className:'verdict '+s.verdict, style:{marginTop:'12px'}},
      s.verdict === 'right' ? '✓ Correct!' : '✗ Wrong. Answer: ' + (q.kind==='mc' ? q.choices[parseInt(s.correctAnswer)] : s.correctAnswer)));

    // Rapid mode: show the full explanation (with repeated addition) + Next button.
    // Challenge mode: auto-advance quickly, skip the explanation.
    if (mode === 'challenge') {
      setTimeout(() => { if (state.challenge.running) challengeNext(); }, 1400);
    } else {
      const eb = h('div', {className:'explain-box', style:{marginTop:'12px'}});
      eb.innerHTML = `<span class="k">Why</span><div class="title">Full explanation</div><div class="body">${esc(fullExplanation(q))}</div>`;
      card.appendChild(eb);
      const nextBar = h('div', {className:'qact'});
      nextBar.appendChild(h('div', {className:'g'}));
      nextBar.appendChild(h('div', {className:'g'}, [h('button', {className:'btn big', onClick:onNextClick}, 'Next question →')]));
      card.appendChild(nextBar);
    }
  }
  host.appendChild(card);
}

async function submitQuick(s, mode) {
  if (!s.current || s.verdict) return;
  const q = s.current;
  let ans = '';
  if (q.kind === 'mc') { if (s.picked === null) return; ans = String(s.picked); }
  else { if (!s.typed.trim()) return; ans = s.typed.trim(); }
  try {
    const outcome = currentOutcome();
    const r = await api.submit({outcome_id:outcome.id, question_id:q.id, answer:ans, hints_used:0, mode:mode});
    s.verdict = r.correct ? 'right' : 'wrong';
    s.correctAnswer = r.answer;
    s.questionsAsked++;
    if (r.correct) s.score++;
    await refreshMastery();
  } catch(e) { s.verdict = 'wrong'; s.questionsAsked++; }
  render();
}

// ─────────── RAPID FIRE (no timer) ───────────
function rapidReset() {
  state.rapid = {running:false, score:0, questionsAsked:0, current:null, picked:null, typed:'', verdict:null, correctAnswer:null};
}
function renderRapid(host) {
  const outcome = currentOutcome();
  const r = state.rapid;
  panelHead(host, 'Rapid fire', `<em>Rapid</em> fire`, "One question after another. No timer — take your time. Stop whenever.");

  const header = h('div', {className:'blitz-header'});
  header.appendChild(h('div', {}, [
    h('div', {style:{fontFamily:'var(--mono)', fontSize:'11px', letterSpacing:'0.14em', textTransform:'uppercase', color:'var(--ink-faint)', fontWeight:'600'}}, 'Answered'),
    h('div', {className:'blitz-score', html: `${r.questionsAsked}`}),
  ]));
  header.appendChild(h('div', {style:{textAlign:'right'}}, [
    h('div', {style:{fontFamily:'var(--mono)', fontSize:'11px', letterSpacing:'0.14em', textTransform:'uppercase', color:'var(--ink-faint)', fontWeight:'600'}}, 'Correct'),
    h('div', {className:'blitz-score', html: `${r.score}<span class="u">of ${r.questionsAsked}</span>`}),
  ]));
  host.appendChild(header);

  if (!r.running && r.questionsAsked === 0) {
    const start = h('div', {className:'qcard', style:{textAlign:'center'}});
    start.innerHTML = `<div style="font-family:var(--serif);font-size:22px;font-weight:600;margin-bottom:10px;color:var(--ink-strong);">Rapid fire: ${esc(outcome.name)}</div>
      <p style="color:var(--ink-muted);margin:0 0 20px;">Quick-fire questions from this topic, one after another. No timer. Answer, click Next, keep going. Stop whenever you're done.</p>`;
    start.appendChild(h('button', {className:'btn big', onClick:startRapid}, 'Start'));
    host.appendChild(start);
    return;
  }

  if (r.running) {
    renderQuickQuestion(host, r, 'rapid', () => submitQuick(r, 'rapid'), rapidNext);
    // Stop button
    host.appendChild(h('div', {style:{marginTop:'14px', textAlign:'center'}}, [
      h('button', {className:'btn ghost', onClick:stopRapid}, 'End round'),
    ]));
    return;
  }

  // Finished
  const done = h('div', {className:'blitz-summary'});
  const pct = r.questionsAsked > 0 ? Math.round(r.score / r.questionsAsked * 100) : 0;
  done.innerHTML = `<div class="big">${r.score} / ${r.questionsAsked}</div><div class="lbl">Round complete · ${pct}% correct</div>`;
  done.appendChild(h('div', {style:{marginTop:'22px', display:'flex', gap:'10px', justifyContent:'center'}}, [
    h('button', {className:'btn big', onClick:()=>{rapidReset(); startRapid();}}, 'Play again'),
    h('button', {className:'btn ghost', onClick:()=>{rapidReset(); render();}}, 'Back'),
  ]));
  host.appendChild(done);
}
function startRapid() {
  const q = pickRandomQuestion(); if (!q) return;
  state.rapid = {running:true, score:0, questionsAsked:0, current:q, picked:null, typed:'', verdict:null, correctAnswer:null};
  render();
}
function rapidNext() {
  if (!state.rapid.running) return;
  const q = pickRandomQuestion();
  state.rapid.current = q; state.rapid.picked = null; state.rapid.typed = ''; state.rapid.verdict = null; state.rapid.correctAnswer = null;
  render();
}
function stopRapid() { state.rapid.running = false; render(); }

// ─────────── CHALLENGE (60-second timer) ───────────
function challengeReset() {
  if (state.challenge.tid) { clearInterval(state.challenge.tid); state.challenge.tid = null; }
  state.challenge = {running:false, timeLeft:60, score:0, questionsAsked:0, current:null, picked:null, typed:'', tid:null, verdict:null, correctAnswer:null};
}
function renderChallenge(host) {
  const outcome = currentOutcome();
  const b = state.challenge;
  panelHead(host, 'Challenge', `<em>60-second</em> challenge`, "Same questions, but with a clock. Answer as many as you can. Correct = +1, no penalty for wrong.");

  const header = h('div', {className:'blitz-header'});
  header.appendChild(h('div', {}, [
    h('div', {style:{fontFamily:'var(--mono)', fontSize:'11px', letterSpacing:'0.14em', textTransform:'uppercase', color:'var(--ink-faint)', fontWeight:'600'}}, 'Time'),
    h('div', {className:'blitz-timer' + (b.timeLeft <= 10 && b.running ? ' low' : '')}, b.running ? String(b.timeLeft) + 's' : '60s'),
  ]));
  header.appendChild(h('div', {style:{textAlign:'right'}}, [
    h('div', {style:{fontFamily:'var(--mono)', fontSize:'11px', letterSpacing:'0.14em', textTransform:'uppercase', color:'var(--ink-faint)', fontWeight:'600'}}, 'Score'),
    h('div', {className:'blitz-score', html: `${b.score}<span class="u">/ ${b.questionsAsked}</span>`}),
  ]));
  host.appendChild(header);

  if (!b.running && b.questionsAsked === 0) {
    const start = h('div', {className:'qcard', style:{textAlign:'center'}});
    start.innerHTML = `<div style="font-family:var(--serif);font-size:22px;font-weight:600;margin-bottom:10px;color:var(--ink-strong);">Challenge: ${esc(outcome.name)}</div>
      <p style="color:var(--ink-muted);margin:0 0 20px;">60 seconds on the clock. Ready?</p>`;
    start.appendChild(h('button', {className:'btn big', onClick:startChallenge}, 'Start challenge'));
    host.appendChild(start);
    return;
  }

  if (!b.running && b.questionsAsked > 0) {
    const done = h('div', {className:'blitz-summary'});
    const pct = b.questionsAsked > 0 ? Math.round(b.score / b.questionsAsked * 100) : 0;
    done.innerHTML = `<div class="big">${b.score} / ${b.questionsAsked}</div><div class="lbl">Time's up · ${pct}% accuracy</div>`;
    done.appendChild(h('div', {style:{marginTop:'22px', display:'flex', gap:'10px', justifyContent:'center'}}, [
      h('button', {className:'btn big', onClick:()=>{challengeReset(); startChallenge();}}, 'Play again'),
      h('button', {className:'btn ghost', onClick:()=>{challengeReset(); render();}}, 'Back'),
    ]));
    host.appendChild(done);
    return;
  }

  renderQuickQuestion(host, b, 'challenge', () => submitQuick(b, 'blitz'), null);
}
function startChallenge() {
  const q = pickRandomQuestion(); if (!q) return;
  state.challenge = {running:true, timeLeft:60, score:0, questionsAsked:0, current:q, picked:null, typed:'', tid:null, verdict:null, correctAnswer:null};
  state.challenge.tid = setInterval(() => {
    state.challenge.timeLeft--;
    if (state.challenge.timeLeft <= 0) {
      clearInterval(state.challenge.tid);
      state.challenge.tid = null;
      state.challenge.running = false;
    }
    if (state.active === 'challenge') render();
  }, 1000);
  render();
}
function challengeNext() {
  if (!state.challenge.running) return;
  const q = pickRandomQuestion();
  state.challenge.current = q; state.challenge.picked = null; state.challenge.typed = ''; state.challenge.verdict = null; state.challenge.correctAnswer = null;
  render();
}

// ═══════════════════ Panel: MASTERY ═══════════════════
function renderMastery(host) {
  const c = currentCourse();
  const rows = c.outcomes.map(o => {
    const m = state.mastery.find(x => x.outcome_id === o.id) || {score:0, attempts:0, correct:0};
    return {courseLabel:c.label, ...m, code:o.code, name:o.name, strand:o.strand, outcome_id:o.id};
  });
  const totalAtt = rows.reduce((s,r)=>s+r.attempts, 0);
  const totalCorrect = rows.reduce((s,r)=>s+r.correct, 0);
  const avg = rows.length ? rows.reduce((s,r)=>s+r.score, 0)/rows.length : 0;
  const touched = rows.filter(r => r.attempts>0).length;
  panelHead(host, 'Progress', `<em>My</em> progress`, "Each star reflects how well you know a topic. Practice → stars go up.");
  const sum = h('div', {className:'msum'});
  sum.innerHTML = `
    <div class="stat"><div class="lbl">Avg stars</div><div class="val"><em>${(avg*5).toFixed(1)}</em><span class="u">of 5</span></div></div>
    <div class="stat"><div class="lbl">Topics tried</div><div class="val">${touched}<span class="u">of ${rows.length}</span></div></div>
    <div class="stat"><div class="lbl">Questions done</div><div class="val">${totalAtt}</div></div>
    <div class="stat"><div class="lbl">Accuracy</div><div class="val">${totalAtt?Math.round(totalCorrect/totalAtt*100):0}<span class="u">%</span></div></div>`;
  host.appendChild(sum);
  rows.sort((a,b)=>a.score-b.score);
  const list = h('div', {className:'mlist'});
  rows.forEach(r => {
    const row = h('div', {className:'mrow'});
    const n = Math.round(r.score * 5);
    const starHtml = [1,2,3,4,5].map(i => `<svg viewBox="0 0 24 24" fill="${i<=n?'currentColor':'none'}" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="${i<=n?'on':''}"><path d="M12 2l3 6 7 1-5 5 1 7-6-3-6 3 1-7-5-5 7-1z"/></svg>`).join('');
    row.innerHTML = `<div><div class="out-code">${r.code} · ${esc(r.strand)}</div><div class="out-name">${esc(r.name)}</div><div class="out-sub">${r.attempts} tries · ${r.correct} correct</div></div>
      <div class="mstars">${starHtml}</div><div class="n">${Math.round(r.score*100)}<span class="u">%</span></div>`;
    list.appendChild(row);
  });
  host.appendChild(list);
}

// ═══════════════════ Panel: TEACH ═══════════════════
async function renderTeach(host) {
  const outcome = currentOutcome();
  if (state.teach.sources.length === 0) { try { state.teach.sources = await api.sources(); } catch(e){} }
  panelHead(host, 'Teach the AI', `Content <em>library</em>`, "Give the AI a PDF or paragraph. It chunks, embeds, and cites it back next time you ask.");

  const wrap = h('div', {className:'teach-wrap'});
  const left = h('div', {});
  const drop = h('div', {className:'drop' + (state.teach.hot ? ' hot' : ''),
    onDragOver:e=>{e.preventDefault(); state.teach.hot=true; drop.classList.add('hot');},
    onDragLeave:()=>{state.teach.hot=false; drop.classList.remove('hot');},
    onDrop:e=>{e.preventDefault(); drop.classList.remove('hot'); const f=e.dataTransfer.files?.[0]; if(f) doUpload(f);},
    onClick:()=>document.getElementById('fi').click()});
  drop.innerHTML = `<div class="big">Drop a file here, or click to pick one</div><div class="small">PDF · DOCX · TXT · MD</div>`;
  drop.appendChild(h('input', {id:'fi', type:'file', accept:'.pdf,.docx,.txt,.md', style:{display:'none'},
    onChange:e=>{const f=e.target.files?.[0]; if(f) doUpload(f);}}));
  left.appendChild(drop);

  const attach = h('label', {style:{display:'flex', alignItems:'center', gap:'8px', marginTop:'12px', fontFamily:'var(--mono)', fontSize:'12px', color:'var(--ink-muted)'}});
  const cb = h('input', {type:'checkbox', onChange:e=>{state.teach.attach = e.target.checked;}});
  if (state.teach.attach) cb.checked = true;
  attach.appendChild(cb);
  attach.appendChild(document.createTextNode(` Attach to current topic (${outcome.code})`));
  left.appendChild(attach);

  if (state.teach.busy) left.appendChild(h('div', {className:'loading'}, 'Extracting, chunking, embedding…'));
  if (state.teach.err) left.appendChild(h('div', {className:'error'}, state.teach.err));
  if (state.teach.msg) left.appendChild(h('div', {className:'verdict right', style:{marginTop:'10px'}}, state.teach.msg));

  const sl = h('div', {className:'slist'});
  if (state.teach.sources.length === 0) sl.appendChild(h('div', {className:'empty'}, 'Nothing uploaded yet.'));
  else state.teach.sources.forEach(s => {
    const row = h('div', {className:'sitem'});
    row.innerHTML = `<div><div class="title">${esc(s.title)}</div><div class="meta">${s.kind} · ${(s.bytes_size/1024).toFixed(1)} kB · ${s.outcome_id ? 'attached to '+s.outcome_id : 'available everywhere'}</div></div>
      <div class="stats">${s.chunks_n} pieces<br><span style="opacity:.7">${new Date(s.created_at).toLocaleDateString()}</span></div>`;
    row.appendChild(h('button', {className:'del', onClick:async()=>{if(confirm('Delete this source?')){ await api.del(s.id); state.teach.sources = state.teach.sources.filter(x=>x.id!==s.id); render(); }}}, 'Delete'));
    sl.appendChild(row);
  });
  left.appendChild(sl);
  wrap.appendChild(left);

  const right = h('div', {className:'card'});
  right.innerHTML = `<div class="card-title">…or paste text</div><div class="card-sub">no file needed</div>`;
  const tf = h('div', {className:'field', style:{marginBottom:'10px'}}); tf.innerHTML = `<label>Title</label>`;
  tf.appendChild(h('input', {value:state.teach.pt, placeholder:'Notes · fractions', onInput:e=>{state.teach.pt = e.target.value;}}));
  right.appendChild(tf);
  const xf = h('div', {className:'field', style:{marginBottom:'12px'}}); xf.innerHTML = `<label>What to teach</label>`;
  xf.appendChild(h('textarea', {value:state.teach.px, placeholder:'Paste a paragraph or definitions…', style:{minHeight:'180px'}, onInput:e=>{state.teach.px = e.target.value;}}));
  right.appendChild(xf);
  right.appendChild(h('button', {className:'btn big', disabled:state.teach.busy || !state.teach.px.trim(), onClick:doPaste}, 'Teach the AI'));
  right.appendChild(h('div', {style:{marginTop:'20px', paddingTop:'16px', borderTop:'1px solid var(--rule)', fontSize:'13px', color:'var(--ink-muted)', lineHeight:'1.6'}, html:
    `<div style="font-family:var(--mono);font-size:10.5px;letter-spacing:0.14em;text-transform:uppercase;color:var(--accent);margin-bottom:6px;font-weight:600;">How it works</div>
     Extract → chunk (~500 tokens) → embed via <code style="background:var(--inset);padding:1px 5px;border-radius:3px;font-family:var(--mono);font-size:0.87em;color:var(--accent-ink)">nomic-embed-text</code> (Ollama) → cosine similarity retrieval → top-3 chunks pinned into system prompt → tutor cites as [title].`}));
  wrap.appendChild(right);
  host.appendChild(wrap);

  async function doUpload(file) {
    state.teach.busy = true; state.teach.err = null; state.teach.msg = null; render();
    try { const s = await api.upload(file, state.teach.attach ? outcome.id : null);
          state.teach.msg = `Added "${s.title}" — ${s.chunks_n} pieces.`;
          state.teach.sources = await api.sources(); }
    catch(e) { state.teach.err = e.message; }
    finally { state.teach.busy = false; render(); }
  }
  async function doPaste() {
    if (!state.teach.px.trim()) return;
    state.teach.busy = true; state.teach.err = null; state.teach.msg = null; render();
    try { const s = await api.paste({title:state.teach.pt, text:state.teach.px, outcome_id: state.teach.attach ? outcome.id : null});
          state.teach.msg = `Added "${s.title}" — ${s.chunks_n} pieces.`;
          state.teach.pt = ''; state.teach.px = '';
          state.teach.sources = await api.sources(); }
    catch(e) { state.teach.err = e.message; }
    finally { state.teach.busy = false; render(); }
  }
}

// ═══════════════════ Panel: LOG ═══════════════════
const KL = {chat:'Tutor chat', practice:'Practice', flashcard:'Flashcard', blitz:'Challenge', rapid:'Rapid fire', teach:'Content added'};
async function renderLog(host) {
  panelHead(host, 'History', `<em>My</em> history`, "Everything you did — chats, practice, flashcards, blitz, uploads.");
  try { state.log = await api.log(); } catch(e){}
  const top = h('div', {style:{display:'flex', alignItems:'center', gap:'8px', marginBottom:'10px'}});
  top.appendChild(h('div', {style:{fontFamily:'var(--mono)', fontSize:'11.5px', color:'var(--ink-faint)', letterSpacing:'0.06em', textTransform:'uppercase'}}, `${state.log.length} events`));
  top.appendChild(h('div', {style:{flex:1}}));
  top.appendChild(h('button', {className:'btn ghost', onClick:async()=>{if(confirm('Clear log?')){ await api.clearLog(); state.log=[]; render(); }}}, 'Clear'));
  host.appendChild(top);
  if (!state.log.length) { host.appendChild(h('div', {className:'empty'}, 'No events yet.')); return; }
  const list = h('div', {className:'llist'});
  state.log.forEach(e => {
    const d = new Date(e.created_at);
    const row = h('div', {className:'lrow'});
    row.innerHTML = `<div class="when">${d.toLocaleDateString(undefined,{month:'short',day:'numeric'})}<br>${d.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'})}</div>
      <div class="what"><span class="kind">${KL[e.kind]||e.kind}</span><span class="det">${esc(e.detail||'')}</span></div>
      <div class="badge">${e.outcome_id||''}</div>`;
    list.appendChild(row);
  });
  host.appendChild(list);
}

// ═══════════════════ boot ═══════════════════
(async function() {
  try {
    const [cs, hh, mm] = await Promise.all([api.courses(), api.health(), api.mastery()]);
    state.courses = cs; state.health = hh; state.mastery = mm;
    if (state.view === 'app' && state.pickGrade && state.pickSubject) {
      const c = state.courses.find(cc => cc.grade === state.pickGrade && cc.subject === state.pickSubject);
      if (c) { state.courseId = c.id; state.outcomeId = c.outcomes[0]?.id || ''; }
      else state.view = 'landing';
    }
    render();
  } catch(e) {
    document.getElementById('app').innerHTML = '<div style="padding:40px;color:var(--risk);font-family:var(--sans);">Backend unreachable: ' + e.message + '</div>';
  }
})();
</script>
</body>
</html>"""


def _open_browser_soon():
    time.sleep(1.2)
    try: webbrowser.open("http://localhost:8000")
    except Exception: pass


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        print("\nMissing deps. Run:\n\n  pip install -r requirements.txt\n", file=sys.stderr)
        sys.exit(1)
    print("─" * 60)
    print(" Project X")
    print("─" * 60)
    print(f"  Serving at:  http://localhost:8000")
    print(f"  Database:    {DB_PATH}")
    print(f"  Ollama at:   {OLLAMA_HOST}  (chat: {OLLAMA_CHAT_MODEL})")
    print("  Ctrl+C to stop.")
    print("─" * 60)
    threading.Thread(target=_open_browser_soon, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
