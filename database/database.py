"""SQLite prediction history. Stores no images and no personal data: only the (sanitised) file name,
the model output, the decision threshold, the model version and a timestamp."""
import datetime
import sqlite3
from contextlib import closing
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    filename TEXT,
    predicted_class TEXT NOT NULL,
    probability REAL NOT NULL,
    threshold REAL NOT NULL,
    model_version TEXT,
    species TEXT,
    species_probability REAL,
    source TEXT NOT NULL DEFAULT 'web'
)"""


def _connect(path):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(path)) as conn:
        conn.execute(SCHEMA)
        conn.commit()


def add_prediction(path, *, filename, predicted_class, probability, threshold, model_version,
                   species=None, species_probability=None, source="web"):
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with closing(_connect(path)) as conn:
        cur = conn.execute(
            "INSERT INTO predictions (timestamp, filename, predicted_class, probability, threshold, model_version, "
            "species, species_probability, source) VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, (filename or "")[:120], predicted_class, float(probability), float(threshold), model_version,
             species, None if species_probability is None else float(species_probability), source))
        conn.commit()
        return cur.lastrowid


def list_predictions(path, limit=200):
    with closing(_connect(path)) as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def count(path):
    with closing(_connect(path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


def clear_history(path):
    with closing(_connect(path)) as conn:
        n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.execute("DELETE FROM predictions")
        conn.commit()
        return n
