import sqlite3
DB_PATH = "rfp_evaluation.db"
criteria = [
    ("Technical Capability", "Architecture, integrations, scalability, technical fit", 30, 10, 1),
    ("Implementation Plan", "Timeline, milestones, staffing, and risk plan", 20, 10, 1),
    ("Commercial Value", "Pricing clarity, total cost, and assumptions", 20, 10, 1),
    ("Security & Compliance", "Controls, certifications, privacy, and auditability", 20, 10, 1),
    ("Support & Experience", "Support model, similar projects, and references", 10, 10, 1),
]
with sqlite3.connect(DB_PATH) as conn:
    conn.execute("""CREATE TABLE IF NOT EXISTS evaluation_criteria (
        criterion_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
        description TEXT NOT NULL, weight REAL NOT NULL, max_score REAL NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 1)""")
    conn.executemany("""INSERT OR IGNORE INTO evaluation_criteria
        (name, description, weight, max_score, is_active) VALUES (?, ?, ?, ?, ?)""", criteria)
print("Criteria seeded.")
