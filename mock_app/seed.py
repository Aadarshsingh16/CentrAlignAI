import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "bills.db"

SEED_BILLS = [
    {
        "vendor": "Globex",
        "invoice_no": "GLX-2026-012",
        "amount": 2100.00,
        "due_date": "2026-07-30",
        "status": "paid",
    },
    {
        "vendor": "Acme Corporation",
        "invoice_no": "ACM-2026-101",
        "amount": 3200.00,
        "due_date": "2026-09-01",
        "status": "pending",
    },
    {
        "vendor": "Umbrella",
        "invoice_no": "UMB-2026-001",
        "amount": 1450.00,
        "due_date": "2026-08-15",
        "status": "pending",
    },
    {
        "vendor": "Initech",
        "invoice_no": "INT-2026-005",
        "amount": 1800.00,
        "due_date": "2026-06-15",
        "status": "paid",
    },
    {
        "vendor": "Massive Dynamic",
        "invoice_no": "MD-2026-001",
        "amount": 62500.00,
        "due_date": "2026-05-20",
        "status": "paid",
    },
    {
        "vendor": "Cyberdyne Systems",
        "invoice_no": "CYB-2026-001",
        "amount": 1500.00,
        "due_date": "2026-04-10",
        "status": "paid",
    },
]

def get_db_path(db_path=None) -> Path:
    if db_path is not None:
        return Path(db_path)
    env_path = os.getenv("MOCK_APP_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH

def init_db(db_path=None):
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor TEXT NOT NULL,
            invoice_no TEXT UNIQUE NOT NULL,
            amount REAL NOT NULL,
            due_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def reset_db(db_path=None):
    """
    Restores the exact same seed data every time.
    Called before evals and between test runs.
    """
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS bills")
    cursor.execute("""
        CREATE TABLE bills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor TEXT NOT NULL,
            invoice_no TEXT UNIQUE NOT NULL,
            amount REAL NOT NULL,
            due_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    for b in SEED_BILLS:
        cursor.execute(
            """
            INSERT INTO bills (vendor, invoice_no, amount, due_date, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (b["vendor"], b["invoice_no"], b["amount"], b["due_date"], b["status"]),
        )
    conn.commit()
    conn.close()

def get_all_bills(db_path=None) -> list[dict]:
    """
    Directly reads the SQLite database file and returns all bills as dictionaries.
    Used for verifier and ground truth checks.
    """
    path = get_db_path(db_path)
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, vendor, invoice_no, amount, due_date, status, created_at FROM bills ORDER BY id ASC")
    rows = cursor.fetchall()
    bills = [dict(row) for row in rows]
    conn.close()
    return bills

if __name__ == "__main__":
    reset_db()
    print(f"Database reset and seeded at {get_db_path()}. Total bills: {len(get_all_bills())}")
