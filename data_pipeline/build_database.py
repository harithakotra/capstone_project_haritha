"""
build_database.py
-------------------
Creates a normalized SQLite database (books.db) from cleaned_books.csv,
using a two-table schema linked by a primary/foreign key:

    categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
    books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL,
          price_inr REAL, rating INTEGER, in_stock INTEGER,
          category_id INTEGER REFERENCES categories(category_id))

Re-run this script any time to rebuild books.db from scratch (it deletes
any existing file first, so it's always a clean, reproducible build).
"""

import csv
import os
import sqlite3

CLEANED_INPUT_FILE = "cleaned_books.csv"
DB_FILE = "books.db"

SCHEMA_SQL = """
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    price_inr   REAL NOT NULL,
    rating      INTEGER NOT NULL,
    in_stock    INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""


def main():
    # Start fresh every time so the DB is always reproducible from the CSV.
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()
    cursor.executescript(SCHEMA_SQL)

    with open(CLEANED_INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cleaned_rows = list(reader)

    # --- Populate categories table first (so books can reference it) ---
    category_names = sorted(set(row["category"] for row in cleaned_rows))
    cursor.executemany(
        "INSERT INTO categories (category_name) VALUES (?);",
        [(name,) for name in category_names],
    )
    conn.commit()

    # Build a lookup: category_name -> category_id
    cursor.execute("SELECT category_id, category_name FROM categories;")
    category_id_by_name = {name: cid for cid, name in cursor.fetchall()}

    # --- Populate books table ---
    book_rows = []
    for row in cleaned_rows:
        book_rows.append((
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            category_id_by_name[row["category"]],
        ))

    cursor.executemany(
        """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?);""",
        book_rows,
    )
    conn.commit()

    # Sanity check counts
    cursor.execute("SELECT COUNT(*) FROM books;")
    book_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM categories;")
    category_count = cursor.fetchone()[0]

    print(f"Built {DB_FILE}: {category_count} categories, {book_count} books.")

    conn.close()


if __name__ == "__main__":
    main()
