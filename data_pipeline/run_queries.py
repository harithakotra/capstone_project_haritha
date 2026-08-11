"""
run_queries.py
---------------
Runs the required SQL queries against books.db and prints + saves their
output. Collectively these demonstrate: SELECT/WHERE, ORDER BY, LIMIT,
DISTINCT, IN/BETWEEN, and a JOIN across the two tables.

Output is both printed to the console and written to queries_output.txt
so it can be included in the submission.
"""

import sqlite3

DB_FILE = "books.db"
OUTPUT_FILE = "queries_output.txt"

QUERIES = [
    {
        "id": "Q1",
        "description": "SELECT / WHERE - books priced under £20 (in GBP)",
        "sql": """
            SELECT title, price_gbp, price_inr, rating
            FROM books
            WHERE price_gbp < 20
            ORDER BY price_gbp ASC;
        """,
    },
    {
        "id": "Q2",
        "description": "ORDER BY / LIMIT - 10 most expensive books overall",
        "sql": """
            SELECT title, price_gbp, price_inr
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10;
        """,
    },
    {
        "id": "Q3",
        "description": "DISTINCT - list of distinct category names present in the books table",
        "sql": """
            SELECT DISTINCT c.category_name
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            ORDER BY c.category_name;
        """,
    },
    {
        "id": "Q4",
        "description": "BETWEEN - books priced between £25 and £45 (inclusive)",
        "sql": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 25 AND 45
            ORDER BY price_gbp ASC;
        """,
    },
    {
        "id": "Q5",
        "description": "IN - books with a rating of 4 or 5 stars",
        "sql": """
            SELECT title, rating, price_gbp
            FROM books
            WHERE rating IN (4, 5)
            ORDER BY rating DESC, price_gbp ASC;
        """,
    },
    {
        "id": "Q6",
        "description": "JOIN - top 3 highest-rated (then cheapest) books per category",
        "sql": """
            SELECT c.category_name, b.title, b.rating, b.price_gbp
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            WHERE (
                SELECT COUNT(*)
                FROM books b2
                WHERE b2.category_id = b.category_id
                  AND (b2.rating > b.rating
                       OR (b2.rating = b.rating AND b2.price_gbp < b.price_gbp))
            ) < 3
            ORDER BY c.category_name, b.rating DESC, b.price_gbp ASC;
        """,
    },
]


def run_all_queries():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    output_lines = []

    for q in QUERIES:
        header = f"\n{'=' * 70}\n{q['id']}: {q['description']}\n{'=' * 70}"
        print(header)
        output_lines.append(header)

        sql_display = f"SQL:\n{q['sql'].strip()}\n"
        print(sql_display)
        output_lines.append(sql_display)

        cursor.execute(q["sql"])
        col_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        header_row = " | ".join(col_names)
        print(header_row)
        print("-" * len(header_row))
        output_lines.append(header_row)
        output_lines.append("-" * len(header_row))

        for row in rows:
            row_str = " | ".join(str(v) for v in row)
            print(row_str)
            output_lines.append(row_str)

        footer = f"({len(rows)} row(s) returned)"
        print(footer)
        output_lines.append(footer)

    conn.close()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\nSaved full query output to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_all_queries()
