"""
pandas_verify.py
-----------------
1. Reads two of the SQL query results directly into pandas DataFrames using
   pd.read_sql(...).
2. Separately reproduces the JOIN query's result (Q6: top 3 highest-rated
   books per category) using pd.merge(...) on in-memory DataFrames loaded
   from the raw 'categories' and 'books' tables -- i.e. no SQL JOIN at all,
   just pandas -- and shows that it matches the SQL version.
"""

import sqlite3
import pandas as pd

DB_FILE = "books.db"


def main():
    conn = sqlite3.connect(DB_FILE)

    # ------------------------------------------------------------------
    # 1a. pd.read_sql for Q1-style query (SELECT/WHERE + ORDER BY)
    # ------------------------------------------------------------------
    cheap_books_sql = """
        SELECT title, price_gbp, price_inr, rating
        FROM books
        WHERE price_gbp < 20
        ORDER BY price_gbp ASC;
    """
    df_cheap_books = pd.read_sql(cheap_books_sql, conn)
    print("=" * 70)
    print("pd.read_sql -> books under £20")
    print("=" * 70)
    print(df_cheap_books.to_string(index=False))
    print(f"({len(df_cheap_books)} rows)\n")

    # ------------------------------------------------------------------
    # 1b. pd.read_sql for the JOIN query (Q6: top 3 per category)
    # ------------------------------------------------------------------
    top3_per_category_sql = """
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
    """
    df_join_via_sql = pd.read_sql(top3_per_category_sql, conn)
    print("=" * 70)
    print("pd.read_sql -> top 3 highest-rated books per category (SQL JOIN)")
    print("=" * 70)
    print(df_join_via_sql.to_string(index=False))
    print(f"({len(df_join_via_sql)} rows)\n")

    # ------------------------------------------------------------------
    # 2. Reproduce the same JOIN result using pd.merge on in-memory
    #    DataFrames pulled from the raw tables -- no SQL JOIN involved.
    # ------------------------------------------------------------------
    df_categories = pd.read_sql("SELECT * FROM categories;", conn)
    df_books = pd.read_sql("SELECT * FROM books;", conn)

    # Merge books with their category name (equivalent of the SQL JOIN)
    df_merged = pd.merge(
        df_books, df_categories,
        on="category_id", how="inner"
    )

    # Rank within each category by rating (desc) then price (asc),
    # exactly mirroring the SQL query's ordering/tiebreak logic.
    df_merged_sorted = df_merged.sort_values(
        by=["category_name", "rating", "price_gbp"],
        ascending=[True, False, True],
    )
    df_merged_sorted["rank_in_category"] = (
        df_merged_sorted.groupby("category_name").cumcount() + 1
    )
    df_top3_via_pandas = df_merged_sorted[df_merged_sorted["rank_in_category"] <= 3]

    df_top3_via_pandas = df_top3_via_pandas[
        ["category_name", "title", "rating", "price_gbp"]
    ].reset_index(drop=True)

    print("=" * 70)
    print("pd.merge -> top 3 highest-rated books per category (pure pandas, no SQL JOIN)")
    print("=" * 70)
    print(df_top3_via_pandas.to_string(index=False))
    print(f"({len(df_top3_via_pandas)} rows)\n")

    # ------------------------------------------------------------------
    # 3. Confirm both approaches produce equivalent output.
    # ------------------------------------------------------------------
    df_sql_comparable = df_join_via_sql.reset_index(drop=True)
    df_pandas_comparable = df_top3_via_pandas.reset_index(drop=True)

    are_equal = df_sql_comparable.equals(df_pandas_comparable)
    print("=" * 70)
    print(f"SQL JOIN result matches pandas merge result: {are_equal}")
    print("=" * 70)

    conn.close()
    return are_equal


if __name__ == "__main__":
    main()
