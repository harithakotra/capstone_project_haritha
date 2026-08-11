# data_pipeline — Catalog Scrape → Clean → Convert → Store → Query

A raw-to-relational data pipeline that scrapes book catalog data from
[books.toscrape.com](http://books.toscrape.com) (a public scraping-practice
site), cleans and type-converts it, applies a fixed-rate GBP→INR currency
conversion, loads it into a normalized SQLite database, and queries it with
both SQL and pandas.

## Pipeline overview

```
scrape.py            -->  raw_books.csv
clean_transform.py    -->  cleaned_books.csv
build_database.py      -->  books.db  (SQLite, 2-table normalized schema)
run_queries.py          -->  queries_output.txt  (5+ required SQL queries)
pandas_verify.py         -->  console output (pd.read_sql vs pd.merge)
```

## Install & run

```bash
pip install requests beautifulsoup4 pandas

# 1. Scrape live data (requires normal internet access to books.toscrape.com)
python scrape.py

# 2. Clean & type-convert
python clean_transform.py

# 3. Build the normalized SQLite database from the cleaned CSV
python build_database.py

# 4. Run and save the required SQL queries
python run_queries.py

# 5. Verify SQL JOIN vs pandas merge produce equivalent results
python pandas_verify.py
```

Each script is idempotent and reads only the output of the previous step, so
you can re-run any stage independently as long as its input file/DB exists.

> **Note on this repository's included sample run:** the environment this
> pipeline was originally assembled in did not have outbound network access
> to books.toscrape.com, so `_demo_only_generate_sample_raw_csv.py` was used
> to fabricate a `raw_books.csv` in the *exact same shape* `scrape.py`
> produces (same columns, same `"£xx.xx"` price formatting, same
> `"One".."Five"` rating words, same availability phrasing) purely so
> `clean_transform.py` → `build_database.py` → `run_queries.py` →
> `pandas_verify.py` could be built and demonstrated end-to-end. Running
> `scrape.py` for real on a machine with internet access produces a
> `raw_books.csv` that is a drop-in replacement — no other script changes.
> `_demo_only_generate_sample_raw_csv.py` is **not** part of the graded
> pipeline itself.

## Scraping approach

`scrape.py` scrapes 5 category listing pages (the sidebar categories on the
homepage), following each category's "Next" pagination link until it runs
out, rather than scraping the mixed "All products" pages. This gives clean,
already-labeled `category` values with no extra parsing, and comfortably
clears the ≥60-books / ≥3-categories requirement.

For each book it captures: `title`, `price` (raw GBP text, e.g. `"£51.77"`),
`star_rating` (raw text, e.g. `"Three"`), `availability` (raw text, e.g.
`"In stock"`), and `category`.

## Cleaning & type-conversion decisions

| Raw field       | Cleaned column | Type  | Parsing rule |
|-----------------|----------------|-------|--------------|
| `price`         | `price_gbp`    | float | Strip the `£` symbol, cast to float |
| `star_rating`   | `rating`       | int   | Map word → number: One=1 … Five=5 |
| `availability`  | `in_stock`     | bool  | `True` if the text contains "in stock" (case-insensitive), else `False` |
| *(derived)*     | `price_inr`    | float | `price_gbp * 105.50` (fixed rate, see below) |

**Row-handling policy for unparseable fields** (assignment requires this be
stated and justified):

- **`price` / `availability` — drop the row.** These are treated as
  *structural* fields: a book's identity in this dataset is meaningless
  without a valid price and a valid stock status, and there's no sensible
  "typical" value to substitute that wouldn't risk quietly distorting price
  or availability analysis downstream. If the source page renders a
  never-seen-before format that can't be parsed at all, the safest thing is
  to drop that one row and keep the pipeline running rather than crash.

- **`star_rating` — median-impute.** Rating is a bounded ordinal scale
  (1–5), and a genuinely unparseable rating word doesn't invalidate the
  book's price/availability/category data, which remain useful. Rather than
  discard an otherwise-good row, a missing rating is filled with the
  **median of all successfully-parsed ratings** in the batch — a
  conservative, distribution-preserving choice that avoids skewing the
  rating column toward an arbitrary constant.

In the demo run included in this repo, 0 rows were dropped and 1 rating was
median-imputed (median = 3) — see the printed output of
`clean_transform.py`.

## Currency conversion (required, graded baseline)

```
price_inr = price_gbp * 105.50
```

**1 GBP = 105.50 INR** is a fixed, project-defined constant for this
assignment — not a live or historical market rate — so it requires no
external API call, no network access, and no date reference. This is the
exact rate used to compute every `price_inr` value in `books.db`, and it is
the only conversion path that is graded.

*(The assignment also allows an optional, ungraded stretch: looking up a
free keyless currency API with explicit HTTP status-code handling, falling
back to the fixed rate on any failure. That optional path was not exercised
here since it doesn't affect the required, graded `price_inr` values, which
use the fixed rate exclusively.)*

## Database schema

Two tables, linked by a primary/foreign key:

```sql
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
    in_stock    INTEGER NOT NULL,           -- 0/1
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
```

`build_database.py` always rebuilds `books.db` from scratch from
`cleaned_books.csv`, so the database is fully reproducible — you don't need
to commit the binary `.db` file if you'd rather regenerate it.

## Required SQL queries (`run_queries.py` → `queries_output.txt`)

| ID | Clause(s) demonstrated       | What it does |
|----|-------------------------------|--------------|
| Q1 | SELECT / WHERE                | Books priced under £20 |
| Q2 | ORDER BY / LIMIT               | 10 most expensive books overall |
| Q3 | DISTINCT + JOIN                | Distinct category names (via a join to `categories`) |
| Q4 | BETWEEN                       | Books priced between £25 and £45 |
| Q5 | IN                             | Books rated 4 or 5 stars |
| Q6 | JOIN                           | Top 3 highest-rated (then cheapest) books per category |

All six are executed and their full output is saved to
`queries_output.txt`.

## pandas verification (`pandas_verify.py`)

- Loads Q1 (books under £20) and Q6 (top-3-per-category JOIN) into
  DataFrames via `pd.read_sql(...)`.
- Independently reloads the raw `categories` and `books` tables into
  DataFrames and reproduces the Q6 result using **`pd.merge(...)`** plus a
  pandas `groupby`/`cumcount` rank — no SQL JOIN involved at all.
- Confirms both results are row-for-row identical
  (`df_sql.equals(df_pandas)` → `True` in the included demo run).

## Files in this module

```
data_pipeline/
├── scrape.py                              # real scraper (needs internet access)
├── _demo_only_generate_sample_raw_csv.py   # sandbox-only stand-in for scrape.py's output
├── clean_transform.py                      # cleaning + type conversion + price_inr
├── build_database.py                       # builds books.db from cleaned_books.csv
├── run_queries.py                          # runs + saves the 5+ required SQL queries
├── pandas_verify.py                        # pd.read_sql vs pd.merge equivalence check
├── raw_books.csv                           # scraped output (sample run included)
├── cleaned_books.csv                       # cleaned output (sample run included)
├── books.db                                # SQLite database (sample run included)
├── queries_output.txt                      # saved query output (sample run included)
└── README.md                               # this file
```
