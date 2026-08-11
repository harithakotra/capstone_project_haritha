"""
_demo_only_generate_sample_raw_csv.py
--------------------------------------
NOT part of the graded submission.

This sandbox does not have network access to books.toscrape.com, so this
script fabricates a raw_books.csv with the exact same shape/format that
scrape.py produces (same column names, same "£xx.xx" price formatting, same
"One".."Five" rating words, same availability phrasing), purely so the rest
of the pipeline (clean_transform.py, build_database.py, run_queries.py,
pandas_verify.py) can be built, run, and demonstrated end-to-end here.

When you run scrape.py on a machine with normal internet access, it will
produce a real raw_books.csv that is a drop-in replacement for this one --
no other script needs to change.
"""

import csv
import random

random.seed(42)

CATEGORIES = ["Travel", "Mystery", "Historical Fiction", "Sequential Art", "Classics"]

TITLE_WORDS_A = ["Shadow", "Silent", "Broken", "Hidden", "Last", "Golden", "Winter",
                  "Forgotten", "Secret", "Distant", "Endless", "Quiet", "Wild", "Lost"]
TITLE_WORDS_B = ["Garden", "River", "City", "House", "Journey", "Kingdom", "Letters",
                  "Road", "Sky", "Harbor", "Forest", "Portrait", "Chronicle", "Voyage"]

RATINGS = ["One", "Two", "Three", "Four", "Five"]

# A couple of intentionally messy rows to prove the cleaning step is robust,
# e.g. an availability string with unusual whitespace and a rating word that
# doesn't map cleanly.
MESSY_INJECTIONS = [
    {"star_rating": "Zero", "availability": "In stock"},   # invalid rating word
    {"star_rating": "Three", "availability": "Unavailable"},  # unusual availability text
]

rows = []
book_num = 0
for category in CATEGORIES:
    n_books = random.randint(13, 18)  # ensures >=60 total across 5 categories
    for _ in range(n_books):
        book_num += 1
        title = f"{random.choice(TITLE_WORDS_A)} {random.choice(TITLE_WORDS_B)} {book_num}"
        price = round(random.uniform(10.0, 60.0), 2)
        rating = random.choice(RATINGS)
        # Real site phrasing is "In stock" for in-stock items.
        availability = "In stock"

        rows.append({
            "title": title,
            "price": f"£{price:.2f}",
            "star_rating": rating,
            "availability": availability,
            "category": category,
        })

# Inject a couple of messy rows into the mix to exercise the cleaning logic.
for messy in MESSY_INJECTIONS:
    book_num += 1
    rows.append({
        "title": f"Messy Row {book_num}",
        "price": "£25.00",
        "star_rating": messy["star_rating"],
        "availability": messy["availability"],
        "category": "Travel",
    })

with open("raw_books.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "price", "star_rating", "availability", "category"])
    writer.writeheader()
    writer.writerows(rows)

print(f"[DEMO ONLY] Generated raw_books.csv with {len(rows)} rows across {len(CATEGORIES)} categories.")
