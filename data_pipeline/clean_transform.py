"""
clean_transform.py
-------------------
Reads raw_books.csv (produced by scrape.py) and cleans it into proper types:

    price_gbp   float   - stripped of the "£" symbol
    rating      int     - "One".."Five" -> 1..5
    in_stock    bool    - parsed from the availability text
    price_inr   float   - price_gbp * FIXED_GBP_TO_INR_RATE

Fixed conversion rate (per assignment spec): 1 GBP = 105.50 INR.
This is an artificial, project-defined constant -- NOT a live market rate --
so no API call or date reference is needed for the required, graded path.

Row-handling policy for unparseable fields (stated + justified in README):
    - rating: if the star-rating text isn't one of One/Two/Three/Four/Five,
      we treat it as missing and MEDIAN-IMPUTE the integer rating. Ratings
      are a bounded ordinal scale (1-5), so imputing with the dataset median
      preserves the row (and its price/availability/category data, which are
      still valid and useful) rather than discarding otherwise-good data.
    - price / availability: these are considered essential, structural
      fields. If either is missing or fails to parse, we DROP the row,
      since there is no sensible "typical" price or stock status to impute
      that wouldn't risk misleading downstream analysis.

Output: cleaned_books.csv
"""

import csv
import statistics

RAW_INPUT_FILE = "raw_books.csv"
CLEANED_OUTPUT_FILE = "cleaned_books.csv"

# --- Fixed, project-defined conversion rate (required baseline) ---
FIXED_GBP_TO_INR_RATE = 105.50

RATING_WORD_TO_INT = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def parse_price(price_text):
    """
    Strip the currency symbol (and any stray whitespace) and convert to float.
    Returns None if the value can't be parsed, so the caller can decide to
    drop the row.
    """
    if not price_text:
        return None
    cleaned = price_text.strip().lstrip("£").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_rating(rating_text):
    """
    Convert 'One'..'Five' to 1..5. Returns None for anything else (e.g. an
    unexpected word), leaving it to be median-imputed later.
    """
    if rating_text is None:
        return None
    return RATING_WORD_TO_INT.get(rating_text.strip(), None)


def parse_in_stock(availability_text):
    """
    The real site's availability text is phrased like 'In stock (22 available)'
    or simply 'In stock'. Anything containing 'in stock' (case-insensitive) is
    treated as True; anything else ('Out of stock', 'Unavailable', etc.) is
    treated as False. Returns None only if the field is completely empty,
    so the caller can drop that row (availability is a structural field).
    """
    if not availability_text or not availability_text.strip():
        return None
    return "in stock" in availability_text.strip().lower()


def main():
    with open(RAW_INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    print(f"Loaded {len(raw_rows)} raw rows from {RAW_INPUT_FILE}")

    parsed_rows = []
    dropped_count = 0

    for row in raw_rows:
        price_gbp = parse_price(row.get("price"))
        in_stock = parse_in_stock(row.get("availability"))

        # price and availability are structural: drop the row if either fails.
        if price_gbp is None or in_stock is None:
            dropped_count += 1
            continue

        rating = parse_rating(row.get("star_rating"))  # may be None -> imputed below

        parsed_rows.append({
            "title": row.get("title"),
            "price_gbp": price_gbp,
            "rating": rating,  # placeholder, possibly None
            "in_stock": in_stock,
            "category": row.get("category"),
        })

    print(f"Dropped {dropped_count} row(s) with unparseable price/availability.")

    # --- Median-impute any missing ratings ---
    known_ratings = [r["rating"] for r in parsed_rows if r["rating"] is not None]
    if known_ratings:
        median_rating = round(statistics.median(known_ratings))
    else:
        median_rating = 3  # neutral fallback if somehow every rating failed to parse

    imputed_count = 0
    for r in parsed_rows:
        if r["rating"] is None:
            r["rating"] = median_rating
            imputed_count += 1

    print(f"Median-imputed {imputed_count} missing rating value(s) using median = {median_rating}.")

    # --- Fixed-rate currency conversion (required, graded path) ---
    for r in parsed_rows:
        r["price_inr"] = round(r["price_gbp"] * FIXED_GBP_TO_INR_RATE, 2)

    # --- Write cleaned CSV ---
    fieldnames = ["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]
    with open(CLEANED_OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in parsed_rows:
            writer.writerow({
                "title": r["title"],
                "category": r["category"],
                "price_gbp": r["price_gbp"],
                "price_inr": r["price_inr"],
                "rating": r["rating"],
                "in_stock": int(r["in_stock"]),  # store as 0/1 for CSV/SQLite friendliness
            })

    print(f"Wrote {len(parsed_rows)} cleaned rows to {CLEANED_OUTPUT_FILE}")
    categories = sorted(set(r["category"] for r in parsed_rows))
    print(f"Categories present: {categories}")


if __name__ == "__main__":
    main()
