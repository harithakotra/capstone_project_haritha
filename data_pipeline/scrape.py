"""
scrape.py
---------
Scrapes book data (title, price, star rating, availability, category) from
books.toscrape.com, a public site built specifically for scraping practice.

Strategy: walk at least 3 category listing pages (following "Next" pagination
links inside each category) so we get a naturally category-grouped dataset,
rather than scraping the mixed "All products" pages. This guarantees clean
category labels without needing to visit each book's detail page.

Run this on a machine with normal internet access:
    python scrape.py

Output: raw_books.csv  (title, price, star_rating, availability, category)
"""

import csv
import time
import sys
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
HOME_URL = BASE_URL + "index.html"

# How many categories to scrape. The assignment requires >= 3 categories and
# >= 60 total books. We scrape 5 categories to comfortably clear both bars
# even if some categories are small.
NUM_CATEGORIES = 5

# Be a polite scraper: pause briefly between requests.
REQUEST_DELAY_SECONDS = 0.5


def get_session():
    """A requests.Session reuses the underlying TCP connection across
    requests, which is faster and friendlier to the server than opening a
    fresh connection every time."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (educational scraping exercise)"})
    return session


def get_category_links(session):
    """
    Fetch the homepage and pull out the list of (category_name, category_url)
    pairs from the left-hand sidebar navigation.
    """
    resp = session.get(HOME_URL, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # The sidebar categories live under div.side_categories ul.nav-list -> nested <li><a>
    links = soup.select("div.side_categories ul.nav-list li ul li a")

    categories = []
    for a in links:
        name = a.get_text(strip=True)
        href = a["href"]  # relative, e.g. "category/books/travel_2/index.html"
        full_url = BASE_URL + "catalogue/" + href if not href.startswith("catalogue") else BASE_URL + href
        categories.append((name, full_url))
    return categories


def parse_rating(rating_p_tag):
    """
    The star rating is encoded as a CSS class, e.g. <p class="star-rating Three">.
    We pull the word ("One".."Five") out of the class list.
    """
    classes = rating_p_tag.get("class", [])
    for c in classes:
        if c != "star-rating":
            return c
    return None


def scrape_category(session, category_name, start_url, rows):
    """
    Scrape every book on every paginated page of a single category,
    following the "Next" link until it disappears.
    """
    next_url = start_url
    page_count = 0

    while next_url:
        resp = session.get(next_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        pods = soup.select("article.product_pod")
        for pod in pods:
            title = pod.h3.a["title"]  # full, untruncated title lives in the title attribute
            price_text = pod.select_one("p.price_color").get_text(strip=True)
            rating_text = parse_rating(pod.select_one("p.star-rating"))
            availability_text = pod.select_one("p.instock.availability").get_text(strip=True)

            rows.append({
                "title": title,
                "price": price_text,
                "star_rating": rating_text,
                "availability": availability_text,
                "category": category_name,
            })

        page_count += 1

        # Look for a "Next" pagination link; if present, build the next URL.
        next_link = soup.select_one("li.next a")
        if next_link:
            # next_url's directory + the relative href of the "next" link
            base_dir = next_url.rsplit("/", 1)[0]
            next_url = base_dir + "/" + next_link["href"]
            time.sleep(REQUEST_DELAY_SECONDS)
        else:
            next_url = None

    print(f"  '{category_name}': scraped {page_count} page(s), running total rows so far captured.")


def main():
    session = get_session()

    print("Fetching category list from homepage...")
    all_categories = get_category_links(session)
    if not all_categories:
        print("ERROR: could not find any categories on the homepage. Site structure may have changed.")
        sys.exit(1)

    # Pick the first NUM_CATEGORIES categories that appear in the sidebar.
    chosen_categories = all_categories[:NUM_CATEGORIES]
    print(f"Chosen categories: {[c[0] for c in chosen_categories]}")

    rows = []
    for name, url in chosen_categories:
        print(f"Scraping category: {name}")
        scrape_category(session, name, url, rows)

    if len(rows) < 60:
        print(f"WARNING: only scraped {len(rows)} rows across {len(chosen_categories)} categories. "
              f"Increase NUM_CATEGORIES to pick up more books.")
    else:
        print(f"Done. Scraped {len(rows)} total book rows across {len(chosen_categories)} categories.")

    fieldnames = ["title", "price", "star_rating", "availability", "category"]
    with open("raw_books.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved raw_books.csv")


if __name__ == "__main__":
    main()
