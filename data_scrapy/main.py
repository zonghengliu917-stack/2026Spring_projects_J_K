import argparse
import csv
import os
from typing import Optional

import requests
from bs4 import BeautifulSoup, Tag


BASE_URL = "https://m.compassedu.hk"
DETAIL_URL_TEMPLATE = BASE_URL + "/newst/{id}"


def clean_text(text: str) -> str:
    """Collapse all whitespace in *text* to single spaces and strip leading/trailing whitespace."""
    return " ".join(text.split())


def fetch_html(session: requests.Session, url: str) -> str:
    """Fetch the HTML content of *url* using *session*.

    Args:
        session: A ``requests.Session`` object used for the HTTP request.
        url: The fully-qualified URL to fetch.

    Returns:
        The response body as a decoded string.

    Raises:
        requests.HTTPError: If the server returns a 4xx or 5xx status code.
    """
    resp = session.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text


def is_valid_detail_page(soup: BeautifulSoup) -> bool:
    """Return True if *soup* contains a CompassEdu offer detail panel with an info-box."""
    return bool(soup.select_one(".module-panel .info-box"))


def find_info_panel(soup: BeautifulSoup) -> Optional[Tag]:
    """Return the first ``.module-panel`` element that contains an ``.info-box``, or None."""
    for panel in soup.select(".module-panel"):
        if panel.select_one(".info-box"):
            return panel
    return None


def find_experience_panel(soup: BeautifulSoup) -> Optional[Tag]:
    """Return the first ``.module-panel`` containing experience items, or None.

    Looks for either ``.experience_box`` or ``.experience_text`` child elements.
    """
    for panel in soup.select(".module-panel"):
        if panel.select(".experience_box") or panel.select(".experience_text"):
            return panel
    return None


def parse_detail_page(html: str) -> dict[str, str]:
    """Parse a CompassEdu offer detail page and return its fields as a flat dict.

    Extracts the page title, the last five info-box rows (label → value), and any
    numbered experience items joined into a single ``"experience"`` string.

    Args:
        html: Raw HTML string of the detail page.

    Returns:
        A dict mapping field names (e.g. ``"录取学校"``) to their string values.
        The ``"experience"`` key is present only when experience items are found.
    """
    soup = BeautifulSoup(html, "html.parser")
    data: dict[str, str] = {}

    title_el = soup.select_one("h1")
    if title_el:
        data["title"] = clean_text(title_el.get_text(" ", strip=True))

    info_panel = find_info_panel(soup)
    if info_panel:
        rows = info_panel.select(".info-box .row")
        rows = rows[-5:] if len(rows) >= 5 else rows
        for row in rows:
            name_el = row.select_one(".name")
            detail_el = row.select_one(".detail")
            if not name_el or not detail_el:
                continue
            name = clean_text(name_el.get_text(" ", strip=True))
            detail = clean_text(detail_el.get_text(" ", strip=True))
            if name:
                data[name] = detail

    experience_panel = find_experience_panel(soup)
    exp_items: list[str] = []
    if experience_panel:
        texts = [
            clean_text(t.get_text(" ", strip=True))
            for t in experience_panel.select(".experience_text")
        ]
        if not texts:
            texts = [
                clean_text(t.get_text(" ", strip=True))
                for t in experience_panel.select(".experience_box")
            ]
        for idx, text in enumerate([t for t in texts if t], start=1):
            exp_items.append(f"{idx}. {text}")
    if exp_items:
        data["experience"] = "\n".join(exp_items)

    return data


def write_csv(rows: list[dict[str, str]], output_path: str) -> None:
    """Write *rows* to a CSV file at *output_path*, creating parent directories as needed.

    Column order is determined by the union of all keys across *rows*, in the order
    they are first encountered.

    Args:
        rows: List of dicts where each dict represents one scraped offer record.
        output_path: Destination file path (relative or absolute).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    """Entry point: parse CLI arguments and scrape CompassEdu offer detail pages.

    Iterates from ``--start-id`` down to ``--min-id``, fetching each
    ``/newst/{id}`` page. Valid pages are parsed and appended to the output CSV
    incrementally. Stops early after ``--max-consecutive-fails`` consecutive
    missing or invalid pages.
    """
    parser = argparse.ArgumentParser(
        description="Scrape CompassEdu offer list and detail pages."
    )
    parser.add_argument(
        "--output",
        default="output/compass_offers.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=49554,
        help="Start ID for /newst/{id}.",
    )
    parser.add_argument(
        "--min-id",
        type=int,
        default=0,
        help="Minimum ID to scan down to (inclusive).",
    )
    parser.add_argument(
        "--max-consecutive-fails",
        type=int,
        default=50,
        help="Stop after N consecutive missing/invalid pages.",
    )
    args = parser.parse_args()

    session = requests.Session()
    all_rows: list[dict] = []
    consecutive_fails = 0

    for current_id in range(args.start_id, args.min_id - 1, -1):
        url = DETAIL_URL_TEMPLATE.format(id=current_id)
        try:
            html = fetch_html(session, url)
        except requests.HTTPError:
            consecutive_fails += 1
            print(f"{current_id}: 页面不存在，连续失败 {consecutive_fails}")
            if consecutive_fails >= args.max_consecutive_fails:
                break
            continue

        soup = BeautifulSoup(html, "html.parser")
        if not is_valid_detail_page(soup):
            consecutive_fails += 1
            print(f"{current_id}: 无有效内容，连续失败 {consecutive_fails}")
            if consecutive_fails >= args.max_consecutive_fails:
                break
            continue

        consecutive_fails = 0
        all_rows.append(parse_detail_page(html))
        write_csv(all_rows, args.output)
        print(f"{current_id}: 已保存，累计 {len(all_rows)} 条")
    print("抓取完成")
    print(f"共获取 {len(all_rows)} 条数据")
    print(f"已保存到 {args.output}")


if __name__ == "__main__":
    main()
