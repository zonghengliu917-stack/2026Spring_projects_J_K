"""
This is the preprocessing of the dataset, which focuses on matching the school names between the offer dataset 
and the cost dataset, and building a cost profile for each school. It also standardizes the GPA to a 4.0 scale 
for better comparison. The output is a CSV file with matched schools and their cost profiles, which can be used 
for further analysis.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


_HERE = Path(__file__).parent
DEFAULT_OFFERS = _HERE / "compass_offers_processed.csv"
DEFAULT_COSTS = _HERE / "International_Education_Costs.csv"

# Many schools have multiple names or variations, I record them here manually.
MANUAL_ALIASES = {
    "hong kong university of science and technology": "HKUST",
    "hkust": "HKUST",
    "university of michigan ann arbor": "University of Michigan",
    "university of wisconsin madison": "University of Wisconsin-Madison",
    "wisconsin madison": "University of Wisconsin-Madison",
    "ohio state university": "Ohio State University",
    # Confirmed from fuzzy_review — exact names verified against cost CSV
    "the university of texas at austin": "University of Texas Austin",
    "royal holloway university of london": "Royal Holloway University",
    "university of colorado boulder": "University of Colorado",
}


@dataclass
class MatchRecord:
    offer_school: str
    matched_school: str
    offer_count: int
    match_type: str


NUM_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def contains_phd_text(text: str) -> bool:
    """Return True if *text* indicates a PhD-level program.

    PhD offers are excluded from this analysis because they are typically
    funded by scholarships and have a fundamentally different cost structure
    from self-funded master's programs.

    Checks for the Chinese character '博士', the substring 'phd', and 'doctor'
    (case-insensitive).
    """
    s = (text or "").strip()
    if not s:
        return False
    low = s.lower()
    if "博士" in s:
        return True
    if "phd" in low:
        return True
    if "doctor" in low:
        return True
    return False


def is_offer_phd_row(row: Dict[str, str]) -> bool:
    """Return True if the offer row represents a PhD admission.

    Checks both the English (``录取专业_en``) and Chinese (``录取专业``) major
    columns using :func:`contains_phd_text`.
    """
    major_en = row.get("录取专业_en") or ""
    major_cn = row.get("录取专业") or ""
    return contains_phd_text(major_en) or contains_phd_text(major_cn)


def is_cost_phd_row(row: Dict[str, str]) -> bool:
    """Return True if the cost dataset row represents a PhD-level program.

    Checks the ``Level`` and ``Program`` columns using :func:`contains_phd_text`.
    """
    level = row.get("Level") or ""
    program = row.get("Program") or ""
    return contains_phd_text(level) or contains_phd_text(program)


def is_cost_master_row(row: Dict[str, str]) -> bool:
    """Return True if the cost dataset row represents a master's-level program.

    Matches any ``Level`` value that contains the substring ``'master'``
    (case-insensitive), e.g. ``"Master's"``, ``"Masters"``.
    """
    level = (row.get("Level") or "").strip().lower()
    return "master" in level


def has_chinese_char(text: str) -> bool:
    """Return True if *text* contains at least one CJK Unified Ideograph (U+4E00–U+9FFF)."""
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def is_chinese_student_row(row: Dict[str, str]) -> bool:
    """Return True if the offer row likely belongs to a Chinese domestic student.

    Heuristic: the undergraduate school name (``毕业学校``) contains at least
    one Chinese character.  This is used to decide which GPA conversion rule
    to apply when the raw GPA is on a 100-point scale.
    """
    school_cn = row.get("毕业学校") or ""
    if has_chinese_char(school_cn):
        return True
    return False


def parse_numeric(text: str) -> Optional[float]:
    """Extract and return the first numeric value found in *text*, or None.

    Handles integers and decimals, including negative numbers.
    Returns None if *text* is empty, ``'nan'``, or contains no numeric pattern.
    """
    s = (text or "").strip()
    if not s or s.lower() == "nan":
        return None
    m = NUM_PATTERN.search(s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def pct_to_gpa4_china_linear(score: float) -> float:
    """Convert a Chinese 100-point score to a 4.0-scale GPA using the linear formula.

    Formula: ``GPA = (score - 50) / 10``, clipped to the range ``[0.0, 4.0]``.
    This is the common mainland Chinese conversion used in many application guidelines.
    """
    # Common mainland conversion: GPA = (score - 50) / 10, clipped to [0, 4].
    gpa = (score - 50.0) / 10.0
    if gpa < 0:
        gpa = 0.0
    if gpa > 4.0:
        gpa = 4.0
    return gpa


def pct_to_gpa4_china_table(score: float) -> float:
    """Convert a Chinese 100-point score to a 4.0-scale GPA using a step-table lookup.

    Uses the stepped conversion table commonly referenced in Chinese university
    application materials (e.g. 90+ → 4.0, 85–89 → 3.7, …, below 60 → 0.0).
    """
    # Chinese step conversion table often used in school applications.
    if score >= 90:
        return 4.0
    if score >= 85:
        return 3.7
    if score >= 82:
        return 3.3
    if score >= 78:
        return 3.0
    if score >= 75:
        return 2.7
    if score >= 72:
        return 2.3
    if score >= 68:
        return 2.0
    if score >= 64:
        return 1.5
    if score >= 60:
        return 1.0
    return 0.0


def standardize_gpa_to_4(
    row: Dict[str, str], gpa_rule: str
) -> Tuple[str, str, str]:
    """
    Returns (gpa_numeric_raw, gpa_4_standardized, gpa_status)
    """
    raw = parse_numeric(row.get("GPA") or "")
    if raw is None:
        return "", "", "missing_or_invalid"

    # Already on a 4.x scale.
    if 0 <= raw <= 4.5:
        gpa4 = min(raw, 4.0)
        return f"{raw:.4f}", f"{gpa4:.4f}", "already_4_scale"

    # Convert percentage scale for Chinese students.
    if 50 <= raw <= 100:
        if is_chinese_student_row(row):
            if gpa_rule == "china_table":
                gpa4 = pct_to_gpa4_china_table(raw)
            else:
                gpa4 = pct_to_gpa4_china_linear(raw)
            return f"{raw:.4f}", f"{gpa4:.4f}", "converted_from_100"
        return f"{raw:.4f}", "", "percent_non_china_not_converted"

    return f"{raw:.4f}", "", "out_of_range"


def normalize_school_name(name: str) -> str:
    """Return a normalized version of *name* for deterministic school-name matching.

    Transformations applied (in order):
    1. Strip whitespace and lowercase.
    2. Replace em-dash / en-dash with a hyphen.
    3. Remove a leading ``'the '``.
    4. Replace ``'&'`` with ``'and'``.
    5. Remove all characters except letters, digits, spaces, and hyphens.
    6. Replace hyphens with spaces and collapse repeated whitespace.
    """
    # Normalize school names for deterministic matching.
    s = (name or "").strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"^the\s+", "", s)
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_school_counter(path: Path, school_col: str, encoding: str) -> Tuple[int, int, Counter]:
    """Read *path* and return row count, missing-school count, and per-school frequency.

    Args:
        path: Path to the CSV file.
        school_col: Name of the column containing school names.
        encoding: File encoding (e.g. ``'utf-8-sig'``).

    Returns:
        A tuple of ``(total_rows, missing_school_rows, school_counter)`` where
        ``school_counter`` maps each school name to its occurrence count.

    Raises:
        ValueError: If *school_col* is not found in the CSV header.
    """
    row_count = 0
    missing_school = 0
    school_counter: Counter = Counter()

    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        if school_col not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{school_col}' not found in {path}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            row_count += 1
            school = (row.get(school_col) or "").strip()
            if not school:
                missing_school += 1
                continue
            school_counter[school] += 1

    return row_count, missing_school, school_counter


def read_csv_rows(path: Path, encoding: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read all rows from *path* and return the header and row list.

    Args:
        path: Path to the CSV file.
        encoding: File encoding (e.g. ``'utf-8-sig'``).

    Returns:
        A tuple of ``(fieldnames, rows)`` where *fieldnames* is the ordered list
        of column headers and *rows* is a list of ``DictReader`` row dicts.
    """
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def build_school_counter(rows: List[Dict[str, str]], school_col: str) -> Tuple[int, Counter]:
    """Count per-school frequencies from an already-loaded list of rows.

    Args:
        rows: List of row dicts (e.g. from :func:`read_csv_rows`).
        school_col: Name of the column containing school names.

    Returns:
        A tuple of ``(missing_school_count, school_counter)`` where
        ``school_counter`` maps each non-empty school name to its occurrence count.
    """
    missing_school = 0
    school_counter: Counter = Counter()
    for row in rows:
        school = (row.get(school_col) or "").strip()
        if not school:
            missing_school += 1
            continue
        school_counter[school] += 1
    return missing_school, school_counter


def unique_school_list(rows: List[Dict[str, str]], school_col: str) -> List[str]:
    """Return a deduplicated list of school names preserving first-seen order.

    Args:
        rows: List of row dicts.
        school_col: Name of the column containing school names.

    Returns:
        Unique, non-empty school names in the order they first appear in *rows*.
    """
    schools: List[str] = []
    seen = set()
    for row in rows:
        school = (row.get(school_col) or "").strip()
        if school and school not in seen:
            seen.add(school)
            schools.append(school)
    return schools


def to_float(value: str) -> float:
    """Convert *value* to float, returning 0.0 if conversion fails or value is empty."""
    try:
        return float((value or "").strip())
    except (ValueError, AttributeError):
        return 0.0


def calc_total_cost_usd(row: Dict[str, str]) -> float:
    """Calculate the estimated total cost of attendance in USD for one cost-dataset row.

    Formula: ``tuition * duration + rent * 12 * duration + visa + insurance``

    All fields are read via :func:`to_float` so missing values default to 0.

    Args:
        row: A cost-dataset row dict containing ``Duration_Years``, ``Tuition_USD``,
             ``Rent_USD``, ``Visa_Fee_USD``, and ``Insurance_USD`` keys.

    Returns:
        Total estimated cost in USD as a float.
    """
    # calculate total cost in USD based on tuition, living cost, visa, insurance, and duration.
    duration = to_float(row.get("Duration_Years", ""))
    tuition = to_float(row.get("Tuition_USD", ""))
    rent = to_float(row.get("Rent_USD", ""))
    visa = to_float(row.get("Visa_Fee_USD", ""))
    insurance = to_float(row.get("Insurance_USD", ""))
    return tuition * duration + rent * 12.0 * duration + visa + insurance


def build_school_cost_profile(
    cost_rows: List[Dict[str, str]],
    school_col: str,
) -> Dict[str, Dict[str, str]]:
    """Build a cost profile dict keyed by school name from the cost dataset.

    For each school, selects the single most representative row:
    - Prefers master's-level rows over other levels.
    - Among the selected pool, picks the row whose total cost is closest to the
      pool's median total cost.

    Args:
        cost_rows: Filtered cost-dataset rows (PhD rows already removed).
        school_col: Name of the column containing school names.

    Returns:
        A dict mapping each school name to a flat dict of cost profile fields
        (e.g. ``cost_tuition_usd``, ``cost_total_usd``, ``cost_country``, …).
    """
    # build the profile for each school based on cost rows. If multiple rows exist for the same school, prefer master level, otherwise use median total cost to pick the most representative row.
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in cost_rows:
        school = (row.get(school_col) or "").strip()
        if school:
            grouped[school].append(row)

    profiles: Dict[str, Dict[str, str]] = {}
    for school, rows in grouped.items():
        master_rows = [r for r in rows if is_cost_master_row(r)]
        pool = master_rows if master_rows else rows

        totals = [calc_total_cost_usd(r) for r in pool]
        median_total = statistics.median(totals)
        chosen = min(pool, key=lambda r: abs(calc_total_cost_usd(r) - median_total))

        profiles[school] = {
            "cost_rows_non_phd": str(len(rows)),
            "cost_master_rows_non_phd": str(len(master_rows)),
            "cost_profile_source": "master_preferred" if master_rows else "non_phd_fallback",
            "cost_country": chosen.get("Country", ""),
            "cost_city": chosen.get("City", ""),
            "cost_university": chosen.get("University", ""),
            "cost_program": chosen.get("Program", ""),
            "cost_level": chosen.get("Level", ""),
            "cost_duration_years": chosen.get("Duration_Years", ""),
            "cost_tuition_usd": chosen.get("Tuition_USD", ""),
            "cost_living_cost_index": chosen.get("Living_Cost_Index", ""),
            "cost_rent_usd": chosen.get("Rent_USD", ""),
            "cost_visa_fee_usd": chosen.get("Visa_Fee_USD", ""),
            "cost_insurance_usd": chosen.get("Insurance_USD", ""),
            "cost_exchange_rate": chosen.get("Exchange_Rate", ""),
            "cost_total_usd": f"{median_total:.2f}",
        }
    return profiles


def read_cost_schools(path: Path, school_col: str, encoding: str) -> List[str]:
    """Return a deduplicated list of school names from the cost CSV, preserving order.

    Args:
        path: Path to the cost CSV file.
        school_col: Name of the column containing school names.
        encoding: File encoding (e.g. ``'utf-8-sig'``).

    Returns:
        Unique, non-empty school names in the order they first appear.

    Raises:
        ValueError: If *school_col* is not found in the CSV header.
    """
    schools: List[str] = []
    seen = set()
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        if school_col not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{school_col}' not found in {path}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            school = (row.get(school_col) or "").strip()
            if school and school not in seen:
                seen.add(school)
                schools.append(school)
    return schools


def best_candidates(
    offer_school: str,
    cost_schools: Sequence[str],
    top_n: int,
    cutoff: float,
) -> List[Tuple[str, float]]:
    """Return the top fuzzy-match candidates for *offer_school* from *cost_schools*.

    Uses :func:`difflib.get_close_matches` for candidate retrieval, then ranks
    results by :class:`difflib.SequenceMatcher` similarity ratio (descending).

    Args:
        offer_school: The school name from the offers dataset to match.
        cost_schools: Sequence of school names from the cost dataset.
        top_n: Maximum number of candidates to return.
        cutoff: Minimum similarity ratio threshold (0–1) for a candidate to qualify.

    Returns:
        List of ``(school_name, similarity_score)`` tuples sorted by score descending.
        Empty list if no candidates meet the cutoff.
    """
    cands = difflib.get_close_matches(offer_school, cost_schools, n=top_n, cutoff=cutoff)
    scored = []
    for c in cands:
        score = difflib.SequenceMatcher(None, offer_school.lower(), c.lower()).ratio()
        scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def resolve_costs_path(offers_path: Path, costs_arg: Optional[Path]) -> Tuple[Path, str]:
    """
    Resolve the costs path.
    Priority:
    1) explicit --costs
    2) offers_dir / International_Education_Costs.csv
    3) parent_of_offers_dir / International_Education_Costs.csv
    4) hardcoded default (if exists)
    """
    if costs_arg is not None:
        if costs_arg.exists():
            return costs_arg, "explicit --costs"
        raise FileNotFoundError(f"Costs file not found: {costs_arg}")

    candidate_names = [
        "International_Education_Costs.csv",
        "international_education_costs.csv",
    ]
    search_dirs = [
        offers_path.parent,
        offers_path.parent.parent,
    ]
    for folder in search_dirs:
        for name in candidate_names:
            candidate = folder / name
            if candidate.exists():
                return candidate, f"auto-detected in {folder}"

    if DEFAULT_COSTS.exists():
        return DEFAULT_COSTS, "fallback default path"

    searched_text = ", ".join(str(x) for x in search_dirs)
    raise FileNotFoundError(
        "Could not find International_Education_Costs.csv automatically. "
        f"Searched in: {searched_text}. Please pass --costs <path>."
    )


def main() -> None:
    """Entry point: match offer schools to cost schools and export an enriched CSV.

    Reads the offers and costs CSV files, filters out PhD rows, runs a four-stage
    school-name matching pipeline (exact → alias → normalized → fuzzy), standardizes
    GPA values to a 4.0 scale, and writes the matched output CSV.  Prints a
    human-readable summary of match coverage and GPA conversion statistics to stdout.
    """
    parser = argparse.ArgumentParser(
        description="Check school matching quality between offer and cost datasets."
    )
    parser.add_argument("--offers", type=Path, default=DEFAULT_OFFERS)
    parser.add_argument(
        "--costs",
        type=Path,
        default=None,
        help="Optional. If omitted, auto-detect from offers folder and its parent.",
    )
    parser.add_argument("--offer-school-col", default="录取学校_en")
    parser.add_argument("--cost-school-col", default="University")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--top-schools", type=int, default=25)
    parser.add_argument("--top-suspicious", type=int, default=40)
    parser.add_argument("--fuzzy-cutoff", type=float, default=0.78)
    parser.add_argument("--fuzzy-top-n", type=int, default=3)
    parser.add_argument("--fuzzy-auto-threshold", type=float, default=0.92)
    parser.add_argument("--fuzzy-min-gap", type=float, default=0.03)
    parser.add_argument(
        "--gpa-rule",
        choices=["china_linear", "china_table"],
        default="china_linear",
        help=(
            "GPA conversion rule for percentage scores in Chinese students. "
            "china_linear: (score-50)/10; china_table: step table."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <offers_stem>_matched_no_phd.csv",
    )
    args = parser.parse_args()

    if not args.offers.exists():
        raise FileNotFoundError(f"Offers file not found: {args.offers}")
    costs_path, costs_source = resolve_costs_path(args.offers, args.costs)

    offer_fieldnames, offer_rows_all = read_csv_rows(args.offers, args.encoding)
    if args.offer_school_col not in offer_fieldnames:
        raise ValueError(
            f"Column '{args.offer_school_col}' not found in {args.offers}. "
            f"Available columns: {offer_fieldnames}"
        )

    cost_fieldnames, cost_rows_all = read_csv_rows(costs_path, args.encoding)
    if args.cost_school_col not in cost_fieldnames:
        raise ValueError(
            f"Column '{args.cost_school_col}' not found in {costs_path}. "
            f"Available columns: {cost_fieldnames}"
        )

    offer_rows = [r for r in offer_rows_all if not is_offer_phd_row(r)]
    cost_rows = [r for r in cost_rows_all if not is_cost_phd_row(r)]

    offer_phd_removed = len(offer_rows_all) - len(offer_rows)
    cost_phd_removed = len(cost_rows_all) - len(cost_rows)

    missing_offer_school, offer_counter = build_school_counter(offer_rows, args.offer_school_col)
    cost_schools = unique_school_list(cost_rows, args.cost_school_col)
    cost_school_set = set(cost_schools)

    print("[basic_check]")
    print(f"offers_file = {args.offers}")
    print(f"costs_file = {costs_path}  # {costs_source}")
    print(f"offer_rows_before_filter = {len(offer_rows_all)}")
    print(f"offer_rows_removed_contains_phd = {offer_phd_removed}")
    print(f"offer_rows_after_filter = {len(offer_rows)}")
    print(f"offer_school_missing_rows = {missing_offer_school}")
    print(f"offer_unique_schools = {len(offer_counter)}")
    print(f"cost_rows_before_filter = {len(cost_rows_all)}")
    print(f"cost_rows_removed_contains_phd = {cost_phd_removed}")
    print(f"cost_rows_after_filter = {len(cost_rows)}")
    print(f"cost_unique_schools = {len(cost_schools)}")
    print(f"gpa_rule = {args.gpa_rule}")

    print()
    print(f"[top_offer_schools] top_n = {args.top_schools}")
    for school, cnt in offer_counter.most_common(args.top_schools):
        print(f"{school} = {cnt}")

    # 1) exact matches
    final_matches: Dict[str, MatchRecord] = {}
    for school, cnt in offer_counter.items():
        if school in cost_school_set:
            final_matches[school] = MatchRecord(school, school, cnt, "exact")

    # Remaining after exact
    remaining = {
        s: c for s, c in offer_counter.items() if s not in final_matches
    }

    # 2) manual alias
    alias_hits = 0
    for school, cnt in list(remaining.items()):
        key = normalize_school_name(school)
        alias_target = MANUAL_ALIASES.get(key)
        if alias_target and alias_target in cost_school_set:
            final_matches[school] = MatchRecord(school, alias_target, cnt, "alias")
            alias_hits += 1
            remaining.pop(school)

    # 3) normalized unique matching
    cost_norm_map: Dict[str, set] = defaultdict(set)
    for c in cost_schools:
        cost_norm_map[normalize_school_name(c)].add(c)

    ambiguous_normalized = []
    normalized_hits = 0
    for school, cnt in list(remaining.items()):
        key = normalize_school_name(school)
        candidates = sorted(cost_norm_map.get(key, set()))
        if len(candidates) == 1:
            final_matches[school] = MatchRecord(
                school, candidates[0], cnt, "normalized_unique"
            )
            normalized_hits += 1
            remaining.pop(school)
        elif len(candidates) > 1:
            ambiguous_normalized.append((school, cnt, candidates))

    # 4) fuzzy candidates for remaining schools
    fuzzy_auto_hits = 0
    fuzzy_review = []
    fuzzy_no_candidate = []
    for school, cnt in list(remaining.items()):
        cand_scores = best_candidates(
            school, cost_schools, args.fuzzy_top_n, args.fuzzy_cutoff
        )
        if not cand_scores:
            fuzzy_no_candidate.append((school, cnt))
            continue

        best_name, best_score = cand_scores[0]
        second_score = cand_scores[1][1] if len(cand_scores) > 1 else 0.0
        gap = best_score - second_score

        if best_score >= args.fuzzy_auto_threshold and gap >= args.fuzzy_min_gap:
            final_matches[school] = MatchRecord(school, best_name, cnt, "fuzzy_auto")
            fuzzy_auto_hits += 1
            remaining.pop(school)
        else:
            fuzzy_review.append((school, cnt, cand_scores))

    # coverage summary
    total_offer_rows_with_school = sum(offer_counter.values())
    matched_rows = sum(m.offer_count for m in final_matches.values())
    matched_unique = len(final_matches)
    coverage_rows = (
        matched_rows / total_offer_rows_with_school * 100
        if total_offer_rows_with_school
        else 0.0
    )
    coverage_unique = (
        matched_unique / len(offer_counter) * 100 if offer_counter else 0.0
    )

    type_counter = Counter(m.match_type for m in final_matches.values())
    type_rows = defaultdict(int)
    for m in final_matches.values():
        type_rows[m.match_type] += m.offer_count

    print()
    print("[match_summary]")
    print(f"matched_offer_rows = {matched_rows}")
    print(f"total_offer_rows = {total_offer_rows_with_school}")
    print(f"matched_offer_rows_pct = {coverage_rows:.2f}")
    print(f"matched_unique_schools = {matched_unique}")
    print(f"total_unique_offer_schools = {len(offer_counter)}")
    print(f"matched_unique_schools_pct = {coverage_unique:.2f}")

    print()
    print("[match_type_breakdown]")
    for t in ["exact", "alias", "normalized_unique", "fuzzy_auto"]:
        print(f"{t}_unique_schools = {type_counter[t]}")
        print(f"{t}_offer_rows = {type_rows[t]}")

    print()
    print(f"[top_matched_schools] top_n = {args.top_schools}")
    matched_sorted = sorted(
        final_matches.values(), key=lambda x: x.offer_count, reverse=True
    )
    for m in matched_sorted[: args.top_schools]:
        print(
            f"{m.offer_school} -> {m.matched_school} | "
            f"type={m.match_type} | count={m.offer_count}"
        )

    ambiguous_normalized.sort(key=lambda x: x[1], reverse=True)
    print()
    print(f"[normalized_ambiguous] n = {len(ambiguous_normalized)}")
    for school, cnt, cands in ambiguous_normalized[: args.top_suspicious]:
        cand_text = "; ".join(cands)
        print(f"{school} | count={cnt} | candidates={cand_text}")

    fuzzy_review.sort(key=lambda x: x[1], reverse=True)
    print()
    print(f"[fuzzy_review] n = {len(fuzzy_review)}")
    for school, cnt, cands in fuzzy_review[: args.top_suspicious]:
        cand_text = "; ".join([f"{n} ({s:.3f})" for n, s in cands])
        print(f"{school} | count={cnt} | candidates={cand_text}")

    fuzzy_no_candidate.sort(key=lambda x: x[1], reverse=True)
    print()
    print(f"[no_candidate] n = {len(fuzzy_no_candidate)}")
    for school, cnt in fuzzy_no_candidate[: args.top_suspicious]:
        print(f"{school} | count={cnt}")

    # Build output CSV (filtered offers only) with match + cost profile columns.
    if args.output_csv is None:
        output_csv = args.offers.with_name(f"{args.offers.stem}_matched_no_phd.csv")
    else:
        output_csv = args.output_csv

    if output_csv.parent and not output_csv.parent.exists():
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    cost_profile = build_school_cost_profile(cost_rows, args.cost_school_col)

    cost_cols = [
        "cost_rows_non_phd",
        "cost_master_rows_non_phd",
        "cost_profile_source",
        "cost_country",
        "cost_city",
        "cost_university",
        "cost_program",
        "cost_level",
        "cost_duration_years",
        "cost_tuition_usd",
        "cost_living_cost_index",
        "cost_rent_usd",
        "cost_visa_fee_usd",
        "cost_insurance_usd",
        "cost_exchange_rate",
        "cost_total_usd",
    ]

    match_cols = ["match_status", "match_type", "matched_school"]
    gpa_cols = [
        "gpa_raw_numeric",
        "gpa_4_standardized",
        "gpa_standardization_status",
        "gpa_standardization_rule",
    ]
    output_cols = list(offer_fieldnames)
    for col in match_cols + cost_cols + gpa_cols:
        if col not in output_cols:
            output_cols.append(col)

    gpa_status_counter: Counter = Counter()

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_cols)
        writer.writeheader()

        for row in offer_rows:
            school = (row.get(args.offer_school_col) or "").strip()
            rec = final_matches.get(school)

            out_row = dict(row)
            if rec is None:
                out_row["match_status"] = "unmatched"
                out_row["match_type"] = ""
                out_row["matched_school"] = ""
                for c in cost_cols:
                    out_row[c] = ""
            else:
                out_row["match_status"] = "matched"
                out_row["match_type"] = rec.match_type
                out_row["matched_school"] = rec.matched_school

                profile = cost_profile.get(rec.matched_school, {})
                for c in cost_cols:
                    out_row[c] = profile.get(c, "")

            gpa_raw, gpa_4, gpa_status = standardize_gpa_to_4(row, args.gpa_rule)
            out_row["gpa_raw_numeric"] = gpa_raw
            out_row["gpa_4_standardized"] = gpa_4
            out_row["gpa_standardization_status"] = gpa_status
            out_row["gpa_standardization_rule"] = args.gpa_rule
            gpa_status_counter[gpa_status] += 1

            writer.writerow({c: out_row.get(c, "") for c in output_cols})

    print()
    print("[output]")
    print(f"output_csv = {output_csv}")
    print(f"output_rows = {len(offer_rows)}")
    print("output_note = phd rows removed before matching and export")
    print(f"gpa_converted_from_100_rows = {gpa_status_counter['converted_from_100']}")
    print(f"gpa_already_4_scale_rows = {gpa_status_counter['already_4_scale']}")
    print(f"gpa_missing_or_invalid_rows = {gpa_status_counter['missing_or_invalid']}")
    print(f"gpa_out_of_range_rows = {gpa_status_counter['out_of_range']}")


if __name__ == "__main__":
    main()
