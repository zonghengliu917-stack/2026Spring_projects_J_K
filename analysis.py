"""
IS597PR Final Project — Analysis Script
Authors: Kiara (hantarita), Jack (zonghengliu917-stack)

Overarching question:
    What drives the cost of studying abroad for Chinese graduate students —
    the destination country, the school's popularity, or the student's own
    academic background?

Research questions implemented here:
    RQ1 — Is Popularity Priced?
        Do universities that attract more Chinese applicants tend to cost more,
        and does this relationship hold uniformly across destination countries?
    RQ2 — Does Undergraduate Tier Predict Destination Cost? (coming soon)
    RQ3 — Does GPA Have Different Purchasing Power Across Markets? (coming soon)

Usage:
    python analysis.py                        # run all RQs
    python analysis.py --rq 1                 # run RQ1 only
    python analysis.py --output results/      # custom output folder

Data sources:
    Dataset A: compass_offers_processed_matched_no_phd.csv
               (scraped from compassedu.hk, preprocessed by preprocessing.py)
    Dataset B: International_Education_Costs.csv
               (international education cost dataset, publicly available)

Citations:
    CompassEdu offer data: https://m.compassedu.hk
    International Education Costs dataset: see International_Education_Costs.csv
"""
from __future__ import annotations

import argparse
import difflib
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.formula.api as smf

_HERE = Path(__file__).parent
DEFAULT_DATA   = _HERE / "compass_offers_processed_matched_no_phd.csv"
DEFAULT_TIER   = _HERE / "university_tier.csv"
DEFAULT_OUTPUT = _HERE / "output"

# Consistent color palette for the five destination markets
COUNTRY_COLORS: Dict[str, str] = {
    "UK":        "#4C72B0",
    "USA":       "#DD8452",
    "Australia": "#55A868",
    "Hong Kong": "#C44E52",
    "Singapore": "#8172B2",
}

MIN_N_FOR_TEST = 10   # minimum group size to run a significance test


# ── Data loading ─────────────────────────────────────────────────────────────

def load_matched_data(path: Path) -> pd.DataFrame:
    """Load the preprocessed matched CSV and coerce numeric columns.

    Args:
        path: Path to compass_offers_processed_matched_no_phd.csv.

    Returns:
        DataFrame with all rows (matched and unmatched).

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df["cost_total_usd"]      = pd.to_numeric(df["cost_total_usd"],      errors="coerce")
    df["gpa_4_standardized"]  = pd.to_numeric(df["gpa_4_standardized"],  errors="coerce")
    df["cost_tuition_usd"]    = pd.to_numeric(df["cost_tuition_usd"],    errors="coerce")
    df["cost_rent_usd"]       = pd.to_numeric(df["cost_rent_usd"],       errors="coerce")
    df["cost_duration_years"] = pd.to_numeric(df["cost_duration_years"], errors="coerce")
    return df


def filter_matched(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where match_status is 'matched'.

    Args:
        df: Full DataFrame from :func:`load_matched_data`.

    Returns:
        Filtered DataFrame with only successfully matched offer rows.
    """
    return df[df["match_status"] == "matched"].copy()


# ── RQ1 helpers ──────────────────────────────────────────────────────────────

def aggregate_by_school(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matched offer rows into one summary row per destination school.

    Args:
        df: Matched-only DataFrame (output of :func:`filter_matched`).

    Returns:
        DataFrame with columns: matched_school, cost_country, offer_count,
        mean_total_cost. Rows with missing cost_total_usd are excluded.
        Sorted by offer_count descending.
    """
    clean = df.dropna(subset=["cost_total_usd"])
    agg = (
        clean
        .groupby("matched_school", as_index=False)
        .agg(
            offer_count    =("matched_school",  "count"),
            mean_total_cost=("cost_total_usd",  "mean"),
            cost_country   =("cost_country",    "first"),
        )
        .sort_values("offer_count", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def compute_spearman(
    x: pd.Series,
    y: pd.Series,
) -> Tuple[float, float]:
    """Compute Spearman rank correlation and two-sided p-value.

    Args:
        x: First variable (e.g. offer_count).
        y: Second variable (e.g. mean_total_cost).

    Returns:
        Tuple of (rho, p_value). Both are NaN if fewer than 3 observations.
    """
    if len(x) < 3:
        return float("nan"), float("nan")
    rho, p = scipy.stats.spearmanr(x, y)
    return float(rho), float(p)


def spearman_summary(
    label: str,
    x: pd.Series,
    y: pd.Series,
    min_n: int = MIN_N_FOR_TEST,
) -> str:
    """Return a formatted one-line Spearman result string.

    Suppresses the significance test and adds a caveat when n < *min_n*.

    Args:
        label: Display label (e.g. country name or 'All schools').
        x: First variable.
        y: Second variable.
        min_n: Minimum n required to report a p-value.

    Returns:
        A single human-readable result line.
    """
    n = len(x)
    rho, p = compute_spearman(x, y)
    if n < min_n:
        return (
            f"  {label:<14} n={n:>3}  rho={rho:+.3f}  "
            f"[descriptive only — n < {min_n}, no significance test]"
        )
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    return f"  {label:<14} n={n:>3}  rho={rho:+.3f}  p={p:.4f}  {sig}"


# ── RQ1 visualisation ─────────────────────────────────────────────────────────

def _label_top_schools(
    ax: plt.Axes,
    agg: pd.DataFrame,
    top_n: int = 12,
) -> None:
    """Annotate the *top_n* schools by offer_count with their short names.

    Args:
        ax: Axes object to draw on.
        agg: Aggregated school DataFrame (must have matched_school, offer_count,
             mean_total_cost columns).
        top_n: Number of highest-traffic schools to label.
    """
    top = agg.nlargest(top_n, "offer_count")
    for _, row in top.iterrows():
        name = row["matched_school"]
        # Shorten long names to keep the plot readable
        short = name.replace("University of ", "U. of ").replace("University", "U.")
        short = textwrap.shorten(short, width=22, placeholder="…")
        ax.annotate(
            short,
            xy=(row["offer_count"], row["mean_total_cost"]),
            xytext=(6, 2),
            textcoords="offset points",
            fontsize=6.5,
            color="#333333",
        )


def plot_rq1_scatter(
    agg: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Save a scatter plot of offer_count vs mean_total_cost, coloured by country.

    Args:
        agg: Aggregated school DataFrame from :func:`aggregate_by_school`.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for country, group in agg.groupby("cost_country"):
        color = COUNTRY_COLORS.get(country, "#888888")
        ax.scatter(
            group["offer_count"],
            group["mean_total_cost"],
            label=country,
            color=color,
            s=60,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    _label_top_schools(ax, agg)

    ax.set_xscale("log")
    ax.set_xlabel("Number of Admissions (log scale)", fontsize=11)
    ax.set_ylabel("Estimated Total Cost (USD)", fontsize=11)
    ax.set_title(
        "RQ1: Is Popularity Priced?\n"
        "Offer volume vs total program cost by destination school",
        fontsize=12,
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v:,.0f}")
    )
    ax.legend(title="Country", fontsize=9, title_fontsize=9)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    out = output_dir / "rq1_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── RQ1 orchestrator ─────────────────────────────────────────────────────────

def run_rq1(
    matched: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Execute all RQ1 hypotheses and save results.

    H1a: Globally, offer_count and mean_total_cost are significantly positively
         correlated (Spearman ρ > 0, p < 0.05).
    H1b: This correlation is driven by the USA market; UK and Australia show
         weaker or insignificant within-country correlations.

    Args:
        matched: Matched-only DataFrame (from :func:`filter_matched`).
        output_dir: Directory for output files (PNG + results text).
    """
    agg = aggregate_by_school(matched)

    lines: List[str] = [
        "=" * 60,
        "RQ1 Results — Is Popularity Priced?",
        "=" * 60,
        "",
        f"Schools analysed: {len(agg)}",
        f"Total offer rows: {agg['offer_count'].sum():,}",
        "",
        "── H1a: Global Spearman correlation ──",
    ]

    # H1a — all 71 schools
    global_line = spearman_summary(
        "All schools", agg["offer_count"], agg["mean_total_cost"]
    )
    lines.append(global_line)
    lines.append("")
    lines.append("── H1b: Per-country Spearman correlations ──")

    # H1b — per country
    for country in ["UK", "USA", "Australia", "Hong Kong", "Singapore"]:
        sub = agg[agg["cost_country"] == country]
        lines.append(
            spearman_summary(country, sub["offer_count"], sub["mean_total_cost"])
        )

    lines += [
        "",
        "── Top 10 schools by offer volume ──",
        f"  {'School':<45} {'Country':<12} {'Offers':>7} {'Cost (USD)':>12}",
        "  " + "-" * 78,
    ]
    for _, row in agg.head(10).iterrows():
        lines.append(
            f"  {row['matched_school']:<45} {row['cost_country']:<12} "
            f"{row['offer_count']:>7,} ${row['mean_total_cost']:>11,.0f}"
        )

    # Save text results
    result_text = "\n".join(lines)
    result_path = output_dir / "rq1_results.txt"
    result_path.write_text(result_text, encoding="utf-8")
    print(result_text)

    # Save plot
    plot_path = plot_rq1_scatter(agg, output_dir)
    print(f"\n[RQ1] Plot saved → {plot_path}")
    print(f"[RQ1] Results saved → {result_path}")


# ── RQ2 helpers ──────────────────────────────────────────────────────────────

TIER_ORDER = ["985", "211", "Other"]


def load_tier_lookup(tier_csv: Path) -> Dict[str, str]:
    """Load university_tier.csv into a dict mapping school name to tier.

    Args:
        tier_csv: Path to a CSV with columns ``school_name`` and ``tier``.

    Returns:
        Dict of {school_name: '985' | '211' | 'Other'}.

    Raises:
        FileNotFoundError: If *tier_csv* does not exist.
    """
    if not tier_csv.exists():
        raise FileNotFoundError(f"Tier lookup file not found: {tier_csv}")
    lookup: Dict[str, str] = {}
    with tier_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lookup[row["school_name"].strip()] = row["tier"].strip()
    return lookup


def assign_tier(
    school: str,
    tier_lookup: Dict[str, str],
    known_schools: List[str],
    cutoff: float = 0.85,
) -> str:
    """Return the tier for one Chinese undergraduate school name.

    Tries an exact lookup first, then falls back to fuzzy matching via
    :func:`difflib.get_close_matches`.  Non-Chinese or unrecognised schools
    default to ``'Other'``.

    Args:
        school: Raw value from the ``毕业学校`` column.
        tier_lookup: Dict from :func:`load_tier_lookup`.
        known_schools: Pre-computed list of keys in *tier_lookup*.
        cutoff: Minimum similarity ratio for fuzzy acceptance (0–1).

    Returns:
        One of ``'985'``, ``'211'``, or ``'Other'``.
    """
    if not school or not school.strip():
        return "Other"
    school = school.strip()
    if school in tier_lookup:
        return tier_lookup[school]
    matches = difflib.get_close_matches(school, known_schools, n=1, cutoff=cutoff)
    if matches:
        return tier_lookup[matches[0]]
    return "Other"


def add_tier_column(
    df: pd.DataFrame,
    tier_lookup: Dict[str, str],
) -> pd.DataFrame:
    """Add a ``tier`` column to *df* based on the ``毕业学校`` column.

    Args:
        df: Matched offer DataFrame.
        tier_lookup: Dict from :func:`load_tier_lookup`.

    Returns:
        Copy of *df* with a new ``tier`` column (985 / 211 / Other).
    """
    known_schools = list(tier_lookup.keys())
    df = df.copy()
    df["tier"] = df["毕业学校"].apply(
        lambda s: assign_tier(
            str(s) if pd.notna(s) else "", tier_lookup, known_schools
        )
    )
    df["tier"] = pd.Categorical(df["tier"], categories=TIER_ORDER, ordered=True)
    return df


# ── RQ2 statistics ────────────────────────────────────────────────────────────

def run_rq2_kruskal(df: pd.DataFrame) -> Dict[str, object]:
    """Test H2a: does total cost differ significantly across undergraduate tiers?

    Runs a Kruskal-Wallis test across 985 / 211 / Other groups, then pairwise
    Mann-Whitney U post-hoc tests if the omnibus result is significant.

    Args:
        df: DataFrame with ``tier`` and ``cost_total_usd`` columns.

    Returns:
        Dict with keys ``stat``, ``p``, ``group_medians``, and ``pairwise``.
    """
    groups = {
        tier: df.loc[df["tier"] == tier, "cost_total_usd"].dropna().values
        for tier in TIER_ORDER
    }
    stat, p = scipy.stats.kruskal(*groups.values())

    pairwise: Dict[str, Tuple[float, float]] = {}
    if p < 0.05:
        pairs = [("985", "211"), ("985", "Other"), ("211", "Other")]
        for a, b in pairs:
            u, pu = scipy.stats.mannwhitneyu(
                groups[a], groups[b], alternative="two-sided"
            )
            pairwise[f"{a} vs {b}"] = (float(u), float(pu))

    return {
        "stat": float(stat),
        "p": float(p),
        "group_medians": {t: float(np.median(v)) for t, v in groups.items()},
        "pairwise": pairwise,
    }


def run_rq2_chisquare(df: pd.DataFrame) -> Dict[str, object]:
    """Test H2b: does undergraduate tier predict destination country?

    Builds a tier × country contingency table and runs a chi-square test.
    Only the five main destination countries are included to avoid sparse cells.

    Args:
        df: DataFrame with ``tier`` and ``cost_country`` columns.

    Returns:
        Dict with keys ``chi2``, ``p``, ``dof``, and ``contingency`` (DataFrame).
    """
    main_countries = list(COUNTRY_COLORS.keys())
    sub = df[df["cost_country"].isin(main_countries)]
    contingency = pd.crosstab(sub["tier"], sub["cost_country"])
    # Reorder columns for consistency
    contingency = contingency.reindex(
        columns=[c for c in main_countries if c in contingency.columns]
    )
    chi2, p, dof, _ = scipy.stats.chi2_contingency(contingency)
    return {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "contingency": contingency,
    }


def run_rq2_ols(df: pd.DataFrame) -> Dict[str, object]:
    """Test H2c: does the tier effect on cost shrink after controlling for country?

    Fits two OLS models:
      Model 1 — cost ~ tier dummies (no country control)
      Model 2 — cost ~ tier dummies + country dummies

    Because cost_total_usd is a school-level constant (every applicant to the
    same school shares the same value), cluster-robust standard errors are used
    to avoid pseudo-replication inflating significance.

    Args:
        df: DataFrame with ``tier``, ``cost_country``, and ``cost_total_usd``.

    Returns:
        Dict with keys ``model1``, ``model2``, ``coef_drop_pct`` (percentage
        reduction in the 985 coefficient after adding country dummies), and
        ``matched_school`` group labels used for clustering.
    """
    data = df.dropna(subset=["cost_total_usd"]).copy()
    data["is_985"] = (data["tier"] == "985").astype(int)
    data["is_211"] = (data["tier"] == "211").astype(int)

    m1 = smf.ols("cost_total_usd ~ is_985 + is_211", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["matched_school"]}
    )
    m2 = smf.ols(
        "cost_total_usd ~ is_985 + is_211 + C(cost_country)", data=data
    ).fit(cov_type="cluster", cov_kwds={"groups": data["matched_school"]})

    coef1 = m1.params.get("is_985", float("nan"))
    coef2 = m2.params.get("is_985", float("nan"))
    drop_pct = (coef1 - coef2) / abs(coef1) * 100 if coef1 != 0 else float("nan")

    return {"model1": m1, "model2": m2, "coef_drop_pct": float(drop_pct)}


# ── RQ2 visualisation ─────────────────────────────────────────────────────────

def plot_rq2_cost_by_tier(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save a box plot of total cost by undergraduate tier.

    Args:
        df: DataFrame with ``tier`` and ``cost_total_usd`` columns.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    data_by_tier = [
        df.loc[df["tier"] == t, "cost_total_usd"].dropna().values
        for t in TIER_ORDER
    ]
    bp = ax.boxplot(
        data_by_tier,
        tick_labels=TIER_ORDER,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    colors = ["#E07B54", "#5B9BD5", "#A8C888"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Annotate median values
    for i, tier in enumerate(TIER_ORDER):
        med = np.median(data_by_tier[i])
        ax.text(
            i + 1, med, f"  ${med:,.0f}",
            va="center", fontsize=8, color="#333333"
        )

    ax.set_xlabel("Undergraduate Tier", fontsize=11)
    ax.set_ylabel("Estimated Total Cost (USD)", fontsize=11)
    ax.set_title(
        "RQ2 — H2a: Total Program Cost by Undergraduate Tier",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    out = output_dir / "rq2_cost_by_tier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_rq2_country_by_tier(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save a stacked bar chart of destination country distribution by tier.

    Args:
        df: DataFrame with ``tier`` and ``cost_country`` columns.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    main_countries = list(COUNTRY_COLORS.keys())
    sub = df[df["cost_country"].isin(main_countries)]
    counts = pd.crosstab(sub["tier"], sub["cost_country"])
    counts = counts.reindex(
        index=TIER_ORDER,
        columns=[c for c in main_countries if c in counts.columns],
        fill_value=0,
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(TIER_ORDER))
    for country in pct.columns:
        color = COUNTRY_COLORS.get(country, "#888888")
        vals = pct[country].values
        bars = ax.bar(TIER_ORDER, vals, bottom=bottom, label=country, color=color, alpha=0.85)
        # Label segments that are wide enough to read
        for bar, val, bot in zip(bars, vals, bottom):
            if val >= 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bot + val / 2,
                    f"{val:.0f}%",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold",
                )
        bottom += vals

    ax.set_xlabel("Undergraduate Tier", fontsize=11)
    ax.set_ylabel("Share of Applicants (%)", fontsize=11)
    ax.set_title(
        "RQ2 — H2b: Destination Country Distribution by Undergraduate Tier",
        fontsize=12, fontweight="bold",
    )
    ax.legend(title="Country", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    out = output_dir / "rq2_country_by_tier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── RQ2 orchestrator ──────────────────────────────────────────────────────────

def run_rq2(
    matched: pd.DataFrame,
    tier_csv: Path,
    output_dir: Path,
) -> None:
    """Execute all RQ2 hypotheses and save results.

    H2a: 985 students pay significantly more than 211 and Other (Kruskal-Wallis).
    H2b: Undergraduate tier significantly predicts destination country (chi-square).
    H2c: After controlling for country, the tier cost effect shrinks substantially
         (OLS with cluster-robust standard errors).

    Args:
        matched: Matched-only DataFrame (from :func:`filter_matched`).
        tier_csv: Path to university_tier.csv.
        output_dir: Directory for output files.
    """
    tier_lookup = load_tier_lookup(tier_csv)
    df = add_tier_column(matched, tier_lookup)

    tier_counts = df["tier"].value_counts().to_dict()

    lines: List[str] = [
        "=" * 60,
        "RQ2 Results — Does Undergraduate Tier Predict Destination Cost?",
        "=" * 60,
        "",
        "── Tier distribution ──",
        *(f"  {t}: {tier_counts.get(t, 0):,} applicants" for t in TIER_ORDER),
        "",
    ]

    # H2a — Kruskal-Wallis
    kw = run_rq2_kruskal(df)
    sig = "***" if kw["p"] < 0.001 else ("**" if kw["p"] < 0.01 else ("*" if kw["p"] < 0.05 else "n.s."))
    lines += [
        "── H2a: Kruskal-Wallis — cost across tiers ──",
        f"  H={kw['stat']:.2f},  p={kw['p']:.4f}  {sig}",
        "  Median total cost:",
        *(f"    {t}: ${kw['group_medians'][t]:,.0f}" for t in TIER_ORDER),
    ]
    if kw["pairwise"]:
        lines.append("  Post-hoc Mann-Whitney U (two-sided):")
        for pair, (u, pu) in kw["pairwise"].items():
            ps = "***" if pu < 0.001 else ("**" if pu < 0.01 else ("*" if pu < 0.05 else "n.s."))
            lines.append(f"    {pair}: U={u:.0f}, p={pu:.4f}  {ps}")
    lines.append("")

    # H2b — Chi-square
    chi = run_rq2_chisquare(df)
    cs = "***" if chi["p"] < 0.001 else ("**" if chi["p"] < 0.01 else ("*" if chi["p"] < 0.05 else "n.s."))
    lines += [
        "── H2b: Chi-square — tier vs destination country ──",
        f"  χ²={chi['chi2']:.2f},  df={chi['dof']},  p={chi['p']:.4f}  {cs}",
        "  Contingency table (row = tier, col = country):",
    ]
    for row_label, row_data in chi["contingency"].iterrows():
        lines.append("    " + f"{row_label:<8}" + "  ".join(f"{c:>12}: {v:>5}" for c, v in row_data.items()))
    lines.append("")

    # H2c — OLS
    ols = run_rq2_ols(df)
    m1, m2 = ols["model1"], ols["model2"]
    lines += [
        "── H2c: OLS — does tier effect shrink after controlling for country? ──",
        "  (Cluster-robust SEs by destination school)",
        f"  Model 1 (no country): is_985 coef={m1.params['is_985']:+,.0f}, "
        f"p={m1.pvalues['is_985']:.4f},  R²={m1.rsquared:.3f}",
        f"  Model 2 (+ country):  is_985 coef={m2.params['is_985']:+,.0f}, "
        f"p={m2.pvalues['is_985']:.4f},  R²={m2.rsquared:.3f}",
        f"  Coefficient drop after adding country: {ols['coef_drop_pct']:.1f}%",
        "",
    ]

    # Supplementary note: USA selection rate by tier
    # USA is the most expensive market; if 985 students disproportionately
    # choose USA it would further explain their higher costs.
    usa_total = df[df["cost_country"] == "USA"]
    tier_totals = df["tier"].value_counts()
    usa_by_tier = usa_total["tier"].value_counts()
    usa_note_lines = [
        "── Supplementary: USA selection rate by tier ──",
        f"  (USA total applicants: {len(usa_total)} — small sample, descriptive only)",
    ]
    for t in TIER_ORDER:
        n = usa_by_tier.get(t, 0)
        total = tier_totals.get(t, 1)
        usa_note_lines.append(f"  {t:<8}: {n:>4} / {total:>6} = {n/total*100:.2f}%")
    usa_note_lines += [
        "  Note: 'Other' has the highest USA rate (3.56%), not 985 (2.47%).",
        "  985 students are concentrated in Singapore (NTU/NUS), which is",
        "  expensive but still below top US programs — consistent with a",
        "  'high-budget value-optimisation' pattern rather than pure prestige-seeking.",
        "",
    ]
    lines += [""] + usa_note_lines

    result_text = "\n".join(lines)
    result_path = output_dir / "rq2_results.txt"
    result_path.write_text(result_text, encoding="utf-8")
    print(result_text)

    p1 = plot_rq2_cost_by_tier(df, output_dir)
    p2 = plot_rq2_country_by_tier(df, output_dir)
    print(f"[RQ2] Plots saved → {p1.name}, {p2.name}")
    print(f"[RQ2] Results saved → {result_path}")


# ── RQ3 helpers ──────────────────────────────────────────────────────────────

VALID_GPA_STATUSES = {"converted_from_100", "already_4_scale"}


def filter_rq3_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with a valid standardised GPA and a matched cost.

    Excludes rows where GPA conversion failed, GPA is zero (score below 60,
    indicating likely data quality issues), or cost is missing.

    Args:
        df: Matched-only DataFrame from :func:`filter_matched`.

    Returns:
        Filtered DataFrame ready for regression analysis.
    """
    mask = (
        df["gpa_standardization_status"].isin(VALID_GPA_STATUSES)
        & (df["gpa_4_standardized"] > 0.0)
        & df["cost_total_usd"].notna()
    )
    return df[mask].copy()


# ── RQ3 statistics ────────────────────────────────────────────────────────────

def run_rq3_ols(df: pd.DataFrame) -> Tuple[object, object]:
    """Fit two OLS models to test whether GPA predicts total cost.

    Model 1 (H3a): cost ~ GPA  — no country control
    Model 2 (H3b): cost ~ GPA + country dummies  — controls for destination

    Both models use cluster-robust standard errors grouped by destination school,
    because cost_total_usd is a school-level constant (not individual variation).

    Args:
        df: Filtered DataFrame from :func:`filter_rq3_data`.

    Returns:
        Tuple of (model1, model2) fitted statsmodels OLS results.
    """
    cluster = {"cov_type": "cluster", "cov_kwds": {"groups": df["matched_school"]}}
    m1 = smf.ols("cost_total_usd ~ gpa_4_standardized", data=df).fit(**cluster)
    m2 = smf.ols(
        "cost_total_usd ~ gpa_4_standardized + C(cost_country)", data=df
    ).fit(**cluster)
    return m1, m2


def compute_within_country_correlations(
    df: pd.DataFrame,
    min_n: int = MIN_N_FOR_TEST,
) -> pd.DataFrame:
    """Compute per-country Spearman correlation between GPA and total cost.

    Args:
        df: Filtered DataFrame from :func:`filter_rq3_data`.
        min_n: Countries with fewer rows than this are skipped.

    Returns:
        DataFrame with columns: country, n, rho, p_value, significant.
        Sorted by rho descending.
    """
    rows = []
    for country, group in df.groupby("cost_country"):
        if len(group) < min_n:
            continue
        rho, p = compute_spearman(group["gpa_4_standardized"], group["cost_total_usd"])
        rows.append({
            "country": country,
            "n": len(group),
            "rho": rho,
            "p_value": p,
            "significant": p < 0.05,
        })
    return pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)


# ── RQ3 visualisation ─────────────────────────────────────────────────────────

def plot_rq3_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save a scatter plot of GPA vs total cost coloured by country.

    Each country gets its own regression line. Only the five main destination
    countries are shown to keep the plot readable.

    Args:
        df: Filtered DataFrame from :func:`filter_rq3_data`.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for country, group in df.groupby("cost_country"):
        if country not in COUNTRY_COLORS:
            continue
        color = COUNTRY_COLORS[country]
        ax.scatter(
            group["gpa_4_standardized"], group["cost_total_usd"],
            color=color, s=8, alpha=0.25, zorder=2,
        )
        # Regression line per country
        x = group["gpa_4_standardized"].values
        y = group["cost_total_usd"].values
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, color=color, linewidth=2,
                label=f"{country} (slope={m:+,.0f})")

    ax.set_xlabel("Standardised GPA (4.0 scale)", fontsize=11)
    ax.set_ylabel("Estimated Total Cost (USD)", fontsize=11)
    ax.set_title(
        "RQ3 — GPA vs Total Program Cost by Destination Country",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=9, title="Country (OLS slope)")
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    out = output_dir / "rq3_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_rq3_coef(m1: object, m2: object, output_dir: Path) -> Path:
    """Save a bar chart comparing the GPA coefficient before and after country control.

    Args:
        m1: Simple OLS model (no country dummies).
        m2: Country-controlled OLS model.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    coefs = [m1.params["gpa_4_standardized"], m2.params["gpa_4_standardized"]]
    cis   = [
        m1.conf_int().loc["gpa_4_standardized"].values,
        m2.conf_int().loc["gpa_4_standardized"].values,
    ]
    labels = ["Model 1\n(GPA only)", "Model 2\n(GPA + Country)"]
    colors = ["#DD8452", "#4C72B0"]

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (coef, ci, label, color) in enumerate(zip(coefs, cis, labels, colors)):
        ax.bar(i, coef, color=color, alpha=0.8, width=0.5)
        ax.errorbar(i, coef, yerr=[[coef - ci[0]], [ci[1] - coef]],
                    fmt="none", color="black", capsize=6, linewidth=1.5)
        ax.text(i, coef + (ci[1] - coef) + 500, f"${coef:,.0f}",
                ha="center", fontsize=10, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("GPA Coefficient (USD per 1-point GPA increase)", fontsize=10)
    ax.set_title(
        "RQ3 — H3b: Does the GPA Effect Shrink\nAfter Controlling for Country?",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    out = output_dir / "rq3_coef_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── RQ3 orchestrator ──────────────────────────────────────────────────────────

def run_rq3(matched: pd.DataFrame, output_dir: Path) -> None:
    """Execute all RQ3 hypotheses and save results.

    H3a: GPA and total cost are significantly positively correlated (simple OLS).
    H3b: After adding country dummies, the GPA coefficient drops > 50%,
         showing that country choice mediates most of the GPA-cost relationship.
    H3c: Within-country GPA-cost correlations vary: positive in some markets,
         near-zero in others.

    Args:
        matched: Matched-only DataFrame from :func:`filter_matched`.
        output_dir: Directory for output files.
    """
    df = filter_rq3_data(matched)
    m1, m2 = run_rq3_ols(df)
    within = compute_within_country_correlations(df)

    coef1 = m1.params["gpa_4_standardized"]
    coef2 = m2.params["gpa_4_standardized"]
    drop_pct = (coef1 - coef2) / abs(coef1) * 100

    def sig(p: float) -> str:
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))

    lines: List[str] = [
        "=" * 60,
        "RQ3 Results — Does GPA Have Different Purchasing Power Across Markets?",
        "=" * 60,
        "",
        f"Rows after GPA/cost filter: {len(df):,}",
        f"GPA range: {df['gpa_4_standardized'].min():.2f} – {df['gpa_4_standardized'].max():.2f}",
        "",
        "── H3a: Simple OLS — does GPA predict cost? ──",
        f"  cost ~ GPA:  coef={coef1:+,.0f} USD/GPA-point",
        f"  p={m1.pvalues['gpa_4_standardized']:.4f}  {sig(m1.pvalues['gpa_4_standardized'])}",
        f"  R²={m1.rsquared:.4f}",
        "",
        "── H3b: Country-controlled OLS — does GPA effect shrink? ──",
        "  (Cluster-robust SEs by destination school)",
        f"  cost ~ GPA + country:  coef={coef2:+,.0f} USD/GPA-point",
        f"  p={m2.pvalues['gpa_4_standardized']:.4f}  {sig(m2.pvalues['gpa_4_standardized'])}",
        f"  R²={m2.rsquared:.4f}",
        f"  GPA coefficient drop after adding country: {drop_pct:.1f}%",
        "",
        "── H3c: Within-country Spearman (GPA vs cost) ──",
        f"  {'Country':<14} {'n':>6}  {'rho':>7}  {'p':>8}",
        "  " + "-" * 42,
    ]
    for _, row in within.iterrows():
        lines.append(
            f"  {row['country']:<14} {int(row['n']):>6}  "
            f"{row['rho']:>+.3f}  {row['p_value']:>8.4f}  {sig(row['p_value'])}"
        )

    result_text = "\n".join(lines)
    result_path = output_dir / "rq3_results.txt"
    result_path.write_text(result_text, encoding="utf-8")
    print(result_text)

    p1 = plot_rq3_scatter(df, output_dir)
    p2 = plot_rq3_coef(m1, m2, output_dir)
    print(f"\n[RQ3] Plots saved → {p1.name}, {p2.name}")
    print(f"[RQ3] Results saved → {result_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the requested research questions.

    Example:
        python analysis.py
        python analysis.py --rq 1 --output my_results/
    """
    parser = argparse.ArgumentParser(
        description="IS597PR Final Project Analysis — RQ1/RQ2/RQ3"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to the matched offers CSV (default: auto-detect from script folder).",
    )
    parser.add_argument(
        "--tier",
        type=Path,
        default=DEFAULT_TIER,
        help="Path to university_tier.csv for RQ2 (default: auto-detect).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for plots and result text files.",
    )
    parser.add_argument(
        "--rq",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which research question(s) to run (default: all).",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df      = load_matched_data(args.data)
    matched = filter_matched(df)

    if args.rq in ("1", "all"):
        run_rq1(matched, args.output)

    if args.rq in ("2", "all"):
        run_rq2(matched, args.tier, args.output)

    if args.rq in ("3", "all"):
        run_rq3(matched, args.output)


if __name__ == "__main__":
    main()
