"""
Tests for analysis.py helper functions.

Covers: filter_matched, aggregate_by_school, compute_spearman,
spearman_summary, load_tier_lookup, assign_tier, add_tier_column,
run_rq2_kruskal, run_rq2_chisquare, filter_rq3_data,
compute_within_country_correlations, load_matched_data.

Run with:
    pytest tests/test_analysis.py -v
    pytest tests/ --cov=analysis --cov-report=term-missing
"""
import math
from pathlib import Path

import pandas as pd
import pytest

from analysis import (
    MIN_N_FOR_TEST,
    TIER_ORDER,
    VALID_GPA_STATUSES,
    add_tier_column,
    aggregate_by_school,
    assign_tier,
    compute_spearman,
    compute_within_country_correlations,
    filter_matched,
    filter_rq3_data,
    load_matched_data,
    load_tier_lookup,
    run_rq2_chisquare,
    run_rq2_kruskal,
    spearman_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_tier_csv(tmp_path: Path) -> Path:
    """A small university_tier.csv for tests that need file I/O."""
    p = tmp_path / "university_tier.csv"
    p.write_text(
        "school_name,tier\n"
        "北京大学,985\n"
        "清华大学,985\n"
        "对外经济贸易大学,211\n"
        "河海大学,211\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def sample_matched_df() -> pd.DataFrame:
    """A small matched-only DataFrame for RQ1/RQ2 tests."""
    return pd.DataFrame(
        {
            "match_status":    ["matched", "matched", "matched", "matched"],
            "matched_school":  ["Oxford", "Oxford", "MIT", "NUS"],
            "cost_country":    ["UK",     "UK",     "USA", "Singapore"],
            "cost_total_usd":  [60_000.0, 60_000.0, 200_000.0, 100_000.0],
            "毕业学校":         ["北京大学", "清华大学", "对外经济贸易大学", "河海大学"],
            "gpa_4_standardized":        [3.8, 3.5, 3.2, 3.0],
            "gpa_standardization_status":["already_4_scale"] * 4,
        }
    )


# ── filter_matched ────────────────────────────────────────────────────────────

class TestFilterMatched:
    def test_keeps_only_matched_rows(self) -> None:
        df = pd.DataFrame(
            {"match_status": ["matched", "unmatched", "matched"], "v": [1, 2, 3]}
        )
        result = filter_matched(df)
        assert len(result) == 2
        assert (result["match_status"] == "matched").all()

    def test_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"match_status": ["matched"], "v": [1]})
        result = filter_matched(df)
        result["v"] = 99
        assert df["v"].iloc[0] == 1  # original unchanged

    def test_empty_input(self) -> None:
        df = pd.DataFrame({"match_status": pd.Series([], dtype=str)})
        assert len(filter_matched(df)) == 0

    def test_all_unmatched_returns_empty(self) -> None:
        df = pd.DataFrame({"match_status": ["unmatched", "unmatched"]})
        assert len(filter_matched(df)) == 0


# ── aggregate_by_school ───────────────────────────────────────────────────────

class TestAggregateBySchool:
    def test_offer_count_is_row_count_per_school(
        self, sample_matched_df: pd.DataFrame
    ) -> None:
        agg = aggregate_by_school(sample_matched_df)
        oxford_count = agg.loc[agg["matched_school"] == "Oxford", "offer_count"].iloc[0]
        assert oxford_count == 2

    def test_mean_total_cost_is_correct(
        self, sample_matched_df: pd.DataFrame
    ) -> None:
        agg = aggregate_by_school(sample_matched_df)
        mit_cost = agg.loc[agg["matched_school"] == "MIT", "mean_total_cost"].iloc[0]
        assert mit_cost == pytest.approx(200_000.0)

    def test_country_passthrough(self, sample_matched_df: pd.DataFrame) -> None:
        agg = aggregate_by_school(sample_matched_df)
        nus_country = agg.loc[agg["matched_school"] == "NUS", "cost_country"].iloc[0]
        assert nus_country == "Singapore"

    def test_rows_with_missing_cost_excluded(self) -> None:
        df = pd.DataFrame(
            {
                "match_status":   ["matched", "matched"],
                "matched_school": ["A", "B"],
                "cost_country":   ["UK", "USA"],
                "cost_total_usd": [50_000.0, float("nan")],
            }
        )
        agg = aggregate_by_school(df)
        assert len(agg) == 1
        assert agg["matched_school"].iloc[0] == "A"

    def test_sorted_by_offer_count_descending(
        self, sample_matched_df: pd.DataFrame
    ) -> None:
        agg = aggregate_by_school(sample_matched_df)
        counts = agg["offer_count"].tolist()
        assert counts == sorted(counts, reverse=True)


# ── compute_spearman ──────────────────────────────────────────────────────────

class TestComputeSpearman:
    def test_perfect_positive_correlation(self) -> None:
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([1, 2, 3, 4, 5])
        rho, p = compute_spearman(x, y)
        assert rho == pytest.approx(1.0)
        assert p < 0.05

    def test_perfect_negative_correlation(self) -> None:
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([5, 4, 3, 2, 1])
        rho, p = compute_spearman(x, y)
        assert rho == pytest.approx(-1.0)
        assert p < 0.05

    def test_returns_float_tuple(self) -> None:
        x = pd.Series([1, 2, 3])
        y = pd.Series([3, 1, 2])
        rho, p = compute_spearman(x, y)
        assert isinstance(rho, float)
        assert isinstance(p, float)

    def test_fewer_than_three_observations_returns_nan(self) -> None:
        x = pd.Series([1, 2])
        y = pd.Series([2, 1])
        rho, p = compute_spearman(x, y)
        assert math.isnan(rho)
        assert math.isnan(p)

    def test_rho_bounded_between_minus_one_and_one(self) -> None:
        x = pd.Series([1, 3, 2, 5, 4, 7, 6])
        y = pd.Series([2, 1, 4, 3, 6, 5, 7])
        rho, _ = compute_spearman(x, y)
        assert -1.0 <= rho <= 1.0


# ── spearman_summary ──────────────────────────────────────────────────────────

class TestSpearmanSummary:
    def test_large_n_shows_p_value(self) -> None:
        x = pd.Series(range(20))
        y = pd.Series(range(20))
        line = spearman_summary("Test", x, y, min_n=10)
        assert "p=" in line
        assert "***" in line

    def test_small_n_shows_caveat(self) -> None:
        x = pd.Series([1, 2, 3])
        y = pd.Series([3, 2, 1])
        line = spearman_summary("SmallGroup", x, y, min_n=10)
        assert "descriptive only" in line
        assert "p=" not in line

    def test_label_appears_in_output(self) -> None:
        x = pd.Series(range(15))
        y = pd.Series(range(15))
        line = spearman_summary("Australia", x, y)
        assert "Australia" in line


# ── load_tier_lookup ──────────────────────────────────────────────────────────

class TestLoadTierLookup:
    def test_returns_correct_mapping(self, minimal_tier_csv: Path) -> None:
        lookup = load_tier_lookup(minimal_tier_csv)
        assert lookup["北京大学"] == "985"
        assert lookup["对外经济贸易大学"] == "211"

    def test_all_entries_loaded(self, minimal_tier_csv: Path) -> None:
        lookup = load_tier_lookup(minimal_tier_csv)
        assert len(lookup) == 4

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_tier_lookup(tmp_path / "nonexistent.csv")


# ── assign_tier ───────────────────────────────────────────────────────────────

class TestAssignTier:
    @pytest.fixture
    def lookup_and_known(self) -> tuple:
        lookup = {
            "北京大学": "985",
            "清华大学": "985",
            "对外经济贸易大学": "211",
            "河海大学": "211",
        }
        return lookup, list(lookup.keys())

    def test_exact_985_match(self, lookup_and_known: tuple) -> None:
        lookup, known = lookup_and_known
        assert assign_tier("北京大学", lookup, known) == "985"

    def test_exact_211_match(self, lookup_and_known: tuple) -> None:
        lookup, known = lookup_and_known
        assert assign_tier("河海大学", lookup, known) == "211"

    def test_unknown_school_returns_other(self, lookup_and_known: tuple) -> None:
        lookup, known = lookup_and_known
        assert assign_tier("深圳大学", lookup, known) == "Other"

    def test_empty_string_returns_other(self, lookup_and_known: tuple) -> None:
        lookup, known = lookup_and_known
        assert assign_tier("", lookup, known) == "Other"

    def test_whitespace_only_returns_other(self, lookup_and_known: tuple) -> None:
        lookup, known = lookup_and_known
        assert assign_tier("   ", lookup, known) == "Other"

    def test_cutoff_zero_matches_loosely(self, lookup_and_known: tuple) -> None:
        # With cutoff=0, even weak matches are accepted
        lookup, known = lookup_and_known
        result = assign_tier("北京大学", lookup, known, cutoff=0.0)
        assert result in ("985", "211", "Other")


# ── add_tier_column ───────────────────────────────────────────────────────────

class TestAddTierColumn:
    def test_tier_column_added(self, minimal_tier_csv: Path) -> None:
        lookup = load_tier_lookup(minimal_tier_csv)
        df = pd.DataFrame({"毕业学校": ["北京大学", "河海大学", "深圳大学"]})
        result = add_tier_column(df, lookup)
        assert "tier" in result.columns

    def test_known_985_school_assigned_correctly(
        self, minimal_tier_csv: Path
    ) -> None:
        lookup = load_tier_lookup(minimal_tier_csv)
        df = pd.DataFrame({"毕业学校": ["北京大学"]})
        result = add_tier_column(df, lookup)
        assert str(result["tier"].iloc[0]) == "985"

    def test_unknown_school_gets_other(self, minimal_tier_csv: Path) -> None:
        lookup = load_tier_lookup(minimal_tier_csv)
        df = pd.DataFrame({"毕业学校": ["一个不存在的学校XYZ"]})
        result = add_tier_column(df, lookup)
        assert str(result["tier"].iloc[0]) == "Other"

    def test_original_dataframe_not_mutated(self, minimal_tier_csv: Path) -> None:
        lookup = load_tier_lookup(minimal_tier_csv)
        df = pd.DataFrame({"毕业学校": ["北京大学"]})
        add_tier_column(df, lookup)
        assert "tier" not in df.columns


# ── run_rq2_kruskal ───────────────────────────────────────────────────────────

class TestRunRq2Kruskal:
    @pytest.fixture
    def tier_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "tier": pd.Categorical(
                    ["985"] * 50 + ["211"] * 50 + ["Other"] * 50,
                    categories=TIER_ORDER,
                    ordered=True,
                ),
                "cost_total_usd": (
                    [150_000.0] * 50 + [80_000.0] * 50 + [50_000.0] * 50
                ),
            }
        )

    def test_returns_stat_and_p(self, tier_df: pd.DataFrame) -> None:
        result = run_rq2_kruskal(tier_df)
        assert "stat" in result
        assert "p" in result

    def test_significant_difference_detected(self, tier_df: pd.DataFrame) -> None:
        result = run_rq2_kruskal(tier_df)
        assert result["p"] < 0.05

    def test_group_medians_returned(self, tier_df: pd.DataFrame) -> None:
        result = run_rq2_kruskal(tier_df)
        assert set(result["group_medians"].keys()) == {"985", "211", "Other"}
        assert result["group_medians"]["985"] == pytest.approx(150_000.0)

    def test_pairwise_tests_run_when_significant(
        self, tier_df: pd.DataFrame
    ) -> None:
        result = run_rq2_kruskal(tier_df)
        assert len(result["pairwise"]) > 0


# ── run_rq2_chisquare ─────────────────────────────────────────────────────────

class TestRunRq2Chisquare:
    @pytest.fixture
    def country_tier_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "tier": pd.Categorical(
                    ["985"] * 40 + ["211"] * 30 + ["Other"] * 30,
                    categories=TIER_ORDER,
                    ordered=True,
                ),
                "cost_country": (
                    ["Singapore"] * 30 + ["UK"] * 10
                    + ["UK"] * 20 + ["Australia"] * 10
                    + ["UK"] * 25 + ["Hong Kong"] * 5
                ),
            }
        )

    def test_returns_chi2_p_dof(self, country_tier_df: pd.DataFrame) -> None:
        result = run_rq2_chisquare(country_tier_df)
        assert "chi2" in result
        assert "p" in result
        assert "dof" in result

    def test_contingency_table_returned(
        self, country_tier_df: pd.DataFrame
    ) -> None:
        result = run_rq2_chisquare(country_tier_df)
        assert isinstance(result["contingency"], pd.DataFrame)


# ── filter_rq3_data ───────────────────────────────────────────────────────────

class TestFilterRq3Data:
    def _base_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gpa_standardization_status": ["already_4_scale"] * 4,
                "gpa_4_standardized": [3.5, 3.2, 0.0, 2.8],
                "cost_total_usd": [80_000.0, 90_000.0, 70_000.0, float("nan")],
            }
        )

    def test_removes_zero_gpa(self) -> None:
        result = filter_rq3_data(self._base_df())
        assert (result["gpa_4_standardized"] > 0.0).all()

    def test_removes_missing_cost(self) -> None:
        result = filter_rq3_data(self._base_df())
        assert result["cost_total_usd"].notna().all()

    def test_removes_invalid_status(self) -> None:
        df = pd.DataFrame(
            {
                "gpa_standardization_status": ["already_4_scale", "out_of_range"],
                "gpa_4_standardized": [3.5, 3.2],
                "cost_total_usd": [80_000.0, 90_000.0],
            }
        )
        result = filter_rq3_data(df)
        assert len(result) == 1
        assert result["gpa_standardization_status"].iloc[0] == "already_4_scale"

    def test_keeps_valid_rows(self) -> None:
        df = pd.DataFrame(
            {
                "gpa_standardization_status": ["converted_from_100"],
                "gpa_4_standardized": [3.7],
                "cost_total_usd": [100_000.0],
            }
        )
        result = filter_rq3_data(df)
        assert len(result) == 1

    def test_returns_copy(self) -> None:
        df = pd.DataFrame(
            {
                "gpa_standardization_status": ["already_4_scale"],
                "gpa_4_standardized": [3.5],
                "cost_total_usd": [80_000.0],
            }
        )
        result = filter_rq3_data(df)
        result["cost_total_usd"] = 0
        assert df["cost_total_usd"].iloc[0] == 80_000.0


# ── compute_within_country_correlations ───────────────────────────────────────

class TestComputeWithinCountryCorrelations:
    def _country_df(self) -> pd.DataFrame:
        """DataFrame with two countries, one above and one below min_n."""
        return pd.DataFrame(
            {
                "cost_country": ["UK"] * 20 + ["Tiny"] * 3,
                "gpa_4_standardized": list(range(1, 21)) + [1.0, 2.0, 3.0],
                "cost_total_usd": (
                    [i * 5_000 for i in range(1, 21)] + [10_000, 20_000, 30_000]
                ),
            }
        )

    def test_country_below_min_n_excluded(self) -> None:
        result = compute_within_country_correlations(self._country_df(), min_n=10)
        assert "Tiny" not in result["country"].values

    def test_country_above_min_n_included(self) -> None:
        result = compute_within_country_correlations(self._country_df(), min_n=10)
        assert "UK" in result["country"].values

    def test_returns_dataframe_with_expected_columns(self) -> None:
        result = compute_within_country_correlations(self._country_df(), min_n=10)
        for col in ("country", "n", "rho", "p_value", "significant"):
            assert col in result.columns

    def test_positive_correlation_detected(self) -> None:
        result = compute_within_country_correlations(self._country_df(), min_n=10)
        uk_rho = result.loc[result["country"] == "UK", "rho"].iloc[0]
        assert uk_rho > 0.9  # near-perfect positive correlation


# ── load_matched_data ─────────────────────────────────────────────────────────

class TestLoadMatchedData:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_matched_data(tmp_path / "ghost.csv")

    def test_loads_csv_and_coerces_numeric_columns(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text(
            "match_status,cost_total_usd,gpa_4_standardized\n"
            "matched,80000,3.5\n"
            "unmatched,,\n",
            encoding="utf-8",
        )
        df = load_matched_data(p)
        assert df["cost_total_usd"].dtype == float
        assert pd.isna(df["cost_total_usd"].iloc[1])
