import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.scoring import (  # noqa: E402
    DEFAULT_WEIGHTS,
    SCORING_PROFILES,
    SKILL_CAP_FLOOR,
    PreparedCv,
    PreparedJob,
    category_component,
    compute_final_score,
    core_keyword_coverage,
    enrich_cv_skills,
    experience_component,
    experience_zone_score,
    get_skill_embedding_tuning,
    infer_seniority_years,
    jobtype_component,
    location_component,
    normalize_priority_keyword,
    priority_keyword_component,
    qualification_component,
    resolve_priority_keywords,
    salary_component,
    set_skill_embedding_tuning,
    skills_component,
    split_priority_keyword_terms,
    weights_for_profile,
)


def make_job(**overrides) -> PreparedJob:
    defaults = dict(
        text="",
        semantic_text="",
        tokens=set(),
        keywords=[],
        keyword_set=set(),
        skills_canonical=set(),
        location="",
        category="",
        jobtype="",
        min_experience_years=None,
        salary_min=None,
        salary_max=None,
        title="",
        keyword_terms_raw=[],
        scoring_profile=None,
    )
    defaults.update(overrides)
    return PreparedJob(**defaults)


def make_cv(**overrides) -> PreparedCv:
    defaults = dict(
        payload=object(),
        text="",
        text_tokens=set(),
        title_tokens=set(),
        keywords=[],
        keyword_set=set(),
        skills_canonical=set(),
        location="",
        category="",
        jobtype="",
        experience_years=None,
        salary_expected_min=None,
        salary_expected_max=None,
        qualified=None,
    )
    defaults.update(overrides)
    return PreparedCv(**defaults)


# ── skills_component ────────────────────────────────────────────────────


def test_skills_component_no_signal_when_job_has_no_skills():
    job = make_job(skills_canonical=set())
    value, ok, hits = skills_component(job, frozenset({"Python"}))
    assert (value, ok, hits) == (0.5, False, [])


def test_skills_component_no_signal_when_cv_has_no_skills():
    job = make_job(skills_canonical={"Python"})
    value, ok, hits = skills_component(job, frozenset())
    assert (value, ok, hits) == (0.0, False, [])


def test_skills_component_partial_coverage():
    job = make_job(skills_canonical={"Python", "SQL", "Docker", "Git"})
    value, ok, hits = skills_component(job, frozenset({"Python", "SQL"}))
    assert ok is True
    assert value == 0.5
    assert hits == ["Python", "SQL"]


def test_skills_component_full_coverage():
    job = make_job(skills_canonical={"Python", "SQL"})
    value, ok, _hits = skills_component(job, frozenset({"Python", "SQL", "Docker"}))
    assert ok is True
    assert value == 1.0


def test_skills_component_semantic_credit_tops_up_unmatched():
    job = make_job(skills_canonical={"Python", "SQL"})
    # SQL not present literally, but the injected credit function grants
    # partial credit for it (simulates an embedding near-miss).
    def credit_fn(unmatched, cv_skills):
        return 0.5 if "SQL" in unmatched else 0.0

    value, ok, hits = skills_component(job, frozenset({"Python"}), credit_fn)
    assert ok is True
    assert hits == ["Python"]  # exact hits unaffected by semantic credit
    assert value == 0.75  # (1.0 + 0.5) / 2


def test_skills_component_semantic_credit_never_exceeds_1():
    job = make_job(skills_canonical={"Python"})

    def credit_fn(unmatched, cv_skills):
        return 5.0  # absurdly high, must be capped

    value, ok, _hits = skills_component(job, frozenset({"Java"}), credit_fn)
    assert value == 1.0


# ── split_priority_keyword_terms / normalize_priority_keyword ───────────


def test_split_priority_keyword_terms_strips_bullets_and_header():
    raw = "Mots Clés :\n- Laravel\n• VueJS\n1. Docker"
    terms = split_priority_keyword_terms(raw)
    assert terms == ["Laravel", "VueJS", "Docker"]


def test_split_priority_keyword_terms_preserves_repeats():
    raw = "Laravel, Laravel, Laravel"
    terms = split_priority_keyword_terms(raw)
    assert terms == ["Laravel", "Laravel", "Laravel"]


def test_split_priority_keyword_terms_does_not_split_on_slash():
    # Unlike the general keywords field, "/" is kept intact so
    # normalize_priority_keyword can apply its own slash-combo handling.
    terms = split_priority_keyword_terms("MOA / AMOA")
    assert terms == ["MOA / AMOA"]


def test_normalize_priority_keyword_falls_back_to_none_for_unknown_term():
    assert normalize_priority_keyword("TRM XYZZY") is None


def test_normalize_priority_keyword_splits_on_slash_when_both_sides_agree():
    # Both "javascript" and "js" normalize to the same canonical via
    # _TECH_SKILLS aliases -- the combined raw string doesn't match anything
    # on its own, but each side does and they agree.
    canonical = normalize_priority_keyword("JavaScript/JS")
    assert canonical is not None


# ── enrich_cv_skills ───────────────────────────────────────────────────


def test_enrich_cv_skills_adds_literal_unknown_term_found_in_cv_text():
    job = make_job(keyword_terms_raw=["TRM"])
    cv = make_cv(skills_canonical=set(), text="Expérience sur le référentiel TRM en 2023.")
    enriched = enrich_cv_skills(job, cv)
    assert "TRM" in enriched


def test_enrich_cv_skills_skips_term_not_present_in_cv_text():
    job = make_job(keyword_terms_raw=["TRM"])
    cv = make_cv(skills_canonical=set(), text="Aucun rapport avec ce sigle.")
    enriched = enrich_cv_skills(job, cv)
    assert "TRM" not in enriched


def test_enrich_cv_skills_never_mutates_original_set():
    job = make_job(keyword_terms_raw=["TRM"])
    original = {"Python"}
    cv = make_cv(skills_canonical=original, text="TRM partout ici.")
    enrich_cv_skills(job, cv)
    assert original == {"Python"}  # untouched


# ── priority_keyword_component / resolve_priority_keywords ──────────────


def test_priority_keyword_component_no_signal_when_job_has_none():
    job = make_job(keyword_terms_raw=[])
    value, ok, hits = priority_keyword_component(job, frozenset({"Laravel"}))
    assert (value, ok, hits) == (0.5, False, [])


def test_priority_keyword_component_deduplicates_by_canonical():
    # Three raw lines naming the same tool differently should count as ONE
    # required canonical, not three.
    job = make_job(keyword_terms_raw=["Laravel", "laravel", "LARAVEL"])
    value, ok, hits = priority_keyword_component(job, frozenset({"Laravel"}))
    assert ok is True
    assert value == 1.0
    assert len(hits) == 1


def test_resolve_priority_keywords_matched_vs_all():
    job = make_job(keyword_terms_raw=["Laravel", "VueJS", "Docker"])
    matched, all_terms = resolve_priority_keywords(job, frozenset({"Laravel"}))
    assert len(all_terms) == 3
    assert len(matched) == 1


# ── core_keyword_coverage ────────────────────────────────────────────────


def test_core_keyword_coverage_no_penalty_when_no_priority_keywords():
    job = make_job(keyword_terms_raw=[])
    assert core_keyword_coverage(job, frozenset()) == 1.0


def test_core_keyword_coverage_repeated_keyword_missing_from_cv():
    job = make_job(keyword_terms_raw=["SAS", "SAS", "SAS", "Excel"], title="Data Analyst")
    coverage = core_keyword_coverage(job, frozenset({"Excel"}))
    assert coverage == 0.0  # the one "core" (repeated 3x) canonical is missing


def test_core_keyword_coverage_repeated_keyword_present_in_cv():
    job = make_job(keyword_terms_raw=["SAS", "SAS", "SAS"], title="Data Analyst")
    coverage = core_keyword_coverage(job, frozenset({"SAS"}))
    assert coverage == 1.0


def test_core_keyword_coverage_title_mention_counts_as_core():
    job = make_job(keyword_terms_raw=["SAS"], title="Data Analyst Expert SAS")
    coverage_missing = core_keyword_coverage(job, frozenset())
    coverage_present = core_keyword_coverage(job, frozenset({"SAS"}))
    assert coverage_missing == 0.0
    assert coverage_present == 1.0


def test_core_keyword_coverage_title_alternation_disables_title_check():
    # "/" in the title signals alternative labels, not one headline tool --
    # the title-based core-keyword check must not fire here.
    job = make_job(
        keyword_terms_raw=["Concepteur Decisionnel"],
        title="Data Analyst / Concepteur Decisionnel Senior",
    )
    # Not repeated (only 1 occurrence) and title check disabled -> no core
    # keyword at all -> no penalty regardless of CV content.
    assert core_keyword_coverage(job, frozenset()) == 1.0


def test_core_keyword_coverage_final_score_applies_multiplicative_penalty():
    job = make_job(
        skills_canonical=set(),
        keyword_terms_raw=["SAS", "SAS", "SAS"],
        title="Data Analyst",
    )
    cv = make_cv(skills_canonical=set())
    result = compute_final_score(job, cv, similarity=1.0, rerank_score=None)
    assert result.core_keyword_coverage == 0.0
    # Multiplicative penalty floor (0.45) must have been applied.
    assert result.score < 50.0


# ── seniority inference ──────────────────────────────────────────────────


def test_infer_seniority_years_from_title():
    job = make_job(title="Développeur Java Senior")
    assert infer_seniority_years(job) == 5


def test_infer_seniority_years_expert_outranks_others():
    job = make_job(title="Consultant Expert SAS")
    assert infer_seniority_years(job) == 8


def test_infer_seniority_years_returns_zero_when_no_signal():
    job = make_job(title="Développeur Web", text="Développeur Web")
    assert infer_seniority_years(job) == 0


def test_experience_component_uses_inferred_years_when_no_explicit_requirement():
    job = make_job(min_experience_years=None, title="Ingénieur Senior")
    cv = make_cv(experience_years=5)
    value, ok, required_years = experience_component(job, cv)
    assert ok is True
    assert required_years == 5.0
    # Inferred requirement is dampened toward the neutral 0.75, so a CV that
    # exactly meets it (zone score 1.0) should land strictly between 0.75
    # and 1.0, not at the full 1.0 an explicit match would get.
    assert 0.75 < value < 1.0


def test_experience_component_explicit_requirement_not_dampened():
    job = make_job(min_experience_years=5)
    cv = make_cv(experience_years=5)
    value, ok, required_years = experience_component(job, cv)
    assert (value, ok, required_years) == (1.0, True, 5.0)


# ── experience_zone_score (unchanged core, still covered) ───────────────


def test_experience_zone_equal():
    assert experience_zone_score(5, 5) == 1.0


def test_experience_zone_comfortably_over_is_full_credit():
    assert experience_zone_score(6, 3) == 1.0


def test_experience_zone_severely_over_hits_floor():
    assert experience_zone_score(8, 2) == 0.85


def test_experience_zone_shortfall_is_linear():
    assert experience_zone_score(5, 10) == 0.5


# ── jobtype / category / location / salary / qualification ──────────────


def test_jobtype_component_match():
    job = make_job(jobtype="cdi")
    cv = make_cv(jobtype="cdi")
    assert jobtype_component(job, cv) == (1.0, True)


def test_category_component_match():
    job = make_job(category="developpement web")
    cv = make_cv(category="developpement web")
    assert category_component(job, cv) == (1.0, True)


def test_location_component_substring_match():
    job = make_job(location="paris")
    cv = make_cv(location="paris 15e")
    assert location_component(job, cv) == (1.0, True)


def test_salary_component_within_budget():
    job = make_job(salary_max=50000)
    cv = make_cv(salary_expected_min=45000)
    assert salary_component(job, cv) == (1.0, True)


def test_qualification_component_none_is_no_signal():
    cv = make_cv(qualified=None)
    assert qualification_component(cv) == (0.5, False)


# ── scoring profiles ──────────────────────────────────────────────────────


def test_weights_for_profile_falls_back_to_default_on_unknown_name():
    weights = weights_for_profile("nom-inconnu")
    assert weights == DEFAULT_WEIGHTS


def test_weights_for_profile_returns_named_preset():
    weights = weights_for_profile("priorite_experience")
    assert weights == SCORING_PROFILES["priorite_experience"]
    assert weights["experience"] > DEFAULT_WEIGHTS["experience"]


def test_weights_for_profile_none_uses_base_or_default():
    assert weights_for_profile(None) == DEFAULT_WEIGHTS
    custom_base = {**DEFAULT_WEIGHTS, "semantic": 0.99}
    assert weights_for_profile(None, base=custom_base) == custom_base


def test_compute_final_score_uses_job_scoring_profile_via_explicit_weights():
    job = make_job(min_experience_years=3)
    cv = make_cv(experience_years=8)  # far over -> lower zone score than "equal"
    profile_weights = weights_for_profile("priorite_experience")
    result = compute_final_score(job, cv, similarity=0.5, weights=profile_weights)
    assert result.weights == profile_weights


# ── compute_final_score : moyenne pondérée renormalisée (cœur inchangé) ──


def test_final_score_renormalizes_over_missing_components():
    job = make_job(skills_canonical={"Python", "SQL"})
    cv = make_cv(skills_canonical={"Python", "SQL"})
    result = compute_final_score(job, cv, similarity=1.0, rerank_score=None)

    expected_weight_total = DEFAULT_WEIGHTS["semantic"] + DEFAULT_WEIGHTS["skills"]
    expected_final = (
        DEFAULT_WEIGHTS["semantic"] * 1.0 + DEFAULT_WEIGHTS["skills"] * 1.0
    ) / expected_weight_total
    assert result.score == round(expected_final * 100, 2)
    assert set(result.low_confidence_components) == {
        "keywords",
        "experience",
        "jobtype",
        "category",
        "location",
        "salary",
        "qualification",
    }


def test_final_score_uses_rerank_score_over_raw_similarity():
    job = make_job()
    cv = make_cv()
    result = compute_final_score(job, cv, similarity=0.1, rerank_score=0.9)
    assert result.semantic == 0.9


def test_skill_coverage_caps_the_final_score():
    job = make_job(skills_canonical={"Python", "SQL", "Docker", "Kubernetes"})
    cv = make_cv(skills_canonical={"Python"})  # 1/4 coverage = 0.25
    result = compute_final_score(job, cv, similarity=1.0, rerank_score=None)

    cap = (SKILL_CAP_FLOOR + (1 - SKILL_CAP_FLOOR) * 0.25) * 100
    assert result.score <= round(cap, 2)


def test_skill_coverage_cap_does_not_hold_back_perfect_coverage():
    job = make_job(skills_canonical={"Python"})
    cv = make_cv(skills_canonical={"Python"})
    result = compute_final_score(job, cv, similarity=1.0, rerank_score=None)
    assert result.score == 100.0


def test_custom_weights_are_respected():
    job = make_job(skills_canonical={"Python"})
    cv = make_cv(skills_canonical={"Python"})
    custom = dict(DEFAULT_WEIGHTS)
    custom["skills"] = 0.0
    result = compute_final_score(job, cv, similarity=0.5, rerank_score=None, weights=custom)
    assert result.score == 50.0


# ── réglage live du crédit sémantique de compétences ──────────────────────


@pytest.fixture(autouse=True)
def _reset_skill_embedding_tuning_override():
    set_skill_embedding_tuning(None, None)
    yield
    set_skill_embedding_tuning(None, None)


def test_get_skill_embedding_tuning_defaults_when_no_override():
    threshold, max_credit, overridden = get_skill_embedding_tuning(0.6, 0.8)
    assert (threshold, max_credit, overridden) == (0.6, 0.8, False)


def test_set_skill_embedding_tuning_overrides_only_the_given_field():
    set_skill_embedding_tuning(threshold=0.45, max_credit=None)
    threshold, max_credit, overridden = get_skill_embedding_tuning(0.6, 0.8)
    assert threshold == 0.45
    assert max_credit == 0.8  # non fourni -> valeur par défaut inchangée
    assert overridden is True


def test_set_skill_embedding_tuning_both_none_clears_override():
    set_skill_embedding_tuning(threshold=0.3, max_credit=0.5)
    set_skill_embedding_tuning(None, None)
    threshold, max_credit, overridden = get_skill_embedding_tuning(0.6, 0.8)
    assert (threshold, max_credit, overridden) == (0.6, 0.8, False)


def test_set_skill_embedding_tuning_accumulates_across_calls():
    set_skill_embedding_tuning(threshold=0.4, max_credit=None)
    set_skill_embedding_tuning(threshold=None, max_credit=0.9)
    threshold, max_credit, overridden = get_skill_embedding_tuning(0.6, 0.8)
    assert (threshold, max_credit, overridden) == (0.4, 0.9, True)
