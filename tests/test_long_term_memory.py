"""Tests for the long-term memory module (src/memory/long_term.py).

Test scenarios
--------------
1. test_load_memory           — loads JSON, validates required fields, returns list
2. test_role_specific_retrieval — agent="reasoner", medication query returns relevant rules
3. test_metadata_filter       — agent + topic filter returns matching rules only
4. test_read_only_raises      — add_rule() and save() raise RuntimeError when read_only=True
5. test_v3_state_integration  — fake question → state has global/retrieval/reasoner/verifier memory
6. test_v2_no_memory          — ENABLE_MEMORY=false → all memory fields are empty lists
7. test_official_mode_prevents_writes — read_only=True, allow_writes=False

Additional unit tests cover schema validation, search scoring, and edge cases.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
RULES_FILE = REPO_ROOT / "memory" / "long_term_rules.json"


@pytest.fixture
def mem():
    """A fresh (writable) LongTermMemory loaded from the real rules file."""
    from medqa_multi_agents.memory.long_term import LongTermMemory
    m = LongTermMemory(str(RULES_FILE), read_only=False)
    m.load()
    return m


@pytest.fixture
def readonly_mem():
    """A read-only LongTermMemory loaded from the real rules file."""
    from medqa_multi_agents.memory.long_term import LongTermMemory
    m = LongTermMemory(str(RULES_FILE), read_only=True)
    m.load()
    return m


# ---------------------------------------------------------------------------
# 1. Test long-term memory loading
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_returns_list(self, mem):
        """load() returns a non-empty list of dicts."""
        rules = mem.load()
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_load_required_fields(self, mem):
        """Every rule has all required schema fields."""
        from medqa_multi_agents.memory.long_term import REQUIRED_FIELDS
        for rule in mem.load():
            missing = REQUIRED_FIELDS - rule.keys()
            assert missing == set(), f"Rule {rule.get('id')} missing: {missing}"

    def test_load_file_not_found(self, tmp_path):
        """load() raises FileNotFoundError for missing file."""
        from medqa_multi_agents.memory.long_term import LongTermMemory
        m = LongTermMemory(str(tmp_path / "nonexistent.json"), read_only=True)
        with pytest.raises(FileNotFoundError):
            m.load()

    def test_load_missing_field_raises(self, tmp_path):
        """load() raises ValueError if a record is missing required fields."""
        from medqa_multi_agents.memory.long_term import LongTermMemory
        bad = [{"id": "bad_001", "agent": "global"}]  # missing many fields
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        m = LongTermMemory(str(p), read_only=True)
        with pytest.raises(ValueError, match="missing required fields"):
            m.load()

    def test_load_returns_known_ids(self, mem):
        """Rules file contains expected initial rule IDs."""
        ids = {r["id"] for r in mem.load()}
        assert "global_001" in ids
        assert "retrieval_001" in ids
        assert "reasoner_001" in ids
        assert "verifier_001" in ids
        assert "finalizer_001" in ids

    def test_len(self, mem):
        """__len__ reflects the number of loaded rules."""
        assert len(mem) == 9


# ---------------------------------------------------------------------------
# 2. Test role-specific retrieval
# ---------------------------------------------------------------------------

class TestRoleSpecificRetrieval:
    def test_reasoner_returns_reasoner_rules(self, readonly_mem):
        """get_rules_for_agent('reasoner') returns only reasoner rules."""
        rules = readonly_mem.get_rules_for_agent("reasoner", query="patient develops symptoms")
        for r in rules:
            assert r["agent"] == "reasoner"

    def test_medication_query_returns_pharmacology_rule(self, readonly_mem):
        """Medication-related query for reasoner returns the pharmacology rule."""
        rules = readonly_mem.get_rules_for_agent(
            "reasoner",
            query="patient develops dry cough after starting medication",
        )
        rule_ids = {r["id"] for r in rules}
        # reasoner_002 is about adverse drug effects / medication
        assert "reasoner_002" in rule_ids, f"Expected reasoner_002 in {rule_ids}"

    def test_global_rules_returned_for_global_agent(self, readonly_mem):
        """get_rules_for_agent('global') returns global rules."""
        rules = readonly_mem.get_rules_for_agent("global", query="format output answer")
        assert any(r["agent"] == "global" for r in rules)

    def test_top_k_respected(self, readonly_mem):
        """get_rules_for_agent respects top_k parameter."""
        rules = readonly_mem.get_rules_for_agent("retrieval_planner", query="diagnosis", top_k=1)
        assert len(rules) <= 1

    def test_unknown_agent_returns_empty(self, readonly_mem):
        """Unknown agent name returns empty list."""
        rules = readonly_mem.get_rules_for_agent("nonexistent_agent", query="anything")
        assert rules == []


# ---------------------------------------------------------------------------
# 3. Test metadata filter
# ---------------------------------------------------------------------------

class TestMetadataFilter:
    def test_filter_by_agent_and_topic(self, readonly_mem):
        """Filter by agent=retrieval_planner, topic=pharmacology returns matching rules only."""
        rules = readonly_mem.search(query="drug", agent="retrieval_planner", topic="pharmacology")
        for r in rules:
            assert r["agent"] == "retrieval_planner"
            assert r["topic"] == "pharmacology"

    def test_filter_by_memory_type(self, readonly_mem):
        """Filter by memory_type=verifier_checklist returns only verifier rules."""
        rules = readonly_mem.search(query="evidence", memory_type="verifier_checklist")
        for r in rules:
            assert r["memory_type"] == "verifier_checklist"

    def test_filter_by_tags(self, readonly_mem):
        """Filter by tags=['pharmacology'] returns rules that have that tag."""
        rules = readonly_mem.search(query="drug", tags=["pharmacology"])
        for r in rules:
            assert "pharmacology" in r["tags"]

    def test_filter_no_match_returns_empty(self, readonly_mem):
        """Filters with no matching rules return empty list."""
        rules = readonly_mem.search(
            query="anything",
            agent="retrieval_planner",
            topic="definitely_not_a_real_topic_xyz",
        )
        assert rules == []

    def test_search_without_filter_returns_all_top_k(self, readonly_mem):
        """search() without metadata filters returns top-k across all agents."""
        rules = readonly_mem.search(query="diagnosis symptoms", top_k=5)
        assert len(rules) <= 5


# ---------------------------------------------------------------------------
# 4. Test read-only behaviour
# ---------------------------------------------------------------------------

class TestReadOnly:
    def test_add_rule_raises_in_read_only_mode(self, readonly_mem):
        """add_rule() raises RuntimeError when read_only=True."""
        new_rule = {
            "id": "test_001",
            "agent": "global",
            "memory_type": "procedural_rule",
            "topic": "test",
            "rule": "Test rule.",
            "source": "test",
            "tags": ["test"],
            "confidence": 0.5,
            "created_at": "2026-07-01",
        }
        with pytest.raises(RuntimeError, match="read-only"):
            readonly_mem.add_rule(new_rule)

    def test_save_raises_in_read_only_mode(self, readonly_mem):
        """save() raises RuntimeError when read_only=True."""
        with pytest.raises(RuntimeError, match="read-only"):
            readonly_mem.save()

    def test_add_rule_succeeds_in_write_mode(self, mem):
        """add_rule() works when read_only=False."""
        before = len(mem)
        new_rule = {
            "id": "test_writable_001",
            "agent": "global",
            "memory_type": "procedural_rule",
            "topic": "test",
            "rule": "A writable test rule.",
            "source": "test",
            "tags": ["test"],
            "confidence": 0.5,
            "created_at": "2026-07-01",
        }
        mem.add_rule(new_rule)
        assert len(mem) == before + 1

    def test_add_rule_missing_fields_raises(self, mem):
        """add_rule() raises ValueError for incomplete rule record."""
        with pytest.raises(ValueError, match="missing required fields"):
            mem.add_rule({"id": "incomplete"})

    def test_official_config_is_read_only(self):
        """LongTermMemory default read_only=True matches official config requirement."""
        from medqa_multi_agents.memory.long_term import LongTermMemory
        m = LongTermMemory("memory/long_term_rules.json")
        assert m.read_only is True, "Default must be read_only=True for official eval safety"


# ---------------------------------------------------------------------------
# 5. Test V3 state integration
# ---------------------------------------------------------------------------

class TestV3StateIntegration:
    def test_load_memory_node_populates_state(self, monkeypatch):
        """_node_load_memory() produces state with all four memory fields (ENABLE_MEMORY=true)."""
        import medqa_multi_agents as pkg
        # Directly patch the module-level flag — no reload needed
        monkeypatch.setattr(pkg, "ENABLE_MEMORY", True)

        from medqa_multi_agents import _node_load_memory

        fake_state: dict = {
            "question": "A 45-year-old patient presents with dry cough after starting lisinopril.",
            "global_memory": [],
            "retrieval_memory": [],
            "reasoner_memory": [],
            "verifier_memory": [],
        }

        result = _node_load_memory(fake_state)
        assert "global_memory" in result
        assert "retrieval_memory" in result
        assert "reasoner_memory" in result
        assert "verifier_memory" in result
        assert isinstance(result["global_memory"], list)
        assert isinstance(result["retrieval_memory"], list)
        assert isinstance(result["reasoner_memory"], list)
        assert isinstance(result["verifier_memory"], list)

    def test_load_memory_returns_rule_dicts(self, monkeypatch):
        """Memory fields contain rule dicts with expected schema fields (ENABLE_MEMORY=true)."""
        import medqa_multi_agents as pkg
        monkeypatch.setattr(pkg, "ENABLE_MEMORY", True)

        from medqa_multi_agents import _node_load_memory

        result = _node_load_memory({
            "question": "What drug causes dry cough?",
            "global_memory": [], "retrieval_memory": [],
            "reasoner_memory": [], "verifier_memory": [],
        })

        for field in ("global_memory", "retrieval_memory", "reasoner_memory", "verifier_memory"):
            for rule in result[field]:
                assert "id" in rule
                assert "rule" in rule
                assert "agent" in rule


# ---------------------------------------------------------------------------
# 6. Test V2 does not use explicit long-term memory
# ---------------------------------------------------------------------------

class TestV2NoMemory:
    def test_load_memory_returns_empty_lists_when_disabled(self, monkeypatch):
        """When ENABLE_MEMORY=false, _node_load_memory returns empty lists."""
        import medqa_multi_agents as pkg
        monkeypatch.setattr(pkg, "ENABLE_MEMORY", False)

        from medqa_multi_agents import _node_load_memory

        result = _node_load_memory({
            "question": "What is the treatment for hypertension?",
            "global_memory": [], "retrieval_memory": [],
            "reasoner_memory": [], "verifier_memory": [],
        })

        assert result["global_memory"] == []
        assert result["retrieval_memory"] == []
        assert result["reasoner_memory"] == []
        assert result["verifier_memory"] == []

    def test_v2_memory_fields_are_empty_in_state(self, monkeypatch):
        """V2 state has empty memory lists (ENABLE_MEMORY=false)."""
        import medqa_multi_agents as pkg
        monkeypatch.setattr(pkg, "ENABLE_MEMORY", False)

        from medqa_multi_agents import _node_load_memory

        state = {
            "question": "test question",
            "global_memory": [], "retrieval_memory": [],
            "reasoner_memory": [], "verifier_memory": [],
        }
        result = _node_load_memory(state)
        for field in ("global_memory", "retrieval_memory", "reasoner_memory", "verifier_memory"):
            assert result[field] == []


# ---------------------------------------------------------------------------
# 7. Test official mode prevents writes
# ---------------------------------------------------------------------------

class TestOfficialMode:
    def test_default_is_read_only(self):
        """LongTermMemory defaults to read_only=True."""
        from medqa_multi_agents.memory.long_term import LongTermMemory
        m = LongTermMemory("memory/long_term_rules.json")
        assert m.read_only is True

    def test_module_singleton_is_read_only(self):
        """The module-level long_term_memory instance is read-only."""
        from medqa_multi_agents.memory.long_term import long_term_memory
        assert long_term_memory.read_only is True

    def test_official_mode_add_rule_blocked(self):
        """Official mode: add_rule() raises RuntimeError."""
        from medqa_multi_agents.memory.long_term import LongTermMemory
        m = LongTermMemory(str(RULES_FILE), read_only=True)
        m.load()
        with pytest.raises(RuntimeError):
            m.add_rule({
                "id": "x", "agent": "global", "memory_type": "procedural_rule",
                "topic": "test", "rule": ".", "source": "test",
                "tags": [], "confidence": 1.0, "created_at": "2026-07-01",
            })

    def test_official_mode_save_blocked(self):
        """Official mode: save() raises RuntimeError."""
        from medqa_multi_agents.memory.long_term import LongTermMemory
        m = LongTermMemory(str(RULES_FILE), read_only=True)
        m.load()
        with pytest.raises(RuntimeError):
            m.save()


# ---------------------------------------------------------------------------
# Misc unit tests
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_returns_tuple(self):
        from medqa_multi_agents.memory.long_term import extract_keywords
        assert isinstance(extract_keywords("hypertension diabetes mellitus"), tuple)

    def test_max_three(self):
        from medqa_multi_agents.memory.long_term import extract_keywords
        kw = extract_keywords("hypertension diabetes mellitus cardiomegaly pulmonary")
        assert len(kw) <= 3

    def test_deduplication(self):
        from medqa_multi_agents.memory.long_term import extract_keywords
        kw = extract_keywords("hypertension hypertension hypertension diabetes")
        assert kw.count("hypertension") <= 1

    def test_empty_string(self):
        from medqa_multi_agents.memory.long_term import extract_keywords
        kw = extract_keywords("")
        assert isinstance(kw, tuple)


class TestFormatRulesForPrompt:
    def test_empty_list(self, readonly_mem):
        result = readonly_mem.format_rules_for_prompt([])
        assert "No relevant" in result

    def test_non_empty_list(self, readonly_mem):
        rules = readonly_mem.load()[:2]
        result = readonly_mem.format_rules_for_prompt(rules)
        assert "[" in result  # contains id brackets

    def test_exported_from_package(self):
        from medqa_multi_agents.memory import LongTermMemory, long_term_memory
        assert callable(long_term_memory.search)
        assert callable(long_term_memory.get_rules_for_agent)
