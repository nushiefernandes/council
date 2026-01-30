#!/usr/bin/env python3
"""
Tests for the enhanced skill system in crew.py.

These tests verify the skill discovery, description extraction, listing,
and querying functionality across multiple skill locations:
- ~/.claude/skills/ (user skills)
- ~/.claude/plugins/marketplaces/*/plugins/*/skills/ (plugin skills)

Run with: pytest test_skill_system.py -v
"""

import pytest
from pathlib import Path

# Import the functions to test from crew.py
# Note: list_skills_tool and query_skill are @tool decorated, so we need the underlying functions
from crew import find_skill, extract_description

# For the tool-decorated functions, we import and call .func to get the underlying function
import crew


def call_list_skills():
    """Helper to call list_skills_tool (which is a CrewAI Tool object)."""
    # The @tool decorator wraps functions - access via .func or just call directly
    return crew.list_skills_tool.func()


def call_query_skill(skill_name: str, query: str):
    """Helper to call query_skill (which is a CrewAI Tool object)."""
    return crew.query_skill.func(skill_name, query)


class TestFindSkill:
    """Tests for find_skill() function."""

    def test_find_postgres_best_practices_in_user_skills(self):
        """find_skill('postgres-best-practices') should find the user skill."""
        result = find_skill("postgres-best-practices")
        assert result is not None, "postgres-best-practices skill should be found"
        # AGENTS.md is preferred over SKILL.md if both exist
        assert "postgres-best-practices" in str(result)
        assert result.exists()

    def test_find_frontend_design_in_plugins(self):
        """find_skill('frontend-design') should find the plugin version."""
        result = find_skill("frontend-design")
        assert result is not None, "frontend-design skill should be found in plugins"
        assert "plugins" in str(result), f"frontend-design should be from plugins, got {result}"
        assert "frontend-design" in str(result)
        assert str(result).endswith("SKILL.md") or str(result).endswith("AGENTS.md")

    def test_find_baby_council_prefers_agents_md(self):
        """find_skill('baby-council') should prefer AGENTS.md over SKILL.md."""
        result = find_skill("baby-council")
        assert result is not None, "baby-council skill should be found"
        assert str(result).endswith("AGENTS.md"), f"Should prefer AGENTS.md, got {result}"

    def test_find_nonexistent_returns_none(self):
        """find_skill('nonexistent') should return None."""
        result = find_skill("nonexistent-skill-xyz-123")
        assert result is None, "Nonexistent skill should return None"

    def test_find_skill_case_sensitivity(self):
        """Test that skill names are matched correctly."""
        result_lower = find_skill("postgres-best-practices")
        assert result_lower is not None, "Lowercase skill name should work"

    def test_find_react_best_practices(self):
        """find_skill('react-best-practices') should find the user skill."""
        result = find_skill("react-best-practices")
        assert result is not None, "react-best-practices skill should be found"
        assert "react-best-practices" in str(result)


class TestExtractDescription:
    """Tests for extract_description() function."""

    def test_extract_description_from_skill_file(self):
        """extract_description should parse description from skill files."""
        skill_path = find_skill("postgres-best-practices")
        if skill_path and skill_path.exists():
            description = extract_description(skill_path)
            assert description is not None
            assert isinstance(description, str)
            assert len(description) > 0

    def test_extract_description_from_frontend_design(self):
        """extract_description should work with plugin SKILL.md files."""
        skill_path = find_skill("frontend-design")
        if skill_path:
            description = extract_description(skill_path)
            assert description is not None
            assert isinstance(description, str)

    def test_extract_description_handles_missing_file(self):
        """extract_description should handle missing files gracefully."""
        fake_path = Path("/nonexistent/path/SKILL.md")
        result = extract_description(fake_path)
        # Should return "No description available", not raise an exception
        assert result == "No description available"

    def test_extract_description_truncates_long_descriptions(self):
        """extract_description should truncate long descriptions."""
        skill_path = find_skill("frontend-design")
        if skill_path:
            description = extract_description(skill_path, max_length=50)
            assert len(description) <= 53  # 50 + "..."


class TestListSkillsTool:
    """Tests for list_skills_tool() function."""

    def test_list_skills_returns_string(self):
        """list_skills should return a string."""
        result = call_list_skills()
        assert isinstance(result, str)

    def test_list_skills_finds_multiple_skills(self):
        """list_skills should find multiple skills."""
        result = call_list_skills()
        assert "Available skills" in result
        # Count skill entries (lines starting with "  - ")
        skill_lines = [l for l in result.split('\n') if l.strip().startswith('- ')]
        assert len(skill_lines) >= 5, f"Expected 5+ skills, found {len(skill_lines)}"

    def test_list_skills_includes_frontend_design(self):
        """list_skills should include 'frontend-design' from plugins."""
        result = call_list_skills()
        assert "frontend-design" in result, "frontend-design should be in skill list"

    def test_list_skills_excludes_ralph(self):
        """list_skills should NOT include 'ralph' (has no SKILL.md)."""
        result = call_list_skills()
        # ralph exists but has no SKILL.md, only ralph.sh
        lines = result.split('\n')
        ralph_lines = [l for l in lines if '- ralph:' in l]
        assert len(ralph_lines) == 0, f"ralph should not be listed: {ralph_lines}"

    def test_list_skills_includes_user_skills(self):
        """list_skills should include skills from ~/.claude/skills/."""
        result = call_list_skills()
        # These should be in user skills directory
        assert "postgres-best-practices" in result or "react-best-practices" in result

    def test_list_skills_deduplicates(self):
        """list_skills should dedupe skills that exist in multiple locations."""
        result = call_list_skills()
        # frontend-design exists in multiple plugin dirs
        count = result.count("- frontend-design:")
        assert count <= 1, f"frontend-design should appear at most once, found {count}"


class TestQuerySkill:
    """Tests for query_skill() function."""

    def test_query_frontend_design_typography(self):
        """query_skill('frontend-design', 'typography') should return guidance."""
        result = call_query_skill("frontend-design", "typography")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Error" not in result or "typography" in result.lower()

    def test_query_nonexistent_skill_returns_helpful_error(self):
        """query_skill for nonexistent skill should return helpful error."""
        result = call_query_skill("nonexistent-xyz-123", "anything")
        assert result is not None
        assert "not found" in result.lower()
        assert "list_skills" in result

    def test_query_postgres_best_practices(self):
        """query_skill('postgres-best-practices', 'index') should return guidance."""
        result = call_query_skill("postgres-best-practices", "index")
        assert result is not None
        assert isinstance(result, str)

    def test_query_with_no_matches(self):
        """query_skill with non-matching query should return helpful message."""
        result = call_query_skill("frontend-design", "xyznonexistent123")
        assert result is not None
        assert "No matches" in result or "list_skills" in result

    def test_query_baby_council(self):
        """query_skill('baby-council', 'agents') should work."""
        result = call_query_skill("baby-council", "agents")
        assert result is not None
        assert isinstance(result, str)


class TestSkillDiscoveryLocations:
    """Tests verifying skill discovery across all expected locations."""

    def test_user_skills_directory_exists(self):
        """Verify ~/.claude/skills/ directory is checked."""
        user_skills = Path.home() / ".claude" / "skills"
        assert user_skills.exists()

    def test_plugin_marketplaces_directory_exists(self):
        """Verify plugin marketplaces directory is checked."""
        marketplaces = Path.home() / ".claude" / "plugins" / "marketplaces"
        assert marketplaces.exists()

    def test_finds_skills_in_nested_plugin_structure(self):
        """Verify skills in plugins/*/skills/*/ structure are found."""
        result = find_skill("frontend-design")
        assert result is not None
        assert "plugins" in str(result)


class TestEdgeCases:
    """Edge case tests for the skill system."""

    def test_skill_with_hyphens(self):
        """Test handling of skill names with hyphens."""
        result = find_skill("postgres-best-practices")
        assert result is not None

    def test_find_skill_with_trailing_slash(self):
        """Test that trailing slash doesn't break things."""
        # This should return None gracefully
        result = find_skill("postgres-best-practices/")
        # Implementation may or may not handle this - just shouldn't crash
        assert result is None or isinstance(result, Path)

    def test_list_skills_consistent_format(self):
        """Verify list_skills returns consistent format on multiple calls."""
        result1 = call_list_skills()
        result2 = call_list_skills()
        assert type(result1) == type(result2)
        # Should have same number of skills
        count1 = result1.count("- ")
        count2 = result2.count("- ")
        assert count1 == count2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
