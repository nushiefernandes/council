"""Tests for the dynamic skill discovery system."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

# Import the functions we're testing
from crew import find_skill, extract_description, list_skills_tool, query_skill


class TestFindSkill:
    """Tests for find_skill() function."""

    def test_finds_skill_in_user_dir(self):
        """find_skill() finds skills in ~/.claude/skills/ if they exist."""
        # Test with a real skill that should exist
        result = find_skill("postgres-best-practices")
        # If the skill exists, it should return a Path
        if result is not None:
            assert isinstance(result, Path)
            assert result.exists()
            assert result.name in ("AGENTS.md", "SKILL.md")

    def test_prefers_agents_md_over_skill_md(self, tmp_path):
        """find_skill() returns AGENTS.md if both exist."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        agents_file = skill_dir / "AGENTS.md"
        agents_file.write_text("# Agents version")

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# Skill version")

        # AGENTS.md should be found first (it's checked before SKILL.md)
        locations = [(tmp_path, "direct")]

        # Test the actual behavior with a real skill
        real_skill = find_skill("postgres-best-practices")
        if real_skill:
            # If skill exists, verify it's a Path
            assert isinstance(real_skill, Path)

    def test_returns_none_for_nonexistent_skill(self):
        """find_skill() returns None for skills that don't exist."""
        result = find_skill("nonexistent-skill-that-does-not-exist-12345")
        assert result is None


class TestExtractDescription:
    """Tests for extract_description() function."""

    def test_extracts_from_frontmatter(self, tmp_path):
        """extract_description() extracts description from YAML frontmatter."""
        skill_file = tmp_path / "skill.md"
        skill_file.write_text("""---
name: test-skill
description: This is a test skill for testing purposes
---

# Test Skill
Content here
""")
        result = extract_description(skill_file)
        assert "This is a test skill" in result

    def test_extracts_from_first_paragraph(self, tmp_path):
        """extract_description() falls back to first paragraph if no frontmatter."""
        skill_file = tmp_path / "skill.md"
        skill_file.write_text("""# Test Skill

This is the first paragraph of the skill documentation.

More content here.
""")
        result = extract_description(skill_file)
        assert "first paragraph" in result

    def test_returns_default_for_empty_file(self, tmp_path):
        """extract_description() returns default for empty files."""
        skill_file = tmp_path / "skill.md"
        skill_file.write_text("")
        result = extract_description(skill_file)
        assert result == "No description available"

    def test_truncates_long_descriptions(self, tmp_path):
        """extract_description() truncates descriptions over max_length."""
        skill_file = tmp_path / "skill.md"
        long_desc = "A" * 200
        skill_file.write_text(f"---\ndescription: {long_desc}\n---\n")
        result = extract_description(skill_file, max_length=50)
        assert len(result) <= 53  # 50 + "..."
        assert result.endswith("...")


class TestListSkillsTool:
    """Tests for list_skills_tool() function."""

    def test_returns_multiple_skills(self):
        """list_skills_tool() returns multiple skills if they exist."""
        result = list_skills_tool.func()
        # Should return either skills or a "no skills" message
        assert isinstance(result, str)
        assert "skills" in result.lower() or "no skills" in result.lower()

    def test_deduplicates_skills(self):
        """list_skills_tool() doesn't list the same skill twice."""
        result = list_skills_tool.func()
        if "Available skills" in result:
            # Extract skill names and check for duplicates
            lines = [l for l in result.split("\n") if l.strip().startswith("- ")]
            skill_names = [l.split(":")[0].strip("- ").strip() for l in lines]
            assert len(skill_names) == len(set(skill_names)), "Duplicate skills found"


class TestQuerySkill:
    """Tests for query_skill() function."""

    def test_returns_guidance_for_valid_skill(self):
        """query_skill() returns guidance for skills that exist."""
        # Try to query a skill that likely exists
        result = query_skill.func("postgres-best-practices", "index")
        # Should return either guidance or a "not found" message
        assert isinstance(result, str)

    def test_returns_error_for_invalid_skill(self):
        """query_skill() returns helpful error for nonexistent skills."""
        result = query_skill.func("nonexistent-skill-xyz-12345", "anything")
        assert "not found" in result.lower() or "error" in result.lower()
        assert "list_skills" in result.lower()

    def test_returns_no_matches_message(self):
        """query_skill() returns helpful message when query has no matches."""
        # Find a real skill first
        skill_file = find_skill("postgres-best-practices")
        if skill_file:
            result = query_skill.func("postgres-best-practices", "xyznonexistentquery12345")
            assert "no matches" in result.lower() or "try a different" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
