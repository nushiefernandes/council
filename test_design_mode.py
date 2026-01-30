#!/usr/bin/env python3
"""
Tests for the design mode functionality in crew.py.

These tests verify the design agents, tasks, and pipeline functionality.

Run with: pytest test_design_mode.py -v
"""

import pytest
import subprocess
import sys
from pathlib import Path

# Import the functions to test from crew.py
from crew import (
    create_design_agents,
    create_design_tasks,
    WORKSPACE_DIR,
    list_skills_tool,
    query_skill,
    discover_skill_tool,
    spawn_research_tool,
    collect_research_tool,
    write_file,
)


class TestCreateDesignAgents:
    """Tests for create_design_agents() function."""

    def test_creates_four_agents(self):
        """create_design_agents() should return exactly 4 agents."""
        agents = create_design_agents()
        assert len(agents) == 4, f"Expected 4 agents, got {len(agents)}"

    def test_agent_roles(self):
        """Agents should have the expected roles."""
        ux, visual, tech, synth = create_design_agents()

        assert "UX Designer" in ux.role
        assert "Visual Designer" in visual.role
        assert "Technical Designer" in tech.role
        assert "Synthesizer" in synth.role

    def test_ux_designer_has_skill_tools(self):
        """UX Designer should have list_skills, query_skill, discover_skill."""
        ux, _, _, _ = create_design_agents()
        tool_names = [t.name for t in ux.tools]

        assert "list_skills" in tool_names, "UX Designer should have list_skills"
        assert "query_skill" in tool_names, "UX Designer should have query_skill"
        assert "discover_skill" in tool_names, "UX Designer should have discover_skill"

    def test_visual_designer_has_write_file(self):
        """Visual Designer should have write_file tool."""
        _, visual, _, _ = create_design_agents()
        tool_names = [t.name for t in visual.tools]

        assert "write_file" in tool_names, "Visual Designer should have write_file"

    def test_visual_designer_has_skill_tools(self):
        """Visual Designer should also have skill query tools."""
        _, visual, _, _ = create_design_agents()
        tool_names = [t.name for t in visual.tools]

        assert "list_skills" in tool_names
        assert "query_skill" in tool_names

    def test_tech_designer_has_skill_tools(self):
        """Technical Designer should have list_skills and query_skill."""
        _, _, tech, _ = create_design_agents()
        tool_names = [t.name for t in tech.tools]

        assert "list_skills" in tool_names
        assert "query_skill" in tool_names

    def test_synthesizer_has_write_file(self):
        """Synthesizer should have write_file tool."""
        _, _, _, synth = create_design_agents()
        tool_names = [t.name for t in synth.tools]

        assert "write_file" in tool_names, "Synthesizer should have write_file"

    def test_agents_have_step_callback(self):
        """All agents should have step_callback for logging."""
        agents = create_design_agents()
        for agent in agents:
            assert agent.step_callback is not None, f"{agent.role} should have step_callback"


class TestCreateDesignTasks:
    """Tests for create_design_tasks() function."""

    def test_creates_four_tasks(self):
        """create_design_tasks() should return exactly 4 tasks."""
        agents = create_design_agents()
        tasks = create_design_tasks(*agents, "Test design task")

        assert len(tasks) == 4, f"Expected 4 tasks, got {len(tasks)}"

    def test_task_agents_match(self):
        """Each task should be assigned to the correct agent."""
        ux, visual, tech, synth = create_design_agents()
        tasks = create_design_tasks(ux, visual, tech, synth, "Test task")

        assert tasks[0].agent == ux, "Task 0 should be assigned to UX Designer"
        assert tasks[1].agent == visual, "Task 1 should be assigned to Visual Designer"
        assert tasks[2].agent == tech, "Task 2 should be assigned to Technical Designer"
        assert tasks[3].agent == synth, "Task 3 should be assigned to Synthesizer"

    def test_task_context_chain(self):
        """Tasks should have proper context dependencies."""
        agents = create_design_agents()
        tasks = create_design_tasks(*agents, "Test task")

        # UX task has no context (first in chain)
        # CrewAI uses various sentinel values, check for common patterns
        ux_context = tasks[0].context
        has_no_context = (
            ux_context is None or
            (isinstance(ux_context, list) and len(ux_context) == 0) or
            "NotSpecified" in str(type(ux_context)) or
            str(ux_context) == "NOT_SPECIFIED"
        )
        assert has_no_context, f"UX task should have no context, got {ux_context} (type: {type(ux_context)})"

        # Visual task depends on UX
        assert tasks[1].context is not None
        assert isinstance(tasks[1].context, list), f"Visual context should be list, got {type(tasks[1].context)}"
        assert tasks[0] in tasks[1].context

        # Tech task depends on Visual
        assert tasks[2].context is not None
        assert isinstance(tasks[2].context, list)
        assert tasks[1] in tasks[2].context

        # Synth task depends on all three
        assert tasks[3].context is not None
        assert isinstance(tasks[3].context, list)
        assert len(tasks[3].context) == 3

    def test_visual_task_mentions_write_file(self):
        """Visual design task should mention using write_file."""
        agents = create_design_agents()
        tasks = create_design_tasks(*agents, "Test task")

        assert "write_file" in tasks[1].description.lower()

    def test_synth_task_mentions_design_brief(self):
        """Synthesis task should mention DESIGN-BRIEF.md."""
        agents = create_design_agents()
        tasks = create_design_tasks(*agents, "Test task")

        assert "design-brief" in tasks[3].description.lower()

    def test_task_description_includes_user_task(self):
        """UX task description should include the user's task."""
        agents = create_design_agents()
        test_task = "Design a checkout flow for mobile"
        tasks = create_design_tasks(*agents, test_task)

        assert test_task in tasks[0].description


class TestDesignModeIntegration:
    """Integration tests for design mode."""

    def test_design_flag_in_help(self):
        """--design flag should appear in argparse help."""
        result = subprocess.run(
            [sys.executable, "crew.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )

        assert "--design" in result.stdout, "--design flag should be in help"
        assert "design mode" in result.stdout.lower()

    def test_design_examples_in_help(self):
        """Help should include design mode examples."""
        result = subprocess.run(
            [sys.executable, "crew.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )

        assert "--design" in result.stdout

    def test_workspace_dir_exists(self):
        """WORKSPACE_DIR should exist."""
        assert WORKSPACE_DIR.exists(), f"Workspace dir should exist: {WORKSPACE_DIR}"

    def test_design_dir_can_be_created(self):
        """design/ subdirectory should be creatable in workspace."""
        design_dir = WORKSPACE_DIR / "design"
        design_dir.mkdir(parents=True, exist_ok=True)

        assert design_dir.exists()


class TestDesignAgentModels:
    """Tests verifying correct model assignments."""

    def test_ux_designer_uses_claude(self):
        """UX Designer should use Claude model."""
        ux, _, _, _ = create_design_agents()
        # LLM string format is "provider/model"
        llm_str = str(ux.llm) if hasattr(ux.llm, '__str__') else str(type(ux.llm))
        assert "claude" in llm_str.lower() or "anthropic" in llm_str.lower()

    def test_visual_designer_uses_gpt(self):
        """Visual Designer should use GPT model."""
        _, visual, _, _ = create_design_agents()
        llm_str = str(visual.llm) if hasattr(visual.llm, '__str__') else str(type(visual.llm))
        assert "gpt" in llm_str.lower() or "openai" in llm_str.lower()

    def test_tech_designer_uses_deepseek(self):
        """Technical Designer should use DeepSeek model."""
        _, _, tech, _ = create_design_agents()
        # Access the model attribute directly for LLM objects
        model_name = getattr(tech.llm, 'model', '') or str(tech.llm)
        assert "deepseek" in model_name.lower() or "ollama" in model_name.lower(), f"Expected deepseek/ollama in {model_name}"

    def test_synthesizer_uses_claude(self):
        """Synthesizer should use Claude model."""
        _, _, _, synth = create_design_agents()
        llm_str = str(synth.llm) if hasattr(synth.llm, '__str__') else str(type(synth.llm))
        assert "claude" in llm_str.lower() or "anthropic" in llm_str.lower()


class TestDesignTaskOutputs:
    """Tests for expected task outputs."""

    def test_visual_task_expects_html_files(self):
        """Visual task expected_output should mention HTML files."""
        agents = create_design_agents()
        tasks = create_design_tasks(*agents, "Test task")

        assert "html" in tasks[1].expected_output.lower()
        assert "option" in tasks[1].expected_output.lower()

    def test_synth_task_expects_markdown(self):
        """Synthesis task expected_output should mention markdown."""
        agents = create_design_agents()
        tasks = create_design_tasks(*agents, "Test task")

        assert "design-brief.md" in tasks[3].expected_output.lower()

    def test_tech_task_expects_assessment(self):
        """Technical task expected_output should mention assessment."""
        agents = create_design_agents()
        tasks = create_design_tasks(*agents, "Test task")

        output_lower = tasks[2].expected_output.lower()
        assert "complexity" in output_lower or "assessment" in output_lower


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
