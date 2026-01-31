from __future__ import annotations

"""Design pipeline runner.

Incorporates:
- Clean architecture separation (context store, stages, output writer).
- Performance enhancements:
  - LLM config caching
  - Agent pooling per stage/provider to reduce repeated construction overhead
  - String template building avoids repeated formatting work
- Robust error handling.

Note: This runner expects `crewai` to be installed at runtime.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .design_context import DesignContext, DesignContextStore
from .design_stages import DESIGN_STAGES, DesignStage
from .errors import StageDependencyError, StageExecutionError
from .llm_config import get_llm_for_provider
from .output_writer import OutputPaths, write_synthesis_outputs

# Import skill tools from main crew module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from crew import list_skills_tool, query_skill, discover_skill_tool, write_file


@dataclass(frozen=True, slots=True)
class RunResult:
    stage_results: dict[str, str]
    output_paths: Optional[OutputPaths] = None


class DesignPipelineRunner:
    """Runs the design pipeline with checkpoint and resume support."""

    def __init__(
        self,
        workspace_dir: Path,
        checkpoint_mode: bool = True,
        prompt_fn: Callable[[str], str] | None = None,
        verbose: bool = True,
    ):
        self.workspace_dir = workspace_dir
        self.checkpoint_mode = checkpoint_mode
        self.context_store = DesignContextStore(workspace_dir)
        self.stages: list[DesignStage] = DESIGN_STAGES
        self._prompt_fn = prompt_fn or input
        self._verbose = verbose

        # Performance: cache constructed agents per stage (or provider+role).
        self._agent_cache: dict[str, object] = {}

    def _build_task_description(self, stage: DesignStage, context: DesignContext) -> str:
        """Inject previous stage results into task description."""
        replacements: dict[str, str] = {"task": context.task_description}
        for stage_name, result in context.stage_results.items():
            replacements[f"previous_{stage_name}"] = result
        try:
            return stage.task_description_template.format(**replacements)
        except KeyError as exc:
            missing = str(exc)
            raise StageExecutionError(
                f"Stage '{stage.name}' template missing placeholder for {missing}. "
                f"Available: {sorted(replacements.keys())}"
            ) from exc

    def _get_tools_for_stage(self, stage_name: str) -> list:
        """Get appropriate tools for each design stage."""
        if stage_name == "ux_design":
            # UX Designer needs skills for guidelines + research
            return [list_skills_tool, query_skill, discover_skill_tool]
        elif stage_name == "visual_design":
            # Visual Designer needs skills + file writing for mockups
            return [list_skills_tool, query_skill, write_file]
        elif stage_name == "technical_design":
            # Technical Designer needs skills for best practices
            return [list_skills_tool, query_skill]
        elif stage_name == "synthesis":
            # Synthesizer needs file writing for final outputs
            return [list_skills_tool, query_skill, write_file]
        return []

    def _create_agent_for_stage(self, stage: DesignStage):
        """Create or reuse a CrewAI agent for the given stage."""
        cache_key = f"{stage.name}:{stage.llm_provider}"
        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        try:
            from crewai import Agent  # type: ignore

            tools = self._get_tools_for_stage(stage.name)

            agent = Agent(
                role=stage.agent_role,
                goal=stage.agent_goal,
                backstory=stage.agent_backstory,
                llm=get_llm_for_provider(stage.llm_provider),
                tools=tools,
                verbose=self._verbose,
                allow_delegation=False,
            )
            self._agent_cache[cache_key] = agent
            return agent
        except Exception as exc:
            raise StageExecutionError(f"Failed to create agent for stage '{stage.name}': {exc}") from exc

    def _create_task_for_stage(self, stage: DesignStage, agent, context: DesignContext):
        """Create a CrewAI task with injected context."""
        try:
            from crewai import Task  # type: ignore

            description = self._build_task_description(stage, context)
            return Task(
                description=description,
                expected_output=stage.expected_output,
                agent=agent,
            )
        except Exception as exc:
            raise StageExecutionError(f"Failed to create task for stage '{stage.name}': {exc}") from exc

    def _run_single_stage(self, stage: DesignStage, context: DesignContext) -> str:
        """Execute a single stage and return its result."""
        try:
            from crewai import Crew  # type: ignore

            agent = self._create_agent_for_stage(stage)
            task = self._create_task_for_stage(stage, agent, context)
            crew = Crew(agents=[agent], tasks=[task], verbose=self._verbose)
            result = crew.kickoff()
            return str(result)
        except StageExecutionError:
            raise
        except Exception as exc:
            raise StageExecutionError(f"Stage '{stage.name}' failed during execution: {exc}") from exc

    def _prompt_for_approval(self, stage_name: str, result: str) -> bool:
        """In checkpoint mode, ask user to approve stage result."""
        if not self.checkpoint_mode:
            return True

        print(f"\n{'=' * 60}")
        print(f"STAGE COMPLETE: {stage_name}")
        print(f"{'=' * 60}")
        preview = result[:2000] + ("..." if len(result) > 2000 else "")
        print(preview)
        print(f"{'=' * 60}")

        try:
            response = self._prompt_fn("\nApprove and continue? [Y/n]: ").strip().lower()
        except EOFError:
            # Non-interactive environment: default to approve.
            return True
        return response in ("", "y", "yes")

    def run(self, task_description: str, resume: bool = False) -> RunResult:
        """Run the full design pipeline.

        Args:
            task_description: The design task to accomplish.
            resume: If True, resume from last checkpoint.

        Returns:
            RunResult with stage results and output paths if synthesis completed.
        """
        # Initialize or resume context
        if resume:
            loaded = self.context_store.load()
            if loaded is None:
                context = DesignContext(task_description=task_description)
            else:
                # If user passed a new task description, keep persisted one.
                context = loaded
        else:
            self.context_store.clear()
            context = DesignContext(task_description=task_description)

        self.context_store.save(context)

        # Run remaining stages
        for stage in self.stages:
            if stage.name in context.stage_results:
                if self._verbose:
                    print(f"Skipping completed stage: {stage.name}")
                continue

            missing_deps = [dep for dep in stage.depends_on if dep not in context.stage_results]
            if missing_deps:
                raise StageDependencyError(
                    f"Stage '{stage.name}' missing dependencies: {missing_deps}"
                )

            if self._verbose:
                print(f"\n>>> Running stage: {stage.name}")

            result = self._run_single_stage(stage, context)

            if not self._prompt_for_approval(stage.name, result):
                if self._verbose:
                    print("Stage not approved. Stopping pipeline.")
                return RunResult(stage_results=context.stage_results)

            context = context.with_stage_result(stage.name, result)
            self.context_store.save(context)

        output_paths: Optional[OutputPaths] = None
        synthesis = context.stage_results.get("synthesis")
        if synthesis:
            output_paths = write_synthesis_outputs(self.workspace_dir, synthesis)

        return RunResult(stage_results=context.stage_results, output_paths=output_paths)
