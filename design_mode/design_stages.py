from __future__ import annotations

"""Stage configuration for the design pipeline."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class DesignStage:
    """Configuration for a single design pipeline stage."""

    name: str
    agent_role: str
    agent_goal: str
    agent_backstory: str
    task_description_template: str
    expected_output: str
    llm_provider: str = "claude"  # claude, gpt, deepseek
    depends_on: list[str] = field(default_factory=list)


DESIGN_STAGES: list[DesignStage] = [
    DesignStage(
        name="ux_design",
        agent_role="UX Designer (Claude)",
        agent_goal="Design intuitive user flows, information architecture, and interaction patterns",
        agent_backstory="""You are a senior UX designer focused on how interfaces FEEL to use.
You think about user journeys, cognitive load, thumb reachability on mobile,
and emotional resonance. You avoid complexity unless it serves the user.

Start by running list_skills to see available expertise, then query relevant
skills like web-design-guidelines for accessibility guidance.""",
        task_description_template=(
            """
Analyze the design requirements and create UX specifications:

TASK: {task}

First, run list_skills to see available expertise. Query relevant skills for guidance.

Provide:
1. User goals for this screen/flow
2. Information hierarchy (what's most important?)
3. Interaction patterns (tap, swipe, scroll behaviors)
4. Mobile-first layout recommendations (thumb zones, reachability)
5. Accessibility considerations

Focus on HOW IT FEELS TO USE, not how it looks.
""".strip()
        ),
        expected_output="""UX specification with:
- User goals and success metrics
- Information architecture
- Interaction flow description
- Mobile layout guidance
- Accessibility notes""",
        llm_provider="claude",
        depends_on=[],
    ),
    DesignStage(
        name="visual_design",
        agent_role="Visual Designer (GPT)",
        agent_goal="Create distinctive, memorable visual designs that avoid generic AI aesthetics",
        agent_backstory="""You are a bold visual designer who hates cookie-cutter interfaces.
You think about typography (never use Inter/Arial), color palettes with personality,
spatial composition, and motion/micro-interactions.

Run list_skills first, then query_skill("frontend-design", "aesthetics") for guidance.
Use write_file to save mockups as viewable HTML files.""",
        task_description_template=(
            """
Based on the UX specification, create 2-3 DISTINCT visual design directions.

ORIGINAL TASK: {task}

UX ANALYSIS:
{previous_ux_design}

First, run list_skills and query_skill for frontend-design guidelines.

For EACH direction:
1. Choose a unique aesthetic (e.g., warm minimalism, bold typography, soft organic)
2. Define typography (distinctive fonts, NOT Inter/Arial/system fonts)
3. Define color palette (with CSS variables)
4. Create a viewable HTML mockup with embedded CSS

REQUIREMENTS:
- Each direction must be visually DIFFERENT (not variations of same theme)
- Use write_file to save: design/option-a.html, design/option-b.html, etc.
- Mobile-first (375px width, use max-width for desktop)
- Include hover states and micro-interactions where appropriate
""".strip()
        ),
        expected_output="""2-3 HTML mockup files saved to workspace/design/:
- design/option-a.html
- design/option-b.html
- design/option-c.html (optional)

Each viewable in browser with complete styling.""",
        llm_provider="gpt",
        depends_on=["ux_design"],
    ),
    DesignStage(
        name="technical_design",
        agent_role="Technical Designer (DeepSeek)",
        agent_goal="Review designs for technical feasibility, mobile performance, and implementation complexity",
        agent_backstory="""You review design proposals with an engineer's eye.
You flag animations that cause jank, layouts that break on small screens,
and complexity that slows development. Suggest simpler alternatives when needed.

Use list_skills and query_skill for react-best-practices guidance.""",
        task_description_template=(
            """
Review the visual designs for technical feasibility.

ORIGINAL TASK: {task}

UX ANALYSIS:
{previous_ux_design}

VISUAL DESIGN:
{previous_visual_design}

Run list_skills and query react-best-practices for guidance.

For each design direction, assess:
1. Implementation complexity (1-10 scale)
2. Mobile performance concerns (animations, repaints, large images)
3. Accessibility issues (contrast, focus states, screen reader)
4. React component structure recommendations
5. CSS concerns (browser support, layout stability)

Flag issues but don't kill creativity - suggest alternatives that preserve the design intent.
""".strip()
        ),
        expected_output="""Technical assessment for each design:
- Complexity rating with justification
- Performance flags and fixes
- Accessibility audit results
- Recommended component breakdown
- CSS/implementation notes""",
        llm_provider="deepseek",
        depends_on=["ux_design", "visual_design"],
    ),
    DesignStage(
        name="synthesis",
        agent_role="Design Synthesizer (Claude)",
        agent_goal="Combine design perspectives into clear options for human decision-making",
        agent_backstory="""You synthesize input from UX, visual, and technical designers
into 2-3 distinct design directions. Present tradeoffs clearly without
making the decision for the human. Save output as DESIGN-BRIEF.md.""",
        task_description_template=(
            """
Synthesize all input into a design brief for human decision-making.

ORIGINAL TASK: {task}

UX DESIGN:
{previous_ux_design}

VISUAL DESIGN:
{previous_visual_design}

TECHNICAL DESIGN:
{previous_technical_design}

Create a summary document using write_file("design/DESIGN-BRIEF.md", content) that:
1. Names each direction clearly (e.g., "Warm Minimalism", "Bold Editorial")
2. Shows a side-by-side comparison table
3. Lists pros/cons from UX, visual, and technical perspectives
4. Notes which direction best fits the stated product vision
5. Does NOT make the decision - presents options for human choice

Also include links to the mockup files (design/option-a.html, etc.)
""".strip()
        ),
        expected_output="""design/DESIGN-BRIEF.md containing:
- Direction summaries with names
- Comparison matrix
- Tradeoffs from each perspective
- Recommendation (without deciding)
- Links to mockup files""",
        llm_provider="claude",
        depends_on=["ux_design", "visual_design", "technical_design"],
    ),
]
