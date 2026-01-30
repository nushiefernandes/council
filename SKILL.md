---
name: baby-council
description: Multi-model deliberation council using Claude Opus 4.5, GPT-5.2, and DeepSeek. Invokes CrewAI for planning, building, and dual-perspective code review.
---

# /baby-council

> **Implementation details are in `AGENTS.md`.** Do NOT read `crew.py` or explore `~/.council/crewai-council/` - everything you need is in this skill folder.

A multi-model AI council that uses 3 different LLMs working together:
- **Claude Opus 4.5** (Anthropic) - Architecture & Security Review
- **GPT-5.2** (OpenAI) - Implementation
- **DeepSeek Coder** (Ollama/Local) - Performance Review

## When to Use

Use this skill when the user wants multiple AI models to collaborate on a task:
- `/baby-council Build a REST API for user auth`
- `/baby-council Create a CLI tool that does X`
- "Run the council on this task"
- "Get multiple AI perspectives on building X"

## What It Does

The council runs a 5-stage pipeline:

| Stage | Agent | Model | Focus |
|-------|-------|-------|-------|
| 1 | Architect A | Claude Opus 4.5 | Clean architecture design |
| 2 | Architect B | DeepSeek | Performance review of design |
| 3 | Builder | GPT-5.2 | Implementation |
| 4 | Reviewer A | Claude Opus 4.5 | Security & edge cases |
| 5 | Reviewer B | DeepSeek | Performance & efficiency |

## Prerequisites

Before running, verify:
1. `ANTHROPIC_API_KEY` is set
2. `OPENAI_API_KEY` is set
3. Ollama is running (`curl http://localhost:11434/api/tags`)
4. DeepSeek model is pulled (`ollama pull deepseek-coder-v2:16b`)

## How to Invoke

Run the CrewAI council:

```bash
~/.council/crewai-council/venv/bin/python ~/.council/crewai-council/crew.py "<task description>"
```

### With Checkpoints (Human-in-the-Loop)

Add `--checkpoint` to pause for approval between stages:

```bash
~/.council/crewai-council/venv/bin/python ~/.council/crewai-council/crew.py --checkpoint "<task description>"
```

When checkpoints are enabled, the council pauses 3 times:
1. **After Planning** - Review architecture from both architects before building
2. **After Building** - Review generated files before code review
3. **After Review** - See final review results before completion

At each checkpoint, you can:
- Press **A** (or Enter) to approve and continue
- Press **R** to reject and abort the session

## Output

- **Console**: Full deliberation output from all 5 agents
- **Workspace**: Generated files saved to `~/.council/crewai-council/workspace/`

## Presenting Results

After the council completes:
1. Summarize what each agent contributed
2. Highlight key decisions from architects
3. Note any issues found by reviewers
4. Point user to workspace for generated code

## Example Session

User: `/baby-council Build a Python function that validates email addresses`

Claude should:
1. Confirm prerequisites are met
2. Run the command above
3. Wait for completion (can take 2-5 minutes)
4. Summarize: "The council completed. Claude designed a regex-based approach, DeepSeek suggested performance optimizations, GPT-5.2 implemented it, and both reviewers found minor issues..."
5. Show workspace path for generated files

## Future Features (Not Yet Implemented)

- `--output json` - Structured JSON output
- `--resume <session>` - Resume a previous session
- `--models` - Override default model selection
