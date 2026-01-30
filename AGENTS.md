# baby-council Agent Reference

> **Do NOT read `crew.py` or explore `~/.council/crewai-council/`.** This file contains everything needed to invoke the council.

## Invocation

```bash
~/.council/crewai-council/venv/bin/python ~/.council/crewai-council/crew.py "<task>"
```

## Flags

| Flag | Purpose |
|------|---------|
| `--checkpoint` | Pause between stages (A=approve, R=reject) |
| `-y` / `--yes` | Skip provider confirmation |
| `--no-ollama` | Skip Ollama auto-start |

## Prerequisites

Before running, verify:
1. `ANTHROPIC_API_KEY` is set
2. `OPENAI_API_KEY` is set
3. `deepseek-coder-v2:16b` model installed (`ollama list | grep deepseek-coder-v2`)

**Note**: The council auto-starts Ollama if not running (use `--no-ollama` to disable).

## Starting Ollama (if not running)

```bash
# Start Ollama server (runs in background)
ollama serve &

# Pull DeepSeek model if not installed
ollama pull deepseek-coder-v2:16b

# Verify running
curl -s http://localhost:11434/api/tags | grep deepseek
```

## Live Monitoring

Watch deliberation in real-time:
```bash
tail -f ~/.council/crewai-council/workspace/deliberation.jsonl | jq -c
```

## Pipeline

| Stage | Agent | Model | Focus |
|-------|-------|-------|-------|
| Plan | Architect A | Claude Opus 4.5 | Clean architecture |
| Plan | Architect B | DeepSeek | Performance review |
| Build | Builder | GPT-5.2 | Implementation |
| Review | Reviewer A | Claude Opus 4.5 | Security & edge cases |
| Review | Reviewer B | DeepSeek | Performance & efficiency |

## Agent Capabilities

| Agent | Tools Available |
|-------|-----------------|
| Architects | `discover_skill` (GitHub search), `spawn_research`/`collect_research` (web research) |
| Builder | `write_file` (saves directly to workspace) |
| Reviewers | `query_skill` (postgres/react/web-design), `discover_skill`, `spawn_research`/`collect_research` |

**Note**: Builder writes files directly - no manual copy needed after completion.

## Output Locations

- **Console**: Full deliberation from all 5 agents
- **Files**: `~/.council/crewai-council/workspace/`
- **Log**: `~/.council/crewai-council/workspace/deliberation.jsonl`

## After Completion

1. Summarize what each agent contributed
2. Highlight key decisions from architects
3. Note issues found by reviewers
4. Point user to workspace for generated code

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| Missing API key | Env var not set | `export ANTHROPIC_API_KEY=...` |
| Ollama not running | Server stopped | `ollama serve &` (or let council auto-start) |
| Model not found | DeepSeek not pulled | `ollama pull deepseek-coder-v2:16b` |
| Provider check fails | API key invalid or <20 chars | Verify key is complete |
