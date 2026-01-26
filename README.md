# Council

A 5-agent AI deliberation system using CrewAI with three different LLMs working together.

## Agents

| Stage | Agent | Model | Focus |
|-------|-------|-------|-------|
| Planning | Architect A | Claude Opus 4.5 | Clean architecture design |
| Planning | Architect B | DeepSeek | Performance review |
| Building | Builder | GPT-5.2 | Implementation |
| Review | Reviewer A | Claude Opus 4.5 | Security & edge cases |
| Review | Reviewer B | DeepSeek | Performance & efficiency |

## Prerequisites

```bash
# API Keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Ollama with DeepSeek (local)
brew install ollama
ollama serve
ollama pull deepseek-coder-v2:16b
```

## Usage

```bash
# Activate virtual environment
source ~/.council/crewai-council/venv/bin/activate

# Run the council
python crew.py "Build a REST API for user authentication"
```

## Output

- Console shows full deliberation from all 5 agents
- Generated files saved to `workspace/`

## Directory Structure

```
~/.council/crewai-council/
├── crew.py          # Main implementation
├── config.yaml      # Configuration
├── venv/            # Python virtual environment
├── workspace/       # Generated output files
└── sessions/        # Session history (future)
```
