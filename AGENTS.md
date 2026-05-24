# TimesFM — Agent Entry Point

This repository ships a first-party **Agent Skill** for TimesFM at:

```
timesfm-forecasting/
└── SKILL.md    ← read this for the full skill
```

## Install the skill

Copy the skill directory into your agent's skills folder:

```bash
# Cursor / Claude Code / OpenCode / Codex (global install)
cp -r timesfm-forecasting/ ~/.cursor/skills/
cp -r timesfm-forecasting/ ~/.claude/skills/

# Or project-level
cp -r timesfm-forecasting/ .cursor/skills/
```

Any agent that supports the open [Agent Skills standard](https://agentskills.io) will discover it automatically.

## Working in this repo

If you are developing TimesFM itself (not using it), the source lives in `src/timesfm/`.
Archived v1/v2 code and notebooks are in `v1/`.

Run tests:

```bash
pytest v1/tests/
```

See `README.md` for full developer setup.

## Personal notes

- I'm using this fork primarily to experiment with fine-tuning on custom time-series datasets.
- The `v1/` notebooks are handy for quick sanity checks before running full evaluations.
- Note: `pytest v1/tests/` requires the optional dev dependencies (`pip install -e '.[dev]'`); easy to miss if setting up a fresh venv.
