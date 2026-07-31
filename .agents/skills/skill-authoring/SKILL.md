---
name: skill-authoring
description: Write or revise agent skills (SKILL.md packages) under .agents/skills/ or ~/.agents/skills/. Use when creating a new skill, renaming or restructuring one, improving a skill's description so it triggers correctly, or checking a skill against the Agent Skills standard and pi's discovery/validation rules.
---

# Author a great skill

A skill is a directory with a `SKILL.md` (frontmatter + instructions) plus
optional `scripts/`, `references/`, and `assets/`. Skills load from
`.agents/skills/` (project, shared via git) and `~/.agents/skills/` (personal).
Root-level `.md` files directly inside `.agents/skills/` are **ignored** —
every skill must be its own directory containing `SKILL.md`.

## How the agent actually consumes a skill

1. At startup only the **name + description** enter the system prompt.
2. When a task matches, the agent `read`s the full SKILL.md (or the user
   forces it with `/skill:<name>`).
3. The agent follows the instructions; relative paths resolve against the
   skill directory.

This progressive disclosure drives every rule below.

## Frontmatter

```yaml
---
name: my-skill        # <=64 chars, lowercase a-z 0-9 and single hyphens only;
                      # no leading/trailing hyphens. Required.
description: ...      # <=1024 chars. Required — WITHOUT it the skill never loads.
---
```

## The description is the trigger

It is the ONLY part always in context, so it decides whether the skill ever
fires. Write it as **what it does + when to use it**, including the phrases a
user would actually say.

- Bad: `Helps with deployments.` (fires nowhere, or everywhere)
- Good: `Add or modify an OpenTela deployment recipe under
  deployments/<service-kind>/<site>/. Use when creating a new recipe, porting
  one to a new site, or serving a different model.`

Keep it under ~4 sentences; trim until every clause changes when it triggers.

## Body principles

1. **Instructions, not essays.** Imperative steps, copy-runnable commands,
   decision tables. Delete anything the agent already knows generically; only
   non-obvious, repo-specific, hard-won knowledge earns its tokens.
2. **Lean SKILL.md, deep references.** Keep SKILL.md roughly under 200 lines.
   Move long detail to `references/` linked one level deep
   (`See [X](references/x.md)`); the agent loads it only when needed.
3. **Scripts over prose for mechanics.** If a block is "run these exact
   commands," ship it in `scripts/` and show the invocation — don't paste a
   wall of shell that must be transcribed.
4. **Say what breaks.** For every non-obvious rule, name the failure it
   prevents — agents adapt rules correctly only when they know the reason
   (same convention as this repo's recipes).
5. **Fail loudly with the fix.** If the skill needs a tool/env var, the first
   section checks it and prints the exact install/export command.
6. **No invented facts.** Commands, flags, and versions must exist; if unsure,
   verify with `--help`/docs before writing them into a skill.
7. **Generalize on purpose.** Defaults + env overrides, not hardcoded values,
   unless the hardcoded value IS the documented site default.

## House style (this repo)

- Skills mirror recipe conventions: copy-and-run, comments saying what breaks
  without a setting, exact fix commands instead of obscure failures.
- Reference real repo files as canonical examples rather than re-deriving.
- Commit skills with the recipes they support — project skills are shared.

## Validate before committing

```bash
# name: lowercase-hyphenated, <=64 chars; description present and <=1024
for f in .agents/skills/*/SKILL.md; do
  awk '/^---$/{c++; next} c==1 && /^name:/{print FILENAME": "$0}' "$f"
done

# force-load and exercise it
#   /skill:<name>            in an interactive pi session
```

Name collisions across locations warn and keep the first found — check
`~/.agents/skills/` for same-named skills before adding a project one.

## Anti-patterns to reject

- Vague or missing description (missing = silently unloaded).
- SKILL.md that duplicates what the model already does by default.
- Duplicating content that belongs in the repo (recipes, READMEs) — link to
  the canonical file instead, so the skill can't drift stale.
- Multi-hop reference chains (SKILL.md -> a.md -> b.md); keep it one level.
- Skills that only restate a CLI's `--help`.
