# Claude Code Architecture System
## Master Index & Navigation

**Languages:** TypeScript, Rust
**Purpose:** Modular instruction system for architecture, refactoring, and code quality

---

## Agent System

Agents are automatically invoked by Claude based on task context. Each agent has:
- **Isolated context window** — Separate from main thread
- **Scoped tools** — Only tools needed for the task
- **Reference docs** — Automatically consulted before responding

### Available Agents

| Agent | Invoke | Tools | Auto-Triggers |
|-------|--------|-------|---------------|
| `@engineer` | **DEFAULT MODE** | All | All code tasks — enforces critical thinking |
| `@architect` | Design tasks | Read, Glob, Grep, Bash | "design", "pattern", "structure" |
| `@refactor` | Code improvement | Read, Edit, Write, Bash | "refactor", "clean up", "simplify" |
| `@reviewer` | Quality checks | Read, Glob, Grep, Bash | "review", "check", "validate" |
| `@supabase` | Database & RLS | All + MCP | "supabase", "RLS", "migration", "policy" |
| `@linear` | Work breakdown | Read, Glob + MCP | "break down", "issues", "project plan", "linear" |
| `@git` | Version control | Read, Glob, Bash + MCP (git, github) | "commit", "branch", "rebase", "PR", "merge", "git" |

### Personality Mode

**`@engineer` should be active for ALL coding interactions.** It enforces:
- Critical analysis before implementation
- Alternative consideration (never first idea only)
- Failure mode identification
- Direct communication without enthusiasm theater
- SOLID/patterns/smells compliance

---

## Agents

### [@architect](agents/architect.md)
**Purpose:** Pattern selection, architecture decisions, system design
**Use when:**
- Designing new features or systems
- Choosing between design patterns
- Structuring modules and dependencies
- Making build-vs-buy decisions

### [@refactor](agents/refactor.md)
**Purpose:** Code smell detection, refactoring operations, code transformation
**Use when:**
- Improving existing code quality
- Detecting and fixing code smells
- Applying specific refactoring techniques
- Reducing technical debt

### [@reviewer](agents/reviewer.md)
**Purpose:** Code review, SOLID compliance, quality gates
**Use when:**
- Reviewing PRs or code changes
- Checking SOLID principle adherence
- Validating error handling patterns
- Ensuring type safety

---

## Reference Documentation

| Document | Contents | Used By |
|----------|----------|---------|
| [patterns.md](docs/patterns.md) | 23 GoF design patterns | Architect, Reviewer |
| [techniques.md](docs/techniques.md) | 60+ refactoring techniques | Refactor, Reviewer |
| [smells.md](docs/smells.md) | Code smell detection matrix | Refactor, Reviewer |
| [types.md](docs/types.md) | Type system (Result, branded, unions) | All agents |
| [errors.md](docs/errors.md) | Error handling architecture | All agents |
| [solid.md](docs/solid.md) | SOLID principles reference | All agents |

---

## Cross-Reference: Problem → Solution

```
PROBLEM TYPE                          → AGENT    → PRIMARY DOCS
─────────────────────────────────────────────────────────────────
"How should I structure this?"        → Architect → patterns.md
"Which pattern fits this problem?"    → Architect → patterns.md
"This code is messy, fix it"          → Refactor  → smells.md, techniques.md
"Is this code good quality?"          → Reviewer  → solid.md, smells.md
"How do I handle errors here?"        → Any       → errors.md, types.md
"What type should this be?"           → Any       → types.md
```

---

## Decision Flowchart

```
START: What do you need?

├─ [BUILDING NEW CODE]
│   └─ Use @architect
│       ├─ Read: docs/patterns.md
│       ├─ Read: docs/types.md
│       └─ Read: docs/errors.md
│
├─ [IMPROVING EXISTING CODE]
│   └─ Use @refactor
│       ├─ Read: docs/smells.md (detect problems)
│       ├─ Read: docs/techniques.md (apply fixes)
│       └─ Read: docs/solid.md (validate result)
│
└─ [EVALUATING CODE QUALITY]
    └─ Use @reviewer
        ├─ Read: docs/solid.md
        ├─ Read: docs/smells.md
        └─ Read: docs/errors.md
```

---

## File Structure

```
.claude/
├── INDEX.md                 ← You are here
├── agents/
│   ├── architect.md         ← Pattern selection, system design
│   ├── refactor.md          ← Smell detection, code transformation
│   └── reviewer.md          ← Code review, quality gates
└── docs/
    ├── patterns.md          ← Design patterns catalog
    ├── techniques.md        ← Refactoring techniques catalog
    ├── smells.md            ← Code smell detection guide
    ├── types.md             ← Type system patterns
    ├── errors.md            ← Error handling architecture
    └── solid.md             ← SOLID principles reference
```