---
id: scaffold-base
title: "Project Scaffolding Instructions"
description: "Core instructions for scaffolding new projects, including CLI tooling, integration modules, and generation workflows."
category: scaffold
tags: [scaffold, instructions, setup, cli]
type: guideline
version: "1.0"
created: 2025-12-06
updated: 2025-12-06
related: [../guidelines/quick-start, ../agents/project-scaffold-agent]
---
## Table of Contents

- [Generation Instructions](#generation-instructions)
- [Post Generation Cleanup](#post-generation-cleanup)
- [Integration Modules](#integration-modules)
  - [NextJS ^16 App Router](#nextjs-16-app-router)


- [Generation Instructions](#generation-instructions)
- [Post Generation Cleanup](#post-generation-cleanup)
- [Integration Modules](#integration-modules)
  - [NextJS ^16 App Router](#nextjs-16-app-router)


# Project Scaffolding Instructions

These instructions must be followed modularly when requested to scaffold a NextJS 16
App Router + React 19 project. 

This file acts as a scaffold decision router that offers directions for modularized tooling related to integrating specific functionality into a new project. The integrations available for new projects are under "## Integration Modules"

The @./.agent/scaffold/cli-tooling.sh provides useful cli-tool installation instructions and scripts.

llm=agent

## Generation Instructions

1. Make plan @./.(llm)/scaffold/plan.md that integrates all project requirements using bash scripts (integration modules). There will be instructions in the bash scripts for parameters for customizing the generated project.
2. Execute plan:
    - If script fails - Review commands that failed in bash script. Run manually outside of bash script. If manual run fails give detailed report to user.
    - If script succeeds - Proceed to next objective in task.
3. After plan:
    - If plan fails - Review failing instruction and attempt recovery. If recovery fails, give user a detailed report of what failed.
    - If plan succeeds - Notify user, and proceed to ##Post Generation Cleanup

## Post Generation Cleanup

1. After all bash scripts are ran, initialize git repository with bash script.
    - Set default branch: `prod`
    - Set developement branch: `dev`
    - Ask user if they would like to integrate playwright tests into git-actions to protect `prod` or be ran with every action.
2. If host enabled, run netlify bash script.
    - If user initialized git repository, when creating netlify project with cli, link to github repository by default and specify the branch configuration:
        - Main production site only deploys on pushes to `prod`
        - Preview developement site only deploys on pushes to `dev`
    - If user did not initialize get repository, use cli to push code directly with no branch management. Write pnpm scripts to help with deploying development sites and production sites.


## Integration Modules

- {module}: @./.agent/scaffold/[module].sh
- tooling: @./.agent/scaffold/cli-tooling.sh

### NextJS ^16 App Router

Official Documentation: 