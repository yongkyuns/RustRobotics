# Learning Paths

This page is for readers who want a structured route through the material rather
than a one-page quick start.

## If you are new to robotics

Use this order:

1. {doc}`overview`
2. {doc}`core_concepts`
3. {doc}`tutorials/pendulum`
4. {doc}`tutorials/localization`
5. {doc}`tutorials/path_planning`
6. {doc}`tutorials/slam`
7. {doc}`tutorials/robot`

Why this order:

- the pendulum tutorial builds control intuition
- localization introduces uncertainty and inference
- path planning introduces search and algorithm comparison
- SLAM combines uncertainty with mapping
- the robot tutorial connects the earlier ideas to a richer runtime

## If you are mainly interested in control

1. {doc}`tutorials/pendulum`
2. {doc}`tutorials/robot`

This path is best if you want to move from a simple unstable system to a more
realistic runtime loop.

## If you are mainly interested in estimation

1. {doc}`tutorials/localization`
2. {doc}`tutorials/slam`
3. {doc}`tutorials/robot`

This path is best if you want to focus on uncertainty, belief updates, and how
those ideas connect to larger systems.

## If you are mainly interested in planning

1. {doc}`tutorials/path_planning`
2. {doc}`tutorials/robot`

This path is best if you care more about search, heuristics, and practical
route generation.

## If you want the shortest complete tour

Read:

1. {doc}`quickstart`
2. {doc}`tutorials/pendulum`
3. {doc}`tutorials/localization`
4. {doc}`tutorials/path_planning`
5. {doc}`tutorials/robot`

That gives a compact view of control, estimation, planning, and runtime.

## If you are reading as an experienced engineer

A practical order is:

1. {doc}`overview`
2. {doc}`core_concepts`
3. whichever tutorial is closest to your current problem

Then, while reading each chapter, focus on:

- assumptions
- complexity and memory cost
- comparison tables
- practical use cases
- failure modes

The chapters are written so they can be used independently after the overview.
