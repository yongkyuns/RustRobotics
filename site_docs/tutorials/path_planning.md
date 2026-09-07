# Path Planning Tutorial

Path planning is not just "find a path." A planner optimizes over a particular representation of
free space, using a particular cost model, collision convention, and search budget. This chapter
compares Dijkstra, A*, Theta*, and RRT while making those assumptions visible.

```{raw} html
<div class="sim-embed-card">
  <iframe
    class="sim-embed-frame"
    data-sim-mode="path_planning"
    data-sim-path="?mode=path_planning&embed=focused&ui=20260907a"
    title="Rust Robotics path planning simulator"
    loading="lazy"
  ></iframe>
</div>
```

## Learning goals

By the end of this chapter, you should be able to:

- distinguish graph search from sampling-based planning
- explain what an admissible heuristic changes in A*
- separate shortest graph cost from geometric path quality
- identify collision and connectivity conventions that affect the answer
- compare planners using path cost and search effort rather than animation alone
- explain why RRT needs a seed/iteration budget to make comparisons reproducible

## Start with the representation

Dijkstra and A* operate on a grid graph. The current A* implementation is **8-connected**:

- cardinal steps cost `1`
- diagonal steps cost `sqrt(2)`
- Euclidean distance is used as the heuristic

The current grid convention allows a diagonal step whenever its destination cell is free; it does
not yet reject a diagonal merely because the two adjacent cardinal cells are blocked. In other
words, the grid planner currently permits corner cutting. That convention should be kept in mind
when judging path clearance or physical feasibility for a non-point robot.

Theta* keeps a graph-search structure but adds line-of-sight shortcuts. RRT instead samples
continuous free space and grows a tree.

```{admonition} The representation is part of the algorithm
:class: note-shell

A shortest path on an 8-connected point-robot grid is not automatically the shortest collision-free
trajectory for a robot with radius, dynamics, or turning limits. Before comparing planners, define
what counts as a legal move and what cost you are minimizing.
```

## Dijkstra: cost without goal guidance

Dijkstra expands states in increasing accumulated cost `g(n)`. With nonnegative edge costs it is
complete and optimal on the represented graph. It has no estimate of which direction is promising,
so it commonly explores much more of the map than necessary.

Its value in this tutorial is not just historical: Dijkstra gives a strong reference for checking
A* optimal cost on deterministic grid problems.

## A*: use an admissible heuristic to focus search

A* ranks candidates by

$$
f(n)=g(n)+h(n).
$$

With an admissible heuristic, A* should retain optimal graph cost while usually expanding fewer
states than Dijkstra. The current implementation reports `iterations` as **unique node expansions**;
stale duplicate heap entries are no longer counted as expansions.

That metric distinction matters. Priority queues often contain an older entry for a cell after a
better path to the same cell has been discovered. Counting queue pops as expansions exaggerates
actual search work and makes comparisons harder to interpret.

## Theta*: path shape depends on graph constraints

A grid can force a staircase-shaped path even when the search algorithm is optimal on that grid.
Theta* adds line-of-sight shortcuts so a path can connect across grid cells at arbitrary angles.

This is a useful reminder that "optimal search" and "good geometry" are different properties. A*
can be perfectly optimal for the graph you gave it while the graph remains a poor approximation of
the motion problem you care about.

## RRT: search by sampling

RRT samples continuous free space and incrementally extends a tree toward those samples. Instead of
systematically enumerating a fixed graph, it uses randomized exploration to discover feasible
connections.

The basic RRT family is valuable in spaces where a uniform grid is awkward or expensive, but a
single run is not a fair deterministic benchmark. Path quality and success depend on random samples,
iteration budget, step/connection rules, collision checking, and any post-processing.

## Experiment 1: A* should match Dijkstra's graph cost

**Question:** What does a good heuristic buy without changing the optimum?

1. Use one fixed grid environment, start, and goal.
2. Run Dijkstra and record the final path cost and visited/expanded cells.
3. Run A* on the same problem.
4. Compare path cost first, then search effort.

**Prediction:** both should produce the same optimal graph cost under the same movement convention,
while A* should usually expand fewer states because Euclidean distance points the search toward the
goal.

This comparison is also a numerical invariant suitable for CI, not just a visual demo.

## Experiment 2: shortest graph path versus path shape

**Question:** Why can an optimal grid path still look undesirable?

1. Use an environment with a long diagonal route and obstacles that do not force tight turns.
2. Compare A* and Theta*.
3. Look at path length, number of direction changes, and obstacle clearance.
4. Ask which differences come from search and which come from representation/line-of-sight rules.

**Prediction:** Theta* can remove staircase artifacts even when A* was already optimal on the
underlying grid.

## Experiment 3: collision convention matters

**Question:** Can a path be legal for a point grid cell but questionable for a physical robot?

1. Construct or select a narrow diagonal passage near obstacle corners.
2. Observe whether the grid planner accepts the diagonal transition.
3. Imagine inflating obstacles by a robot radius or forbidding diagonal corner cutting.
4. Re-evaluate whether the same route should remain valid.

**What this teaches:** collision semantics are part of the planning problem. A future
footprint-aware mode should make these rules configurable and test them explicitly.

## Experiment 4: RRT needs a statistical comparison

**Question:** How should a randomized planner be compared fairly?

1. Fix the environment, start/goal, and iteration budget.
2. Run RRT several times.
3. Record success rate, path length, and search-tree size.
4. Compare the distribution of outcomes instead of choosing the best-looking run.

A reproducible benchmark should additionally expose/fix the random seed. Until that is available in
the interactive surface, treat run-to-run variation as part of what the experiment is showing.

## Complexity and memory

For graph search with `V` vertices and `E` edges, priority-queue implementations of Dijkstra and A*
are commonly bounded around `O(E log V)`, with memory dominated by frontier, predecessor, score,
and closed/visited state. The heuristic does not change the worst-case form but can dramatically
change how much of the graph is explored on a particular problem.

Theta* pays additional line-of-sight checks. RRT cost is better discussed in terms of sample count,
nearest-neighbor work, and collision checks rather than the size of a pre-existing graph.

## What to measure

Prefer measurements that correspond to your real objective:

- success/failure
- path cost under the planner's cost model
- geometric path length
- minimum clearance when available
- number of unique expanded states
- tree/sample count for sampling-based planners
- runtime under a fixed map and budget
- variability across random seeds

## Common mistakes

- equating graph optimality with physical trajectory optimality
- comparing A* and Dijkstra without holding movement/cost conventions fixed
- calling heap pops "expanded nodes"
- ignoring corner-cutting and robot-footprint assumptions
- judging RRT from one lucky or unlucky run
- comparing planner speed without a fixed map, budget, and termination rule

## Where to go next

SLAM combines geometry with uncertainty and numerical optimization. The same discipline applies:
state the factors and objective, inspect the residuals, and distinguish sparse problem structure
from the actual linear algebra backend used by the implementation.
