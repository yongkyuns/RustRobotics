# Core Concepts

This page collects the main ideas that shape the project and the documentation.

## Portable implementations

One of the project’s core ideas is portability.

That does not only mean “code that can compile in more than one place.” It
means algorithm implementations that are not trapped inside one UI or one demo.

In practical terms, portability here means:

- algorithms can be used in native and web experiences
- the same core logic can be studied through interactive demos
- the learning surface does not require a separate implementation from the
  algorithmic surface

This matters for both accessibility and maintainability. A concept is easier to
teach when it can be demonstrated in more than one environment.

## Interactivity as a teaching tool

Interactivity is not a cosmetic feature of this project. It is a teaching tool.

For example:

- in control, it makes overshoot and settling visible
- in localization, it makes uncertainty visible
- in planning, it makes search effort and path quality visible
- in SLAM, it makes drift and correction visible

A static explanation can define an algorithm. A live simulator can expose the
failure modes and tradeoffs.

## Straightforward technical presentation

The project should remain technically straightforward. That means:

- avoiding unnecessary internal jargon in the documentation
- preferring clear problem statements over framework language
- discussing architecture only when it affects learning or portability
- keeping the focus on algorithms, tradeoffs, and applications

This is why the tutorials prioritize:

- theory
- comparisons
- complexity
- memory cost
- practical use cases

instead of deep implementation mechanics.

## Practical questions every tutorial should answer

Each chapter should make it easy for a reader to answer:

- what problem is being solved?
- what are the standard algorithms for this problem?
- where are they used?
- what are their strengths and limitations?
- how expensive are they in compute and memory?
- what should I observe in the simulator?

If a page cannot answer those questions, it is probably still too
implementation-centric.

## How complexity should be discussed

Complexity should not be treated as a decorative big-O line. In robotics,
runtime cost and memory cost often shape what is practical.

Good documentation should talk about:

- asymptotic cost
- practical online cost
- offline versus online work
- what scales with horizon, graph size, particle count, or state dimension
- when memory becomes a limiting factor

That is one of the main differences between a mathematically valid algorithm and
an algorithm that is genuinely usable.

## Why comparisons matter

Most tutorials become more useful when they compare algorithms rather than
explaining only one in isolation.

Examples:

- PID versus LQR versus MPC versus PPO
- particle filters versus Gaussian filters
- Dijkstra versus A* versus Theta* versus RRT
- EKF-SLAM versus graph-SLAM

Comparisons help readers understand not only what an algorithm is, but why one
method is chosen over another.
