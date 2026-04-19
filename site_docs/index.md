# Rust Robotics

```{raw} html
<section class="hero-band">
  <div class="hero-brand">
    <div class="hero-copy">
      <div class="hero-kicker">Portable Algorithms + Interactive Simulation</div>
      <h1 class="hero-title">Rust Robotics</h1>
      <p class="hero-lead">
        Rust Robotics is a project for learning, comparing, and demonstrating robotics
        algorithms through reusable implementations and live simulations that run in
        native and web environments.
      </p>
    </div>
  <img class="hero-logo" src="_static/RustyRobot.png" alt="RustyRobot mascot" />
</div>
</section>
```

Rust Robotics has two practical goals:

- a library of robotics algorithm implementations
- an interactive simulation layer for learning, testing, and demonstration

This site is written to be readable by beginners while still being useful to
experienced engineers. The main focus is not internal project structure. The
focus is:

- what each algorithm is for
- what assumptions it makes
- where it is used
- how it compares to alternatives
- what it costs in compute and memory
- and how to interpret its behavior in the simulator

## Start here

If you are new to the project, start with:

1. {doc}`quickstart`
2. {doc}`getting_started`
3. {doc}`overview`
4. {doc}`core_concepts`
5. {doc}`tutorials/pendulum`

That order moves from smaller, easier-to-isolate algorithmic ideas toward a
larger runtime loop.

## How to use this site

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Quickstart
:class-card: landing-card

Read the shortest path through the project if you want to get useful value
quickly.

{doc}`quickstart`
:::

:::{grid-item-card} Learning paths
:class-card: landing-card

Use a structured reading path if you want to study by topic or by background.

{doc}`getting_started`
:::

:::{grid-item-card} Overview
:class-card: landing-card

Read the short project overview for the main purpose of the library and the
simulation layer.

{doc}`overview`
:::

:::{grid-item-card} Core concepts
:class-card: landing-card

Read the main ideas behind the project: portability, interactivity, and how the
documentation is intended to teach.

{doc}`core_concepts`
:::

:::{grid-item-card} Tutorials
:class-card: landing-card

The tutorial pages are the core of the site. Each chapter combines theory,
algorithm comparison, complexity discussion, and a live simulator.

{doc}`tutorials/index`
:::

:::{grid-item-card} Essays
:class-card: landing-card

The blog section is for broader project essays and higher-level narrative
writing that does not fit naturally into a single tutorial chapter.

{doc}`blog/index`
:::
::::

```{admonition} What makes this project different
:class: note-shell

The core educational promise of Rust Robotics is that the same ideas should be
visible in multiple settings:

- as reusable Rust implementations
- as native interactive applications
- as web-accessible simulations
- and, where appropriate, as portable logic that can move toward real hardware
```

## Main topics

The core subjects covered right now are:

- control systems
- localization
- path planning
- SLAM
- robot runtime execution

The tutorials are meant to feel closer to compact textbook chapters than to
feature walkthroughs.

## Reference and implementation details

This site is not trying to replace rustdoc. Rustdoc remains the right place for:

- module-level API reference
- source-oriented implementation details
- symbol navigation

This site is the explanatory layer on top of that.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Resources

quickstart
getting_started
overview
core_concepts
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Essays

blog/index
```
