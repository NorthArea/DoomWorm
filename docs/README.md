# Documentation map

Two tracks, one platform, one repository.

```text
docs/
  knowledge/     what either track learned that the other can use.
                 Read the relevant file before an experiment.
  doom/          the Doom player: its findings, stages, assumptions, results,
                 trained brains
  broom/         the home robot: the same four, plus the hardware protocol
  history/       plans that earlier section numbers refer to
```

The rule that keeps the two apart: a track's folder holds everything about
*its task*; `knowledge/` holds only what is true without naming a task, and
every entry there cites the run it came from. See `knowledge/README.md`.

Each track's `findings.md` is its own deliverable — one row per question, with
the worm's number, the best self-trained number and a verdict. Neither track
edits the other's.
