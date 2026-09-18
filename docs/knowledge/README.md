# Knowledge: what one track learned that the other can use

Two tracks share this repository and one platform. They do not share a task —
one plays Doom, the other drives a floor robot — so most of what each learns
stays in its own folder (`docs/doom/`, `docs/broom/`). This directory is for the
rest: things we found out that are **true regardless of the task**.

## What belongs here

A finding is promoted here when both of these hold:

1. it was **measured**, and the entry names the track, the run and the numbers
   it came from — never an opinion, never a plan;
2. it can be **stated without naming the task**. "Restricting the genome to the
   interface synapses beats tuning all 5905 at the same budget" is knowledge.
   "The worm cannot reach the exit on doom2" is a track result.

If a claim fails the second test, it stays in the track. If it fails the first,
it is not written down at all.

## What is here

| File | What it holds |
|---|---|
| `connectome.md` | facts about the wiring itself: what the topology allows and forbids |
| `search.md` | how to find weights in a fixed topology: what worked, at what budget |
| `sensing.md` | how to put a world in front of 302 neurons: where a signal must land |
| `method.md` | how to run an experiment here without fooling yourself |

## How to use it

Before an experiment, read the file that covers what you are about to change.
Several of today's entries exist because one track spent hours discovering
something the other would otherwise repeat.

Both directions count. A robot finding can settle a Doom question and the other
way round: the tracks differ in task, not in nervous system.

## How to add to it

When a measurement on your track produces something task-independent, add a row
in the right file with:

- the claim, in one sentence, with the numbers;
- **where it came from**: track, run directory, date;
- what it means for someone who does not work on your track.

Keep the track's own story in the track's own `findings.md`; put here only the
part the sibling could act on.
