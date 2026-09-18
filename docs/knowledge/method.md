# How to run an experiment here without fooling yourself

Lessons about our own process. Cheaper to read than to rediscover.

## Never edit the source while a lane is running

An edit that was broken for ninety seconds — an import added after its first
use — crashed six benchmark subprocesses that happened to start inside that
window. Lanes are idempotent, so nothing was lost but time. Finish the edit,
run `make check`, then launch.

## Snapshot artefacts after the lane says it is done

`docs/brains/` was copied while a lane was still training, so several files
were mid-run checkpoints. The committed PPO brain scored −4.08 where its
published row said +3.30, and nobody noticed until a row was replayed a day
later. An evolution writes a checkpoint every generation, so a mid-run snapshot
is a *valid* brain — just not the one in the table.

Every published row must be replayable from what is committed. Check by
replaying one.

## Make the optimiser stronger than the problem before you conclude anything

Half the negative results in this project were ambiguous between "the
architecture cannot" and "our search did not". Fixing the search did not make
the wall move, and only then did the wall mean something. See `search.md`.

## Separate "cannot represent" from "could not find"

The way to settle it: let a search that is allowed to look ahead find good
actions, then try to **teach them back** to the network. If the network can be
taught what the search found, the architecture could always express it. This is
planned as the Doom track's stages 21-23 and applies unchanged to any track.

## Write the negative result down, with its numbers

Of the ideas tried on the Doom track in a single day — a place memory as a
layer, aim-damped turning, neuron noise, change detectors — none produced the
behaviour it was built for, and all four produced the *same* side effect: being
stuck got cheaper. That pattern was only visible because each failure was
recorded with its numbers instead of being deleted.

## Say what part of the model is ours

Real topology, contact-count magnitudes, literature signs for one transmitter
class, and our own search for every value. When a result is reported, the claim
is about the topology as a constraint — not about the animal. See
`connectome.md`.

## Clearing the scratch desk is a procedure, not a delete

`runs/` reached 191 MB of brains, rows and logs, and most of it backed a
published number. What made it safe to clear:

1. every lane's rows aggregated into a table that says how many seeds it came
   from (`scripts/archive_runs.py` -> `docs/<track>/results/<lane>/`), with the
   lane log beside it;
2. the brains copied to `docs/<track>/brains/`, including the losing arm of any
   comparison the choice was made on;
3. **every saved brain loaded by the code before anything was removed** — 145
   of 145. That check caught benchmark result files sitting in the brain
   folders, and three hybrid brains pointing at a reflex whose path had moved.

Do those three, in that order, and the desk is scratch again. Skip the third
and a row silently stops replaying, which has already happened here once.

## What a search scores must be what it executes

The first run of the proposal search said it was *worse* than the brain alone
on two levels of three: doom4 +0.90 against −3.28, doom2 +0.63 against −1.22.
It was not. Each branch was scored with the brain deciding afresh every tick,
and then only the branch's *first* intent was applied, held for five ticks. The
judge and the player were doing different things.

Replanning every tick, so that the scored branch and the executed one agree:
**+3.13 against +1.40** on the same six maps. The sign of the result came from
the mismatch, not from the method.

Cost of the correct version: about a hundred times an ordinary episode, 17
seconds against a fraction of one. Too slow to train with, fine for generating
a teacher.

**For any track:** whenever a controller evaluates a plan and then executes
something else — a held action, a smoothed action, a different horizon — the
evaluation is measuring a policy nobody runs. Check that first, before
believing a negative result about the search.

## A search over proposals is measured against a random proposer, or not at all

A rollout over the connectome's proposals beat the connectome alone: +3.13
against +1.40. The obvious reading — the wiring proposes well — is wrong. The
same rollout over **uniformly random intents** scored +3.45.

The look-ahead does the work; what generates the candidates barely matters. Any
"the network plus search is better" claim needs the random-proposer arm before
it means anything, and it is a two-minute run.

## Ask what chance would give before reading an agreement rate

The competence map reported that the brain's own first proposal was the one the
rollout kept in 42-51 % of decisions, against 17 % expected from six
exchangeable candidates. That looked like a strong signal and was an artefact:
the first proposal was chosen 41 % of the time and in **every one** of those its
score only *tied* the best. `max()` returns the first maximum, and a reward made
of coarse increments produces ties constantly.

The fix is not a different tie-break, it is the question: with a discrete
reward, how often are the candidates indistinguishable? Measure that first.

## A teacher you cannot imitate is not a teacher

Before building an expert-iteration loop, check that the expert's choice is a
**function of the state**. Ours was not: a constant fitted per situation scored
0.7144 against 0.7149 for one global constant — no situation-dependent
structure at all — while a brain trained towards it reached 0.845, worse than
the constant. The teacher was selecting lucky noise in hindsight, and hindsight
does not generalise to a policy.

Two cheap checks, in this order: does a per-situation constant beat a global
one, and does the expert beat a random proposer. If either says no, there is
nothing to distil and the loop will burn hours proving it slowly.

## Ask the same state twice

The cheapest test of whether anything is a function of the state costs one
extra evaluation per decision: answer the same state **twice**, from the same
internal snapshot, and compare.

    within    how far the two answers to one state lie apart -- the noise
    between   how far answers to different states lie apart -- the signal

Their ratio is the whole diagnosis, and it carries its own baseline: at 1.0 the
quantity says nothing about the state, whatever it looks like in a plot. It
needs no labels, no fitted model and no training run, and it applies to a
policy, a teacher, a score, a learned feature, anything with a state and an
output.

It is also how to tell a fix from a rescaling. Averaging the best k branches
looked like it was making our teacher deterministic — the spread of its
decisions fell from 0.708 to 0.163 as k grew. The ratio never moved: 1.03,
0.97, 0.94, 1.03. Both halves shrank together, because there was nothing under
the noise to uncover.

## A silent answer may be a badly chosen question

When the ratio says a controller is not responding to the state, ask *where it
was measured* before concluding it cannot respond. The same estimator along
three trajectories separated the two readings for us: steering itself, the
worm's answer varied five times more between states than within one; under the
search's steering, 1.3 times. Same brain, same estimator, same number of
states.

So a demonstration set is recorded **on the trajectory the student will
actually live on**. An expert that drives the system somewhere the student
never goes teaches it about a region where its own behaviour is undefined —
and the resulting failure looks exactly like "the architecture cannot express
this", which is the one conclusion it does not support.

## A control also tells you how big nothing is

The same-state-twice ratio needs a control for a second reason, beyond the one
above: it says what the ratio does when there is genuinely no structure. Ours
came back at 0.93 on one width and 1.04 on another, from a proposer drawing
uniform noise — so at that sample size anything in roughly [0.86, 1.12] is what
*nothing* looks like, and a worm scoring 0.9 is not a finding.

Run the control at every setting you run the measurement at, not once. Then the
bar is a measured band rather than the number 1, and a result has to leave it.

## Two implementations of one loop will not stay the same

A rollout must restore the brain's snapshot **before every branch**. Ours did,
in the library. Two later probes reimplemented the same loop and left it out,
so branch k started wherever branch k-1's rollout had driven the neurons: the
proposals stopped being draws from one state, and a measurement of what the
state implied measured the drift instead. The published verdict survived the
fix, but two of its numbers did not.

The fix is not "be careful in probes". It is that the loop lives in the library
with the invariant in its docstring and a test that asserts the brain is left
where it was found, and a probe that wants branches calls it.

## Beating a constant is not playing better

Our student fit its teacher better than a constant did — 0.1579 against 0.1639,
the first time in this project that happened — and then scored 0.29 where the
untaught brain scored 0.31. An imitation error is a distance in action space
averaged over every tick. A reward is collected by rare events. Nothing makes
a few percent of the first turn into any of the second, and the direction of a
small improvement is not even guaranteed.

So an imitation number is a diagnostic, never a result. The result is the
behaviour, measured the same way every other row is, on maps nothing trained
on. Report both: the fit says whether the learning worked, the benchmark says
whether it mattered, and they answer different questions.

## Check the baseline's own error bar before naming a collapse

Three maps said our taught student had collapsed: 1.97 untaught against 0.23
taught. Twenty maps said the untaught brain scores 0.31, and the collapse was
the baseline's sampling noise. A weak controller's reward is mostly variance,
so a drop measured against a small-sample baseline can be entirely the
baseline.

Before writing down a difference, ask what the *unchanged* arm scores on the
larger sample. It costs one more run and it has saved us from publishing a
dramatic number twice now.

## Measure the floor under your floor, and the ceiling over it

Two numbers turn a score into a result, and a project can run for weeks without
either. Ours did.

**The ceiling**: what the task pays if everything on it is collected, counted
off the environment rather than estimated — and then bounded by what the episode
can physically arrive at. Ours moves 0.2 units a tick, so 600 ticks buy 120
units of travel across a floor of 283 cells: most of the exploration reward was
on the map and out of reach in the time given. Without both numbers a reward is
a number with no scale, and "the first positive row" can be 3 % of what was
actually available.

Take the bound from the system's own constant, not from driving it and
measuring: our first two attempts at that measured how far it gets before
hitting a wall, which is a fact about the map.

**The floor under the floor**: what a policy with *no behaviour* scores —
uniform actions, no relation to the observations — benchmarked under exactly
the conditions the real rows use. Not a shuffled network, not an untrained one:
those are still structured objects, and ours turned out to score the same as
everything else. Twenty-four rows of careful comparison between trained brains
and trained controls all sat inside the band of flailing at random, and nothing
in any of those rows could have revealed it.

Measure both before the first comparison, not after the twentieth. They cost
one afternoon and they decide whether any later difference is worth reading.

And give the flailing floor enough episodes. A single lucky kill was worth more
here than any trained brain's entire episode, so a two-episode mean of a random
policy was one coin flip -- it read as 8.55 against the worm's 2.35 until the
sample grew, and then it read as −1.02.
