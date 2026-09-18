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
