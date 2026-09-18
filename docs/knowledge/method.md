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
