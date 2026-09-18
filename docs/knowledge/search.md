# Finding weights in a fixed topology

How to train 5905 synapses you are not allowed to rewire. Every number below
was measured on the Doom track at a budget of 1000 evaluations, 3 seeds, and
each is stated so the other track can reuse it.

## Cut the genome to the interface, not to the paths

Three scopes, same budget:

| scope | weights | result |
|---|---|---|
| all synapses | 5905 | the baseline |
| **the interface** | **1819** | **+2.2 and +6.9 on two tasks** |
| sensory outputs only | 506 | −10 and −2: too small |

The interface is every synapse that *leaves* a neuron a channel is injected
into or *enters* a motor group. It tunes how loudly the senses speak and how
loudly the muscles listen, and leaves the animal's interneurons alone.

**For the other track:** use `interface_synapses(connectome, "interface")` and
`WormBrain.trainable`. Expect a few points at no cost. Do not try to narrow by
reachability (see `connectome.md`).

## sep-CMA-ES beats fixed-sigma mutation, and makes a negative result mean something

The project's original search mutated every weight by a fixed sigma and kept
the best. On a 20-dimensional sphere that reaches −0.005 where separable
CMA-ES reaches −4e-6; on a 10^6-conditioned ellipsoid the fixed step does not
move at all. On the real task, same budget: +2.2 points on one level and +16 on
the harder one, and **three times tighter across seeds** (±1.1 against ±7.9).

`wormlab.learning.cmaes`, proven on benchmark functions before use
(`tests/test_cmaes.py`). Use `TrainConfig(search="cma")`.

**For the other track:** the reason to switch is not the points. Until the
optimiser is demonstrably stronger than the problem, "the connectome cannot do
this" only ever meant "our mutation did not find it".

## More budget buys very little

2.4x the training maps and 2.4x the generations moved reward by 6 points and
changed no behaviour at all (no aiming appeared). Measured on the Doom track,
2026-09-16.

**For the other track:** if a behaviour is absent, more generations is the
least promising thing to try. Look at what the reward pays for first.

## A stronger optimiser can find a *worse* behaviour

The reward paid steadily for exploring and rarely for a hit. Under the strong
search the brain emptied its magazine and collected exploration; under the weak
one it fired a third as often and killed three times as much.

**For the other track:** when a reward has a safe, dull, always-available term,
improving the search makes the policy duller. That is a property of the reward,
and it shows up *more* the better your optimiser is.

## Gains in the genome: not on an exponential scale

Putting the adapter's gains in the genome as `base * 8 ** gene` made the worst
row of its lane (−6.93 against −0.63 without). A small step changes the turn
rate several-fold and the search falls into a region it cannot climb out of. A
linear range is the thing to try; the idea is not dead, the parametrisation is.
