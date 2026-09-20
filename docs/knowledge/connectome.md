# The wiring itself

Facts about the *C. elegans* connectome as a computational substrate, measured
here, true whatever the animal is asked to do.

## The pharyngeal group is an island, and it cost us the whole shooting story

The pharyngeal nervous system — where the pumping neurons M3, M4 and MC live —
is joined to the rest of the connectome by **exactly five connections, all
through the pair RIPL/RIPR**. The trigger group's seventeen input neurons are
all pharyngeal. No sensory channel reaches it by any other path.

Measured on the Doom track, 2026-09-17. Consequence: a mapping that puts an
output on a pharyngeal neuron and its input anywhere else is not a hard task,
it is a **disconnected** one. Untrained, the real connectome answered 0.134
with a target on the gun line against 0.157 with none — pure noise — while its
degree-preserving shuffle, which has 399 crossing connections, answered in 77 %
of ticks. That difference was read as "shuffled wiring is better at aiming" for
a week. It was our mapping.

**For the other track:** before mapping any output onto a named group, check how
many connections cross into the region that group sits in. `RIP` is the only
door into the pharynx.

## Two hops from the sensors, almost everything is reachable

Counting the synapses that a sensory current can influence *and* that can
influence a motor group: 57 % within one hop each way, **99.3 % within two**,
100 % within three. Measured on the Doom track's sensory and motor tables,
2026-09-17.

**For the other track:** you cannot narrow a search by "only the synapses on
the path from sense to muscle" — that is nearly all of them. The cut that does
narrow it is structural, not topological (see `search.md`).

## What the connectome does and does not give us

The data (Cook et al. 2019) gives: which neuron connects to which, whether the
connection is chemical or a gap junction, and **how many synaptic contacts** the
pair has — 1 to 401, mean 5.5. That count is the only quantity, and we use it as
the initial weight, normalised per target neuron.

It does not give: the sign of a connection, its strength in physical units, gap
junction conductances, intrinsic neuron properties, or any of the neuropeptide
signalling that travels outside synapses at all.

We set signs from the literature for the GABAergic set (McIntire et al. 1993;
Gendrel, Atlas & Hobert 2016) and learn the rest. So the honest description of
what either track studies is: **real topology, contact-count magnitudes,
literature signs for part of the network, and our own search for the values.**
Every conclusion is about the topology as a constraint, not about the animal.

**For the other track:** the cheapest step towards a more real animal is more
signs, not more training — neurotransmitter atlases and the CeNGEN expression
data would settle the sign of far more connections than our GABA list does.

## What this wiring is measurably good at, and what it is not

Grouping 3600 searched decisions by situation (Doom track, 2026-09-18): the
connectome's own first proposal is the best of six in 42-51 % of them, where
chance is 17 %. The proposal distribution is well centred — this is not a
network guessing.

The spread across situations is the useful part. It is **most** right where its
own reflex lives (51 % with a wall ahead, and a search over its proposals adds
almost nothing there, +0.06). It is **least** right wherever the task is to
orient onto a thing: a search adds twice as much (+0.13) in any situation
involving a target to engage.

**For the other track:** expect the connectome to carry avoidance and gradient
following for free, and expect to pay for anything that requires facing a
chosen object. Budget the effort accordingly, and measure by situation — one
reward number hides this completely.

## What the wiring gives is a bias, not a capacity

The clearest case we have. The hand-written floor's most expensive reflex is
coming round onto a target off the bow, and the question was whether the
connectome can express it at all.

Measured with no world in the way -- one side channel held steady, the intent
read back -- the answer is yes: the right sign, a turn that holds while the
signal does, and one that settles when the target centres (0.030). All three
training seeds agree on the sign, and neither the shuffled connectome nor a
random topology of the same size does.

Then train *directly* for that reflex, nothing else scored: the connectome
reaches 0.735, its shuffle 0.499 -- and a random topology 0.979, the best of the
three. Every topology can represent it. The real one is not special at
representing it.

So the two measurements say different things and both are needed. Capacity is
not where the anatomy shows up; **what it does is decide what you find when the
reward never asks for it directly.** Trained on a game that pays only for kills
and exits, the real wiring lands the correct sign anyway and the controls do
not. That is an inductive bias, and it is the kind of claim a shuffled control
can actually establish.

Read it with the sample in mind: three seeds agreeing is one chance in eight.

## And a reflex that exists can still be useless

The same circuit, swept over signal strength, turns at 0.007 when the sector
channel reads 0.05 -- a target across the room -- against the 0.6 the
hand-written floor uses. At the gun's range it manages 0.055. The reflex is
present and its gain is out by an order of magnitude exactly where the task
needs it.

A brain trained harder (stage 18's CMA run) does have the gain, 0.49 at 0.05,
and pays for it by never stopping: it spins past. Presence, gain and
termination are three separate properties, and a circuit that has one of them
looks from the outside exactly like a circuit that has none.
