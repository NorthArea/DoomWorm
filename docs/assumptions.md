# Assumptions & decisions

Plan rule 7: document assumptions. Add an entry per stage as decisions are made.

| Date | Stage | Assumption / decision | Why |
|------|-------|-----------------------|-----|
| 2026-09-14 | scaffold | Neuron model v1 = Leaky Integrate-and-Fire (no Hodgkin-Huxley). | Plan §2.3. |
| 2026-09-14 | scaffold | Connectome topology is fixed; only synaptic weights are trainable. | Plan §16. |
| 2026-09-14 | scaffold | No large ML model between environment and brain; adapters are simple transforms. | Plan §2.1. |
| 2026-09-14 | scaffold | CPU-only, NumPy. No GPU/Rust until the hypothesis works. | Plan §42. |
| 2026-09-14 | scaffold | ViZDoom/Gymnasium are not dependencies yet; added at stage 18. | Plan §44. |
| 2026-09-14 | 0 | Synchronous update: a step reads activities from the previous tick, so a signal travels one synapse per tick and neuron order is irrelevant. | Determinism; matches Plan §40 replay requirement. |
| 2026-09-14 | 0 | Activity is binary (0/1) and potential resets to 0 on firing. Leak is applied before input: `p = p*(1-decay) + I`. | Simplest model that satisfies Plan §6 acceptance criteria. |
| 2026-09-14 | 0 | Pure Python dicts, no NumPy in the brain yet. | 3 neurons; vectorise only when the 302-neuron connectome demands it (Plan §42). |
| 2026-09-14 | 1 | World is a bounded box; walls are sensed and collide like obstacles. Obstacles are circles. | Simplest geometry with exact ray casts. |
| 2026-09-14 | 1 | Sensor reading = `1 - distance/range` clamped to [0,1]; rays at +45°, 0°, -45° from heading, range 4.0. | Proximity, not distance, so "closer" is "stronger" (Plan §8 uses the same convention for food). |
| 2026-09-14 | 1 | On collision the agent keeps its rotation but does not move; collision is counted. | Avoids tunnelling and gives a clean signal for later reward (Plan §9). |
| 2026-09-14 | 1 | Adapters are introduced now: `SensoryAdapter` (channel -> neuron, gain, tonic currents) and `MotorAdapter` (two neuron ids -> wheels). | Plan §2.2 wants independent layers from the start; both are tiny. |
| 2026-09-14 | 1 | Motor neurons run on a tonic current of 1.0 and sensors inhibit the *opposite* wheel. Front also excites the left wheel so a front+right conflict still turns right instead of freezing. | Hand tuning only (Plan §7); first wiring deadlocked and was fixed. |
| 2026-09-14 | 2 | Food is a gradient sense, not line-of-sight: signal = `1/distance` clamped to [0,1], routed to one of three sectors (front ±30°, left, right) by bearing. Food behind is still sensed on a side. | Plan §8 formula; chemosensation in the worm is not directional vision. |
| 2026-09-14 | 2 | Only the *nearest* food item is sensed, so the three food channels are mutually exclusive. | Binary neurons cannot compare magnitudes; two foods on opposite sides deadlocked the hand-wired brain. |
| 2026-09-14 | 2 | Hunger grows `hunger_rate` per tick, saturates at 1.0 (`World.starved`), resets to 0 on eating. Food is consumed on contact. Starvation has no consequence yet. | Consequence is a reward matter (stage 3). |
| 2026-09-14 | 2 | Hunger is exposed as an observation channel and gates food attraction in the demo through a HUNGRY interneuron. | Shows motivation observably: a sated agent drives past food. |
| 2026-09-14 | 2 | Shared `experiments/episode.py` holds the closed loop, trace record and rendering; stage demos only build scenarios. | Avoids copy-paste between stage demos. |
| 2026-09-14 | 3 | Reward lives in `learning/reward.py` as `RewardConfig` + `RewardTracker`; it takes plain event flags (ate, collided, starved) and position, not the World, so any environment can feed it. | Plan §9 single source of numbers; keeps learning decoupled from simple_2d. |
| 2026-09-14 | 3 | Exploration bonus counts distinct `floor(x/cell)`, `floor(y/cell)` cells with cell 1.0; collision penalty per tick, capped at 20 per episode; starvation scored once and ends the episode. | Plan §8-9 as revised. |
| 2026-09-14 | 3 | Episode ends inside `run_episode` on `world.starved` or after `steps`. Demo `hunger_rate` moved from 0.01 to 0.004 (starves in 250 ticks). | Plan §8 range 0.002-0.005; the stage-2 demo still eats both items and then starves, since respawn is stage 4. |
| 2026-09-14 | 4 | Genome = weight vector of the stage-2 topology (12 synapses) in insertion order; thresholds, decay and adapter wiring (tonic 1.0 on motors) stay fixed. Initial weights uniform in [-1, 1], mutation Gaussian σ=0.3, clipped to [-2, 2], truncation selection with 20% elites, no crossover. | Plan §10 algorithm literally; simplest thing that shows learning works. |
| 2026-09-14 | 4 | Fitness = mean total reward over 4 fixed train seeds (random start pose, 2 food items with seeded respawn, fixed obstacle). Held-out seeds 1000+ report generalisation. | Plan §10 revised: never one map. Obstacle layout stays fixed until stage 12. |
| 2026-09-14 | 4 | The closed loop moved from `experiments/episode.py` to `doomworm/episode.py` and scores ticks through a `TickScorer` protocol, so `learning` can import the loop without a cycle. | Layering: episode < learning < experiments. |
| 2026-09-14 | 4 | Brain JSON format v1: neurons (id, threshold, decay), synapses (source, target, weight, kind), meta (scenario, seeds, fitness...). `doomworm play` rebuilds adapters from `meta.scenario`. | Plan §40. Adapters are scenario code, not data, for now. |
| 2026-09-14 | 4 | Known failure mode of the evolved brain: on some seeds (e.g. 1000) it freezes when obstacle-right and food-right fire together and both motors are inhibited. | Binary neurons; addressed by graded activity from stage 5 (Plan §2.3), not by tweaking this stage. |
| 2026-09-14 | 5 | Dataset: Cook et al. 2019 hermaphrodite tables (SI 5 corrected July 2020, SI 4 cell lists) from wormwiring.org, extracted to CSV by `scripts/fetch_connectome.py`. No explicit license on the site; used as published SI for research, cited in `data/connectome/cook2019/README.md`. | Plan §11. |
| 2026-09-14 | 5 | 302 neurons = 83 sensory + 91 interneurons + 126 motor + CANL/CANR (type `neuron`, unclassified). Muscles, glia and end organs dropped. Dataset "weight" is EM serial sections of connectivity, not a synapse count. 38 chemical self-loops and 14 gap-junction self entries kept as in the source. | Keep the data as published; interpretation lives in `build_network`. |
| 2026-09-14 | 5 | Gap junction pair (a, b) becomes two positive electrical synapses a->b and b->a. Chemical sign: negative iff the source is one of the 26 classic GABA neurons (DD, VD, RME, RIS, AVL, DVB), else positive. Training may flip signs later. | Plan §11 revised. GABA-uptake-only neurons (Gendrel 2016) are not treated as inhibitory. |
| 2026-09-14 | 5 | Weight normalisation: per target, `w = sign * gain * weight / sum(abs incoming weights)`, so every neuron's absolute incoming weights sum to `gain` (default 1.0). | Bounded input per tick regardless of hub size; hubs and leaves get equal drive, which is a simplification to revisit at stage 6. |
| 2026-09-14 | 5 | Graded neuron mode added (`Neuron.graded`): `activity = clip(potential / threshold, 0, 1)`, no reset. Binary mode stays default for stages 0-4 code; connectome networks are graded. Brain JSON writes `graded` per neuron, format version unchanged (field optional, default false). | Plan §2.3 revised. |
