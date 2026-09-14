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
