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
