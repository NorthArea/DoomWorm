# Findings: does the natural network play Doom better?

> This is the **doom** track. What we learned here that holds whatever the
> task is lives in `docs/knowledge/`, and the sibling track (the home robot,
> `docs/broom/`) draws on the same file. Cross-track lessons are not
> repeated in this document.

The project's result. One row per question: the connectome worm against the
best network trained from scratch, under the same budget and the same
benchmark. Numbers are mean reward over >= 3 training seeds unless marked; the
profit column is worm minus best self-trained, in reward units. Verdict:
**win** / **tie** (inside the spread) / **loss**.

Rows measured on the vacuum robot and the kit car, which this project used to
carry as a second track, are not here: they live in the git history up to
`1bf247a`.

| # | Condition | Worm | Best self-trained | Profit | Verdict | Rows |
|---|---|---|---|---|---|---|
| 1 same task, same budget — **re-measured** | `doom4` after the two fixes (stage 12 lane), same protocol: 3 training maps, evolution 40 x 25, 12 unseen maps, 3 seeds. The nets are unchanged by the fixes and keep their rows | curriculum −2.11 ± 6.0 (0.19 kills on 13.8 shots); real −2.80 ± 7.9 (0.17 kills on 20.3 shots) | PPO −3.3 ± 6.5; rnn −8.5 ± 1.7; ncp −9.5 ± 1.1 | +1.2 vs PPO (curriculum) | **tie** with PPO, **win** over the same-size net, still **loss** to the `doomguy` floor (+4.8) | `runs/benchmark_doom_b2d/train_doom4/` |
| 1 controls — **re-measured** | same | real −2.80 ± 7.9 | shuffled −2.15 ± 5.6 (0.06 kills on 45.9 shots); random −6.24 ± 3.3 | −0.65 vs shuffled, inside the spread | **tie with its own controls on reward — and a clear win on aim**: the connectome lands 0.17 kills on 20 shots against the shuffle's 0.06 on 46. Accuracy 2.8 % against 0.5 %. The old 11.6-point deficit to the shuffle was our blind trigger, and fixing it moved the worm +10.9 while leaving the control untouched | same |
| 1 same task, same budget (superseded) | `doom4` (gun, one monster, exit), 3 training maps, evolution 40 x 25 / PPO 300k, 12 unseen maps, 3 seeds | real −13.7 ± 5.2; curriculum −8.1 ± 4.0 | PPO −3.3 ± 6.5; rnn −8.5 ± 1.7; ncp −9.5 ± 1.1 | −10.4 (curriculum −4.8) | **loss** — and the whole field loses to the zero-learning `doomguy` floor (+4.8), so at this budget nobody learned Doom | `results/b/b2_doom_2026-09-16.md` |
| 1 controls (superseded) | same | real −13.7 | random −4.9 ± 6.2; shuffled −2.1 ± 6.8 | −11.6 vs shuffled | **loss to its own controls** — the first axis where the wiring is worse than the same wiring shuffled | same |
| 2 a harder level — **re-measured** | `doom6` (apartment, two walking monsters, gun), after the fixes, 3 seeds | real −29.80 ± 4.7; curriculum −40.55 ± 7.4 | shuffled −23.06 ± 12.0; random −26.26 ± 5.4; ncp −21.3 ± 2.3 | −6.7 vs its own shuffle; −8.5 vs ncp | **loss**, and the fixes did not carry: the worm gained 10.3 here but the curriculum worm *lost* 16.0 (−24.6 before), consistently across all three seeds. The floor takes +27.2 with 1.33 kills and 75 % exits | `runs/benchmark_doom_b2d/train_doom6/` |
| 3 the real engine — **re-measured** | the same layouts run by ViZDoom, brains from the fixed lane, no retraining, 3 seeds | `vizdoom4`: curriculum +2.30 ± 3.9 (0.25 kills, survives every episode); real −1.85 ± 7.0. `vizdoom6`: real −6.68 ± 8.0; curriculum −9.38 ± 6.1 with 0.44 kills, the most of any row | `vizdoom4`: shuffled −1.91 ± 4.7; random −5.06 ± 5.5. `vizdoom6`: shuffled −3.76 ± 6.5; random −4.87 ± 3.9 | +4.2 on `vizdoom4`; −2.9 on `vizdoom6` | **tie**: the worm family leads on `vizdoom4` and trails on `vizdoom6`. And the fixes cost points *here* — the curriculum worm was +7.1 on `vizdoom4` before them | `runs/benchmark_doom_b2d/train_doom*/seed*/eval_vizdoom*` |
| 2 a harder level (superseded) | `doom6`, before the fixes | real −40.1 ± 3.3; curriculum −24.6 ± 1.0 | ncp −21.3 ± 2.3; PPO −33.0 ± 6.4; rnn −46.2 ± 4.0 | −18.8 (curriculum −3.3) | **loss** | `results/b/b2_doom_2026-09-16.md` |
| 3 the real engine (superseded) | the same seeded layouts run by ViZDoom, no retraining | curriculum 7.1 ± 3.5 — the best trained brain on `vizdoom4` | PPO 0.8 ± 2.6; rnn −1.1 ± 0.9; ncp −2.2 ± 0.5 | +6.3 | **win among trained brains** (still under the floor's +27.4): the engine reverses the simulator's order and the worm family comes out on top | same, `runs/benchmark_doom/train_doom4/*/eval_vizdoom*` |
| 4 a map nobody drew | the stock ViZDoom scenario `defend_the_center`, brains from `doom4`/seed 0, 4 episodes each | real 3.2 ± 16.8 with 2.0 kills; curriculum −32.4 with no shot fired | rnn −29.6 and PPO −37.3, neither ever fires; shuffled 14.9 with 4.0 kills | +32.8 vs PPO, −19.6 vs the floor | **win over the trained nets, loss to the floor**: outside our own generator only the worm family pulls the trigger at all. Four episodes, one seed — a direction, not a measurement | `runs/benchmark_doom/stock/leaderboard.md` |
| 8 the control that undid it (stage 23) | the same search with the proposals drawn **at random** instead of from the brain: uniform intents, same six candidates, same twelve-tick rollout, same six maps | search over the worm's proposals +3.13 | **search over random proposals +3.45** | −0.32: the connectome is, if anything, the worse proposer | **the win belongs to the rollout, not to the wiring.** A look-ahead that picks the best of six futures does the work whatever produces the six. Stage 21's row stands as a number and falls as an interpretation | `scripts/search_probe.py`, the control in `docs/doom/results/` |
| 8 why there was nothing to teach (stage 23) | the teacher's own actions, 3600 logged ticks: is the choice a function of the state? | a constant at the teacher's mean scores an imitation error of **0.7149**; a constant fitted **per situation** scores 0.7144 — five ten-thousandths better | the brain distilled towards the teacher reached 0.845, *worse than the constant*; the teacher's own deterministic answer scores 0.859 | — | **the teacher is not a policy.** Its actions carry no situation-dependent structure at all: the same mean everywhere (forward 0.01, turn +0.30) and all the magnitude in the spread (sd 0.6-0.7). It selects lucky noise in hindsight, and hindsight cannot be imitated | `runs/distill` (regenerate with `scripts/distill_lane.py`) |
| 9 can consensus repair the teacher (stage 23b, **re-measured**) | average the best *k* branches instead of executing the luckiest one, then search the **same state twice** from the same neural snapshot: `within` is how far the two answers to one state lie apart, `between` how far answers to different states lie apart | k=1: within 0.937, between 0.914 — k=3: 0.562 / 0.588 | k=6: 0.432 / 0.413 — k=12: 0.270 / 0.312 | ratios 1.03, 0.96, 1.04, 0.86 | **no, and the control says how to read that.** The random proposer — a quantity with no state dependence whatsoever — scores 0.93 and 1.04 on the same sweep, so at 100 decisions anything in roughly [0.86, 1.12] is what *nothing* looks like. Every consensus width lands inside it. The first printing of this row used a draw that let each branch start where the previous one's rollout left the neurons; the numbers below are from the fixed one, and the verdict did not change | `scripts/teacher_probe.py`, seed 3000, 100 decisions |
| 9 so is the worm silent, or is it where it stands (stage 23b) | the same estimator — the mean of 48 draws from one snapshot, twice — run along three trajectories, so only the driver differs | its own noisy control: spread across states **0.277**, same state twice 0.052 — a signal **5×** its own error | the search driving: 0.194 across states, 0.147 same state twice — **1.3×** | deterministic driving: 0.174 / 0.073 | **the wiring is not silent; the search's states are.** Where the worm steers itself its answer varies five times more between states than within one. Where the *search* steers, the per-state noise triples and the between-state variation does not follow. The demonstrations of stage 23 were recorded in exactly the place where the connectome has least to say | `scripts/teacher_probe.py --drivers-only`, seeds 3000-3002, agreeing to ±0.05. Unaffected by the draw bug above: this estimator restored the snapshot before every draw from the start |
| 10 the gate stage 23 never ran (stage 24) | DAgger's correction: the **student** drives and the search labels what it should have done there. Before collecting anything, label each visited state twice and compare within against between — with the random proposer run at every width, because it says how big nothing is | the worm proposing: 12 candidates best 1 → ratio 0.93, 24 best 8 → **0.75**, 24 best 24 → **0.51** | the control at the same widths: 0.98, 0.93, 1.02 | separation of 0.51 against 1.02 | **the labels are a function of the state**, and stage 23 missed it two ways at once: the wrong trajectory *and* no averaging. At best-of-1 — what stage 23 recorded — the worm sits at 0.93, inside the control's band | `scripts/dagger_lane.py --gate`, seeds 3000-3002 |
| 10 the budget that was hiding behind the verdict (stage 24) | the same imitation training at 480 evaluations against 4800, on 1819 interface weights | 480 evals: search arm 0.1833 → 0.1890, *worse than where it started* | 4800 evals: 0.1833 → **0.1579** against a constant's 0.1639 | the control arm, self-distillation, 0.1558 → **0.1161** against 0.1201 | **both beat the constant — the first time in this project a distilled student has.** Stage 23's "worse than a constant" was a budget artefact sitting on top of a real one: thirty generations over 1819 weights never left the starting point, and its trace jumps without trending, which is what that looks like | `scripts/dagger_lane.py --train --generations 150 --population 32` |
| 10 and it still does not play better (stage 24) | the taught student alone, no search at runtime, on 20 maps nothing trained on | untaught **0.31** | taught by the search **0.29** | taught by its own noise-averaged self **0.26** | **fitting the teacher improved and the behaviour did not.** Beating the constant by 4 % in action-space L1 buys nothing in reward, and the search's labels buy nothing over labels with no search in them at all. Expert iteration is closed with every stated reason eliminated: right trajectory, ten times the budget, a gate that verified the target had structure, and a control that says the look-ahead contributed none of it | `scripts/dagger_lane.py --benchmark --first-seed 4000 --seeds 20` |
| 11 the denominator nobody had counted (stage 25) | what each map pays if everything on it is collected — the exit, every monster at a kill plus the hit that lands it, every floor cell the agent could stand in — and then how much of that an episode can physically arrive at: the agent moves 0.2 units a tick, so 600 ticks buy 120 units of travel, a straight line that never turns or doubles back | on the map: `doom1` 42.4, `doom2` 38.6 | `doom4` 50.3, `doom6` 57.2 | **reachable in the time given: 22.0, 22.0, 34.0, 46.0** | **every reward in this file was a number without a scale.** The project's best worm row, +3.30 on `doom4`, is **9.7 %** of what that episode could have reached. The first positive row ever celebrated, +0.66, is 3 %. The hand-written floor's +27.2 on `doom6` is **59 %**, which is the only figure here that reads like playing the game. Most of the exploration term is on the map and out of reach: `doom1`'s floor is 32.4 and 600 ticks of travel can touch 12.0 of it at the very best | `scripts/reach_lane.py --ceiling` |
| 11 the floor under the floor (stage 25) | uniform intents with no relation to the channels, benchmarked under exactly the published conditions — 600 ticks, 12 unseen maps, ideal sensors, 24 episodes a cell | flailing: `doom1` **0.37 ± 1.2**, `doom2` **−0.20 ± 2.1** | `doom4` **−1.02 ± 3.7**, `doom6` **−24.52 ± 28.1** | the hand-written floor over the same cells: 11.69, 4.41, 4.78, 27.15 | **nothing trained in this project is distinguishable from doing nothing.** On `doom4` the worm scores −2.80, its shuffle −2.15, the best net −2.11 — all at or *below* flailing's −1.02. On `doom6` every arm from −21.3 to −40.55 is at or below flailing, and the curriculum worm's −40.55 is 16 points below it — past the ±7.0 this benchmark can read (row 12), so that one is measurably *worse* than having no behaviour at all. Twenty-four rows of worm-against-control were run entirely inside the band of a policy with no behaviour, and the only thing that has ever clearly left it is the hand-written floor | `scripts/reach_lane.py --floor`, `runs/reach/` |
| 11 and depth does not buy it either (stage 25) | the same rollout search swept over how far it looks: 6, 12, 24 and 48 ticks, 8 candidates, a decision every tick, `doom4`, 4 maps | the worm proposing: 4.47, 4.43, 1.42, 4.40 | the control proposing: 4.40, 4.40, 4.40, 4.48 | and with **no search at all**, the control scores 4.50 | **the look-ahead depth changes nothing, and neither does quadrupling it.** Every arm's 4.4 is the same single monster on the same one of four maps — a kill and three hits — plus 0.4 of exploration against the 3.2 those 300 ticks could have walked. Not one arm at any depth reached an exit. Stage 21's search was never finding a policy; it was finding a trigger pull | `scripts/reach_probe.py` |
| 12 what the floor knows (stage 26) | the hand-written floor with one reflex switched off at a time, the tick falling through to the next; same 12 seeds, 2 repeats, 600 ticks, same code path as every row above. Scored as a share of what stage 25 says an episode can reach | intact **39 %** | `shoot` removed **−53 %**, `escape` removed −17 %, `exit` removed −7 % | `face` removed 2 %; `hurt` 38 %, `avoid` 39 % — no cost at all | **four of its seven reflexes carry it, and the gun is most of it.** Taking away the decision to fire costs 92 points across the four levels; taking away contact-escape 56; the exit gradient 46; coming round onto a target 37. `hurt` and `avoid` are dead weight here — both need an empty magazine, which these levels rarely produce | `scripts/ablation_lane.py --leave-one-out`, `runs/ablation/` |
| 12 one reflex alone (stage 26) | the inverse: keep one, remove the other six | **`exit` alone on `doom1`: 11.31** against the intact floor's 11.69 | `escape` alone: −0.85 and −5.82 on the monster-free levels | `hurt`, `shoot`, `avoid` alone: **identical** at −18.60 / −19.20 / −22.50 / −34.51 | **on a level with no monsters the whole floor is one rule.** Gradient-following alone scores 97 % of what all seven reflexes score. The three identical rows are not a measurement of those reflexes: with everything else gone the tick falls through to "forward", and none of the three fires often enough to change it — that signature is the always-forward policy, which is *worse than flailing at random* (−18.6 against +0.37), because a policy that jams into a wall pays the collision penalty for the rest of the episode | `scripts/ablation_lane.py --only-one` |
| 12 what this benchmark can read (stage 26) | the error of the mean behind every row in this file: the published ± is the spread over episodes, so a mean of *n* of them carries std/√n | `doom1` ±0.2, `doom2` ±4.2 | `doom4` ±6.2 at 24 episodes, ±4.7 at 48 | `doom6` **±7.0** at 24 episodes | **differences smaller than this are not differences.** Doubling the seeds moved the *unchanged* floor on `doom6` from 27.15 to 20.80 and on `doom4` from 4.78 to 7.83 — both inside their own interval, and both larger than several margins this file has reported as results. Row 2's "−6.7 vs its own shuffle" on `doom6` is one of them | `scripts/ablation_lane.py --resolution` |
| 9 the competence map (stage 22) | 3600 searched decisions on `doom4`, six proposals each, grouped by the situation the channels describe | ~~42-51 % agreement~~ **withdrawn**: the first proposal was chosen 41 % of the time and in *every one* of those its score merely **tied** the best, so the column measured `max()` taking the first of equal scores, not competence | — | the search adds **twice as much** wherever a monster is involved (+0.13) as in any exit situation (+0.07) | what survives the correction is the **gain** column, which is a real difference in score: a rollout adds twice as much wherever a monster must be engaged (+0.13) as in any exit situation (+0.07), and least of all with a wall ahead (+0.06), which is where the animal's own reflex lives. And two thirds of the decisions (2423 of 3600) are spent with the exit off to one side, the situation it cannot resolve | `docs/doom/results/competence/doom4.md` |
| 8 search over the worm's own proposals (stage 21) | with noise in the neurons the brain is a distribution: draw N intents from one state, roll each forward in a copy of the world, keep the first action of the best branch. The connectome proposes, the rollout judges. `doom4`, six unseen maps, the project's best brain | **+3.13 with the search** against +1.40 for the same brain alone and +1.07 for it made stochastic | — (the search is over the connectome's own output, so there is no self-trained arm) | +1.73, more than double | **the search wins, and the connectome has nothing to do with it** — see the control below. It costs about a hundred times an ordinary episode and needs a simulator | `docs/doom/results/` (stage 21), `scripts/search_probe.py` |
| 8 the first answer was wrong, and why | the same search with a five-tick hold | −3.28 on `doom4`, −1.22 on `doom2`: *worse* than the brain alone | — | — | **a methodological artefact, not a result**: branches were scored with the brain deciding every tick and then only the first intent was applied and held. Judge and player were doing different things; making them agree flipped the sign. Recorded in `docs/knowledge/method.md` | same |
| 7 the ceiling, with everything outside the brain tuned (stage 18) | `doom4`, three seeds, the same 1000 evaluations as every row above. Two levers, neither of which touches the network: a genome cut to the 1819 interface synapses (those leaving a sensory neuron or entering a motor group, with the animal's interneurons left alone), and sep-CMA-ES in place of fixed-sigma mutation | **CMA + interface +0.66 ± 3.4** — the worm's first positive row in the project; CMA alone −0.56; mutation + interface −0.63; the shipped protocol −2.80. On `doom6` CMA alone −22.10 against −38.43, a gain of 16 | (the nets are not re-run: these levers are about searching the connectome) | +3.5 over the shipped protocol on `doom4`, +16 on `doom6` | **the ceiling is the architecture, not the optimiser** — which is what this row was built to establish. A search with an adaptive step and a per-coordinate scale, proven on the sphere and on a 10^6-conditioned ellipsoid, lands in the same place the crude one did: about zero on `doom4`, no kills on the classic map, and not one exit anywhere. Still under the `doomguy` floor (+4.8) | `runs/squeeze/benchmark/`, `runs/squeeze2/benchmark/` |
| 7 what the strong search changed | same rows, looking at behaviour rather than score | CMA fires to the bottom of the magazine (36-50 shots) and collects exploration and survival; mutation on the interface genome fires 8.4 times and kills three times as often (0.19 against 0.08) | — | — | **a different strategy, not a better one**: CMA found the safe, unaimed policy and found it reliably (spread ±1.1 against ±7.9). Aiming survives better under the weaker search, which suits a reward where a hit is rare and exploration is steady | same |
| 7 gains as genes | the adapter's `gain_drive`, `gain_turn` and trigger gain put in the genome as `base * 8 ** gene`, so the search picks them instead of the author | −6.93 ± 7.3, the worst row of the lane | — | −7.6 against the same setup without them | **failed**: on an exponential scale a small step changes the turn rate several-fold, and the search falls into a bad region it cannot climb out of. A linear parametrisation or a narrower range might work; this one does not | same |
| 6 genome size (stage 18) | how much of the connectome the search may move, mutation, three seeds | interface (1819 synapses) `doom4` −0.63, `doom6` −31.55 | full genome (5905) −2.80 and −38.43; sensory only (506) −13.01 and −40.66 | +2.2 and +6.9 for the interface cut | **worth it**: tuning how loudly the senses speak and the muscles listen beats tuning everything, at the same budget. Path-reachability cannot narrow it further — within two hops of the sensors 99.3 % of synapses can already reach a motor group | `runs/squeeze/benchmark/` |
| 5 memory as a layer (stage 16) | a place memory outside the brain: it marks the cells odometry has visited and reports the nearest *unvisited* ground as a gradient, in the same form as the food smell. The brain reads it and still decides. Trained and benchmarked under the condition, `doom4`, 3 seeds | worm+memory: `doom6` −24.28 ± 7.3 (best row on that level, was −38.43), `doom2` +1.28, `doom4` −8.36 (worse than without) | shuffled+memory −35.17 on `doom6`, −6.86 on `doom4`: the same layer *hurts* the shuffled control everywhere | +14.2 on `doom6`, −5.6 on `doom4` | **the hypothesis failed on its own terms**: exits stayed at 0.00 on every level, which is what the layer was built to buy. What it did buy is wall-grinding — collisions fell 6-50x (doom4 19.6 -> 0.3, doom6 25.4 -> 4.1) — and a large gain on the hardest level, and only for the real wiring | `runs/memory/benchmark/` |
| 5 hybrids | a frozen trained brain as a reflex module under a small net; only the net trained | hybrid 43.2 ± 1.2 | the same net alone 37.7 ± 1.4 | +5.5 over the bare net | **the worm helps as a part, and only up to itself**: it lifts a from-scratch net by 5.5 points, exactly to the worm's own level, and steadies it across seeds. Measured on the conditions of the old second track; to be re-run on Doom | `runs/c7/benchmark/leaderboard.md` |
| 6 shooting, before the fixes | what the trigger actually does on `doom4` | real 0.00 kills on 3.1 shots; curriculum 0.03 on 29.9 | PPO 0.00 on 11.2; rnn and ncp never fire | floor: 0.50 kills on 5.7 shots | **superseded** — every row above was measured while the trigger was unreachable and the monster was only a source of pain. See the two fixes below; stage 12 re-measures them | same |

## What was wrong with the rows above

Two defects, both ours, both in the mappings rather than in the connectome, and
both found by measuring untrained brains:

- **The trigger was unreachable.** It sits on the pharyngeal pumping group
  M3/M4/MC, and the pharyngeal nervous system is an island: exactly five
  connections cross its boundary, all through RIPL/RIPR. Untrained, the real
  worm pulled the trigger in 0.0 % of ticks and answered 0.134 with a monster
  on the gun line against 0.157 with none — pure noise. Its shuffled control
  has 399 crossing connections after the rewiring and fired in 77 % of ticks.
  **That is why the control beat the connectome on every shooting row**, not
  better wiring. With `aim` delivered to RIP, the animal's own door, the same
  untrained worm answers 0.906 against 0.143.
- **There was nothing to turn toward.** The gun is bolted to the chassis, so
  aiming is turning, and the monster reached the brain only as `danger_*` into
  ASH — the escape pathway, which turns the body *away*. The `prey_*` channels
  now read it as a lateralised attractant on CEPD, the dopaminergic pair that
  in the animal reports food under the nose and slows it down.

With both in place and **no training at all**, the connectome on the stock map
went from never firing and dying in every episode to 0.50 kills and surviving
three quarters of them. Against a standing target it closes from 10 units dead
ahead and lands 20 hits; 60 degrees off the bow it still cannot come round.

## After the two fixes (2026-09-17)

The shooting rows were re-measured on `doom4` with the same protocol. The worm
gained +10.9 points, the curriculum worm +6.0, and the shuffled control did not
move at all — which is the signature the explanation predicted: the deficit was
a trigger our mapping had put out of reach, not the wiring. What is left is a
tie on reward between the connectome, its controls and PPO, with the worm ahead
on the one measure that is about aiming rather than volume: three times the
kills on half the shots.

`doom6` and the engine tell a different story, and it is not a good one: on the
harder level the worm still trails its own shuffle, and the curriculum worm —
the best brain the project had — lost 16 points there and 4.8 in the engine.
Whatever the two fixes bought on `doom4`, they did not generalise.

The one quantity that holds in all four runs is **aim**. Kills per shot:

| run | worm | curriculum | shuffled | random |
|---|---|---|---|---|
| `doom4` | 0.82 % | **1.41 %** | 0.12 % | 0.17 % |
| `doom6` | 0.14 % | **0.28 %** | 0.14 % | 0.21 % |
| `vizdoom4` | 1.00 % | **2.14 %** | 0.24 % | 0.52 % |
| `vizdoom6` | 0.34 % | **0.96 %** | 0.28 % | 0.31 % |

The connectome family aims two to nine times better than its own controls in
every run, and the curriculum worm is the best of them everywhere. On raw
reward there is no such ordering. That is the project's one stable signal so
far, and it is about *what it does with a shot*, not about winning.

Memory did not buy the exit. The novelty gradient pulls toward unexplored
ground, and the exit is a particular place; going round a wall means moving
*against* the target gradient for a while while still holding it as the goal,
and the brain has neither a held goal nor a way to come round. The layer
removed the symptom — grinding against a wall — without supplying the thing
that was missing.

## The ceiling, measured with a strong search (2026-09-17)

Everything in this section leaves the network alone: the topology, the neuron
model and the composition of the circuit are the animal's. What changed is how
the weights are found and how the world is translated for it.

| lever | result |
|---|---|
| genome cut to the interface (1819 of 5905) | +2.2 on `doom4`, +6.9 on `doom6` |
| sep-CMA-ES instead of fixed-sigma mutation | +2.2 more on `doom4`, +16 on `doom6` |
| the two together | **+0.66 on `doom4`** — the worm's first positive row |
| adapter gains as genes | −7.6: worse, on an exponential scale the search falls off a cliff |
| a place memory as a layer (stage 16) | collisions 6-50x down, no exits |
| aim-damped turn | worse (43 degrees against 26): the aim neurons fire for other reasons too |
| motor turn gain, by hand | the body does come round — best bearing 65 degrees at gain 6, 21 at gain 48 — but never into the four degrees a hit needs |

And the wall is in the same place as before. No exits anywhere, no kills on the
classic map, and the hand-written floor at +4.8 still ahead. The search is not
what is holding it back: sep-CMA-ES adapts its step and scales every coordinate
separately, it solves a 10^6-conditioned ellipsoid that the old mutation cannot
touch, and on this task it converges three times more tightly (±1.1 against
±7.9) to a slightly better number.

One behavioural finding is worth more than the scores. Under the strong search
the worm empties its magazine and collects exploration and survival; under the
weak one, on the interface genome, it fires a third as often and kills three
times as much. The reward pays steadily for wandering and rarely for a hit, so
the better the optimiser, the more surely it finds the safe policy. Aiming
survives *because* the search is bad at following that gradient.

## Why it is bad at this, in numbers

Mean reward per episode, broken into what pays and what costs, 12 unseen maps,
the benchmark's own path:

| `doom4` | exits | what pays | what costs |
|---|---|---|---|
| floor | 0.33 | kill +5.0, exit +3.3, hit +3.0, explore +2.2 | collision −7.5, damage −1.3 |
| worm | 0.00 | kill +2.5, explore +2.2, hit +1.5 | — |
| curriculum | 0.00 | explore +3.2, kill +1.7, hit +1.3 | collision −8.3, damage −3.3, death −1.7 |
| shuffled | 0.00 | explore +1.1, kill +0.8 | collision −3.3 |

| `doom6` | exits | what pays | what costs |
|---|---|---|---|
| floor | 0.75 | kill +13.3, hit +8.0, exit +7.5, explore +2.4 | collision −4.0 |
| worm | 0.00 | hit +1.0, explore +0.6 | damage −16.7, death −8.3, collision −7.4 |
| curriculum | 0.00 | hit +0.8, explore +0.6 | damage −30.0, death −15.0 |
| shuffled | 0.00 | explore +0.8, hit +0.5 | damage −16.7, death −8.3, collision −12.3 |

Three things fall out of this, and none of them is "the connectome is a bad
network":

1. **No trained brain has ever reached an exit — 0.00 in every row.** The exit
   is the single largest item on the board (+10) and the floor collects it in a
   third to three quarters of its episodes. The worm senses the exit as a smell
   through walls; climbing that gradient walks it into the wall between. Going
   *around* needs a memory of where it has already pushed, and the brain has
   none: its neurons leak over a few ticks and there is no map anywhere in the
   contract. So roughly half the available reward is structurally out of reach.
2. **On the harder level the difference is not scoring, it is dying.** The
   floor takes zero damage on `doom6` and collects +13.3 of kills; the trained
   brains take −16.7 to −30.0 of damage and −8.3 to −15.0 of death. The floor
   engages at range and retreats when hit — a policy of three hand-written
   rules. The worm has no notion of range, because it cannot turn onto a target
   it is not already facing: measured, 0 of 31 sensory pairs land a single hit
   from 40-90 degrees off the bow.
3. **What is left for the worm to earn is crumbs.** Exploration pays +0.6 to
   +3.2 an episode, and that is what evolution optimises, because it is the
   only term a wandering taxis animal can reliably collect. The reward function
   is not wrong; it simply pays most for two behaviours — reaching a place and
   winning a fight — that this brain has no machinery to produce.

The honest summary is that the task asks for **navigation and target
selection**, and the *C. elegans* connectome is a chemotaxis controller: climb
a gradient, reverse when something hurts. It does that part visibly better than
its own shuffled copy — that is the aim column. It has no circuit for "go round
the wall" or "keep facing the thing that is shooting me", and no amount of
weight tuning inside a fixed topology has produced one in 25 generations, or in
60 (the wider-budget pilot moved reward by 6 points and aim not at all).

## What the map says so far

- **Read every row below against stage 25 first.** An episode can reach 22-46
  points depending on the level, and a policy with no behaviour at all scores
  between +0.4 and −24.5 of that.
  Every trained brain in this file, the worm and its controls alike, lands in
  that band. The comparisons are real and the differences between them are
  measured honestly; what stage 25 adds is that they are differences between
  ways of not playing the game.
- On the easy level the worm now ties PPO and its own controls and beats the
  same-size net. On the harder level it still trails its own shuffle, and the
  curriculum worm — the project's best brain — got *worse* there after the
  fixes. The fixes bought aim, not competence.
- The worm's advantages show up **off the training distribution**: in the real
  engine, and on a map from someone else's generator, it is the only trained
  family that fights at all.
- Nothing trained has beaten the hand-written floor, on any level, ever. That
  is the honest headline of the project so far.
- The reason is structural, not a tuning problem: no trained brain has reached
  a single exit, and on the harder level they die instead of engaging. Both
  behaviours need machinery the connectome does not have — a memory of where it
  has been, and a way to come round onto a target.
- Expert iteration is finished, and it failed for a reason that is not the
  learning loop. Every candidate explanation was removed one at a time — the
  trajectory, the budget, whether the target had any structure — and the
  student still plays 0.29 against 0.31 untaught. The connectome's answer *is*
  a function of the state; the content of that dependence is not what this
  reward pays for, because the reward is collected by rare events that a
  correction of 0.15 in intent units never reaches.
- A teacher recorded off the student's own trajectory taught nothing, and the
  measurement says why: the connectome's answer is five times more
  state-dependent than noisy where it steers itself, and no more than noisy
  where the search takes it. Distillation was asking the brain to reproduce
  its own behaviour in the one region where that behaviour is undefined.
- The connectome is worth something **as a component**: bolted under a
  from-scratch net it carries that net up to the worm's own level, and makes it
  far steadier across seeds — but not past it.
