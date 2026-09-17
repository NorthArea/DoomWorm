# Stage 23.0: the platform after removing the mini-Doom mechanics

Same command on the commit before the removal (`dc91568`, in a git worktree)
and after it: the scripted Roomba controller under the needs layer on the
default benchmark (apartment / clean / vacuum sensors, 6 unseen maps x 3
noise repeats, 800 steps). The controller is deterministic, so any change in
the simulator, the layer or the loop would move these numbers.

```
uv run broomworm benchmark --scripted roomba --planner needs
```

| code | episodes | reward | coverage | collisions | damage | dockings | survived | ticks |
|---|---|---|---|---|---|---|---|---|
| before (dc91568) | 18 | 48.12 ± 13.46 | 0.24 ± 0.04 | 17.7 | 0.0 | 4.5 | 1.0 | 800.0 |
| after (stage 23.0) | 18 | 48.12 ± 13.46 | 0.24 ± 0.04 | 17.7 | 0.0 | 4.5 | 1.0 | 800.0 |

Identical. Removed: `Enemy`, hitscan and the gun, the exit, the trigger output
of `Drive` and of every candidate, `FireAdapter`, the pharyngeal `fire` motor
group, the `aim` / `prey` sensory routes and channels, `ammo`, the kills / hits
/ shots / exited counters and the Doom leaderboard columns. Old brain and
benchmark JSON files still load.
