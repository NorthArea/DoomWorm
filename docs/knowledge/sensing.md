# Putting a world in front of 302 neurons

Where a signal lands decides what the network can do with it. Every entry was
measured; none of it is task-specific.

## Where a channel lands is chosen by outcome, never by the story

Twice now a plausible story picked the wrong neuron pair. Scoring the 31 free
sensory pairs by *turn bias* put ASG on top; scoring the same pairs by **hits
landed** put CEPD on top with 20 of 27 shots while ASG managed under two. The
bias was two orders of magnitude below the body's own tonic turn, so it
predicted nothing.

**For the other track:** rank candidate landing sites by the outcome you
actually want, on the untrained network, before training anything. A one-minute
sweep beats an afternoon of reasoning.

## A level and its change are different senses, and only one of them is free

All of our channels were levels — `1/distance` and the like. The animal's are
not: ASEL fires on a *rising* concentration, ASER on a falling one, and the
falling edge is what raises its turn rate.

Adding relative-change channels (a fraction of what is already felt, Weber, not
a difference) did nothing at all untrained: −18.53 → −18.49. A derivative is
worthless to random weights. By contrast a channel that feeds an existing
reflex pays immediately — the prey channel took an untrained network from never
firing to 0.5 kills an episode on a map it had never seen.

**For the other track:** a new channel pays untrained only if it lands where a
reflex already uses it. Anything the network must *learn to read* has to be
measured after training or not at all.

## Compute a derivative once per tick, never in the observation

The first implementation computed the change inside `observe()` and read zero
forever: the loop looks at the world twice per tick and nothing moves in
between. It belongs in the step.

## Constant noise is jitter, not search

Adding the animal's stochasticity — a gaussian current per neuron, a synapse
that transmits with probability below one — was worth +12.1 and +7.7 reward on
two tasks untrained, and produced **one** episode of directed behaviour in
thirty-six. The animal's version is not constant: the reversal rate rises when
the gradient worsens. Without that coupling, noise buys robustness to being
stuck, not a way out of it.

## The body's own asymmetry can drown any sensory signal

A trained brain drove with a fixed wheel difference of 0.63 — a permanent slow
turn. The sensory signals we were feeding it moved the wheels by 0.04. Raising
the motor turn gain from 6 to 48 moved the best bearing it could reach from 65
degrees to 21, which no amount of re-routing had done.

**For the other track:** check the scale of the tonic output before concluding
that a channel does not work. Adapter gains are a free parameter and we had
never swept them.
