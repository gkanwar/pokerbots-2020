# Building bots

Common modules are stored as version-controlled master copies in in /lib. Running `python build.py name_of_bot` will check the dependencies of that bot, copy the modules from /lib and zip up the bot into /robots. Any local copies of the modules will be overwritten.

# General strategy

The bot ranks all the possible hands it could have at any stage. Pre-flop, it uses some approximant to all-in equity. Post-flop, it uses `eval7` to evaluate each possible hand with the board and rank based on the made-hand rankings (ignoring straights). **Should just rank on equity instead (against some estimate of opponent range).**

Facing a bet, the bot calculates what fraction of its range it wants to defend using some minimum defence frequency:

`MDF = pot / (pot + bet)`

so e.g. we defend against infinitesimal bets with 100% and against infinite beats with 0%.

The bot then has some fraction of these hands that it will raise. It then figures out a bluff frequency to make that bet balanced:

`bluff frequency = bet / (pot + bet)`

These bluffs are then constructed to be the hands just below the MDF threshold (i.e. too weak to call, but the next strongest). So the overall range will look like [strongest] RRCCCCCRRFFF [weakest] where R=raise/bet, C=check/call, F=fold.

The range of hands the bot has after each action should then be reduced.

These are all concepts from "toy game" single-street poker variants, and I'm just mis-applying them to the full poker game tree to get something that's approximately sensible.

# Hacks/hard-coded values

There are a bunch of free parameters that I've just hardcoded for now based on what seems somewhat reasonable:

 * preflop raise from SB = 85%
 * preflop raise (facing bet/call) = 33%
 * preflop raise size = 1x pot
 * postflop raise fraction = 15%
 * postflop raise/bet size = 0.75x pot
 * get rid of all bluffs (other bots didn't seem to bluff much)
 * the range is restricted after each action pre-flop, but not restricted post-flop (seemed to get us into trouble because we'd end up in spots with no strong hands or no weak hands and end up overvaluing weak hands / undervaluing strong hands)