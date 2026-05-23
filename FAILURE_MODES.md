# FAILURE_MODES.md — Real Home Failure Modes

HomeBrain must be designed around these cases from the beginning.

## Perception failures

- dark room
- overexposed window
- reflective floor
- mirror/glass door
- black furniture
- thin chair legs
- white wall / low texture
- camera dirty or occluded
- motion blur

## Geometry failures

- monocular depth wrong scale
- floor/rug boundary confusion
- cords mistaken for harmless texture
- low obstacles below camera line
- stairs/cliffs not visible enough
- wheel slip corrupts odometry

## Dynamic home failures

- human walks through
- pet walks/sleeps in path
- toy/box appears after map was built
- door opens/closes
- chair moves
- laundry/cords move
- person temporarily blocks path

## Planner failures

- loops forever
- over-cleans already-covered area
- gets stuck under furniture
- wedges into chair legs
- avoids too much and gives up
- drives into uncertainty instead of slowing/stopping

## Memory failures

- maps a moving person as wall
- forgets already-cleaned zones
- believes old free-space after object moved
- accumulates pose drift
- cannot relocalize after recovery

## Required behaviors

- slow down on uncertainty
- stop on high dynamic risk
- decay temporary obstacles
- preserve persistent static structure
- recover with reverse/turn/search
- mark zones as uncertain instead of lying
- produce debug output explaining stop/recovery reason
