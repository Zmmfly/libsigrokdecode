##

'''

DShot is the digital ESC protocol which replaced the analog PWM and
OneShot variants. It is used with BLHeli_S, BLHeli_32 and Bluejay ESC
firmware, and supports bidirectional telemetry on the same signal wire.

'''

from .pd import Decoder
