##

'''

ExpressLRS uses the CRSF serial protocol between the receiver and the
flight controller, and between the radio and the transmitter module.
This decoder reassembles those frames, verifies their checksum, and
annotates the payload of the frames which ELRS uses.

Note that the over-the-air layer is not covered: ExpressLRS transmits
LoRa chirps (or FLRC/FSK in some modes) which need an RF receiver and
demodulation, neither of which a logic analyzer can provide.

'''

from .pd import Decoder
