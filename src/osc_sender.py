"""Thin wrapper around a UDP OSC client for sending analysis results.

This module centralizes all outgoing OSC communication so the rest of the code
can emit genre, mood, and control data with simple method calls. It hides the
details of the underlying python-osc client and provides a consistent place to
extend or customize OSC addressing.
"""

from pythonosc.udp_client import SimpleUDPClient

class OSCSender:
    def __init__(self, ip="127.0.0.1", port=12000, path="/genre"):
        self.client = SimpleUDPClient(ip, port)
        self.path = path

    def send(self, path, value):
        self.client.send_message(path, value)

    def send_silence(self, value):
        self.client.send_message("/WASEssentia/audio/silence", value)
