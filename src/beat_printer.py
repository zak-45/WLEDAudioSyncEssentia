"""Tiny utility for visualizing beats as a spinning console character.

This module provides a minimal stateful helper that cycles through a set of
spinner characters so beat events can be displayed as a simple animation in
the terminal. It is used purely for human-friendly feedback and debugging and
does not affect any audio or lighting logic.
"""

class BeatPrinter:
    """A simple class to manage the state of a spinning character for printing."""

    def __init__(self):
        self.state: int = 0
        self.spinner_chars = "¼▚-▞"

    def get_char(self) -> str:
        char = self.spinner_chars[self.state]
        self.state = (self.state + 1) % len(self.spinner_chars)
        return char
