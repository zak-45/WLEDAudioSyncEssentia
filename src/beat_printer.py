class BeatPrinter:
    """A simple class to manage the state of a spinning character for printing."""

    def __init__(self):
        self.state: int = 0
        self.spinner_chars = "¼▚-▞"

    def get_char(self) -> str:
        char = self.spinner_chars[self.state]
        self.state = (self.state + 1) % len(self.spinner_chars)
        return char
