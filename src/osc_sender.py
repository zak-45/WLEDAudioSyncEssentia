from pythonosc.udp_client import SimpleUDPClient

class OSCSender:
    def __init__(self, ip="127.0.0.1", port=57120, path="/genre"):
        self.client = SimpleUDPClient(ip, port)
        self.path = path

    def send(self, items):
        """
        items: list of (label, value)
        """
        for label, value in items:
            self.client.send_message(
                f"{self.path}/{label.replace(' ', '_').replace('&', 'and')}",
                float(value)
            )

    def send_silence(self):
        self.client.send_message("/genre", [])
