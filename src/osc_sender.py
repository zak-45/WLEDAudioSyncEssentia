from pythonosc.udp_client import SimpleUDPClient

class OSCSender:
    def __init__(self, ip="127.0.0.1", port=57120, path="/genre"):
        self.client = SimpleUDPClient(ip, port)
        self.path = path

    def send(self, path, value):
        self.client.send_message(path, value)

    def send_silence(self, value):
        self.client.send_message("/WASEssentia/silence", value)
