from rich.console import Console
from rich.panel import Panel
from rich.text import Text

install_path = "WLEDAudioSyncEssentia"

console = Console()

message = Text()
message.append("The application has been extracted successfully.\n\n", style="bold green")
message.append("Extraction folder:\n", style="bold")
message.append(f"{install_path}\n\n", style="cyan")
message.append("➡ Please navigate to this folder and run the application.", style="yellow")

class msg():
    def init(self):
        pass
    @staticmethod
    def message():
        console.print(
            Panel(
                message,
                title="Installation Complete",
                border_style="green",
                padding=(1, 2),
            )
        )
