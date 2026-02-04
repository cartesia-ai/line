from pathlib import Path


class SimpleLogger:
    def __init__(self, log_name="report"):
        # Get current working directory and append 'reports'
        reports_dir = Path.cwd() / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Corrected path
        self.log_path = str(reports_dir / f"{log_name}.txt")

        header = f"Assessment for {log_name}\n"
        self._write(header)
        self._write("-" * len(header) + "\n")

    def info(self, message: str):
        self._write(message + "\n")

    def warning(self, message: str):
        self._write("WARNING: " + message + "\n")

    def error(self, message: str):
        self._write("ERROR: " + message + "\n")

    def _write(self, message: str):
        with open(self.log_path, "a") as f:
            f.write(message)
