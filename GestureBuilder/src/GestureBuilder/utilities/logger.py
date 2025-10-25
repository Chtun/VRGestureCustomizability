import os
from pathlib import Path

class Logger:
    def __init__(self, *files, output_folder: Path, filename="log.txt"):
        """
        Redirects prints to multiple files and stdout optionally.
        
        Args:
            files: Any file-like objects to also write to.
            output_folder (str, optional): Folder to save log file. Creates folder if it doesn't exist.
            filename (str): Name of the log file in the output folder.
        """
        self.files = list(files)

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            log_path = os.path.join(output_folder, filename)
            log_file = open(log_path, "w")
            self.files.append(log_file)

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()
        
    def close(self):
        # Only close files we opened ourselves
        for f in self.files:
            f.close()
        self.files = []

